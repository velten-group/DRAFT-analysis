"""Fit the native SCENIC+ R2G separately within each stage. A variant of ours; official SCENIC+ fits R2G once across all cells.

Reuses the expression, global topics and imputed accessibility of a completed run, so topic training still sees all cells. Each
stage trains on its own cells only, with the same candidates, GBM settings and seed. No Hi-C label is read. Stages finished
earlier are reused, so an interrupted run can simply be started again.

    python comparisons/scenicplus_r2g/stage_fit.py --run RUN --cpus 8
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from native import digest, infer_r2g, load_native
from workflow import check_files, read_json, resolve_run, write_json

LOG = logging.getLogger(__name__)
OUTPUT = "stage_fit_v1"


def stage_table(api, expression, accessibility, search, temp_dir, cpus, seed):
    """One stage's scores; every candidate must be scored exactly once."""
    table = infer_r2g(api, expression, accessibility, search, temp_dir, cpus, seed)
    expected = set(zip(search.Name, search.Gene))
    if len(table) != len(expected) or set(zip(table.region, table.target)) != expected:
        raise ValueError("Stage R2G does not score every candidate exactly once.")
    if not np.isfinite(table.importance).all() or (table.importance < 0).any():
        raise ValueError("Stage R2G importance is invalid.")
    return table


def fit_stages(run, cpus=8, api=None):
    run = resolve_run(run)
    manifest = read_json(run / "manifest.json")
    check_files(run, manifest["files_sha256"])
    check_files(run, read_json(run / "imputation_complete.json")["files_sha256"])
    if manifest["upstream"] != read_json(HERE / "upstream_api.json"):
        raise ValueError("Native source pins differ from preparation.")
    api = api or load_native()
    out = run / OUTPUT
    out.mkdir(exist_ok=True)
    features = read_json(run / "features.json")
    cells = pd.read_csv(run / "cells.tsv.gz", sep="\t")
    peaks = read_json(run / "imputed_peaks.json")
    search = pd.read_csv(run / "search_space.tsv.gz", sep="\t")
    search = search.loc[search.Name.isin(peaks)].copy()
    expression = np.load(run / "rna_expression.npy", mmap_mode="r")
    imputed = np.load(run / "imputed_accessibility.npy", mmap_mode="r")
    if expression.shape != (len(cells), len(features["genes"])) or imputed.shape != (
        len(peaks),
        len(cells),
    ):
        raise ValueError("Expression, accessibility and their labels differ.")
    summary = []
    for stage in sorted(cells.time_idx.unique()):
        path = out / f"t{stage:02d}_region_to_gene.tsv.gz"
        keep = np.flatnonzero(cells.time_idx.to_numpy() == stage)
        names = cells.cell_id.iloc[keep].tolist()
        rna = pd.DataFrame(
            np.asarray(expression[keep]), index=names, columns=features["genes"]
        )
        if not path.exists():
            LOG.info("Stage %d: %d cells, %d pairs", stage, len(keep), len(search))
            access = pd.DataFrame(
                np.asarray(imputed[:, keep]).T, index=names, columns=peaks
            )
            table = stage_table(
                api, rna, access, search, out / "joblib_tmp", cpus, manifest["seed"]
            )
            part = out / f"t{stage:02d}_part.tsv.gz"
            table.to_csv(part, sep="\t", index=False)
            part.replace(path)
        table = pd.read_csv(path, sep="\t")
        summary.append(
            {
                "time_idx": int(stage),
                "n_cells": len(keep),
                "n_pairs": len(table),
                "n_positive": int(table.importance.gt(0).sum()),
                "n_unexpressed_targets": int(
                    (rna[search.Gene.unique()].max() == 0).sum()
                ),
            }
        )
    pd.DataFrame(summary).to_csv(out / "stage_summary.tsv", sep="\t", index=False)
    names = sorted(p.name for p in out.glob("t*_region_to_gene.tsv.gz"))
    write_json(
        out / "stage_fit_report.json",
        {
            "run": str(run),
            "validation_only": manifest["validation_only"],
            "seed": manifest["seed"],
            "temporal_scope": "per_stage",
            "scope": "native R2G GBM within each stage; global topics and imputation reused; not the official all-cell fit",
            "files_sha256": {n: digest(out / n) for n in names},
            "code_sha256": digest(Path(__file__)),
            "native_adapter_sha256": digest(HERE / "native.py"),
        },
    )
    print(pd.DataFrame(summary).to_string(index=False))
    print(f"Stage R2G complete: {out}", flush=True)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--cpus", type=int, default=8)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    fit_stages(args.run, args.cpus)
