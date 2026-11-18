"""Official Dictys context-specific OU reconstruction with the mouse DRAFT prior."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mouse_benchmark"))
from common import (
    digest,
    mark_complete,
    model_run,
    read_json,
    resolve_prepared,
    validate_counts,
    verify_complete,
)

PIN = "f409a632eec99c2cbc71f3c69b779964649bfc8b"
NETWORK_SHA256 = "18a03d4120e5d89c5981795b6c465cdb2b904c85e236611744b0815f928b2439"


def native_software() -> dict:
    # Dictys imports matplotlib through its package initializer; keep its cache local.
    cache = Path(__file__).resolve().parent / "runs" / ".mplconfig"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import dictys.network
    import pyro
    import torch

    api = Path(dictys.network.__file__)
    canonical_hash = hashlib.sha256(
        api.read_bytes().replace(b"\r\n", b"\n")
    ).hexdigest()
    if canonical_hash != NETWORK_SHA256:
        raise RuntimeError(
            f"Dictys network API differs from pinned commit {PIN}; reinstall the pinned source."
        )
    if pyro.__version__ != "1.9.1":
        raise RuntimeError(
            "Use pyro-ppl==1.9.1 in the isolated mouse_dictys environment."
        )
    return {
        "dictys_commit": PIN,
        "network_sha256": canonical_hash,
        "network_file_sha256": digest(api),
        "torch": torch.__version__,
        "pyro": pyro.__version__,
        "numpy": importlib.metadata.version("numpy"),
    }


def train(
    prepared: Path,
    cpus: int,
    device: str,
    steps: int = 4000,
    clusters: list[int] | None = None,
) -> Path:
    manifest = read_json(prepared / "manifest.json")
    if not manifest["raw_counts_available"]:
        raise ValueError(
            "Original RNA counts are missing. Prepare a new run with --counts-root and --confirm-raw-counts; normalized cluster tables cannot be used for official Dictys."
        )
    software = native_software()
    import torch
    from dictys.network import normalize, reconstruct

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but unavailable. Explicitly choose --device cpu for a CPU run."
        )
    if steps != 4000 and not manifest.get("validation_only"):
        raise ValueError(
            "Biological comparisons use the prespecified 4000 steps; short runs are software tests only."
        )
    configuration = {
        "software": software,
        "cpus": cpus,
        "device": device,
        "steps": steps,
        "seed": manifest["seed"],
        "lr": 0.01,
        "lrd": 0.999,
        "npc": 0,
        "dtype": "float",
        "normalization": 3,
        "model": "official Dictys context-specific OU model, one network per cluster",
        "prior": "original DRAFT per-cluster binary allowed TF->TG mask; native footprinting not run",
        "score": "absolute official normalized direct edge weight; signed native weights also retained",
    }
    run = model_run("mouse_dictys", prepared, configuration)
    selected = clusters or manifest["clusters"]
    if not set(selected).issubset(manifest["clusters"]):
        raise ValueError("Requested cluster is not in the prepared inputs.")
    for cid in selected:
        folder = run / f"C{cid}"
        folder.mkdir(exist_ok=True)
        if (folder / "complete.json").exists():
            verify_complete(folder)
            print(f"C{cid}: verified completed Dictys output", flush=True)
            continue
        inputs = prepared / f"C{cid}"
        counts = pd.read_csv(inputs / "counts.tsv.gz", sep="\t", index_col=0)
        validate_counts(counts)
        prior = pd.read_csv(inputs / "prior.tsv.gz", sep="\t")
        regulators = (
            (inputs / "regulators.txt").read_text(encoding="utf-8").splitlines()
        )
        mask = pd.DataFrame(0, index=regulators, columns=counts.index, dtype=np.int8)
        tf_idx = {name: i for i, name in enumerate(regulators)}
        gene_idx = {name: i for i, name in enumerate(counts.index)}
        for row in prior.itertuples(index=False):
            mask.iat[tf_idx[row.TF], gene_idx[row.TG]] = 1
        mask.to_csv(folder / "binlinking.tsv.gz", sep="\t")
        print(
            f"C{cid}: official Dictys OU, {steps} steps on {device}; log {folder / 'dictys.log'}",
            flush=True,
        )
        handler = logging.FileHandler(folder / "dictys.log", mode="w", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        root_log = logging.getLogger()
        old_level = root_log.level
        root_log.addHandler(handler)
        root_log.setLevel(logging.INFO)
        try:
            reconstruct(
                str(inputs / "counts.tsv.gz"),
                str(folder / "binlinking.tsv.gz"),
                str(folder / "net_weight.tsv.gz"),
                str(folder / "net_meanvar.tsv.gz"),
                str(folder / "net_covfactor.tsv.gz"),
                str(folder / "net_loss.tsv.gz"),
                str(folder / "net_stats.tsv.gz"),
                lr=0.01,
                lrd=0.999,
                nstep=steps,
                npc=0,
                model="ou",
                nstep_report=100,
                rseed=manifest["seed"],
                device=device,
                dtype="float",
                nth=cpus,
            )
            normalize(
                str(folder / "net_weight.tsv.gz"),
                str(folder / "net_meanvar.tsv.gz"),
                str(folder / "net_covfactor.tsv.gz"),
                str(folder / "net_nweight.tsv.gz"),
                norm=3,
                nth=cpus,
            )
        finally:
            root_log.removeHandler(handler)
            handler.close()
            root_log.setLevel(old_level)
        weights = pd.read_csv(folder / "net_nweight.tsv.gz", sep="\t", index_col=0)
        if not np.isfinite(weights.to_numpy()).all():
            raise ValueError(
                "Official Dictys produced non-finite weights; no substitute scores are generated."
            )
        rows = []
        for edge in prior.itertuples(index=False):
            value = float(weights.loc[edge.TF, edge.TG])
            rows.append(
                {
                    "TF": edge.TF,
                    "TG": edge.TG,
                    "score": abs(value),
                    "signed_score": value,
                }
            )
        pd.DataFrame(rows).to_csv(folder / "edges.tsv.gz", sep="\t", index=False)
        mark_complete(
            folder,
            {
                "model": "Dictys",
                "cluster": cid,
                "n_edges": len(rows),
                "n_genes": counts.shape[0],
                "n_cells": counts.shape[1],
                "input_manifest_sha256": digest(prepared / "manifest.json"),
                "validation_only": manifest.get("validation_only", False),
                "score": configuration["score"],
            },
        )
        print(f"C{cid}: official Dictys complete", flush=True)
    print(f"Dictys results: {run}", flush=True)
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-run", type=Path, default=Path("latest"))
    parser.add_argument("--cpus", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--clusters", type=int, nargs="+")
    args = parser.parse_args()
    if args.cpus < 1:
        parser.error("--cpus must be positive")
    train(
        resolve_prepared(args.prepared_run),
        args.cpus,
        args.device,
        clusters=args.clusters,
    )


if __name__ == "__main__":
    main()
