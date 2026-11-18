"""Frozen full-prior scores for repeated evaluation; no model training."""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SCDORI = HERE.parent / "scdori_mondo"
sys.path.insert(0, str(SCDORI))

from compare_stage_hic import (
    choose_draft_checkpoint,
    digest,
    load_draft,
    raw_draft_scores,
    read_json,
)
from hic_common import map_positives, region_key
from rescore_hic import native_r2g_reference, resolve_native_r2g
from scenic_hic import map_stage_edges

LOG = logging.getLogger(__name__)


def build_cache(scenic_run, root=None, checkpoint=Path("auto"), draft_loader=None):
    run = resolve_native_r2g(scenic_run)
    scenic_report = read_json(run / "r2g_report.json")
    comparison = Path(scenic_report["comparison"]).resolve()
    parent = read_json(comparison / "comparison_report.json")
    networks = Path(parent["stage_networks"]).resolve()
    if not comparison.is_relative_to(SCDORI / "runs") or not networks.is_relative_to(
        SCDORI / "runs"
    ):
        raise ValueError("Original comparison must stay inside isolated scDoRI runs.")
    if digest(networks / "stage_report.json") != parent["stage_report_sha256"]:
        raise ValueError("Parent stage report changed.")
    stage = read_json(networks / "stage_report.json")
    root = (
        Path(root or Path(stage["input_sources"]["TG.tsv"]["path"]).parent)
        .expanduser()
        .resolve()
    )
    for name, source in stage["input_sources"].items():
        if digest(root / name) != source["sha256"]:
            raise ValueError(f"Shared input source changed: {name}")
    for name in ["stage_arrays.npz", "cells.tsv.gz"]:
        if digest(networks / name) != stage["files_sha256"][name]:
            raise ValueError(f"Shared stage arrays/cells changed: {name}")
    features = read_json(run / "features.json")
    genes, peaks = features["genes"], features["peaks"]
    if len(genes) != parent["n_common_genes"] or len(peaks) != parent["n_common_peaks"]:
        raise ValueError("Native R2G and original feature pool counts differ.")
    table, scenic_meta, _ = native_r2g_reference(
        run, comparison, networks, genes, peaks, parent["validation_only"]
    )
    checkpoint = choose_draft_checkpoint(Path(checkpoint), root)
    if digest(checkpoint) != parent["draft_checkpoint_sha256"]:
        raise ValueError("DRAFT checkpoint differs from the existing comparison.")
    truths = {
        assay: {
            "path": str(root / f"{assay}_hic.tsv"),
            "sha256": digest(root / f"{assay}_hic.tsv"),
        }
        for assay in ["spg", "spc"]
    }
    for assay, record in truths.items():
        if record["sha256"] != parent["sources"][assay]["sha256"]:
            raise ValueError(f"Hi-C labels changed since original comparison: {assay}")
    signature = {
        "native_report_sha256": digest(run / "r2g_report.json"),
        "parent_report_sha256": digest(comparison / "comparison_report.json"),
        "stage_report_sha256": digest(networks / "stage_report.json"),
        "draft_checkpoint_sha256": digest(checkpoint),
        "truth_sources": truths,
        "cache_code_sha256": digest(Path(__file__)),
        "draft_scoring_sources": {
            name: digest(HERE.parents[1] / name)
            for name in [
                "comparisons/scdori_mondo/compare_stage_hic.py",
                "comparisons/scdori_mondo/hic_common.py",
                "comparisons/scdori_mondo/scenic_hic.py",
                "comparisons/scdori_mondo/rescore_hic.py",
                "src/drosophila_gnn/infer.py",
                "src/drosophila_gnn/analysis/repro.py",
                "src/drosophila_gnn/models/gnn.py",
                "src/drosophila_gnn/preprocess.py",
                "src/drosophila_gnn/graph_builder.py",
                "configs/germ_reproduce.yaml",
            ]
        },
    }
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[
        :12
    ]
    cache = run / f"within_prior_score_cache_{key}"
    if (cache / "cache_report.json").exists():
        report = read_json(cache / "cache_report.json")
        if report["signature"] != signature:
            raise ValueError("Score cache provenance differs.")
        for name, expected in report["files_sha256"].items():
            if digest(cache / name) != expected:
                raise ValueError(f"Score cache changed: {name}")
        LOG.info("Reusing frozen complete-prior scores: %s", cache)
        return cache
    gi = {g: i for i, g in enumerate(genes)}
    pi = {region_key(p): i for i, p in enumerate(peaks)}
    if len(gi) != len(genes) or len(pi) != len(peaks):
        raise ValueError("Duplicate shared feature identifiers.")
    with np.load(networks / "stage_arrays.npz", allow_pickle=False) as arrays:
        old_genes, old_peaks = arrays["genes"].tolist(), arrays["peaks"].tolist()
        source_pairs = [
            (pi[region_key(old_peaks[p])], gi[old_genes[g]], j)
            for j, (g, p) in enumerate(zip(arrays["pair_gene"], arrays["pair_peak"]))
            if old_genes[g] in gi and region_key(old_peaks[p]) in pi
        ]
        if not source_pairs:
            raise ValueError("No fixed common RE–TG candidates.")
        p, g, positions = np.asarray(source_pairs, dtype=np.int64).T
        codes = p * len(genes) + g
        order = np.argsort(codes)
        p, g, positions, codes = [x[order] for x in [p, g, positions, codes]]
        if len(np.unique(codes)) != len(codes):
            raise ValueError("Duplicate fixed candidate pairs.")
        scdori = arrays["re_tg_scores"][:, positions].copy()
    # Freeze the candidate universe and every method's scores before reading labels.
    tg = pd.read_csv(root / "TG.tsv", sep="\t", dtype=str).fillna("")
    scenic_edges, _ = map_stage_edges(table.assign(time_idx=0), peaks, genes, tg)
    scenic = np.asarray(
        [scenic_edges[0].get((int(a), int(b)), 0) for a, b in zip(p, g)],
        dtype=np.float64,
    )
    candidate_codes = set(codes.tolist())
    outside = [
        (a, b) for a, b in scenic_edges[0] if a * len(genes) + b not in candidate_codes
    ]
    if outside:
        raise ValueError("Native R2G contains scores outside shared fixed candidates.")
    LOG.info(
        "Loading existing DRAFT checkpoint once to score %d fixed candidates; no training",
        len(codes),
    )
    draft = (draft_loader or load_draft)(root, checkpoint)
    model_seed = draft["checkpoint"].get("seed")
    if model_seed != stage["seed"] or model_seed != scenic_meta["seed"]:
        raise ValueError("Frozen model seeds differ.")
    if draft["germline_types"] != [f"Germline_{t}" for t in range(15)]:
        raise ValueError("DRAFT stage order differs.")
    observed = draft["artifacts"]["common"]["anno"]
    observed = observed.loc[
        observed.cell_type.isin(draft["germline_types"]), ["cell_id", "cell_type"]
    ].copy()
    observed["time_idx"] = observed.cell_type.str.extract(r"(\d+)$")[0].astype(int)
    cells = pd.read_csv(networks / "cells.tsv.gz", sep="\t")
    if set(zip(observed.cell_id, observed.time_idx)) != set(
        zip(cells.cell_id, cells.time_idx)
    ):
        raise ValueError("Frozen DRAFT and baseline cell-stage cohorts differ.")
    aliases = dict(zip(tg.gene_name, tg.FBgn_ID))
    aliases.update({gene: gene for gene in tg.FBgn_ID})
    dp = {region_key(str(peak)): j for j, peak in enumerate(draft["maps"]["re"])}
    dg = {
        aliases.get(str(gene), str(gene)): j
        for j, gene in enumerate(draft["maps"]["tg"])
    }
    if len(dp) != len(draft["maps"]["re"]) or len(dg) != len(draft["maps"]["tg"]):
        raise ValueError("Ambiguous frozen DRAFT node identifiers.")
    draft_values = raw_draft_scores(
        draft["out_re"],
        draft["out_tg"],
        np.asarray([dp[region_key(peaks[j])] for j in p]),
        np.asarray([dg[genes[j]] for j in g]),
    )
    if scdori.shape != draft_values.shape or draft_values.shape != (15, len(codes)):
        raise ValueError("Frozen model score shapes differ.")
    if not all(np.isfinite(values).all() for values in [draft_values, scdori, scenic]):
        raise ValueError("Nonfinite frozen model scores.")
    positive_positions, coverage = {}, []
    for assay in ["spg", "spc"]:
        positives, audit = map_positives(root / f"{assay}_hic.tsv", peaks, genes, tg)
        gold_codes = positives[:, 0] * len(genes) + positives[:, 1]
        inside = np.isin(gold_codes, codes)
        if inside.sum() < 4:
            raise ValueError(f"Too few unique within-prior positives: {assay}")
        positive_positions[assay] = np.searchsorted(codes, gold_codes[inside])
        coverage.append(
            {
                "dataset": assay,
                **audit,
                "n_inside_prior": int(inside.sum()),
                "n_outside_prior": int((~inside).sum()),
            }
        )
    cache.mkdir(exist_ok=True)
    np.savez_compressed(
        cache / "scores.npz",
        pair_codes=codes,
        n_genes=np.asarray(len(genes)),
        draft=draft_values,
        scdori=scdori,
        scenicplus=scenic,
        positives_spg=positive_positions["spg"],
        positives_spc=positive_positions["spc"],
    )
    pd.DataFrame(coverage).to_csv(cache / "coverage.tsv", sep="\t", index=False)
    report = {
        "signature": signature,
        "scenic_run": str(run),
        "comparison": str(comparison),
        "validation_only": bool(parent["validation_only"]),
        "model_seed": model_seed,
        "n_candidates": len(codes),
        "stage_labels": stage["stage_labels"],
        "draft_device": str(draft["out_re"].device),
        "score_scope": "complete shared fixed RE–TG prior; original DRAFT sigmoid decoder, native scDoRI stage coefficients, new native global SCENIC+ R2G; no model training, score normalization, Top-K or significance filtering",
        "files_sha256": {
            name: digest(cache / name) for name in ["scores.npz", "coverage.tsv"]
        },
    }
    (cache / "cache_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    LOG.info("Frozen score cache complete: %s", cache)
    return cache
