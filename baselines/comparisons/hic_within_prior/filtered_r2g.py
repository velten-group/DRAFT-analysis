"""Filter the refitted global SCENIC+ R2G by stage with official DAR/DEG, then score the fixed Hi-C pairs. Nothing is trained.

Copies the original study's step: scenicplus diff_features.get_differential_features at 3ec82fa (their pipeline commit) with
its defaults, on our germline cells only. A pair keeps its global score at a stage if its RE is a DAR and its TG a DEG of that
stage; otherwise it scores zero. All filters are fixed before the Hi-C pairs are read. DAR-only and DEG-only are diagnostics.

    python comparisons/hic_within_prior/filtered_r2g.py --evaluation EVAL --data-root RAW
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scdori_mondo"))
from compare_stage_hic import calibration_threshold, digest, read_json
from hic_common import metrics, region_key

sys.path.insert(0, str(HERE))
from published_scenic import load_inputs
from workflow import ASSAYS, STAGE_LABELS, seed_sample

VARIANTS = {
    "both": "SCENIC+ R2G DAR+DEG filtered",
    "dar": "SCENIC+ R2G DAR filtered",
    "deg": "SCENIC+ R2G DEG filtered",
}
MIN_DISPERSION = {"DAR": 0.05, "DEG": 0.5}  # official values
ADJUSTED_P, LOG2_FC = 0.05, np.log2(1.5)  # official defaults


def differential(adata, stages, min_disp):
    """Official steps and defaults: each stage against the rest. Adata is (cell, feature) and is changed in place."""
    import scanpy as sc

    adata.obs["stage"] = pd.Categorical(np.asarray(stages).astype(str))
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata, min_mean=0.0125, max_mean=3, min_disp=min_disp, max_disp=np.inf
    )
    variable = adata[:, adata.var.highly_variable].copy()
    sc.tl.rank_genes_groups(
        variable, "stage", method="wilcoxon", corr_method="bonferroni"
    )
    selected = {}
    for group in variable.uns["rank_genes_groups"]["names"].dtype.names:
        table = sc.get.rank_genes_groups_df(variable, group=group)
        keep = (table.pvals_adj <= ADJUSTED_P) & (table.logfoldchanges >= LOG2_FC)
        selected[int(group)] = set(table.names[keep])
    return selected, variable.n_vars


def stage_masks(selected, names, key=str):
    """Boolean (stage, feature) from the features selected at each stage."""
    index = {key(name): j for j, name in enumerate(names)}
    masks = np.zeros((len(STAGE_LABELS), len(names)), dtype=bool)
    for stage, features in selected.items():
        masks[stage, [index[key(f)] for f in features if key(f) in index]] = True
    return masks


def normalize_cell(value):
    return re.sub(
        r"#s?0*([0-9]+)$", r"_sample\1", str(value).strip(), flags=re.IGNORECASE
    )


def official_masks(native, root, features):
    import anndata as ad

    cells = pd.read_csv(native / "cells.tsv.gz", sep="\t")
    stages = cells.time_idx.to_numpy()
    rna = ad.read_h5ad(root / "A_rna_focus.h5ad")
    ids = pd.Index([normalize_cell(c) for c in rna.obs_names])
    rows = ids.get_indexer(cells.cell_id.map(normalize_cell))
    if (rows < 0).any():
        raise ValueError("A comparison cell is missing in the RNA data.")
    expression = ad.AnnData(
        rna.layers["counts"][rows], var=pd.DataFrame(index=rna.var_names)
    ).copy()
    del rna
    peaks = read_json(native / "imputed_peaks.json")
    imputed = np.load(native / "imputed_accessibility.npy").T.astype(np.float32)
    if imputed.shape != (len(cells), len(peaks)):
        raise ValueError("Imputed accessibility and its labels differ.")
    access = ad.AnnData(imputed, var=pd.DataFrame(index=peaks))
    deg, n_deg = differential(expression, stages, MIN_DISPERSION["DEG"])
    dar, n_dar = differential(access, stages, MIN_DISPERSION["DAR"])
    counts = pd.DataFrame(
        {
            "time_idx": range(len(STAGE_LABELS)),
            "stage_label": STAGE_LABELS,
            "n_cells": np.bincount(stages, minlength=len(STAGE_LABELS)),
            "n_DAR": [len(dar[s]) for s in range(len(STAGE_LABELS))],
            "n_DEG": [len(deg[s]) for s in range(len(STAGE_LABELS))],
            "n_variable_regions": n_dar,
            "n_variable_genes": n_deg,
        }
    )
    return (
        stage_masks(dar, features["peaks"], region_key),
        stage_masks(deg, features["genes"]),
        counts,
    )


def filtered_scores(scores, codes, n_genes, dar, deg):
    """Global (candidate) scores to (stage, candidate); a filtered pair scores zero."""
    keep_dar, keep_deg = dar[:, codes // n_genes], deg[:, codes % n_genes]
    return {
        "both": scores * (keep_dar & keep_deg),
        "dar": scores * keep_dar,
        "deg": scores * keep_deg,
    }


def stage_row(assay, seed, stage, model, labels, score, calibration):
    test = ~calibration
    cutoff = calibration_threshold(labels[calibration], score[calibration])
    return {
        "dataset": assay,
        "evaluation_seed": seed,
        "time_idx": stage,
        "stage_label": STAGE_LABELS[stage],
        "model": model,
        "F1_protocol": "calibrated_within_prior_test",
        "F1": metrics(labels[test], score[test], cutoff, inclusive=True)["f1"],
        "AUROC": roc_auc_score(labels[test], score[test]),
        "AP": average_precision_score(labels[test], score[test]),
    }


def write_results(evaluation, report, arrays, variants, out):
    """Add (stage, candidate) models to the frozen pairs; the labels are first read here."""
    codes, n_genes = arrays["pair_codes"], int(arrays["n_genes"])
    scores = arrays["scenicplus"]
    previous = pd.read_csv(evaluation / "metrics_by_stage.tsv", sep="\t")
    seed = report["evaluation_seed"]
    rows, coverage = [], []
    for assay_idx, (assay, times) in enumerate(ASSAYS.items()):
        saved = pd.read_csv(evaluation / f"pairs_{assay}_s{seed}.tsv.gz", sep="\t")
        positions, labels, calibration = seed_sample(
            codes, arrays[f"positives_{assay}"], n_genes, seed, assay_idx
        )
        if not (
            np.array_equal(saved.candidate_idx, positions)
            and np.array_equal(saved.label, labels)
            and np.array_equal(saved.fold.eq("calibration"), calibration)
        ):
            raise ValueError("Saved pairs/folds differ from the frozen evaluation.")
        test = ~calibration
        for stage in times:
            old = previous.loc[
                previous.dataset.eq(assay)
                & previous.time_idx.eq(stage)
                & previous.model.eq("SCENIC+ R2G"),
                "AUROC",
            ].item()
            if not np.isclose(
                old, roc_auc_score(labels[test], scores[positions][test]), atol=1e-12
            ):
                raise ValueError("Original R2G AUROC does not match saved scores.")
            for model, values in variants.items():
                score = values[stage, positions]
                rows.append(
                    stage_row(assay, seed, stage, model, labels, score, calibration)
                )
                coverage.append(
                    {
                        "dataset": assay,
                        "time_idx": stage,
                        "model": model,
                        "fraction_nonzero_positive": float(
                            np.mean(score[labels == 1] > 0)
                        ),
                        "fraction_nonzero_negative": float(
                            np.mean(score[labels == 0] > 0)
                        ),
                    }
                )
    points = pd.concat([previous, pd.DataFrame(rows)], ignore_index=True)
    coverage = pd.DataFrame(coverage)
    points.to_csv(out / "metrics_by_stage.tsv", sep="\t", index=False)
    coverage.to_csv(out / "evaluation_coverage.tsv", sep="\t", index=False)
    summary = (
        points.loc[points.model.isin([*variants, "SCENIC+ R2G", "DRAFT"])]
        .groupby(["dataset", "model"], sort=False)
        .AUROC.agg(["mean", "min", "max"])
        .join(
            coverage.groupby(["dataset", "model"])[
                ["fraction_nonzero_positive", "fraction_nonzero_negative"]
            ].mean()
        )
    )
    summary.to_csv(out / "summary.tsv", sep="\t")
    print(summary.round(4).to_string())
    return summary


def run(evaluation, data_root=None, masks=None):
    evaluation = Path(evaluation).expanduser().resolve()
    report, cache, features, _, tg_path = load_inputs(evaluation, data_root)
    native = Path(read_json(cache / "cache_report.json")["scenic_run"])
    # Every filter is fixed here; Hi-C pairs are first read in write_results.
    dar, deg, counts = masks or official_masks(native, tg_path.parent, features)
    with np.load(cache / "scores.npz", allow_pickle=False) as a:
        arrays = {name: a[name] for name in a.files}
    variants = filtered_scores(
        arrays["scenicplus"], arrays["pair_codes"], int(arrays["n_genes"]), dar, deg
    )
    out = evaluation / f"filtered_r2g_{uuid.uuid4().hex[:10]}"
    out.mkdir()
    counts.to_csv(out / "differential_counts.tsv", sep="\t", index=False)
    print(counts.to_string(index=False))
    write_results(
        evaluation,
        report,
        arrays,
        {VARIANTS[name]: values for name, values in variants.items()},
        out,
    )
    (out / "filtered_r2g_report.json").write_text(
        json.dumps(
            {
                "evaluation": str(evaluation),
                "validation_only": bool(report["validation_only"]),
                "code_sha256": digest(Path(__file__)),
                "differential": "scenicplus 3ec82fa get_differential_features defaults on germline cells only",
                "adoption_rule": "use DAR+DEG in the figure only if its mean nonzero share of Hi-C pairs is at least 5% in both assays; fixed before results",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Filtered R2G: {out}", flush=True)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args()
    run(args.evaluation, args.data_root)
