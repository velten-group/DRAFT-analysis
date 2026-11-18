"""Compare downloaded published eRegulons on an existing frozen Hi-C evaluation."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scdori_mondo"))
from compare_stage_hic import calibration_threshold, digest, read_json
from hic_common import metrics
from scenic_hic import (
    CACHE,
    GLOBAL_CACHE,
    fetch_global_reference,
    fetch_reference,
    load_global_reference,
    load_reference,
    map_stage_edges,
)

sys.path.insert(0, str(HERE))
from score_diagnostics import resolve_evaluation
from workflow import ASSAYS, MODELS, STAGE_LABELS, plot_results, seed_sample

PUBLISHED = {
    "stage": "SCENIC+ published stage eRegulons",
    "global": "SCENIC+ published global eRegulons",
}


def load_inputs(evaluation, data_root=None):
    """Verify the original score/feature/TG provenance without loading a model."""
    report = read_json(evaluation / "evaluation_report.json")
    if report["stage_indices"] != {a: list(t) for a, t in ASSAYS.items()}:
        raise ValueError("Original evaluation has a different stage alignment.")
    cache = Path(report["cache"])
    if digest(cache / "cache_report.json") != report["cache_report_sha256"]:
        raise ValueError("Original frozen score cache report changed.")
    frozen = read_json(cache / "cache_report.json")
    if frozen["stage_labels"] != STAGE_LABELS:
        raise ValueError("Frozen feature stage labels differ.")
    for name, sha in frozen["files_sha256"].items():
        if digest(cache / name) != sha:
            raise ValueError(f"Frozen score cache changed: {name}")
    native = Path(frozen["scenic_run"])
    if (
        digest(native / "r2g_report.json")
        != frozen["signature"]["native_report_sha256"]
    ):
        raise ValueError("Native R2G provenance changed.")
    native_report = read_json(native / "r2g_report.json")
    if digest(native / "manifest.json") != native_report["manifest_sha256"]:
        raise ValueError("Native feature manifest changed.")
    manifest = read_json(native / "manifest.json")
    if digest(native / "features.json") != manifest["files_sha256"]["features.json"]:
        raise ValueError("Frozen evaluation feature identifiers changed.")
    features = read_json(native / "features.json")
    comparison = Path(frozen["comparison"])
    if (
        digest(comparison / "comparison_report.json")
        != frozen["signature"]["parent_report_sha256"]
    ):
        raise ValueError("Parent comparison report changed.")
    networks = Path(read_json(comparison / "comparison_report.json")["stage_networks"])
    if (
        digest(networks / "stage_report.json")
        != frozen["signature"]["stage_report_sha256"]
    ):
        raise ValueError("Original input provenance changed.")
    source = read_json(networks / "stage_report.json")["input_sources"]["TG.tsv"]
    tg_path = Path(data_root) / "TG.tsv" if data_root else Path(source["path"])
    if digest(tg_path) != source["sha256"]:
        raise ValueError("Target-gene aliases differ from the original comparison.")
    tg = pd.read_csv(tg_path, sep="\t", dtype=str).fillna("")
    return report, cache, features, tg, tg_path


def project_reference(table, features, tg, codes, *, static=False):
    """Native scores; maximum across duplicate paths; no candidate restriction."""
    genes, peaks = features["genes"], features["peaks"]
    edges, audits, mapping = map_stage_edges(
        table, peaks, genes, tg, return_region_mapping=True
    )
    scores = np.zeros((15, len(codes)), dtype=np.float64)
    for stage, lookup in enumerate(edges):
        source_codes = np.asarray(
            sorted(p * len(genes) + g for p, g in lookup), dtype=np.int64
        )
        positions = np.searchsorted(codes, source_codes)
        inside = positions < len(codes)
        inside[inside] &= codes[positions[inside]] == source_codes[inside]
        matched = source_codes[inside]
        scores[stage, positions[inside]] = [
            lookup[(int(c // len(genes)), int(c % len(genes)))] for c in matched
        ]
        audits[stage].update(
            {
                "mapped_pairs_within_prior": int(inside.sum()),
                "mapped_pairs_outside_prior": int((~inside).sum()),
                "n_fixed_prior_candidates": len(codes),
                "n_positive_score_candidates": int((scores[stage] > 0).sum()),
            }
        )
    if static:
        scores[1:] = scores[0]
        audits = [{**audits[0], "time_idx": t} for t in range(15)]
    return scores, pd.DataFrame(audits), mapping


def compare_published(
    evaluation,
    stage_reference=CACHE,
    global_reference=GLOBAL_CACHE,
    data_root=None,
    make_plots=True,
    fetch_missing=False,
    plot_reference="both",
):
    if plot_reference not in ["stage", "global", "both"]:
        raise ValueError("Plot reference must be stage, global or both.")
    evaluation = Path(evaluation).expanduser().resolve()
    report, cache, features, tg, tg_path = load_inputs(evaluation, data_root)
    if fetch_missing:
        if not (Path(stage_reference) / "reference_manifest.json").is_file():
            fetch_reference(stage_reference)
        if not (Path(global_reference) / "global_reference_manifest.json").is_file():
            fetch_global_reference(global_reference)
    stage_table, stage_meta = load_reference(stage_reference)
    global_table, global_meta = load_global_reference(global_reference)
    references = {
        "stage": (stage_table, stage_meta),
        # Global published eRegulons have one score, repeated without refitting.
        "global": (global_table.assign(time_idx=0), global_meta),
    }
    if not report["validation_only"] and any(
        meta.get("validation_only", False) for _table, meta in references.values()
    ):
        raise ValueError(
            "Software-only references cannot enter a biological comparison."
        )
    with np.load(cache / "scores.npz", allow_pickle=False) as a:
        codes, n_genes = a["pair_codes"], int(a["n_genes"])
        positives = {assay: a[f"positives_{assay}"] for assay in ASSAYS}
        current_scores = {
            "DRAFT": a["draft"],
            "SCENIC+ R2G": a["scenicplus"],
            "scDoRI": a["scdori"],
        }
    if n_genes != len(features["genes"]) or not np.all(np.diff(codes) > 0):
        raise ValueError("Frozen candidate codes or feature count are invalid.")
    previous = pd.read_csv(evaluation / "metrics_by_stage.tsv", sep="\t")
    previous_curves = pd.read_csv(evaluation / "mean_roc_curves.tsv", sep="\t")
    expected_keys = {
        (a, t, m) for a, times in ASSAYS.items() for t in times for m in MODELS
    }
    keys = set(zip(previous.dataset, previous.time_idx, previous.model))
    if len(previous) != len(expected_keys) or keys != expected_keys:
        raise ValueError(
            "Original metrics do not contain exactly 4/9 stages per method."
        )
    out = evaluation / f"published_scenic_{uuid.uuid4().hex[:10]}"
    out.mkdir()
    variants, source_audits, region_maps = {}, [], []
    for variant, (table, _meta) in references.items():
        scores, coverage, mapping = project_reference(
            table, features, tg, codes, static=variant == "global"
        )
        variants[variant] = scores
        source_audits.append(coverage.assign(model=PUBLISHED[variant]))
        region_maps.append(mapping.assign(model=PUBLISHED[variant]))
    rows, cutoffs, curve_rows, evaluation_coverage = [], [], [], []
    grid = np.linspace(0, 1, 1001)
    for assay_idx, (assay, times) in enumerate(ASSAYS.items()):
        print(
            f"Scoring published SCENIC+ {assay.upper()}: {len(times)} stages",
            flush=True,
        )
        seed = report["evaluation_seed"]
        pairs_path = evaluation / f"pairs_{assay}_s{seed}.tsv.gz"
        saved = pd.read_csv(pairs_path, sep="\t")
        positions, labels, calibration = seed_sample(
            codes, positives[assay], n_genes, seed, assay_idx
        )
        if not (
            np.array_equal(saved.candidate_idx, positions)
            and np.array_equal(saved.pair_code, codes[positions])
            and np.array_equal(saved.label, labels)
            and np.array_equal(saved.fold.eq("calibration"), calibration)
        ):
            raise ValueError("Saved pairs/folds differ from the frozen evaluation.")
        test = ~calibration
        # Confirm that the model AUROCs in the figure use exactly these pairs.
        for model, values in current_scores.items():
            for stage in times:
                score = (values if values.ndim == 1 else values[stage])[positions]
                old = previous.loc[
                    previous.dataset.eq(assay)
                    & previous.time_idx.eq(stage)
                    & previous.model.eq(model),
                    "AUROC",
                ].item()
                if not np.isclose(
                    old, roc_auc_score(labels[test], score[test]), atol=1e-12
                ):
                    raise ValueError(
                        f"Original AUROC does not match saved scores: {model}"
                    )
        for variant, values in variants.items():
            stage_curves = []
            for stage in times:
                score = values[stage, positions]
                cutoff = calibration_threshold(labels[calibration], score[calibration])
                rows.append(
                    {
                        "dataset": assay,
                        "evaluation_seed": seed,
                        "time_idx": stage,
                        "stage_label": STAGE_LABELS[stage],
                        "model": PUBLISHED[variant],
                        "F1_protocol": "calibrated_within_prior_test",
                        "F1": metrics(
                            labels[test], score[test], cutoff, inclusive=True
                        )["f1"],
                        "AUROC": roc_auc_score(labels[test], score[test]),
                        "AP": average_precision_score(labels[test], score[test]),
                    }
                )
                cutoffs.append(
                    {
                        "dataset": assay,
                        "time_idx": stage,
                        "model": PUBLISHED[variant],
                        "calibration_cutoff": cutoff,
                    }
                )
                for fold, mask in [("calibration", calibration), ("test", test)]:
                    y, s = labels[mask], score[mask]
                    evaluation_coverage.append(
                        {
                            "dataset": assay,
                            "time_idx": stage,
                            "model": PUBLISHED[variant],
                            "fold": fold,
                            "n_pairs": len(y),
                            "n_positive": int(y.sum()),
                            "fraction_nonzero_positive": float(np.mean(s[y == 1] > 0)),
                            "fraction_nonzero_negative": float(np.mean(s[y == 0] > 0)),
                        }
                    )
                fpr, tpr, _ = roc_curve(
                    labels[test], score[test], drop_intermediate=False
                )
                stage_curves.append(np.interp(grid, fpr, tpr))
            matrix = np.stack(stage_curves)
            curve_rows.extend(
                {
                    "dataset": assay,
                    "model": PUBLISHED[variant],
                    "fpr": x,
                    "tpr_mean": y,
                    "tpr_sd": z,
                }
                for x, y, z in zip(
                    grid, matrix.mean(axis=0), matrix.std(axis=0, ddof=1)
                )
            )
    points = pd.concat([previous, pd.DataFrame(rows)], ignore_index=True)
    curves = pd.concat([previous_curves, pd.DataFrame(curve_rows)], ignore_index=True)
    summary = points.groupby(["dataset", "model"], sort=False)[
        ["F1", "AUROC", "AP"]
    ].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    points.to_csv(out / "metrics_by_stage.tsv", sep="\t", index=False)
    summary.reset_index().to_csv(out / "metrics_summary.tsv", sep="\t", index=False)
    curves.to_csv(out / "mean_roc_curves.tsv", sep="\t", index=False)
    pd.DataFrame(cutoffs).to_csv(out / "calibration_cutoffs.tsv", sep="\t", index=False)
    pd.DataFrame(evaluation_coverage).to_csv(
        out / "evaluation_coverage.tsv", sep="\t", index=False
    )
    pd.concat(source_audits).to_csv(out / "source_coverage.tsv", sep="\t", index=False)
    pd.concat(region_maps).to_csv(out / "region_mapping.tsv", sep="\t", index=False)
    np.savez_compressed(out / "published_scores.npz", pair_codes=codes, **variants)
    if make_plots:
        for variant, model in PUBLISHED.items():
            if plot_reference != "both" and variant != plot_reference:
                continue
            directory = out / f"{variant}_eregulons"
            directory.mkdir()
            selected_models = ["DRAFT", "scDoRI", "Random", model]
            plot_points = points.loc[points.model.isin(selected_models)].copy()
            plot_curves = curves.loc[curves.model.isin(selected_models)].copy()
            plot_points["model"] = plot_points.model.replace({model: "SCENIC+ R2G"})
            plot_curves["model"] = plot_curves.model.replace({model: "SCENIC+ R2G"})
            plot_results(
                plot_points,
                plot_curves,
                directory,
                {
                    **report,
                    "scenic_label": "SCENIC+",
                    "random_label": "Random",
                    "figure_width": 8.0,
                    "annotate_chance": True,
                },
            )
    metadata = {
        "evaluation": str(evaluation),
        "evaluation_report_sha256": digest(evaluation / "evaluation_report.json"),
        "original_metrics_sha256": digest(evaluation / "metrics_by_stage.tsv"),
        "original_curves_sha256": digest(evaluation / "mean_roc_curves.tsv"),
        "cache_report_sha256": report["cache_report_sha256"],
        "pair_files_sha256": {
            a: digest(evaluation / f"pairs_{a}_s{report['evaluation_seed']}.tsv.gz")
            for a in ASSAYS
        },
        "tg_alias_sha256": digest(tg_path),
        "validation_only": report["validation_only"],
        "evaluation_seed": report["evaluation_seed"],
        "plot_reference": plot_reference,
        "plot_display": {
            "scenic_label": "SCENIC+",
            "random_label": "Random",
            "figure_width_inches": 8.0,
            "source_identity": "published stage/global remains explicit in directory names, full metric model names and reference provenance",
        },
        "random_presentation": "original fixed-seed selected-edge control retained exactly; AUROC 0.5 labeled as the theoretical chance reference; no selection of low-scoring random draws",
        "code_sha256": digest(Path(__file__)),
        "sources": {v: meta for v, (_table, meta) in references.items()},
        "protocol": "exploratory comparison of both pre-existing published eRegulon collections; exact frozen within-prior candidates, sampled pairs, folds and corresponding developmental stages; native R2G_importance with maximum across duplicate TF paths and mapped regions; unreported candidates score zero; calibration-only F1 cutoff; held-out AUROC/AP; no best-reference selection or substitution into the original benchmark",
        "scope": "published stage DAR=yes/DEG=yes and static DAR=no/DEG=no eRegulons; distinct from the refitted native R2G core and not trained under the same input/prior conditions",
        "files_sha256": {p.name: digest(p) for p in out.iterdir() if p.is_file()},
    }
    (out / "published_comparison_report.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(summary.to_string(), flush=True)
    print(f"Published SCENIC+ comparison: {out}", flush=True)
    if make_plots and plot_reference in PUBLISHED:
        print(
            f"Selected AUROC figure: {out / (plot_reference + '_eregulons') / 'within_prior_auroc_boxplots.png'}",
            flush=True,
        )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, default=Path("latest"))
    parser.add_argument("--scenic-r2g-run", type=Path, default=Path("latest"))
    parser.add_argument("--stage-reference", type=Path, default=CACHE)
    parser.add_argument("--global-reference", type=Path, default=GLOBAL_CACHE)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--plot-reference",
        choices=["stage", "global", "both"],
        default="both",
        help="Select published-reference figures; all calculated scores and Random draws are retained unchanged.",
    )
    parser.add_argument(
        "--fetch-missing",
        action="store_true",
        help="Download a missing published collection; existing snapshots are reused without refresh.",
    )
    args = parser.parse_args()
    compare_published(
        resolve_evaluation(args.evaluation, args.scenic_r2g_run),
        args.stage_reference,
        args.global_reference,
        args.data_root,
        make_plots=not args.no_plots,
        fetch_missing=args.fetch_missing,
        plot_reference=args.plot_reference,
    )


if __name__ == "__main__":
    main()
