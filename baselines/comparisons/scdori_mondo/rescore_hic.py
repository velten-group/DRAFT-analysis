"""Reassess saved Hi-C pairs against new native or public global SCENIC+ scores.

No model reload, training, resampling, fold changes or label-dependent RE mapping.
The native shared-prior R2G fit is global; the optional public DAR=no/DEG=no
reference remains eRegulon-selected. Both retain their explicit source scope.
"""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from compare_stage_hic import calibration_threshold, digest, plot_f1, read_json
from hic_common import metrics, region_key
from scenic_hic import fetch_global_reference, load_global_reference, map_stage_edges
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
SCENIC_R2G = HERE.parent / "scenicplus_r2g"
MODELS = {"DRAFT": "draft", "scDoRI": "scdori", "SCENIC+": "scenicplus"}


def resolve_native_r2g(value):
    if Path(value) == Path("latest"):
        reports = [
            p
            for p in (SCENIC_R2G / "runs").glob("shared_prior_s*/r2g_report.json")
            if not read_json(p).get("validation_only", False)
        ]
        if not reports:
            raise FileNotFoundError("No completed biological native SCENIC+ R2G run.")
        value = max(reports, key=lambda p: p.stat().st_mtime_ns).parent
    run = Path(value).expanduser().resolve()
    if not run.is_relative_to(SCENIC_R2G / "runs"):
        raise ValueError("Native SCENIC+ run must be inside its own runs directory.")
    return run


def native_r2g_reference(value, comparison, networks, genes, peaks, validation_only):
    """Read portable completed native scores; no SCENIC+ runtime import here."""
    run = resolve_native_r2g(value)
    report = read_json(run / "r2g_report.json")
    if report.get("source_kind") != "native_r2g_shared_prior":
        raise ValueError("Not a completed native shared-prior R2G inference.")
    if report["validation_only"] and not validation_only:
        raise ValueError(
            "Synthetic R2G inference cannot enter a biological comparison."
        )
    if digest(run / "manifest.json") != report["manifest_sha256"]:
        raise ValueError("Native R2G manifest changed.")
    manifest = read_json(run / "manifest.json")
    for name, expected in {
        **manifest["files_sha256"],
        **report["files_sha256"],
    }.items():
        if digest(run / name) != expected:
            raise ValueError(f"Native R2G input/output changed: {name}")
    if report["comparison_report_sha256"] != digest(
        comparison / "comparison_report.json"
    ):
        raise ValueError("Native R2G was prepared for a different original comparison.")
    if report["source_cell_table_sha256"] != digest(networks / "cells.tsv.gz"):
        raise ValueError("Native R2G paired cell cohort differs.")
    features = read_json(run / "features.json")
    if features != {"genes": genes, "peaks": peaks}:
        raise ValueError("Native R2G evaluation features/order differ.")
    table = pd.read_csv(run / "region_to_gene.tsv.gz", sep="\t")
    if len(table) != report["n_scored_candidate_pairs"]:
        raise ValueError("Native R2G score count differs from the completion report.")
    table = table.rename(
        columns={"region": "Region", "target": "Gene", "importance": "R2G_importance"}
    )
    if not np.isfinite(table.R2G_importance).all() or (table.R2G_importance < 0).any():
        raise ValueError("Invalid native R2G scores.")
    return table, {"directory": str(run), **report}, run


def relative_scores(scores, calibration):
    """0–1 positive-evidence display scores; NEVER probabilities or decision rules.

    Use s+/(s+ + q), q = median positive calibration score, or 1 if none exist.
    Signed raw values remain untouched for native F1 and rank metrics.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError("Cannot normalize nonfinite scores.")
    positive = scores[np.asarray(calibration, dtype=bool) & (scores > 0)]
    scale = float(np.median(positive)) if len(positive) else 1.0
    values = np.maximum(scores, 0)
    normalized = np.zeros_like(values)
    with np.errstate(over="ignore", divide="ignore"):
        nonzero = values > 0
        normalized[nonzero] = 1 / (1 + scale / values[nonzero])
    return normalized, scale


def evaluate_scores(labels, scores, calibration):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=np.float64)
    calibration = np.asarray(calibration, dtype=bool)
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite native scores.")
    for mask in [calibration, ~calibration]:
        if set(labels[mask]) != {0, 1}:
            raise ValueError("Each saved fold must contain both label classes.")
    threshold = calibration_threshold(labels[calibration], scores[calibration])
    truth, values = labels[~calibration], scores[~calibration]
    prevalence = float(truth.mean())
    all_positive_f1 = 2 * prevalence / (1 + prevalence)
    result = metrics(truth, values, threshold, inclusive=True)
    return {
        "threshold": threshold,
        **result,
        "auroc": float(roc_auc_score(truth, values)),
        "average_precision": float(average_precision_score(truth, values)),
        "n_positive": int(truth.sum()),
        "n_negative": int((truth == 0).sum()),
        "predicted_positive_fraction": float((values >= threshold).mean()),
        "n_nonzero_positive_all": int(((labels == 1) & (scores > 0)).sum()),
        "n_nonzero_positive_test": int(((truth == 1) & (values > 0)).sum()),
        "all_positive_f1": all_positive_f1,
        "f1_delta_all_positive": result["f1"] - all_positive_f1,
        "chance_auroc": 0.5,
        "chance_average_precision": prevalence,
    }


def validate_pairs(pairs):
    required = {"pair_idx", "RE", "TG_id", "label", "fold", "in_prior"}
    if not required.issubset(pairs):
        raise ValueError("Saved pair table is incomplete.")
    if not np.array_equal(pairs.pair_idx, np.arange(len(pairs))):
        raise ValueError("Saved pair order is invalid.")
    if (
        not pairs.label.isin([0, 1]).all()
        or not pairs.fold.isin(["calibration", "test"]).all()
    ):
        raise ValueError("Invalid saved labels or folds.")
    grouped = pairs.assign(RE=pairs.RE.map(region_key)).groupby(["RE", "TG_id"])
    if grouped.fold.nunique().gt(1).any() or grouped.label.nunique().gt(1).any():
        raise ValueError("A physical pair crosses folds or has conflicting labels.")


def rescore(
    comparison, *, data_root=None, scenic_reference=Path("auto"), scenic_r2g_run=None
):
    comparison = Path(comparison).expanduser().resolve()
    if not comparison.is_relative_to(HERE / "runs"):
        raise ValueError("Comparison must be inside the isolated runs directory.")
    report_path = comparison / "comparison_report.json"
    report = read_json(report_path)
    networks = Path(report["stage_networks"]).expanduser().resolve()
    if not networks.is_relative_to(HERE / "runs"):
        raise ValueError("Stage network source is outside the runs directory.")
    if digest(networks / "stage_report.json") != report["stage_report_sha256"]:
        raise ValueError("Parent stage report changed.")
    meta = read_json(networks / "stage_report.json")
    if bool(meta["validation_only"]) != bool(report["validation_only"]):
        raise ValueError("Parent comparison/stage validation scopes differ.")
    root = (
        Path(data_root or Path(meta["input_sources"]["TG.tsv"]["path"]).parent)
        .expanduser()
        .resolve()
    )
    for name in ["TG.tsv", "RE.txt"]:
        if digest(root / name) != meta["input_sources"][name]["sha256"]:
            raise ValueError(f"Feature source changed: {name}")
    arrays_path = networks / "stage_arrays.npz"
    if digest(arrays_path) != meta["files_sha256"]["stage_arrays.npz"]:
        raise ValueError("Stage feature arrays changed.")
    tg = pd.read_csv(root / "TG.tsv", sep="\t", dtype=str).fillna("")
    with np.load(arrays_path, allow_pickle=False) as arrays:
        stage_genes, stage_peaks = arrays["genes"].tolist(), arrays["peaks"].tolist()
    # Reconstruct the original unlabeled feature pool, never predicted endpoints.
    if (comparison / "evaluation_features.json").is_file():
        expected_features = report.get("evaluation_features_sha256")
        if (
            expected_features
            and digest(comparison / "evaluation_features.json") != expected_features
        ):
            raise ValueError("Saved evaluation feature pool changed.")
        features = read_json(comparison / "evaluation_features.json")
        genes, peaks = features["genes"], features["peaks"]
    else:
        gene_pool = set(tg.FBgn_ID)
        peak_pool = {region_key(p) for p in (root / "RE.txt").read_text().splitlines()}
        genes = [g for g in stage_genes if g in gene_pool]
        peaks = [p for p in stage_peaks if region_key(p) in peak_pool]
    if len(genes) != report["n_common_genes"] or len(peaks) != report["n_common_peaks"]:
        raise ValueError(
            "Cannot reconstruct the exact parent feature pool; rerun compare_stage_hic.py to save it explicitly."
        )
    gi, pi = (
        {g: i for i, g in enumerate(genes)},
        {region_key(p): i for i, p in enumerate(peaks)},
    )
    if len(gi) != len(genes) or len(pi) != len(peaks):
        raise ValueError("Ambiguous evaluation feature pool.")
    scenic_reference = Path(scenic_reference)
    native_run = None
    if scenic_r2g_run is not None:
        if scenic_reference != Path("auto"):
            raise ValueError("Choose either a public reference or a native R2G run.")
        table, scenic_meta, native_run = native_r2g_reference(
            scenic_r2g_run,
            comparison,
            networks,
            genes,
            peaks,
            report["validation_only"],
        )
        reference_report_path = native_run / "r2g_report.json"
        scenic_label = "SCENIC+ native R2G (global)"
    else:
        if scenic_reference == Path("auto"):
            scenic_reference = fetch_global_reference()
        table, scenic_meta = load_global_reference(scenic_reference)
        reference_report_path = scenic_reference / "global_reference_manifest.json"
        scenic_label = "SCENIC+ global reference"
    if scenic_meta["validation_only"] and not report["validation_only"]:
        raise ValueError("Synthetic SCENIC+ data cannot enter a biological comparison.")
    # Broadcast a declared static baseline; this does not infer stage variation.
    expanded = pd.concat(
        [table.assign(time_idx=t) for t in range(15)], ignore_index=True
    )
    scenic_edges, audits, mapping = map_stage_edges(
        expanded, peaks, genes, tg, return_region_mapping=True
    )
    out = (native_run or comparison) / f"rescored_hic_{uuid.uuid4().hex[:10]}"
    out.mkdir(exist_ok=False)
    rows, score_hashes, scales = [], {}, []
    for assay, times in [("spg", range(2, 6)), ("spc", range(6, 15))]:
        for universe in ["all_features", "within_prior"]:
            pair_path = comparison / f"pairs_{assay}_{universe}.tsv.gz"
            if not pair_path.exists():
                continue
            pairs = pd.read_csv(pair_path, sep="\t")
            validate_pairs(pairs)
            calibration = pairs.fold.eq("calibration").to_numpy()
            labels = pairs.label.to_numpy()
            codes = [(pi[region_key(p)], gi[g]) for p, g in zip(pairs.RE, pairs.TG_id)]
            score_hashes[pair_path.name] = digest(pair_path)
            pairs.to_csv(out / pair_path.name, sep="\t", index=False)
            for stage in times:
                name = f"scores_{assay}_{universe}_t{stage:02d}.tsv.gz"
                path = comparison / name
                saved = pd.read_csv(path, sep="\t")
                if len(saved) != len(pairs) or not np.array_equal(
                    saved.pair_idx, pairs.pair_idx
                ):
                    raise ValueError(f"Saved score order differs: {name}")
                score_hashes[name] = digest(path)
                scores = saved[["pair_idx", "draft_score", "scdori_score"]].copy()
                scores["scenicplus_score"] = [
                    scenic_edges[stage].get(pair, 0.0) for pair in codes
                ]
                scores["scenicplus_reported"] = [
                    pair in scenic_edges[stage] for pair in codes
                ]
                if universe == "all_features":
                    audits[stage].update(
                        dataset=assay,
                        n_positive_pairs=int(labels.sum()),
                        n_positive_reported=int(
                            ((labels == 1) & scores.scenicplus_reported).sum()
                        ),
                        n_positive_nonzero=int(
                            ((labels == 1) & scores.scenicplus_score.gt(0)).sum()
                        ),
                    )
                for model, key in MODELS.items():
                    values = scores[f"{key}_score"].to_numpy()
                    normalized, scale = relative_scores(values, calibration)
                    scores[f"{key}_relative_score"] = normalized
                    common = {
                        "dataset": assay,
                        "time_idx": stage,
                        "stage_label": meta["stage_labels"][stage],
                        "universe": universe,
                        "model": model,
                    }
                    scales.append({**common, "positive_calibration_median": scale})
                    rows.append(
                        {
                            **common,
                            "protocol": "calibrated_disjoint_test",
                            **evaluate_scores(labels, values, calibration),
                        }
                    )
                scores.to_csv(out / name, sep="\t", index=False)
    if not rows:
        raise ValueError("No complete saved pair/score tables found.")
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "hic_metrics_by_stage.tsv", sep="\t", index=False)
    index = ["dataset", "time_idx", "stage_label", "universe"]
    wide_tables = []
    for metric in ["f1", "auroc", "average_precision"]:
        wide = (
            summary.pivot(index=index, columns="model", values=metric)
            .reindex(columns=MODELS)
            .reset_index()
        )
        wide.insert(len(index), "metric", metric)
        wide_tables.append(wide)
    pd.concat(wide_tables, ignore_index=True).to_csv(
        out / "hic_metrics_comparison.tsv", sep="\t", index=False
    )
    pd.DataFrame(scales).to_csv(
        out / "relative_score_scales.tsv", sep="\t", index=False
    )
    pd.DataFrame(audits).to_csv(out / "scenic_coverage.tsv", sep="\t", index=False)
    mapping.to_csv(out / "scenic_region_mapping.tsv", sep="\t", index=False)
    (out / "rescore_report.json").write_text(
        json.dumps(
            {
                "parent_comparison": str(comparison),
                "parent_report_sha256": digest(report_path),
                "input_pair_score_sha256": score_hashes,
                "validation_only": bool(report["validation_only"]),
                "scenic_reference": scenic_meta,
                "scenic_reference_manifest_sha256": digest(reference_report_path),
                "code_sha256": digest(Path(__file__)),
                "evaluation_features": {"genes": genes, "peaks": peaks},
                "normalization": "s_positive/(s_positive + median_positive_calibration_score), scale 1 if no positive calibration scores; zero/no-support stays zero; display only, not probability; native signed scores retained for all metrics",
                "evaluation": "reuse exact saved pairs, labels, folds and native DRAFT/scDoRI scores; per-method F1 cutoff fitted only to calibration; AUROC and average precision on untouched test fold, using native scores; global SCENIC+ scores are a constant baseline across stages, not independent temporal observations",
                "scenic_source_kind": "native_r2g_shared_prior"
                if native_run
                else "published_global_eregulon_selected",
            },
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    plot_f1(summary, out, {**meta, "scenic_label": scenic_label})
    print(wide_tables[0].to_string(index=False), flush=True)
    print(
        "Test-fold means (descriptive; repeated stages are not independent assays):",
        flush=True,
    )
    print(
        summary.groupby(["universe", "model"])[
            [
                "f1",
                "auroc",
                "average_precision",
                "all_positive_f1",
                "chance_average_precision",
            ]
        ]
        .mean()
        .to_string(),
        flush=True,
    )
    print(f"Rescored comparison: {out}", flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, default=Path("latest"))
    parser.add_argument("--run", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--scenic-reference", type=Path, default=Path("auto"))
    parser.add_argument(
        "--scenic-r2g-run",
        type=Path,
        help="Completed native SCENIC+ RE–TG run, or latest; uses new scores instead of public eRegulons.",
    )
    args = parser.parse_args()
    if args.comparison == Path("latest") and args.scenic_r2g_run is not None:
        # Pin the original saved pairs/folds used to prepare this inference.
        args.scenic_r2g_run = resolve_native_r2g(args.scenic_r2g_run)
        args.comparison = Path(
            read_json(args.scenic_r2g_run / "r2g_report.json")["comparison"]
        )
        if args.run is not None and not args.comparison.resolve().is_relative_to(
            args.run.expanduser().resolve()
        ):
            parser.error("Native R2G was prepared from a different scDoRI run.")
    if args.comparison == Path("latest"):
        if args.run is None:
            parser.error("Pass --run or --comparison /EXACT/COMPARISON/PATH.")
        run = args.run.expanduser().resolve()
        if not run.is_relative_to(HERE / "runs"):
            parser.error("Run must be inside the isolated runs directory.")
        reports = list(
            run.glob("stage_networks_*/comparison_hic_*/comparison_report.json")
        )
        if not reports:
            parser.error(
                "No saved three-method comparison found; run compare_stage_hic.py first."
            )
        args.comparison = max(reports, key=lambda p: p.stat().st_mtime_ns).parent
    rescore(
        args.comparison,
        data_root=args.data_root,
        scenic_reference=args.scenic_reference,
        scenic_r2g_run=args.scenic_r2g_run,
    )


if __name__ == "__main__":
    main()
