"""Read-only score diagnostics; alternatives use calibration labels only."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scdori_mondo"))

from compare_stage_hic import digest, read_json
from hic_common import region_key
from rescore_hic import resolve_native_r2g

# Import the analysis entry point from its own directory, before baseline paths.
sys.path.insert(0, str(HERE))
from workflow import ASSAYS, MODELS, seed_sample


def resolve_evaluation(value, scenic_run):
    if Path(value) == Path("latest"):
        reports = list(
            resolve_native_r2g(scenic_run).glob("within_prior_*/evaluation_report.json")
        )
        if not reports:
            raise FileNotFoundError("No completed within-prior evaluation.")
        value = max(reports, key=lambda p: p.stat().st_mtime_ns).parent
    return Path(value).expanduser().resolve()


def map_r2g_alternative(table, features, codes, original):
    """Official importance × abs(Spearman rho), mapped to every frozen candidate."""
    pi = {region_key(p): j for j, p in enumerate(features["peaks"])}
    gi = {g: j for j, g in enumerate(features["genes"])}
    n = len(gi)
    raw, weighted = {}, {}
    n_invalid_rho = 0
    for row in table.itertuples(index=False):
        code = pi[region_key(row.region)] * n + gi[row.target]
        if code in raw:
            raise ValueError("Duplicate native R2G candidate.")
        importance, rho = float(row.importance), float(row.rho)
        raw[code] = importance
        n_invalid_rho += int(not np.isfinite(rho))
        weighted[code] = importance * abs(rho) if np.isfinite(rho) else 0.0
    native = np.asarray([raw.get(int(c), 0.0) for c in codes])
    if not np.allclose(native, original, rtol=1e-10, atol=1e-12):
        raise ValueError("Native R2G importance differs from the frozen score cache.")
    return np.asarray([weighted.get(int(c), 0.0) for c in codes]), n_invalid_rho


def load_alternatives(cache, frozen, codes, scores):
    native = Path(frozen["scenic_run"])
    native_report = read_json(native / "r2g_report.json")
    if (
        digest(native / "r2g_report.json")
        != frozen["signature"]["native_report_sha256"]
    ):
        raise ValueError("Native R2G report changed since score caching.")
    if digest(native / "manifest.json") != native_report["manifest_sha256"]:
        raise ValueError("Native feature manifest changed.")
    manifest = read_json(native / "manifest.json")
    for name, expected in {
        **manifest["files_sha256"],
        **native_report["files_sha256"],
    }.items():
        if digest(native / name) != expected:
            raise ValueError(f"Native R2G source/output changed: {name}")
    features = read_json(native / "features.json")
    weighted, invalid = map_r2g_alternative(
        pd.read_csv(native / "region_to_gene.tsv.gz", sep="\t"),
        features,
        codes,
        scores["SCENIC+ R2G"],
    )
    comparison = Path(frozen["comparison"])
    if (
        digest(comparison / "comparison_report.json")
        != frozen["signature"]["parent_report_sha256"]
    ):
        raise ValueError("Original comparison report changed.")
    networks = Path(read_json(comparison / "comparison_report.json")["stage_networks"])
    if (
        digest(networks / "stage_report.json")
        != frozen["signature"]["stage_report_sha256"]
    ):
        raise ValueError("Stage score source changed.")
    stage = read_json(networks / "stage_report.json")
    if (
        digest(networks / "stage_arrays.npz")
        != stage["files_sha256"]["stage_arrays.npz"]
    ):
        raise ValueError("scDoRI coefficient archive changed.")
    gi = {g: j for j, g in enumerate(features["genes"])}
    pi = {region_key(p): j for j, p in enumerate(features["peaks"])}
    with np.load(networks / "stage_arrays.npz", allow_pickle=False) as a:
        genes, peaks = a["genes"].tolist(), a["peaks"].tolist()
        mapping = [
            (pi[region_key(peaks[p])] * len(gi) + gi[genes[g]], j)
            for j, (g, p) in enumerate(zip(a["pair_gene"], a["pair_peak"]))
            if genes[g] in gi and region_key(peaks[p]) in pi
        ]
        mapping.sort()
        if not np.array_equal(np.asarray([c for c, _j in mapping]), codes):
            raise ValueError("scDoRI static-weight candidate mapping differs.")
        positions = [j for _c, j in mapping]
        if not np.allclose(
            a["re_tg_scores"][:, positions], scores["scDoRI"], rtol=1e-10, atol=1e-12
        ):
            raise ValueError("scDoRI stage coefficients differ from the frozen cache.")
        static = a["global_gene_peak"][positions].copy()
    return {
        "SCENIC+ importance_x_abs_rho": ("SCENIC+ R2G", weighted),
        "scDoRI static decoder weight": ("scDoRI", static),
    }, {
        "n_native_r2g_nonfinite_rho": invalid,
        "nonfinite_rho_policy": "zero correlation evidence, recorded; original importance untouched",
    }


def within_target_auc(y, score, target):
    values = []
    for group in np.unique(target):
        mask = target == group
        if len(np.unique(y[mask])) == 2:
            values.append(roc_auc_score(y[mask], score[mask]))
    return (float(np.mean(values)) if values else np.nan), len(values)


def paired_target_bootstrap(y, matrix, target, repeats=100, seed=2025):
    """Resample TG clusters; retain shared pairs, stage dependence and pairing."""
    _groups, inverse = np.unique(target, return_inverse=True)
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(repeats):
        counts = np.bincount(
            rng.integers(0, len(_groups), size=len(_groups)), minlength=len(_groups)
        )
        weights = counts[inverse]
        if weights[y == 0].sum() == 0 or weights[y == 1].sum() == 0:
            continue
        results.append(
            [
                np.mean(
                    [roc_auc_score(y, stage, sample_weight=weights) for stage in method]
                )
                for method in matrix
            ]
        )
    if not results:
        raise ValueError("No bootstrap draw contains both classes.")
    return np.asarray(results)


def audit_evaluation(evaluation, repeats=100, alternatives=None):
    evaluation = Path(evaluation).resolve()
    report = read_json(evaluation / "evaluation_report.json")
    if repeats < 20:
        raise ValueError("Use at least 20 bootstrap repeats.")
    if alternatives is not None and not report["validation_only"]:
        raise ValueError(
            "Injected software alternatives cannot enter biological diagnostics."
        )
    cache = Path(report["cache"])
    if digest(cache / "cache_report.json") != report["cache_report_sha256"]:
        raise ValueError("Frozen score cache report changed.")
    frozen = read_json(cache / "cache_report.json")
    for name, sha in frozen["files_sha256"].items():
        if digest(cache / name) != sha:
            raise ValueError(f"Frozen scores changed: {name}")
    with np.load(cache / "scores.npz", allow_pickle=False) as a:
        codes, n_genes = a["pair_codes"], int(a["n_genes"])
        scores = {
            "DRAFT": a["draft"],
            "SCENIC+ R2G": a["scenicplus"],
            "scDoRI": a["scdori"],
        }
        positives = {assay: a[f"positives_{assay}"] for assay in ASSAYS}
    if alternatives is None:
        alternatives, source_audit = load_alternatives(cache, frozen, codes, scores)
    else:
        source_audit = {"validation_only": True}
    variants = {f"{model} current": (model, values) for model, values in scores.items()}
    variants.update(alternatives)
    for name, (_model, values) in variants.items():
        if (
            values.shape not in [(len(codes),), (15, len(codes))]
            or not np.isfinite(values).all()
        ):
            raise ValueError(f"Invalid diagnostic variant: {name}")
    out = evaluation / f"score_diagnostics_{uuid.uuid4().hex[:10]}"
    out.mkdir()
    rows, ci_rows = [], []
    with np.load(
        evaluation / "random_edges_by_stage.npz", allow_pickle=False
    ) as selected:
        context_path = Path(report["random_context"])
        if (
            digest(context_path / "context_report.json")
            != report["random_context_report_sha256"]
        ):
            raise ValueError("Actual Random selection context changed.")
        from random_control import load_context

        context = load_context(context_path, cache)
        for assay_idx, (assay, times) in enumerate(ASSAYS.items()):
            print(
                f"Auditing {assay.upper()}: calibration scoring variants and {repeats} paired TG-cluster resamples",
                flush=True,
            )
            seed = report["evaluation_seed"]
            table = pd.read_csv(evaluation / f"pairs_{assay}_s{seed}.tsv.gz", sep="\t")
            positions, labels, calibration = seed_sample(
                codes, positives[assay], n_genes, seed, assay_idx
            )
            if not (
                np.array_equal(table.candidate_idx, positions)
                and np.array_equal(table.label, labels)
                and np.array_equal(table.fold.eq("calibration"), calibration)
                and np.array_equal(table.pair_code, codes[positions])
            ):
                raise ValueError(
                    "Saved evaluation pairs/folds differ from frozen protocol."
                )
            target = codes[positions] % n_genes
            for name, (model, values) in variants.items():
                for stage in times:
                    score = (values if values.ndim == 1 else values[stage])[positions][
                        calibration
                    ]
                    y = labels[calibration]
                    conditional, n_targets = within_target_auc(
                        y, score, target[calibration]
                    )
                    rows.append(
                        {
                            "dataset": assay,
                            "time_idx": stage,
                            "model": model,
                            "variant": name,
                            "fold": "calibration",
                            "calibration_AUROC": roc_auc_score(y, score),
                            "within_target_macro_AUROC": conditional,
                            "n_evaluable_targets": n_targets,
                            "fraction_zero": float(np.mean(score == 0)),
                            "fraction_negative": float(np.mean(score < 0)),
                            "n_unique_scores": len(np.unique(score)),
                        }
                    )
            test = ~calibration
            matrices = []
            for model in MODELS:
                if model == "Random":
                    data = np.stack(
                        [
                            np.isin(
                                context["candidate_codes"][positions[test]],
                                selected[f"t{t:02d}"],
                            ).astype(float)
                            for t in times
                        ]
                    )
                else:
                    values = scores[model]
                    data = np.stack(
                        [
                            (values if values.ndim == 1 else values[t])[positions[test]]
                            for t in times
                        ]
                    )
                matrices.append(data)
            matrix = np.stack(matrices)
            boot = paired_target_bootstrap(
                labels[test], matrix, target[test], repeats, seed + assay_idx
            )
            observed = [
                np.mean([roc_auc_score(labels[test], s) for s in m]) for m in matrix
            ]
            for j, model in enumerate(MODELS):
                low, high = np.quantile(boot[:, j], [0.025, 0.975])
                dl, dh = np.quantile(
                    boot[:, j] - boot[:, MODELS.index("Random")], [0.025, 0.975]
                )
                ci_rows.append(
                    {
                        "dataset": assay,
                        "model": model,
                        "current_test_AUROC": observed[j],
                        "AUROC_bootstrap_low": low,
                        "AUROC_bootstrap_high": high,
                        "paired_delta_vs_random_low": dl,
                        "paired_delta_vs_random_high": dh,
                        "n_valid_bootstrap": len(boot),
                    }
                )
    calibration_table = pd.DataFrame(rows)
    calibration_table.to_csv(
        out / "calibration_score_diagnostics.tsv", sep="\t", index=False
    )
    calibration_summary = (
        calibration_table.groupby(["dataset", "model", "variant"], sort=False)[
            [
                "calibration_AUROC",
                "within_target_macro_AUROC",
                "fraction_zero",
                "fraction_negative",
            ]
        ]
        .mean()
        .reset_index()
    )
    calibration_summary.to_csv(out / "calibration_summary.tsv", sep="\t", index=False)
    ci_table = pd.DataFrame(ci_rows)
    ci_table.to_csv(out / "current_test_uncertainty.tsv", sep="\t", index=False)
    (out / "diagnostic_report.json").write_text(
        json.dumps(
            {
                "validation_only": report["validation_only"],
                "evaluation": str(evaluation),
                "evaluation_report_sha256": digest(
                    evaluation / "evaluation_report.json"
                ),
                "code_sha256": digest(Path(__file__)),
                "cache_report_sha256": report["cache_report_sha256"],
                "random_selection_sha256": digest(
                    evaluation / "random_edges_by_stage.npz"
                ),
                "bootstrap_repeats": repeats,
                "source_audit": source_audit,
                "alternatives_policy": "all alternatives reported on calibration fold only; no winner chosen, no test-based score flipping, seed search, filtering, retraining or overwrite of original metrics/figures",
                "uncertainty": "95% percentile intervals from paired target-gene cluster bootstrap of the CURRENT raw model and actual Random scores; all stage scores share each draw's weights; descriptive uncertainty conditional on fixed models/pairs, not biological replicate inference",
                "within_target": "diagnostic calibration-only macro AUROC for TG groups containing both classes; different from pooled AUROC and not a replacement primary metric",
                "scenic_source": "official enhancer_to_gene.calculate_regions_to_genes_relationships also outputs importance_x_abs_rho",
                "scdori_variant": "saved native global_gene_peak coefficient without stage accessibility multiplication; an explicitly distinct static readout, not an optimized model",
            },
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    print(calibration_summary.to_string(index=False), flush=True)
    print(ci_table.to_string(index=False), flush=True)
    print(f"Read-only score diagnostics: {out}", flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, default=Path("latest"))
    parser.add_argument("--scenic-r2g-run", type=Path, default=Path("latest"))
    parser.add_argument("--bootstrap-repeats", type=int, default=100)
    args = parser.parse_args()
    audit_evaluation(
        resolve_evaluation(args.evaluation, args.scenic_r2g_run), args.bootstrap_repeats
    )


if __name__ == "__main__":
    main()
