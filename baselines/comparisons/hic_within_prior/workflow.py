"""Within-prior model evaluation and the fixed-edge-count random control."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scdori_mondo"))

from compare_stage_hic import calibration_threshold, digest, grouped_calibration_split
from hic_common import metrics
from random_control import draw_stages, load_context, prepare_context, set_metrics

MODELS = ["DRAFT", "SCENIC+ R2G", "scDoRI", "Random"]
COLORS = {
    "DRAFT": "red",
    "Dictys": "tab:blue",
    "scDORI": "tab:orange",
    "scMNTI": "tab:green",
    "SCENIC+": "tab:red",
    "random_control": "tab:purple",
}
MODEL_COLORS = {
    "DRAFT": COLORS["DRAFT"],
    "SCENIC+ R2G": COLORS["SCENIC+"],
    "scDoRI": COLORS["scDORI"],
    "Random": COLORS["random_control"],
}
FONT_STYLE = {
    "font.size": 18,
    "axes.titlesize": 24,
    "axes.labelsize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 16,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
}
CONFIG = yaml.safe_load(
    (HERE.parents[1] / "configs/germ_reproduce.yaml").read_text(encoding="utf-8")
)
ASSAYS = {
    assay: tuple(CONFIG["reproduce"]["hic"][f"{assay}_timepoints"])
    for assay in ["spg", "spc"]
}
ASSAY_LABELS = {"spg": "SpG", "spc": "SpC"}
STAGE_LABELS = json.loads(
    (HERE.parents[1] / "configs/germline_stage_labels.json").read_text(encoding="utf-8")
)["labels"]


def seed_sample(codes, positives, n_genes, seed, assay_idx):
    """Uniform 1:1 prior negatives and physical-pair-disjoint grouped folds."""
    positives = np.asarray(positives, dtype=np.int64)
    eligible = np.setdiff1d(np.arange(len(codes)), positives)
    if len(positives) < 4 or len(eligible) < 4:
        raise ValueError("Need at least four unique candidate positives/negatives.")
    sampling_rng = np.random.default_rng(np.random.SeedSequence([seed, assay_idx, 1]))
    negative = sampling_rng.choice(eligible, size=len(positives), replace=True)
    positions = np.r_[positives, negative]
    labels = np.r_[
        np.ones(len(positives), dtype=int), np.zeros(len(negative), dtype=int)
    ]
    pairs = np.column_stack([codes[positions] // n_genes, codes[positions] % n_genes])
    split_seed = int(np.random.SeedSequence([seed, assay_idx, 2]).generate_state(1)[0])
    calibration = grouped_calibration_split(pairs, labels, n_genes, split_seed)
    return positions, labels, calibration


def evaluate(
    cache,
    seeds=(2025,),
    output=None,
    random_seed=12345,
    random_context=None,
    data_root=None,
):
    cache = Path(cache).resolve()
    report = json.loads((cache / "cache_report.json").read_text())
    for name, sha in report["files_sha256"].items():
        if digest(cache / name) != sha:
            raise ValueError(f"Frozen score cache changed: {name}")
    seeds = [int(s) for s in seeds]
    if len(seeds) != 1 or seeds[0] < 0:
        raise ValueError(
            "Use one fixed nonnegative seed; plotting observations are stages."
        )
    if report["stage_labels"] != STAGE_LABELS:
        raise ValueError(
            "Frozen network stage labels differ from the formal stage order."
        )
    if random_seed < 0:
        raise ValueError("Random selection seed must be nonnegative.")
    context_path = (
        Path(random_context) if random_context else prepare_context(cache, data_root)
    )
    context = load_context(context_path, cache)
    selected = draw_stages(context, random_seed)
    out = (
        Path(output)
        if output
        else cache.parent / f"within_prior_stage_evaluation_{uuid.uuid4().hex[:10]}"
    )
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        out / "random_edges_by_stage.npz",
        n_re=context["n_re"],
        n_tg=context["n_tg"],
        **{f"t{t:02d}": edges for t, edges in enumerate(selected)},
    )
    with np.load(cache / "scores.npz", allow_pickle=False) as arrays:
        codes, n_genes = arrays["pair_codes"], int(arrays["n_genes"])
        values = {
            "DRAFT": arrays["draft"],
            "SCENIC+ R2G": arrays["scenicplus"],
            "scDoRI": arrays["scdori"],
        }
        for model, data in values.items():
            expected = (len(codes),) if model == "SCENIC+ R2G" else (15, len(codes))
            if data.shape != expected or not np.isfinite(data).all():
                raise ValueError(f"Invalid frozen score shape/values: {model}")
        if len(context["candidate_codes"]) != len(codes):
            raise ValueError("Random-control candidate mapping length differs.")
        rows, thresholds, curve_rows, saved_curves, random_audit = [], [], [], [], []
        grid = np.linspace(0, 1, 1001)
        for assay_idx, (assay, times) in enumerate(ASSAYS.items()):
            for seed in seeds:
                positions, labels, calibration = seed_sample(
                    codes, arrays[f"positives_{assay}"], n_genes, seed, assay_idx
                )
                sampled = pd.DataFrame(
                    {
                        "pair_idx": np.arange(len(positions)),
                        "candidate_idx": positions,
                        "pair_code": codes[positions],
                        "label": labels,
                        "fold": np.where(calibration, "calibration", "test"),
                    }
                )
                sampled.to_csv(
                    out / f"pairs_{assay}_s{seed}.tsv.gz", sep="\t", index=False
                )
                test = ~calibration
                truth = labels[test]
                stage_random = {}
                for stage in times:
                    stage_random[stage] = set_metrics(
                        selected[stage], context[f"gold_{assay}"]
                    )
                    random_audit.append(
                        {
                            "dataset": assay,
                            "time_idx": stage,
                            "stage_label": STAGE_LABELS[stage],
                            "random_seed": random_seed,
                            "n_full_pool_re": int(context["n_re"]),
                            "n_full_pool_tg": int(context["n_tg"]),
                            **stage_random[stage],
                        }
                    )
                for model in MODELS:
                    for stage in times:
                        if model == "Random":
                            score = np.isin(
                                context["candidate_codes"][positions], selected[stage]
                            ).astype(float)
                            f1 = stage_random[stage]["F1"]
                        elif model == "SCENIC+ R2G":
                            score = values[model][positions]
                        else:
                            score = values[model][stage, positions]
                        if model != "Random":
                            cutoff = calibration_threshold(
                                labels[calibration], score[calibration]
                            )
                            f1 = metrics(truth, score[test], cutoff, inclusive=True)[
                                "f1"
                            ]
                            thresholds.append(
                                {
                                    "dataset": assay,
                                    "evaluation_seed": seed,
                                    "time_idx": stage,
                                    "model": model,
                                    "calibration_cutoff": cutoff,
                                }
                            )
                        test_score = score[test]
                        stat = {
                            "F1": f1,
                            "AUROC": float(roc_auc_score(truth, test_score)),
                            "AP": float(average_precision_score(truth, test_score)),
                        }
                        rows.append(
                            {
                                "dataset": assay,
                                "evaluation_seed": seed,
                                "time_idx": stage,
                                "stage_label": STAGE_LABELS[stage],
                                "model": model,
                                "F1_protocol": "full_pool_set"
                                if model == "Random"
                                else "calibrated_within_prior_test",
                                **stat,
                            }
                        )
                        fpr, tpr, _ = roc_curve(
                            truth, test_score, drop_intermediate=False
                        )
                        saved_curves.append(
                            (assay, stage, model, np.interp(grid, fpr, tpr))
                        )
        for assay in ASSAYS:
            for model in MODELS:
                curves = np.stack(
                    [c for a, _t, m, c in saved_curves if a == assay and m == model]
                )
                mean, sd = curves.mean(axis=0), curves.std(axis=0, ddof=1)
                curve_rows.extend(
                    {
                        "dataset": assay,
                        "model": model,
                        "fpr": float(x),
                        "tpr_mean": float(y),
                        "tpr_sd": float(z),
                    }
                    for x, y, z in zip(grid, mean, sd)
                )
    stage_table, curves = pd.DataFrame(rows), pd.DataFrame(curve_rows)
    stage_table.to_csv(out / "metrics_by_stage.tsv", sep="\t", index=False)
    pd.DataFrame(
        [
            {"dataset": assay, "time_idx": t, "stage_label": STAGE_LABELS[t]}
            for assay, times in ASSAYS.items()
            for t in times
        ]
    ).to_csv(out / "stage_alignment.tsv", sep="\t", index=False)
    summary = stage_table.groupby(["dataset", "model"], sort=False)[
        ["F1", "AUROC", "AP"]
    ].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(out / "metrics_summary.tsv", sep="\t", index=False)
    pd.DataFrame(thresholds).to_csv(
        out / "calibration_cutoffs.tsv", sep="\t", index=False
    )
    pd.DataFrame(random_audit).to_csv(
        out / "random_set_audit.tsv", sep="\t", index=False
    )
    curves.to_csv(out / "mean_roc_curves.tsv", sep="\t", index=False)
    np.savez_compressed(
        out / "roc_by_stage.npz",
        fpr=grid,
        **{
            f"{assay}_t{stage:02d}_m{MODELS.index(model)}": curve
            for assay, stage, model, curve in saved_curves
        },
    )
    metadata = {
        "scope": "within_prior",
        "metrics": ["F1", "AUROC", "AP"],
        "primary_metric": "AUROC",
        "evaluation_seed": seeds[0],
        "random_seed": random_seed,
        "random_context": str(context_path.resolve()),
        "random_context_report_sha256": digest(context_path / "context_report.json"),
        "random_f1_scope": "original full DRAFT Cartesian-product edge-set F1; differs from the three models' calibrated within-prior held-out F1",
        "random_rank_scope": "binary membership in the actual selected network on the SAME within-prior held-out pairs as the three models; no continuous random ranking or F1 calibration",
        "observation_unit": "developmental stage",
        "stage_indices": {assay: list(times) for assay, times in ASSAYS.items()},
        "stage_labels": STAGE_LABELS,
        "stage_config_sha256": digest(HERE.parents[1] / "configs/germ_reproduce.yaml"),
        "stage_labels_sha256": digest(
            HERE.parents[1] / "configs/germline_stage_labels.json"
        ),
        "plot_colors": MODEL_COLORS,
        "fixed_model_seed": report["model_seed"],
        "models": MODELS,
        "validation_only": report["validation_only"],
        "cache": str(cache),
        "cache_report_sha256": digest(cache / "cache_report.json"),
        "code_sha256": digest(Path(__file__)),
        "protocol": "fixed trained models and one evaluation seed; uniform 1:1 negative sampling with replacement within complete shared prior and physical-pair-disjoint 50/50 calibration/test splits; three models use calibrated held-out F1 and raw-score AUROC/AP; Random uses original full-pool edge-set F1, explicitly labeled as a different denominator; AUROC/AP of all four methods share held-out assay pairs, with Random scored by actual selected-edge 0/1 membership; 4 SpG/9 SpC stage points are descriptive",
        "random": "original notebook unique Cartesian-product selection; per-stage budget equals original DRAFT mapped prior edge count; full original RE/TG node pool and all mapped assay positives, including outside-prior positives; RNG seed 12345 by default and all 15 stage draws in original order; no Hi-C-informed selection, F1 threshold tuning, learned-score shuffle or imposed low values",
        "roc": "four mean ROC curves per assay, equally averaged across only its corresponding stages; bands show one SD of TPR across stages; legend AUROC is mean +/- stage SD from raw stage scores, not integration of the display grid; stage scores are descriptive, not independent biological replicates",
        "scenic_scope": "new native shared-prior GLOBAL R2G scores, not published stage-filtered eRegulons",
    }
    (out / "evaluation_report.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8"
    )
    plot_results(stage_table, curves, out, metadata)
    print(summary.to_string(), flush=True)
    print(f"Within-prior figures and metrics: {out}", flush=True)
    return out


def metric_limits(values, metric):
    """Keep all stage points visible, and include the AUROC chance reference."""
    values = np.asarray(values, dtype=float)
    if metric == "AUROC":
        values = np.r_[values, 0.5]
    lower, upper = float(values.min()), float(values.max())
    padding = max(0.008, (upper - lower) * 0.15)
    return max(-0.01, lower - padding), min(1.01, upper + padding)


def cache_from_evaluation(value, scenic_run):
    """Read an existing evaluation's verified cache without loading model inputs."""
    if Path(value) == Path("latest"):
        from rescore_hic import resolve_native_r2g

        run = resolve_native_r2g(scenic_run)
        reports = list(run.glob("within_prior_*/evaluation_report.json"))
        if not reports:
            raise FileNotFoundError("No completed within-prior evaluation in this run.")
        value = max(reports, key=lambda p: p.stat().st_mtime_ns).parent
    previous = Path(value).expanduser().resolve()
    report = json.loads(
        (previous / "evaluation_report.json").read_text(encoding="utf-8")
    )
    cache = Path(report["cache"]).resolve()
    if digest(cache / "cache_report.json") != report["cache_report_sha256"]:
        raise ValueError("Previous evaluation's frozen score cache report changed.")
    return cache


def plot_results(points, curves, out, metadata):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    figure_width = metadata.get("figure_width", 9.5)

    def save(fig, name):
        for suffix in ["png", "pdf", "svg"]:
            fig.savefig(out / f"{name}.{suffix}", dpi=300)
        plt.close(fig)

    def boxes(ax, assay, metric):
        data = [
            points.loc[
                points.dataset.eq(assay) & points.model.eq(model), metric
            ].to_numpy()
            for model in MODELS
        ]
        result = ax.boxplot(
            data,
            positions=np.arange(1, 5),
            widths=0.5,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.5},
        )
        for box, model in zip(result["boxes"], MODELS):
            box.set_facecolor(MODEL_COLORS[model])
        for j, samples in enumerate(data, 1):
            spread = min(0.24, 0.035 * (len(samples) - 1))
            ax.scatter(
                np.full(len(samples), j) + np.linspace(-spread, spread, len(samples)),
                samples,
                s=70,
                color=MODEL_COLORS[MODELS[j - 1]],
                edgecolor="black",
                linewidth=1.0,
                zorder=3,
            )
        random_label = metadata.get(
            "random_label",
            "Random\nset F1" if metric == "F1" else "Random\nselected edges",
        )
        ax.set_xticks(
            range(1, 5),
            [
                "DRAFT",
                metadata.get("scenic_label", "SCENIC+\nR2G"),
                "scDORI",
                random_label,
            ],
        )
        ax.set(
            xlim=(0.4, 4.6),
            ylim=metric_limits(np.concatenate(data), metric),
            xlabel="Method",
            ylabel=metric,
            title=f"RE→TG ({ASSAY_LABELS[assay]} Hi-C): {metric}",
        )
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
        ax.grid(axis="y", color="#EEEEEE", linewidth=0.8)
        ax.set_axisbelow(True)
        if metric == "AUROC":
            ax.axhline(0.5, color="#777777", linestyle=":", linewidth=1)
            if metadata.get("annotate_chance", False):
                ax.annotate(
                    "Chance AUROC = 0.5",
                    xy=(0.02, 0.5),
                    xycoords=ax.get_yaxis_transform(),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="left",
                    va="bottom",
                    fontsize=10,
                    color="#555555",
                )
        ax.text(
            0.5,
            -0.28,
            (
                "Model F1: held-out prior pairs · Random: full-pool set F1\n"
                if metric == "F1"
                else "Within shared prior · "
            )
            + f"{len(ASSAYS[assay])} developmental stages · evaluation seed {metadata['evaluation_seed']}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=11,
        )

    def roc_panel(ax, assay):
        for model in MODELS:
            data = curves.loc[curves.dataset.eq(assay) & curves.model.eq(model)]
            auc = points.loc[points.dataset.eq(assay) & points.model.eq(model), "AUROC"]
            name = (
                "scDORI"
                if model == "scDoRI"
                else ("Random selection" if model == "Random" else model)
            )
            if model == "SCENIC+ R2G":
                name = metadata.get("scenic_label", model).replace("\n", " ")
            elif model == "Random":
                name = metadata.get("random_label", name).replace("\n", " ")
            ax.plot(
                np.r_[0, data.fpr],
                np.r_[0, data.tpr_mean],
                color=MODEL_COLORS[model],
                linewidth=2,
                linestyle="--" if model == "Random" else "-",
                label=f"{name}: {auc.mean():.3f} ± {auc.std(ddof=1):.3f}",
            )
            ax.fill_between(
                data.fpr,
                np.maximum(0, data.tpr_mean - data.tpr_sd),
                np.minimum(1, data.tpr_mean + data.tpr_sd),
                color=MODEL_COLORS[model],
                alpha=0.10,
                linewidth=0,
            )
        ax.plot([0, 1], [0, 1], color="#DDDDDD", linestyle=":", zorder=0)
        ax.set(
            xlim=(0, 1),
            ylim=(0, 1.01),
            xlabel="False positive rate",
            ylabel="True positive rate",
            title=f"RE→TG ({ASSAY_LABELS[assay]} Hi-C): ROC",
        )
        ax.legend(
            loc="lower right", facecolor="white", edgecolor="none", framealpha=0.95
        )
        ax.text(
            0.5,
            -0.24,
            "Within shared prior · mean ROC ± 1 SD across corresponding stages",
            transform=ax.transAxes,
            ha="center",
            fontsize=11,
        )

    with plt.rc_context(FONT_STYLE):
        for assay in ASSAYS:
            for metric in ["AUROC", "F1", "AP"]:
                fig, ax = plt.subplots(
                    figsize=(figure_width, 5.5), constrained_layout=True
                )
                boxes(ax, assay, metric)
                if metadata["validation_only"]:
                    fig.suptitle("SYNTHETIC SOFTWARE TEST", fontsize=11)
                save(fig, f"{assay}_{metric.lower()}_boxplot")
            fig, ax = plt.subplots(figsize=(figure_width, 5.5), constrained_layout=True)
            roc_panel(ax, assay)
            if metadata["validation_only"]:
                fig.suptitle("SYNTHETIC SOFTWARE TEST", fontsize=11)
            save(fig, f"{assay}_roc_curves")
        for metric in ["AUROC", "F1", "AP"]:
            fig, axes = plt.subplots(
                2, 1, figsize=(figure_width, 10.8), constrained_layout=True
            )
            for ax, assay in zip(axes, ASSAYS):
                boxes(ax, assay, metric)
            if metadata["validation_only"]:
                fig.suptitle("SYNTHETIC SOFTWARE TEST", fontsize=11)
            save(fig, f"within_prior_{metric.lower()}_boxplots")
        fig, axes = plt.subplots(
            2, 1, figsize=(figure_width, 10.8), constrained_layout=True
        )
        for ax, assay in zip(axes, ASSAYS):
            roc_panel(ax, assay)
        if metadata["validation_only"]:
            fig.suptitle("SYNTHETIC SOFTWARE TEST", fontsize=11)
        save(fig, "within_prior_roc_curves")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenic-r2g-run", type=Path, default=Path("latest"))
    parser.add_argument(
        "--from-evaluation",
        type=Path,
        help="Existing evaluation folder or latest: reuse its frozen scores without inference.",
    )
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--draft-checkpoint", type=Path, default=Path("auto"))
    parser.add_argument(
        "--evaluation-seed",
        "--evaluation-seeds",
        type=int,
        nargs="+",
        default=[2025],
        dest="evaluation_seeds",
        help="One fixed evaluation seed; each plotted observation is a stage.",
    )
    parser.add_argument("--cpus", type=int, default=8)
    parser.add_argument(
        "--random-seed",
        type=int,
        default=12345,
        help="Original fixed-edge-count selection seed; default matches the notebook.",
    )
    parser.add_argument("--device", choices=["cpu", "auto"], default="cpu")
    args = parser.parse_args()
    if args.cpus < 1:
        parser.error("CPU count must be positive.")
    if len(args.evaluation_seeds) != 1 or args.evaluation_seeds[0] < 0:
        parser.error(
            "Provide one evaluation seed; observations are developmental stages."
        )
    if args.random_seed < 0:
        parser.error("Random selection seed must be nonnegative.")
    if args.from_evaluation is not None:
        cache = cache_from_evaluation(args.from_evaluation, args.scenic_r2g_run)
        evaluate(
            cache,
            args.evaluation_seeds,
            random_seed=args.random_seed,
            data_root=args.data_root,
        )
        return
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch

    torch.set_num_threads(args.cpus)
    from score_cache import build_cache

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    cache = build_cache(args.scenic_r2g_run, args.data_root, args.draft_checkpoint)
    evaluate(
        cache,
        args.evaluation_seeds,
        random_seed=args.random_seed,
        data_root=args.data_root,
    )


if __name__ == "__main__":
    main()
