"""Mondo network profiles of DRAFT and its ablations, and Mondo RNA expression (Supp. Fig. 10)."""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator, make_interp_spline

from ..utils import load_yaml_config, resolve_config_path
from .continuous_activity import gaussian_smooth_1d
from .germline_stages import germline_stage_labels, set_germline_stage_axis
from .scenic_reference import load_scenic_profile

MODEL_STYLES = {
    "gru": ("DRAFT (GAT + GRU)", "#E5A000"),
    "no_gru": ("DRAFT w/o GRU", "#4C78A8"),
    "separate_gat": ("separate GAT", "#4C78A8"),
    "linear_gru": ("Linear + GRU (GAT ablation)", "#009E73"),
}
NETWORK_FIELDS = (
    "active_thr",
    "min_score_keep",
    "top_k_global",
    "n_interp",
    "sigma_time",
    "rescue_gamma",
    "rescue_m_min",
    "rescue_m_max",
    "rescue_eps",
)


def _read_table(path: Path, columns: list[str]) -> pd.DataFrame:
    table = pd.read_csv(path, sep="\t")
    if table.empty or not set(columns).issubset(table.columns):
        raise ValueError(f"Missing plot rows or columns {columns}: {path}")
    if not np.isfinite(table[columns].to_numpy(dtype=float)).all():
        raise ValueError(f"Non-finite plot values: {path}")
    if table.time.duplicated().any() or not table.time.is_monotonic_increasing:
        raise ValueError(f"Plot times must be unique and increasing: {path}")
    if (table[columns].to_numpy(dtype=float) < 0).any():
        raise ValueError(f"Plot values must be nonnegative: {path}")
    return table


def _load_run(directory: Path, variant: str, seed: int, gene: str) -> dict[str, Any]:
    reference = directory / "reference_manifest.json"
    report_path = (
        reference if reference.is_file() else directory / "analysis_report.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected_encoder = "linear" if variant == "linear_gru" else "gat"
    if (
        report["seed"] != seed
        or report["variant"] != variant
        or bool(report["use_gru"]) != (variant not in {"no_gru", "separate_gat"})
        or report.get("spatial_encoder", "gat") != expected_encoder
    ):
        raise ValueError(
            f"Saved seed/model does not match {variant} seed {seed}: {directory}"
        )
    if variant == "separate_gat" and (
        not report.get("temporal_loss_disabled", False)
        or float(report.get("lambda_smooth_used", -1)) != 0
        or any(p != -1 for p in report.get("training_parent", [0]))
    ):
        raise ValueError(
            f"separate GAT has not disabled temporal training: {directory}"
        )
    if "gene" in report and report["gene"].casefold() != gene.casefold():
        raise ValueError(f"Reference gene does not match {gene}: {directory}")
    if reference.is_file():
        network = report["network_settings"]
    else:
        network_path = directory / "network_dense" / "dense_network_report.json"
        network = json.loads(network_path.read_text(encoding="utf-8"))["settings"]
    settings = {key: network[key] for key in NETWORK_FIELDS}
    centrality_path = directory / "mondo_dynamics_centrality.tsv"
    expression_path = directory / "mondo_dynamics_expression.tsv"
    centrality = _read_table(centrality_path, ["time", "centrality_raw"])
    expression = _read_table(expression_path, ["time", "mean_expr", "n_cells"])
    if not np.array_equal(expression.time, expression.time.astype(int)):
        raise ValueError(f"Expression times must be integer stages: {directory}")
    if (expression.n_cells <= 0).any():
        raise ValueError(f"Expression stages must contain cells: {directory}")
    if not np.allclose(centrality.time.iloc[[0, -1]], expression.time.iloc[[0, -1]]):
        raise ValueError(f"Centrality and expression ranges differ: {directory}")
    if not expression.time.isin(centrality.time).all():
        raise ValueError(
            f"Dense centrality is missing integer-stage values: {directory}"
        )
    sources = {}
    source_paths = [centrality_path, expression_path, report_path]
    if not reference.is_file():
        source_paths.append(network_path)
    for path in source_paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if (
            reference.is_file()
            and path.name in report.get("table_sha256", {})
            and digest != report["table_sha256"][path.name]
        ):
            raise ValueError(f"Manuscript reference table checksum differs: {path}")
        sources[path.name] = {"path": str(path), "sha256": digest}
    return {
        "variant": variant,
        "seed": seed,
        "directory": str(directory),
        "centrality": centrality,
        "expression": expression,
        "network_settings": settings,
        "sources": sources,
    }


def _interpolate(times: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    # The same nonnegative cubic display as the manuscript's plot_smooth_curve.
    values = np.asarray(values, dtype=float)
    result = (
        make_interp_spline(times, values, k=3)(grid)
        if len(times) >= 4
        else np.interp(grid, times, values)
    )
    return np.maximum(result, 0)


def plot_scenic_comparison(
    display: pd.DataFrame,
    dense: pd.DataFrame,
    expression: pd.DataFrame,
    scenic: pd.DataFrame,
    scenic_metadata: dict[str, Any],
    settings: dict[str, Any],
    directory: Path,
) -> tuple[list[str], dict[str, Any]]:
    """Compare temporal shapes with explicitly independent maximum normalization."""
    seed, gene = int(settings["seed"]), str(settings["gene"])
    prefix = f"mondo_seed{seed}_scenic_comparison"
    curves = display.copy(deep=True)
    styles = {k: v for k, v in MODEL_STYLES.items() if k in display}
    curves["scenic_target_count"] = PchipInterpolator(
        scenic.time.to_numpy(dtype=float), scenic.target_count.to_numpy(dtype=float)
    )(display.time)
    maxima = {}
    for column in [*styles, "mean_expr", "scenic_target_count"]:
        if column in styles:
            anchors = dense.loc[dense.variant.eq(column), "centrality_smooth"]
        elif column == "mean_expr":
            anchors = expression.mean_expr
        else:
            anchors = scenic.target_count
        maximum = max(float(curves[column].max()), float(anchors.max()))
        maxima[column] = maximum
        curves[column] = curves[column] / maximum if maximum > 0 else 0.0
    curves.to_csv(directory / f"{prefix}_curves.tsv", sep="\t", index=False)
    scenic.to_csv(directory / f"{prefix}_raw_counts.tsv", sep="\t", index=False)
    figures = []
    purple, red = "#795099", "#D55E00"
    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 14,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    ):
        fig, ax = plt.subplots(figsize=tuple(settings.get("figsize", [11, 7])))
        for variant, (label, color) in styles.items():
            ax.plot(curves.time, curves[variant], color=color, linewidth=3, label=label)
        ax.plot(
            curves.time,
            curves.mean_expr,
            color=purple,
            linewidth=2.5,
            linestyle="--",
            label=f"Mean {gene.upper()} expression",
        )
        ax.plot(
            curves.time,
            curves.scenic_target_count,
            color=red,
            linewidth=2.8,
            linestyle="-.",
            label="SCENIC+ stage-filtered targets",
        )
        if maxima["scenic_target_count"] > 0:
            ax.scatter(
                scenic.time,
                scenic.target_count / maxima["scenic_target_count"],
                color=red,
                s=30,
                zorder=3,
            )
        if maxima["mean_expr"] > 0:
            ax.scatter(
                expression.time,
                expression.mean_expr / maxima["mean_expr"],
                color=purple,
                s=20,
                alpha=0.35,
            )
        set_germline_stage_axis(ax)
        ax.set_xlim(
            float(expression.time.min()) - 0.4, float(expression.time.max()) + 0.4
        )
        ax.set_ylim(-0.035, 1.08)
        ax.set_xlabel("Germline trajectory", fontsize=17, labelpad=10)
        ax.set_ylabel("Relative profile (each maximum = 1)", fontsize=17, labelpad=12)
        ax.spines[["top", "right"]].set_visible(False)
        fig.legend(
            ax.lines,
            [line.get_label() for line in ax.lines],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.93),
            ncol=2,
            frameon=False,
            fontsize=12,
        )
        fig.suptitle(
            settings.get(
                "scenic_comparison_title",
                f"{gene.upper()} temporal profiles (DRAFT models: seed {seed})",
            ),
            y=0.985,
            fontsize=18,
        )
        fig.text(
            0.5,
            0.025,
            f"SCENIC+: unique targets; DAR={scenic_metadata['subset_dar']}, DEG={scenic_metadata['subset_deg']}",
            ha="center",
            fontsize=11,
            color=red,
        )
        fig.subplots_adjust(left=0.10, right=0.97, bottom=0.24, top=0.73)
        try:
            for suffix in ["png", "pdf", "svg"]:
                path = directory / f"{prefix}.{suffix}"
                fig.savefig(path, dpi=int(settings.get("dpi", 600)), facecolor="white")
                figures.append(str(path))
        finally:
            plt.close(fig)
    caption = (
        f"**Supplementary Fig. Sx | Relative temporal profiles of {gene.upper()}.** "
        f"DRAFT and its two ablations use seed {seed}; their profiles are Gaussian-smoothed "
        "selected-network centrality followed by cubic interpolation. RNA is the shared "
        "stage-wise mean log-normalized expression. SCENIC+ is the number of distinct target "
        f"genes returned by the [dAWA eRegulon Table]({scenic_metadata['source_url']}) with "
        f"DAR={scenic_metadata['subset_dar']} and DEG={scenic_metadata['subset_deg']}; multiple "
        "regions linked to the same target are counted once. SCENIC+ stage counts are shown "
        "as dots, connected by shape-preserving cubic interpolation without Gaussian smoothing. "
        "Each displayed profile is divided by its own maximum (including its anchor values), "
        "so this panel compares timing and shape, not absolute strength or shared activity units. "
        "The SCENIC+ readout is a stage-filtered target count, not RSS or AUCell activity. "
        "No between-seed uncertainty is shown. Original-scale DRAFT centrality and RNA remain "
        "available in the companion dual-axis panel.\n"
    )
    if "separate_gat" in styles:
        caption += (
            "The separate GAT retains the shared DRAFT graph encoder but has no GRU "
            "and no adjacent-stage training regularization. Export interpolation and "
            "display smoothing match the reference; they are not recurrent learning. "
            "The companion stage-anchor panel shows unsmoothed original-stage values.\n"
        )
    (directory / "scenic_comparison_caption.md").write_text(caption, encoding="utf-8")
    return figures, {
        **scenic_metadata,
        "normalization_maxima": maxima,
        "normalization": "independent division by each display/anchor maximum; zero profiles remain zero",
        "interpolation": "PCHIP for SCENIC+ counts; no Gaussian pass; cubic for DRAFT and RNA as in the original panel",
        "interpretation": "timing/shape only; target counts and weighted centrality have different units",
    }


def plot_stage_anchors(runs, expression, scenic, settings, directory):
    """Show original-stage readouts only, without temporal smoothing or splines."""
    seed = int(settings["seed"])
    table = pd.DataFrame({"time": expression.time.to_numpy()})
    styles = {run["variant"]: MODEL_STYLES[run["variant"]] for run in runs}
    for run in runs:
        anchors = run["centrality"].set_index("time").loc[table.time, "centrality_raw"]
        table[run["variant"]] = anchors.to_numpy(dtype=float)
    table["mean_expr"] = expression.mean_expr.to_numpy()
    if scenic is not None:
        table["scenic_target_count"] = (
            scenic.set_index("time")
            .loc[table.time, "target_count"]
            .to_numpy(dtype=float)
        )
    raw = table.copy(deep=True)
    for column in table.columns.drop("time"):
        maximum = float(table[column].max())
        table[column] = table[column] / maximum if maximum > 0 else 0.0
    prefix = f"mondo_seed{seed}_stage_anchors"
    raw.to_csv(directory / f"{prefix}_raw.tsv", sep="\t", index=False)
    table.to_csv(directory / f"{prefix}_normalized.tsv", sep="\t", index=False)
    figures = []
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 13,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    ):
        fig, ax = plt.subplots(figsize=tuple(settings.get("figsize", [11, 7])))
        for variant, (label, color) in styles.items():
            ax.plot(
                table.time,
                table[variant],
                color=color,
                lw=2.5,
                marker="o",
                ms=4,
                label=label,
            )
        ax.plot(
            table.time,
            table.mean_expr,
            color="#795099",
            ls="--",
            lw=2.3,
            marker="o",
            ms=4,
            label="Mean MONDO expression",
        )
        if scenic is not None:
            ax.plot(
                table.time,
                table.scenic_target_count,
                color="#D55E00",
                ls="-.",
                lw=2.3,
                marker="o",
                ms=4,
                label="SCENIC+ stage-filtered targets",
            )
        set_germline_stage_axis(ax)
        ax.set(
            xlabel="Germline trajectory",
            ylabel="Relative profile (each maximum = 1)",
            ylim=(-0.035, 1.08),
        )
        ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(
            settings.get(
                "stage_anchor_title",
                f"MONDO temporal profiles (original stage anchors; seed {seed})",
            ),
            y=0.985,
            fontsize=17,
        )
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.94),
            ncol=2,
            frameon=False,
            fontsize=12,
        )
        fig.subplots_adjust(left=0.10, right=0.97, bottom=0.24, top=0.73)
        for suffix in ["png", "pdf", "svg"]:
            path = directory / f"{prefix}.{suffix}"
            fig.savefig(path, dpi=int(settings.get("dpi", 600)), facecolor="white")
            figures.append(str(path))
        plt.close(fig)
    (directory / "stage_anchor_caption.md").write_text(
        "MONDO temporal profiles at the original stage anchors. The separate GAT uses "
        "the shared DRAFT spatial encoder with GRU and adjacent-stage smoothness loss "
        "disabled. All curves use original-stage selected-network centrality, mean "
        "log-normalized RNA including zeros, or published SCENIC+ unique-target counts. "
        "Each profile is divided by its own maximum. Dots mark stage-level values; "
        "straight connecting lines are display guides. No Gaussian smoothing, dense "
        "intermediate samples or cubic interpolation enters this panel. Network scores "
        "are not measured TF activity.\n",
        encoding="utf-8",
    )
    return figures


def run_mondo_supplement(
    config_path: str | Path, overrides: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Read saved numeric results only; create a fresh publication figure directory."""
    settings = load_yaml_config(config_path)
    settings.update({k: v for k, v in (overrides or {}).items() if v is not None})
    settings["base_dir"] = str(
        Path(
            settings.get("base_dir", Path(config_path).resolve().parent.parent)
        ).resolve()
    )
    seed, gene = int(settings["seed"]), str(settings["gene"])
    sigma = float(settings.get("sigma_time", 0.5))
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma_time must be positive and finite.")
    ablation_root = resolve_config_path(settings, settings["ablation_root"])
    gat_variant = settings.get("gat_variant", "no_gru")
    if gat_variant not in {"no_gru", "separate_gat"}:
        raise ValueError("gat_variant must be no_gru or separate_gat.")
    directories = {
        "gru": resolve_config_path(settings, settings["draft_tables"]),
        gat_variant: ablation_root / f"seed_{seed}" / gat_variant,
        "linear_gru": ablation_root / f"seed_{seed}" / "linear_gru",
    }
    runs = [
        _load_run(directory, variant, seed, gene)
        for variant, directory in directories.items()
    ]
    styles = {k: MODEL_STYLES[k] for k in directories}
    baseline = runs[0]
    expression = baseline["expression"]
    for run in runs[1:]:
        if run["network_settings"] != baseline["network_settings"]:
            raise ValueError(
                f"Network export/selection settings differ for {run['variant']}."
            )
        if not np.array_equal(run["centrality"].time, baseline["centrality"].time):
            raise ValueError(f"Dense time grids differ for {run['variant']}.")
        other = run["expression"]
        if (
            not np.array_equal(other.time, expression.time)
            or not np.array_equal(other.n_cells, expression.n_cells)
            or not np.allclose(
                other.mean_expr, expression.mean_expr, rtol=1e-5, atol=1e-7
            )
        ):
            raise ValueError(
                f"RNA stage means or matched cell counts differ for {run['variant']}."
            )

    scenic, scenic_metadata = None, None
    if settings.get("scenic_reference"):
        scenic, scenic_metadata = load_scenic_profile(
            resolve_config_path(settings, settings["scenic_reference"]),
            gene=gene,
            dar=str(settings.get("scenic_dar", "yes")).lower(),
            deg=str(settings.get("scenic_deg", "yes")).lower(),
            stage_times=expression.time.to_numpy(dtype=float),
        )

    times = baseline["centrality"].time.to_numpy(dtype=float)
    display_times = np.linspace(times[0], times[-1], 300)
    display = pd.DataFrame({"time": display_times})
    frames, summary = [], []
    for run in runs:
        centrality = run["centrality"].copy(deep=True)
        # Smooth raw values once, with identical stage-unit sigma for every model.
        centrality["centrality_smooth"] = gaussian_smooth_1d(
            centrality.centrality_raw.to_numpy(), times, sigma
        )
        centrality["seed"], centrality["variant"] = seed, run["variant"]
        frames.append(centrality)
        display[run["variant"]] = _interpolate(
            times, centrality.centrality_smooth.to_numpy(), display_times
        )
        stage_rows = centrality.loc[centrality.time.isin(expression.time)]
        summary.append(
            {
                "seed": seed,
                "variant": run["variant"],
                "raw_stage_peak": float(
                    stage_rows.loc[stage_rows.centrality_raw.idxmax(), "time"]
                ),
                "gaussian_peak_time": float(
                    centrality.loc[centrality.centrality_smooth.idxmax(), "time"]
                ),
            }
        )
    display["mean_expr"] = _interpolate(
        expression.time.to_numpy(dtype=float),
        expression.mean_expr.to_numpy(),
        display_times,
    )
    output_root = resolve_config_path(settings, settings["output_root"])
    directory = output_root / f"seed_{seed}_{uuid.uuid4().hex[:12]}"
    directory.mkdir(parents=True, exist_ok=False)
    prefix = f"mondo_seed{seed}_supplement"
    dense = pd.concat(frames, ignore_index=True)
    dense.to_csv(directory / f"{prefix}_centrality.tsv", sep="\t", index=False)
    expression.to_csv(directory / f"{prefix}_expression.tsv", sep="\t", index=False)
    display.to_csv(directory / f"{prefix}_display_curves.tsv", sep="\t", index=False)
    pd.DataFrame(summary).to_csv(
        directory / f"{prefix}_summary.tsv", sep="\t", index=False
    )

    title = settings.get(
        "title", f"{gene.upper()} dynamics: DRAFT and ablations (seed {seed})"
    )
    figures = []
    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 14,
            "axes.linewidth": 1.1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    ):
        fig, left = plt.subplots(figsize=tuple(settings.get("figsize", [11, 7])))
        right = left.twinx()
        for variant, (label, color) in styles.items():
            raw = dense.loc[dense.variant.eq(variant)]
            left.scatter(
                raw.time,
                raw.centrality_raw,
                color=color,
                alpha=0.20,
                s=24,
                edgecolors="none",
            )
            left.plot(
                display.time, display[variant], color=color, linewidth=3, label=label
            )
        purple = "#795099"
        right.scatter(
            expression.time,
            expression.mean_expr,
            color=purple,
            alpha=0.3,
            s=25,
            edgecolors="none",
        )
        right.plot(
            display.time,
            display.mean_expr,
            color=purple,
            linewidth=2.6,
            linestyle="--",
            label=f"Mean {gene.upper()} expression",
        )
        left.set_xlabel("Germline trajectory", fontsize=17, labelpad=12)
        left.set_ylabel("Network centrality", fontsize=17, labelpad=12)
        right.set_ylabel(
            "Mean log-normalized expression", color=purple, fontsize=17, labelpad=12
        )
        right.tick_params(axis="y", colors=purple)
        right.spines["right"].set_color(purple)
        left.spines["right"].set_visible(False)
        right.spines["left"].set_visible(False)
        for ax in [left, right]:
            ax.spines["top"].set_visible(False)
        set_germline_stage_axis(left)
        left.set_xlim(times[0] - 0.4, times[-1] + 0.4)
        centrality_max = max(
            float(dense.centrality_raw.max()),
            float(display[list(styles)].max().max()),
            1,
        )
        expr_max = max(
            float(display.mean_expr.max()), float(expression.mean_expr.max()), 0.01
        )
        left.set_ylim(-0.035 * centrality_max, 1.08 * centrality_max)
        right.set_ylim(-0.025 * expr_max, 1.08 * expr_max)
        lines = left.lines + right.lines
        fig.legend(
            lines,
            [line.get_label() for line in lines],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.93),
            ncol=2,
            frameon=False,
            fontsize=13,
        )
        fig.suptitle(title, y=0.985, fontsize=18)
        fig.subplots_adjust(left=0.10, right=0.88, bottom=0.24, top=0.77)
        try:
            for suffix in ["png", "pdf", "svg"]:
                path = directory / f"{prefix}.{suffix}"
                fig.savefig(path, dpi=int(settings.get("dpi", 600)), facecolor="white")
                figures.append(str(path))
        finally:
            plt.close(fig)
    caption = (
        f"**Supplementary Fig. Sx | {gene.upper()} network dynamics in DRAFT and ablation models.** "
        f"Selected-network centrality is shown for the manuscript DRAFT model (GAT + GRU), "
        f"{styles[gat_variant][0]}, and the GAT-to-Linear ablation retaining GRU, all using seed {seed} "
        "(left axis). The dashed purple curve shows the stage-wise mean log-normalized RNA "
        "expression, including zero-expression cells (right axis). Centrality is the sum of "
        "exported outgoing TF–TG edge scores after the same network-selection procedure. "
        f"Centrality curves use Gaussian smoothing (sigma = {sigma:g} stage units) followed "
        "by cubic interpolation; expression uses cubic interpolation of stage means. "
        "Dots show values before plotting smoothing. Dense centrality samples derive from "
        "interpolated model representations and do not constitute additional biological "
        "observations. Curves represent one seed; no between-seed uncertainty is shown.\n"
    )
    if gat_variant == "separate_gat":
        caption += (
            "The separate GAT variant retains the shared DRAFT spatial encoder and "
            "removes both GRU recurrence and adjacent-stage training regularization; "
            "the smoothness coefficient is fixed to zero. Display smoothing is retained "
            "as described above and is separate from training regularization.\n"
        )
    (directory / "figure_caption.md").write_text(caption, encoding="utf-8")
    scenic_report = None
    if scenic is not None:
        added, scenic_report = plot_scenic_comparison(
            display, dense, expression, scenic, scenic_metadata, settings, directory
        )
        figures.extend(added)
    if gat_variant == "separate_gat":
        figures.extend(
            plot_stage_anchors(runs, expression, scenic, settings, directory)
        )
    report = {
        "seed": seed,
        "gene": gene,
        "title": title,
        "figures": figures,
        "output_dir": str(directory),
        "network_settings": baseline["network_settings"],
        "sigma_time": sigma,
        "scenic_comparison": scenic_report,
        "model_labels": {k: v[0] for k, v in styles.items()},
        "gat_variant": gat_variant,
        "stage_labels": germline_stage_labels(),
        "sources": [
            {k: v for k, v in run.items() if k not in {"centrality", "expression"}}
            for run in runs
        ],
        "methods": {
            "centrality": "sum of selected outgoing edge prob; original values, no rescaling",
            "centrality_display": "Gaussian smoothing of raw dense values once, then nonnegative cubic interpolation",
            "expression": "shared stage-wise lognorm mean including zeros; cubic interpolation only",
            "uncertainty": "one seed per model; no SD band",
            "execution": "read-only saved tables; no model training, inference, or network export",
            "stage_anchor_panel": "original-stage raw readouts only; no Gaussian smoothing, intermediate dense samples or spline interpolation",
        },
    }
    (directory / "figure_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
