"""Continuous TF activity visualization on dense network exports."""

from __future__ import annotations

import glob
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline

from ..checkpoint import save_checkpoint_metadata
from ..io import strip_copy_suffix
from ..utils import ensure_directory, load_yaml_config


def _section(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    return value if isinstance(value, dict) else {}


def _default_dense_file_patterns() -> list[str]:
    return [
        "TF_TG_dense_t*.tsv",
        "t*_TF-TG.tsv",
        "TF_TG_globalTopK_rescue_dense_*.tsv",
        "TF_TG_top10000_*.tsv",
    ]


def _continuous_activity_settings(config: dict[str, Any]) -> dict[str, Any]:
    section = _section(config, "continuous_activity")
    lineage = str(config.get("lineage", "")).lower()
    tf_list = section.get("tf_list")
    if tf_list is None:
        tf_symbol = section.get("tf_symbol")
        tf_list = [tf_symbol] if tf_symbol else []
    elif not isinstance(tf_list, list):
        tf_list = [tf_list]

    colors = section.get("tf_colors", {})
    if not isinstance(colors, dict):
        colors = {}

    data_cfg = _section(config, "data")
    tf_annot_path = section.get("tf_annot_path", data_cfg.get("tf_list", "TF.tsv"))

    return {
        "network_dir": str(section.get("network_dir", f"result/{lineage}/network_dense")),
        "output_dir": str(section.get("output_dir", f"result/{lineage}/continuous_activity")),
        "file_patterns": section.get(
            "file_patterns",
            _default_dense_file_patterns(),
        ),
        "tf_annot_path": str(tf_annot_path),
        "tf_gene_col": str(section.get("tf_gene_col", "gene_name")),
        "tf_list": [str(value) for value in tf_list if str(value).strip()],
        "sigma_time": float(section.get("sigma_time", 0.5)),
        "output_figure": str(section.get("output_figure", "Figure_MultiTF_Centrality.pdf")),
        "output_table": str(section.get("output_table", "multi_tf_activity.tsv")),
        "save_fig": bool(section.get("save_fig", True)),
        "save_table": bool(section.get("save_table", True)),
        "show_raw_points": bool(section.get("show_raw_points", True)),
        "prefer_prob": bool(section.get("prefer_prob", True)),
        "index_base": str(section.get("index_base", "infer")),
        "tf_colors": {str(key).lower(): str(value) for key, value in colors.items()},
        "figure_width": float(section.get("figure_width", 7.0)),
        "figure_height": float(section.get("figure_height", 5.5)),
        "title": str(section.get("title", "Continuous Network Dynamics")),
        "xlabel": str(section.get("xlabel", "Normalized Time")),
        "ylabel": str(section.get("ylabel", "Centrality")),
    }


def _parse_time_label(value: str) -> float | None:
    try:
        return float(str(value))
    except ValueError:
        return None


def extract_time_from_filename(path: str | Path) -> float | None:
    base = Path(path).name
    patterns = [
        r"^TF_TG_dense_t(.+)\.tsv$",
        r"^t([0-9]+(?:\.[0-9]+)?)_TF-TG\.tsv$",
        r"^TF_TG_globalTopK_rescue_dense_(.+)\.tsv$",
        r"^TF_TG_top10000_(.+)\.tsv$",
    ]
    for pattern in patterns:
        match = re.match(pattern, base)
        if match:
            raw = match.group(1)
            if raw.startswith("t"):
                raw = raw[1:]
            try:
                return float(raw)
            except ValueError:
                return None
    return None


def gaussian_smooth_1d(y: np.ndarray, t: np.ndarray, sigma: float) -> np.ndarray:
    values = np.asarray(y, dtype=float)
    times = np.asarray(t, dtype=float)

    if np.any(np.isnan(values)):
        series = pd.Series(values)
        values = (
            series.interpolate(limit_direction="both")
            .bfill()
            .ffill()
            .to_numpy(dtype=float)
        )

    if len(times) > 1:
        unique_times = np.sort(np.unique(times))
        dt = float(np.median(np.diff(unique_times))) if len(unique_times) > 1 else 1.0
    else:
        dt = 1.0

    radius = max(1, int(round(3 * sigma / dt)))
    offsets = np.arange(-radius, radius + 1)
    kernel_t = offsets * dt
    weights = np.exp(-0.5 * (kernel_t / sigma) ** 2)
    weights = weights / weights.sum()

    padded = np.pad(values, (radius, radius), mode="edge")
    return np.convolve(padded, weights, mode="valid")


def plot_smooth_curve(
    ax: Any,
    x: np.ndarray,
    y_smooth: np.ndarray,
    color: str,
    label: str,
    raw_y: np.ndarray | None = None,
) -> None:
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y_smooth, dtype=float)

    if raw_y is not None:
        ax.scatter(x_values, raw_y, color=color, alpha=0.28, s=50, edgecolors="none", zorder=1)

    unique_x = np.unique(x_values)
    if len(unique_x) >= 4:
        frame = pd.DataFrame({"x": x_values, "y": y_values}).groupby("x", as_index=False)["y"].mean()
        x2 = frame["x"].to_numpy(dtype=float)
        y2 = frame["y"].to_numpy(dtype=float)
        if len(x2) >= 4:
            try:
                x_new = np.linspace(x2.min(), x2.max(), 300)
                spline = make_interp_spline(x2, y2, k=3)
                y_new = spline(x_new)
                if np.min(y2) >= 0:
                    y_new = np.clip(y_new, 0, None)
                ax.plot(x_new, y_new, color=color, linewidth=3.2, label=label, zorder=2)
                return
            except Exception:
                pass

    ax.plot(x_values, y_values, color=color, linewidth=3.2, label=label, zorder=2)


def pick_color(tf_lower: str, idx: int, default_cycle: list[str], user_colors: dict[str, str]) -> str:
    if tf_lower in user_colors:
        return user_colors[tf_lower]
    return default_cycle[idx % len(default_cycle)]


def _load_tf_annotation(path: Path, gene_col: str) -> tuple[list[str], dict[int, str]]:
    frame = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    if gene_col not in frame.columns:
        lower = {str(column).lower(): str(column) for column in frame.columns}
        if gene_col.lower() in lower:
            gene_col = lower[gene_col.lower()]
        elif "gene_name" in lower:
            gene_col = lower["gene_name"]
        elif "symbol" in lower:
            gene_col = lower["symbol"]
        else:
            gene_col = str(frame.columns[0])
    tf_names = frame[gene_col].astype(str).map(strip_copy_suffix).tolist()
    idx2tf_name = {idx + 1: name for idx, name in enumerate(tf_names)}
    return tf_names, idx2tf_name


def _clean_exported_tf_names(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
        .str.replace(r"^tf:", "", regex=True, case=False)
        .map(strip_copy_suffix)
    )


def _discover_dense_files(network_dir: Path, file_patterns: list[str]) -> list[tuple[float, Path]]:
    pairs: list[tuple[float, Path]] = []
    for pattern in file_patterns:
        for path_str in glob.glob(str(network_dir / pattern)):
            time_value = extract_time_from_filename(path_str)
            if time_value is not None:
                pairs.append((time_value, Path(path_str)))
    dedup: dict[str, tuple[float, Path]] = {}
    for time_value, path in pairs:
        dedup[str(path.resolve())] = (time_value, path.resolve())
    unique_pairs = list(dedup.values())
    unique_pairs.sort(key=lambda item: (item[0], item[1].name))
    return unique_pairs


def _find_col_by_lower(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    column_map = {str(column).lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        if candidate in column_map:
            return column_map[candidate]
    return None


def _parse_tf_idx_series(series: pd.Series, index_base: str, tf_count: int) -> pd.Series:
    parsed = pd.to_numeric(series, errors="coerce").astype("Int64")
    base = index_base.lower()
    if base == "zero":
        return parsed + 1
    if base == "one":
        return parsed

    valid = parsed.dropna().astype(int)
    if valid.empty:
        return parsed
    if int(valid.min()) == 0:
        return parsed + 1
    if int(valid.max()) == tf_count:
        return parsed
    if int(valid.max()) < tf_count:
        return parsed + 1
    return parsed


def _centrality_series(frame: pd.DataFrame, prefer_prob: bool) -> pd.Series:
    col_map = {str(column).lower(): str(column) for column in frame.columns}
    if prefer_prob and "prob" in col_map:
        column = col_map["prob"]
        return frame.groupby("tf_name_lower")[column].sum()
    if "prob" in col_map:
        column = col_map["prob"]
        return frame.groupby("tf_name_lower")[column].sum()
    for candidate in ["score_cos", "score_dot", "score"]:
        if candidate in col_map:
            column = col_map[candidate]
            return frame.groupby("tf_name_lower")[column].sum()

    tg_column = _find_col_by_lower(frame, ["tg_idx", "tg"])
    if tg_column is None:
        tg_column = str(frame.columns[1] if len(frame.columns) > 1 else frame.columns[0])
    return frame.groupby("tf_name_lower")[tg_column].nunique()


def run_continuous_activity_from_config(config: dict[str, Any]) -> dict[str, Any]:
    settings = _continuous_activity_settings(config)
    if not settings["tf_list"]:
        raise ValueError("continuous_activity.tf_list or continuous_activity.tf_symbol must be provided.")

    base_dir = Path(config.get("base_dir", ".")).expanduser().resolve()
    network_dir = Path(settings["network_dir"]).expanduser()
    if not network_dir.is_absolute():
        network_dir = base_dir / network_dir
    network_dir = network_dir.resolve()
    output_dir = Path(settings["output_dir"]).expanduser()
    if not output_dir.is_absolute():
        output_dir = base_dir / output_dir
    output_dir = ensure_directory(output_dir.resolve())

    tf_annot_path = Path(settings["tf_annot_path"]).expanduser()
    if not tf_annot_path.is_absolute():
        tf_annot_path = base_dir / tf_annot_path
    tf_names, idx2tf_name = _load_tf_annotation(tf_annot_path.resolve(), settings["tf_gene_col"])

    configured_patterns = [str(pattern) for pattern in settings["file_patterns"]]
    pairs = _discover_dense_files(network_dir, configured_patterns)
    used_patterns = configured_patterns
    if not pairs:
        fallback_patterns = _default_dense_file_patterns()
        if configured_patterns != fallback_patterns:
            pairs = _discover_dense_files(network_dir, fallback_patterns)
            if pairs:
                used_patterns = fallback_patterns
    if not pairs:
        raise FileNotFoundError(
            f"No dense TF-TG files matched under {network_dir} with patterns {configured_patterns}."
        )

    times = np.asarray([item[0] for item in pairs], dtype=float)
    files = [item[1] for item in pairs]
    tf_list_lower = [value.lower() for value in settings["tf_list"]]
    activity: dict[str, list[float]] = {value: [] for value in tf_list_lower}
    for time_value, path in pairs:
        try:
            frame = pd.read_csv(path, sep="\t")
        except Exception:
            for tf_lower in tf_list_lower:
                activity[tf_lower].append(0.0)
            continue

        frame = frame.copy()
        tf_name_col = _find_col_by_lower(frame, ["tf", "tf_name", "tf_symbol", "tf_gene"])
        if tf_name_col is not None:
            frame["tf_name"] = _clean_exported_tf_names(frame[tf_name_col])
        else:
            tf_idx_col = _find_col_by_lower(frame, ["tf_idx", "tfid", "tf_index"])
            if tf_idx_col is None:
                for tf_lower in tf_list_lower:
                    activity[tf_lower].append(0.0)
                continue
            tf_idx_one_based = _parse_tf_idx_series(frame[tf_idx_col], settings["index_base"], len(tf_names))
            frame["tf_name"] = tf_idx_one_based.map(idx2tf_name)
        frame = frame.dropna(subset=["tf_name"])
        frame["tf_name_lower"] = frame["tf_name"].astype(str).str.lower()

        centrality = _centrality_series(frame, settings["prefer_prob"])
        for tf_lower in tf_list_lower:
            value = float(centrality.get(tf_lower, 0.0))
            activity[tf_lower].append(value)

    for tf_lower in tf_list_lower:
        activity[tf_lower] = list(np.asarray(activity[tf_lower], dtype=float))

    if len(np.unique(times)) < len(times):
        merged = pd.DataFrame({"time": times})
        for tf_lower in tf_list_lower:
            merged[tf_lower] = np.asarray(activity[tf_lower], dtype=float)
        merged = merged.groupby("time", as_index=False).mean(numeric_only=True)
        times = merged["time"].to_numpy(dtype=float)
        for tf_lower in tf_list_lower:
            activity[tf_lower] = list(merged[tf_lower].to_numpy(dtype=float))

    smooth: dict[str, np.ndarray] = {}
    for tf_lower in tf_list_lower:
        smooth[tf_lower] = gaussian_smooth_1d(np.asarray(activity[tf_lower], dtype=float), times, settings["sigma_time"])

    fig, ax = plt.subplots(figsize=(settings["figure_width"], settings["figure_height"]))
    default_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#000000"])

    for idx, tf_name in enumerate(settings["tf_list"]):
        tf_lower = tf_name.lower()
        color = pick_color(tf_lower, idx, default_cycle, settings["tf_colors"])
        plot_smooth_curve(
            ax,
            times,
            smooth[tf_lower],
            color=color,
            label=tf_name,
            raw_y=np.asarray(activity[tf_lower], dtype=float) if settings["show_raw_points"] else None,
        )

    ax.set_title(settings["title"], loc="center", fontsize=20, pad=15)
    ax.set_xlabel(settings["xlabel"], fontsize=18, fontweight="medium")
    ax.set_ylabel(settings["ylabel"], fontsize=18, fontweight="medium")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    y_max = max(float(np.max(np.asarray(activity[tf_lower], dtype=float))) for tf_lower in tf_list_lower)
    ax.set_ylim(0, y_max * 1.05 if y_max > 0 else 1.0)

    n_tf = len(settings["tf_list"])
    ncol = 1 if n_tf <= 6 else 2
    legend_fs = 16 if n_tf <= 6 else 12
    ax.legend(frameon=False, fontsize=legend_fs, loc="upper right", ncol=ncol)
    plt.tight_layout()

    figure_path = output_dir / settings["output_figure"]
    table_path = output_dir / settings["output_table"]

    if settings["save_fig"]:
        fig.savefig(figure_path, dpi=300, transparent=True)
    plt.close(fig)

    rows = []
    for tf_name in settings["tf_list"]:
        tf_lower = tf_name.lower()
        for time_value, raw_value, smooth_value in zip(times.tolist(), activity[tf_lower], smooth[tf_lower].tolist()):
            rows.append(
                {
                    "time": float(time_value),
                    "tf_query": str(tf_name),
                    "centrality_raw": float(raw_value),
                    "centrality_smooth": float(smooth_value),
                }
            )
    table = pd.DataFrame(rows).sort_values(["tf_query", "time"], kind="mergesort").reset_index(drop=True)
    if settings["save_table"]:
        table.to_csv(table_path, sep="\t", index=False)

    report = {
        "analysis": "continuous_activity",
        "output_dir": str(output_dir),
        "network_dir": str(network_dir),
        "file_count": int(len(files)),
        "used_file_patterns": used_patterns,
        "times": [float(value) for value in times.tolist()],
        "tf_list": settings["tf_list"],
        "figure_path": str(figure_path) if settings["save_fig"] else None,
        "table_path": str(table_path) if settings["save_table"] else None,
        "settings": settings,
    }
    save_checkpoint_metadata(output_dir / "continuous_activity_report.json", report)
    return report


def run_continuous_activity(config: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(config, (str, Path)):
        loaded = load_yaml_config(config)
        report = run_continuous_activity_from_config(loaded)
        report["config_path"] = str(Path(config).resolve())
        return report
    return run_continuous_activity_from_config(config)
