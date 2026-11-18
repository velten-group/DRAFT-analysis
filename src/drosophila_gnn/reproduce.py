"""Embeddings, stage distances and Hi-C scores of the germline and soma models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from network_evaluation.temporal import json_finite

from .analysis.repro import (
    build_germ_labels,
    build_soma_labels,
    build_tg_alias2idx,
    f1_at_threshold_over_time,
    make_unique_labels,
    map_hic_pairs,
    plot_smoothness_heatmap,
    save_embeddings_by_label,
    save_table,
    summarize_smoothness,
    tf_tg_prob_frobenius_dist_matrix,
)
from .checkpoint import save_checkpoint_metadata
from .infer import reload_germ_from_checkpoint, reload_soma_from_checkpoint
from .utils import (
    emit_progress,
    ensure_directory,
    load_yaml_config,
    resolve_config_path,
)

DEFAULT_GERM_CELLTYPES = [
    "GSC",
    "Gb",
    "Sp-2",
    "Sp-4",
    "Sp-8",
    "Sp-16",
    "1deg Sc",
    "1deg Sc",
    "1deg Sc",
    "2deg Sc",
    "2deg Sc",
    "2deg Sc",
    "2deg Sc",
    "2deg Sc",
    "2deg Sc",
]


def _section(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    return value if isinstance(value, dict) else {}


def _reproduce_output_dir(config: dict[str, Any], default_name: str) -> Path:
    reproduce_cfg = _section(config, "reproduce")
    return ensure_directory(resolve_config_path(config, reproduce_cfg.get("output_dir", default_name)))


def _save_embeddings_enabled(config: dict[str, Any]) -> bool:
    reproduce_cfg = _section(config, "reproduce")
    return bool(reproduce_cfg.get("save_embeddings", True))


def _cluster_ids(config: dict[str, Any], time_count: int) -> list[str]:
    reproduce_cfg = _section(config, "reproduce")
    provided = reproduce_cfg.get("cluster_ids")
    if isinstance(provided, list) and len(provided) == time_count:
        return [str(value) for value in provided]
    return [f"t{idx}" for idx in range(time_count)]


def _resolve_hic_path(config: dict[str, Any], key: str, default_name: str) -> Path:
    reproduce_cfg = _section(config, "reproduce")
    hic_cfg = reproduce_cfg.get("hic", {})
    value = None
    if isinstance(hic_cfg, dict):
        value = hic_cfg.get(key)
    base_dir = Path(config.get("base_dir", ".")).expanduser().resolve()
    target = Path(value) if value else Path(default_name)
    if not target.is_absolute():
        target = base_dir / target
    return target.resolve()


def _smoothness_settings(config: dict[str, Any]) -> dict[str, Any]:
    reproduce_cfg = _section(config, "reproduce")
    smooth_cfg = reproduce_cfg.get("smoothness", {})
    if not isinstance(smooth_cfg, dict):
        smooth_cfg = {}
    return {
        "row_bs": int(smooth_cfg.get("row_bs", 1024)),
        "col_bs": int(smooth_cfg.get("col_bs", 4096)),
        "save_heatmap": bool(smooth_cfg.get("save_heatmap", True)),
    }


def _hic_settings(config: dict[str, Any]) -> dict[str, Any]:
    reproduce_cfg = _section(config, "reproduce")
    hic_cfg = reproduce_cfg.get("hic", {})
    if not isinstance(hic_cfg, dict):
        hic_cfg = {}
    return {
        "enabled": bool(hic_cfg.get("enabled", True)),
        "threshold": float(hic_cfg.get("threshold", 0.5)),
        "rng_seed": int(hic_cfg.get("rng_seed", 42)),
        "pair_batch": int(hic_cfg.get("pair_batch", 200_000)),
        "spc_timepoints": [int(value) for value in hic_cfg.get("spc_timepoints", list(range(6, 15)))],
        "spg_timepoints": [int(value) for value in hic_cfg.get("spg_timepoints", list(range(2, 6)))],
    }


def _save_matrix(path: str | Path, matrix: np.ndarray) -> str:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path, matrix, fmt="%.10f", delimiter="\t")
    return str(output_path)


def reproduce_soma(config_path: str | Path) -> dict[str, Any]:
    config = load_yaml_config(config_path)
    output_dir = _reproduce_output_dir(config, "result/soma/reproduce")
    smoothness_cfg = _smoothness_settings(config)

    inference = reload_soma_from_checkpoint(config)
    out_tf = inference["out_tf"]
    out_re = inference["out_re"]
    out_tg = inference["out_tg"]
    time_count = int(out_tf.size(1))
    checkpoint_meta = inference.get("checkpoint", {}).get("meta", {})

    labels = build_soma_labels(
        time_count,
        somatic_names=inference.get("somatic_names"),
        cluster_ids=_cluster_ids(config, time_count),
    )

    if _save_embeddings_enabled(config):
        saved_embeddings = save_embeddings_by_label(out_tf, out_re, out_tg, labels, output_dir)
    else:
        saved_embeddings = []

    dist_mat = tf_tg_prob_frobenius_dist_matrix(
        out_tf,
        out_tg,
        row_bs=smoothness_cfg["row_bs"],
        col_bs=smoothness_cfg["col_bs"],
    )
    smoothness = summarize_smoothness(dist_mat)

    if len(labels) >= time_count:
        plot_labels = labels[:time_count]
    else:
        plot_labels = labels + [f"t{idx}" for idx in range(len(labels), time_count)]

    heatmap_path = None
    if smoothness_cfg["save_heatmap"]:
        heatmap_path = output_dir / "smoothness_tf_tg_heatmap_YlOrRd_somatic_norm.png"
    saved_heatmap = plot_smoothness_heatmap(
        smoothness["dist_mat_norm"],
        plot_labels,
        float(smoothness["overall_smoothness"]),
        "TF-TG Smoothness Heatmap - Somatic",
        output_path=heatmap_path,
        cbar_label="Frobenius distance",
    )
    matrix_path = _save_matrix(output_dir / "tf_tg_prob_frobenius_dist_matrix.tsv", dist_mat)

    report = {
        "lineage": "soma",
        "config_path": str(Path(config_path).resolve()),
        "output_dir": str(output_dir),
        "artifact_path": inference["artifact_path"],
        "checkpoint_path": inference["checkpoint_path"],
        "seed": checkpoint_meta.get("seed"),
        "labels": labels,
        "saved_embeddings": saved_embeddings,
        "smoothness": {
            "avg_adjacent": float(smoothness["avg_adjacent"]),
            "max_pairwise": float(smoothness["max_pairwise"]),
            "overall_smoothness": float(smoothness["overall_smoothness"]),
            "metric_version": smoothness["metric_version"],
            "velocity_roughness": float(smoothness["velocity_roughness"]),
            "constant_over_time": smoothness["constant_over_time"],
            "largest_jump_interval": smoothness["largest_jump_interval"],
            "adjacent_mean_smoothness": smoothness[
                "adjacent_mean_smoothness"
            ],
            "matrix_path": matrix_path,
            "heatmap_path": saved_heatmap,
        },
        "status": "reproduced",
    }
    report = json_finite(report)
    save_checkpoint_metadata(output_dir / "reproduce_report.json", report)
    return report


def reproduce_germ(config_path: str | Path) -> dict[str, Any]:
    config = load_yaml_config(config_path)
    output_dir = _reproduce_output_dir(config, "result/germ/reproduce")
    smoothness_cfg = _smoothness_settings(config)
    hic_cfg = _hic_settings(config)

    inference = reload_germ_from_checkpoint(config)
    artifacts = inference["artifacts"]
    out_tf = inference["out_tf"]
    out_re = inference["out_re"]
    out_tg = inference["out_tg"]
    time_count = int(out_tf.size(1))
    cluster_ids = _cluster_ids(config, time_count)
    checkpoint = inference.get("checkpoint", {})

    labels = build_germ_labels(
        time_count,
        germline_types=inference.get("germline_types"),
        cluster_ids=cluster_ids,
    )
    if _save_embeddings_enabled(config):
        saved_embeddings = save_embeddings_by_label(out_tf, out_re, out_tg, labels, output_dir)
    else:
        saved_embeddings = []

    maps = artifacts["maps"]
    re2idx = {str(name): idx for idx, name in enumerate(maps["re"])}
    tg2idx = {str(name): idx for idx, name in enumerate(maps["tg"])}
    input_paths = artifacts["common"]["input_paths"]
    tg_alias2idx = build_tg_alias2idx(tg2idx, input_paths.tg_list)

    hic_results: dict[str, Any] = {}
    if hic_cfg["enabled"]:
        evaluations = [
            ("spc", _resolve_hic_path(config, "spc_path", "spc_hic.tsv"), hic_cfg["spc_timepoints"]),
            ("spg", _resolve_hic_path(config, "spg_path", "spg_hic.tsv"), hic_cfg["spg_timepoints"]),
        ]
        for tag, hic_path, timepoints in evaluations:
            emit_progress("PROGRESS", "reproduce:germ", f"hic={tag}")
            mapped = map_hic_pairs(hic_path, re2idx, tg_alias2idx, out_re.device)
            idx_list, f1_values = f1_at_threshold_over_time(
                mapped["pairs_idx"],
                out_re,
                out_tg,
                timepoints=timepoints,
                thr=hic_cfg["threshold"],
                rng_seed=hic_cfg["rng_seed"],
                pair_batch=hic_cfg["pair_batch"],
            )
            frame = pd.DataFrame(
                {
                    "time_idx": idx_list,
                    "label": [cluster_ids[idx] for idx in idx_list],
                    "f1_at_0_5": f1_values,
                }
            )
            table_path = save_table(output_dir / f"hic_f1_{tag}.tsv", frame)
            hic_results[tag] = {
                "source_path": mapped["source_path"],
                "valid_count": int(mapped["valid_count"]),
                "miss_re": int(mapped["miss_re"]),
                "miss_tg": int(mapped["miss_tg"]),
                "timepoints": idx_list,
                "f1_at_0_5": [float(value) for value in f1_values],
                "table_path": table_path,
            }

    dist_mat = tf_tg_prob_frobenius_dist_matrix(
        out_tf,
        out_tg,
        row_bs=smoothness_cfg["row_bs"],
        col_bs=smoothness_cfg["col_bs"],
    )
    smoothness = summarize_smoothness(dist_mat)

    reproduce_cfg = _section(config, "reproduce")
    configured_labels = reproduce_cfg.get("germ_celltypes")
    if isinstance(configured_labels, list) and len(configured_labels) == time_count:
        plot_labels = make_unique_labels([str(value) for value in configured_labels])
    elif len(DEFAULT_GERM_CELLTYPES) == time_count:
        plot_labels = make_unique_labels(DEFAULT_GERM_CELLTYPES)
    else:
        plot_labels = list(cluster_ids)

    heatmap_path = None
    if smoothness_cfg["save_heatmap"]:
        heatmap_path = output_dir / "smoothness_tf_tg_heatmap_YlOrRd_germ_norm.png"
    saved_heatmap = plot_smoothness_heatmap(
        smoothness["dist_mat_norm"],
        plot_labels,
        float(smoothness["overall_smoothness"]),
        "TF-TG Smoothness Heatmap--Germline ",
        output_path=heatmap_path,
        cbar_label=None,
    )
    matrix_path = _save_matrix(output_dir / "tf_tg_prob_frobenius_dist_matrix.tsv", dist_mat)

    report = {
        "lineage": "germ",
        "config_path": str(Path(config_path).resolve()),
        "output_dir": str(output_dir),
        "checkpoint_path": inference["checkpoint_path"],
        "seed": checkpoint.get("seed"),
        "timestamp": checkpoint.get("timestamp"),
        "labels": labels,
        "saved_embeddings": saved_embeddings,
        "hic": hic_results,
        "smoothness": {
            "avg_adjacent": float(smoothness["avg_adjacent"]),
            "max_pairwise": float(smoothness["max_pairwise"]),
            "overall_smoothness": float(smoothness["overall_smoothness"]),
            "metric_version": smoothness["metric_version"],
            "velocity_roughness": float(smoothness["velocity_roughness"]),
            "constant_over_time": smoothness["constant_over_time"],
            "largest_jump_interval": smoothness["largest_jump_interval"],
            "adjacent_mean_smoothness": smoothness[
                "adjacent_mean_smoothness"
            ],
            "matrix_path": matrix_path,
            "heatmap_path": saved_heatmap,
            "plot_labels": plot_labels,
        },
        "status": "reproduced",
    }
    report = json_finite(report)
    save_checkpoint_metadata(output_dir / "reproduce_report.json", report)
    return report


def reproduce_lineage_model(config_path: str | Path) -> dict[str, Any]:
    config = load_yaml_config(config_path)
    lineage = str(config.get("lineage", "")).lower()
    if lineage in {"soma", "somatic"}:
        return reproduce_soma(config_path)
    if lineage in {"germ", "germline"}:
        return reproduce_germ(config_path)
    raise ValueError(f"Unsupported lineage '{config.get('lineage')}'. Expected soma or germ.")
