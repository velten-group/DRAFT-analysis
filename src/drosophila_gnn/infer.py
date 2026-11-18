"""Checkpoint loading and inference helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .checkpoint import find_latest_checkpoint, torch_load_compat
from .graph_builder import build_germ_graph_artifacts, unify_feature_dim
from .models.gnn import build_model_for_lineage
from .utils import resolve_config_path


def _select_runtime_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _reproduce_section(config: dict[str, Any]) -> dict[str, Any]:
    value = config.get("reproduce", {})
    return value if isinstance(value, dict) else {}


def _soma_artifact_path(config: dict[str, Any]) -> Path:
    reproduce_cfg = _reproduce_section(config)
    explicit = reproduce_cfg.get("artifact_path")
    if explicit:
        return resolve_config_path(config, explicit)
    artifacts_cfg = config.get("artifacts", {})
    artifact_dir = resolve_config_path(config, artifacts_cfg.get("dir", "result/soma/train/artifacts"))
    return artifact_dir / "soma_graphs.pt"


def _soma_checkpoint_path(config: dict[str, Any]) -> Path:
    reproduce_cfg = _reproduce_section(config)
    explicit = reproduce_cfg.get("checkpoint_path")
    if explicit:
        return resolve_config_path(config, explicit)
    checkpoints_cfg = config.get("checkpoints", {})
    checkpoint_dir = resolve_config_path(config, checkpoints_cfg.get("dir", "result/soma/model"))
    pattern = str(reproduce_cfg.get("checkpoint_pattern", "*.pt"))
    return find_latest_checkpoint(checkpoint_dir, pattern=pattern)


def _germ_checkpoint_path(config: dict[str, Any]) -> Path:
    reproduce_cfg = _reproduce_section(config)
    explicit = reproduce_cfg.get("checkpoint_path")
    if explicit:
        return resolve_config_path(config, explicit)
    checkpoints_cfg = config.get("checkpoints", {})
    checkpoint_dir = resolve_config_path(config, checkpoints_cfg.get("dir", "result/germ/model"))
    pattern = str(reproduce_cfg.get("checkpoint_pattern", "*.pt"))
    return find_latest_checkpoint(checkpoint_dir, pattern=pattern)


def reload_soma_from_checkpoint(config: dict[str, Any]) -> dict[str, Any]:
    artifact_path = _soma_artifact_path(config)
    checkpoint_path = _soma_checkpoint_path(config)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Soma artifact was not found: {artifact_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Soma checkpoint was not found: {checkpoint_path}")

    device = _select_runtime_device()
    artifact = torch_load_compat(artifact_path, map_location="cpu")
    graphs_sel = artifact["graphs"]
    topo_order = list(artifact["topo_order"])
    parents_list = artifact["parents_list"]
    time_names = list(artifact["time_names"])
    in_channels = int(artifact["in_channels"])
    t_sel = int(artifact["T_sel"])

    graphs_somatic = [graph.to(device) for graph in graphs_sel]
    checkpoint = torch_load_compat(checkpoint_path, map_location=device)
    hp = checkpoint["hp"]

    model = build_model_for_lineage(
        "soma",
        in_channels=in_channels,
        hidden_channels=int(hp["hidden_channels"]),
        out_channels=int(hp["out_channels"]),
        heads=int(hp["gat_heads"]),
        dropout_rate=float(hp["dropout_rate"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    with torch.no_grad():
        out_tf, out_re, out_tg = model(graphs_somatic, topo_order, parents_list)

    return {
        "lineage": "soma",
        "device": str(device),
        "artifact_path": str(artifact_path),
        "checkpoint_path": str(checkpoint_path),
        "graphs_somatic": graphs_somatic,
        "topo_somatic": topo_order,
        "parents_somatic": parents_list,
        "somatic_names": time_names,
        "in_channels": in_channels,
        "T_sel": t_sel,
        "maps": artifact.get("maps"),
        "hp": hp,
        "checkpoint": checkpoint,
        "model": model,
        "out_tf": out_tf,
        "out_re": out_re,
        "out_tg": out_tg,
    }


def reload_germ_from_checkpoint(config: dict[str, Any]) -> dict[str, Any]:
    artifacts = build_germ_graph_artifacts(config)
    graphs_sel = artifacts["graphs_germline"]
    if not graphs_sel:
        raise ValueError("No germline graphs were constructed for germ reproduction.")

    device = graphs_sel[0]["TF"].x.device
    checkpoint_path = _germ_checkpoint_path(config)
    checkpoint = torch_load_compat(checkpoint_path, map_location="cpu")
    hp = checkpoint.get("hp")
    if hp is None:
        raise RuntimeError("The germ checkpoint does not contain 'hp'; model reconstruction failed.")

    in_channels = checkpoint.get("in_channels")
    if in_channels is None:
        in_channels = graphs_sel[0]["TF"].x.size(1)
    in_channels = int(in_channels)
    unify_feature_dim(graphs_sel, in_channels)

    topo_order = checkpoint.get("topo_order", artifacts.get("topo_order", list(range(len(graphs_sel)))))
    parent = checkpoint.get(
        "parent",
        artifacts.get("parent", [-1] + list(range(len(graphs_sel) - 1))),
    )
    topo_order = list(topo_order)
    parent = list(parent)

    model = build_model_for_lineage(
        "germ",
        in_channels=in_channels,
        hidden_channels=int(hp["hidden_channels"]),
        out_channels=int(hp["out_channels"]),
        heads=int(hp["gat_heads"]),
        dropout_rate=float(hp["dropout_rate"]),
        use_gru=bool(hp.get("use_gru", True)),
        spatial_encoder=str(hp.get("spatial_encoder", "gat")),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        out_tf, out_re, out_tg = model(graphs_sel, topo_order, parent)

    return {
        "lineage": "germ",
        "device": str(device),
        "checkpoint_path": str(checkpoint_path),
        "artifacts": artifacts,
        "graphs_germline": graphs_sel,
        "topo_order": topo_order,
        "parent": parent,
        "germline_types": artifacts.get("germline_types"),
        "maps": artifacts.get("maps"),
        "hp": hp,
        "checkpoint": checkpoint,
        "model": model,
        "in_channels": in_channels,
        "out_tf": out_tf,
        "out_re": out_re,
        "out_tg": out_tg,
    }


def run_inference(config: dict[str, Any]) -> dict[str, Any]:
    lineage = str(config.get("lineage", "")).lower()
    if lineage in {"soma", "somatic"}:
        return reload_soma_from_checkpoint(config)
    if lineage in {"germ", "germline"}:
        return reload_germ_from_checkpoint(config)
    raise ValueError(f"Unsupported lineage '{config.get('lineage')}'. Expected soma or germ.")
