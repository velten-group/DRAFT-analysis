"""Stages that load a checkpoint: embeddings and stage distances, network export.

Each function runs the manuscript code (drosophila_gnn) with the manuscript configuration; only input and output
paths are changed (draft_analysis.config). Inference runs on the CPU when no GPU is present.
"""
from __future__ import annotations

from pathlib import Path

from .. import config, paths


def reproduce(lineage: str) -> dict:
    """Stage embeddings and the TF-TG Frobenius distance matrix between stages (Fig. 4c, Supp. Fig. 7).

    Writes results/fly/<lineage>/reproduce/: embedding_*.txt per stage and tf_tg_prob_frobenius_dist_matrix.tsv.
    """
    from drosophila_gnn.reproduce import reproduce_germ, reproduce_soma

    over = {"reproduce": {"checkpoint_path": str(config.checkpoint(lineage)), "hic": {"enabled": False}}}
    if lineage == "soma":
        over["reproduce"]["artifact_path"] = str(paths.data("fly", "checkpoints", "soma_graphs.pt"))
    cfg = config.load(f"{lineage}_reproduce", over)
    cfg["reproduce"]["smoothness"] = {**cfg["reproduce"].get("smoothness", {}), "save_heatmap": False}
    path = config.write(cfg, f"{lineage}_reproduce")
    return (reproduce_germ if lineage == "germ" else reproduce_soma)(path)


def network(lineage: str, seed: int = 2025, out: Path | None = None) -> dict:
    """Stage-specific networks: strict TF-TG and TF-RE-TG networks (network/) and dense 50,000-edge TF-TG networks on
    the interpolated time axis (network_dense/). Seeds other than 2025 are the germline replicate models (Fig. 5a)."""
    from drosophila_gnn.analysis.network import export_lineage_networks

    base = out or (paths.FLY / lineage if seed == 2025 else paths.FLY / f"germ_seed{seed}")
    over = {"reproduce": {"checkpoint_path": str(config.checkpoint(lineage, seed))}}
    if lineage == "soma":
        over["reproduce"]["artifact_path"] = str(paths.data("fly", "checkpoints", "soma_graphs.pt"))
    cfg = config.load(f"{lineage}_network", over)
    cfg["network"]["output_dir"] = str(base / "network")
    cfg["network"]["dense"]["output_dir"] = str(base / "network_dense")
    path = config.write(cfg, f"{lineage}_network" + ("" if seed == 2025 else f"_seed{seed}"))
    return export_lineage_networks(path)
