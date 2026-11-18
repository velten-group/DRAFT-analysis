"""Control models of Supplementary Figs. 10 and 11: network export from their checkpoints (nothing is trained).

- DRAFT ablations (seeds 2025, 2027, 2028): without GRU, and Linear + GRU (GAT replaced by a linear layer). They use the
  manuscript data, prior and network settings; the architecture is read from the checkpoint.
- Prior controls (seeds 2025, 2026): models trained on a redrawn TF-RE prior, "tf-uniform" (TFs of every RE drawn
  uniformly) and "shuffle" (degree-preserving). Their input graphs are rebuilt with that prior, deposited next to the
  checkpoint. The tf-uniform models and "shuflow2025" (the shuffle seed-2025 model) are exported with the dense score
  threshold off, so the 50,000-edge cap sets every stage.
Outputs: results/checkpoint/fly/controls/<name>/network_dense (and the strict network/ of the same export).
"""
from __future__ import annotations

from pathlib import Path

from .. import config, paths

# name: (checkpoint folder under DATA/fly/checkpoints, TF-RE prior file in that folder or None, dense threshold or None)
CONTROLS = {
    **{f"nogru{s % 100}": (f"ablation/seed_{s}/no_gru", None, None) for s in (2025, 2027, 2028)},
    **{f"lingru{s % 100}": (f"ablation/seed_{s}/linear_gru", None, None) for s in (2025, 2027, 2028)},
    "tfu2025": ("prior_tfuniform/seed_2025", "TF_RE_tfuniform_seed2025.tsv", -1.0),
    "tfu2026": ("prior_tfuniform/seed_2026", "TF_RE_tfuniform_seed2026.tsv", -1.0),
    "shuf2025": ("prior_shuffle/seed_2025", "TF_RE_shuffled_seed2025.tsv", None),
    "shuf2026": ("prior_shuffle/seed_2026", "TF_RE_shuffled_seed2026.tsv", None),
    "shuflow2025": ("prior_shuffle/seed_2025", "TF_RE_shuffled_seed2025.tsv", -1.0),
}


def dense_dir(name: str) -> Path:
    return paths.FLY / "controls" / name / "network_dense"


def export(name: str) -> dict:
    from drosophila_gnn.analysis.network import export_lineage_networks

    folder, prior, threshold = CONTROLS[name]
    ckpt = paths.data("fly", "checkpoints", *folder.split("/"), "model.pt")
    over = {"reproduce": {"checkpoint_path": str(ckpt)}}
    if prior:
        over["data"] = {"tf_re_prior": str(paths.data("fly", "checkpoints", *folder.split("/"), prior))}
    cfg = config.load("germ_network", over)
    cfg["network"]["output_dir"] = str(paths.FLY / "controls" / name / "network")
    cfg["network"]["dense"]["output_dir"] = str(dense_dir(name))
    if threshold is not None:
        cfg["network"]["dense"]["min_score_keep"] = threshold
    return export_lineage_networks(config.write(cfg, f"germ_network_{name}"))


def run() -> None:
    for name in CONTROLS:
        print(f"   {name}", flush=True)
        export(name)
