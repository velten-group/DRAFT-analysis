"""Cell-cell communication timing (Fig. 6, Supp. Figs. 12-14): checkpoint -> receptor-side scores -> LR timing calls.

1. scripts/export_ccc_full_scores.py: for every integer stage of the germline (15) and soma (9) model, the cosine of every
   active TF-TG pair and its percentile; per receptor and stage the mean percentile over the active TFs (readout A).
2. scripts/run_ccc_unified.py with configs/fly/ccc_unified.yaml: LR timing rule, receptor gate 5%, ligand gate 5%,
   soma axis aligned to the germline by the frozen DTW path (SdA/SdB in parallel); AUROC for literature pairs, strips.
   The expression-based baselines (CellChat, LIANA+) are read from their official runs (deposited outputs).
3. scripts/list_ccc_peak_shifts.py: early/late calls of every LR pair in the five germline models (seeds 2024-2028),
   used to select the pairs of Fig. 6d; the full scores of every seed are exported from its checkpoint.
4. Supplementary Figs. 12-14: scripts/ccc_seed_stability.py (call stability across the five models, random controls),
   the same calls under the original sequential alignment (SdA then SdB; the published Supp. Figs. 12 and 13 were
   drawn from it), and scripts/benchmark_ccc_alignment.py (literature AUROC under each alignment).

The scripts receive absolute paths through the rewritten configuration.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

from .. import config, paths

SCRIPTS = paths.REPO / "scripts"
SEEDS = (2024, 2025, 2026, 2027, 2028)
RUN_NAME = "receptor_input_ungated_gate0.05_lig0.05"


def _run(script: str, *args: str) -> None:
    subprocess.run([sys.executable, str(SCRIPTS / script), *args], check=True, cwd=paths.REPO)


def full_scores_dir(lineage: str, seed: int = 2025) -> Path:
    return paths.FLY / "ccc_full_scores" / (lineage if seed == 2025 else f"germ_seed{seed}")


def full_scores(lineage: str, seed: int = 2025) -> Path:
    """Receptor-side full-score export of one model (needs the network export of the same model for its check)."""
    out = full_scores_dir(lineage, seed)
    net_cfg = paths.FLY / "configs" / (f"{lineage}_network" + ("" if seed == 2025 else f"_seed{seed}") + ".yaml")
    args = ["--lineage", lineage, "--config", str(net_cfg), "--checkpoint", str(config.checkpoint(lineage, seed)),
            "--output-dir", str(out)]
    if lineage == "soma":
        args += ["--artifact", str(paths.data("fly", "checkpoints", "soma_graphs.pt"))]
    _run("export_ccc_full_scores.py", *args)
    return out


def unified_config() -> Path:
    cfg = config.load("ccc_unified")
    cfg["output_dir"] = str(paths.FLY / "ccc_unified")
    cfg["full_scores"] = {"germline": str(full_scores_dir("germ")), "soma": str(full_scores_dir("soma"))}
    return config.write(cfg, "ccc_unified")


def unified() -> Path:
    """LR timing calls, strips (Fig. 6d) and the AUROC of the earliness score for literature pairs (Fig. 6c)."""
    _run("run_ccc_unified.py", "--config", str(unified_config()))
    run = paths.FLY / "ccc_unified" / RUN_NAME
    _run("ccc_effect_sizes.py", "--run-dir", str(run))
    return run


def seed_full_scores_dir(seed: int) -> Path:
    out = full_scores_dir("germ", seed)
    if not (out / "receptor_input_ungated.tsv").exists():
        raise FileNotFoundError(f"{out}: run `draft-analysis checkpoint --stages ccc seeds` first")
    return out


def peak_shifts(cfg: Path | None = None) -> Path:
    """Early/late calls in the five germline models (Fig. 6d selection, Supp. Figs. 12, 13)."""
    cfg = cfg or unified_config()
    args = ["--config", str(cfg)]
    for s in SEEDS:
        args += ["--germ-seed", str(s), str(seed_full_scores_dir(s))]
    _run("list_ccc_peak_shifts.py", *args)
    return Path(yaml.safe_load(Path(cfg).read_text(encoding="utf-8"))["output_dir"]) / PEAK_SHIFTS


PEAK_SHIFTS = "peak_shifts_gate0.05_lig0.05"
SEQUENTIAL = "ccc_unified_sequential"


def sequential_config() -> Path:
    """The sequential configuration: SdA and SdB are successive soma time points on the original DTW path (used for
    Supp. Figs. 12 and 13)."""
    cfg = yaml.safe_load(unified_config().read_text(encoding="utf-8"))
    cfg.pop("soma_parallel_branches")
    cfg.pop("soma_branch_label", None)
    cfg["alignment"] = str(config.CONFIGS / "fly" / "ccc_frozen_dtw_soma_to_germ.tsv")
    cfg["output_dir"] = str(paths.FLY / SEQUENTIAL)
    return config.write(cfg, SEQUENTIAL)


def seed_stability(cfg: Path) -> Path:
    """Agreement of the calls between the five germline models, with uniform and permutation controls (seed 2025)."""
    _run("ccc_seed_stability.py", "--config", str(cfg))
    return Path(yaml.safe_load(cfg.read_text(encoding="utf-8"))["output_dir"]) / PEAK_SHIFTS


def supplementary() -> None:
    """Supp. Figs. 12-14: stability and agreement under the current and the sequential alignment; alignment benchmark."""
    seed_stability(unified_config())
    seq = sequential_config()
    peak_shifts(seq)
    seed_stability(seq)
    _run("benchmark_ccc_alignment.py", "--config", str(unified_config()))
