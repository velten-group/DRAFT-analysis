"""Single-TF target lists from the germline checkpoint and their consensus eRegulons (Fig. 4d, ovo).

1. drosophila_gnn single-TF step (configs/fly/germ_single_tf.yaml) with the settings of the manuscript run: TFs mondo and
   ovo, stage-specific Top lists of the targets expressed in 10-70% of the stage's cells (Top 50 for G0-G2, Top 25 for G3),
   two REs per target.
2. scripts/aggregate_single_tf_regulon.R (base R) combines the stage lists of one TF within its edge cluster (frozen
   numbering) by reciprocal-rank fusion and drops CG-numbered genes.

The R script reads <root>/germ/{edge_cluster,network,reproduce,single_tf}; <root> is results/checkpoint/fly, where
germ/edge_cluster holds the frozen edge clusters (numbering.install_frozen).
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .. import config, paths

# number of top targets per germline stage, as in the manuscript
STAGE_TOPN = {0: 50, 1: 50, 2: 50, 3: 25}
R_SCRIPT = paths.REPO / "scripts" / "aggregate_single_tf_regulon.R"


def single_tf(stage: int, tfs=("mondo", "ovo"), min_expr: float = 0.1, max_expr: float = 0.7,
              out: Path | None = None) -> Path:
    """Top targets and their REs for each TF at one germline stage; writes regulon_TG_RE_<tf>_Germline_<stage>.csv."""
    from drosophila_gnn.analysis.single_tf import run_single_tf_analysis

    out = out or paths.FLY / "germ" / "single_tf" / "runs" / f"G{stage}"
    cfg = config.load("germ_single_tf", {"reproduce": {"checkpoint_path": str(config.checkpoint("germ"))}})
    cfg["single_tf"].update({"output_dir": str(out), "tf_symbols": list(tfs), "time_for_top": stage,
                             "topn_tg": STAGE_TOPN[stage], "min_expr_frac": min_expr, "max_expr_frac": max_expr,
                             "save_fig": False})
    run_single_tf_analysis(config.write(cfg, f"germ_{out.parent.parent.name}_G{stage}"))
    return out


def consensus(tf: str, times: list[int], topn: int, highlight: list[str]) -> Path:
    """Consensus regulon of `tf` over `times` with the R script; returns the folder holding its tables."""
    root = paths.FLY
    dest = root / "germ" / "single_tf"
    for t in times:  # the R script reads the stage lists from germ/single_tf
        src = dest / "runs" / f"G{t}" / f"regulon_TG_RE_{tf}_Germline_{t}.csv"
        shutil.copyfile(src, dest / src.name)
    cmd = [rscript(), str(R_SCRIPT), "--result-root", str(root), "--tf", tf, "--times", ",".join(map(str, times)),
           "--topn", str(topn), "--highlight-targets", ",".join(highlight)]
    subprocess.run(cmd, check=True)
    return dest


def rscript() -> str:
    found = shutil.which("Rscript")
    if found:
        return found
    for base in (Path("C:/Program Files/R"),):
        hits = sorted(base.glob("R-*/bin/Rscript.exe"))
        if hits:
            return str(hits[-1])
    raise FileNotFoundError("Rscript was not found; install R (base R only is needed) or put Rscript on PATH.")


def ovo() -> Path:
    """Fig. 4d: ovo at GSC (G0) within its edge cluster, top 20 named targets, stg and exu highlighted."""
    single_tf(0)
    return consensus("ovo", [0], 20, ["stg", "exu"])
