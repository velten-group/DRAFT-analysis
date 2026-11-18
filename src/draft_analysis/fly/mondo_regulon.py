"""Fig. 5c: early germline Mondo eRegulon (GSC to Sp-4).

1. Single-TF Top lists of Mondo at GSC, Gb, Sp-2 and Sp-4 from the seed-2025 checkpoint, targets expressed in 10-75% of
   the stage's cells (Top 50 for G0-G2, Top 25 for G3).
2. scripts/aggregate_single_tf_regulon.R lists Mondo's edge-cluster candidates with their mean cosine. Fig. 5c uses its own
   edge clusters (Mondo in EC11; DATA/fly/external/fig5c_edge_cluster), not the frozen numbering of Fig. 4
   (Mondo in EC9).
3. Consensus (Python port of the R rule): reciprocal-rank fusion (k = 60) with absent
   stages contributing 0, CG-numbered and non-coding RNA genes removed, top 20; the two REs of each target ranked by
   stages selected, occurrences and distance to the TSS.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from .. import paths
from ..panels import save
from . import regulon

TIMES = [0, 1, 2, 3]
STAGES = ["GSC", "Gb", "Sp-2", "Sp-4"]
K = 60
CG = re.compile(r"CG[0-9]+.*", re.IGNORECASE)
RNA = re.compile(r"CR\d+|mir-.*|.*:CR\d+.*|[^:]*RNA[^:]*:.*|mt:(lr|sr)RNA|mt:tRNA:.*")
HIGHLIGHT = ["kuz", "mamo"]
ROOT = paths.FLY / "fig5c_root"


def link_dir(target: Path, link: Path) -> None:
    """Directory link: a symlink, or a junction on Windows when symlinks are not permitted."""
    if link.exists():
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(target, link, target_is_directory=True)
    except OSError:
        import _winapi

        _winapi.CreateJunction(str(target), str(link))


def stage_lists_dir() -> Path:
    return ROOT / "germ" / "single_tf"


def run_stage_lists() -> None:
    for t in TIMES:
        out = regulon.single_tf(t, tfs=("mondo",), max_expr=0.75,
                                out=paths.FLY / "germ" / "single_tf_max0.75" / "runs" / f"G{t}")
        dest = stage_lists_dir()
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out / f"regulon_TG_RE_mondo_Germline_{t}.csv", dest / f"regulon_TG_RE_mondo_Germline_{t}.csv")


def run_r_candidates() -> Path:
    ec = ROOT / "germ" / "edge_cluster"
    ec.mkdir(parents=True, exist_ok=True)
    for name in ("edge_clusters_12k_topk.csv", "edges_selected_topk.csv"):
        shutil.copyfile(paths.data("fly", "external", "fig5c_edge_cluster", name), ec / name)
    link_dir(paths.FLY / "germ" / "network", ROOT / "germ" / "network")
    link_dir(paths.FLY / "germ" / "reproduce", ROOT / "germ" / "reproduce")
    subprocess.run([regulon.rscript(), str(regulon.R_SCRIPT), "--result-root", str(ROOT), "--tf", "mondo",
                    "--times", "0,1,2,3", "--topn", "20", "--highlight-targets", ",".join(HIGHLIGHT)], check=True)
    return stage_lists_dir() / "mondo_G0-G3_consensus_all_candidates.tsv"


def _split(values):
    parts = pd.Series(values).str.extract(r"^(.*) \((FBgn[^)]*)\)$")
    return parts[0].tolist(), parts[1].tolist()


def _stage_lists(folder: Path):
    lists, frames = {}, {}
    for t in TIMES:
        frame = pd.read_csv(folder / f"regulon_TG_RE_mondo_Germline_{t}.csv")
        frame["TG_name"], frame["TG_id"] = _split(frame.TG)
        order = list(dict.fromkeys(frame.TG_id))
        lists[t] = pd.Series(np.arange(1, len(order) + 1), index=order, dtype=float)
        frames[t] = frame.assign(stage_label=STAGES[t], time_idx=t)
    return lists, frames


def _aggregate(lists, members, cos):
    ids = sorted({g for s in lists.values() for g in s.index if g in members})
    ranks = pd.DataFrame({t: lists[t].reindex(ids) for t in TIMES})
    lengths = {t: len(lists[t]) for t in TIMES}
    norm = pd.DataFrame({t: (ranks[t] - 0.5) / max(1, lengths[t]) for t in TIMES})
    table = pd.DataFrame({"TG_id": ids, "TG": [members[g] for g in ids]}).set_index("TG_id")
    table["rrf_score"] = (1.0 / (K + ranks)).fillna(0).mean(axis=1)  # absent stages contribute 0
    table["geometric_mean_normalized_rank"] = np.exp(np.log(norm.fillna(1.0)).mean(axis=1))
    table["mean_cos_sim"] = cos.reindex(ids)
    table["stages_listed"] = ranks.notna().sum(axis=1)
    for t in TIMES:
        table[f"rank_{STAGES[t]}"] = ranks[t]
    table = table.reset_index().sort_values(["rrf_score", "geometric_mean_normalized_rank", "mean_cos_sim", "TG"],
                                            ascending=[False, True, False, True], kind="mergesort")
    table = table.reset_index(drop=True)
    drop = table.TG.map(lambda s: bool(CG.fullmatch(s)) or bool(RNA.fullmatch(s)))
    kept = table.loc[~drop].reset_index(drop=True)
    kept.insert(0, "consensus_rank", np.arange(1, len(kept) + 1))
    return kept


def _re_links(top, frames) -> pd.DataFrame:
    rows = pd.concat(frames.values(), ignore_index=True)
    out = []
    for target in top.itertuples(index=False):
        group = rows.loc[rows.TG_id.eq(target.TG_id)]
        summary = group.groupby("RE").agg(
            selected_stages=("stage_label", lambda s: ";".join(dict.fromkeys(s))),
            selected_stage_count=("time_idx", "nunique"), occurrence_count=("RE", "size"),
            mean_abs_distance_to_TSS_bp=("dist_REcenter_to_TSS_bp", lambda d: d.abs().mean())).reset_index()
        summary["distance_order"] = summary.mean_abs_distance_to_TSS_bp.fillna(np.inf)
        summary = summary.sort_values(["selected_stage_count", "occurrence_count", "distance_order", "RE"],
                                      ascending=[False, False, True, True], kind="mergesort").head(2)
        for rank, row in enumerate(summary.itertuples(index=False), 1):
            out.append({"consensus_rank": target.consensus_rank, "TG": target.TG, "TG_id": target.TG_id, "RE": row.RE,
                        "RE_rank": rank, "selected_stages": row.selected_stages,
                        "selected_stage_count": row.selected_stage_count,
                        "mean_abs_distance_to_TSS_bp": row.mean_abs_distance_to_TSS_bp})
    return pd.DataFrame(out)


def consensus() -> tuple[pd.DataFrame, pd.DataFrame]:
    lists, frames = _stage_lists(stage_lists_dir())
    r_all = pd.read_csv(stage_lists_dir() / "mondo_G0-G3_consensus_all_candidates.tsv", sep="\t")
    table = _aggregate(lists, dict(zip(r_all.TG_id, r_all.TG)), r_all.set_index("TG_id").mean_cos_sim)
    top = table.head(20)
    return top, _re_links(top, frames)


def panel_c() -> None:
    top, links = consensus()
    keep = [c for c in top.columns if c in ("consensus_rank", "TG", "TG_id", "rrf_score", "geometric_mean_normalized_rank",
                                            "mean_cos_sim", "stages_listed") or c.startswith("rank_")]
    save("main", "fig5_Fig_5c_0.tsv", top[keep].assign(highlighted=top.TG.isin(HIGHLIGHT)))
    save("main", "fig5_Fig_5c_1.tsv", links)


def run() -> None:
    run_stage_lists()
    run_r_candidates()
    panel_c()
