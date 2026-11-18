"""Supplementary panel tables of the checkpoint line: Supp. Figs. 8-9 here; 7 (supp_baselines), 10-11 (supp_peaks),
12-14 (supp_ccc), 5-6 (fig3), 1-4 (sim).

8   top 10 TFs (243-TF list) per frozen edge cluster by distinct targets, mean edge score; cluster stage composition
9   topic GO terms (Enrichr results) and topic weights per stage (frozen numbering)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import paths
from ..fly import numbering
from . import fig4, save

K = "supplementary"
# stage names of the original notebooks (spermatocyte sub-stages collapsed), used in the Supp. Fig. 8 panel titles
NOTEBOOK_STAGES = ["GSC", "Gb", "Sp-2", "Sp-4", "Sp-8", "Sp-16"] + ["1° Sc"] * 3 + ["2° Sc"] * 6


def ec_specificity(prop: np.ndarray, counts: np.ndarray, labels: list[str], min_edges=50, min_max_prop=0.45,
                   min_gap=0.15, topk=3) -> pd.DataFrame:
    n_edges = counts.sum(axis=1)
    top_idx = np.argsort(-prop, axis=1)[:, :topk]
    top_props = np.take_along_axis(prop, top_idx, axis=1)
    max_prop, second = top_props[:, 0], top_props[:, 1]
    gap = max_prop - second
    mix = [", ".join(f"{labels[top_idx[e, m]]}:{top_props[e, m]:.2f}" for m in range(topk)) for e in range(len(prop))]
    spec = (n_edges >= min_edges) & (max_prop >= min_max_prop) & (gap >= min_gap)
    return pd.DataFrame({"edge_cluster": np.arange(len(prop)),
                         "specific_to": [labels[top_idx[e, 0]] if spec[e] else "mixed" for e in range(len(prop))],
                         "max_prop": max_prop, "second_prop": second, "gap": gap, "n_edges": n_edges, "top_mix": mix})


def supp8() -> None:
    edges = pd.read_csv(numbering.frozen("edge_cluster", "edges_selected_topk.csv"))
    clusters = pd.read_csv(numbering.frozen("edge_cluster", "edge_clusters_12k_topk.csv"))
    counts = pd.read_csv(numbering.frozen("edge_cluster", "edgeCluster_x_cellCluster_counts_topk.csv"), index_col=0)
    counts = counts.to_numpy(float)
    prop = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
    is_te = lambda s: s.astype(str).str.contains(fig4.TE)
    with_ec = edges[["tf_idx", "tg_idx", "tf_name", "tg_name", "score"]].merge(
        clusters[["tf_idx", "tg_idx", "edge_cluster"]], on=["tf_idx", "tg_idx"], how="left")
    with_ec, cl = with_ec[~is_te(with_ec.tf_name)], clusters[~is_te(clusters.tf_name)]
    keys = ["edge_cluster", "tf_idx", "tf_name"]
    stats = cl.groupby(keys).tg_idx.nunique().reset_index(name="degree").merge(
        with_ec.groupby(keys).score.mean().reset_index(name="mean_score"), on=keys, how="left")
    keep = pd.read_csv(numbering.frozen("tf_list", "TF_tsv_FlyBase_plus_exceptions.tsv"), sep="\t", dtype=str)
    keep = keep.included.astype(str).eq("True").to_numpy()
    stats = stats[keep[stats.tf_idx]]
    top = stats.sort_values(["edge_cluster", "degree"], ascending=[True, False]).groupby("edge_cluster").head(10)
    save(K, "supp08_Supp_Fig_8_0.tsv", top[["edge_cluster", "tf_idx", "tf_name", "degree", "mean_score"]])
    save(K, "supp08_Supp_Fig_8_1.tsv", ec_specificity(prop, counts, NOTEBOOK_STAGES))


def supp9() -> None:
    rows = []
    for t in range(12):
        e = pd.read_csv(paths.data("fly", "external", "enrichr_topic_GO",
                                   f"topic_T{t}_enrich_GO_Biological_Process_2018_fly.csv"))
        e["score"] = -np.log10(np.maximum(pd.to_numeric(e["Adjusted P-value"], errors="coerce"), 1e-300))
        e = e.dropna(subset=["score"]).sort_values("score", ascending=False).head(10)
        rows.append(e.assign(topic=f"T{t}", rank=range(1, len(e) + 1))[
            ["topic", "rank", "Term", "Overlap", "P-value", "Adjusted P-value", "score", "Genes"]])
    tp = pd.read_csv(numbering.frozen("topic", "topic_x_stage_prop.tsv"), sep="\t", index_col=0)
    save(K, "supp09_Supp_Fig_9_0.tsv", pd.concat(rows))
    save(K, "supp09_Supp_Fig_9_1.tsv", tp.rename_axis("topic").reset_index())


def build() -> None:
    from . import supp_baselines, supp_ccc, supp_peaks

    supp8()
    supp9()
    supp_baselines.build()
    supp_peaks.build()
    supp_ccc.build()
