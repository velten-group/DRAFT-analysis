"""Fig. 3 panel tables (mouse reprogramming, unpaired DRAFT) and Supp. Figs. 5-6, from the mouse stages.

a  cells per cluster and sample of origin (cluster expression tables)
b  balanced AUROC and official AUPR per cell cluster of DRAFT and four other methods (mouse/benchmark.py); mean and
   SD over the seven clusters
c  network distances between clusters (raw, and divided by each method's maximum) and lineage continuity
d  edge clusters x cell clusters (frozen numbering, recomputed and checked) and the six candidate TFs
e  topic weights per cluster (frozen) and the six GO callouts
S5 top 12 TFs (out-degree) of each edge cluster; S6 top 10 GO terms of each topic
"""
from __future__ import annotations

import pandas as pd

from ..mouse import CLUSTERS, benchmark, modules, out
from . import save

K = "main"
GOLDS = ["ChIP", "Perturb", "Perturb+ChIP"]
LOCAL, RECEIVED = ["DRAFT", "scMTNI+Prior", "Random"], ["Dictys", "SCENIC"]  # row blocks of the published AUROC table
DISTANCE_METHODS = [("DRAFT", "DRAFT"), ("Dictys", "Dictys"), ("scMTNI+Prior", "scMTNI+Prior"), ("SCENIC", "SCENIC")]


def panel_a() -> None:
    save(K, "fig3_Fig_3a_0.tsv", modules.composition())


def panel_b() -> None:
    auc = pd.read_csv(out("benchmark", "auroc_metrics_by_cluster.tsv"), sep="\t", float_precision="round_trip")
    apr = pd.read_csv(out("benchmark", "aupr", "metrics_by_cluster.tsv"), sep="\t", float_precision="round_trip")
    key = {f"C{c}": i for i, c in enumerate(CLUSTERS)}
    auc = auc.assign(_b=auc.model.isin(RECEIVED).astype(int), _c=auc.cluster.map(key), _g=auc.gold_standard.map(
        GOLDS.index), _m=auc.model.map((LOCAL + RECEIVED).index)).sort_values(["_b", "_c", "_g", "_m"])
    auc = auc[["gold_standard", "cluster", "model", "AUROC", "n_positive", "n_negative"]].reset_index(drop=True)
    apr = apr.assign(_g=apr.gold_standard.map(GOLDS.index), _m=apr.model.map(benchmark.METHODS.index)).sort_values(
        ["_g", "_m", "cluster"])[["gold_standard", "cluster", "model", "AUPR", "n_positive", "n_pairs"]]
    apr = apr.reset_index(drop=True)
    auc_s = auc.groupby(["gold_standard", "model"]).AUROC.agg(["mean", lambda s: s.std(ddof=1)]).reset_index()
    auc_s.columns = ["gold_standard", "model", "AUROC_mean", "AUROC_SD"]
    apr_s = apr.groupby(["gold_standard", "model"]).AUPR.agg(["mean", lambda s: s.std(ddof=1), "size"]).reset_index()
    apr_s.columns = ["gold_standard", "model", "AUPR_mean", "AUPR_SD", "n_clusters"]
    for i, df in enumerate([auc_s.merge(apr_s, on=["gold_standard", "model"]), auc, apr]):
        save(K, f"fig3_Fig_3b_{i}.tsv", df)


def panel_c() -> None:
    raw = modules.distances()
    cont = pd.DataFrame([{"model": m, **modules.continuity(raw[m])} for m, _ in DISTANCE_METHODS])
    blocks = [cont]
    for m, _ in DISTANCE_METHODS:
        d = raw[m]
        blocks.append((d / d.to_numpy().max()).rename_axis("cluster").reset_index())
        blocks.append(d.rename_axis("cluster").reset_index())
    for i, df in enumerate(blocks):
        save(K, f"fig3_Fig_3c_{i}.tsv", df)


def panel_d() -> None:
    counts = pd.read_csv(modules.frozen("edge", "edgeCluster_x_cellCluster_counts_from_network_all.csv"), index_col=0)
    counts.index = [f"EC{x}" for x in counts.index]
    counts = counts[["C4", "C8", "C1", "C5", "C7", "C3", "C2"]]
    props = counts.div(counts.sum(axis=1), axis=0)
    totals = counts.sum(axis=1).rename("edge_occurrences_across_cell_clusters")
    for i, df in enumerate([props.rename_axis("edge_cluster").reset_index(),
                            counts.rename_axis("edge_cluster").reset_index(),
                            totals.rename_axis("edge_cluster").reset_index(), modules.selected_tfs()]):
        save(K, f"fig3_Fig_3d_{i}.tsv", df)


def panel_e() -> None:
    topics = pd.read_csv(modules.frozen("topic10", "topic_proportions.tsv"), sep="\t", index_col=0)
    save(K, "fig3_Fig_3e_0.tsv", topics[["C4", "C8", "C1", "C5", "C7", "C3", "C2"]].rename_axis("topic").reset_index())
    save(K, "fig3_Fig_3e_1.tsv", modules.selected_go())


def supplementary() -> None:
    namespace, helpers = modules.module_statistics()
    tf = modules.tf_top12(namespace)
    save("supplementary", "supp05_Supp_Fig_5_0.tsv",
         tf[["edge_cluster", "tf_idx", "tf_name", "degree", "mean_score", "panel_title"]])
    save("supplementary", "supp06_Supp_Fig_6_0.tsv", modules.go_top10(helpers))


def build() -> None:
    modules.numbering_check()
    panel_a()
    panel_b()
    panel_c()
    panel_d()
    panel_e()
    supplementary()
