"""Fig. 4 panel tables from the fly pipeline results.

a  cell numbers per annotated cell type (RNA h5ad)
b  Hi-C AUROC per stage on the frozen evaluation pairs: DRAFT rescored from the checkpoint and SCENIC+ (per-stage R2G)
   from its stage fits (hic stage); scDoRI and Random from the frozen within-prior inputs (see docs/pipeline.md)
c  Frobenius distance between stage networks (reproduce step) and smoothness along each path
d  ovo eRegulon (single-TF step + consensus R script)
e  edge clusters x stages (frozen numbering); f  topics x stages (frozen numbering); g  curated summary + topic GO terms
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from .. import config, paths
from ..fly import numbering
from . import save

K = "main"
SOMA = ["CySC", "eSCC-1", "eSCC-2", "eSCC-3", "lSCC-1", "lSCC-2", "lSCC-3", "SdA", "SdB"]
TE = re.compile(r"transposable[_\s-]*element|retrotranspos|transposon|^RR\d+_transposable_element$", re.IGNORECASE)
CELL_GROUPS = {  # Fig. 4a bars: annotated cell types -> obs cell_type groups of the RNA h5ad
    "GSC": ["Germline_0"], "Gb": ["Germline_1"], "Sp-2": ["Germline_2"], "Sp-4": ["Germline_3"], "Sp-8": ["Germline_4"],
    "Sp-16": ["Germline_5"], "1° Sc": [f"Germline_{i}" for i in (6, 7, 8)],
    "2° Sc": [f"Germline_{i}" for i in range(9, 15)], "Hub": ["Hub_0"], "CySC": ["Soma_0"],
    "eSCC": ["Soma_1", "Soma_2", "Soma_3"], "lSCC": ["Soma_4", "Soma_5", "Soma_6"], "SdA": ["Soma_7a"],
    "SdB": ["Soma_7b"]}
HIC = {"spg": ("SpG", range(2, 6)), "spc": ("SpC", range(6, 15))}
HIC_METHODS = {"DRAFT": "DRAFT", "SCENIC+ R2G per stage": "SCENIC+", "scDoRI": "scDoRI", "Random": "Random"}


def annotations() -> dict:
    return json.loads((config.CONFIGS / "fly" / "figure4g_annotations.json").read_text(encoding="utf-8"))


def panel_a() -> None:
    import anndata as ad

    obs = ad.read_h5ad(paths.data("fly", "raw", "A_rna_focus.h5ad"), backed="r").obs["cell_type"].astype(str)
    counts = obs.value_counts()
    lineage = ["Germline"] * 8 + ["Hub"] + ["Somatic"] * 5
    save(K, "fig4_Fig_4a_0.tsv", pd.DataFrame({"cell_type": list(CELL_GROUPS), "lineage": lineage,
                                               "n_cells": [int(counts[g].sum()) for g in CELL_GROUPS.values()]}))


def hic_auroc() -> pd.DataFrame:
    """AUROC per Hi-C assay, stage and method on the held-out (test) half of the frozen balanced pairs.

    DRAFT and SCENIC+ (per-stage R2G) scores come from the hic stage (fly/hic.py: checkpoint and stage fits);
    scDoRI coefficients and Random edges are deposited inputs."""
    from sklearn.metrics import roc_auc_score

    from ..fly import hic

    z = np.load(hic.cache_dir() / "scores.npz")
    own = np.load(paths.result("fly", "germ", "hic_within_prior", "scores.npz"))
    if not np.array_equal(own["pair_codes"], z["pair_codes"]):
        raise ValueError("hic stage scored a different candidate set")
    ctx = np.load(hic.cache_dir() / "random_context" / "context.npz")
    ev = hic.evaluation_dir()
    rnd = np.load(ev / "random_edges_by_stage.npz")
    labels = numbering.germline_labels()
    rows = []
    for ds, (_, times) in HIC.items():
        pairs = pd.read_csv(ev / f"pairs_{ds}_s2025.tsv.gz", sep="\t")
        test = pairs[pairs.fold.eq("test")]
        pos, y = test.candidate_idx.to_numpy(), test.label.to_numpy()
        for t in times:
            scores = {"DRAFT": own["draft"][t, pos], "SCENIC+ R2G per stage": own["scenicplus_per_stage"][t, pos],
                      "scDoRI": z["scdori"][t, pos],
                      "Random": np.isin(ctx["candidate_codes"][pos], rnd[f"t{t:02d}"]).astype(float)}
            for model, s in scores.items():
                rows.append({"dataset": ds, "time_idx": t, "stage_label": labels[t], "model": model,
                             "AUROC": float(roc_auc_score(y, s))})
    return pd.DataFrame(rows)


def panel_b() -> None:
    m = hic_auroc()
    b = m.assign(method=m.model.map(HIC_METHODS), assay=m.dataset.map({k: v[0] for k, v in HIC.items()}))
    b = b.assign(order_a=b.assay.map({"SpG": 0, "SpC": 1}), order_m=b.method.map({n: i for i, n in
                                                                                    enumerate(HIC_METHODS.values())}))
    b = b.sort_values(["order_a", "order_m", "time_idx"])[["assay", "time_idx", "stage_label", "method", "AUROC"]]
    save(K, "fig4_Fig_4b_0.tsv", b.reset_index(drop=True))


def smoothness_along(d: np.ndarray, order: list[int]) -> float:
    """1 - mean adjacent distance / maximum pairwise distance, within one lineage path."""
    sub = d[np.ix_(order, order)]
    return float(1 - np.diag(sub, 1).mean() / sub.max())


def panel_c() -> None:
    germ = numbering.germline_labels()
    blocks, smooth = [], []
    for lineage, labels, title in (("germ", germ, "Germline"), ("soma", SOMA, "Soma")):
        d = np.loadtxt(paths.FLY / lineage / "reproduce" / "tf_tg_prob_frobenius_dist_matrix.tsv",
                       delimiter="\t")
        if title == "Germline":
            smooth.append({"lineage": "Germline", "path": "GSC to 2° Sc_6", "smoothness": smoothness_along(d, list(range(len(d))))})
        else:
            smooth.append({"lineage": "Soma", "path": "CySC to lSCC-3, then SdA", "smoothness": smoothness_along(d, list(range(8)))})
            smooth.append({"lineage": "Soma", "path": "CySC to lSCC-3, then SdB",
                           "smoothness": smoothness_along(d, [*range(7), 8])})
        blocks.append(pd.DataFrame(d / d.max(), index=labels, columns=labels).reset_index().rename(columns={"index": "stage"}))
        blocks.append(pd.DataFrame(d, index=labels, columns=labels).reset_index().rename(columns={"index": "stage"}))
    for i, df in enumerate([pd.DataFrame(smooth)] + blocks):
        save(K, f"fig4_Fig_4c_{i}.tsv", df)


def panel_d() -> None:
    st = paths.FLY / "germ" / "single_tf"
    top = pd.read_csv(st / "ovo_G0-G0_consensus_top20.tsv", sep="\t")
    reg = pd.read_csv(st / "regulon_TG_RE_ovo_G0-G0_consensus.csv")
    hl = set(annotations()["highlighted_ovo_targets"])
    d_top = top[["consensus_rank", "TG", "TG_id", "mean_cos_sim", "edge_cluster"]].assign(highlighted=top.TG.isin(hl))
    d_re = reg[["consensus_rank", "TG", "RE", "rank_re", "mean_abs_distance_to_TSS_bp"]]
    save(K, "fig4_Fig_4d_0.tsv", d_top)
    save(K, "fig4_Fig_4d_1.tsv", d_re[d_re.rank_re.le(2)].reset_index(drop=True))


def ec_tf_ranks() -> pd.DataFrame:
    """Rank of each TF within each frozen edge cluster by its number of distinct targets, among the 243 listed TFs."""
    clusters = pd.read_csv(numbering.frozen("edge_cluster", "edge_clusters_12k_topk.csv"))
    keep = pd.read_csv(numbering.frozen("tf_list", "TF_tsv_FlyBase_plus_exceptions.tsv"), sep="\t", dtype=str)
    included = set(keep.loc[keep.included.eq("True"), "gene_name"])
    no_te = clusters[~clusters.tf_name.astype(str).str.contains(TE)]
    deg = no_te.groupby(["edge_cluster", "tf_idx", "tf_name"], sort=False).tg_idx.nunique().reset_index(name="degree")
    deg = deg.sort_values(["edge_cluster", "degree"], ascending=[True, False], kind="mergesort")
    filt = deg[deg.tf_name.isin(included)].copy()
    filt["rank_tf243"] = filt.groupby("edge_cluster").cumcount() + 1
    return filt


def panels_efg() -> None:
    ann = annotations()
    germ = numbering.germline_labels()
    ec_node: dict[str, list[str]] = {}
    for node, spec in ann["nodes"].items():
        for ec in spec["edge_clusters"]:
            ec_node.setdefault(f"EC{ec}", []).append(node)
    cnt = pd.read_csv(numbering.frozen("edge_cluster", "edgeCluster_x_cellCluster_counts_topk.csv"), index_col=0)
    cnt = cnt.to_numpy(float)
    labels = [f"EC{k}" for k in range(len(cnt))]
    prop = pd.DataFrame(cnt / cnt.sum(1, keepdims=True), index=labels, columns=germ)
    save(K, "fig4_Fig_4e_0.tsv", prop.reset_index().rename(columns={"index": "edge_cluster"}))
    save(K, "fig4_Fig_4e_1.tsv", pd.DataFrame(cnt, index=labels, columns=germ).reset_index()
         .rename(columns={"index": "edge_cluster"}))
    save(K, "fig4_Fig_4e_2.tsv", pd.DataFrame({"edge_cluster": labels, "edges_counted_at_every_stage (bars)": cnt.sum(1),
                                               "used_in_g_for_node": [", ".join(ec_node.get(e, [])) for e in labels]}))
    tp = pd.read_csv(numbering.frozen("topic", "topic_x_stage_prop.tsv"), sep="\t", index_col=0)
    save(K, "fig4_Fig_4f_0.tsv", tp.rename_axis("topic").reset_index())
    save(K, "fig4_Fig_4f_1.tsv", pd.DataFrame({"topic": tp.index, "used_in_g_for_node": [
        ", ".join(ann["topic_node"].get(t, [])) for t in tp.index]}))

    # g: every badge TF must rank <= 10 among the 243 TFs in one of its node's edge clusters
    ranks = ec_tf_ranks()
    for node, spec in ann["nodes"].items():
        for tf in spec["TFs"]:
            ok = ranks[(ranks.tf_name == tf) & ranks.edge_cluster.isin(spec["edge_clusters"]) & (ranks.rank_tf243 <= 10)]
            if ok.empty:
                raise AssertionError(f"Fig. 4g badge {tf} at {node} is not in the top 10 of its edge clusters")
    nodes = pd.DataFrame([{"node": n, "edge_clusters": ", ".join(f"EC{e}" for e in s["edge_clusters"]),
                           "TFs": ", ".join(s["TFs"])} for n, s in ann["nodes"].items()])
    badges = pd.read_csv(config.CONFIGS / "fly" / "figure4g_badge_sources.tsv", sep="\t")
    boxes = pd.DataFrame([{"box": b["box"], "topics": b["topics"], "listed_terms": "; ".join(b["listed_terms"]),
                           "node": b["node"]} for b in ann["boxes"]])
    go = []
    for t in sorted({x.strip() for b in ann["boxes"] for x in b["topics"].split(",")}, key=lambda s: int(s[1:])):
        e = pd.read_csv(paths.data("fly", "external", "enrichr_topic_GO",
                                   f"topic_{t}_enrich_GO_Biological_Process_2018_fly.csv"))
        e = e[e["Adjusted P-value"].lt(0.05)][["Term", "Overlap", "P-value", "Adjusted P-value", "Genes"]]
        go.append(e.assign(topic=t))
    go_tab = pd.concat(go)[["topic", "Term", "Overlap", "P-value", "Adjusted P-value", "Genes"]]
    for i, df in enumerate([nodes, badges, boxes, go_tab]):
        save(K, f"fig4_Fig_4g_{i}.tsv", df)


def build() -> None:
    panel_a()
    panel_b()
    panel_c()
    panel_d()
    panels_efg()
