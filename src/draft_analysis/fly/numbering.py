"""Edge clusters (k-means) and topics (LDA) of the germline networks, and the frozen numbering used in the manuscript.

Cluster and topic numbers are arbitrary labels: a different network export, library version or random state renumbers
them. Every figure and table therefore reads the frozen copies in DATA/fly/frozen_numbering (deposited with the data),
whose SHA-256 values are listed in its MANIFEST.json. This module recomputes both from the networks exported here and
reports whether the recomputation reproduces the frozen files; downstream panels always use the frozen files.

Edge clusters: drosophila_gnn edge-cluster step (configs/fly/germ_edge_cluster.yaml) on the strict networks.
Topics: the procedure of the original notebooks (GNN_merge_germ-repro.ipynb cell 3 and topic_filtered.ipynb), which
differs from the packaged topic step: per stage the TFs, REs and TGs non-zero in >= 5% of the stage's cells are active,
the global top 10,000 TF-TG pairs by embedding cosine form the stage network, and LDA (12 topics, batch, 300 iterations,
random_state 42) is fitted to the edge x stage occurrence matrix.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config, paths

FROZEN_FILES = {
    "edge_cluster": ["edges_selected_topk.csv", "edge_clusters_12k_topk.csv", "edgeCluster_x_cellCluster_counts_topk.csv",
                     "edgeCluster_x_cellCluster_prop_topk.csv"],
    "topic": ["topic_x_stage_prop.tsv", "lda_W.npz", "topic_top500_edge_TGs.json"],
}


def frozen(*parts: str) -> Path:
    return paths.data("fly", "frozen_numbering", *parts)


def assert_frozen() -> None:
    """The frozen files match the SHA-256 listed in MANIFEST.json."""
    manifest = json.loads(frozen("MANIFEST.json").read_text(encoding="utf-8"))
    bad = [rel for rel, sha in manifest["sha256"].items()
           if hashlib.sha256(frozen(*rel.split("/")).read_bytes()).hexdigest() != sha]
    if bad:
        raise RuntimeError(f"frozen numbering files changed: {bad}")


def install_frozen() -> Path:
    """Copy the frozen edge-cluster tables to results/checkpoint/fly/germ/edge_cluster, where downstream steps read them."""
    import shutil

    assert_frozen()
    dest = paths.FLY / "germ" / "edge_cluster"
    dest.mkdir(parents=True, exist_ok=True)
    for name in FROZEN_FILES["edge_cluster"]:
        shutil.copyfile(frozen("edge_cluster", name), dest / name)
    return dest


def germline_labels() -> list[str]:
    return json.loads((config.CONFIGS / "fly" / "germline_stage_labels.json").read_text(encoding="utf-8"))["labels"]


# ------------------------------------------------------------------------------------------------ edge clusters
def edge_clusters() -> dict:
    """Run the edge-cluster step on results/fly/germ/network and compare its tables with the frozen ones."""
    from drosophila_gnn.analysis.edge_cluster import run_edge_cluster

    cfg = config.load("germ_edge_cluster")
    cfg["edge_cluster"]["save_fig"] = False
    cfg["edge_cluster"]["output_dir"] = str(paths.FLY / "germ" / "edge_cluster_recomputed")
    run_edge_cluster(config.write(cfg, "germ_edge_cluster"))
    out = Path(cfg["edge_cluster"]["output_dir"])
    report = {}
    for name in FROZEN_FILES["edge_cluster"]:
        new, ref = pd.read_csv(out / name), pd.read_csv(frozen("edge_cluster", name))
        # edges with tied scores can be listed in another order: compare the tables sorted by edge
        keys = [c for c in ("cell_cluster", "tf_idx", "tg_idx") if c in ref.columns]
        if keys and set(keys) <= set(new.columns):
            new, ref = (df.sort_values(keys, kind="stable").reset_index(drop=True) for df in (new, ref))
        same_shape = new.shape == ref.shape and list(new.columns) == list(ref.columns)
        report[name] = {"rows": [len(new), len(ref)], "identical": bool(same_shape and new.equals(ref))}
        if same_shape and not report[name]["identical"]:
            num = [c for c in new.select_dtypes("number").columns if c not in keys]
            other = [c for c in new.columns if c not in num]
            report[name]["same_edges_and_labels"] = bool(new[other].equals(ref[other]))
            report[name]["max_abs_diff"] = float((new[num] - ref[num]).abs().max().max()) if len(num) else None
    sel_new = pd.read_csv(out / "edges_selected_topk.csv")
    sel_ref = pd.read_csv(frozen("edge_cluster", "edges_selected_topk.csv"))
    key = ["cell_cluster", "tf_idx", "tg_idx"] if "cell_cluster" in sel_ref else list(sel_ref.columns[:3])
    report["selected_edges_same_set"] = bool(
        set(map(tuple, sel_new[key].to_numpy())) == set(map(tuple, sel_ref[key].to_numpy())))
    _save_report(report, "edge_cluster_vs_frozen.json")
    return report


# ------------------------------------------------------------------------------------------------ topics
def topic_networks(out: Path | None = None) -> Path:
    """Notebook top-10,000 TF-TG network per stage from the seed-2025 germline checkpoint (topic_filtered.ipynb input)."""
    import torch
    import torch.nn.functional as F

    from drosophila_gnn.analysis import network as nw

    out = out or paths.FLY / "germ" / "topic_notebook"
    net_dir = out / "network_top10000"
    net_dir.mkdir(parents=True, exist_ok=True)
    cfg = config.load("germ_network", {"reproduce": {"checkpoint_path": str(config.checkpoint("germ"))}})
    cfg["network"]["output_dir"] = str(out / "unused_strict")
    cfg["network"]["dense"]["output_dir"] = str(out / "unused_dense")
    context = nw._network_context(cfg)
    graphs, inf = context["graphs_sel"], context["inference"]
    out_tf, out_tg = inf["out_tf"], inf["out_tg"]
    active_rows = []
    with torch.no_grad():
        for t, g in enumerate(graphs):
            xtf, xre, xtg = nw._select_valid_cols(g["TF"].x, g["RE"].x, g["TG"].x)
            itf = torch.where((xtf != 0).float().mean(dim=1) >= 0.05)[0]
            ire = torch.where((xre != 0).float().mean(dim=1) >= 0.05)[0]
            itg = torch.where((xtg != 0).float().mean(dim=1) >= 0.05)[0]
            if itf.numel() == 0:
                itf = torch.arange(g["TF"].x.size(0))
            if itg.numel() == 0:
                itg = torch.arange(g["TG"].x.size(0))
            active_rows.append({"time_idx": t, "n_TF_active": int(itf.numel()), "n_RE_active": int(ire.numel()),
                                "n_TG_active": int(itg.numel())})
            s = F.normalize(out_tf[:, t, :][itf], dim=1) @ F.normalize(out_tg[:, t, :][itg], dim=1).T
            vals, flat = torch.topk(s.flatten(), min(10_000, s.numel()))
            tf_idx = itf[(flat // itg.numel()).long()].cpu().numpy()
            tg_idx = itg[(flat % itg.numel()).long()].cpu().numpy()
            pd.DataFrame({"time_idx": t, "TF_idx": tf_idx, "TG_idx": tg_idx, "TF_name": tf_idx.astype(str),
                          "TG_name": tg_idx.astype(str), "score_cos": vals.cpu().numpy()}).to_csv(
                net_dir / f"t{t:02d}_TF_TG_top10000.tsv", sep="\t", index=False)
    pd.DataFrame(active_rows).to_csv(net_dir / "active_counts.tsv", sep="\t", index=False)
    return net_dir


def topic_lda(net_dir: Path, out: Path) -> None:
    """topic_filtered.ipynb: edge x stage counts of the top-10,000 networks, LDA with 12 topics."""
    from scipy.sparse import coo_matrix
    from sklearn.decomposition import LatentDirichletAllocation

    stages = germline_labels()
    dfs = []
    for t in range(len(stages)):
        df = pd.read_csv(net_dir / f"t{t:02d}_TF_TG_top10000.tsv", sep="\t")
        df["prob"] = (df["score_cos"].astype(float) + 1.0) / 2.0
        s = df["prob"].to_numpy()
        k = min(34_400, len(df))
        part = np.argpartition(-s, kth=k - 1)[:k]
        part = part[np.argsort(-s[part])]
        sel = df.iloc[part].reset_index(drop=True)
        dfs.append(pd.DataFrame({"cell_cluster": t, "tf_idx": sel.TF_idx.astype(int), "tg_idx": sel.TG_idx.astype(int),
                                 "score": sel.prob.astype(float)}))
    edges = pd.concat(dfs, ignore_index=True)
    tf_names = pd.unique(pd.read_csv(paths.data("fly", "raw", "TF.tsv"), sep="\t", dtype=str).fillna("")["gene_name"]
                         .astype(str)).tolist()
    tg_names = pd.unique(pd.read_csv(paths.data("fly", "raw", "TG.tsv"), sep="\t", dtype=str).fillna("")["gene_name"]
                         .astype(str)).tolist()
    edges = edges[edges.tf_idx.isin(range(len(tf_names))) & edges.tg_idx.isin(range(len(tg_names)))].copy()
    edges["tg_name"] = edges.tg_idx.map(dict(enumerate(tg_names)))
    codes, keys = pd.MultiIndex.from_frame(edges[["tf_idx", "tg_idx"]]).factorize()
    counts = coo_matrix((np.ones(len(codes)), (codes, edges.cell_cluster.to_numpy(int))),
                        shape=(len(keys), len(stages))).tocsr()
    lda = LatentDirichletAllocation(n_components=12, learning_method="batch", max_iter=300, random_state=42,
                                    evaluate_every=-1, verbose=0)
    w = lda.fit_transform(counts)
    h = lda.components_
    pd.DataFrame(h / h.sum(1, keepdims=True), index=[f"T{i}" for i in range(12)], columns=stages).to_csv(
        out / "topic_x_stage_prop.tsv", sep="\t")
    key2tg = (edges.groupby(["tf_idx", "tg_idx", "tg_name"]).size().reset_index(name="cnt")
              .sort_values(["tf_idx", "tg_idx", "cnt"], ascending=[True, True, False]).drop_duplicates(["tf_idx", "tg_idx"]))
    key2tg = {(int(a), int(b)): str(n) for a, b, n in zip(key2tg.tf_idx, key2tg.tg_idx, key2tg.tg_name)}
    edge_list = list(keys)
    lists = {f"T{t}": sorted({key2tg[(int(edge_list[i][0]), int(edge_list[i][1]))] for i in np.argsort(-w[:, t])[:500]})
             for t in range(12)}
    (out / "topic_top500_edge_TGs.json").write_text(json.dumps(lists, indent=1), encoding="utf-8")
    np.savez_compressed(out / "lda_W.npz", W=w, tf_idx=np.array([a for a, _ in edge_list]),
                        tg_idx=np.array([b for _, b in edge_list]),
                        tg_name=np.array([key2tg[(int(a), int(b))] for a, b in edge_list]))


def topics() -> dict:
    """Recompute the topic networks and the LDA, and compare both with the frozen copies."""
    out = paths.FLY / "germ" / "topic_notebook"
    net_dir = topic_networks(out)
    topic_lda(net_dir, out)
    report = {}
    same_net = []
    for t in range(15):
        a, b = (pd.read_csv(p, sep="\t").sort_values(["TF_idx", "TG_idx"], kind="stable").reset_index(drop=True)
                for p in (net_dir / f"t{t:02d}_TF_TG_top10000.tsv",
                          frozen("topic", "network_top10000", f"t{t:02d}_TF_TG_top10000.tsv")))
        same_net.append(bool(a[["TF_idx", "TG_idx"]].equals(b[["TF_idx", "TG_idx"]])
                             and np.allclose(a.score_cos, b.score_cos, atol=1e-6, rtol=0)))
    report["network_top10000_identical_stages"] = int(sum(same_net))
    a = pd.read_csv(out / "topic_x_stage_prop.tsv", sep="\t", index_col=0)
    b = pd.read_csv(frozen("topic", "topic_x_stage_prop.tsv"), sep="\t", index_col=0)
    report["topic_x_stage_prop_max_abs_diff"] = float((a - b).abs().to_numpy().max())
    lists_new = json.loads((out / "topic_top500_edge_TGs.json").read_text(encoding="utf-8"))
    lists_ref = json.loads(frozen("topic", "topic_top500_edge_TGs.json").read_text(encoding="utf-8"))
    report["topic_top500_TG_lists_identical"] = lists_new == lists_ref
    _save_report(report, "topic_vs_frozen.json")
    return report


def _save_report(report: dict, name: str) -> None:
    p = paths.result("fly", "germ", "checks", name)
    p.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
