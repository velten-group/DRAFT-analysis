"""Fig. 3a, c-e and Supp. Figs. 5-6 from the checkpoint: the cells of the original mouse notebooks, then the figure tables.

1. scripts/mouse/rerun_mouse_notebooks.py executes the hash-checked cells of the original notebooks
   (scripts/mouse/original_notebooks): network export from the checkpoint's saved cluster score matrices, model
   distances (Fig. 3c, DRAFT), 12 edge clusters (k-means, seed 5465), 10-topic LDA (seed 42) and GO BP 2021
   over-representation against the 7,764 network targets. No model is trained or run forward.
2. Edge-cluster and topic numbers are labels: the recomputed tables are compared with the frozen copies
   (data/mouse/frozen_numbering, as for the fly), which every figure table then reads.
3. Distances of the baselines (Fig. 3c): unsquared Frobenius distances between the cluster networks of Dictys, SCENIC
   (native scores) and scMTNI+Prior (published consensus), on the checkpoint's TF x TG axes; continuity along the
   lineage (scripts/mouse/mouse_lineage_continuity.py).
"""
from __future__ import annotations

import json
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd

from .. import paths
from . import CLUSTERS, SCRIPTS, native, out, scripts_on_path

NOTEBOOKS = SCRIPTS / "original_notebooks"
ORDER = [f"C{c}" for c in CLUSTERS]
# TFs of Fig. 3d (literature-informed candidates) and GO callouts of Fig. 3e: curated choices, as the Fig. 4g annotations
TF_CANDIDATES = (("Foxs1", "C4"), ("Fosl1", "C4"), ("Egr1", "C4"), ("Esrrb", "C5"), ("Zscan10", "C5"), ("Klf5", "C5"))
GO_CALLOUTS = (("T9", "C1;C5", "GO:0006281", "DNA repair"), ("T9", "C1;C5", "GO:0000723", "Telomere maintenance"),
               ("T6", "C3", "GO:0030198", "Extracellular matrix organization"),
               ("T6", "C3", "GO:0030199", "Collagen fibril organization"),
               ("T8", "C7;C3", "GO:0042981", "Regulation of apoptotic process"),
               ("T8", "C7;C3", "GO:0090383", "Phagosome acidification"))


def run_dir():
    return out("notebook_run")


def replay() -> None:
    """Fresh run of the original notebook cells into results/checkpoint/mouse/notebook_run."""
    import shutil

    for d in (run_dir(), out("notebook_figures")):
        if d.exists():
            shutil.rmtree(d)
    subprocess.run([sys.executable, str(SCRIPTS / "rerun_mouse_notebooks.py"), "--source-root", str(NOTEBOOKS),
                    "--checkpoint", str(paths.data("mouse", "checkpoint", "grn_best.pt")),
                    "--data-dir", str(paths.data("mouse", "raw")),
                    "--go-library", str(paths.data("mouse", "raw", "resources", "GO_Biological_Process_2021.gmt")),
                    "--run-dir", str(run_dir()), "--output-dir", str(out("notebook_figures"))],
                   check=True, cwd=paths.REPO)


# ---------------------------------------------------------------------------------------------- composition, 3a
def composition() -> pd.DataFrame:
    scripts_on_path()
    from lineage_composition import read_inputs

    rows, _, _, _ = read_inputs(paths.data("mouse", "raw"))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------- distances, 3c
def distances() -> dict[str, pd.DataFrame]:
    """Raw unsquared Frobenius distances between the seven cluster networks of each method (cluster order C4 ... C5)."""
    scripts_on_path()
    import torch
    from network_distances import consensus_vector, frobenius_distances
    from official_aupr_protocol import canonical_edges
    from scipy import sparse

    ck = torch.load(paths.data("mouse", "checkpoint", "grn_best.pt"), map_location="cpu", weights_only=False)
    tfs, tgs = ck["meta"]["TF_list"], ck["meta"]["TG_list"]
    del ck
    model = pd.read_csv(run_dir() / "model_distance.tsv", sep="\t", index_col=0, float_precision="round_trip")
    result = {"DRAFT": model.loc[ORDER, ORDER]}
    features = json.loads((native("features") / "features.json").read_text(encoding="utf-8"))
    ftf, ftg = features["TF"], features["TG"]
    ti, gi = {s: i for i, s in enumerate(ftf)}, {s: i for i, s in enumerate(ftg)}
    for method, folder in (("Dictys", "dictys"), ("SCENIC", "scenic")):
        vectors = []
        for cid in CLUSTERS:
            frame, _ = canonical_edges(pd.read_csv(native(folder) / f"C{cid}" / "edges.tsv.gz", sep="\t",
                                                   float_precision="round_trip"), ftf, ftg)
            codes = frame.TF.map(ti).to_numpy(np.int64) * len(ftg) + frame.TG.map(gi).to_numpy(np.int64)
            vectors.append(sparse.csr_array((frame.score.to_numpy(), (np.zeros(len(frame), dtype=int), codes)),
                                            shape=(1, len(ftf) * len(ftg))))
        result[method] = pd.DataFrame(frobenius_distances(vectors), index=ORDER, columns=ORDER)
    vectors = [consensus_vector(paths.data("mouse", "raw", "scMTNI_plus_prior", f"cluster{c}_consensus_edges.txt"),
                                tfs, tgs)[0] for c in CLUSTERS]
    result["scMTNI+Prior"] = pd.DataFrame(frobenius_distances(vectors), index=ORDER, columns=ORDER)
    return result


def continuity(raw: pd.DataFrame) -> dict:
    scripts_on_path()
    from mouse_lineage_continuity import lineage_summary

    s = lineage_summary(raw)
    return {"successful_mean_continuity": s["Successful"]["mean_continuity"],
            "unsuccessful_mean_continuity": s["Unsuccessful"]["mean_continuity"],
            "mean_continuity": s["overall"]["mean_continuity"], "normalization_maximum": s["overall"]["diameter"]}


# ---------------------------------------------------------------------------------------------- modules, 3d-e, S5-6
def numbering_check() -> dict:
    """Recomputed edge clusters and topic weights vs the frozen copies (results/checkpoint/mouse/checks)."""
    frozen = paths.data("mouse", "frozen_numbering")
    run = run_dir()
    report = {}
    for name in ("edge_clusters_12k_from_network_all.csv", "edgeCluster_x_cellCluster_counts_from_network_all.csv",
                 "edges_selected_from_network_all.csv"):
        a, b = pd.read_csv(run / "edge" / name), pd.read_csv(frozen / "edge" / name)
        report[name] = {"rows": [len(a), len(b)], "identical": bool(a.equals(b))}
    a = pd.read_csv(run / "topic10" / "topic_proportions.tsv", sep="\t", index_col=0)
    b = pd.read_csv(frozen / "topic10" / "topic_proportions.tsv", sep="\t", index_col=0)
    report["topic_proportions.tsv"] = {"max_abs_diff": float(np.abs(a.to_numpy() - b.to_numpy()).max())}
    p = out("checks", "numbering_vs_frozen.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def frozen(*parts: str):
    return paths.data("mouse", "frozen_numbering", *parts)


def module_statistics():
    """Edge-cluster TF statistics and topic helpers of the original notebooks (scripts/mouse/edge_cluster_statistics.py)."""
    scripts_on_path()
    from edge_cluster_statistics import source_statistics

    return source_statistics(NOTEBOOKS, frozen())


def selected_tfs() -> pd.DataFrame:
    """Target counts, dominant edge cluster and module specificity of the six Fig. 3d TFs in their cell cluster."""
    edge = frozen("edge")
    raw = pd.read_csv(edge / "edge_clusters_12k_from_network_all.csv")
    spec = pd.read_csv(edge / "edgeCluster_top20_TF_module_specificity.csv")
    spec["rank"] = spec.groupby("edge_cluster").S_kt.rank(method="min", ascending=False).astype(int)
    nets = {c: pd.read_csv(edge / f"edges_{c}_all.csv").merge(raw[["tf_idx", "tg_idx", "edge_cluster"]],
                                                              on=["tf_idx", "tg_idx"], validate="one_to_one")
            for c in sorted({c for _, c in TF_CANDIDATES})}
    rows = []
    for gene, cluster in TF_CANDIDATES:
        hits = nets[cluster].loc[nets[cluster].tf_name == gene]
        degrees = hits.groupby("edge_cluster").tg_idx.nunique().sort_index()
        if (degrees == degrees.max()).sum() != 1:
            raise ValueError(f"ambiguous dominant EC for {gene}")
        module = int(degrees.idxmax())
        stat = spec.loc[(spec.edge_cluster == module) & (spec.tf_name == gene)].iloc[0]
        display = 1 if cluster == "C4" else 4
        rows.append({"tf": gene, "original_cluster": cluster, "cluster_tg_degree": hits.tg_idx.nunique(),
                     "dominant_module_at_original_cluster": f"EC{module}",
                     "dominant_module_cluster_tg_degree": int(degrees.loc[module]),
                     "whole_module_tf_tg_degree": raw.loc[(raw.edge_cluster == module) & (raw.tf_name == gene),
                                                          "tg_idx"].nunique(),
                     "module_specificity": float(stat.S_kt), "module_specificity_rank": int(stat["rank"]),
                     "highlighted_modules": f"EC{display}",
                     "highlighted_module_cluster_tg_degree": int(degrees.loc[display])})
    return pd.DataFrame(rows)


def selected_go() -> pd.DataFrame:
    tables, rows = {}, []
    for topic, clusters, go_id, label in GO_CALLOUTS:
        if topic not in tables:
            tables[topic] = pd.read_csv(frozen("topic10", "local_go", f"{topic}_local_go.tsv"), sep="\t")
        m = tables[topic].loc[tables[topic].Term.str.contains(go_id, regex=False)]
        if len(m) != 1:
            raise ValueError(f"expected one enrichment row for {topic}/{go_id}")
        rows.append({"tag": f"P{list(tables).index(topic) + 1}", "topic": topic, "cell_clusters": clusters,
                     "display_label": label, **m.iloc[0].to_dict()})
    return pd.DataFrame(rows)


def tf_top12(namespace) -> pd.DataFrame:
    stats, titles = namespace["stats_tf"], namespace["ec2title"]
    parts = []
    for ec in sorted(stats.edge_cluster.unique()):
        sub = stats.loc[stats.edge_cluster == ec].copy()
        sub.sort_values("degree", ascending=False, inplace=True)
        plotted = sub.head(12).copy().sort_values("degree", ascending=True)
        parts.append(plotted.assign(panel_title=f"EC{ec} | {titles[ec]}"))
    return pd.concat(parts, ignore_index=True)


def go_top10(helpers) -> pd.DataFrame:
    with frozen("topic10", "lda_artifacts.pkl").open("rb") as handle:
        art = pickle.load(handle)
    suffixes = helpers["_topic_cluster_suffixes_from_H"](art["H"], art["CLUSTERS"], thr=0.10, topn=3, digits=3)
    parts = []
    for t in range(art["H"].shape[0]):
        frame = helpers["prep_enrichment_bar_df"](pd.read_csv(frozen("topic10", "local_go", f"T{t}_local_go.tsv"),
                                                              sep="\t"), top_terms=10)
        parts.append(frame.assign(topic=f"T{t}", panel_title=f"Topic{t}{suffixes[t]}"))
    return pd.concat(parts, ignore_index=True)
