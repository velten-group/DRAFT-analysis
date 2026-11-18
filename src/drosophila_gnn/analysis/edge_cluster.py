"""Edge clusters of the stage networks (k-means on the stage pattern of each TF-TG edge)."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from ..checkpoint import save_checkpoint_metadata
from ..utils import ensure_directory, load_yaml_config, resolve_config_path
from .network import discover_network_time_indices, load_and_select_all


DEFAULT_GERM_CELLTYPES = [
    "GSC",
    "Gb",
    "Sp-2",
    "Sp-4",
    "Sp-8",
    "Sp-16",
    "1deg Sc",
    "1deg Sc",
    "1deg Sc",
    "2deg Sc",
    "2deg Sc",
    "2deg Sc",
    "2deg Sc",
    "2deg Sc",
    "2deg Sc",
]


def _section(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    return value if isinstance(value, dict) else {}


def _edge_cluster_settings(config: dict[str, Any]) -> dict[str, Any]:
    section = _section(config, "edge_cluster")
    lineage = str(config.get("lineage", "")).lower()
    clusters = section.get("clusters")
    if clusters is not None and not isinstance(clusters, list):
        clusters = [clusters]
    celltypes = section.get("celltypes")
    if celltypes is not None and not isinstance(celltypes, list):
        celltypes = [celltypes]
    selection_mode = str(section.get("selection_mode", "topk")).strip().lower()
    if selection_mode not in {"topk", "threshold", "both", "all", "tf_rescue"}:
        raise ValueError(f"Unsupported edge_cluster.selection_mode={selection_mode!r}.")
    return {
        "network_dir": str(section.get("network_dir", f"result/{lineage}/network")),
        "output_dir": str(section.get("output_dir", f"result/{lineage}/edge_cluster")),
        "clusters": [int(value) for value in clusters] if clusters is not None else None,
        "celltypes": [str(value) for value in celltypes] if celltypes is not None else None,
        "selection_mode": selection_mode,
        "top_n_per_cluster": int(section.get("top_n_per_cluster", 14_200)),
        "score_threshold": float(section.get("score_threshold", 0.80)),
        "n_clusters": int(section.get("n_clusters", 12)),
        "random_state": int(section.get("random_state", 4256)),
        "save_fig": bool(section.get("save_fig", True)),
        "top_tf_per_cluster": int(section.get("top_tf_per_cluster", 20)),
        "top_tg_per_cluster": int(section.get("top_tg_per_cluster", 20)),
        "heatmap_cmap": str(section.get("heatmap_cmap", "OrRd")),
        "specificity_min_edges": int(section.get("specificity_min_edges", 50)),
        "specificity_min_max_prop": float(section.get("specificity_min_max_prop", 0.45)),
        "specificity_min_gap": float(section.get("specificity_min_gap", 0.15)),
        "specificity_topk": int(section.get("specificity_topk", 3)),
        "topk_per_ec": int(section.get("topk_per_ec", 10)),
        "tf_grid_nrows": int(section.get("tf_grid_nrows", 3)),
        "tf_grid_ncols": int(section.get("tf_grid_ncols", 4)),
        "score_vmin": float(section.get("score_vmin", 0.5)),
        "score_vmax": float(section.get("score_vmax", 1.0)),
    }


def _cluster_labels(settings: dict[str, Any], config: dict[str, Any], clusters: list[int]) -> list[str]:
    if settings["celltypes"] is not None and len(settings["celltypes"]) == len(clusters):
        return settings["celltypes"]
    lineage = str(config.get("lineage", "")).lower()
    if lineage in {"germ", "germline"} and len(clusters) == len(DEFAULT_GERM_CELLTYPES):
        return DEFAULT_GERM_CELLTYPES
    return [f"t{cluster:02d}" for cluster in clusters]


def plot_heatmap_with_ec_bar(
    matrix: np.ndarray,
    xticklabels: list[str],
    yticklabels: list[str],
    title: str,
    path: str | Path,
    ec_counts: np.ndarray,
    cmap_name: str,
) -> str:
    fig = plt.figure(figsize=(max(10, 0.8 * len(xticklabels) + 3.0), max(6, 0.55 * len(yticklabels) + 1.5)))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.25, 6.0], wspace=0.15)
    ax_bar = fig.add_subplot(grid[0, 0])
    ax_heat = fig.add_subplot(grid[0, 1])

    y_pos = np.arange(len(yticklabels))
    ax_bar.barh(y_pos, ec_counts, color="#b45a3c")
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels([])
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("n", fontsize=12)
    ax_bar.tick_params(axis="x", labelsize=11)
    ax_bar.tick_params(axis="y", length=0)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    image = ax_heat.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap_name)
    ax_heat.set_xticks(np.arange(len(xticklabels)))
    ax_heat.set_xticklabels(xticklabels, rotation=45, ha="right", fontsize=12)
    ax_heat.set_yticks(np.arange(len(yticklabels)))
    ax_heat.set_yticklabels(yticklabels, fontsize=12)
    ax_heat.set_title(title, fontsize=16)
    colorbar = fig.colorbar(image, ax=ax_heat, fraction=0.046, pad=0.04)
    colorbar.ax.tick_params(labelsize=11)
    fig.tight_layout()

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def infer_ec_specificity_topmix(
    prop_mat: np.ndarray,
    counts_mat: np.ndarray,
    xt_labels: list[str],
    min_edges: int = 50,
    min_max_prop: float = 0.45,
    min_gap: float = 0.15,
    topk: int = 3,
) -> pd.DataFrame:
    n_ec, n_clusters = prop_mat.shape
    n_edges = counts_mat.sum(axis=1)
    keep_topk = min(topk, n_clusters)
    top_idx = np.argsort(-prop_mat, axis=1)[:, :keep_topk]
    top_props = np.take_along_axis(prop_mat, top_idx, axis=1)
    max_prop = top_props[:, 0]
    second_prop = top_props[:, 1] if keep_topk >= 2 else np.zeros(n_ec, dtype=float)
    gap = max_prop - second_prop

    top_mix = []
    for edge_cluster in range(n_ec):
        pairs = [f"{xt_labels[top_idx[edge_cluster, rank]]}:{top_props[edge_cluster, rank]:.2f}" for rank in range(keep_topk)]
        top_mix.append(", ".join(pairs))

    is_specific = (n_edges >= min_edges) & (max_prop >= min_max_prop) & (gap >= min_gap)
    specific_to = [
        xt_labels[top_idx[edge_cluster, 0]] if is_specific[edge_cluster] else "mixed"
        for edge_cluster in range(n_ec)
    ]

    return pd.DataFrame(
        {
            "edge_cluster": np.arange(n_ec, dtype=int),
            "specific_to": specific_to,
            "max_prop": max_prop,
            "second_prop": second_prop,
            "gap": gap,
            "n_edges": n_edges,
            "top_mix": top_mix,
        }
    )


def _export_selected_edges(
    network_dir: str | Path,
    clusters: list[int],
    labels: list[str],
    output_dir: Path,
    settings: dict[str, Any],
) -> tuple[dict[int, pd.DataFrame], pd.DataFrame]:
    selected_by_cluster = load_and_select_all(
        network_dir,
        clusters,
        mode=settings["selection_mode"],
        top_n=settings["top_n_per_cluster"],
        thr=settings["score_threshold"],
    )
    selection_suffix = settings["selection_mode"]
    frames: list[pd.DataFrame] = []
    for cluster, label in zip(clusters, labels):
        frame = selected_by_cluster[cluster].copy()
        if frame.empty:
            continue
        frame["cell_cluster"] = int(cluster)
        frame["cell_type"] = str(label)
        frame["tf_idx"] = frame["TF_idx"]
        frame["tg_idx"] = frame["TG_idx"]
        frame["tf_name"] = frame["TF_name"].astype(str)
        frame["tg_name"] = frame["TG_name"].astype(str)
        frame["score"] = frame["prob"].astype(float)
        frame_out = frame[["cell_cluster", "cell_type", "tf_idx", "tg_idx", "tf_name", "tg_name", "score"]]
        frame_out.to_csv(output_dir / f"edges_{label}_{selection_suffix}.csv", index=False)
        frames.append(frame_out)

    df_edges = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["cell_cluster", "cell_type", "tf_idx", "tg_idx", "tf_name", "tg_name", "score"])
    )
    df_edges.to_csv(output_dir / f"edges_selected_{selection_suffix}.csv", index=False)
    return selected_by_cluster, df_edges


def _export_top_nodes_by_degree(
    df_edges: pd.DataFrame,
    labels_by_cluster: dict[int, str],
    top_tf_per_cluster: int,
    top_tg_per_cluster: int,
    output_dir: Path,
    selection_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tf_degree = (
        df_edges.groupby(["cell_cluster", "tf_idx", "tf_name"])["tg_idx"]
        .nunique()
        .reset_index(name="degree")
        .sort_values(["cell_cluster", "degree", "tf_name"], ascending=[True, False, True], kind="mergesort")
    )
    tg_degree = (
        df_edges.groupby(["cell_cluster", "tg_idx", "tg_name"])["tf_idx"]
        .nunique()
        .reset_index(name="degree")
        .sort_values(["cell_cluster", "degree", "tg_name"], ascending=[True, False, True], kind="mergesort")
    )
    top_tf = tf_degree.groupby("cell_cluster", group_keys=False).head(top_tf_per_cluster).reset_index(drop=True)
    top_tg = tg_degree.groupby("cell_cluster", group_keys=False).head(top_tg_per_cluster).reset_index(drop=True)
    top_tf["cell_type"] = top_tf["cell_cluster"].map(labels_by_cluster)
    top_tg["cell_type"] = top_tg["cell_cluster"].map(labels_by_cluster)
    top_tf.to_csv(output_dir / f"top_TF_by_degree_{selection_mode}.csv", index=False)
    top_tg.to_csv(output_dir / f"top_TG_by_degree_{selection_mode}.csv", index=False)
    return top_tf, top_tg


def _edge_binary_matrix(df_edges: pd.DataFrame, clusters: list[int]) -> tuple[list[tuple[Any, Any]], np.ndarray]:
    edge_tuples = pd.MultiIndex.from_frame(df_edges[["tf_idx", "tg_idx"]]).unique().tolist()
    edge_to_row = {edge: row_idx for row_idx, edge in enumerate(edge_tuples)}
    cluster_to_col = {cluster: col_idx for col_idx, cluster in enumerate(clusters)}
    matrix = np.zeros((len(edge_tuples), len(clusters)), dtype=float)
    for cluster, group in df_edges.groupby("cell_cluster"):
        col_idx = cluster_to_col[int(cluster)]
        for edge in pd.MultiIndex.from_frame(group[["tf_idx", "tg_idx"]]).unique().tolist():
            matrix[edge_to_row[edge], col_idx] = 1.0
    return edge_tuples, matrix


def _tf_centrality_summary(
    df_edges: pd.DataFrame,
    df_edge_clusters: pd.DataFrame,
    df_ec_spec: pd.DataFrame,
    output_dir: Path,
    settings: dict[str, Any],
) -> tuple[pd.DataFrame, list[str]]:
    te_regex = re.compile(
        r"(transposable[_\s-]*element|retrotranspos|transposon|^RR\d+_transposable_element$)",
        flags=re.IGNORECASE,
    )

    def is_te(name: str) -> bool:
        return bool(te_regex.search(str(name))) if name is not None else False

    edge_with_ec = pd.merge(
        df_edges[["tf_idx", "tg_idx", "tf_name", "tg_name", "score"]],
        df_edge_clusters[["tf_idx", "tg_idx", "edge_cluster"]],
        on=["tf_idx", "tg_idx"],
        how="left",
    )
    edge_with_ec_no_te = edge_with_ec[~edge_with_ec["tf_name"].apply(is_te)].copy()
    df_edge_clusters_no_te = df_edge_clusters[~df_edge_clusters["tf_name"].apply(is_te)].copy()

    deg_per_ec_tf = (
        df_edge_clusters_no_te.groupby(["edge_cluster", "tf_idx", "tf_name"])["tg_idx"]
        .nunique()
        .reset_index(name="degree")
    )
    score_per_ec_tf = (
        edge_with_ec_no_te.groupby(["edge_cluster", "tf_idx", "tf_name"])["score"]
        .mean()
        .reset_index(name="mean_score")
    )
    stats_tf = pd.merge(
        deg_per_ec_tf,
        score_per_ec_tf,
        on=["edge_cluster", "tf_idx", "tf_name"],
        how="left",
    )

    summary_rows: list[pd.DataFrame] = []
    for edge_cluster in sorted(stats_tf["edge_cluster"].unique().tolist()):
        sub = stats_tf[stats_tf["edge_cluster"] == edge_cluster].copy()
        if sub.empty:
            continue
        sub.sort_values(["degree", "mean_score", "tf_name"], ascending=[False, False, True], inplace=True, kind="mergesort")
        sub = sub.head(settings["topk_per_ec"]).reset_index(drop=True)
        sub.insert(1, "rank", np.arange(1, len(sub) + 1))
        spec_row = df_ec_spec[df_ec_spec["edge_cluster"] == edge_cluster]
        if not spec_row.empty:
            sub["specific_to"] = str(spec_row.iloc[0]["specific_to"])
            sub["top_mix"] = str(spec_row.iloc[0]["top_mix"])
        summary_rows.append(sub)

    summary = (
        pd.concat(summary_rows, ignore_index=True)
        if summary_rows
        else pd.DataFrame(columns=["edge_cluster", "rank", "tf_idx", "tf_name", "degree", "mean_score", "specific_to", "top_mix"])
    )
    summary_path = output_dir / f"ec_top{settings['topk_per_ec']}_TF_centrality_bar_data_{settings['selection_mode']}.csv"
    summary.to_csv(summary_path, index=False)

    saved_figures: list[str] = []
    if settings["save_fig"] and not summary.empty:
        cmap = mpl.cm.get_cmap(settings["heatmap_cmap"]).copy()
        cmap.set_bad((0.85, 0.85, 0.85, 1.0))
        norm = mpl.colors.Normalize(vmin=settings["score_vmin"], vmax=settings["score_vmax"])

        nrows = settings["tf_grid_nrows"]
        ncols = settings["tf_grid_ncols"]
        per_page = nrows * ncols
        ecs = sorted(summary["edge_cluster"].unique().tolist())
        n_pages = int(math.ceil(len(ecs) / per_page))

        for page in range(n_pages):
            chunk = ecs[page * per_page : (page + 1) * per_page]
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 12))
            axes_arr = np.asarray(axes).ravel()
            for axis_idx, ax in enumerate(axes_arr):
                if axis_idx >= len(chunk):
                    ax.axis("off")
                    continue
                edge_cluster = int(chunk[axis_idx])
                sub = summary[summary["edge_cluster"] == edge_cluster].copy()
                if sub.empty:
                    ax.axis("off")
                    continue
                sub = sub.sort_values(["degree", "mean_score"], ascending=[True, True], kind="mergesort")
                colors = cmap(norm(sub["mean_score"].fillna(settings["score_vmin"]).to_numpy()))
                ax.barh(sub["tf_name"], sub["degree"], color=colors)
                first = sub.iloc[0]
                title = f"EC{edge_cluster} | {first.get('specific_to', 'mixed')}"
                ax.set_title(title, fontsize=12)
                ax.set_xlabel("Degree", fontsize=12)
                ax.tick_params(axis="x", labelsize=12)
                ax.tick_params(axis="y", labelsize=14)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
            fig.suptitle("Top TF centrality per edge cluster", fontsize=18)
            fig.tight_layout()
            out_path = output_dir / (
                f"ec_TF_centrality_grid_{nrows}x{ncols}_page{page + 1:02d}_{settings['selection_mode']}.png"
            )
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            saved_figures.append(str(out_path))

    return summary, saved_figures


def run_edge_cluster_from_config(config: dict[str, Any]) -> dict[str, Any]:
    settings = _edge_cluster_settings(config)
    output_dir = ensure_directory(resolve_config_path(config, settings["output_dir"]))
    network_dir = resolve_config_path(config, settings["network_dir"])
    clusters = settings["clusters"] or discover_network_time_indices(network_dir)
    labels = _cluster_labels(settings, config, clusters)
    labels_by_cluster = {cluster: label for cluster, label in zip(clusters, labels)}

    _, df_edges = _export_selected_edges(
        network_dir=network_dir,
        clusters=clusters,
        labels=labels,
        output_dir=output_dir,
        settings=settings,
    )
    _export_top_nodes_by_degree(
        df_edges=df_edges,
        labels_by_cluster=labels_by_cluster,
        top_tf_per_cluster=settings["top_tf_per_cluster"],
        top_tg_per_cluster=settings["top_tg_per_cluster"],
        output_dir=output_dir,
        selection_mode=settings["selection_mode"],
    )

    if df_edges.empty:
        raise ValueError(f"No edges were selected from {network_dir} with mode={settings['selection_mode']}.")

    edge_tuples, matrix = _edge_binary_matrix(df_edges, clusters)
    scaler = StandardScaler(with_mean=True, with_std=True)
    matrix_z = scaler.fit_transform(matrix) if matrix.shape[0] > 1 else matrix
    km = KMeans(n_clusters=settings["n_clusters"], n_init=10, random_state=settings["random_state"])
    labels_pred = km.fit_predict(matrix_z) if matrix.shape[0] > 1 else np.zeros((matrix.shape[0],), dtype=int)

    tf_name_map = (
        df_edges.groupby(["tf_idx", "tg_idx"])["tf_name"]
        .first()
        .to_dict()
    )
    tg_name_map = (
        df_edges.groupby(["tf_idx", "tg_idx"])["tg_name"]
        .first()
        .to_dict()
    )
    df_edge_clusters = pd.DataFrame(
        {
            "tf_idx": [edge[0] for edge in edge_tuples],
            "tg_idx": [edge[1] for edge in edge_tuples],
            "tf_name": [tf_name_map[(edge[0], edge[1])] for edge in edge_tuples],
            "tg_name": [tg_name_map[(edge[0], edge[1])] for edge in edge_tuples],
            "edge_cluster": labels_pred,
        }
    )
    edge_clusters_path = output_dir / f"edge_clusters_{settings['n_clusters']}k_{settings['selection_mode']}.csv"
    df_edge_clusters.to_csv(edge_clusters_path, index=False)

    edge_with_ec = pd.merge(
        df_edges,
        df_edge_clusters[["tf_idx", "tg_idx", "edge_cluster"]],
        on=["tf_idx", "tg_idx"],
        how="left",
    )
    counts_df = (
        pd.crosstab(edge_with_ec["edge_cluster"], edge_with_ec["cell_cluster"])
        .reindex(index=range(settings["n_clusters"]), columns=clusters, fill_value=0)
    )
    counts = counts_df.to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        prop = np.divide(counts, np.maximum(counts.sum(axis=1, keepdims=True), 1e-12))
    prop_df = pd.DataFrame(prop, index=counts_df.index, columns=counts_df.columns)
    counts_path = output_dir / f"edgeCluster_x_cellCluster_counts_{settings['selection_mode']}.csv"
    prop_path = output_dir / f"edgeCluster_x_cellCluster_prop_{settings['selection_mode']}.csv"
    counts_df.to_csv(counts_path)
    prop_df.to_csv(prop_path)

    heatmap_paths: list[str] = []
    if settings["save_fig"]:
        xt = [labels_by_cluster[cluster] for cluster in clusters]
        yt = [f"EC{k}" for k in range(settings["n_clusters"])]
        ec_edge_counts = counts.sum(axis=1)
        heatmap_paths.append(
            plot_heatmap_with_ec_bar(
                counts,
                xticklabels=xt,
                yticklabels=yt,
                title="Edge cluster",
                path=output_dir / f"heatmap_counts_{settings['n_clusters']}k_{settings['selection_mode']}.png",
                ec_counts=ec_edge_counts,
                cmap_name=settings["heatmap_cmap"],
            )
        )
        heatmap_paths.append(
            plot_heatmap_with_ec_bar(
                prop,
                xticklabels=xt,
                yticklabels=yt,
                title="Edge cluster",
                path=output_dir / f"heatmap_prop_{settings['n_clusters']}k_{settings['selection_mode']}.png",
                ec_counts=ec_edge_counts,
                cmap_name=settings["heatmap_cmap"],
            )
        )

    df_ec_spec = infer_ec_specificity_topmix(
        prop,
        counts,
        _cluster_labels(settings, config, clusters),
        min_edges=settings["specificity_min_edges"],
        min_max_prop=settings["specificity_min_max_prop"],
        min_gap=settings["specificity_min_gap"],
        topk=settings["specificity_topk"],
    )
    specificity_path = output_dir / f"edgeCluster_specificity_{settings['selection_mode']}.csv"
    df_ec_spec.to_csv(specificity_path, index=False)

    tf_summary, tf_figures = _tf_centrality_summary(
        df_edges=df_edges,
        df_edge_clusters=df_edge_clusters,
        df_ec_spec=df_ec_spec,
        output_dir=output_dir,
        settings=settings,
    )

    report = {
        "analysis": "edge_cluster",
        "output_dir": str(output_dir),
        "network_dir": str(network_dir),
        "clusters": clusters,
        "cluster_labels": labels,
        "selection_mode": settings["selection_mode"],
        "selected_edge_count": int(len(df_edges)),
        "union_edge_count": int(len(edge_tuples)),
        "n_edge_clusters": int(settings["n_clusters"]),
        "edge_clusters_path": str(edge_clusters_path),
        "counts_path": str(counts_path),
        "prop_path": str(prop_path),
        "specificity_path": str(specificity_path),
        "heatmap_paths": heatmap_paths,
        "tf_summary_path": str(
            output_dir / f"ec_top{settings['topk_per_ec']}_TF_centrality_bar_data_{settings['selection_mode']}.csv"
        ),
        "tf_figure_paths": tf_figures,
        "settings": settings,
    }
    save_checkpoint_metadata(output_dir / "edge_cluster_report.json", report)
    return report


def run_edge_cluster(config: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(config, (str, Path)):
        loaded = load_yaml_config(config)
        report = run_edge_cluster_from_config(loaded)
        report["config_path"] = str(Path(config).resolve())
        return report
    return run_edge_cluster_from_config(config)
