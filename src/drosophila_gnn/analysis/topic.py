"""Regulatory topics (LDA) of the stage networks and their enrichment."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.decomposition import LatentDirichletAllocation

from ..checkpoint import save_checkpoint_metadata
from ..io import strip_copy_suffix
from ..utils import ensure_directory, load_yaml_config, resolve_config_path
from .network import build_edge_dataframe, discover_network_time_indices


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


def _topic_settings(config: dict[str, Any]) -> dict[str, Any]:
    section = _section(config, "topic")
    lineage = str(config.get("lineage", "")).lower()
    data_section = _section(config, "data")
    clusters = section.get("clusters")
    if clusters is not None and not isinstance(clusters, list):
        clusters = [clusters]
    celltypes = section.get("celltypes")
    if celltypes is not None and not isinstance(celltypes, list):
        celltypes = [celltypes]
    figsize = section.get("enrich_grid_figsize", [15, 20])
    if not isinstance(figsize, list) or len(figsize) != 2:
        figsize = [15, 20]
    selection_mode = str(section.get("selection_mode", "topk")).strip().lower()
    if selection_mode not in {"topk", "threshold", "both", "all", "tf_rescue"}:
        raise ValueError(f"Unsupported topic.selection_mode={selection_mode!r}.")
    return {
        "network_dir": str(section.get("network_dir", f"result/{lineage}/network")),
        "output_dir": str(section.get("output_dir", f"result/{lineage}/topic")),
        "clusters": [int(value) for value in clusters] if clusters is not None else None,
        "celltypes": [str(value) for value in celltypes] if celltypes is not None else None,
        "selection_mode": selection_mode,
        "top_n_per_cluster": int(section.get("top_n_per_cluster", 34_400)),
        "score_threshold": float(section.get("score_threshold", 0.85)),
        "tf_list_file": str(section.get("tf_list_file", data_section.get("tf_list", "data/raw/TF.tsv"))),
        "tg_list_file": str(section.get("tg_list_file", data_section.get("tg_list", "data/raw/TG.tsv"))),
        "heatmap_cmap": str(section.get("heatmap_cmap", "OrRd")),
        "n_topics": int(section.get("n_topics", 12)),
        "random_state": int(section.get("random_state", 42)),
        "lda_max_iter": int(section.get("lda_max_iter", 300)),
        "lda_learning_method": str(section.get("lda_learning_method", "batch")),
        "save_fig": bool(section.get("save_fig", True)),
        "do_enrich": bool(section.get("do_enrich", True)),
        "enrich_db": str(section.get("enrich_db", "GO_Biological_Process_2021")),
        "topk_edges_per_topic_for_enrich": int(section.get("topk_edges_per_topic_for_enrich", 500)),
        "topk_terms_per_topic_bar": int(section.get("topk_terms_per_topic_bar", 10)),
        "topic_topk_mix": int(section.get("topic_topk_mix", 3)),
        "topic_min_max_prop": float(section.get("topic_min_max_prop", 0.45)),
        "topic_min_gap": float(section.get("topic_min_gap", 0.15)),
        "save_single_enrich_png": bool(section.get("save_single_enrich_png", True)),
        "enrich_grid_ncols": int(section.get("enrich_grid_ncols", 3)),
        "enrich_grid_per_page": int(section.get("enrich_grid_per_page", 12)),
        "enrich_grid_figsize": (float(figsize[0]), float(figsize[1])),
        "enrich_grid_dpi": int(section.get("enrich_grid_dpi", 300)),
    }


def _cluster_labels(settings: dict[str, Any], config: dict[str, Any], clusters: list[int]) -> list[str]:
    if settings["celltypes"] is not None and len(settings["celltypes"]) == len(clusters):
        return settings["celltypes"]
    lineage = str(config.get("lineage", "")).lower()
    if lineage in {"germ", "germline"} and len(clusters) == len(DEFAULT_GERM_CELLTYPES):
        return DEFAULT_GERM_CELLTYPES
    return [f"t{cluster:02d}" for cluster in clusters]


def plot_topic_cellcluster_heatmap(
    h_matrix: np.ndarray,
    clusters: list[int],
    x_labels: list[str],
    path: str | Path,
    cmap_name: str,
) -> str:
    with np.errstate(divide="ignore", invalid="ignore"):
        row_sum = h_matrix.sum(axis=1, keepdims=True)
        h_prop = np.divide(h_matrix, np.maximum(row_sum, 1e-12))
    y_labels = [f"T{topic}" for topic in range(h_matrix.shape[0])]

    fig_width = max(7.0, 0.75 * len(x_labels)) + 1.0
    fig_height = max(5.0, 0.55 * len(y_labels))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(h_prop, aspect="auto", interpolation="nearest", cmap=cmap_name)
    ax.set_xticks(np.arange(len(clusters)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=12)
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=12)
    ax.set_title("Topic Analysis", fontsize=16)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("proportion")
    colorbar.ax.tick_params(labelsize=11)
    fig.tight_layout()

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def _safe_label(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_-]+", "_", str(value))


def _read_ordered_gene_names(path: str | Path) -> list[str]:
    frame = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    name_col = "gene_name" if "gene_name" in frame.columns else frame.columns[0]
    return pd.unique(frame[name_col].astype(str).map(strip_copy_suffix)).tolist()


def _remap_tf_tg_names_from_tables(
    df_edges: pd.DataFrame,
    tf_path: str | Path,
    tg_path: str | Path,
) -> pd.DataFrame:
    frame = df_edges.copy()
    if {"tf_name", "tg_name"}.issubset(frame.columns):
        tf_has_names = frame["tf_name"].astype(str).str.strip().ne("").all()
        tg_has_names = frame["tg_name"].astype(str).str.strip().ne("").all()
        if tf_has_names and tg_has_names:
            frame["tf_name"] = frame["tf_name"].astype(str).map(strip_copy_suffix)
            frame["tg_name"] = frame["tg_name"].astype(str).map(strip_copy_suffix)
            return frame

    frame["tf_idx"] = pd.to_numeric(frame["tf_idx"], errors="coerce")
    frame["tg_idx"] = pd.to_numeric(frame["tg_idx"], errors="coerce")
    frame = frame.dropna(subset=["tf_idx", "tg_idx"]).copy()
    frame["tf_idx"] = frame["tf_idx"].astype(int)
    frame["tg_idx"] = frame["tg_idx"].astype(int)

    tf_names = _read_ordered_gene_names(tf_path)
    tg_names = _read_ordered_gene_names(tg_path)
    idx2tf_name = dict(enumerate(tf_names))
    idx2tg_name = dict(enumerate(tg_names))

    valid_tf_idx = set(idx2tf_name.keys())
    valid_tg_idx = set(idx2tg_name.keys())
    frame = frame[frame["tf_idx"].isin(valid_tf_idx)].copy()
    frame = frame[frame["tg_idx"].isin(valid_tg_idx)].copy()
    if frame.empty:
        return frame

    if "tf_name" not in frame.columns:
        frame["tf_name"] = ""
    if "tg_name" not in frame.columns:
        frame["tg_name"] = ""
    tf_missing = frame["tf_name"].astype(str).str.strip().eq("")
    tg_missing = frame["tg_name"].astype(str).str.strip().eq("")
    frame.loc[tf_missing, "tf_name"] = frame.loc[tf_missing, "tf_idx"].map(idx2tf_name)
    frame.loc[tg_missing, "tg_name"] = frame.loc[tg_missing, "tg_idx"].map(idx2tg_name)
    frame["tf_name"] = frame["tf_name"].astype(str).map(strip_copy_suffix)
    frame["tg_name"] = frame["tg_name"].astype(str).map(strip_copy_suffix)
    return frame


def _prep_enrich_df(enrich_df: pd.DataFrame, top_terms: int) -> pd.DataFrame:
    if enrich_df is None or enrich_df.empty:
        return pd.DataFrame()

    frame = enrich_df.copy()
    pval_col = None
    for candidate in ["Adjusted P-value", "Adjusted P-value ", "Adjusted P value", "adj_pvalue", "FDR"]:
        if candidate in frame.columns:
            pval_col = candidate
            break
    if pval_col is None:
        lower = {str(column).lower(): str(column) for column in frame.columns}
        for candidate in ["adjusted p-value", "adjusted p value", "adj_pvalue", "fdr"]:
            if candidate in lower:
                pval_col = lower[candidate]
                break
    if pval_col is None:
        return pd.DataFrame()

    term_col = "Term" if "Term" in frame.columns else frame.columns[0]
    numeric = pd.to_numeric(frame[pval_col], errors="coerce")
    score = -np.log10(np.maximum(numeric.to_numpy(dtype=float), 1e-300))
    prepared = pd.DataFrame({"Term": frame[term_col].astype(str), "score": score})
    prepared = prepared.replace([np.inf, -np.inf], np.nan).dropna().sort_values("score", ascending=False)
    return prepared.head(top_terms).reset_index(drop=True)


def infer_topic_celltype_pref(
    h_matrix: np.ndarray,
    clusters: list[int],
    cid2celltype: dict[int, str] | None = None,
    topk: int = 3,
    min_max_prop: float = 0.45,
    min_gap: float = 0.15,
) -> dict[int, str]:
    with np.errstate(divide="ignore", invalid="ignore"):
        row_sum = h_matrix.sum(axis=1, keepdims=True)
        h_prop = np.divide(h_matrix, np.maximum(row_sum, 1e-12))

    if cid2celltype is not None:
        x_labels = [cid2celltype.get(cluster, f"t{cluster:02d}") for cluster in clusters]
    else:
        x_labels = [f"t{cluster:02d}" for cluster in clusters]

    n_topics, n_clusters = h_prop.shape
    keep_topk = min(int(topk), n_clusters)
    top_idx = np.argsort(-h_prop, axis=1)[:, :keep_topk]
    top_props = np.take_along_axis(h_prop, top_idx, axis=1)
    max_prop = top_props[:, 0]
    second_prop = top_props[:, 1] if keep_topk >= 2 else np.zeros(n_topics)
    gap = max_prop - second_prop
    is_specific = (max_prop >= min_max_prop) & (gap >= min_gap)

    topic2title: dict[int, str] = {}
    for topic in range(n_topics):
        if is_specific[topic]:
            topic2title[topic] = f"T{topic} | {x_labels[top_idx[topic, 0]]} ({max_prop[topic]:.2f})"
        else:
            pairs = [f"{x_labels[top_idx[topic, rank]]}:{top_props[topic, rank]:.2f}" for rank in range(keep_topk)]
            topic2title[topic] = f"T{topic} | mixed: " + ", ".join(pairs)
    return topic2title


def draw_enrichment_bars(
    enrich_df: pd.DataFrame,
    topic_id: int,
    path: str | Path,
    db_name: str,
    top_terms: int,
) -> str | None:
    prepared = _prep_enrich_df(enrich_df, top_terms)
    if prepared.empty:
        return None

    fig_height = max(6.0, 0.62 * len(prepared))
    fig, ax = plt.subplots(figsize=(6, fig_height))
    y_pos = np.arange(len(prepared))
    ax.barh(y_pos, prepared["score"].to_numpy())
    ax.set_yticks(y_pos)
    ax.set_yticklabels(prepared["Term"].tolist(), fontsize=14)
    ax.invert_yaxis()
    ax.set_xlabel("-log10(adj p)", fontsize=16)
    ax.set_title(f"Topic T{topic_id} - {db_name}", fontsize=18, pad=8)
    ax.tick_params(axis="x", labelsize=14)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def plot_enrichment_grid(
    enrich_df_by_topic: dict[int, pd.DataFrame],
    n_topics: int,
    db_name: str,
    topic2title: dict[int, str] | None,
    output_dir: Path,
    top_terms: int,
    ncols: int,
    per_page: int,
    figsize: tuple[float, float],
    dpi: int,
) -> list[str]:
    nrows = int(math.ceil(per_page / ncols))
    pages = int(math.ceil(n_topics / per_page))
    safe_db = _safe_label(db_name)
    saved_paths: list[str] = []

    for page in range(pages):
        start = page * per_page
        end = min((page + 1) * per_page, n_topics)
        topic_ids = list(range(start, end))

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=figsize,
            gridspec_kw={"wspace": 1.15, "hspace": 0.55},
        )
        axes_arr = np.asarray(axes).ravel()

        for axis_idx, ax in enumerate(axes_arr):
            if axis_idx >= len(topic_ids):
                ax.axis("off")
                continue

            topic_id = topic_ids[axis_idx]
            prepared = _prep_enrich_df(enrich_df_by_topic.get(topic_id, pd.DataFrame()), top_terms)
            title = topic2title.get(topic_id, f"T{topic_id}") if topic2title is not None else f"T{topic_id}"

            if prepared.empty:
                ax.axis("off")
                ax.text(0.5, 0.5, title + "\nNo results", ha="center", va="center", fontsize=20)
                continue

            y_pos = np.arange(len(prepared))
            ax.barh(y_pos, prepared["score"].to_numpy())
            ax.set_yticks(y_pos)
            ax.set_yticklabels(prepared["Term"].tolist(), fontsize=13)
            ax.tick_params(axis="y", labelsize=13, pad=2)
            ax.invert_yaxis()
            ax.set_title(title, fontsize=20, pad=6)
            ax.tick_params(axis="x", labelsize=13)
            ax.grid(axis="x", linestyle="--", alpha=0.25)
            if axis_idx // ncols == nrows - 1:
                ax.set_xlabel("-log10(adj p)", fontsize=13)
            else:
                ax.set_xlabel("")
            max_score = float(np.nanmax(prepared["score"].to_numpy()))
            ax.set_xlim(0, max_score * 1.08 if max_score > 0 else 1.0)

        fig.suptitle(f"Enrichment per Topic - {db_name} (page {page + 1}/{pages})", fontsize=22)
        fig.subplots_adjust(left=0.15, right=0.985, top=0.90, bottom=0.10)
        out_png = output_dir / f"topic_enrich_grid_{nrows}x{ncols}_{safe_db}_page{page + 1:02d}.png"
        fig.savefig(out_png, dpi=dpi)
        plt.close(fig)
        saved_paths.append(str(out_png))

    return saved_paths


def run_enrichment_per_topic(
    w_matrix: np.ndarray,
    h_matrix: np.ndarray,
    clusters: list[int],
    edge_keys: list[tuple[Any, Any]],
    df_edges_all: pd.DataFrame,
    output_dir: Path,
    settings: dict[str, Any],
    cid2celltype: dict[int, str] | None = None,
) -> dict[str, Any]:
    try:
        import gseapy as gp
    except Exception as exc:
        return {
            "enabled": False,
            "status": "skipped",
            "reason": f"gseapy unavailable: {exc}",
            "csv_paths": [],
            "png_paths": [],
        }

    tmp = (
        df_edges_all.groupby(["tf_idx", "tg_idx", "tg_name"])
        .size()
        .reset_index(name="cnt")
        .sort_values(["tf_idx", "tg_idx", "cnt"], ascending=[True, True, False], kind="mergesort")
    )
    key_to_tgname = {
        (int(row.tf_idx), int(row.tg_idx)): str(row.tg_name)
        for _, row in tmp.drop_duplicates(subset=["tf_idx", "tg_idx"]).iterrows()
    }

    def tg_name_from_key(tf_idx: int, tg_idx: int) -> str:
        return key_to_tgname.get((tf_idx, tg_idx), f"TG{tg_idx}")

    topic2title = infer_topic_celltype_pref(
        h_matrix,
        clusters,
        cid2celltype=cid2celltype,
        topk=settings["topic_topk_mix"],
        min_max_prop=settings["topic_min_max_prop"],
        min_gap=settings["topic_min_gap"],
    )

    n_topics = w_matrix.shape[1]
    safe_db = _safe_label(settings["enrich_db"])
    k_edges = int(min(settings["topk_edges_per_topic_for_enrich"], w_matrix.shape[0]))
    enrich_df_by_topic: dict[int, pd.DataFrame] = {}
    csv_paths: list[str] = []
    png_paths: list[str] = []

    for topic in range(n_topics):
        idxs = np.argsort(-w_matrix[:, topic])[:k_edges]
        tg_names = sorted({tg_name_from_key(int(edge_keys[idx][0]), int(edge_keys[idx][1])) for idx in idxs})
        if len(tg_names) < 3:
            enrich_df_by_topic[topic] = pd.DataFrame()
            continue

        try:
            enr = gp.enrichr(
                gene_list=tg_names,
                gene_sets=[settings["enrich_db"]],
                outdir=None,
                no_plot=True,
                cutoff=1.0,
            )
            results = enr.results.copy() if isinstance(enr.results, pd.DataFrame) else list(enr.results.values())[0]
        except Exception:
            enrich_df_by_topic[topic] = pd.DataFrame()
            continue

        enrich_df_by_topic[topic] = results
        csv_path = output_dir / f"topic_T{topic}_enrich_{safe_db}.csv"
        results.to_csv(csv_path, index=False)
        csv_paths.append(str(csv_path))

        if settings["save_fig"] and settings["save_single_enrich_png"]:
            png_path = draw_enrichment_bars(
                results,
                topic_id=topic,
                path=output_dir / f"topic_T{topic}_enrich_{safe_db}.png",
                db_name=settings["enrich_db"],
                top_terms=settings["topk_terms_per_topic_bar"],
            )
            if png_path is not None:
                png_paths.append(png_path)

    if settings["save_fig"]:
        grid_paths = plot_enrichment_grid(
            enrich_df_by_topic=enrich_df_by_topic,
            n_topics=n_topics,
            db_name=settings["enrich_db"],
            topic2title=topic2title,
            output_dir=output_dir,
            top_terms=settings["topk_terms_per_topic_bar"],
            ncols=settings["enrich_grid_ncols"],
            per_page=settings["enrich_grid_per_page"],
            figsize=settings["enrich_grid_figsize"],
            dpi=settings["enrich_grid_dpi"],
        )
        png_paths.extend(grid_paths)

    return {
        "enabled": True,
        "status": "completed",
        "csv_paths": csv_paths,
        "png_paths": png_paths,
    }


def run_topic_analysis_from_config(config: dict[str, Any]) -> dict[str, Any]:
    settings = _topic_settings(config)
    output_dir = ensure_directory(resolve_config_path(config, settings["output_dir"]))
    network_dir = resolve_config_path(config, settings["network_dir"])
    clusters = settings["clusters"] or discover_network_time_indices(network_dir)
    cluster_labels = _cluster_labels(settings, config, clusters)
    cid2celltype = {cluster: label for cluster, label in zip(clusters, cluster_labels)}

    df_edges = build_edge_dataframe(
        network_dir=network_dir,
        tids=clusters,
        mode=settings["selection_mode"],
        top_n=settings["top_n_per_cluster"],
        thr=settings["score_threshold"],
    )
    df_edges = _remap_tf_tg_names_from_tables(
        df_edges,
        tf_path=resolve_config_path(config, settings["tf_list_file"]),
        tg_path=resolve_config_path(config, settings["tg_list_file"]),
    )
    if df_edges.empty:
        raise ValueError(f"No edges were selected from {network_dir} with mode={settings['selection_mode']}.")

    counts = (
        df_edges.groupby(["tf_idx", "tg_idx", "cell_cluster"])
        .size()
        .reset_index(name="count")
    )
    edge_keys = pd.MultiIndex.from_frame(counts[["tf_idx", "tg_idx"]]).unique().tolist()
    edge_to_row = {edge: idx for idx, edge in enumerate(edge_keys)}
    cluster_to_col = {cluster: idx for idx, cluster in enumerate(clusters)}
    row_idx = counts.apply(lambda row: edge_to_row[(row["tf_idx"], row["tg_idx"])], axis=1).to_numpy(dtype=int)
    col_idx = counts["cell_cluster"].map(cluster_to_col).to_numpy(dtype=int)
    data = counts["count"].to_numpy(dtype=float)
    c_sparse = coo_matrix((data, (row_idx, col_idx)), shape=(len(edge_keys), len(clusters))).tocsr()

    lda = LatentDirichletAllocation(
        n_components=settings["n_topics"],
        learning_method=settings["lda_learning_method"],
        max_iter=settings["lda_max_iter"],
        random_state=settings["random_state"],
        evaluate_every=-1,
        verbose=0,
    )
    w_matrix = lda.fit_transform(c_sparse)
    h_matrix = lda.components_

    h_df = pd.DataFrame(h_matrix, index=[f"T{topic}" for topic in range(h_matrix.shape[0])], columns=[f"c{cluster}" for cluster in clusters])
    h_df.to_csv(output_dir / "topic_x_cluster_lda.csv")
    with np.errstate(divide="ignore", invalid="ignore"):
        h_prop = np.divide(h_matrix, np.maximum(h_matrix.sum(axis=1, keepdims=True), 1e-12))
    pd.DataFrame(h_prop, index=h_df.index, columns=h_df.columns).to_csv(output_dir / "topic_x_cluster_prop_lda.csv")

    heatmap_path = None
    if settings["save_fig"]:
        heatmap_path = plot_topic_cellcluster_heatmap(
            h_matrix,
            clusters=clusters,
            x_labels=cluster_labels,
            path=output_dir / "topic_x_celltype_heatmap_lda.png",
            cmap_name=settings["heatmap_cmap"],
        )

    enrichment_report = {
        "enabled": False,
        "status": "skipped",
        "csv_paths": [],
        "png_paths": [],
    }
    if settings["do_enrich"]:
        enrichment_report = run_enrichment_per_topic(
            w_matrix=w_matrix,
            h_matrix=h_matrix,
            clusters=clusters,
            edge_keys=edge_keys,
            df_edges_all=df_edges,
            output_dir=output_dir,
            settings=settings,
            cid2celltype=cid2celltype,
        )

    report = {
        "analysis": "topic",
        "output_dir": str(output_dir),
        "network_dir": str(network_dir),
        "clusters": clusters,
        "cluster_labels": cluster_labels,
        "selection_mode": settings["selection_mode"],
        "selected_edge_count": int(len(df_edges)),
        "n_topics": int(settings["n_topics"]),
        "topic_x_cluster_path": str(output_dir / "topic_x_cluster_lda.csv"),
        "topic_x_cluster_prop_path": str(output_dir / "topic_x_cluster_prop_lda.csv"),
        "heatmap_path": heatmap_path,
        "enrichment": enrichment_report,
        "settings": settings,
    }
    save_checkpoint_metadata(output_dir / "topic_report.json", report)
    return report


def run_topic_analysis(config: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(config, (str, Path)):
        loaded = load_yaml_config(config)
        report = run_topic_analysis_from_config(loaded)
        report["config_path"] = str(Path(config).resolve())
        return report
    return run_topic_analysis_from_config(config)
