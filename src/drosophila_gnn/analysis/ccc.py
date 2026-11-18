"""Cell-cell communication: receptor scores, DTW alignment and LR timing."""

from __future__ import annotations

import glob
import os
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import entropy
from sklearn.cluster import KMeans

from ..checkpoint import save_checkpoint_metadata
from ..utils import ensure_directory, load_yaml_config

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


EDGE_GLOBS = (
    "TF_TG_dense_t*.tsv",
    "TF_TG_globalTopK_rescue_dense_*.tsv",
    "TF_TG_top*.tsv",
    "TF_TG_*.tsv",
    "t*_TF-TG.tsv",
)


def _section(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    return value if isinstance(value, dict) else {}


def _resolve_path(base_dir: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _ccc_settings(config: dict[str, Any]) -> dict[str, Any]:
    section = _section(config, "ccc")
    data_cfg = _section(config, "data")
    return {
        "seed": int(section.get("seed", 4122)),
        "output_dir": str(section.get("output_dir", "result/ccc")),
        "network_dir_soma": str(section.get("network_dir_soma", "result/soma/network_dense")),
        "network_dir_germ": str(section.get("network_dir_germ", "result/germ/network_dense")),
        "tg_annot_soma": str(section.get("tg_annot_soma", data_cfg.get("tg_list", "TG.tsv"))),
        "tg_annot_germ": str(section.get("tg_annot_germ", data_cfg.get("tg_list", "TG.tsv"))),
        "tf_annot_soma": str(section.get("tf_annot_soma", data_cfg.get("tf_list", "TF.tsv"))),
        "tf_annot_germ": str(section.get("tf_annot_germ", data_cfg.get("tf_list", "TF.tsv"))),
        "ccc_csv": str(section.get("ccc_csv", "ccc.csv")),
        "tf_filter_path": str(section.get("tf_filter_path", "TF_filtered.tsv")),
        "sigma_time": float(section.get("sigma_time", 0.7)),
        "finger_r": int(section.get("finger_r", 20)),
        "norm_eps": float(section.get("norm_eps", 1e-8)),
        "use_log1p": bool(section.get("use_log1p", False)),
        "use_clip": bool(section.get("use_clip", False)),
        "clip_q": float(section.get("clip_q", 0.99)),
        "warp_fp_w": float(section.get("warp_fp_w", 1.0)),
        "warp_ent_w": float(section.get("warp_ent_w", 100.0)),
        "warp_lam_d2": float(section.get("warp_lam_d2", 100.0)),
        "warp_steps": int(section.get("warp_steps", 2000)),
        "warp_lr": float(section.get("warp_lr", 1e-3)),
        "dtw_fp_w": float(section.get("dtw_fp_w", 1.0)),
        "dtw_ent_w": float(section.get("dtw_ent_w", 0.1)),
        "dtw_band": section.get("dtw_band", 6),
        "dtw_lam_d2": float(section.get("dtw_lam_d2", 1.0)),
        "dtw_n_iter": int(section.get("dtw_n_iter", 50)),
        "alignment_method": str(section.get("alignment_method", "auto")),
        "min_common_times": int(section.get("min_common_times", 6)),
        "alignment_n_boot": int(section.get("alignment_n_boot", 200)),
        "tie_tol": float(section.get("tie_tol", 1e-6)),
        "prefer_on_tie": str(section.get("prefer_on_tie", "DTW")),
        "corr_thresh": float(section.get("corr_thresh", 0.2)),
        "top_k_prog": int(section.get("top_k_prog", 200)),
        "kmeans_clusters": int(section.get("kmeans_clusters", 3)),
        "kmeans_random_state": int(section.get("kmeans_random_state", 42)),
        "kmeans_n_init": int(section.get("kmeans_n_init", 10)),
        "gap_frac": float(section.get("gap_frac", 0.03)),
        "heatmap_figure": str(section.get("heatmap_figure", "LR_short_axis.png")),
        "peak_table_tsv": str(section.get("peak_table_tsv", "LR_peak_stage_table.tsv")),
        "peak_table_csv": str(section.get("peak_table_csv", "LR_peak_stage_table.csv")),
        "stage_table": str(section.get("stage_table", "stage_boundaries.tsv")),
        "warp_cont_tsv": str(section.get("warp_cont_tsv", "time_warp_continuous_soma_to_germ.tsv")),
        "warp_dtw_tsv": str(section.get("warp_dtw_tsv", "time_warp_dtw_soma_to_germ.tsv")),
        "alignment_bar_figure": str(section.get("alignment_bar_figure", "alignment_method_comparison.png")),
        "alignment_bar_tsv": str(section.get("alignment_bar_tsv", "alignment_method_comparison.tsv")),
        "heatmap_width": float(section.get("heatmap_width", 16.0)),
        "heatmap_height": float(section.get("heatmap_height", 3.5)),
        "group_meta": {
            0: {"name": "GSC/Gb", "color": "#E64B35"},
            1: {"name": "Sp-2,4,8,16", "color": "#4DBBD5"},
            2: {"name": "Spermatocytes", "color": "#00A087"},
        },
    }


def _set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass


def _clean_cols(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output.columns = (
        output.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
        .str.lower()
    )
    return output


def _clean_gene_names(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
        .str.lower()
        .str.replace(r"^(tf:|tg:)", "", regex=True)
    )


def parse_time_from_filename(path: str | Path) -> float:
    stem = Path(path).stem
    match = re.match(r"^TF_TG_dense_t([-+]?[0-9]*\.?[0-9]+)$", stem)
    if match:
        return float(match.group(1))
    match = re.match(r"^t([-+]?[0-9]*\.?[0-9]+)_(?:TF-TG|TF_TG.*)$", stem)
    if match:
        return float(match.group(1))
    label = stem.split("_")[-1]
    match = re.match(r"([-+]?[0-9]*\.?[0-9]+)", label)
    if not match:
        raise ValueError(f"Cannot parse time from filename: {path}")
    return float(match.group(1))


def list_edge_files(network_dir: str | Path) -> list[Path]:
    directory = Path(network_dir)
    for pattern in EDGE_GLOBS:
        files = [Path(path).resolve() for path in glob.glob(str(directory / pattern))]
        if files:
            return sorted(files, key=parse_time_from_filename)
    raise ValueError(f"No recognizable TF-TG files were found under {directory}.")


def gaussian_smooth_over_time(df: pd.DataFrame, times: np.ndarray, sigma: float) -> pd.DataFrame:
    values = df.to_numpy(dtype=float)
    times = np.asarray(times, dtype=float)
    count = len(times)
    diff = times.reshape(count, 1) - times.reshape(1, count)
    kernel = np.exp(-0.5 * (diff / sigma) ** 2)
    kernel = kernel / kernel.sum(axis=1, keepdims=True)
    smoothed = values @ kernel.T
    return pd.DataFrame(smoothed, index=df.index, columns=df.columns)


def collapse_duplicate_timecols(df: pd.DataFrame, ndigits: int = 6) -> tuple[pd.DataFrame, np.ndarray]:
    output = df.copy()
    output.columns = np.round(np.asarray(output.columns, dtype=float), ndigits)
    output = output.T.groupby(level=0).mean().T
    output = output.loc[:, sorted(output.columns)]
    return output, output.columns.to_numpy(dtype=float)


def load_activity_from_grn(network_dir: str | Path, tg_annot_path: str | Path, prob_col: str = "prob") -> tuple[pd.DataFrame, np.ndarray]:
    tg = pd.read_csv(tg_annot_path, sep="\t")
    tg = _clean_cols(tg)
    if "gene_name" not in tg.columns:
        raise ValueError(f"{tg_annot_path} must contain gene_name.")
    if "tg_idx" not in tg.columns:
        tg["tg_idx"] = np.arange(1, len(tg) + 1)
    tg["gene_name"] = _clean_gene_names(tg["gene_name"])
    tg = tg.drop_duplicates(subset=["tg_idx"]).copy()
    tg_idx_to_name = tg.set_index("tg_idx")["gene_name"]
    tg_gene_order = tg["gene_name"].tolist()

    series_list: list[pd.Series] = []
    times: list[float] = []
    for path in list_edge_files(network_dir):
        frame = pd.read_csv(path, sep="\t")
        frame = _clean_cols(frame)
        if prob_col not in frame.columns:
            raise ValueError(f"{path} is missing {prob_col}. Columns: {list(frame.columns)}")
        if "tg" in frame.columns:
            frame["tg"] = _clean_gene_names(frame["tg"])
            grouped = frame.groupby("tg")[prob_col].sum()
            grouped = grouped.reindex(tg_gene_order, fill_value=0.0)
            grouped.index = tg_gene_order
        elif "tg_idx" in frame.columns:
            frame["tg_idx"] = pd.to_numeric(frame["tg_idx"], errors="coerce").astype("Int64")
            grouped = frame.groupby("tg_idx")[prob_col].sum()
            grouped = grouped.reindex(tg_idx_to_name.index, fill_value=0.0)
            grouped.index = tg_idx_to_name.values
        else:
            raise ValueError(f"{path} must contain tg or tg_idx.")
        series_list.append(grouped)
        times.append(parse_time_from_filename(path))

    ordered = sorted(zip(times, series_list), key=lambda item: item[0])
    sorted_times = [item[0] for item in ordered]
    matrix = pd.concat([item[1] for item in ordered], axis=1)
    matrix.columns = sorted_times
    matrix.index = _clean_gene_names(pd.Series(matrix.index, index=matrix.index)).values
    matrix = matrix.groupby(matrix.index).sum().sort_index()
    return matrix, np.asarray(sorted_times, dtype=float)


def load_lr_candidates(ccc_csv: str | Path, act_s: pd.DataFrame, act_g: pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_csv(ccc_csv)
    frame["ligand"] = frame["source_genesymbol"].astype(str).str.lower()
    frame["receptor"] = frame["target_genesymbol"].astype(str).str.lower()
    frame = frame[["ligand", "receptor"]].dropna().drop_duplicates()
    frame = frame[frame["ligand"].isin(act_s.index) & frame["receptor"].isin(act_g.index)]
    return frame.reset_index(drop=True)


def _read_annot(path: str | Path, idx_col: str) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t")
    frame = _clean_cols(frame)
    if "gene_name" not in frame.columns:
        raise ValueError(f"{path} must contain gene_name.")
    if idx_col not in frame.columns:
        frame[idx_col] = np.arange(1, len(frame) + 1)
    frame[idx_col] = pd.to_numeric(frame[idx_col], errors="coerce").astype(int)
    frame["gene_name"] = _clean_gene_names(frame["gene_name"])
    return (
        frame[[idx_col, "gene_name"]]
        .drop_duplicates(subset=[idx_col])
        .set_index(idx_col)
        .sort_index()
    )


def _read_edge_file(path: str | Path, prob_col: str = "prob") -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t")
    frame = _clean_cols(frame)
    if prob_col not in frame.columns:
        for candidate in ["score", "weight", "probability", "p"]:
            if candidate in frame.columns:
                frame = frame.rename(columns={candidate: prob_col})
                break
    if prob_col not in frame.columns:
        raise ValueError(f"{path} is missing {prob_col}.")
    if {"tf", "tg"}.issubset(frame.columns):
        output = frame[["tf", "tg", prob_col]].copy()
        output["tf"] = _clean_gene_names(output["tf"])
        output["tg"] = _clean_gene_names(output["tg"])
        output[prob_col] = pd.to_numeric(output[prob_col], errors="coerce").astype(float)
        return output
    if {"tf_idx", "tg_idx"}.issubset(frame.columns):
        output = frame[["tf_idx", "tg_idx", prob_col]].copy()
        output["tf_idx"] = pd.to_numeric(output["tf_idx"], errors="coerce").astype(int)
        output["tg_idx"] = pd.to_numeric(output["tg_idx"], errors="coerce").astype(int)
        output[prob_col] = pd.to_numeric(output[prob_col], errors="coerce").astype(float)
        return output
    raise ValueError(f"{path} must contain TF/TG or tf_idx/tg_idx.")


def _load_network_series(network_dir: str | Path, tf_annot_path: str | Path, tg_annot_path: str | Path, prob_col: str = "prob") -> tuple[np.ndarray, list[Any], list[str], list[str]]:
    tf = _read_annot(tf_annot_path, "tf_idx")
    tg = _read_annot(tg_annot_path, "tg_idx")
    tf_name_to_row = {name: idx for idx, name in enumerate(tf["gene_name"].tolist())}
    tg_name_to_col = {name: idx for idx, name in enumerate(tg["gene_name"].tolist())}
    tf_idx_to_row = {idx: pos for pos, idx in enumerate(tf.index.tolist())}
    tg_idx_to_col = {idx: pos for pos, idx in enumerate(tg.index.tolist())}

    mats: list[Any] = []
    times: list[float] = []
    for path in list_edge_files(network_dir):
        time_value = parse_time_from_filename(path)
        edges = _read_edge_file(path, prob_col=prob_col)
        if {"tf", "tg"}.issubset(edges.columns):
            edges = edges[edges["tf"].isin(tf_name_to_row) & edges["tg"].isin(tg_name_to_col)].copy()
            if edges.empty:
                continue
            edges = edges.groupby(["tf", "tg"], as_index=False)[prob_col].sum()
            rows = edges["tf"].map(tf_name_to_row).to_numpy(dtype=int)
            cols = edges["tg"].map(tg_name_to_col).to_numpy(dtype=int)
            vals = edges[prob_col].to_numpy(dtype=np.float32)
        else:
            edges = edges[edges["tf_idx"].isin(tf_idx_to_row) & edges["tg_idx"].isin(tg_idx_to_col)].copy()
            if edges.empty:
                continue
            edges = edges.groupby(["tf_idx", "tg_idx"], as_index=False)[prob_col].sum()
            rows = edges["tf_idx"].map(tf_idx_to_row).to_numpy(dtype=int)
            cols = edges["tg_idx"].map(tg_idx_to_col).to_numpy(dtype=int)
            vals = edges[prob_col].to_numpy(dtype=np.float32)
        shape = (len(tf), len(tg))
        if HAS_SCIPY:
            matrix = sp.coo_matrix((vals, (rows, cols)), shape=shape).tocsr()
            matrix.sum_duplicates()
            matrix.eliminate_zeros()
        else:
            matrix = np.zeros(shape, dtype=np.float32)
            np.add.at(matrix, (rows, cols), vals)
        mats.append(matrix)
        times.append(time_value)

    if len(times) < 2:
        raise ValueError(f"Too few timepoints were loaded from {network_dir}.")
    order = np.argsort(times)
    sorted_times = np.asarray(times, dtype=float)[order]
    sorted_mats = [mats[idx] for idx in order]
    return sorted_times, sorted_mats, tf["gene_name"].tolist(), tg["gene_name"].tolist()


def _restrict_to_common(mats: list[Any], tf_names: list[str], tg_names: list[str], tf_common: list[str], tg_common: list[str]) -> list[Any]:
    tf_idx = {name: idx for idx, name in enumerate(tf_names)}
    tg_idx = {name: idx for idx, name in enumerate(tg_names)}
    row_idx = np.asarray([tf_idx[name] for name in tf_common], dtype=int)
    col_idx = np.asarray([tg_idx[name] for name in tg_common], dtype=int)
    output: list[Any] = []
    for matrix in mats:
        if HAS_SCIPY:
            output.append(matrix[row_idx, :][:, col_idx].tocsr())
        else:
            output.append(matrix[np.ix_(row_idx, col_idx)])
    return output


def _apply_log_clip(matrix: Any, settings: dict[str, Any]) -> Any:
    if settings["use_log1p"]:
        if HAS_SCIPY:
            output = matrix.copy()
            output.data = np.log1p(output.data)
            matrix = output
        else:
            matrix = np.log1p(matrix)
    if settings["use_clip"]:
        if HAS_SCIPY:
            q_value = np.quantile(matrix.data, settings["clip_q"]) if matrix.nnz > 0 else 0.0
            output = matrix.copy()
            output.data = np.minimum(output.data, q_value)
            matrix = output
        else:
            q_value = np.quantile(matrix, settings["clip_q"])
            matrix = np.minimum(matrix, q_value)
    return matrix


def _normalize_bipartite(matrix: Any, settings: dict[str, Any]) -> Any:
    matrix = _apply_log_clip(matrix, settings)
    eps = settings["norm_eps"]
    if HAS_SCIPY:
        tf_deg = np.asarray(matrix.sum(axis=1)).ravel() + eps
        tg_deg = np.asarray(matrix.sum(axis=0)).ravel() + eps
        d_tf = sp.diags(1.0 / np.sqrt(tf_deg))
        d_tg = sp.diags(1.0 / np.sqrt(tg_deg))
        output = (d_tf @ matrix @ d_tg).tocsr()
        output.eliminate_zeros()
        return output
    tf_deg = matrix.sum(axis=1) + eps
    tg_deg = matrix.sum(axis=0) + eps
    return (matrix / np.sqrt(tf_deg[:, None])) / np.sqrt(tg_deg[None, :])


def _top_r_svals(matrix: Any, r: int) -> np.ndarray:
    if HAS_SCIPY:
        k = min(r, min(matrix.shape) - 1)
        if k <= 0:
            return np.zeros(r, dtype=np.float32)
        try:
            values = spla.svds(matrix, k=k, which="LM", return_singular_vectors=False)
            values = np.sort(values)[::-1].astype(np.float32)
        except Exception:
            dense = matrix.toarray()
            values = np.linalg.svd(dense, compute_uv=False)[:k].astype(np.float32)
    else:
        values = np.linalg.svd(matrix, compute_uv=False)[: min(r, min(matrix.shape))].astype(np.float32)
    if len(values) < r:
        values = np.pad(values, (0, r - len(values))).astype(np.float32)
    return values[:r]


def _tf_centrality(matrix: Any) -> np.ndarray:
    if HAS_SCIPY:
        return np.asarray(matrix.sum(axis=1)).ravel().astype(np.float32)
    return matrix.sum(axis=1).astype(np.float32)


def _zscore_cols(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    std[std < eps] = 1.0
    return (values - mean) / std


def _zscore_rows(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    mean = values.mean(axis=1, keepdims=True)
    std = values.std(axis=1, keepdims=True)
    std[std < eps] = 1.0
    return (values - mean) / std


def _normalize_time_to_01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return (values - values.min()) / (values.max() - values.min() + 1e-12)


def _get_unique_df(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    output.columns = [float(column) for column in output.columns]
    output = output.T.groupby(level=0).mean().T
    output = output.loc[:, sorted(output.columns)]
    return output


def _calculate_shannon_entropy(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return 0.0
    if np.min(values) < 0:
        values = values - np.min(values)
    total = values.sum()
    if total <= 0:
        return 0.0
    probs = values / total
    return float(entropy(probs))


def _entropy_curve(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    cols = np.asarray([float(column) for column in df.columns], dtype=float)
    values = np.asarray([_calculate_shannon_entropy(df[column].values) for column in cols], dtype=float)
    return cols, values


def _norm01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    v_min = np.min(values)
    v_max = np.max(values)
    if v_max - v_min < 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - v_min) / (v_max - v_min)


def _zscore_np(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return (values - values.mean()) / (values.std() + 1e-12)


def _load_filtered_tfs(path: str | Path) -> list[str] | None:
    if not Path(path).exists():
        return None
    frame = pd.read_csv(path, sep="\t")
    frame.columns = frame.columns.astype(str).str.strip().str.lower()
    if "gene_name" not in frame.columns:
        return None
    frame["gene_name"] = frame["gene_name"].astype(str).str.strip().str.lower()
    if "is_tf" in frame.columns:
        keep = frame["is_tf"].astype(str).str.strip().str.lower().isin(["true", "1", "t", "yes"])
        frame = frame.loc[keep].copy()
    return sorted(set(frame["gene_name"]))


def prepare_entropy_on_net_grid(
    c_s_z: pd.DataFrame,
    c_g_z: pd.DataFrame,
    tau_s_net: np.ndarray,
    tau_g_net: np.ndarray,
    tf_filter_path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    soma_df = _get_unique_df(c_s_z)
    germ_df = _get_unique_df(c_g_z)
    soma_df.index = [str(item).strip().lower() for item in soma_df.index]
    germ_df.index = [str(item).strip().lower() for item in germ_df.index]
    common_tfs = sorted(set(soma_df.index) & set(germ_df.index))
    if not common_tfs:
        raise ValueError("C_s_z and C_g_z do not share any TFs.")
    soma_df = soma_df.loc[common_tfs]
    germ_df = germ_df.loc[common_tfs]
    filtered_tfs = _load_filtered_tfs(tf_filter_path)
    if filtered_tfs is not None:
        keep = sorted(set(filtered_tfs) & set(common_tfs))
        if keep:
            soma_df = soma_df.loc[keep]
            germ_df = germ_df.loc[keep]
    t_s_raw, ent_s = _entropy_curve(soma_df)
    t_g_raw, ent_g = _entropy_curve(germ_df)
    t_s_norm = _norm01(t_s_raw)
    t_g_norm = _norm01(t_g_raw)
    tau_s_net = np.asarray(tau_s_net, dtype=float)
    tau_g_net = np.asarray(tau_g_net, dtype=float)
    ent_s_on_s_net = np.interp(tau_s_net, t_s_norm, ent_s)
    ent_g_on_g_net = np.interp(tau_g_net, t_g_norm, ent_g)
    return _zscore_np(ent_s_on_s_net), _zscore_np(ent_g_on_g_net)


def interp_df_over_time(df: pd.DataFrame, times: np.ndarray, query_times: np.ndarray) -> pd.DataFrame:
    times = np.asarray(times, dtype=float)
    query_times = np.asarray(query_times, dtype=float)
    order = np.argsort(times)
    times = times[order]
    values = df.values[:, order]
    query_clip = np.clip(query_times, times[0], times[-1])
    idx = np.searchsorted(times, query_clip, side="left")
    idx = np.clip(idx, 1, len(times) - 1)
    t0, t1 = times[idx - 1], times[idx]
    y0, y1 = values[:, idx - 1], values[:, idx]
    weight = (query_clip - t0) / (t1 - t0 + 1e-12)
    out = y0 + (y1 - y0) * weight.reshape(1, -1)
    return pd.DataFrame(out, index=df.index, columns=query_times)


def interp_df_time(df: pd.DataFrame, times: np.ndarray, query_times: np.ndarray) -> pd.DataFrame:
    return interp_df_over_time(df, times, query_times)


def learn_monotone_warp_continuous(
    tau_s: np.ndarray,
    fs: np.ndarray,
    tau_g: np.ndarray,
    fg: np.ndarray,
    ent_s: np.ndarray | None = None,
    ent_g: np.ndarray | None = None,
    fp_w: float = 1.0,
    ent_w: float = 0.0,
    lam_d2: float = 1e-2,
    steps: int = 2000,
    lr: float = 1e-3,
) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tau_g_t = torch.tensor(tau_g, dtype=torch.float32, device=device)
    fs_t = torch.tensor(fs, dtype=torch.float32, device=device)
    fg_t = torch.tensor(fg, dtype=torch.float32, device=device)
    ent_s_t = None if ent_s is None else torch.tensor(ent_s, dtype=torch.float32, device=device)
    ent_g_t = None if ent_g is None else torch.tensor(ent_g, dtype=torch.float32, device=device)

    def zscore_torch(values: torch.Tensor) -> torch.Tensor:
        return (values - values.mean()) / (values.std(unbiased=False) + 1e-12)

    def pearson_corr_torch(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        left = left - left.mean()
        right = right - right.mean()
        num = (left * right).mean()
        den = left.std(unbiased=False) * right.std(unbiased=False) + 1e-12
        return num / den

    def interp_g(query: torch.Tensor) -> torch.Tensor:
        idx = torch.searchsorted(tau_g_t, query).clamp(1, len(tau_g_t) - 1)
        u0, u1 = tau_g_t[idx - 1], tau_g_t[idx]
        y0, y1 = fg_t[idx - 1], fg_t[idx]
        weight = (query - u0) / (u1 - u0 + 1e-12)
        return y0 + (y1 - y0) * weight.unsqueeze(-1)

    def interp_ent_g(query: torch.Tensor) -> torch.Tensor:
        idx = torch.searchsorted(tau_g_t, query).clamp(1, len(tau_g_t) - 1)
        u0, u1 = tau_g_t[idx - 1], tau_g_t[idx]
        y0, y1 = ent_g_t[idx - 1], ent_g_t[idx]
        weight = (query - u0) / (u1 - u0 + 1e-12)
        return y0 + (y1 - y0) * weight

    count = len(tau_s)
    alpha = torch.nn.Parameter(torch.zeros(count - 1, device=device))
    optimizer = torch.optim.Adam([alpha], lr=lr)
    eps = 1e-6
    for _ in range(steps):
        delta = F.softplus(alpha) + eps
        warp = torch.cat([torch.zeros(1, device=device), torch.cumsum(delta, dim=0)], dim=0)
        warp = warp / (warp[-1] + 1e-12)
        g_at_w = interp_g(warp)
        fit_fp = ((fs_t - g_at_w) ** 2).sum(dim=1).mean()
        if ent_s_t is not None and ent_g_t is not None:
            ent_g_at_w = interp_ent_g(warp)
            ent_loss = 1.0 - pearson_corr_torch(zscore_torch(ent_s_t), zscore_torch(ent_g_at_w))
        else:
            ent_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        if count >= 3:
            smooth_d2 = ((warp[2:] - 2 * warp[1:-1] + warp[:-2]) ** 2).mean()
        else:
            smooth_d2 = torch.tensor(0.0, dtype=torch.float32, device=device)
        loss = fp_w * fit_fp + ent_w * ent_loss + lam_d2 * smooth_d2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return warp.detach().cpu().numpy()


def isotonic_increasing(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    n0 = len(values)
    levels = values.copy()
    weights = np.ones(n0, dtype=float)
    idx = 0
    while idx < len(levels) - 1:
        if levels[idx] > levels[idx + 1]:
            new_level = (levels[idx] * weights[idx] + levels[idx + 1] * weights[idx + 1]) / (weights[idx] + weights[idx + 1])
            levels[idx] = new_level
            weights[idx] = weights[idx] + weights[idx + 1]
            levels = np.delete(levels, idx + 1)
            weights = np.delete(weights, idx + 1)
            if idx > 0:
                idx -= 1
        else:
            idx += 1
    output = np.empty(n0, dtype=float)
    start = 0
    for level, weight in zip(levels, weights):
        width = int(round(weight))
        output[start : start + width] = level
        start += width
    return output


def smooth_warp_d2_monotone(
    warp0: np.ndarray,
    lam: float = 10.0,
    n_iter: int = 3,
    clip01: bool = True,
    enforce_endpoints: bool = True,
) -> np.ndarray:
    warp0 = np.asarray(warp0, dtype=float).reshape(-1)
    n = len(warp0)
    if n < 3 or lam <= 0:
        output = warp0.copy()
        if clip01:
            output = np.clip(output, 0.0, 1.0)
        return isotonic_increasing(output)

    d2 = np.zeros((n - 2, n), dtype=float)
    for idx in range(n - 2):
        d2[idx, idx] = 1.0
        d2[idx, idx + 1] = -2.0
        d2[idx, idx + 2] = 1.0
    lap = d2.T @ d2

    output = warp0.copy()
    for _ in range(n_iter):
        a_mat = np.eye(n) + lam * lap
        b_vec = warp0.copy()
        if enforce_endpoints:
            a_mat[0, :] = 0.0
            a_mat[0, 0] = 1.0
            b_vec[0] = warp0[0]
            a_mat[-1, :] = 0.0
            a_mat[-1, -1] = 1.0
            b_vec[-1] = warp0[-1]
        output = np.linalg.solve(a_mat, b_vec)
        if clip01:
            output = np.clip(output, 0.0, 1.0)
        output = isotonic_increasing(output)
        if enforce_endpoints:
            output[0] = warp0[0]
            output[-1] = warp0[-1]
            output = isotonic_increasing(output)
        if output[-1] > 1e-12:
            output = output / output[-1]
    return output


def dtw_path(cost: np.ndarray, band: int | None = None) -> list[tuple[int, int]]:
    t_s, t_g = cost.shape
    inf = 1e30
    dp = np.full((t_s + 1, t_g + 1), inf, dtype=np.float64)
    bt = np.zeros((t_s + 1, t_g + 1, 2), dtype=np.int32)
    dp[0, 0] = 0.0
    for i in range(1, t_s + 1):
        j_start, j_end = 1, t_g
        if band is not None:
            j_center = int(round(i * t_g / t_s))
            j_start = max(1, j_center - band)
            j_end = min(t_g, j_center + band)
        for j in range(j_start, j_end + 1):
            current = cost[i - 1, j - 1]
            prevs = [dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1]]
            choice = int(np.argmin(prevs))
            if choice == 0:
                dp[i, j] = prevs[0] + current
                bt[i, j] = (i - 1, j)
            elif choice == 1:
                dp[i, j] = prevs[1] + current
                bt[i, j] = (i, j - 1)
            else:
                dp[i, j] = prevs[2] + current
                bt[i, j] = (i - 1, j - 1)
    i, j = t_s, t_g
    path: list[tuple[int, int]] = []
    while i > 0 or j > 0:
        path.append((i - 1, j - 1))
        i2, j2 = bt[i, j]
        i, j = int(i2), int(j2)
        if i == 0 and j == 0:
            break
    path.reverse()
    return path


def _canon_timecols(df: pd.DataFrame, ndigits: int = 6) -> pd.DataFrame:
    output = df.copy()
    output.columns = np.round(np.asarray(output.columns, dtype=float), ndigits)
    output = output.T.groupby(level=0).mean().T
    cols = np.asarray(sorted(output.columns), dtype=float)
    return output.loc[:, cols]


def _shannon_entropy(values: np.ndarray, eps: float = 1e-12) -> float:
    values = np.asarray(values, float).reshape(-1)
    if values.size == 0:
        return 0.0
    if np.min(values) < 0:
        values = values - np.min(values)
    total = float(np.sum(values))
    if total <= 0:
        return 0.0
    probs = values / (total + eps)
    probs = probs[probs > 0]
    return float(-(probs * np.log(probs + eps)).sum())


def _pearson_corr(left: np.ndarray, right: np.ndarray, eps: float = 1e-12) -> float:
    left = np.asarray(left, float).reshape(-1)
    right = np.asarray(right, float).reshape(-1)
    if left.size < 2 or right.size < 2:
        return 0.0
    left = left - left.mean()
    right = right - right.mean()
    den = (np.linalg.norm(left) * np.linalg.norm(right)) + eps
    return float(left.dot(right) / den)


def entropy_corr_score(
    soma_df: pd.DataFrame,
    other_df: pd.DataFrame,
    min_common_times: int = 6,
    n_boot: int = 200,
    seed: int = 0,
) -> tuple[float, int]:
    common_times = sorted(set(soma_df.columns) & set(other_df.columns))
    count = len(common_times)
    if count < min_common_times:
        return float("nan"), count
    s_curve = np.asarray([_shannon_entropy(soma_df[t].values) for t in common_times], dtype=float)
    o_curve = np.asarray([_shannon_entropy(other_df[t].values) for t in common_times], dtype=float)
    if n_boot <= 1:
        return _pearson_corr(s_curve, o_curve), count
    rng = np.random.default_rng(seed)
    idx = np.arange(count)
    boots = np.empty(n_boot, float)
    for boot_idx in range(n_boot):
        sample = rng.choice(idx, size=count, replace=True)
        boots[boot_idx] = _pearson_corr(s_curve[sample], o_curve[sample])
    return float(np.nanmean(boots)), count


def pick_alignment_method(
    c_s_z: pd.DataFrame,
    c_g_cont_on_s: pd.DataFrame,
    c_g_dtw_on_s: pd.DataFrame,
    min_common_times: int = 6,
    n_boot: int = 200,
    seed_cont: int = 1,
    seed_dtw: int = 2,
    tie_tol: float = 1e-6,
    prefer_on_tie: str = "DTW",
) -> tuple[str, dict[str, Any]]:
    soma_df = _canon_timecols(c_s_z)
    cont_df = _canon_timecols(c_g_cont_on_s)
    dtw_df = _canon_timecols(c_g_dtw_on_s)
    ent_cont, n_cont = entropy_corr_score(soma_df, cont_df, min_common_times=min_common_times, n_boot=n_boot, seed=seed_cont)
    ent_dtw, n_dtw = entropy_corr_score(soma_df, dtw_df, min_common_times=min_common_times, n_boot=n_boot, seed=seed_dtw)
    if not (np.isfinite(ent_cont) and np.isfinite(ent_dtw)):
        raise ValueError(f"Invalid entropy corr values: cont={ent_cont}, dtw={ent_dtw}")
    gap = ent_cont - ent_dtw
    if abs(gap) <= tie_tol:
        chosen = prefer_on_tie
    else:
        chosen = "Warping" if ent_cont > ent_dtw else "DTW"
    metrics = {
        "entropy_corr_cont": float(ent_cont),
        "entropy_corr_dtw": float(ent_dtw),
        "n_common_times_cont": int(n_cont),
        "n_common_times_dtw": int(n_dtw),
        "gap_cont_minus_dtw": float(gap),
        "chosen_method": str(chosen),
        "prefer_on_tie": str(prefer_on_tie),
        "tie_tol": float(tie_tol),
    }
    return chosen, metrics


def _build_alignment_comparison_outputs(
    output_dir: Path,
    settings: dict[str, Any],
    chosen_method: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    rows = [
        {
            "method": "Linear",
            "entropy_corr": float(metrics["entropy_corr_linear"]),
            "n_common_times": int(metrics["n_common_times_linear"]),
            "used_for_choice": False,
            "chosen": False,
        },
        {
            "method": "Warping",
            "entropy_corr": float(metrics["entropy_corr_cont"]),
            "n_common_times": int(metrics["n_common_times_cont"]),
            "used_for_choice": True,
            "chosen": str(chosen_method) == "Warping",
        },
        {
            "method": "DTW",
            "entropy_corr": float(metrics["entropy_corr_dtw"]),
            "n_common_times": int(metrics["n_common_times_dtw"]),
            "used_for_choice": True,
            "chosen": str(chosen_method) == "DTW",
        },
    ]
    table = pd.DataFrame(rows)
    table_path = output_dir / settings["alignment_bar_tsv"]
    table.to_csv(table_path, sep="\t", index=False)

    colors = ["#BDBDBD", "#4DBBD5", "#E64B35"]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bars = ax.bar(table["method"], table["entropy_corr"], color=colors, edgecolor="black", linewidth=0.8)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("Entropy Correlation", fontsize=12)
    ax.set_title("Alignment Method Comparison", fontsize=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    values = table["entropy_corr"].to_numpy(dtype=float)
    span = max(np.nanmax(np.abs(values)), 1e-6)
    offset = max(span * 0.04, 0.01)
    for bar, row in zip(bars, rows):
        value = float(row["entropy_corr"])
        y_pos = value + offset if value >= 0 else value - offset
        va = "bottom" if value >= 0 else "top"
        label = f"{value:.3f}"
        if row["chosen"]:
            label = f"{label}\nchosen"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            y_pos,
            label,
            ha="center",
            va=va,
            fontsize=10,
            fontweight="bold" if row["chosen"] else "normal",
        )

    fig.tight_layout()
    figure_path = output_dir / settings["alignment_bar_figure"]
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "figure_path": str(figure_path),
        "table_path": str(table_path),
        "methods": rows,
    }


def compute_receptor_programs(act_g: pd.DataFrame, receptors: list[str], corr_thresh: float = 0.6, top_k: int | None = 200) -> dict[str, list[str]]:
    genes = list(act_g.index)
    gene_to_idx = {gene: idx for idx, gene in enumerate(genes)}
    values = act_g.values
    z_values = values - values.mean(axis=1, keepdims=True)
    std = z_values.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    z_values = z_values / std
    programs: dict[str, list[str]] = {}
    for receptor in receptors:
        if receptor not in gene_to_idx:
            continue
        idx_r = gene_to_idx[receptor]
        r_vec = z_values[idx_r]
        corr = (z_values * r_vec).mean(axis=1)
        corr[idx_r] = 0.0
        if top_k is not None:
            idx = np.argsort(-np.abs(corr))[:top_k]
            if corr_thresh is not None:
                idx = idx[np.abs(corr[idx]) >= corr_thresh]
        elif corr_thresh is not None:
            idx = np.where(np.abs(corr) >= corr_thresh)[0]
        else:
            idx = np.arange(len(genes))
        prog_genes = [genes[item] for item in idx]
        if prog_genes:
            programs[receptor] = prog_genes
    return programs


def compute_program_curves(programs: dict[str, list[str]], act_g: pd.DataFrame) -> pd.DataFrame:
    payload: dict[str, np.ndarray] = {}
    for receptor, genes in programs.items():
        genes_ok = [gene for gene in genes if gene in act_g.index]
        if not genes_ok:
            continue
        payload[receptor] = act_g.loc[genes_ok].mean(axis=0).values
    return pd.DataFrame.from_dict(payload, orient="index", columns=act_g.columns)


def compute_prog_comm_curves(
    lr_df: pd.DataFrame,
    act_s: pd.DataFrame,
    prog_df: pd.DataFrame,
) -> tuple[dict[tuple[str, str], np.ndarray], pd.DataFrame, np.ndarray]:
    times = np.asarray(act_s.columns, dtype=float)
    prog_reindexed = prog_df.reindex(columns=act_s.columns)
    if prog_reindexed.isna().any().any():
        missing = act_s.columns[prog_reindexed.isna().all(axis=0)]
        raise ValueError(f"prog_df is missing aligned columns: {missing[:10].tolist()}")
    curves: dict[tuple[str, str], np.ndarray] = {}
    records: list[dict[str, Any]] = []
    for _, row in lr_df.iterrows():
        ligand = row["ligand"]
        receptor = row["receptor"]
        if ligand not in act_s.index or receptor not in prog_reindexed.index:
            continue
        ligand_curve = act_s.loc[ligand].values.astype(float)
        program_curve = prog_reindexed.loc[receptor].values.astype(float)
        comm = ligand_curve * program_curve
        curves[(ligand, receptor)] = comm
        records.append({"ligand": ligand, "receptor": receptor, "score_prog": float(comm.sum())})
    scores_df = pd.DataFrame(records).sort_values("score_prog", ascending=False).reset_index(drop=True)
    return curves, scores_df, times


def build_receptor_tfs_from_grn(
    network_dir_germ: str | Path,
    tf_annot_path: str | Path,
    tg_annot_path: str | Path,
    programs: dict[str, list[str]],
) -> dict[str, str]:
    tf = _read_annot(tf_annot_path, "tf_idx")
    tg = _read_annot(tg_annot_path, "tg_idx")
    tf_map = dict(zip(tf.index, tf["gene_name"]))
    tg_map = dict(zip(tg.index, tg["gene_name"]))
    tg_to_tfs: dict[str, set[str]] = defaultdict(set)
    for path in list_edge_files(network_dir_germ):
        frame = _read_edge_file(path, prob_col="prob")
        if {"tf", "tg"}.issubset(frame.columns):
            tf_names = frame["tf"]
            tg_names = frame["tg"]
        else:
            tf_names = frame["tf_idx"].map(tf_map)
            tg_names = frame["tg_idx"].map(tg_map)
        mask = tf_names.notna() & tg_names.notna()
        for tf_name, tg_name in zip(tf_names[mask].values, tg_names[mask].values):
            tg_to_tfs[str(tg_name)].add(str(tf_name))
    receptor_to_tfs: dict[str, str] = {}
    for receptor, genes in programs.items():
        tf_set: set[str] = set()
        for gene in genes:
            tf_set |= tg_to_tfs.get(gene, set())
        receptor_to_tfs[receptor] = ";".join(sorted(tf_set)) if tf_set else ""
    return receptor_to_tfs


def _compute_stage_segments(act_s: pd.DataFrame, act_g: pd.DataFrame) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    common_times = act_s.columns.to_numpy(dtype=float)
    t_min = float(common_times[0])
    t_max = float(common_times[-1])
    t_norm = (common_times - t_min) / (t_max - t_min + 1e-12)
    x_s = act_s.values
    x_g = act_g.values
    diff_s = np.diff(x_s, axis=1)
    diff_g = np.diff(x_g, axis=1)
    delta = (diff_s ** 2).sum(axis=0) + (diff_g ** 2).sum(axis=0)
    t_count = act_s.shape[1]
    if t_count < 3 or delta.size < 2:
        k1 = t_count // 3
        k2 = 2 * t_count // 3
    else:
        split_indices = np.argsort(-delta)[:2]
        split_indices = np.sort(split_indices)
        k1, k2 = int(split_indices[0]), int(split_indices[1])
        k1 = max(0, min(k1, t_count - 3))
        k2 = max(k1 + 1, min(k2, t_count - 2))
    segments = {
        "early": np.arange(0, k1 + 1),
        "mid": np.arange(k1 + 1, k2 + 1),
        "late": np.arange(k2 + 1, t_count),
    }
    records: list[dict[str, Any]] = []
    for name, idxs in segments.items():
        if len(idxs) == 0:
            continue
        i0 = int(idxs[0])
        i1 = int(idxs[-1])
        records.append(
            {
                "stage": name,
                "idx_start": i0,
                "idx_end": i1,
                "t_raw_start": float(common_times[i0]),
                "t_raw_end": float(common_times[i1]),
                "t_norm_start": float(t_norm[i0]),
                "t_norm_end": float(t_norm[i1]),
            }
        )
    return segments, pd.DataFrame(records)


def _build_heatmap_outputs(
    curves_all: dict[tuple[str, str], np.ndarray],
    scores_all: pd.DataFrame,
    times_all: np.ndarray,
    output_dir: Path,
    settings: dict[str, Any],
) -> dict[str, Any]:
    if not curves_all:
        raise ValueError("curves_all is empty.")
    pairs_list = list(curves_all.keys())
    raw_mat = np.vstack([curves_all[pair] for pair in pairs_list])
    row_max = raw_mat.max(axis=1, keepdims=True)
    row_max[row_max == 0] = 1.0
    data_norm = raw_mat / row_max
    keep_idx = raw_mat.max(axis=1) > 0
    data_norm = data_norm[keep_idx]
    raw_mat = raw_mat[keep_idx]
    pairs_list = [pairs_list[idx] for idx in range(len(pairs_list)) if keep_idx[idx]]
    n_pairs = len(pairs_list)
    if n_pairs < settings["kmeans_clusters"]:
        raise ValueError(
            f"Valid LR pair count ({n_pairs}) is smaller than kmeans_clusters ({settings['kmeans_clusters']})."
        )

    peak_indices = np.argmax(data_norm, axis=1)
    peak_times = np.asarray(times_all, dtype=float)[peak_indices]
    kmeans = KMeans(
        n_clusters=settings["kmeans_clusters"],
        random_state=settings["kmeans_random_state"],
        n_init=settings["kmeans_n_init"],
    ).fit(peak_times.reshape(-1, 1))
    centers = kmeans.cluster_centers_.flatten()
    labels = kmeans.labels_
    sorted_center_idx = np.argsort(centers)
    map_label = {old: new for new, old in enumerate(sorted_center_idx)}
    new_labels = np.asarray([map_label[int(label)] for label in labels], dtype=int)

    grouped_indices: list[np.ndarray] = []
    for group_id in range(settings["kmeans_clusters"]):
        current = np.where(new_labels == group_id)[0]
        ordered = current[np.argsort(peak_times[current])]
        grouped_indices.append(ordered)

    gap_width = int(n_pairs * settings["gap_frac"])
    if gap_width < 1:
        gap_width = 1
    nan_gap = np.full((data_norm.shape[1], gap_width), np.nan)

    matrix_parts: list[np.ndarray] = []
    final_labels_list: list[str] = []
    group_label_positions: list[float] = []
    current_col = 0
    for group_id in range(settings["kmeans_clusters"]):
        indices = grouped_indices[group_id]
        sub_matrix = data_norm[indices].T
        matrix_parts.append(sub_matrix)
        final_labels_list.extend([f"{pairs_list[idx][0]}->{pairs_list[idx][1]}" for idx in indices])
        group_width = len(indices)
        group_label_positions.append(current_col + group_width / 2.0)
        current_col += group_width
        if group_id < settings["kmeans_clusters"] - 1:
            matrix_parts.append(nan_gap)
            final_labels_list.extend([""] * gap_width)
            current_col += gap_width
    plotting_matrix = np.hstack(matrix_parts)

    fig = plt.figure(figsize=(settings["heatmap_width"], settings["heatmap_height"]))
    gs = gridspec.GridSpec(1, 3, width_ratios=[0.02, 0.96, 0.02], wspace=0.05)
    ax_hm = fig.add_subplot(gs[1])
    cmap = plt.get_cmap("magma_r").copy()
    cmap.set_bad(color="white")
    image = ax_hm.imshow(
        plotting_matrix,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        origin="lower",
        extent=[0, len(final_labels_list), 0, 1],
    )

    total = len(final_labels_list)
    step = max(1, total // 60)
    tick_locs = np.arange(0, total, step) + 0.5
    tick_txts = [final_labels_list[idx] for idx in range(0, total, step)]
    ax_hm.set_xticks(tick_locs)
    ax_hm.set_xticklabels(tick_txts, rotation=90, fontsize=8)
    for group_id, x_pos in enumerate(group_label_positions):
        meta = settings["group_meta"][group_id]
        ax_hm.text(
            x_pos,
            1.05,
            meta["name"],
            transform=ax_hm.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=14,
            color=meta["color"],
        )
    ax_hm.set_ylabel("Normalized Time", fontsize=18)
    ax_hm.set_yticks([0, 1])
    ax_hm.spines["top"].set_visible(False)
    ax_hm.spines["right"].set_visible(False)
    ax_cb = fig.add_subplot(gs[2])
    plt.colorbar(image, cax=ax_cb, label="Score")
    fig_path = output_dir / settings["heatmap_figure"]
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    score_map = {
        (str(row["ligand"]), str(row["receptor"])): float(row["score_prog"])
        for _, row in scores_all.iterrows()
    }
    summary_df = pd.DataFrame(
        {
            "ligand": [pair[0] for pair in pairs_list],
            "receptor": [pair[1] for pair in pairs_list],
            "pair": [f"{pair[0]}->{pair[1]}" for pair in pairs_list],
            "peak_time": peak_times,
            "peak_stage": [settings["group_meta"][int(group_id)]["name"] for group_id in new_labels],
            "peak_score_raw": raw_mat[np.arange(raw_mat.shape[0]), peak_indices],
            "total_score_raw": [score_map.get(pair, np.nan) for pair in pairs_list],
        }
    ).sort_values(["peak_stage", "peak_time", "peak_score_raw"], ascending=[True, True, False])

    peak_tsv = output_dir / settings["peak_table_tsv"]
    peak_csv = output_dir / settings["peak_table_csv"]
    summary_df.to_csv(peak_tsv, sep="\t", index=False)
    summary_df.to_csv(peak_csv, index=False)

    return {
        "figure_path": str(fig_path),
        "peak_table_tsv": str(peak_tsv),
        "peak_table_csv": str(peak_csv),
        "valid_pair_count": int(n_pairs),
        "heatmap_shape": [int(plotting_matrix.shape[0]), int(plotting_matrix.shape[1])],
        "group_centers": [float(value) for value in centers.tolist()],
    }


def run_ccc_analysis_from_config(config: dict[str, Any]) -> dict[str, Any]:
    settings = _ccc_settings(config)
    base_dir = Path(str(config.get("base_dir", "."))).expanduser().resolve()
    output_dir = ensure_directory(_resolve_path(base_dir, settings["output_dir"]))
    _set_seed(settings["seed"])

    network_dir_soma = _resolve_path(base_dir, settings["network_dir_soma"])
    network_dir_germ = _resolve_path(base_dir, settings["network_dir_germ"])
    tg_annot_soma = _resolve_path(base_dir, settings["tg_annot_soma"])
    tg_annot_germ = _resolve_path(base_dir, settings["tg_annot_germ"])
    tf_annot_soma = _resolve_path(base_dir, settings["tf_annot_soma"])
    tf_annot_germ = _resolve_path(base_dir, settings["tf_annot_germ"])
    ccc_csv = _resolve_path(base_dir, settings["ccc_csv"])
    tf_filter_path = _resolve_path(base_dir, settings["tf_filter_path"])

    act_s_raw, times_s = load_activity_from_grn(network_dir_soma, tg_annot_soma, prob_col="prob")
    act_g_raw, times_g = load_activity_from_grn(network_dir_germ, tg_annot_germ, prob_col="prob")
    act_s = gaussian_smooth_over_time(act_s_raw, times_s, settings["sigma_time"])
    act_g = gaussian_smooth_over_time(act_g_raw, times_g, settings["sigma_time"])
    act_s, _times_s = collapse_duplicate_timecols(act_s)
    act_g, _times_g = collapse_duplicate_timecols(act_g)
    lr_df = load_lr_candidates(ccc_csv, act_s, act_g)

    times_s_net, mats_s, tf_s, tg_s = _load_network_series(network_dir_soma, tf_annot_soma, tg_annot_soma, prob_col="prob")
    times_g_net, mats_g, tf_g, tg_g = _load_network_series(network_dir_germ, tf_annot_germ, tg_annot_germ, prob_col="prob")
    tf_common = sorted(set(tf_s) & set(tf_g))
    tg_common = sorted(set(tg_s) & set(tg_g))
    if len(tf_common) < 5 or len(tg_common) < 5:
        raise ValueError(f"Common TF/TG is too small: TF={len(tf_common)}, TG={len(tg_common)}")
    mats_s = _restrict_to_common(mats_s, tf_s, tg_s, tf_common, tg_common)
    mats_g = _restrict_to_common(mats_g, tf_g, tg_g, tf_common, tg_common)

    fs_list: list[np.ndarray] = []
    cs_list: list[np.ndarray] = []
    for matrix in mats_s:
        abar = _normalize_bipartite(matrix, settings)
        fs_list.append(_top_r_svals(abar, settings["finger_r"]))
        cs_list.append(_tf_centrality(abar))
    fg_list: list[np.ndarray] = []
    cg_list: list[np.ndarray] = []
    for matrix in mats_g:
        abar = _normalize_bipartite(matrix, settings)
        fg_list.append(_top_r_svals(abar, settings["finger_r"]))
        cg_list.append(_tf_centrality(abar))

    fs = np.vstack(fs_list)
    fg = np.vstack(fg_list)
    c_s = pd.DataFrame(np.vstack(cs_list).T, index=tf_common, columns=times_s_net)
    c_g = pd.DataFrame(np.vstack(cg_list).T, index=tf_common, columns=times_g_net)
    fs = _zscore_cols(fs)
    fg = _zscore_cols(fg)
    c_s_z = pd.DataFrame(_zscore_rows(c_s.values), index=c_s.index, columns=c_s.columns)
    c_g_z = pd.DataFrame(_zscore_rows(c_g.values), index=c_g.index, columns=c_g.columns)
    tau_s_net = _normalize_time_to_01(times_s_net)
    tau_g_net = _normalize_time_to_01(times_g_net)

    w_lin_norm_on_s_net = tau_s_net.copy()
    t_germ_on_s_net_lin = times_g_net.min() + w_lin_norm_on_s_net * (times_g_net.max() - times_g_net.min())
    c_g_lin_on_s = interp_df_time(c_g_z, times_g_net, t_germ_on_s_net_lin)
    c_g_lin_on_s.columns = times_s_net

    ent_s_on_s_net, ent_g_on_g_net = prepare_entropy_on_net_grid(
        c_s_z,
        c_g_z,
        tau_s_net,
        tau_g_net,
        tf_filter_path=tf_filter_path,
    )

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

    w_cont_norm_on_s_net = learn_monotone_warp_continuous(
        tau_s_net,
        fs,
        tau_g_net,
        fg,
        ent_s=ent_s_on_s_net,
        ent_g=ent_g_on_g_net,
        fp_w=settings["warp_fp_w"],
        ent_w=settings["warp_ent_w"],
        lam_d2=settings["warp_lam_d2"],
        steps=settings["warp_steps"],
        lr=settings["warp_lr"],
    )
    t_germ_on_s_net_cont = times_g_net.min() + w_cont_norm_on_s_net * (times_g_net.max() - times_g_net.min())
    c_g_cont_on_s = interp_df_time(c_g_z, times_g_net, t_germ_on_s_net_cont)
    c_g_cont_on_s.columns = times_s_net

    act_g_u = act_g.copy()
    act_g_u.columns = [float(column) for column in act_g_u.columns]
    act_g_u = act_g_u.T.groupby(level=0).mean().T
    act_g_u = act_g_u.loc[:, sorted(act_g_u.columns)]
    times_g_act = act_g_u.columns.to_numpy(dtype=float)

    act_s_u = act_s.copy()
    act_s_u.columns = [float(column) for column in act_s_u.columns]
    act_s_u = act_s_u.T.groupby(level=0).mean().T
    act_s_u = act_s_u.loc[:, sorted(act_s_u.columns)]
    times_s_act = act_s_u.columns.to_numpy(dtype=float)

    w_cont_norm_on_s_act = np.interp(
        times_s_act,
        times_s_net,
        w_cont_norm_on_s_net,
        left=w_cont_norm_on_s_net[0],
        right=w_cont_norm_on_s_net[-1],
    )
    t_germ_on_s_act_cont = times_g_act.min() + w_cont_norm_on_s_act * (times_g_act.max() - times_g_act.min())
    act_g_cont = interp_df_over_time(act_g_u, times_g_act, t_germ_on_s_act_cont)
    act_g_cont.columns = times_s_act

    cost_fp = ((fs[:, None, :] - fg[None, :, :]) ** 2).sum(axis=2)
    cost_ent = (ent_s_on_s_net[:, None] - ent_g_on_g_net[None, :]) ** 2
    cost_fp = cost_fp / (cost_fp.mean() + 1e-12)
    cost_ent = cost_ent / (cost_ent.mean() + 1e-12)
    total_cost = settings["dtw_fp_w"] * cost_fp + settings["dtw_ent_w"] * cost_ent
    band = settings["dtw_band"]
    band_value = None if band is None else int(band)
    path = dtw_path(total_cost, band=band_value)

    t_s_count = fs.shape[0]
    matches: list[list[float]] = [[] for _ in range(t_s_count)]
    for i_idx, j_idx in path:
        if 0 <= i_idx < t_s_count and 0 <= j_idx < fg.shape[0]:
            matches[i_idx].append(float(tau_g_net[j_idx]))
    w_dtw_norm_on_s_net = np.zeros(t_s_count, dtype=float)
    for idx in range(t_s_count):
        if matches[idx]:
            w_dtw_norm_on_s_net[idx] = float(np.mean(matches[idx]))
        elif idx > 0:
            w_dtw_norm_on_s_net[idx] = w_dtw_norm_on_s_net[idx - 1]
        else:
            w_dtw_norm_on_s_net[idx] = 0.0
    w_dtw_norm_on_s_net = np.clip(w_dtw_norm_on_s_net, 0.0, 1.0)
    w_dtw_norm_on_s_net[0] = 0.0
    w_dtw_norm_on_s_net[-1] = 1.0
    w_dtw_norm_on_s_net = smooth_warp_d2_monotone(
        w_dtw_norm_on_s_net,
        lam=settings["dtw_lam_d2"],
        n_iter=settings["dtw_n_iter"],
        clip01=True,
        enforce_endpoints=True,
    )
    t_germ_on_s_net_dtw = times_g_net.min() + w_dtw_norm_on_s_net * (times_g_net.max() - times_g_net.min())
    c_g_dtw_on_s = interp_df_time(c_g_z, times_g_net, t_germ_on_s_net_dtw)
    c_g_dtw_on_s.columns = times_s_net

    w_dtw_norm_on_s_act = np.interp(
        times_s_act,
        times_s_net,
        w_dtw_norm_on_s_net,
        left=w_dtw_norm_on_s_net[0],
        right=w_dtw_norm_on_s_net[-1],
    )
    t_germ_on_s_act_dtw = times_g_act.min() + w_dtw_norm_on_s_act * (times_g_act.max() - times_g_act.min())
    act_g_dtw = interp_df_over_time(act_g_u, times_g_act, t_germ_on_s_act_dtw)
    act_g_dtw.columns = times_s_act

    warp_cont_path = output_dir / settings["warp_cont_tsv"]
    pd.DataFrame({"t_soma_net": times_s_net, "t_germ_warped": t_germ_on_s_net_cont}).to_csv(
        warp_cont_path,
        sep="\t",
        index=False,
    )
    warp_dtw_path = output_dir / settings["warp_dtw_tsv"]
    pd.DataFrame({"t_soma_net": times_s_net, "t_germ_warped": t_germ_on_s_net_dtw}).to_csv(
        warp_dtw_path,
        sep="\t",
        index=False,
    )

    requested_method = settings["alignment_method"].strip().lower()
    if requested_method == "auto":
        chosen_method, alignment_metrics = pick_alignment_method(
            c_s_z,
            c_g_cont_on_s,
            c_g_dtw_on_s,
            min_common_times=settings["min_common_times"],
            n_boot=settings["alignment_n_boot"],
            seed_cont=1,
            seed_dtw=2,
            tie_tol=settings["tie_tol"],
            prefer_on_tie=settings["prefer_on_tie"],
        )
    elif requested_method in {"warping", "dtw"}:
        chosen_method = "Warping" if requested_method == "warping" else "DTW"
        _, alignment_metrics = pick_alignment_method(
            c_s_z,
            c_g_cont_on_s,
            c_g_dtw_on_s,
            min_common_times=settings["min_common_times"],
            n_boot=settings["alignment_n_boot"],
            seed_cont=1,
            seed_dtw=2,
            tie_tol=settings["tie_tol"],
            prefer_on_tie=settings["prefer_on_tie"],
        )
        alignment_metrics["chosen_method"] = chosen_method
    else:
        raise ValueError("ccc.alignment_method must be auto, warping, or dtw.")

    ent_linear, n_linear = entropy_corr_score(
        _canon_timecols(c_s_z),
        _canon_timecols(c_g_lin_on_s),
        min_common_times=settings["min_common_times"],
        n_boot=settings["alignment_n_boot"],
        seed=0,
    )
    if not np.isfinite(ent_linear):
        raise ValueError(f"Invalid entropy corr value for linear alignment: {ent_linear}")
    alignment_metrics["entropy_corr_linear"] = float(ent_linear)
    alignment_metrics["n_common_times_linear"] = int(n_linear)
    alignment_comparison = _build_alignment_comparison_outputs(
        output_dir=output_dir,
        settings=settings,
        chosen_method=chosen_method,
        metrics=alignment_metrics,
    )

    act_g_selected = act_g_cont if chosen_method == "Warping" else act_g_dtw
    act_s_aligned = _canon_timecols(act_s)
    act_g_aligned = _canon_timecols(act_g_selected)
    act_g_aligned = act_g_aligned.reindex(columns=act_s_aligned.columns)
    if act_g_aligned.isna().any().any():
        missing = act_s_aligned.columns[act_g_aligned.isna().all(axis=0)]
        raise ValueError(f"Aligned act_g is missing columns on soma grid: {missing[:10].tolist()}")
    common_times = act_s_aligned.columns.to_numpy(dtype=float)
    if len(common_times) < 3:
        raise ValueError(f"Too few aligned timepoints: T={len(common_times)}")

    receptors = sorted(lr_df["receptor"].unique())
    programs = compute_receptor_programs(
        act_g_aligned,
        receptors,
        corr_thresh=settings["corr_thresh"],
        top_k=settings["top_k_prog"],
    )
    prog_df = compute_program_curves(programs, act_g_aligned)
    receptor_to_tfs = build_receptor_tfs_from_grn(
        network_dir_germ,
        tf_annot_germ,
        tg_annot_germ,
        programs,
    )

    segments, stage_df = _compute_stage_segments(act_s_aligned, act_g_aligned)
    stage_path = output_dir / settings["stage_table"]
    stage_df.to_csv(stage_path, sep="\t", index=False)

    stage_score_paths: dict[str, str] = {}
    for name, idxs in segments.items():
        if len(idxs) == 0:
            continue
        act_s_seg = act_s_aligned.iloc[:, idxs]
        prog_seg = prog_df.iloc[:, idxs]
        _curves_seg, scores_seg, _times_seg = compute_prog_comm_curves(lr_df, act_s_seg, prog_seg)
        scores_seg["tf_germline"] = scores_seg["receptor"].map(receptor_to_tfs).fillna("")
        out_path = output_dir / f"comm_scores_LR_program_{name}.tsv"
        scores_seg.to_csv(out_path, sep="\t", index=False)
        stage_score_paths[name] = str(out_path)

    curves_all, scores_all, times_all = compute_prog_comm_curves(lr_df, act_s_aligned, prog_df)
    heatmap_report = _build_heatmap_outputs(curves_all, scores_all, times_all, output_dir, settings)

    report = {
        "analysis": "ccc",
        "output_dir": str(output_dir),
        "network_dir_soma": str(network_dir_soma),
        "network_dir_germ": str(network_dir_germ),
        "ccc_csv": str(ccc_csv),
        "seed": int(settings["seed"]),
        "lr_candidate_count": int(len(lr_df)),
        "program_receptor_count": int(len(programs)),
        "chosen_alignment": str(chosen_method),
        "alignment_metrics": alignment_metrics,
        "alignment_comparison": alignment_comparison,
        "warp_cont_path": str(warp_cont_path),
        "warp_dtw_path": str(warp_dtw_path),
        "stage_table": str(stage_path),
        "stage_score_paths": stage_score_paths,
        "heatmap": heatmap_report,
        "settings": settings,
    }
    save_checkpoint_metadata(output_dir / "ccc_report.json", report)
    return report


def run_ccc_analysis(config: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(config, (str, Path)):
        path = Path(config).resolve()
        loaded = load_yaml_config(path)
        loaded.setdefault("base_dir", str(Path(".").resolve()))
        report = run_ccc_analysis_from_config(loaded)
        report["config_path"] = str(path)
        return report
    return run_ccc_analysis_from_config(config)
