"""Single-TF activity, target genes and regulons."""

from __future__ import annotations

import math
import random
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from ..checkpoint import save_checkpoint_metadata
from ..infer import reload_germ_from_checkpoint, reload_soma_from_checkpoint
from ..preprocess import norm_id, prepare_common_inputs
from ..utils import ensure_directory, load_yaml_config, resolve_config_path
from .network import _candidate_active_re_indices, _select_re_indices_composite
from .repro import safe_label


def _section(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    return value if isinstance(value, dict) else {}


def _single_tf_settings(config: dict[str, Any]) -> dict[str, Any]:
    section = _section(config, "single_tf")
    locus_section = section.get("locus_plot", {})
    if not isinstance(locus_section, dict):
        locus_section = {}
    lineage = str(config.get("lineage", "")).lower()
    default_max_expr_frac = 0.7 if lineage in {"germ", "germline"} else 0.8
    default_time_for_top: Any = 3 if lineage in {"germ", "germline"} else 0
    default_topn_tg: Any = 25 if lineage in {"germ", "germline"} else 50
    stage_labels = section.get("stage_labels")
    if isinstance(stage_labels, tuple):
        stage_labels = list(stage_labels)
    if stage_labels is not None and not isinstance(stage_labels, list):
        stage_labels = [stage_labels]
    tf_symbols = section.get("tf_symbols")
    if tf_symbols is None:
        tf_symbol = section.get("tf_symbol")
        tf_symbols = [tf_symbol] if tf_symbol else []
    elif not isinstance(tf_symbols, list):
        tf_symbols = [tf_symbols]
    target_tgs = locus_section.get("target_tgs")
    if target_tgs is None:
        target_tg = locus_section.get("target_tg")
        target_tgs = [target_tg] if target_tg else []
    elif not isinstance(target_tgs, list):
        target_tgs = [target_tgs]

    return {
        "output_dir": str(section.get("output_dir", f"result/{lineage}/single_tf")),
        "tf_symbols": [str(value) for value in tf_symbols if str(value).strip()],
        "centrality_mode": str(section.get("centrality_mode", "topk_sum")),
        "topk": int(section.get("topk", 500)),
        "threshold": float(section.get("threshold", 0.5)),
        "min_expr_frac": float(section.get("min_expr_frac", 0.1)),
        "max_expr_frac": float(section.get("max_expr_frac", default_max_expr_frac)),
        "time_for_top": section.get("time_for_top", default_time_for_top),
        "topn_tg": section.get("topn_tg", default_topn_tg),
        "topn_re": int(section.get("topn_re", 2)),
        "k_nearest_re": int(section.get("k_nearest_re", 10)),
        "active_frac_re": float(section.get("active_frac_re", 0.05)),
        "max_dist_bp": int(section.get("max_dist_bp", 250_000)),
        "dist_lambda": float(section.get("dist_lambda", 50_000.0)),
        "w_re_tg": float(section.get("w_re_tg", 0.40)),
        "w_dist": float(section.get("w_dist", 0.30)),
        "w_tf_re": float(section.get("w_tf_re", 0.20)),
        "w_act": float(section.get("w_act", 0.10)),
        "user_thresh": float(section.get("user_thresh", 0.20)),
        "output_prefix": str(section.get("output_prefix", "regulon_TG_RE")),
        "stage_labels": [str(value) for value in stage_labels] if stage_labels is not None else None,
        "network_seed": int(section.get("network_seed", 42)),
        "save_fig": bool(section.get("save_fig", True)),
        "save_tables": bool(section.get("save_tables", True)),
        "fs_title": int(section.get("fs_title", 18)),
        "fs_other": int(section.get("fs_other", 12)),
        "fs_gene": int(section.get("fs_gene", 15)),
        "fs_tf_center": int(section.get("fs_tf_center", 20)),
        "activity_dpi": int(section.get("activity_dpi", 300)),
        "network_dpi": int(section.get("network_dpi", 300)),
        "locus_plot": {
            "enabled": bool(locus_section.get("enabled", False)),
            "target_tgs": [str(value) for value in target_tgs if str(value).strip()],
            "window_upstream_bp": int(locus_section.get("window_upstream_bp", 50_000)),
            "window_downstream_bp": int(locus_section.get("window_downstream_bp", 50_000)),
            "track_agg": str(locus_section.get("track_agg", "mean")).lower(),
            "output_dir": str(locus_section.get("output_dir", "locus_plots")),
            "save_table": bool(locus_section.get("save_table", True)),
            "save_fig": bool(locus_section.get("save_fig", True)),
            "dpi": int(locus_section.get("dpi", section.get("network_dpi", 300))),
        },
    }


def _pick_col(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {str(column).lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def build_fbgn_to_name(common: dict[str, Any]) -> dict[str, str]:
    fb2name: dict[str, str] = {}

    tg_df = common.get("tg_df")
    if isinstance(tg_df, pd.DataFrame):
        fb_col = _pick_col(tg_df, ["fbgn_id", "fbgn", "fbgnid"])
        name_col = _pick_col(tg_df, ["gene_name", "symbol", "gene", "name"])
        if fb_col and name_col:
            fb2name.update(dict(zip(tg_df[fb_col].astype(str), tg_df[name_col].astype(str))))

    tf_df = common.get("tf_df")
    if isinstance(tf_df, pd.DataFrame):
        fb_col = _pick_col(tf_df, ["fbgn_id", "fbgn", "fbgnid"])
        name_col = _pick_col(tf_df, ["gene_name", "symbol", "gene", "name"])
        if fb_col and name_col:
            fb2name.update(dict(zip(tf_df[fb_col].astype(str), tf_df[name_col].astype(str))))

    sym2fb = common.get("sym2fb")
    if isinstance(sym2fb, dict):
        for symbol, fbgn in sym2fb.items():
            fbgn = str(fbgn)
            if fbgn not in fb2name:
                fb2name[fbgn] = str(symbol)

    return fb2name


def centrality_from_prob(
    prob: np.ndarray,
    mode: str = "topk_sum",
    topk: int = 500,
    thresh: float = 0.5,
) -> float:
    values = np.asarray(prob, dtype=float)
    if values.size == 0:
        return 0.0
    if mode == "topk_sum":
        keep = int(min(topk, values.size))
        if keep <= 0:
            return 0.0
        idx = np.argpartition(values, -keep)[-keep:]
        return float(values[idx].sum())
    if mode == "topk_mean":
        keep = int(min(topk, values.size))
        if keep <= 0:
            return 0.0
        idx = np.argpartition(values, -keep)[-keep:]
        return float(values[idx].mean())
    if mode == "threshold_count":
        return float(np.sum(values >= thresh))
    if mode == "threshold_sum":
        return float(values[values >= thresh].sum())
    raise ValueError(f"Unsupported centrality mode: {mode}")


def _l2norm_rows(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, eps)


def resolve_tf_id_and_index(
    tf_symbol: str,
    tf_ids: list[str],
    tf_display_names: list[str],
    common: dict[str, Any],
) -> tuple[str, int]:
    tf_id = None
    sym2fb = common.get("sym2fb")
    if isinstance(sym2fb, dict):
        if tf_symbol in sym2fb:
            tf_id = str(sym2fb[tf_symbol])
        else:
            sym2fb_lower = {str(key).lower(): value for key, value in sym2fb.items()}
            if tf_symbol.lower() in sym2fb_lower:
                tf_id = str(sym2fb_lower[tf_symbol.lower()])

    tf_df = common.get("tf_df")
    if tf_id is None and isinstance(tf_df, pd.DataFrame):
        gene_col = _pick_col(tf_df, ["gene_name", "symbol"])
        fbgn_col = _pick_col(tf_df, ["fbgn_id", "fbgn", "fbgnid"])
        if gene_col and fbgn_col:
            hit = tf_df[tf_df[gene_col].astype(str).str.lower() == tf_symbol.lower()]
            if not hit.empty:
                tf_id = str(hit.iloc[0][fbgn_col])

    if tf_id is None:
        for value in tf_ids:
            if str(value).lower() == tf_symbol.lower():
                tf_id = str(value)
                break

    if tf_id is None:
        for value, tf_name in zip(tf_ids, tf_display_names):
            if str(tf_name).lower() == tf_symbol.lower():
                tf_id = str(value)
                break

    if tf_id is None or tf_id not in tf_ids:
        raise RuntimeError(f"Could not resolve TF '{tf_symbol}' inside the TF graph namespace.")

    return tf_id, tf_ids.index(tf_id)


def resolve_time_index(
    time_spec: Any,
    time_labels: list[str],
    time_count: int,
    tf_symbol: str | None = None,
) -> int:
    if isinstance(time_spec, dict):
        if tf_symbol is not None:
            if tf_symbol in time_spec:
                return resolve_time_index(time_spec[tf_symbol], time_labels, time_count, tf_symbol=tf_symbol)
            tf_key = tf_symbol.lower()
            if tf_key in time_spec:
                return resolve_time_index(time_spec[tf_key], time_labels, time_count, tf_symbol=tf_symbol)
        return resolve_time_index(next(iter(time_spec.values())), time_labels, time_count, tf_symbol=tf_symbol)

    if isinstance(time_spec, (int, np.integer)):
        time_idx = int(time_spec)
        if time_idx < 0 or time_idx >= time_count:
            raise ValueError(f"time={time_idx} is outside the valid range [0, {time_count - 1}].")
        return time_idx

    if isinstance(time_spec, str):
        if time_spec in time_labels:
            return time_labels.index(time_spec)
        match = re.search(r"(\d+)$", time_spec)
        if match:
            time_idx = int(match.group(1))
            if 0 <= time_idx < time_count:
                return time_idx
        raise ValueError(f"Could not resolve time label '{time_spec}'.")

    raise TypeError("time_for_top must be int, str, or a tf-specific mapping.")


def resolve_topn(topn_spec: Any, tf_symbol: str) -> int:
    if isinstance(topn_spec, dict):
        if tf_symbol in topn_spec:
            return int(topn_spec[tf_symbol])
        tf_key = tf_symbol.lower()
        if tf_key in topn_spec:
            return int(topn_spec[tf_key])
        return int(next(iter(topn_spec.values())))
    return int(topn_spec)


def _time_labels(lineage: str, inference: dict[str, Any], time_count: int) -> list[str]:
    if lineage in {"germ", "germline"}:
        germline_types = inference.get("germline_types")
        if isinstance(germline_types, list) and len(germline_types) == time_count:
            return [str(value) for value in germline_types]
        return [f"t{idx}" for idx in range(time_count)]
    somatic_names = inference.get("somatic_names")
    if isinstance(somatic_names, list) and len(somatic_names) == time_count:
        return [str(value) for value in somatic_names]
    return [f"t{idx}" for idx in range(time_count)]


def _stage_labels_for_export(settings: dict[str, Any], time_labels: list[str], time_count: int) -> list[str]:
    custom = settings.get("stage_labels")
    if isinstance(custom, list) and len(custom) == time_count:
        return [str(value) for value in custom]
    return [str(value) for value in time_labels]


def _stage_file_token(stage_label: str) -> str:
    text = str(stage_label).replace("°", "deg").replace(" ", "_")
    return safe_label(text)


def _label_to_cells(common: dict[str, Any]) -> dict[str, list[str]]:
    anno = common["anno"]
    expr_df = common["expr_df"]
    celltype_col = _pick_col(anno, ["cell_type", "celltype", "type", "annotation", "cluster", "label"])
    cellid_col = _pick_col(anno, ["cell_id", "cellid", "barcode", "cell", "obs_names", "id"])
    if celltype_col is None or cellid_col is None:
        raise RuntimeError(f"Could not identify anno cell-type/cell-id columns: {list(anno.columns)}")

    expr_cols = set(expr_df.columns.astype(str))
    label2cells: dict[str, list[str]] = {}
    for label in pd.unique(anno[celltype_col].astype(str)):
        cells = anno.loc[anno[celltype_col].astype(str) == str(label), cellid_col].astype(str).tolist()
        label2cells[str(label)] = [cell for cell in cells if cell in expr_cols]
    return label2cells


def _prepare_keep_masks(
    time_labels: list[str],
    tg_ids: list[str],
    expr_df: pd.DataFrame,
    label2cells: dict[str, list[str]],
    fb2name: dict[str, str],
    min_expr_frac: float,
    max_expr_frac: float,
) -> tuple[list[np.ndarray | None], list[np.ndarray | None]]:
    expr_idx = set(expr_df.index.astype(str))
    tg_expr_key: list[str | None] = []
    for tg in map(str, tg_ids):
        if tg in expr_idx:
            tg_expr_key.append(tg)
        else:
            symbol = fb2name.get(tg)
            tg_expr_key.append(str(symbol) if symbol is not None and str(symbol) in expr_idx else None)

    tg_expr_key_arr = np.asarray(tg_expr_key, dtype=object)
    valid_tg_mask = np.asarray([value is not None for value in tg_expr_key_arr], dtype=bool)

    keep_masks: list[np.ndarray | None] = [None] * len(time_labels)
    frac_list: list[np.ndarray | None] = [None] * len(time_labels)
    for time_idx, label in enumerate(time_labels):
        cells = label2cells.get(str(label), [])
        if len(cells) == 0:
            continue
        sub = expr_df.loc[tg_expr_key_arr[valid_tg_mask].tolist(), cells]
        frac_valid = (sub.values > 0).mean(axis=1)
        fracs = np.full(len(tg_ids), np.nan, dtype=float)
        fracs[valid_tg_mask] = frac_valid
        keep = np.isfinite(fracs) & (fracs >= min_expr_frac) & (fracs <= max_expr_frac)
        keep_masks[time_idx] = keep
        frac_list[time_idx] = fracs
    return keep_masks, frac_list


def _parse_re_interval(value: str) -> tuple[str, int, int] | None:
    text = str(value).strip()
    match = re.match(r"^([^:]+):\s*([0-9]+)\s*-\s*([0-9]+)$", text)
    if not match:
        return None
    chrom, start, end = match.group(1), int(match.group(2)), int(match.group(3))
    if start > end:
        start, end = end, start
    chrom = chrom.replace("chr", "")
    return chrom, start, end


def _build_re_metadata(re_names: list[str]) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    re_chr = np.empty(len(re_names), dtype=object)
    re_center = np.full(len(re_names), np.nan, dtype=float)
    chr2reidx: dict[str, list[int]] = {}
    for idx, re_name in enumerate(re_names):
        parsed = _parse_re_interval(re_name)
        if parsed is None:
            re_chr[idx] = None
            continue
        chrom, start, end = parsed
        re_chr[idx] = chrom
        re_center[idx] = (start + end) / 2.0
        chr2reidx.setdefault(chrom, []).append(idx)
    return re_chr, re_center, {chrom: np.asarray(indices, dtype=int) for chrom, indices in chr2reidx.items()}


def _build_tg_metadata(common: dict[str, Any], tg_ids: list[str]) -> dict[str, dict[str, Any]]:
    tg_df = common.get("tg_df")
    if not isinstance(tg_df, pd.DataFrame):
        raise RuntimeError("single_tf requires tg_df metadata to resolve TG coordinates.")

    col_fb = _pick_col(tg_df, ["fbgn_id", "fbgn", "fbgnid"])
    col_gn = _pick_col(tg_df, ["gene_name", "symbol", "gene", "name"])
    col_chr = _pick_col(tg_df, ["chr", "chrom", "chromosome"])
    col_st = _pick_col(tg_df, ["start", "tss", "tx_start"])
    col_en = _pick_col(tg_df, ["end", "tx_end"])
    col_str = _pick_col(tg_df, ["strand", "orientation", "sense"])
    if not (col_fb and col_chr and col_st and col_en):
        raise RuntimeError(f"TG.tsv is missing required FBgn/chr/start/end columns: {list(tg_df.columns)}")

    tg_meta: dict[str, dict[str, Any]] = {}
    tg_set = set(map(str, tg_ids))
    for _, row in tg_df.iterrows():
        fbgn = str(row[col_fb])
        if fbgn not in tg_set:
            continue
        try:
            chrom = str(row[col_chr]).replace("chr", "")
            start = int(float(row[col_st]))
            end = int(float(row[col_en]))
        except Exception:
            continue
        strand = str(row[col_str]).strip() if col_str else None
        tss = end if strand in {"-", "-1", "minus", "rev", "reverse"} else start
        gene_name = str(row[col_gn]) if col_gn else fbgn
        tg_meta[fbgn] = {
            "gene": gene_name,
            "chr": chrom,
            "tss": tss,
            "start": start,
            "end": end,
            "strand": strand,
        }
    return tg_meta


def _build_peak_metadata(common: dict[str, Any]) -> dict[str, Any]:
    peaks_df = common.get("peaks_df")
    if not isinstance(peaks_df, pd.DataFrame):
        raise RuntimeError("single_tf locus plot requires ATAC peaks_df metadata.")
    peak_names = peaks_df.index.astype(str).tolist()
    peak_chr = np.empty(len(peak_names), dtype=object)
    peak_start = np.full(len(peak_names), np.nan, dtype=float)
    peak_end = np.full(len(peak_names), np.nan, dtype=float)
    chr2peakidx: dict[str, list[int]] = {}
    for idx, peak_name in enumerate(peak_names):
        parsed = _parse_re_interval(peak_name)
        if parsed is None:
            peak_chr[idx] = None
            continue
        chrom, start, end = parsed
        peak_chr[idx] = chrom
        peak_start[idx] = float(start)
        peak_end[idx] = float(end)
        chr2peakidx.setdefault(chrom, []).append(idx)
    return {
        "peak_names": peak_names,
        "peak_chr": peak_chr,
        "peak_start": peak_start,
        "peak_end": peak_end,
        "chr2peakidx": {chrom: np.asarray(indices, dtype=int) for chrom, indices in chr2peakidx.items()},
        "peaks_df": peaks_df,
    }


def _norm_gene_key(value: str) -> str:
    return re.sub(r"\s+", "", str(value).strip().lower())


def _tg_aliases(tg_id: str, meta: dict[str, Any] | None, fb2name: dict[str, str]) -> list[str]:
    aliases = [str(tg_id)]
    gene_name = None
    if meta is not None:
        gene_name = meta.get("gene")
    if gene_name is None:
        gene_name = fb2name.get(str(tg_id))
    if gene_name is not None:
        aliases.append(str(gene_name))
        aliases.append(f"{gene_name} ({tg_id})")
    return aliases


def _resolve_requested_tg_indices(
    requested: list[str],
    tg_ids: list[str],
    tg_meta: dict[str, dict[str, Any]],
    fb2name: dict[str, str],
) -> dict[str, int]:
    alias2idx: dict[str, int] = {}
    for idx, tg_id in enumerate(tg_ids):
        meta = tg_meta.get(str(tg_id))
        for alias in _tg_aliases(str(tg_id), meta, fb2name):
            alias2idx.setdefault(_norm_gene_key(alias), idx)

    resolved: dict[str, int] = {}
    missing: list[str] = []
    for item in requested:
        key = _norm_gene_key(item)
        if key in alias2idx:
            resolved[str(item)] = int(alias2idx[key])
        else:
            missing.append(str(item))
    if missing:
        raise RuntimeError(
            "Could not resolve requested target gene(s) in the TG namespace: "
            + ", ".join(missing)
        )
    return resolved


def _top_tg_indices(
    tf_idx: int,
    time_idx: int,
    keep: np.ndarray | None,
    topn: int,
    out_tf_np: np.ndarray,
    out_tg_np: np.ndarray,
) -> np.ndarray:
    if topn <= 0:
        return np.array([], dtype=int)
    htf_tn = _l2norm_rows(out_tf_np[:, time_idx, :])
    htg_tn = _l2norm_rows(out_tg_np[:, time_idx, :])
    tf_vec = htf_tn[tf_idx]
    cos_tg = htg_tn @ tf_vec
    if keep is not None:
        cos_tg = cos_tg.copy()
        cos_tg[~keep] = -np.inf
    finite_mask = np.isfinite(cos_tg)
    if not finite_mask.any():
        return np.array([], dtype=int)
    cand_idx = np.where(finite_mask)[0]
    k = min(topn, cand_idx.size)
    if k <= 0:
        return np.array([], dtype=int)
    top_idx = np.argpartition(cos_tg, -k)[-k:]
    top_idx = top_idx[np.argsort(cos_tg[top_idx])[::-1]]
    return np.asarray(top_idx, dtype=int)


def _select_top_re_indices(
    tg_chr: str | None,
    tg_tss: float | None,
    tg_vec: np.ndarray,
    hre_t: np.ndarray,
    re_names: list[str],
    re_center: np.ndarray,
    chr2reidx: dict[str, np.ndarray],
    k_nearest_re: int,
    topn_re: int,
) -> np.ndarray:
    if topn_re <= 0 or len(re_names) == 0:
        return np.array([], dtype=int)
    top_re_idx: np.ndarray | None = None

    if tg_chr is not None and tg_tss is not None and tg_chr in chr2reidx:
        cand_all = chr2reidx[tg_chr]
        centers = re_center[cand_all]
        finite = np.isfinite(centers)
        cand_all = cand_all[finite]
        centers = centers[finite]
        if cand_all.size > 0:
            abs_dist = np.abs(centers - float(tg_tss))
            k_near = min(k_nearest_re, abs_dist.size)
            if k_near > 0:
                near_loc = np.argpartition(abs_dist, k_near - 1)[:k_near]
                cand_near = cand_all[near_loc]
                sims = hre_t[cand_near] @ tg_vec
                k_sel = min(topn_re, sims.size)
                if k_sel > 0:
                    top_loc = np.argpartition(sims, -k_sel)[-k_sel:]
                    top_loc = top_loc[np.argsort(sims[top_loc])[::-1]]
                    top_re_idx = cand_near[top_loc]

    if top_re_idx is None and tg_tss is not None:
        cand_all = np.arange(len(re_names), dtype=int)
        centers = re_center[cand_all]
        finite = np.isfinite(centers)
        cand_all = cand_all[finite]
        centers = centers[finite]
        if cand_all.size > 0:
            abs_dist = np.abs(centers - float(tg_tss))
            k_near = min(k_nearest_re, abs_dist.size)
            if k_near > 0:
                near_loc = np.argpartition(abs_dist, k_near - 1)[:k_near]
                cand_near = cand_all[near_loc]
                sims = hre_t[cand_near] @ tg_vec
                k_sel = min(topn_re, sims.size)
                if k_sel > 0:
                    top_loc = np.argpartition(sims, -k_sel)[-k_sel:]
                    top_loc = top_loc[np.argsort(sims[top_loc])[::-1]]
                    top_re_idx = cand_near[top_loc]

    if top_re_idx is None:
        cand_all = np.arange(len(re_names), dtype=int)
        sims = hre_t[cand_all] @ tg_vec
        k_sel = min(topn_re, sims.size)
        if k_sel <= 0:
            return np.array([], dtype=int)
        top_loc = np.argpartition(sims, -k_sel)[-k_sel:]
        top_loc = top_loc[np.argsort(sims[top_loc])[::-1]]
        top_re_idx = cand_all[top_loc]

    return np.asarray(top_re_idx, dtype=int)


def _top_tg_re_table(
    tf_symbol: str,
    tf_idx: int,
    time_idx: int,
    stage_label: str,
    top_tg_idx: np.ndarray,
    tg_ids: list[str],
    re_names: list[str],
    re_chr: np.ndarray,
    re_center: np.ndarray,
    re_activity_frac: np.ndarray,
    tg_meta: dict[str, dict[str, Any]],
    out_tf_np: np.ndarray,
    out_re_np: np.ndarray,
    out_tg_np: np.ndarray,
    active_frac_re: float,
    topn_re: int,
    max_dist_bp: int,
    dist_lambda: float,
    w_re_tg: float,
    w_dist: float,
    w_tf_re: float,
    w_act: float,
) -> pd.DataFrame:
    htf_t = _l2norm_rows(out_tf_np[:, time_idx, :])
    htg_t = _l2norm_rows(out_tg_np[:, time_idx, :])
    active_idx = np.where(re_activity_frac >= float(active_frac_re))[0]
    if active_idx.size == 0:
        active_idx = np.arange(len(re_names), dtype=int)
    hre_t = _l2norm_rows(out_re_np[active_idx, time_idx, :])
    act_re_chr = re_chr[active_idx]
    act_re_center = re_center[active_idx]
    act_re_finite = np.isfinite(act_re_center)
    act_re_frac = re_activity_frac[active_idx]
    tf_vec = htf_t[tf_idx]

    rows: list[dict[str, Any]] = []
    for tg_i in top_tg_idx.tolist():
        tg_fb = str(tg_ids[tg_i])
        meta = tg_meta.get(tg_fb)
        tg_gene = meta["gene"] if meta else tg_fb
        tg_show = f"{tg_gene} ({tg_fb})"
        tg_chr = meta["chr"] if meta else None
        tg_tss = meta["tss"] if meta else None
        tg_vec = htg_t[tg_i]

        candidate_rel = _candidate_active_re_indices(
            tg_chr=tg_chr,
            tg_tss=tg_tss,
            act_re_chr=act_re_chr,
            act_re_center=act_re_center,
            act_re_finite=act_re_finite,
            max_dist_bp=max_dist_bp,
        )
        top_re_rel, _top_score = _select_re_indices_composite(
            tf_vec=tf_vec,
            tg_vec=tg_vec,
            re_embed=hre_t,
            candidate_rel=candidate_rel,
            act_re_center=act_re_center,
            act_re_frac=act_re_frac,
            active_frac_re=active_frac_re,
            tg_tss=tg_tss,
            topn_re_per_tg=topn_re,
            dist_lambda=dist_lambda,
            w_re_tg=w_re_tg,
            w_dist=w_dist,
            w_tf_re=w_tf_re,
            w_act=w_act,
            eps=1e-12,
        )
        top_re_idx = active_idx[top_re_rel] if top_re_rel.size > 0 else np.array([], dtype=int)

        for re_i in top_re_idx.tolist():
            re_name = str(re_names[re_i])
            dist = np.nan
            if tg_tss is not None and np.isfinite(re_center[re_i]):
                dist = float(re_center[re_i] - float(tg_tss))
            rows.append(
                {
                    "TF": str(tf_symbol),
                    "stage": str(stage_label),
                    "t_idx": int(time_idx),
                    "TG": tg_show,
                    "RE": re_name,
                    "dist_REcenter_to_TSS_bp": dist,
                }
            )
    return pd.DataFrame(
        rows,
        columns=["TF", "stage", "t_idx", "TG", "RE", "dist_REcenter_to_TSS_bp"],
    )


def _locus_re_table(
    tf_symbol: str,
    tf_idx: int,
    stage_label: str,
    time_idx: int,
    tg_idx: int,
    tg_ids: list[str],
    tg_meta: dict[str, dict[str, Any]],
    fb2name: dict[str, str],
    re_names: list[str],
    re_chr: np.ndarray,
    re_center: np.ndarray,
    re_activity_frac: np.ndarray,
    out_tf_np: np.ndarray,
    out_re_np: np.ndarray,
    out_tg_np: np.ndarray,
    active_frac_re: float,
    topn_re: int,
    max_dist_bp: int,
    dist_lambda: float,
    w_re_tg: float,
    w_dist: float,
    w_tf_re: float,
    w_act: float,
) -> pd.DataFrame:
    htf_t = _l2norm_rows(out_tf_np[:, time_idx, :])
    htg_t = _l2norm_rows(out_tg_np[:, time_idx, :])
    active_idx = np.where(re_activity_frac >= float(active_frac_re))[0]
    if active_idx.size == 0:
        active_idx = np.arange(len(re_names), dtype=int)
    hre_t = _l2norm_rows(out_re_np[active_idx, time_idx, :])
    act_re_chr = re_chr[active_idx]
    act_re_center = re_center[active_idx]
    act_re_finite = np.isfinite(act_re_center)
    act_re_frac = re_activity_frac[active_idx]
    tg_fb = str(tg_ids[tg_idx])
    meta = tg_meta.get(tg_fb)
    tg_gene = (meta or {}).get("gene", fb2name.get(tg_fb, tg_fb))
    tg_chr = (meta or {}).get("chr")
    tg_tss = (meta or {}).get("tss")
    candidate_rel = _candidate_active_re_indices(
        tg_chr=tg_chr,
        tg_tss=tg_tss,
        act_re_chr=act_re_chr,
        act_re_center=act_re_center,
        act_re_finite=act_re_finite,
        max_dist_bp=max_dist_bp,
    )
    top_re_rel, _top_score = _select_re_indices_composite(
        tf_vec=htf_t[tf_idx],
        tg_vec=htg_t[tg_idx],
        re_embed=hre_t,
        candidate_rel=candidate_rel,
        act_re_center=act_re_center,
        act_re_frac=act_re_frac,
        active_frac_re=active_frac_re,
        tg_tss=tg_tss,
        topn_re_per_tg=topn_re,
        dist_lambda=dist_lambda,
        w_re_tg=w_re_tg,
        w_dist=w_dist,
        w_tf_re=w_tf_re,
        w_act=w_act,
        eps=1e-12,
    )
    top_re_idx = active_idx[top_re_rel] if top_re_rel.size > 0 else np.array([], dtype=int)

    rows: list[dict[str, Any]] = []
    for rank, re_idx in enumerate(top_re_idx.tolist(), start=1):
        re_name = str(re_names[re_idx])
        parsed = _parse_re_interval(re_name)
        chrom, start, end = (parsed if parsed is not None else (None, np.nan, np.nan))
        dist = np.nan
        if tg_tss is not None and np.isfinite(re_center[re_idx]):
            dist = float(re_center[re_idx] - float(tg_tss))
        rows.append(
            {
                "TF": str(tf_symbol),
                "stage": str(stage_label),
                "t_idx": int(time_idx),
                "TG": f"{tg_gene} ({tg_fb})",
                "RE": re_name,
                "chr": chrom,
                "start": start,
                "end": end,
                "rank_re": int(rank),
                "dist_REcenter_to_TSS_bp": dist,
            }
        )
    return pd.DataFrame(
        rows,
        columns=["TF", "stage", "t_idx", "TG", "RE", "chr", "start", "end", "rank_re", "dist_REcenter_to_TSS_bp"],
    )


def _aggregate_peak_signal(
    peaks_df: pd.DataFrame,
    peak_indices: np.ndarray,
    cells: list[str],
    agg: str,
) -> np.ndarray:
    if peak_indices.size == 0:
        return np.array([], dtype=float)
    if len(cells) == 0:
        return np.zeros(peak_indices.size, dtype=float)
    values = peaks_df.iloc[peak_indices, :].loc[:, cells].to_numpy(dtype=float)
    if agg == "sum":
        return values.sum(axis=1)
    return values.mean(axis=1)


def _map_cells_to_peak_columns(cells: list[str], peaks_df: pd.DataFrame) -> list[str]:
    peak_map = {norm_id(column): str(column) for column in peaks_df.columns.astype(str)}
    mapped: list[str] = []
    for cell in cells:
        key = norm_id(cell)
        if key in peak_map:
            mapped.append(peak_map[key])
    return mapped


def _re_activity_fraction_for_stage(
    common: dict[str, Any],
    re_names: list[str],
    cells: list[str],
) -> np.ndarray:
    peaks_df = common["peaks_df"]
    peak_cells = _map_cells_to_peak_columns(cells, peaks_df)
    if len(peak_cells) == 0:
        return np.zeros(len(re_names), dtype=float)
    values = peaks_df.loc[re_names, peak_cells].to_numpy(dtype=float)
    return (values != 0).mean(axis=1).astype(float)


def _plot_single_tf_locus(
    tf_symbol: str,
    stage_label: str,
    time_idx: int,
    tg_idx: int,
    tg_ids: list[str],
    tg_meta: dict[str, dict[str, Any]],
    fb2name: dict[str, str],
    peak_meta: dict[str, Any],
    re_table: pd.DataFrame,
    cells: list[str],
    output_path: Path,
    dpi: int,
    window_upstream_bp: int,
    window_downstream_bp: int,
    track_agg: str,
) -> None:
    tg_fb = str(tg_ids[tg_idx])
    meta = tg_meta.get(tg_fb)
    if meta is None:
        raise RuntimeError(f"Missing TG metadata for locus plot target {tg_fb}.")
    tg_gene = str(meta.get("gene", fb2name.get(tg_fb, tg_fb)))
    tg_chr = str(meta["chr"])
    tg_tss = float(meta["tss"])
    tg_start = float(meta.get("start", tg_tss))
    tg_end = float(meta.get("end", tg_tss))
    strand = str(meta.get("strand") or "+")

    x_min = tg_tss - float(window_upstream_bp)
    x_max = tg_tss + float(window_downstream_bp)
    peak_indices_all = peak_meta["chr2peakidx"].get(tg_chr, np.array([], dtype=int))
    if peak_indices_all.size > 0:
        starts = peak_meta["peak_start"][peak_indices_all]
        ends = peak_meta["peak_end"][peak_indices_all]
        mask = np.isfinite(starts) & np.isfinite(ends) & (ends >= x_min) & (starts <= x_max)
        peak_indices = peak_indices_all[mask]
    else:
        peak_indices = np.array([], dtype=int)

    peak_cells = _map_cells_to_peak_columns(cells, peak_meta["peaks_df"])
    signal = _aggregate_peak_signal(peak_meta["peaks_df"], peak_indices, peak_cells, track_agg)
    peak_start = peak_meta["peak_start"][peak_indices] if peak_indices.size > 0 else np.array([], dtype=float)
    peak_end = peak_meta["peak_end"][peak_indices] if peak_indices.size > 0 else np.array([], dtype=float)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12, 5.8),
        sharex=True,
        gridspec_kw={"height_ratios": [3.6, 1.1, 0.9]},
    )
    ax_track, ax_re, ax_gene = axes

    if peak_indices.size > 0:
        widths = np.maximum(peak_end - peak_start, 1.0)
        ax_track.bar(
            peak_start,
            signal,
            width=widths,
            align="edge",
            color="#8bb8f0",
            edgecolor="#5a86c4",
            linewidth=0.3,
        )
    else:
        ax_track.text(0.5, 0.5, "No ATAC peaks in window", transform=ax_track.transAxes, ha="center", va="center")

    for _, row in re_table.iterrows():
        start = float(row["start"]) if pd.notna(row["start"]) else None
        end = float(row["end"]) if pd.notna(row["end"]) else None
        if start is None or end is None:
            continue
        if end < x_min or start > x_max:
            continue
        ax_track.axvspan(start, end, color="#f59e0b", alpha=0.22, linewidth=0)
        ax_re.axvspan(start, end, color="#f59e0b", alpha=0.65, linewidth=0)
        label_x = (start + end) / 2.0
        ax_re.text(label_x, 0.55, str(row["RE"]), rotation=30, ha="center", va="bottom", fontsize=8)

    ax_track.axvline(tg_tss, color="#dc2626", linestyle="--", linewidth=1.0)
    ax_track.set_ylabel(f"ATAC\n({track_agg})", fontsize=10)
    ax_track.set_title(f"{tf_symbol} -> {tg_gene} locus @ {stage_label}", fontsize=13)
    ax_track.spines["top"].set_visible(False)
    ax_track.spines["right"].set_visible(False)

    ax_re.set_ylim(0, 1)
    ax_re.set_yticks([])
    ax_re.set_ylabel("selected RE", fontsize=10)
    ax_re.spines["top"].set_visible(False)
    ax_re.spines["right"].set_visible(False)
    ax_re.spines["left"].set_visible(False)

    gene_y = 0.5
    gene_start = max(min(tg_start, tg_end), x_min)
    gene_end = min(max(tg_start, tg_end), x_max)
    ax_gene.hlines(gene_y, x_min, x_max, color="#9ca3af", linewidth=1.0)
    ax_gene.add_patch(
        plt.Rectangle(
            (gene_start, gene_y - 0.14),
            max(gene_end - gene_start, 1.0),
            0.28,
            facecolor="#2563eb",
            edgecolor="#1d4ed8",
            linewidth=0.8,
        )
    )
    arrow_dir = 1 if strand not in {"-", "-1", "minus", "rev", "reverse"} else -1
    arrow_x = gene_end if arrow_dir > 0 else gene_start
    ax_gene.annotate(
        "",
        xy=(arrow_x + arrow_dir * 0.03 * (x_max - x_min), gene_y),
        xytext=(arrow_x, gene_y),
        arrowprops={"arrowstyle": "->", "color": "#1d4ed8", "lw": 1.2},
    )
    ax_gene.axvline(tg_tss, color="#dc2626", linestyle="--", linewidth=1.0)
    ax_gene.text((gene_start + gene_end) / 2.0, gene_y + 0.22, f"{tg_gene} ({tg_fb})", ha="center", va="bottom", fontsize=10)
    ax_gene.set_ylim(0, 1)
    ax_gene.set_yticks([])
    ax_gene.set_xlabel(f"{tg_chr} coordinate (bp)", fontsize=10)
    ax_gene.spines["top"].set_visible(False)
    ax_gene.spines["right"].set_visible(False)
    ax_gene.spines["left"].set_visible(False)

    ax_gene.set_xlim(x_min, x_max)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _activity_table(
    tf_id: str,
    tf_symbol: str,
    tf_idx: int,
    time_labels: list[str],
    keep_masks: list[np.ndarray | None],
    label2cells: dict[str, list[str]],
    expr_df: pd.DataFrame,
    out_tf_np: np.ndarray,
    out_tg_np: np.ndarray,
    centrality_mode: str,
    topk: int,
    threshold: float,
) -> pd.DataFrame:
    norm_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def get_norm_mats(time_idx: int) -> tuple[np.ndarray, np.ndarray]:
        if time_idx not in norm_cache:
            norm_cache[time_idx] = (
                _l2norm_rows(out_tf_np[:, time_idx, :]),
                _l2norm_rows(out_tg_np[:, time_idx, :]),
            )
        return norm_cache[time_idx]

    def mean_expr_tf_for_label(label: str) -> float:
        cells = label2cells.get(str(label), [])
        if len(cells) == 0:
            return float("nan")
        if tf_id in expr_df.index:
            return float(np.asarray(expr_df.loc[tf_id, cells]).mean())
        if tf_symbol in expr_df.index:
            return float(np.asarray(expr_df.loc[tf_symbol, cells]).mean())
        return float("nan")

    rows: list[dict[str, Any]] = []
    for time_idx, label in enumerate(time_labels):
        keep = keep_masks[time_idx]
        if keep is None:
            rows.append(
                {
                    "t": time_idx,
                    "time_label": str(label),
                    "time_num": time_idx,
                    "centrality": np.nan,
                    "mean_expr": np.nan,
                    "activity": np.nan,
                }
            )
            continue
        htf_tn, htg_tn = get_norm_mats(time_idx)
        tf_vec = htf_tn[tf_idx]
        cos = htg_tn @ tf_vec
        prob = (cos + 1.0) / 2.0
        centrality = centrality_from_prob(prob[keep], mode=centrality_mode, topk=topk, thresh=threshold)
        mean_expr = mean_expr_tf_for_label(str(label))
        time_match = re.search(r"(\d+)$", str(label))
        time_num = int(time_match.group(1)) if time_match else time_idx
        rows.append(
            {
                "t": time_idx,
                "time_label": str(label),
                "time_num": time_num,
                "centrality": centrality,
                "mean_expr": mean_expr,
                "activity": centrality * mean_expr,
            }
        )
    return pd.DataFrame(rows).sort_values(["time_num", "t"], kind="mergesort").reset_index(drop=True)


def _top_tg_table(
    tf_idx: int,
    time_idx: int,
    label: str,
    keep: np.ndarray | None,
    fracs: np.ndarray | None,
    topn: int,
    user_thresh: float,
    tg_ids: list[str],
    fb2name: dict[str, str],
    out_tf_np: np.ndarray,
    out_tg_np: np.ndarray,
) -> tuple[pd.DataFrame, list[str]]:
    if keep is None or fracs is None:
        return pd.DataFrame(), []

    htf_tn = _l2norm_rows(out_tf_np[:, time_idx, :])
    htg_tn = _l2norm_rows(out_tg_np[:, time_idx, :])
    tf_vec = htf_tn[tf_idx]
    cos_all = htg_tn @ tf_vec

    cand_idx = np.where(keep)[0]
    if cand_idx.size == 0:
        return pd.DataFrame(), []
    cos_cand = cos_all[cand_idx]
    order = np.lexsort((cand_idx, -cos_cand))
    top_idx = cand_idx[order[: min(topn, len(order))]]

    rows_top: list[dict[str, Any]] = []
    for rank, tg_idx in enumerate(top_idx.tolist(), start=1):
        tg_id = str(tg_ids[tg_idx])
        rows_top.append(
            {
                "t": int(time_idx),
                "time_label": str(label),
                "rank": int(rank),
                "tg_id": tg_id,
                "cos_sim": float(cos_all[tg_idx]),
                "expr_frac_>0": float(fracs[tg_idx]) if np.isfinite(fracs[tg_idx]) else np.nan,
                "expr_ge_5pct": bool(np.isfinite(fracs[tg_idx]) and fracs[tg_idx] >= 0.05),
                "expr_le_90pct": bool(np.isfinite(fracs[tg_idx]) and fracs[tg_idx] <= 0.90),
                "expr_ge_user_thresh": bool(np.isfinite(fracs[tg_idx]) and fracs[tg_idx] >= user_thresh),
            }
        )

    table = pd.DataFrame(rows_top)
    if table.empty:
        return table, []
    table_named = table.copy()
    table_named["tg_name"] = table_named["tg_id"].map(fb2name).fillna(table_named["tg_id"])
    table_named.insert(4, "tg_id_original", table_named["tg_id"])
    table_named["tg_id"] = table_named["tg_name"]
    table_named = table_named.drop(columns=["tg_name"])
    target_names = _make_unique_labels_hash(table_named["tg_id"].astype(str).tolist())
    return table_named, target_names


def _safe_tag_from_tf_list(tf_symbols: list[str], max_len: int = 120) -> str:
    tag = "_".join(str(value) for value in tf_symbols)
    tag = re.sub(r"[^A-Za-z0-9_-]+", "_", tag)
    if len(tag) > max_len:
        tag = tag[:max_len].rstrip("_")
    return tag


def _make_unique_labels_hash(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    output: list[str] = []
    for name in names:
        count = seen.get(name, 0) + 1
        seen[name] = count
        output.append(name if count == 1 else f"{name}#{count}")
    return output


def _plot_activity_on_ax(
    ax: Any,
    df_activity: pd.DataFrame,
    tf_symbol: str,
    tf_id: str,
    fs_title: int,
    fs_other: int,
) -> None:
    ax.plot(df_activity["time_num"].values, df_activity["activity"].values, marker="o", linewidth=2)
    ax.set_title(f"{tf_symbol} (tf_id={tf_id})", fontsize=fs_title)
    ax.set_xlabel("Time Point", fontsize=fs_other)
    ax.set_ylabel("activity", fontsize=fs_other)
    ax.tick_params(axis="both", which="major", labelsize=fs_other)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _draw_regulon_network_on_ax(
    ax: Any,
    tf_symbol: str,
    targets: list[str],
    time_idx: int,
    network_seed: int,
    fs_title: int,
    fs_gene: int,
    fs_tf_center: int,
) -> None:
    count = len(targets)
    ax.set_title(f"{tf_symbol} regulon @ t={time_idx} (TopTG={count})", fontsize=fs_title)
    if count == 0:
        ax.text(0.5, 0.5, "No targets", ha="center", va="center", transform=ax.transAxes, fontsize=fs_title)
        ax.set_axis_off()
        return

    seed = int(network_seed + abs(hash(tf_symbol)) % 10_000)
    random.seed(seed)
    np.random.seed(seed)

    if count <= 40:
        re_choices = [2, 3]
    elif count <= 120:
        re_choices = [1, 2]
    else:
        re_choices = [1]

    r_out_mean = 4.8 + 0.55 * np.sqrt(count) + 1.0
    r_out_jit = 0.35
    r_in_mean = max(2.0, 0.55 * r_out_mean)
    r_in_jit = 0.25
    noise_theta = 0.10 if count <= 80 else 0.05
    tg_node_size = 260 if count <= 40 else (180 if count <= 120 else 90)
    re_node_size = 80 if count <= 120 else 45

    colors = {
        "left": "#f46d6d",
        "right": "#f0a21b",
        "bottom": "#7db5ff",
        "top": "#d78fdc",
    }

    def tg_color_by_angle(theta: float) -> str:
        angle = theta % (2.0 * np.pi)
        if np.pi / 4 <= angle < 3 * np.pi / 4:
            return colors["top"]
        if 3 * np.pi / 4 <= angle < 5 * np.pi / 4:
            return colors["left"]
        if 5 * np.pi / 4 <= angle < 7 * np.pi / 4:
            return colors["bottom"]
        return colors["right"]

    graph = nx.DiGraph()
    graph.add_node(tf_symbol, kind="tf")
    theta_base = np.linspace(0, 2 * np.pi, count, endpoint=False)
    positions: dict[str, tuple[float, float]] = {tf_symbol: (0.0, 0.0)}
    tg_colors: dict[str, str] = {}

    for idx, tg_name in enumerate(targets):
        theta = theta_base[idx] + np.random.uniform(-noise_theta, noise_theta)
        radius_tg = np.random.normal(r_out_mean, r_out_jit)
        x_tg, y_tg = radius_tg * np.cos(theta), radius_tg * np.sin(theta)
        n_re = random.choice(re_choices)
        re_nodes: list[str] = []
        for re_idx in range(n_re):
            radius_re = np.random.normal(r_in_mean, r_in_jit)
            theta_re = theta + np.random.uniform(-0.12, 0.12)
            x_re, y_re = radius_re * np.cos(theta_re), radius_re * np.sin(theta_re)
            re_name = f"RE_{idx}_{re_idx}"
            graph.add_node(re_name, kind="re")
            positions[re_name] = (x_re, y_re)
            re_nodes.append(re_name)

        graph.add_node(tg_name, kind="tg")
        positions[tg_name] = (x_tg, y_tg)
        for re_name in re_nodes:
            graph.add_edge(tf_symbol, re_name)
            graph.add_edge(re_name, tg_name)
        tg_colors[tg_name] = tg_color_by_angle(theta)

    nx.draw_networkx_edges(graph, positions, width=0.8, edge_color="black", alpha=0.5, arrows=False, ax=ax)
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=[tf_symbol],
        node_size=900,
        node_color="#ffd1b8",
        edgecolors="#444444",
        linewidths=1.8,
        node_shape="h",
        ax=ax,
    )

    re_nodes_all = [node for node in graph.nodes() if graph.nodes[node]["kind"] == "re"]
    rng = np.random.default_rng(0)
    if re_nodes_all:
        re_square_set = set(rng.choice(re_nodes_all, size=len(re_nodes_all) // 2, replace=False))
        re_square = list(re_square_set)
        re_diamond = [node for node in re_nodes_all if node not in re_square_set]
        nx.draw_networkx_nodes(
            graph,
            positions,
            nodelist=re_square,
            node_size=re_node_size,
            node_color="#e6b0b0",
            edgecolors="#e6b0b0",
            node_shape="s",
            ax=ax,
        )
        nx.draw_networkx_nodes(
            graph,
            positions,
            nodelist=re_diamond,
            node_size=re_node_size,
            node_color="#e6b0b0",
            edgecolors="#e6b0b0",
            node_shape="D",
            ax=ax,
        )

    tg_nodes = [node for node in graph.nodes() if graph.nodes[node]["kind"] == "tg"]
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=tg_nodes,
        node_size=tg_node_size,
        node_color=[tg_colors[node] for node in tg_nodes],
        edgecolors="#ffffff",
        linewidths=1.0,
        ax=ax,
    )
    nx.draw_networkx_labels(
        graph,
        positions,
        labels={tf_symbol: tf_symbol.upper()},
        font_size=fs_tf_center,
        font_weight="bold",
        ax=ax,
    )
    for node in tg_nodes:
        x_pos, y_pos = positions[node]
        ax.text(x_pos, y_pos, node, fontsize=fs_gene, fontweight="normal", ha="center", va="center", zorder=13)

    ax.set_aspect("equal")
    ax.set_axis_off()


def _single_tf_context(config: dict[str, Any]) -> dict[str, Any]:
    lineage = str(config.get("lineage", "")).lower()
    if lineage in {"germ", "germline"}:
        inference = reload_germ_from_checkpoint(config)
        common = inference.get("artifacts", {}).get("common")
        if not isinstance(common, dict):
            common = prepare_common_inputs(config)
    elif lineage in {"soma", "somatic"}:
        inference = reload_soma_from_checkpoint(config)
        common = prepare_common_inputs(config)
    else:
        raise ValueError(f"Unsupported lineage '{config.get('lineage')}'. Expected soma or germ.")

    maps = inference.get("maps")
    if not isinstance(maps, dict):
        raise RuntimeError("Checkpoint/artifact did not expose node maps required for single-TF analysis.")

    time_count = int(inference["out_tf"].size(1))
    return {
        "lineage": lineage,
        "inference": inference,
        "common": common,
        "maps": maps,
        "time_count": time_count,
        "time_labels": _time_labels(lineage, inference, time_count),
    }


def run_single_tf_analysis_from_config(config: dict[str, Any]) -> dict[str, Any]:
    settings = _single_tf_settings(config)
    if not settings["tf_symbols"]:
        raise ValueError("single_tf.tf_symbol or single_tf.tf_symbols must be provided.")

    context = _single_tf_context(config)
    output_dir = ensure_directory(resolve_config_path(config, settings["output_dir"]))
    common = context["common"]
    inference = context["inference"]
    maps = context["maps"]
    expr_df = common["expr_df"]
    time_labels = context["time_labels"]
    stage_labels = _stage_labels_for_export(settings, time_labels, context["time_count"])
    label2cells = _label_to_cells(common)
    fb2name = build_fbgn_to_name(common)
    tf_ids = [str(value) for value in maps["tf"]]
    re_names = [str(value) for value in maps["re"]]
    tg_ids = [str(value) for value in maps["tg"]]
    tf_display_names = [str(fb2name.get(tf_id, tf_id)) for tf_id in tf_ids]
    re_chr, re_center, _chr2reidx_unused = _build_re_metadata(re_names)
    tg_meta = _build_tg_metadata(common, tg_ids)
    peak_meta = _build_peak_metadata(common) if settings["locus_plot"]["enabled"] else None
    locus_dir = (
        ensure_directory(output_dir / settings["locus_plot"]["output_dir"])
        if settings["locus_plot"]["enabled"]
        else None
    )

    keep_masks, frac_list = _prepare_keep_masks(
        time_labels=time_labels,
        tg_ids=tg_ids,
        expr_df=expr_df,
        label2cells=label2cells,
        fb2name=fb2name,
        min_expr_frac=settings["min_expr_frac"],
        max_expr_frac=settings["max_expr_frac"],
    )

    out_tf_np = inference["out_tf"].detach().cpu().numpy()
    out_re_np = inference["out_re"].detach().cpu().numpy()
    out_tg_np = inference["out_tg"].detach().cpu().numpy()

    results: list[dict[str, Any]] = []
    saved_tables: list[str] = []
    notebook_rows: list[pd.DataFrame] = []
    locus_saved_figures: list[str] = []
    locus_saved_tables: list[str] = []
    for tf_symbol in settings["tf_symbols"]:
        tf_id, tf_idx = resolve_tf_id_and_index(tf_symbol, tf_ids, tf_display_names, common)
        df_activity = _activity_table(
            tf_id=tf_id,
            tf_symbol=tf_symbol,
            tf_idx=tf_idx,
            time_labels=time_labels,
            keep_masks=keep_masks,
            label2cells=label2cells,
            expr_df=expr_df,
            out_tf_np=out_tf_np,
            out_tg_np=out_tg_np,
            centrality_mode=settings["centrality_mode"],
            topk=settings["topk"],
            threshold=settings["threshold"],
        )
        time_idx_top = resolve_time_index(
            settings["time_for_top"],
            time_labels,
            context["time_count"],
            tf_symbol=tf_symbol,
        )
        stage_label = str(stage_labels[time_idx_top])
        cells_for_stage = label2cells.get(str(time_labels[time_idx_top]), [])
        re_activity_frac = _re_activity_fraction_for_stage(common, re_names, cells_for_stage)
        top_tg_idx = _top_tg_indices(
            tf_idx=tf_idx,
            time_idx=time_idx_top,
            keep=keep_masks[time_idx_top],
            topn=resolve_topn(settings["topn_tg"], tf_symbol),
            out_tf_np=out_tf_np,
            out_tg_np=out_tg_np,
        )
        df_top, targets = _top_tg_table(
            tf_idx=tf_idx,
            time_idx=time_idx_top,
            label=str(time_labels[time_idx_top]),
            keep=keep_masks[time_idx_top],
            fracs=frac_list[time_idx_top],
            topn=resolve_topn(settings["topn_tg"], tf_symbol),
            user_thresh=settings["user_thresh"],
            tg_ids=tg_ids,
            fb2name=fb2name,
            out_tf_np=out_tf_np,
            out_tg_np=out_tg_np,
        )
        df_tg_re = _top_tg_re_table(
            tf_symbol=str(tf_symbol),
            tf_idx=tf_idx,
            time_idx=time_idx_top,
            stage_label=stage_label,
            top_tg_idx=top_tg_idx,
            tg_ids=tg_ids,
            re_names=re_names,
            re_chr=re_chr,
            re_center=re_center,
            re_activity_frac=re_activity_frac,
            tg_meta=tg_meta,
            out_tf_np=out_tf_np,
            out_re_np=out_re_np,
            out_tg_np=out_tg_np,
            active_frac_re=settings["active_frac_re"],
            topn_re=settings["topn_re"],
            max_dist_bp=settings["max_dist_bp"],
            dist_lambda=settings["dist_lambda"],
            w_re_tg=settings["w_re_tg"],
            w_dist=settings["w_dist"],
            w_tf_re=settings["w_tf_re"],
            w_act=settings["w_act"],
        )

        if settings["save_tables"]:
            top_path = output_dir / f"{safe_label(tf_symbol)}_top_tg.tsv"
            df_top.to_csv(top_path, sep="\t", index=False)
            saved_tables.append(str(top_path))
            notebook_path = output_dir / f"{settings['output_prefix']}_{safe_label(tf_symbol)}_{_stage_file_token(stage_label)}.csv"
            df_tg_re.to_csv(notebook_path, index=False)
            saved_tables.append(str(notebook_path))
        notebook_rows.append(df_tg_re)

        locus_outputs: list[dict[str, Any]] = []
        if settings["locus_plot"]["enabled"] and locus_dir is not None and peak_meta is not None:
            requested_tgs = settings["locus_plot"]["target_tgs"]
            if requested_tgs:
                resolved_targets = _resolve_requested_tg_indices(requested_tgs, tg_ids, tg_meta, fb2name)
            else:
                fallback_idx = top_tg_idx[:1].tolist()
                resolved_targets = {}
                for tg_i in fallback_idx:
                    tg_fb = str(tg_ids[tg_i])
                    tg_gene = str((tg_meta.get(tg_fb) or {}).get("gene", fb2name.get(tg_fb, tg_fb)))
                    resolved_targets[tg_gene] = int(tg_i)

            for requested_name, tg_idx_sel in resolved_targets.items():
                df_locus_re = _locus_re_table(
                    tf_symbol=str(tf_symbol),
                    tf_idx=tf_idx,
                    stage_label=stage_label,
                    time_idx=time_idx_top,
                    tg_idx=int(tg_idx_sel),
                    tg_ids=tg_ids,
                    tg_meta=tg_meta,
                    fb2name=fb2name,
                    re_names=re_names,
                    re_chr=re_chr,
                    re_center=re_center,
                    re_activity_frac=re_activity_frac,
                    out_tf_np=out_tf_np,
                    out_re_np=out_re_np,
                    out_tg_np=out_tg_np,
                    active_frac_re=settings["active_frac_re"],
                    topn_re=settings["topn_re"],
                    max_dist_bp=settings["max_dist_bp"],
                    dist_lambda=settings["dist_lambda"],
                    w_re_tg=settings["w_re_tg"],
                    w_dist=settings["w_dist"],
                    w_tf_re=settings["w_tf_re"],
                    w_act=settings["w_act"],
                )
                tg_fb = str(tg_ids[int(tg_idx_sel)])
                tg_gene = str((tg_meta.get(tg_fb) or {}).get("gene", fb2name.get(tg_fb, tg_fb)))
                tg_tag = safe_label(tg_gene)
                stage_tag = _stage_file_token(stage_label)
                locus_prefix = f"{safe_label(tf_symbol)}_{tg_tag}_{stage_tag}"
                table_path = locus_dir / f"{locus_prefix}_locus_re.tsv"
                fig_path = locus_dir / f"{locus_prefix}_locus_atac.png"
                if settings["locus_plot"]["save_table"]:
                    df_locus_re.to_csv(table_path, sep="\t", index=False)
                    locus_saved_tables.append(str(table_path))
                if settings["locus_plot"]["save_fig"]:
                    _plot_single_tf_locus(
                        tf_symbol=str(tf_symbol),
                        stage_label=stage_label,
                        time_idx=time_idx_top,
                        tg_idx=int(tg_idx_sel),
                        tg_ids=tg_ids,
                        tg_meta=tg_meta,
                        fb2name=fb2name,
                        peak_meta=peak_meta,
                        re_table=df_locus_re,
                        cells=cells_for_stage,
                        output_path=fig_path,
                        dpi=settings["locus_plot"]["dpi"],
                        window_upstream_bp=settings["locus_plot"]["window_upstream_bp"],
                        window_downstream_bp=settings["locus_plot"]["window_downstream_bp"],
                        track_agg=settings["locus_plot"]["track_agg"],
                    )
                    locus_saved_figures.append(str(fig_path))
                locus_outputs.append(
                    {
                        "requested_tg": str(requested_name),
                        "tg_idx": int(tg_idx_sel),
                        "tg_id": tg_fb,
                        "tg_gene": tg_gene,
                        "num_selected_re": int(len(df_locus_re)),
                        "table_path": str(table_path) if settings["locus_plot"]["save_table"] else None,
                        "figure_path": str(fig_path) if settings["locus_plot"]["save_fig"] else None,
                    }
                )

        results.append(
            {
                "tf": str(tf_symbol),
                "tf_id": str(tf_id),
                "tf_idx": int(tf_idx),
                "df_activity": df_activity,
                "df_top": df_top,
                "df_tg_re": df_tg_re,
                "t_top": int(time_idx_top),
                "stage": stage_label,
                "targets": targets,
                "locus_outputs": locus_outputs,
            }
        )

    if settings["save_tables"]:
        df_all = pd.concat(notebook_rows, axis=0, ignore_index=True) if notebook_rows else pd.DataFrame(
            columns=["TF", "stage", "t_idx", "TG", "RE", "dist_REcenter_to_TSS_bp"]
        )
        all_path = output_dir / f"{settings['output_prefix']}_ALL.csv"
        df_all.to_csv(all_path, index=False)
        saved_tables.append(str(all_path))

    tf_tag = _safe_tag_from_tf_list(settings["tf_symbols"])
    saved_figures: list[str] = []

    n_items = len(results)
    ncols = 2
    nrows = int(math.ceil(n_items / ncols))

    fig_regulon, axes_regulon = plt.subplots(nrows, ncols, figsize=(7.5 * ncols, 7.5 * nrows), squeeze=False)
    for idx, result in enumerate(results):
        ax = axes_regulon[idx // ncols][idx % ncols]
        _draw_regulon_network_on_ax(
            ax,
            result["tf"],
            result["targets"],
            result["t_top"],
            network_seed=settings["network_seed"],
            fs_title=settings["fs_title"],
            fs_gene=settings["fs_gene"],
            fs_tf_center=settings["fs_tf_center"],
        )
    for idx in range(n_items, nrows * ncols):
        axes_regulon[idx // ncols][idx % ncols].set_axis_off()
    fig_regulon.suptitle("Regulon network (TopTG) per TF", fontsize=settings["fs_title"], y=1.02)
    fig_regulon.tight_layout()
    if settings["save_fig"]:
        if len(results) == 1:
            regulon_name = f"{safe_label(results[0]['tf'])}_regulon.tif"
        else:
            regulon_name = f"{tf_tag}_regulon.tif"
        regulon_path = output_dir / regulon_name
        fig_regulon.savefig(regulon_path, dpi=settings["network_dpi"], bbox_inches="tight")
        saved_figures.append(str(regulon_path))
        if len(results) == 1:
            named_copy = (
                f"{safe_label(results[0]['tf'])}_t{int(results[0]['t_top'])}_"
                f"network_allFilteredTG_n{len(results[0]['targets'])}.tif"
            )
            named_copy_path = output_dir / named_copy
            fig_regulon.savefig(named_copy_path, dpi=settings["network_dpi"], bbox_inches="tight")
            saved_figures.append(str(named_copy_path))
    plt.close(fig_regulon)

    report = {
        "lineage": context["lineage"],
        "output_dir": str(output_dir),
        "checkpoint_path": inference["checkpoint_path"],
        "tf_symbols": settings["tf_symbols"],
        "saved_tables": saved_tables + locus_saved_tables,
        "saved_figures": saved_figures + locus_saved_figures,
        "results": [
            {
                "tf": str(result["tf"]),
                "tf_id": str(result["tf_id"]),
                "tf_idx": int(result["tf_idx"]),
                "t_top": int(result["t_top"]),
                "stage": str(result["stage"]),
                "top_targets": result["targets"],
                "num_tf_re_rows": int(len(result["df_tg_re"])),
                "locus_outputs": result["locus_outputs"],
            }
            for result in results
        ],
        "settings": settings,
    }
    save_checkpoint_metadata(output_dir / "single_tf_report.json", report)
    return report


def run_single_tf_analysis(config_path: str | Path) -> dict[str, Any]:
    config = load_yaml_config(config_path)
    report = run_single_tf_analysis_from_config(config)
    report["config_path"] = str(Path(config_path).resolve())
    return report
