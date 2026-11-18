"""Stage network export from a checkpoint and network-file helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ..checkpoint import save_checkpoint_metadata
from ..infer import reload_germ_from_checkpoint, reload_soma_from_checkpoint
from ..io import InputPaths, strip_copy_suffix
from ..preprocess import fix_dash, prepare_common_inputs
from ..utils import ensure_directory, load_yaml_config, resolve_config_path


def _section(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    return value if isinstance(value, dict) else {}


def safe_label(value: str) -> str:
    return re.sub(r'[\\/:*?"<>|\s]+', "_", str(value))


def natural_key(value: str) -> list[Any]:
    return [int(item) if item.isdigit() else item.lower() for item in re.split(r"(\d+)", str(value))]


def _resolve_relative_path(base_dir: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _network_settings(config: dict[str, Any]) -> dict[str, Any]:
    section = _section(config, "network")
    lineage = str(config.get("lineage", "")).lower()
    default_score_mode = "score_cos" if lineage in {"germ", "germline"} else "none"
    if "strict_score_mode" in section:
        strict_score_mode = str(section.get("strict_score_mode", default_score_mode)).strip().lower()
    elif "include_scores" in section:
        strict_score_mode = "score_cos_prob" if bool(section.get("include_scores", False)) else "none"
    else:
        strict_score_mode = default_score_mode
    return {
        "output_dir": str(section.get("output_dir", f"result/{lineage}/network")),
        "tf_filter_file": str(section.get("tf_filter_file", "TF_filtered.tsv")),
        "active_frac_tf_tg": float(section.get("active_frac_tf_tg", 0.11)),
        "active_frac_re": float(section.get("active_frac_re", 0.05)),
        "top_k_global": int(section.get("top_k_global", 10_000)),
        "min_score_keep": float(section.get("min_score_keep", 0.83)),
        "m_min": int(section.get("m_min", 0)),
        "m_max": int(section.get("m_max", int(int(section.get("top_k_global", 10_000)) * 0.2))),
        "gamma": float(section.get("gamma", 3.0)),
        "eps": float(section.get("eps", 1e-12)),
        "k_nearest_re": int(section.get("k_nearest_re", 10)),
        "topn_re_per_tg": int(section.get("topn_re_per_tg", 2)),
        "max_dist_bp": int(section.get("max_dist_bp", 250_000)),
        "dist_lambda": float(section.get("dist_lambda", 50_000.0)),
        "w_re_tg": float(section.get("w_re_tg", 0.40)),
        "w_dist": float(section.get("w_dist", 0.30)),
        "w_tf_re": float(section.get("w_tf_re", 0.20)),
        "w_act": float(section.get("w_act", 0.10)),
        "strict_score_mode": strict_score_mode,
        "save_report": bool(section.get("save_report", True)),
    }


def _dense_network_settings(config: dict[str, Any]) -> dict[str, Any]:
    network = _section(config, "network")
    section = _section(network, "dense")
    lineage = str(config.get("lineage", "")).lower()
    default_m_max = 0 if lineage in {"soma", "somatic"} else 5
    default_output = f"result/{lineage}/network_dense"
    return {
        "enabled": bool(section.get("enabled", False)),
        "output_dir": str(section.get("output_dir", default_output)),
        "n_interp": int(section.get("n_interp", 4)),
        "sigma_time": float(section.get("sigma_time", 0.7)),
        "active_thr": float(section.get("active_thr", 0.11)),
        "min_score_keep": float(section.get("min_score_keep", 0.82)),
        "top_k_global": int(section.get("top_k_global", 50_000)),
        "rescue_gamma": float(section.get("rescue_gamma", 4.0)),
        "rescue_m_min": int(section.get("rescue_m_min", 0)),
        "rescue_m_max": int(section.get("rescue_m_max", default_m_max)),
        "rescue_eps": float(section.get("rescue_eps", 1e-8)),
        "use_one_based": bool(section.get("use_one_based", False)),
        "keep_id_cols": bool(section.get("keep_id_cols", True)),
        "add_prefix": bool(section.get("add_prefix", False)),
    }


def _safe_get_col(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def _pick_col_ci(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {str(column).lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def _select_valid_cols(
    tf_x: torch.Tensor,
    re_x: torch.Tensor,
    tg_x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    valid = (tf_x != 0).any(dim=0) | (re_x != 0).any(dim=0) | (tg_x != 0).any(dim=0)
    if valid.ndim == 1 and bool(valid.any()):
        return tf_x[:, valid], re_x[:, valid], tg_x[:, valid]
    return tf_x, re_x, tg_x


def _cosine_matrix(left: torch.Tensor, right: torch.Tensor, chunk: int = 1024) -> torch.Tensor:
    try:
        return left @ right.T
    except RuntimeError:
        rows: list[torch.Tensor] = []
        for start in range(0, left.size(0), chunk):
            rows.append(left[start : start + chunk] @ right.T)
        return torch.cat(rows, dim=0)


def _norm_key(value: str) -> str:
    return re.sub(r"\.\d+$", "", str(value).strip())


def _clean_chr(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null", "na", ".", "-1"}:
        return None
    return text.replace("chr", "")


def _to_int_or_none(value: Any) -> int | None:
    try:
        if pd.isna(value):
            return None
        return int(float(str(value)))
    except Exception:
        return None


def parse_re_interval(value: str) -> tuple[str, int, int] | None:
    text = fix_dash(str(value).strip())
    match = re.match(r"^([^:]+):\s*([0-9]+)\s*-\s*([0-9]+)$", text)
    if not match:
        return None
    chrom, start, end = match.group(1), int(match.group(2)), int(match.group(3))
    if start > end:
        start, end = end, start
    chrom = _clean_chr(chrom)
    if chrom is None:
        return None
    return chrom, start, end


def _read_ids_and_names(path: Path, expected_len: int, node_type: str) -> tuple[list[str], list[str]]:
    frame = pd.read_csv(path, sep="\t", dtype=str)
    col_id = _safe_get_col(frame, ["FBgn_ID", "FBgn", "fbgn", "gene_id", "id"])
    col_name = _safe_get_col(frame, ["gene_name", "symbol", "gene"])
    if col_id is not None:
        ids = frame[col_id].astype(str).tolist()
    elif col_name is not None:
        ids = frame[col_name].astype(str).tolist()
    else:
        ids = frame.iloc[:, 0].astype(str).tolist()
    if col_name is not None:
        names = frame[col_name].astype(str).tolist()
    elif col_id is not None:
        names = frame[col_id].astype(str).tolist()
    else:
        names = frame.iloc[:, 0].astype(str).tolist()
    if len(ids) < expected_len or len(names) < expected_len:
        raise ValueError(
            f"{node_type} reference length is shorter than expected: ids={len(ids)}, names={len(names)}, expected={expected_len}."
        )
    if node_type.upper() == "TF":
        names = [strip_copy_suffix(item) for item in names]
    return [str(item) for item in ids[:expected_len]], [str(item) for item in names[:expected_len]]


def _read_re_names(path: Path, expected_len: int) -> list[str]:
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(names) < expected_len:
        raise ValueError(f"RE list at {path} is shorter than expected ({len(names)} < {expected_len}).")
    return [str(item) for item in names[:expected_len]]


def _load_tf_filter(path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path, sep="\t", dtype=str)
    col_gene = _safe_get_col(frame, ["gene_name", "symbol", "gene"])
    col_fbgn = _safe_get_col(frame, ["FBgn_ID", "FBgn", "fbgn"])
    if col_gene is None and col_fbgn is None:
        raise ValueError(f"{path} must contain gene_name/symbol/gene or FBgn_ID/FBgn/fbgn.")
    if col_gene is not None:
        frame[col_gene] = frame[col_gene].astype(str).map(strip_copy_suffix)
    allowed_symbol = set(frame[col_gene].astype(str)) if col_gene is not None else set()
    allowed_fbgn = set(frame[col_fbgn].astype(str)) if col_fbgn is not None else set()
    fbgn2symbol = (
        dict(zip(frame[col_fbgn].astype(str), frame[col_gene].astype(str)))
        if col_gene is not None and col_fbgn is not None
        else {}
    )
    return {
        "frame": frame,
        "allowed_symbol": allowed_symbol,
        "allowed_fbgn": allowed_fbgn,
        "fbgn2symbol": fbgn2symbol,
    }


def _load_tg_meta(path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path, sep="\t", dtype=str)
    col_fb = _pick_col_ci(frame, ["fbgn_id", "fbgn", "fbgnid"])
    col_gene = _pick_col_ci(frame, ["gene_name", "symbol", "gene", "name"])
    col_chr = _pick_col_ci(frame, ["chr", "chrom", "chromosome"])
    col_start = _pick_col_ci(frame, ["start", "tss", "tx_start"])
    col_end = _pick_col_ci(frame, ["end", "tx_end"])
    col_strand = _pick_col_ci(frame, ["strand", "orientation", "sense"])
    if not (col_chr and col_start and col_end):
        raise ValueError(f"{path} must contain chr/start/end columns.")

    tg_meta: dict[str, dict[str, Any]] = {}
    fbgn2symbol: dict[str, str] = {}
    for _, row in frame.iterrows():
        fbgn = str(row[col_fb]).strip() if col_fb is not None and pd.notna(row[col_fb]) else None
        gene = str(row[col_gene]).strip() if col_gene is not None and pd.notna(row[col_gene]) else None
        chrom = _clean_chr(row[col_chr])
        start = _to_int_or_none(row[col_start])
        end = _to_int_or_none(row[col_end])
        strand = str(row[col_strand]).strip() if col_strand is not None and pd.notna(row[col_strand]) else None
        tss = None
        if chrom is not None and start is not None and end is not None and start >= 0 and end >= 0:
            if strand in {"-", "-1", "minus", "rev", "reverse"}:
                tss = end
            else:
                tss = start
        meta = {"fbgn": fbgn, "gene": gene, "chr": chrom, "tss": tss}
        if fbgn:
            tg_meta[_norm_key(fbgn)] = meta
            if gene:
                fbgn2symbol[_norm_key(fbgn)] = gene
        if gene:
            tg_meta[_norm_key(gene)] = meta
    return {"frame": frame, "tg_meta": tg_meta, "fbgn2symbol": fbgn2symbol}


def _resolve_tg_meta(tg_id: str, tg_meta: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    key = _norm_key(tg_id)
    if key in tg_meta:
        return tg_meta[key]
    match = re.search(r"\((FBgn[0-9]+)\)", key)
    if match:
        alt = _norm_key(match.group(1))
        if alt in tg_meta:
            return tg_meta[alt]
    if "|" in key:
        for part in key.split("|"):
            alt = _norm_key(part)
            if alt in tg_meta:
                return tg_meta[alt]
    return None


def _build_re_coordinate_arrays(re_names: list[str]) -> dict[str, np.ndarray]:
    re_chr = np.empty(len(re_names), dtype=object)
    re_center = np.full(len(re_names), np.nan, dtype=float)
    for idx, name in enumerate(re_names):
        parsed = parse_re_interval(name)
        if parsed is None:
            re_chr[idx] = None
            continue
        chrom, start, end = parsed
        re_chr[idx] = chrom
        re_center[idx] = (start + end) / 2.0
    return {"re_chr": re_chr, "re_center": re_center}


def _candidate_active_re_indices(
    tg_chr: str | None,
    tg_tss: float | int | None,
    act_re_chr: np.ndarray,
    act_re_center: np.ndarray,
    act_re_finite: np.ndarray,
    max_dist_bp: int,
) -> np.ndarray:
    if tg_chr is not None and tg_tss is not None:
        rel_pool = np.where(act_re_finite & (act_re_chr == tg_chr))[0]
        if rel_pool.size > 0:
            abs_dist = np.abs(act_re_center[rel_pool] - float(tg_tss))
            rel_pool = rel_pool[abs_dist <= float(max_dist_bp)]
        if rel_pool.size > 0:
            return np.asarray(rel_pool, dtype=int)

    if tg_chr is not None:
        rel_pool = np.where(act_re_chr == tg_chr)[0]
        if rel_pool.size > 0:
            return np.asarray(rel_pool, dtype=int)

    return np.arange(len(act_re_chr), dtype=int)


def _select_re_indices_composite(
    tf_vec: torch.Tensor,
    tg_vec: torch.Tensor,
    re_embed: torch.Tensor,
    candidate_rel: np.ndarray,
    act_re_center: np.ndarray,
    act_re_frac: np.ndarray,
    active_frac_re: float,
    tg_tss: float | int | None,
    topn_re_per_tg: int,
    dist_lambda: float,
    w_re_tg: float,
    w_dist: float,
    w_tf_re: float,
    w_act: float,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    if topn_re_per_tg <= 0 or candidate_rel.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    if isinstance(re_embed, torch.Tensor):
        device = re_embed.device
        dtype = re_embed.dtype
    else:
        device = torch.device("cpu")
        dtype = torch.float32
        re_embed = torch.as_tensor(re_embed, device=device, dtype=dtype)

    if not isinstance(tf_vec, torch.Tensor):
        tf_vec = torch.as_tensor(tf_vec, device=device, dtype=dtype)
    else:
        tf_vec = tf_vec.to(device=device, dtype=dtype)
    if not isinstance(tg_vec, torch.Tensor):
        tg_vec = torch.as_tensor(tg_vec, device=device, dtype=dtype)
    else:
        tg_vec = tg_vec.to(device=device, dtype=dtype)

    candidate_tensor = torch.as_tensor(candidate_rel, device=device, dtype=torch.long)
    s_re_tg = (re_embed[candidate_tensor] @ tg_vec + 1.0) / 2.0
    s_tf_re = (re_embed[candidate_tensor] @ tf_vec + 1.0) / 2.0

    if tg_tss is not None:
        d_abs = np.abs(act_re_center[candidate_rel] - float(tg_tss))
        s_dist_np = np.full(candidate_rel.size, 0.5, dtype=np.float32)
        valid = np.isfinite(d_abs)
        s_dist_np[valid] = np.exp(-d_abs[valid] / float(dist_lambda))
        s_dist = torch.as_tensor(s_dist_np, device=device, dtype=dtype)
    else:
        s_dist = torch.full((candidate_rel.size,), 0.5, device=device, dtype=dtype)

    s_act_np = np.clip(act_re_frac[candidate_rel], float(active_frac_re), 1.0)
    s_act_np = (s_act_np - float(active_frac_re)) / (1.0 - float(active_frac_re) + float(eps))
    s_act = torch.as_tensor(s_act_np, device=device, dtype=dtype)

    score_local = (
        float(w_re_tg) * s_re_tg
        + float(w_dist) * s_dist
        + float(w_tf_re) * s_tf_re
        + float(w_act) * s_act
    )
    keep = min(int(topn_re_per_tg), int(score_local.numel()))
    if keep <= 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    top_pos = torch.topk(score_local, k=keep).indices
    top_rel = candidate_rel[top_pos.detach().cpu().numpy()]
    top_score = score_local[top_pos].detach().cpu().numpy().astype(float)
    return np.asarray(top_rel, dtype=int), np.asarray(top_score, dtype=float)


def _strict_tftg_filename(lineage: str) -> str:
    _ = str(lineage).lower()
    return "t{t:02d}_TF_TG.tsv"


def _strict_tfretg_filename() -> str:
    return "t{t:02d}_TF_RE_TG.tsv"


def _format_network_time_tag(value: str | float | int) -> str:
    try:
        numeric = float(str(value).strip())
    except Exception:
        return safe_label(value)
    if abs(numeric - round(numeric)) < 1e-8:
        return f"{int(round(numeric)):02d}"
    if numeric < 100:
        return f"{numeric:05.2f}"
    return f"{numeric:.2f}"


def _dense_tftg_filename(label: str) -> str:
    return f"TF_TG_dense_t{safe_label(label)}.tsv"


def _tf_names_for_display(tf_ids: list[str], tf_names_file: list[str] | None, fbgn2symbol: dict[str, str]) -> list[str]:
    output: list[str] = []
    for idx, tf_id in enumerate(tf_ids):
        if tf_id in fbgn2symbol:
            output.append(strip_copy_suffix(fbgn2symbol[tf_id]))
        elif tf_names_file is not None and idx < len(tf_names_file):
            output.append(strip_copy_suffix(tf_names_file[idx]))
        else:
            output.append(str(tf_id))
    return output


def _pick_reference_names(
    maps: dict[str, list[str]] | None,
    paths: InputPaths,
    num_tf: int,
    num_tg: int,
    num_re: int,
    tf_filter: dict[str, Any],
    tg_meta_bundle: dict[str, Any],
) -> dict[str, Any]:
    if maps and "tf" in maps and len(maps["tf"]) == num_tf:
        tf_ids = [str(item) for item in maps["tf"]]
        tf_names_file: list[str] | None = None
    else:
        tf_ids, tf_names_file = _read_ids_and_names(paths.tf_list, num_tf, "TF")

    if maps and "tg" in maps and len(maps["tg"]) == num_tg:
        tg_ids = [str(item) for item in maps["tg"]]
        tg_names_file: list[str] | None = None
    else:
        tg_ids, tg_names_file = _read_ids_and_names(paths.tg_list, num_tg, "TG")

    if maps and "re" in maps and len(maps["re"]) == num_re:
        re_names = [str(item) for item in maps["re"]]
    else:
        re_names = _read_re_names(paths.re_list, num_re)

    tf_names = _tf_names_for_display(tf_ids, tf_names_file, tf_filter["fbgn2symbol"])
    if tg_names_file is not None:
        tg_names = [str(item) for item in tg_names_file]
    else:
        tg_names = [tg_meta_bundle["fbgn2symbol"].get(_norm_key(item), str(item)) for item in tg_ids]

    return {
        "tf_ids": tf_ids,
        "tf_names": tf_names,
        "tg_ids": tg_ids,
        "tg_names": tg_names,
        "re_names": re_names,
    }


def _allowed_tf_indices(tf_ids: list[str], tf_names: list[str], tf_filter: dict[str, Any]) -> list[int]:
    allowed: list[int] = []
    for idx, (tf_id, tf_name) in enumerate(zip(tf_ids, tf_names)):
        if tf_id in tf_filter["allowed_fbgn"] or tf_name in tf_filter["allowed_symbol"]:
            allowed.append(idx)
    if not allowed:
        raise ValueError("TF filter has no overlap with the graph TF namespace.")
    return allowed


def _build_tf_frac_df(
    graphs_sel: list[Any],
    tf_ids: list[str],
    allowed_tf_indices: list[int],
) -> tuple[pd.DataFrame, dict[int, pd.Series]]:
    tf_frac_by_t: dict[int, pd.Series] = {}
    for time_idx, graph in enumerate(graphs_sel):
        tf_x, re_x, tg_x = graph["TF"].x, graph["RE"].x, graph["TG"].x
        tf_valid, _, _ = _select_valid_cols(tf_x, re_x, tg_x)
        frac = (tf_valid != 0).float().mean(dim=1).detach().cpu().numpy()
        tf_frac_by_t[time_idx] = pd.Series(frac, index=tf_ids)
    tf_frac_df = pd.DataFrame(tf_frac_by_t).fillna(0.0)
    tf_frac_df = tf_frac_df.loc[[tf_ids[idx] for idx in allowed_tf_indices]]
    other_max: dict[int, pd.Series] = {}
    for time_idx in range(len(graphs_sel)):
        cols_other = [column for column in tf_frac_df.columns if column != time_idx]
        if cols_other:
            other_max[time_idx] = tf_frac_df[cols_other].max(axis=1)
        else:
            other_max[time_idx] = pd.Series(0.0, index=tf_frac_df.index)
    return tf_frac_df, other_max


def _tftg_rows_to_frame(
    top_edges: list[tuple[int, int, float]],
    time_idx: int,
    tf_ids: list[str],
    tf_names: list[str],
    tg_ids: list[str],
    tg_names: list[str],
    score_mode: str,
) -> pd.DataFrame:
    tf_idx = np.array([item[0] for item in top_edges], dtype=int)
    tg_idx = np.array([item[1] for item in top_edges], dtype=int)
    score_cos = np.array([item[2] for item in top_edges], dtype=float)
    payload: dict[str, Any] = {
        "time_idx": np.full(len(top_edges), time_idx, dtype=int),
        "TF_idx": tf_idx,
        "TG_idx": tg_idx,
        "TF_id": [tf_ids[item] for item in tf_idx.tolist()],
        "TF_name": [tf_names[item] for item in tf_idx.tolist()],
        "TG_id": [tg_ids[item] for item in tg_idx.tolist()],
        "TG_name": [tg_names[item] for item in tg_idx.tolist()],
    }
    normalized = str(score_mode).strip().lower()
    if normalized in {"score_cos", "score_cos_prob"}:
        payload["score_cos"] = score_cos
    if normalized == "score_cos_prob":
        payload["prob"] = (score_cos + 1.0) / 2.0
    return pd.DataFrame(payload)


def _export_strict_networks(
    lineage: str,
    output_dir: Path,
    graphs_sel: list[Any],
    out_tf: torch.Tensor,
    out_re: torch.Tensor,
    out_tg: torch.Tensor,
    refs: dict[str, Any],
    tf_filter: dict[str, Any],
    tg_meta_bundle: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    num_tf = int(out_tf.size(0))
    allowed_tf_indices = _allowed_tf_indices(refs["tf_ids"], refs["tf_names"], tf_filter)
    allowed_tf_idx_cpu = torch.tensor(allowed_tf_indices, dtype=torch.long)
    allowed_tf_mask_cpu = torch.zeros(num_tf, dtype=torch.bool)
    allowed_tf_mask_cpu[allowed_tf_idx_cpu] = True

    tf_frac_df, other_max_cache = _build_tf_frac_df(graphs_sel, refs["tf_ids"], allowed_tf_indices)
    re_coord = _build_re_coordinate_arrays(refs["re_names"])

    summary_rows: list[dict[str, Any]] = []
    saved_tftg: list[str] = []
    saved_tfretg: list[str] = []
    tftg_filename_template = _strict_tftg_filename(lineage)
    tfretg_filename_template = _strict_tfretg_filename()

    for time_idx, graph in enumerate(graphs_sel):
        tf_x, re_x, tg_x = graph["TF"].x, graph["RE"].x, graph["TG"].x
        tf_valid, re_valid, tg_valid = _select_valid_cols(tf_x, re_x, tg_x)

        allowed_tf_mask = allowed_tf_mask_cpu.to(tf_x.device)
        allowed_tf_idx = allowed_tf_idx_cpu.to(tf_x.device)

        itf = torch.where((tf_valid != 0).float().mean(dim=1) >= settings["active_frac_tf_tg"])[0]
        ire = torch.where((re_valid != 0).float().mean(dim=1) >= settings["active_frac_re"])[0]
        itg = torch.where((tg_valid != 0).float().mean(dim=1) >= settings["active_frac_tf_tg"])[0]

        itf = itf[allowed_tf_mask[itf]]
        if itf.numel() == 0:
            itf = allowed_tf_idx.clone()
        if ire.numel() == 0:
            ire = torch.arange(re_x.size(0), device=re_x.device)
        if itg.numel() == 0:
            itg = torch.arange(tg_x.size(0), device=tg_x.device)

        summary_rows.append(
            {
                "time_idx": time_idx,
                "n_TF_active": int(itf.numel()),
                "n_RE_active": int(ire.numel()),
                "n_TG_active": int(itg.numel()),
            }
        )

        tf_embed = F.normalize(out_tf[:, time_idx, :][itf], dim=1)
        re_embed = F.normalize(out_re[:, time_idx, :][ire], dim=1)
        tg_embed = F.normalize(out_tg[:, time_idx, :][itg], dim=1)

        score_matrix = _cosine_matrix(tf_embed, tg_embed)
        flat_scores = score_matrix.flatten()
        eligible = torch.where(flat_scores >= settings["min_score_keep"])[0]
        edge_map: dict[tuple[int, int], float] = {}
        rescued_tf = 0

        if eligible.numel() > 0:
            global_keep = min(settings["top_k_global"], int(eligible.numel()))
            eligible_scores = flat_scores[eligible]
            _, rel0 = torch.topk(eligible_scores, global_keep)
            top_idx = eligible[rel0]
            n_tg_active = int(tg_embed.size(0))
            for idx in top_idx.tolist():
                tf_local = idx // n_tg_active
                tg_local = idx % n_tg_active
                tf_global = int(itf[tf_local].item())
                tg_global = int(itg[tg_local].item())
                score_value = float(flat_scores[idx].item())
                prev = edge_map.get((tf_global, tg_global))
                if prev is None or score_value > prev:
                    edge_map[(tf_global, tg_global)] = score_value

        current_series = tf_frac_df[time_idx] if time_idx in tf_frac_df.columns else pd.Series(dtype=float)
        other_series = other_max_cache[time_idx]
        for tf_local, tf_global in enumerate(itf.tolist()):
            tf_id = refs["tf_ids"][tf_global]
            frac_current = float(current_series.get(tf_id, 0.0))
            frac_other = float(other_series.get(tf_id, 0.0))
            specificity = frac_current / (frac_current + frac_other + settings["eps"])
            keep_count = int(
                round(
                    settings["m_min"]
                    + (settings["m_max"] - settings["m_min"]) * (specificity ** settings["gamma"])
                )
            )
            if keep_count <= 0:
                continue
            row = score_matrix[tf_local, :]
            valid = torch.where(row >= settings["min_score_keep"])[0]
            if valid.numel() == 0:
                continue
            keep_count = min(keep_count, int(valid.numel()))
            _, rel = torch.topk(row[valid], keep_count)
            for tg_local in valid[rel].tolist():
                tg_global = int(itg[tg_local].item())
                score_value = float(row[tg_local].item())
                prev = edge_map.get((tf_global, tg_global))
                if prev is None or score_value > prev:
                    edge_map[(tf_global, tg_global)] = score_value
            rescued_tf += 1

        top_edges = [
            (tf_idx, tg_idx, score)
            for (tf_idx, tg_idx), score in edge_map.items()
            if score >= settings["min_score_keep"]
        ]
        top_edges.sort(key=lambda item: -item[2])
        tftg_frame = _tftg_rows_to_frame(
            top_edges,
            time_idx=time_idx,
            tf_ids=refs["tf_ids"],
            tf_names=refs["tf_names"],
            tg_ids=refs["tg_ids"],
            tg_names=refs["tg_names"],
            score_mode=settings["strict_score_mode"],
        )
        tftg_path = output_dir / tftg_filename_template.format(t=time_idx)
        tftg_frame.to_csv(tftg_path, sep="\t", index=False)
        saved_tftg.append(str(tftg_path))
        summary_rows[-1]["eligible_tftg_edges"] = int(eligible.numel())
        summary_rows[-1]["rescued_tf_count"] = int(rescued_tf)
        summary_rows[-1]["kept_tftg_edges"] = int(len(top_edges))

        ire_np = ire.detach().cpu().numpy()
        act_re_chr = re_coord["re_chr"][ire_np]
        act_re_center = re_coord["re_center"][ire_np]
        act_re_finite = np.isfinite(act_re_center)
        re_frac_all = (re_valid != 0).float().mean(dim=1).detach().cpu().numpy()
        act_re_frac = re_frac_all[ire_np]
        tf_all_norm = F.normalize(out_tf[:, time_idx, :], dim=1)
        tg_all_norm = F.normalize(out_tg[:, time_idx, :], dim=1)
        tgre_rows: list[dict[str, Any]] = []

        for tf_global, tg_global, _score in top_edges:
            tf_id = refs["tf_ids"][tf_global]
            tf_name = refs["tf_names"][tf_global]
            tg_id = refs["tg_ids"][tg_global]
            tg_name = refs["tg_names"][tg_global]
            tf_vec = tf_all_norm[tf_global]
            tg_vec = tg_all_norm[tg_global]
            meta = _resolve_tg_meta(tg_id, tg_meta_bundle["tg_meta"])
            tg_chr = meta.get("chr") if meta is not None else None
            tg_tss = meta.get("tss") if meta is not None else None

            candidate_rel = _candidate_active_re_indices(
                tg_chr=tg_chr,
                tg_tss=tg_tss,
                act_re_chr=act_re_chr,
                act_re_center=act_re_center,
                act_re_finite=act_re_finite,
                max_dist_bp=settings["max_dist_bp"],
            )
            if candidate_rel.size == 0:
                continue

            top_rel, _top_score = _select_re_indices_composite(
                tf_vec=tf_vec,
                tg_vec=tg_vec,
                re_embed=re_embed,
                candidate_rel=candidate_rel,
                act_re_center=act_re_center,
                act_re_frac=act_re_frac,
                active_frac_re=settings["active_frac_re"],
                tg_tss=tg_tss,
                topn_re_per_tg=settings["topn_re_per_tg"],
                dist_lambda=settings["dist_lambda"],
                w_re_tg=settings["w_re_tg"],
                w_dist=settings["w_dist"],
                w_tf_re=settings["w_tf_re"],
                w_act=settings["w_act"],
                eps=settings["eps"],
            )
            for re_rel in top_rel.tolist():
                re_global = int(ire[re_rel].item())
                distance = np.nan
                if tg_tss is not None and np.isfinite(act_re_center[re_rel]):
                    distance = float(act_re_center[re_rel] - float(tg_tss))
                tgre_rows.append(
                    {
                        "time_idx": time_idx,
                        "TF_idx": tf_global,
                        "TG_idx": tg_global,
                        "RE_idx": re_global,
                        "TF_id": tf_id,
                        "TF_name": tf_name,
                        "TG_id": tg_id,
                        "TG_name": tg_name,
                        "RE_name": refs["re_names"][re_global],
                        "TG_chr": tg_chr,
                        "TG_tss": tg_tss,
                        "dist_REcenter_to_TSS_bp": distance,
                    }
                )

        tfretg_path = output_dir / tfretg_filename_template.format(t=time_idx)
        pd.DataFrame(tgre_rows).to_csv(tfretg_path, sep="\t", index=False)
        saved_tfretg.append(str(tfretg_path))

    active_path = output_dir / "active_counts.tsv"
    pd.DataFrame(summary_rows).to_csv(active_path, sep="\t", index=False)
    report = {
        "lineage": lineage,
        "output_dir": str(output_dir),
        "active_counts_path": str(active_path),
        "tftg_paths": saved_tftg,
        "tfretg_paths": saved_tfretg,
        "allowed_tf_count": len(allowed_tf_indices),
        "settings": settings,
    }
    if settings["save_report"]:
        save_checkpoint_metadata(output_dir / "network_report.json", report)
    return report


def _make_dense_labels(time_count: int, n_interp: int) -> tuple[torch.Tensor, list[str], list[int], list[bool]]:
    tgt_times: list[float] = []
    labels: list[str] = []
    left_indices: list[int] = []
    is_integer: list[bool] = []
    for time_idx in range(time_count - 1):
        tgt_times.append(float(time_idx))
        labels.append(f"{time_idx + 1:.2f}".rstrip("0").rstrip("."))
        left_indices.append(time_idx)
        is_integer.append(True)
        for step in range(1, n_interp + 1):
            frac = step / (n_interp + 1)
            tgt_times.append(float(time_idx + frac))
            labels.append(f"{time_idx + 1 + frac:.2f}".rstrip("0").rstrip("."))
            left_indices.append(time_idx)
            is_integer.append(False)
    tgt_times.append(float(time_count - 1))
    labels.append(f"{time_count:.2f}".rstrip("0").rstrip("."))
    left_indices.append(time_count - 1)
    is_integer.append(True)
    return torch.tensor(tgt_times, dtype=torch.float32), labels, left_indices, is_integer


def _l2_normalize_rows(values: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return values / values.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)


def resample_embeddings_gaussian(
    embeddings: torch.Tensor,
    src_times: torch.Tensor,
    tgt_times: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    device = embeddings.device
    dtype = embeddings.dtype
    src_times = src_times.to(device=device, dtype=dtype)
    tgt_times = tgt_times.to(device=device, dtype=dtype)
    dt = tgt_times.view(tgt_times.numel(), 1) - src_times.view(1, src_times.numel())
    kernel = torch.exp(-0.5 * (dt / sigma) ** 2)
    kernel = kernel / kernel.sum(dim=1, keepdim=True).clamp_min(1e-12)
    flat = embeddings.permute(1, 0, 2).reshape(src_times.numel(), -1)
    resampled = kernel @ flat
    return resampled.view(tgt_times.numel(), embeddings.size(0), embeddings.size(2)).permute(1, 0, 2).contiguous()


def _dense_name(value: str, node_type: str, add_prefix: bool) -> str:
    return f"{node_type}:{value}" if add_prefix else str(value)


def _dense_tf_whitelist(
    tf_ids: list[str],
    tf_names: list[str],
    tf_filter: dict[str, Any],
) -> torch.Tensor:
    allowed_fbgn = {str(value) for value in tf_filter.get("allowed_fbgn", set())}
    allowed_symbol = {strip_copy_suffix(str(value)) for value in tf_filter.get("allowed_symbol", set())}
    return torch.tensor(
        [
            (str(tf_id) in allowed_fbgn) or (strip_copy_suffix(str(tf_name)) in allowed_symbol)
            for tf_id, tf_name in zip(tf_ids, tf_names)
        ],
        dtype=torch.bool,
    )


def _dense_active_masks(
    graphs_sel: list[Any],
    active_thr: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    time_count = len(graphs_sel)
    num_tf = graphs_sel[0]["TF"].x.size(0)
    num_tg = graphs_sel[0]["TG"].x.size(0)
    tf_active = torch.zeros(time_count, num_tf, dtype=torch.bool)
    tg_active = torch.zeros(time_count, num_tg, dtype=torch.bool)
    tf_frac = torch.zeros(time_count, num_tf, dtype=torch.float32)
    for time_idx, graph in enumerate(graphs_sel):
        tf_x, re_x, tg_x = graph["TF"].x, graph["RE"].x, graph["TG"].x
        valid = (tf_x != 0).any(dim=0) | (re_x != 0).any(dim=0) | (tg_x != 0).any(dim=0)
        if valid.ndim == 1 and bool(valid.any()):
            tf_valid = tf_x[:, valid]
            tg_valid = tg_x[:, valid]
        else:
            tf_valid = tf_x
            tg_valid = tg_x
        tf_frac_now = (tf_valid != 0).float().mean(dim=1).detach().cpu()
        tg_frac_now = (tg_valid != 0).float().mean(dim=1).detach().cpu()
        tf_frac[time_idx] = tf_frac_now
        tf_idx = torch.where(tf_frac_now >= active_thr)[0]
        tg_idx = torch.where(tg_frac_now >= active_thr)[0]
        if tf_idx.numel() == 0:
            tf_idx = torch.arange(tf_x.size(0), dtype=torch.long)
        if tg_idx.numel() == 0:
            tg_idx = torch.arange(tg_x.size(0), dtype=torch.long)
        tf_active[time_idx, tf_idx] = True
        tg_active[time_idx, tg_idx] = True
    return tf_active, tg_active, tf_frac


def _dense_tf_specificity(tf_frac_int: torch.Tensor, eps: float) -> torch.Tensor:
    time_count = int(tf_frac_int.size(0))
    output = torch.zeros_like(tf_frac_int)
    for time_idx in range(time_count):
        current = tf_frac_int[time_idx]
        if time_count == 1:
            other_max = torch.zeros_like(current)
        else:
            others = [idx for idx in range(time_count) if idx != time_idx]
            other_max = tf_frac_int[others].max(dim=0).values
        output[time_idx] = current / (current + other_max + eps)
    return output


def _export_dense_networks(
    lineage: str,
    output_dir: Path,
    graphs_sel: list[Any],
    out_tf: torch.Tensor,
    out_tg: torch.Tensor,
    tf_ids: list[str],
    tf_names: list[str],
    tg_names: list[str],
    tf_filter: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    tf_whitelist = _dense_tf_whitelist(tf_ids, tf_names, tf_filter)
    tf_active_int, tg_active_int, tf_frac_int = _dense_active_masks(graphs_sel, settings["active_thr"])
    tf_spec_int = _dense_tf_specificity(tf_frac_int, settings["rescue_eps"])

    tf_embed_int = out_tf.detach().cpu().float()
    tg_embed_int = out_tg.detach().cpu().float()
    time_count = int(tf_embed_int.size(1))
    src_times = torch.arange(time_count, dtype=torch.float32)
    tgt_times, dense_labels, left_indices, is_integer = _make_dense_labels(time_count, settings["n_interp"])
    tf_embed_dense = resample_embeddings_gaussian(tf_embed_int, src_times, tgt_times, settings["sigma_time"])
    tg_embed_dense = resample_embeddings_gaussian(tg_embed_int, src_times, tgt_times, settings["sigma_time"])

    for dense_idx, flag in enumerate(is_integer):
        if flag:
            time_idx = left_indices[dense_idx]
            tf_embed_dense[:, dense_idx, :] = tf_embed_int[:, time_idx, :]
            tg_embed_dense[:, dense_idx, :] = tg_embed_int[:, time_idx, :]
    tf_embed_dense = _l2_normalize_rows(tf_embed_dense)
    tg_embed_dense = _l2_normalize_rows(tg_embed_dense)

    saved_paths: list[str] = []
    summary_rows: list[dict[str, Any]] = []

    for dense_idx, label in enumerate(dense_labels):
        left_idx = left_indices[dense_idx]
        tf_mask = tf_active_int[left_idx].clone() & tf_whitelist
        tg_mask = tg_active_int[left_idx].clone()
        tf_idx_active = np.where(tf_mask.numpy())[0]
        tg_idx_active = np.where(tg_mask.numpy())[0]
        summary_rows.append(
            {
                "dense_idx": dense_idx,
                "label": label,
                "left_int_idx": left_idx,
                "is_integer": bool(is_integer[dense_idx]),
                "n_TF_active": int(len(tf_idx_active)),
                "n_TG_active": int(len(tg_idx_active)),
            }
        )
        if len(tf_idx_active) == 0 or len(tg_idx_active) == 0:
            continue

        tf_vec = tf_embed_dense[:, dense_idx, :][tf_mask]
        tg_vec = tg_embed_dense[:, dense_idx, :][tg_mask]
        score_matrix = (tf_vec @ tg_vec.T).detach().cpu().numpy()
        keep_mask = score_matrix >= settings["min_score_keep"]
        tf_idx_list = tf_idx_active.tolist()
        tg_idx_list = tg_idx_active.tolist()
        tf_local_map = {int(tf_global): tf_local for tf_local, tf_global in enumerate(tf_idx_list)}

        base_rows: list[tuple[int, int, float]] = []
        for tf_local, tf_global in enumerate(tf_idx_list):
            valid_tg_local = np.flatnonzero(keep_mask[tf_local])
            for tg_local in valid_tg_local.tolist():
                base_rows.append((int(tf_global), int(tg_idx_list[tg_local]), float(score_matrix[tf_local, tg_local])))
        base_rows.sort(key=lambda item: -item[2])

        global_keep = max(0, int(settings["top_k_global"]))
        selected = base_rows[:global_keep] if global_keep > 0 else []
        selected_set = {(tf_idx, tg_idx) for tf_idx, tg_idx, _score in selected}
        selected_counts: dict[int, int] = {}
        for tf_idx, _tg_idx, _score in selected:
            selected_counts[tf_idx] = selected_counts.get(tf_idx, 0) + 1

        rescued_edges = 0
        spec_t = tf_spec_int[left_idx].numpy()
        for tf_global in tf_idx_list:
            specificity = float(spec_t[tf_global])
            keep_count = int(
                round(
                    settings["rescue_m_min"]
                    + (settings["rescue_m_max"] - settings["rescue_m_min"])
                    * (specificity ** settings["rescue_gamma"])
                )
            )
            if keep_count <= 0:
                continue
            current_count = selected_counts.get(int(tf_global), 0)
            if current_count >= keep_count:
                continue

            tf_local = tf_local_map[int(tf_global)]
            row_scores = score_matrix[tf_local]
            valid_tg_local = np.flatnonzero(row_scores >= settings["min_score_keep"])
            if valid_tg_local.size == 0:
                continue
            ordered_tg_local = valid_tg_local[np.argsort(-row_scores[valid_tg_local], kind="mergesort")]
            for tg_local in ordered_tg_local.tolist():
                if current_count >= keep_count:
                    break
                tg_global = int(tg_idx_list[tg_local])
                pair = (int(tf_global), tg_global)
                if pair in selected_set:
                    continue
                selected.append((int(tf_global), tg_global, float(row_scores[tg_local])))
                selected_set.add(pair)
                current_count += 1
                rescued_edges += 1
            selected_counts[int(tf_global)] = current_count

        selected.sort(key=lambda item: -item[2])
        if not selected:
            continue

        summary_rows[-1]["n_base_edges"] = int(len(base_rows))
        summary_rows[-1]["n_topk_edges"] = int(min(len(base_rows), global_keep))
        summary_rows[-1]["n_rescued_edges"] = int(rescued_edges)
        summary_rows[-1]["n_selected_edges"] = int(len(selected))

        tf_idx_global = np.array([item[0] for item in selected], dtype=int)
        tg_idx_global = np.array([item[1] for item in selected], dtype=int)
        score_cos = np.array([item[2] for item in selected], dtype=float)
        payload: dict[str, Any] = {
            "time_label": [label] * len(selected),
            "left_int_idx": np.full(len(selected), left_idx, dtype=int),
            "is_integer": np.full(len(selected), bool(is_integer[dense_idx]), dtype=bool),
            "TF": [_dense_name(tf_names[idx], "TF", settings["add_prefix"]) for idx in tf_idx_global.tolist()],
            "TG": [_dense_name(tg_names[idx], "TG", settings["add_prefix"]) for idx in tg_idx_global.tolist()],
            "score_cos": score_cos,
            "prob": (score_cos + 1.0) / 2.0,
        }
        if settings["keep_id_cols"]:
            if settings["use_one_based"]:
                payload["TF_idx"] = tf_idx_global + 1
                payload["TG_idx"] = tg_idx_global + 1
            else:
                payload["TF_idx"] = tf_idx_global
                payload["TG_idx"] = tg_idx_global
        frame = pd.DataFrame(payload)
        output_path = output_dir / _dense_tftg_filename(label)
        frame.to_csv(output_path, sep="\t", index=False)
        saved_paths.append(str(output_path))

    summary_path = output_dir / "dense_active_counts.tsv"
    pd.DataFrame(summary_rows).to_csv(summary_path, sep="\t", index=False)
    report = {
        "lineage": lineage,
        "output_dir": str(output_dir),
        "summary_path": str(summary_path),
        "dense_paths": saved_paths,
        "settings": settings,
    }
    save_checkpoint_metadata(output_dir / "dense_network_report.json", report)
    return report


def _network_context(config: dict[str, Any]) -> dict[str, Any]:
    lineage = str(config.get("lineage", "")).lower()
    paths = InputPaths.from_config(config)
    base_dir = paths.base_dir
    strict_settings = _network_settings(config)
    dense_settings = _dense_network_settings(config)
    strict_settings["output_dir"] = str(resolve_config_path(base_dir, strict_settings["output_dir"]))
    dense_settings["output_dir"] = str(resolve_config_path(base_dir, dense_settings["output_dir"]))
    tf_filter_path = _resolve_relative_path(base_dir, strict_settings["tf_filter_file"])
    tf_filter = _load_tf_filter(tf_filter_path)
    tg_meta_bundle = _load_tg_meta(paths.tg_list)
    common = prepare_common_inputs(config)

    if lineage in {"soma", "somatic"}:
        inference = reload_soma_from_checkpoint(config)
        graphs_sel = inference["graphs_somatic"]
    elif lineage in {"germ", "germline"}:
        inference = reload_germ_from_checkpoint(config)
        graphs_sel = inference["graphs_germline"]
    else:
        raise ValueError(f"Unsupported lineage '{config.get('lineage')}'. Expected soma or germ.")

    refs = _pick_reference_names(
        maps=inference.get("maps"),
        paths=paths,
        num_tf=int(inference["out_tf"].size(0)),
        num_tg=int(inference["out_tg"].size(0)),
        num_re=int(inference["out_re"].size(0)),
        tf_filter=tf_filter,
        tg_meta_bundle=tg_meta_bundle,
    )

    return {
        "lineage": lineage,
        "paths": paths,
        "strict_settings": strict_settings,
        "dense_settings": dense_settings,
        "tf_filter_path": tf_filter_path,
        "tf_filter": tf_filter,
        "tg_meta_bundle": tg_meta_bundle,
        "common": common,
        "inference": inference,
        "graphs_sel": graphs_sel,
        "refs": refs,
    }


def export_lineage_networks_from_config(config: dict[str, Any]) -> dict[str, Any]:
    context = _network_context(config)
    strict_dir = ensure_directory(context["strict_settings"]["output_dir"])
    strict_report = _export_strict_networks(
        lineage=context["lineage"],
        output_dir=strict_dir,
        graphs_sel=context["graphs_sel"],
        out_tf=context["inference"]["out_tf"],
        out_re=context["inference"]["out_re"],
        out_tg=context["inference"]["out_tg"],
        refs=context["refs"],
        tf_filter=context["tf_filter"],
        tg_meta_bundle=context["tg_meta_bundle"],
        settings=context["strict_settings"],
    )

    dense_report: dict[str, Any] | None = None
    if context["dense_settings"]["enabled"]:
        dense_dir = ensure_directory(context["dense_settings"]["output_dir"])
        dense_report = _export_dense_networks(
            lineage=context["lineage"],
            output_dir=dense_dir,
            graphs_sel=context["graphs_sel"],
            out_tf=context["inference"]["out_tf"],
            out_tg=context["inference"]["out_tg"],
            tf_ids=context["refs"]["tf_ids"],
            tf_names=context["refs"]["tf_names"],
            tg_names=context["refs"]["tg_names"],
            tf_filter=context["tf_filter"],
            settings=context["dense_settings"],
        )

    return {
        "lineage": context["lineage"],
        "strict": strict_report,
        "dense": dense_report,
        "checkpoint_path": context["inference"]["checkpoint_path"],
    }


def export_lineage_networks(config_path: str | Path) -> dict[str, Any]:
    config = load_yaml_config(config_path)
    report = export_lineage_networks_from_config(config)
    report["config_path"] = str(Path(config_path).resolve())
    return report


def discover_network_time_indices(network_dir: str | Path) -> list[int]:
    directory = Path(network_dir)
    patterns = [
        "t*_TF_TG.tsv",
        "t*_TF-TG.tsv",
        "t*_TF_TG_globalTopK_rescue.tsv",
        "t*_TF_TG_top10000.tsv",
    ]
    time_indices: set[int] = set()
    for pattern in patterns:
        for path in directory.glob(pattern):
            match = re.search(
                r"t(\d+)(?:\.\d+)?_(?:TF-TG|TF_TG|TF_TG_(?:globalTopK_rescue|top10000))\.tsv$",
                path.name,
            )
            if match:
                time_indices.add(int(match.group(1)))
    if not time_indices:
        raise FileNotFoundError(f"No exported TF-TG network files were found under {directory}.")
    return sorted(time_indices)


def load_exported_tftg_df(network_dir: str | Path, time_idx: int) -> pd.DataFrame:
    directory = Path(network_dir)
    candidates = [
        directory / f"t{time_idx:02d}_TF_TG.tsv",
        directory / f"t{time_idx:02d}_TF_TG_globalTopK_rescue.tsv",
        directory / f"t{time_idx:02d}_TF_TG_top10000.tsv",
        directory / f"t{time_idx:02d}_TF-TG.tsv",
    ]
    path = next((item for item in candidates if item.exists()), None)
    if path is None:
        raise FileNotFoundError(f"No exported TF-TG file was found for time index {time_idx} under {directory}.")
    frame = pd.read_csv(path, sep="\t")
    frame.columns = [str(column).strip() for column in frame.columns]
    if "TF" in frame.columns and "TF_name" not in frame.columns:
        frame["TF_name"] = frame["TF"].astype(str)
    if "TG" in frame.columns and "TG_name" not in frame.columns:
        frame["TG_name"] = frame["TG"].astype(str)
    if "TF_idx" not in frame.columns and "TF_name" in frame.columns:
        frame["TF_idx"] = frame["TF_name"].astype(str)
    if "TG_idx" not in frame.columns and "TG_name" in frame.columns:
        frame["TG_idx"] = frame["TG_name"].astype(str)
    if "prob" not in frame.columns:
        if "score_cos" in frame.columns:
            frame["prob"] = (frame["score_cos"].astype(float) + 1.0) / 2.0
        elif "score_dot" in frame.columns:
            frame["prob"] = 1.0 / (1.0 + np.exp(-frame["score_dot"].astype(float).to_numpy()))
        else:
            raise ValueError(f"{path} does not contain prob/score_cos/score_dot columns required for downstream selection.")
    return frame


def load_exported_tgre_df(network_dir: str | Path, time_idx: int) -> pd.DataFrame:
    directory = Path(network_dir)
    candidates = [
        directory / f"t{time_idx:02d}_TF_RE_TG.tsv",
        directory / f"t{time_idx:02d}_TG_RE_nearest10_top2_per_pair.tsv",
        directory / f"t{time_idx:02d}_TF-RE-TG.tsv",
    ]
    path = next((item for item in candidates if item.exists()), None)
    if path is None:
        raise FileNotFoundError(candidates[0])
    return pd.read_csv(path, sep="\t")


def select_edges_from_network_df(
    df_tftg: pd.DataFrame,
    mode: str = "topk",
    top_n: int | None = None,
    thr: float | None = None,
) -> pd.DataFrame:
    normalized = str(mode).strip().lower()
    if normalized in {"all", "tf_rescue"}:
        return df_tftg.reset_index(drop=True)
    if normalized not in {"topk", "threshold", "both"}:
        raise ValueError(f"Unsupported strict-network selection mode '{mode}'.")

    if df_tftg.empty:
        return df_tftg.reset_index(drop=True)

    scores = df_tftg["prob"].astype(float).to_numpy()
    idx_topk: pd.Index | None = None
    idx_thr: pd.Index | None = None

    if normalized in {"topk", "both"}:
        keep = len(df_tftg) if top_n is None else min(int(top_n), len(df_tftg))
        if keep > 0:
            if keep == len(df_tftg):
                order = np.argsort(-scores, kind="mergesort")
                idx_topk = pd.Index(order)
            else:
                part = np.argpartition(-scores, kth=keep - 1)[:keep]
                part = part[np.argsort(-scores[part], kind="mergesort")]
                idx_topk = pd.Index(part)
        else:
            idx_topk = pd.Index([], dtype=int)

    if normalized in {"threshold", "both"}:
        cutoff = 0.0 if thr is None else float(thr)
        idx_thr = pd.Index(np.flatnonzero(scores >= cutoff))

    if normalized == "topk":
        idx = idx_topk
    elif normalized == "threshold":
        idx = idx_thr
    else:
        idx = idx_topk.intersection(idx_thr) if idx_topk is not None and idx_thr is not None else pd.Index([], dtype=int)

    if idx is None:
        return df_tftg.iloc[[]].copy()
    return df_tftg.iloc[idx].reset_index(drop=True)


def load_and_select_all(
    network_dir: str | Path,
    tids: list[int],
    mode: str = "topk",
    top_n: int | None = None,
    thr: float | None = None,
) -> dict[int, pd.DataFrame]:
    output: dict[int, pd.DataFrame] = {}
    for time_idx in tids:
        frame = load_exported_tftg_df(network_dir, time_idx)
        output[time_idx] = select_edges_from_network_df(frame, mode=mode, top_n=top_n, thr=thr)
    return output


def build_edge_dataframe(
    network_dir: str | Path,
    tids: list[int] | None = None,
    mode: str = "topk",
    top_n: int | None = None,
    thr: float | None = None,
) -> pd.DataFrame:
    clusters = tids if tids is not None else discover_network_time_indices(network_dir)
    selected = load_and_select_all(network_dir, clusters, mode=mode, top_n=top_n, thr=thr)
    frames: list[pd.DataFrame] = []
    for time_idx in clusters:
        frame = selected[time_idx]
        if frame.empty:
            continue
        frame = frame.copy()
        frame["cell_cluster"] = int(time_idx)
        frame["tf_idx"] = frame["TF_idx"]
        frame["tg_idx"] = frame["TG_idx"]
        frame["tf_name"] = frame["TF_name"].astype(str)
        frame["tg_name"] = frame["TG_name"].astype(str)
        frame["score"] = frame["prob"].astype(float)
        frames.append(frame[["cell_cluster", "tf_idx", "tg_idx", "tf_name", "tg_name", "score"]])
    return (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["cell_cluster", "tf_idx", "tg_idx", "tf_name", "tg_name", "score"])
    )
