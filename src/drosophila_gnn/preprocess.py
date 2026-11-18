"""Input preprocessing: RNA/ATAC matrices, TF/TG/RE lists and priors."""

from __future__ import annotations

import bisect
import re
import subprocess
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from .io import InputPaths, strip_copy_suffix


def pick(colnames: list[str], cand: set[str]) -> str | None:
    for col in colnames:
        if col.lower() in cand:
            return col
    return None


def adata_to_df(adata: ad.AnnData, prefer_layer: str | None = None) -> tuple[pd.DataFrame, str]:
    data_matrix = adata.layers[prefer_layer] if (prefer_layer and prefer_layer in adata.layers) else adata.X
    if sp.issparse(data_matrix):
        data_matrix = data_matrix.tocsr()
        matrix = data_matrix.T.toarray()
    else:
        matrix = np.asarray(data_matrix.T)
    dataframe = pd.DataFrame(
        matrix,
        index=adata.var_names.astype(str),
        columns=adata.obs_names.astype(str),
    )
    return dataframe, (prefer_layer or "X")


def fix_dash(value: str) -> str:
    return str(value).replace("–", "-").replace("—", "-").replace("−", "-")


def parse_re(value: str) -> tuple[str, int, int] | None:
    text = fix_dash(str(value).strip())
    match = re.match(r"^([^:]+):\s*([0-9]+)\s*-\s*([0-9]+)$", text)
    if not match:
        return None
    chrom, start, end = match.group(1), int(match.group(2)), int(match.group(3))
    if start > end:
        start, end = end, start
    return chrom, start, end


def maybe_add_chr(chrom: str) -> str:
    return chrom if str(chrom).startswith("chr") else f"chr{chrom}"


def maybe_drop_chr(chrom: str) -> str:
    return str(chrom)[3:] if str(chrom).startswith("chr") else str(chrom)


def pick_device_by_free_mem(min_required_mb: int = 0) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        per_free = [int(torch.cuda.mem_get_info(i)[0] // (1024**2)) for i in range(torch.cuda.device_count())]
    except Exception:
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                text=True,
            )
            per_free = [int(line.strip()) for line in output.strip().splitlines()]
        except Exception:
            return torch.device("cpu")
    candidates = [(idx, mb) for idx, mb in enumerate(per_free) if mb >= min_required_mb]
    best_idx = max(candidates or list(enumerate(per_free)), key=lambda item: item[1])[0]
    return torch.device(f"cuda:{best_idx}")


def norm_id(value: str) -> str:
    text = str(value).strip()
    return re.sub(r"#s?0*([0-9]+)$", r"_sample\1", text, flags=re.IGNORECASE)


def build_re_overlap_index(re_list: list[str]) -> dict[str, dict[str, list[Any]]]:
    by_chr: dict[str, dict[str, list[Any]]] = {}
    for name in re_list:
        parsed = parse_re(name)
        if not parsed:
            continue
        chrom, start, end = parsed
        entry = by_chr.setdefault(chrom, {"starts": [], "ends": [], "names": []})
        entry["starts"].append(start)
        entry["ends"].append(end)
        entry["names"].append(name)
    for _, entry in by_chr.items():
        order = sorted(range(len(entry["starts"])), key=entry["starts"].__getitem__)
        for key in ("starts", "ends", "names"):
            entry[key] = [entry[key][idx] for idx in order]
    return by_chr


def map_re_to_known(
    re_value: str,
    re2idx: dict[str, int],
    by_chr: dict[str, dict[str, list[Any]]],
) -> str | None:
    text = fix_dash(str(re_value).strip())
    if text in re2idx:
        return text
    parsed = parse_re(text)
    if not parsed:
        return None
    chrom, start, end = parsed
    for alt in (chrom, maybe_add_chr(chrom), maybe_drop_chr(chrom)):
        candidate = f"{alt}:{start}-{end}"
        if candidate in re2idx:
            return candidate
    midpoint = (start + end) // 2
    for alt in (chrom, maybe_add_chr(chrom), maybe_drop_chr(chrom)):
        if alt not in by_chr:
            continue
        entry = by_chr[alt]
        position = bisect.bisect_right(entry["starts"], midpoint) - 1
        for idx in range(max(0, position - 3), min(len(entry["starts"]), position + 4)):
            if entry["starts"][idx] <= midpoint <= entry["ends"][idx]:
                return entry["names"][idx]
    return None


def prepare_common_inputs(config: dict[str, Any]) -> dict[str, Any]:
    paths = InputPaths.from_config(config)

    tf_df = pd.read_csv(paths.tf_list, sep="\t", dtype=str)
    tf_cols = [column.strip() for column in tf_df.columns]
    col_tf_sym = pick(tf_cols, {"gene_name", "symbol", "tf"})
    col_tf_fb = next((column for column in tf_cols if column.lower().startswith("fbgn")), None)
    assert col_tf_sym and col_tf_fb, "TF.tsv requires gene_name/symbol and FBgn_* columns"
    tf_df[col_tf_sym] = tf_df[col_tf_sym].astype(str).map(strip_copy_suffix)
    sym2fb = dict(zip(tf_df[col_tf_sym].astype(str), tf_df[col_tf_fb].astype(str)))

    tg_df = pd.read_csv(paths.tg_list, sep="\t", dtype=str)
    tg_cols = [column.strip() for column in tg_df.columns]
    col_tg_fb = next((column for column in tg_cols if column.lower().startswith("fbgn")), None)
    col_tg_sym = pick(tg_cols, {"gene_name", "symbol", "gene"})
    if col_tg_fb is not None:
        tg_list_all = pd.unique(tg_df[col_tg_fb].dropna().astype(str)).tolist()
    else:
        tg_tmp = tg_df[col_tg_sym].map(sym2fb)
        tg_list_all = pd.unique(tg_tmp.dropna().astype(str)).tolist()

    with paths.re_list.open("r", encoding="utf-8") as handle:
        re_list = pd.unique(pd.Series([line.strip() for line in handle if line.strip()])).tolist()

    adata_rna = ad.read_h5ad(paths.rna_h5ad)
    adata_atac = ad.read_h5ad(paths.atac_h5ad)
    expr_df, used_rna_layer = adata_to_df(adata_rna, prefer_layer="lognorm")
    peaks_df, used_atac_layer = adata_to_df(adata_atac, prefer_layer="counts")

    tf_list = [gene for gene in pd.unique(tf_df[col_tf_fb].dropna().astype(str)) if gene in expr_df.index]
    tg_list = [gene for gene in tg_list_all if gene in expr_df.index]
    num_tf, num_tg, num_re = len(tf_list), len(tg_list), len(re_list)
    tf2idx = {gene: idx for idx, gene in enumerate(tf_list)}
    tg2idx = {gene: idx for idx, gene in enumerate(tg_list)}
    re2idx = {region: idx for idx, region in enumerate(re_list)}

    anno_raw = pd.read_csv(paths.annotation, dtype=str)
    columns = anno_raw.columns.tolist()
    cell_candidates = ["cell_id", "cell", "barcode", "barcodes", "Unnamed: 0", "index", "name", "id"]
    time_candidates = ["GEX_latent_time_bins", "time", "latent_time_bins"]
    ctype_candidates = [
        "GEX_latent_time_bins_annotated",
        "nt_time_bins_annotated",
        "cell_type",
        "annotation",
        "cluster",
        "label",
        "labels",
    ]

    cell_col = next((column for column in cell_candidates if column in columns), columns[0])
    time_col = next((column for column in time_candidates if column in columns), None)
    ctype_col = next((column for column in ctype_candidates if column in columns), None)
    assert ctype_col is not None, f"Cannot find cell type column. Available columns: {columns}"

    anno = anno_raw.rename(columns={cell_col: "cell_raw"})
    if time_col:
        anno = anno.rename(columns={time_col: "time"})
    if ctype_col:
        anno = anno.rename(columns={ctype_col: "cell_type"})
    keep_cols = ["cell_raw"] + (["time"] if "time" in anno.columns else []) + (["cell_type"] if "cell_type" in anno.columns else [])
    anno = anno[keep_cols]

    expr_map = {norm_id(cell): cell for cell in expr_df.columns}
    peaks_map = {norm_id(cell): cell for cell in peaks_df.columns}
    anno["cell_id_norm"] = anno["cell_raw"].map(norm_id)

    common_norm = set(anno["cell_id_norm"]) & set(expr_map) & set(peaks_map)
    anno = anno[anno["cell_id_norm"].isin(common_norm)].copy()
    anno["cell_id"] = anno["cell_id_norm"].map(expr_map)
    out_cols = ["cell_id"] + (["time"] if "time" in anno.columns else []) + (["cell_type"] if "cell_type" in anno.columns else [])
    anno = anno[out_cols]

    by_chr = build_re_overlap_index(re_list)

    tf_re_edges = pd.read_csv(paths.tf_re_prior, sep="\t", dtype=str)
    tf_re_edges.columns = [column.strip().lower() for column in tf_re_edges.columns]
    col_re = pick(list(tf_re_edges.columns), {"re", "peak", "region"})
    col_tf = pick(list(tf_re_edges.columns), {"tf", "tg", "gene", "gene_name"})
    assert col_re and col_tf, f"{paths.tf_re_prior} requires RE and TF columns"
    tf_re_edges = tf_re_edges[[col_re, col_tf]].rename(columns={col_re: "RE", col_tf: "TF_sym"})
    tf_re_edges["TF_sym"] = tf_re_edges["TF_sym"].astype(str).map(strip_copy_suffix)
    tf_re_edges["TF"] = tf_re_edges["TF_sym"].map(sym2fb)
    tf_re_edges["RE_map"] = tf_re_edges["RE"].map(lambda value: map_re_to_known(value, re2idx, by_chr))
    tf_re_edges = tf_re_edges.dropna(subset=["TF", "RE_map"])
    tf_re_edges = tf_re_edges[tf_re_edges["TF"].isin(tf2idx)]
    tf_re_edges = tf_re_edges.drop_duplicates(subset=["TF", "RE_map"])

    pairs_tf_re = {(tf2idx[tf], re2idx[re_name]) for tf, re_name in zip(tf_re_edges["TF"], tf_re_edges["RE_map"])}
    edge_index_tf_re = (
        torch.tensor([list(zip(*pairs_tf_re))[0], list(zip(*pairs_tf_re))[1]], dtype=torch.long)
        if pairs_tf_re
        else torch.zeros((2, 0), dtype=torch.long)
    )
    edge_index_re_tf = edge_index_tf_re.flip(0)

    re_tg_df = pd.read_csv(paths.re_tg_prior, sep="\t", dtype=str)
    re_col_name = next((column for column in re_tg_df.columns if column.lower() in ["re", "peak", "region"]), None)
    tg_col_name = next((column for column in re_tg_df.columns if column.lower() in ["tg", "gene", "target"]), None)

    pairs_re_tg: set[tuple[int, int]] = set()
    if re_col_name and tg_col_name:
        for _, row in re_tg_df.iterrows():
            re_name = fix_dash(str(row[re_col_name]))
            re_map = re_name if re_name in re2idx else map_re_to_known(re_name, re2idx, by_chr)
            tg_gene = str(row[tg_col_name])
            if (re_map in re2idx) and (tg_gene in tg2idx):
                pairs_re_tg.add((re2idx[re_map], tg2idx[tg_gene]))
    else:
        cols = re_tg_df.columns.tolist()
        for _, row in re_tg_df.iterrows():
            tg_gene = str(row[cols[0]])
            chrom = str(row[cols[4]])
            start = int(row[cols[5]])
            end = int(row[cols[6]])
            re_name = f"{chrom}:{start}-{end}"
            if tg_gene in tg2idx and re_name in re2idx:
                pairs_re_tg.add((re2idx[re_name], tg2idx[tg_gene]))

    edge_index_re_tg = (
        torch.tensor([list(zip(*pairs_re_tg))[0], list(zip(*pairs_re_tg))[1]], dtype=torch.long)
        if pairs_re_tg
        else torch.zeros((2, 0), dtype=torch.long)
    )
    edge_index_tg_re = edge_index_re_tg.flip(0)

    return {
        "input_paths": paths,
        "tf_df": tf_df,
        "tg_df": tg_df,
        "sym2fb": sym2fb,
        "tg_list_all": tg_list_all,
        "expr_df": expr_df,
        "peaks_df": peaks_df,
        "used_rna_layer": used_rna_layer,
        "used_atac_layer": used_atac_layer,
        "tf_list": tf_list,
        "tg_list": tg_list,
        "re_list": re_list,
        "num_TF": num_tf,
        "num_TG": num_tg,
        "num_RE": num_re,
        "tf2idx": tf2idx,
        "tg2idx": tg2idx,
        "re2idx": re2idx,
        "anno": anno,
        "by_chr": by_chr,
        "tf_re_edges": tf_re_edges,
        "re_tg_df": re_tg_df,
        "edge_index_tf_re": edge_index_tf_re,
        "edge_index_re_tf": edge_index_re_tf,
        "edge_index_re_tg": edge_index_re_tg,
        "edge_index_tg_re": edge_index_tg_re,
    }
