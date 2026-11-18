"""Stage labels and embeddings, distances between stage networks, smoothness and Hi-C pair scores."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from network_evaluation.temporal import summarize_distances

from ..utils import ensure_directory


def safe_label(value: str) -> str:
    return re.sub(r'[\\/:*?"<>|\s]+', "_", str(value))


def make_unique_labels(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    output: list[str] = []
    for name in names:
        count = seen.get(name, 0) + 1
        seen[name] = count
        output.append(name if count == 1 else f"{name}_{count}")
    return output


def build_soma_labels(
    time_count: int,
    somatic_names: list[str] | None = None,
    cluster_ids: list[str] | None = None,
) -> list[str]:
    if somatic_names is not None and len(somatic_names) == time_count:
        return list(somatic_names)
    if cluster_ids is not None and len(cluster_ids) == time_count:
        return list(cluster_ids)
    return [f"t{idx}" for idx in range(time_count)]


def build_germ_labels(
    time_count: int,
    germline_types: list[str] | None = None,
    cluster_ids: list[str] | None = None,
) -> list[str]:
    if germline_types is not None and len(germline_types) == time_count:
        labels = []
        for idx, name in enumerate(germline_types):
            text = str(name).strip()
            if text.lower().startswith("germline_"):
                labels.append(text.lower())
                continue
            match = re.search(r"(\d+)$", text)
            labels.append(f"germline_{match.group(1)}" if match else f"germline_{idx}")
        return make_unique_labels(labels)
    if cluster_ids is not None and len(cluster_ids) == time_count:
        labels = []
        for idx, value in enumerate(cluster_ids):
            text = str(value).lower()
            labels.append(text if text.startswith("germline_") else f"germline_{idx}")
        return make_unique_labels(labels)
    return [f"germline_{idx}" for idx in range(time_count)]


def save_embeddings_by_label(
    out_tf: torch.Tensor,
    out_re: torch.Tensor,
    out_tg: torch.Tensor,
    labels: list[str],
    output_dir: str | Path,
) -> list[dict[str, str]]:
    target_dir = ensure_directory(output_dir)
    saved_paths: list[dict[str, str]] = []
    for time_idx, label in enumerate(labels):
        tag = safe_label(label)
        tf_path = target_dir / f"embedding_{tag}_TF.txt"
        re_path = target_dir / f"embedding_{tag}_RE.txt"
        tg_path = target_dir / f"embedding_{tag}_TG.txt"
        np.savetxt(
            tf_path,
            out_tf[:, time_idx, :].detach().cpu().numpy(),
            fmt="%.6f",
            delimiter="\t",
        )
        np.savetxt(
            re_path,
            out_re[:, time_idx, :].detach().cpu().numpy(),
            fmt="%.6f",
            delimiter="\t",
        )
        np.savetxt(
            tg_path,
            out_tg[:, time_idx, :].detach().cpu().numpy(),
            fmt="%.6f",
            delimiter="\t",
        )
        saved_paths.append(
            {
                "label": str(label),
                "tf": str(tf_path),
                "re": str(re_path),
                "tg": str(tg_path),
            }
        )
    return saved_paths


def _default_mm_dtype(device: torch.device, mm_dtype: torch.dtype | None) -> torch.dtype:
    if mm_dtype is not None:
        return mm_dtype
    return torch.float16 if device.type == "cuda" else torch.float32


def prob_norm2_blockwise(
    h_tf: torch.Tensor,
    h_tg: torch.Tensor,
    row_bs: int = 1024,
    col_bs: int = 4096,
    device: torch.device | None = None,
    mm_dtype: torch.dtype | None = None,
    sig_dtype: torch.dtype = torch.float32,
) -> float:
    compute_device = device or h_tf.device
    use_gpu = compute_device.type == "cuda"
    inner_dtype = _default_mm_dtype(compute_device, mm_dtype)
    total = 0.0
    n_rows, n_cols = h_tf.size(0), h_tg.size(0)
    for row_start in range(0, n_rows, row_bs):
        row_end = min(row_start + row_bs, n_rows)
        left = h_tf[row_start:row_end].to(compute_device, inner_dtype, non_blocking=True)
        for col_start in range(0, n_cols, col_bs):
            col_end = min(col_start + col_bs, n_cols)
            right = h_tg[col_start:col_end].to(compute_device, inner_dtype, non_blocking=True)
            logits = (left @ right.T).to(sig_dtype)
            probs = torch.sigmoid(logits)
            total += float((probs * probs).sum(dtype=torch.float64))
            del right, logits, probs
        del left
        if use_gpu:
            torch.cuda.empty_cache()
    return total


def prob_cross_blockwise(
    h_tf_t: torch.Tensor,
    h_tg_t: torch.Tensor,
    h_tf_s: torch.Tensor,
    h_tg_s: torch.Tensor,
    row_bs: int = 1024,
    col_bs: int = 4096,
    device: torch.device | None = None,
    mm_dtype: torch.dtype | None = None,
    sig_dtype: torch.dtype = torch.float32,
) -> float:
    compute_device = device or h_tf_t.device
    use_gpu = compute_device.type == "cuda"
    inner_dtype = _default_mm_dtype(compute_device, mm_dtype)
    total = 0.0
    n_rows, n_cols = h_tf_t.size(0), h_tg_t.size(0)
    for row_start in range(0, n_rows, row_bs):
        row_end = min(row_start + row_bs, n_rows)
        left_t = h_tf_t[row_start:row_end].to(compute_device, inner_dtype, non_blocking=True)
        left_s = h_tf_s[row_start:row_end].to(compute_device, inner_dtype, non_blocking=True)
        for col_start in range(0, n_cols, col_bs):
            col_end = min(col_start + col_bs, n_cols)
            right_t = h_tg_t[col_start:col_end].to(compute_device, inner_dtype, non_blocking=True)
            right_s = h_tg_s[col_start:col_end].to(compute_device, inner_dtype, non_blocking=True)
            logits_t = (left_t @ right_t.T).to(sig_dtype)
            logits_s = (left_s @ right_s.T).to(sig_dtype)
            probs_t = torch.sigmoid(logits_t)
            probs_s = torch.sigmoid(logits_s)
            total += float((probs_t * probs_s).sum(dtype=torch.float64))
            del right_t, right_s, logits_t, logits_s, probs_t, probs_s
        del left_t, left_s
        if use_gpu:
            torch.cuda.empty_cache()
    return total


def tf_tg_prob_frobenius_dist_matrix(
    out_tf: torch.Tensor,
    out_tg: torch.Tensor,
    row_bs: int = 1024,
    col_bs: int = 4096,
    mm_dtype: torch.dtype | None = None,
    sig_dtype: torch.dtype = torch.float32,
) -> np.ndarray:
    time_count = out_tf.size(1)
    compute_device = out_tf.device
    norms2 = np.zeros(time_count, dtype=np.float64)
    for time_idx in range(time_count):
        norms2[time_idx] = prob_norm2_blockwise(
            out_tf[:, time_idx, :],
            out_tg[:, time_idx, :],
            row_bs=row_bs,
            col_bs=col_bs,
            device=compute_device,
            mm_dtype=mm_dtype,
            sig_dtype=sig_dtype,
        )
    distances = np.zeros((time_count, time_count), dtype=np.float64)
    for time_idx in range(time_count):
        distances[time_idx, time_idx] = 0.0
        for other_idx in range(time_idx + 1, time_count):
            cross = prob_cross_blockwise(
                out_tf[:, time_idx, :],
                out_tg[:, time_idx, :],
                out_tf[:, other_idx, :],
                out_tg[:, other_idx, :],
                row_bs=row_bs,
                col_bs=col_bs,
                device=compute_device,
                mm_dtype=mm_dtype,
                sig_dtype=sig_dtype,
            )
            distance_sq = norms2[time_idx] + norms2[other_idx] - 2.0 * cross
            distances[time_idx, other_idx] = distances[other_idx, time_idx] = np.sqrt(
                max(0.0, distance_sq)
            )
    return distances


def summarize_smoothness(dist_mat: np.ndarray) -> dict[str, Any]:
    """Smoothness of a stage-distance matrix (mean_continuity) and the adjacent-mean score."""
    temporal = summarize_distances(dist_mat)
    time_count = int(dist_mat.shape[0])
    if time_count > 1:
        adjacent = np.array(
            [dist_mat[idx, idx + 1] for idx in range(time_count - 1)], dtype=np.float64
        )
        avg_adjacent = float(adjacent.mean())
        tri_i, tri_j = np.triu_indices(time_count, k=1)
        max_pairwise = float(dist_mat[tri_i, tri_j].max())
        overall_smooth = (
            1.0 - (avg_adjacent / (max_pairwise + 1e-12)) if max_pairwise > 0 else 0.0
        )
    else:
        avg_adjacent = 0.0
        max_pairwise = 0.0
        overall_smooth = 0.0
    if max_pairwise > 0:
        dist_mat_norm = dist_mat / (max_pairwise + 1e-12)
    else:
        dist_mat_norm = dist_mat.copy()
    return {
        "time_count": time_count,
        "avg_adjacent": avg_adjacent,
        "max_pairwise": max_pairwise,
        "overall_smoothness": temporal["temporal_continuity"],
        "adjacent_mean_smoothness": overall_smooth,
        "dist_mat_norm": dist_mat_norm,
        **temporal,
    }


def plot_smoothness_heatmap(
    dist_mat_norm: np.ndarray,
    plot_labels: list[str],
    overall_smoothness: float,
    title_prefix: str,
    output_path: str | Path | None = None,
    cmap: str = "YlOrRd",
    title_fs: int = 24,
    other_fs: int = 20,
    cbar_label: str | None = "Frobenius distance",
) -> str | None:
    time_count = int(dist_mat_norm.shape[0])
    fig, ax = plt.subplots(figsize=(max(5, time_count * 0.5), max(5, time_count * 0.5)))
    image = ax.imshow(
        dist_mat_norm,
        interpolation="nearest",
        aspect="auto",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_title(
        f"{title_prefix}\nTemporal continuity = {overall_smoothness:.4f}",
        fontsize=title_fs,
    )
    ax.set_xticks(range(time_count))
    ax.set_xticklabels(plot_labels, rotation=90, fontsize=other_fs)
    ax.set_yticks(range(time_count))
    ax.set_yticklabels(plot_labels, fontsize=other_fs)
    ax.tick_params(axis="both", which="major", labelsize=other_fs)
    colorbar = fig.colorbar(image, ax=ax)
    if cbar_label:
        colorbar.set_label(cbar_label, fontsize=other_fs)
    colorbar.ax.tick_params(labelsize=other_fs)
    fig.tight_layout()
    saved_path: str | None = None
    if output_path is not None:
        target_path = Path(output_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target_path, dpi=200)
        saved_path = str(target_path)
    plt.close(fig)
    return saved_path


def build_tg_alias2idx(tg2idx: dict[str, int], tg_list_path: str | Path) -> dict[str, int]:
    alias_map = dict(tg2idx)
    try:
        tg_full = pd.read_csv(tg_list_path, sep="\t", dtype=str).fillna("")
    except Exception:
        return alias_map
    if tg_full.empty:
        return alias_map
    name_col = tg_full.columns[0]
    if "FBgn_ID" not in tg_full.columns:
        return alias_map
    for _, row in tg_full.iterrows():
        symbol = str(row[name_col]).strip()
        fbgn_id = str(row["FBgn_ID"]).strip()
        if symbol in alias_map and fbgn_id:
            alias_map[fbgn_id] = alias_map[symbol]
    return alias_map


def strip_chr(value: str) -> str:
    text = (
        str(value)
        .strip()
        .replace(" ", "")
        .replace("_", "-")
        .replace("..", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
    )
    return text[3:] if text.lower().startswith("chr") else text


def map_hic_pairs(
    hic_path: str | Path,
    re2idx: dict[str, int],
    tg_alias2idx: dict[str, int],
    device: torch.device,
) -> dict[str, Any]:
    data = pd.read_csv(hic_path, sep="\t", header=0, dtype=str).fillna("")
    positives: set[tuple[int, int]] = set()
    missing_re = 0
    missing_tg = 0
    for _, row in data.iterrows():
        tg_idx = tg_alias2idx.get(str(row.get("gene_name", "")).strip())
        if tg_idx is None:
            tg_idx = tg_alias2idx.get(str(row.get("FBgn_ID", "")).strip())
        if tg_idx is None:
            missing_tg += 1
            continue
        re_key = strip_chr(row.get("RE", ""))
        if re_key in re2idx:
            positives.add((re2idx[re_key], tg_idx))
        else:
            missing_re += 1
    positive_list = list(positives)
    if not positive_list:
        raise ValueError(f"{hic_path}: no valid (RE, TG) pairs were mapped.")
    return {
        "pairs_idx": torch.tensor(positive_list, dtype=torch.long, device=device),
        "valid_count": len(positive_list),
        "miss_re": int(missing_re),
        "miss_tg": int(missing_tg),
        "source_path": str(Path(hic_path).resolve()),
    }


def score_pairs_over_time_sigmoid(
    out_left: torch.Tensor,
    out_right: torch.Tensor,
    pairs_idx: torch.Tensor,
    batch: int = 200_000,
    mm_dtype: torch.dtype | None = None,
    sig_dtype: torch.dtype = torch.float32,
) -> list[np.ndarray]:
    compute_device = out_left.device
    use_gpu = compute_device.type == "cuda"
    inner_dtype = _default_mm_dtype(compute_device, mm_dtype)
    left_index = pairs_idx[:, 0].long()
    right_index = pairs_idx[:, 1].long()
    time_count = out_left.size(1)
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for time_idx in range(time_count):
            left = out_left[:, time_idx, :].to(compute_device, inner_dtype, non_blocking=True)
            right = out_right[:, time_idx, :].to(compute_device, inner_dtype, non_blocking=True)
            batched_scores: list[torch.Tensor] = []
            for start in range(0, left_index.numel(), batch):
                stop = min(start + batch, left_index.numel())
                current = slice(start, stop)
                left_vec = left[left_index[current]]
                right_vec = right[right_index[current]]
                logits = (left_vec * right_vec).sum(dim=1).to(sig_dtype)
                batched_scores.append(torch.sigmoid(logits).detach().cpu())
            outputs.append(torch.cat(batched_scores).numpy())
            del left, right
            if use_gpu:
                torch.cuda.empty_cache()
    return outputs


def f1_at_threshold_over_time(
    pos_pairs_idx: torch.Tensor,
    out_left: torch.Tensor,
    out_right: torch.Tensor,
    timepoints: list[int] | None = None,
    thr: float = 0.5,
    rng_seed: int = 42,
    pair_batch: int = 200_000,
    mm_dtype: torch.dtype | None = None,
    sig_dtype: torch.dtype = torch.float32,
) -> tuple[list[int], list[float]]:
    num_left, num_right = out_left.size(0), out_right.size(0)
    pos_pairs_idx = pos_pairs_idx.long()
    num_positive = pos_pairs_idx.size(0)
    rng = np.random.default_rng(rng_seed)

    negative_pairs: list[tuple[int, int]] = []
    positive_set = {(int(i), int(j)) for i, j in pos_pairs_idx.detach().cpu().tolist()}
    while len(negative_pairs) < num_positive:
        left_idx = int(rng.integers(0, num_left))
        right_idx = int(rng.integers(0, num_right))
        if (left_idx, right_idx) not in positive_set:
            negative_pairs.append((left_idx, right_idx))
    neg_pairs_idx = torch.tensor(negative_pairs, dtype=torch.long, device=out_left.device)

    if timepoints is None:
        idx_list = list(range(out_left.size(1)))
    else:
        idx_list = sorted({int(value) for value in timepoints})
        if not all(0 <= value < out_left.size(1) for value in idx_list):
            raise ValueError("All timepoints must be valid embedding indices.")

    pos_scores = score_pairs_over_time_sigmoid(
        out_left,
        out_right,
        pos_pairs_idx,
        batch=pair_batch,
        mm_dtype=mm_dtype,
        sig_dtype=sig_dtype,
    )
    neg_scores = score_pairs_over_time_sigmoid(
        out_left,
        out_right,
        neg_pairs_idx,
        batch=pair_batch,
        mm_dtype=mm_dtype,
        sig_dtype=sig_dtype,
    )

    f1_values: list[float] = []
    for time_idx in idx_list:
        y_true = np.concatenate([np.ones(num_positive), np.zeros(num_positive)])
        y_prob = np.concatenate([pos_scores[time_idx], neg_scores[time_idx]])
        y_pred = (y_prob >= thr).astype(int)
        f1_values.append(float(f1_score(y_true, y_pred)))
    return idx_list, f1_values


def save_table(path: str | Path, frame: pd.DataFrame) -> str:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, sep="\t", index=False)
    return str(output_path)
