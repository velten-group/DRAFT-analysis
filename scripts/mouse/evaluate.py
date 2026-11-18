"""Balanced edge sample with a calibration/test split, and the size-matched Random network of Fig. 3b."""

from __future__ import annotations

import numpy as np
from common import gene_key


def sample_pairs(codes: np.ndarray, gold: np.ndarray, seed: int, cid: int):
    """All gold edges and as many uniformly drawn non-gold edges, each split in half into calibration and test edges."""
    positive = np.flatnonzero(np.isin(codes, gold))
    negative = np.flatnonzero(~np.isin(codes, gold))
    if len(positive) < 4 or len(negative) < len(positive):
        raise ValueError(
            f"C{cid}: need >=4 positive edges and at least as many unreported edges."
        )
    rng = np.random.default_rng(np.random.SeedSequence([seed, cid]))
    positive = rng.permutation(positive)
    negative = rng.choice(negative, size=len(positive), replace=False)
    positions = np.r_[positive, negative]
    labels = np.r_[
        np.ones(len(positive), dtype=int), np.zeros(len(positive), dtype=int)
    ]
    calibration = np.zeros(len(positions), dtype=bool)
    n_cal = len(positive) // 2
    calibration[:n_cal] = True
    calibration[len(positive) : len(positive) + n_cal] = True
    return positions, labels, calibration


def random_network(
    codes: np.ndarray,
    tfs: list[str],
    tgs: list[str],
    genes: list[str],
    seed: int,
    cid: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Uniform random TF-TG edges (no self-pairs), as many as `codes`; returns the scores of `codes`, the chosen edges and the settings."""
    keys = {gene_key(x) for x in genes}
    local_tf = np.array([i for i, name in enumerate(tfs) if gene_key(name) in keys])
    local_tg = np.array([j for j, name in enumerate(tgs) if gene_key(name) in keys])
    target_key_idx = {gene_key(tgs[j]): n for n, j in enumerate(local_tg)}
    holes = np.array(
        [
            n * len(local_tg) + target_key_idx[gene_key(tfs[i])]
            for n, i in enumerate(local_tf)
            if gene_key(tfs[i]) in target_key_idx
        ],
        dtype=np.int64,
    )
    n_pool = len(local_tf) * len(local_tg) - len(holes)
    k = len(codes)  # one random edge per evaluated edge
    if k > n_pool:
        raise ValueError("Random budget exceeds the available non-self TF/TG pool.")
    rng = np.random.default_rng(np.random.SeedSequence([seed, cid]))
    ranks = rng.choice(n_pool, size=k, replace=False)
    flat = ranks + np.searchsorted(holes - np.arange(len(holes)), ranks, side="right")
    chosen = np.sort(
        local_tf[flat // len(local_tg)] * len(tgs) + local_tg[flat % len(local_tg)]
    )
    return (
        np.isin(codes, chosen).astype(float),
        chosen,
        {"k": k, "n_pool": n_pool, "seed": seed},
    )
