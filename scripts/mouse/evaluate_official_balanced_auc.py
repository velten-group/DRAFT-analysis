"""Balanced 1:1 AUROC (Fig. 3b) on the gold-covered TF-TG pairs of the official AUPR.

The sampled test edges are the same for every method.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from evaluate import sample_pairs
from official_aupr_protocol import content_hash


def make_sample(gold, cid):
    if gold.duplicated(["TF", "TG"]).any() or gold.TF.eq(gold.TG).any():
        raise ValueError("The filtered gold standard must contain unique non-self edges.")
    tfs, tgs = sorted(gold.TF.unique()), sorted(gold.TG.unique())
    tf_index, tg_index = {v: i for i, v in enumerate(tfs)}, {v: i for i, v in enumerate(tgs)}
    n_tg = len(tgs)
    codes = np.arange(len(tfs) * n_tg, dtype=np.int64)
    holes = [i * n_tg + tg_index[tf] for i, tf in enumerate(tfs) if tf in tg_index]
    codes = np.delete(codes, holes)
    positive = gold.TF.map(tf_index).to_numpy(dtype=np.int64) * n_tg + gold.TG.map(tg_index).to_numpy(dtype=np.int64)
    positions, labels, calibration = sample_pairs(codes, positive, 2025, cid)
    test_codes, truth = codes[positions[~calibration]], labels[~calibration]
    if truth.sum() * 2 != len(truth):
        raise ValueError("Balanced test must contain exactly equal positive/negative counts.")
    checksum = hashlib.sha256(content_hash({"TF": tfs, "TG": tgs}).encode())
    checksum.update(test_codes.astype("<i8").tobytes())
    checksum.update(truth.astype("u1").tobytes())
    return {"TF": tfs, "TG": tgs, "codes": codes, "gold_codes": positive,
            "test_codes": test_codes, "test_labels": truth, "test_pairs_sha256": checksum.hexdigest()}


def project_scores(frame, sample):
    values = pd.to_numeric(frame.score, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Scores must be finite and nonnegative.")
    tf_index = {v: i for i, v in enumerate(sample["TF"])}
    tg_index = {v: i for i, v in enumerate(sample["TG"])}
    i, j = frame.TF.map(tf_index), frame.TG.map(tg_index)
    valid = i.notna() & j.notna()
    codes = i[valid].to_numpy(dtype=np.int64) * len(tg_index) + j[valid].to_numpy(dtype=np.int64)
    full = np.zeros(len(tf_index) * len(tg_index), dtype=float)
    np.maximum.at(full, codes, values[valid.to_numpy()])
    return full
