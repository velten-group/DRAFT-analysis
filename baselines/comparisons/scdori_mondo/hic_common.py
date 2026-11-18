"""Shared Hi-C ID mapping and balanced pair sampling without model dependencies."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


def region_key(value: str) -> str:
    text = str(value).strip().replace(" ", "").replace("_", "-").replace("..", "-")
    text = text.replace("–", "-").replace("—", "-")
    return re.sub(r"^chr", "", text, flags=re.IGNORECASE)


def map_positives(path: Path, peaks: list[str], genes: list[str], tg: pd.DataFrame):
    table = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    if "RE" not in table or not {"FBgn_ID", "gene_name"}.intersection(table.columns):
        raise ValueError(f"Hi-C labels must identify REs and target genes: {path}")
    pidx = {region_key(p): i for i, p in enumerate(peaks)}
    if len(pidx) != len(peaks):
        raise ValueError("Ambiguous peak names after chr-prefix normalization.")
    gidx = {g: i for i, g in enumerate(genes)}
    aliases = dict(gidx)
    for row in tg.itertuples(index=False):
        if row.FBgn_ID in gidx:
            aliases[str(row.gene_name).strip()] = gidx[row.FBgn_ID]
    pairs = set()
    missing_gene = missing_peak = 0
    for row in table.to_dict("records"):
        gene = aliases.get(row.get("FBgn_ID", ""))
        if gene is None:
            gene = aliases.get(row.get("gene_name", ""))
        if gene is None:
            missing_gene += 1
            continue
        peak = pidx.get(region_key(row["RE"]))
        if peak is None:
            missing_peak += 1
            continue
        pairs.add((peak, gene))
    if not pairs:
        raise ValueError(
            f"No Hi-C positives match the model's non-TF target/peak pool: {path}"
        )
    positives = np.asarray(sorted(pairs), dtype=np.int64)
    return positives, {
        "source_rows": len(table),
        "mapped_unique_positive_pairs": len(positives),
        "missing_gene_rows": missing_gene,
        "missing_peak_rows_after_gene_mapping": missing_peak,
        "mapping": "exact peak coordinates after chr-prefix normalization; FBgn ID then gene-symbol alias; duplicate pairs removed",
    }


def negative_pairs(positives, n_peaks, n_genes, seed, allowed_codes=None):
    """Match DRAFT's balanced uniform sampling with replacement (seed fixed)."""
    positive_set = set(map(tuple, positives.tolist()))
    rng = np.random.default_rng(seed)
    count = len(positives)
    if allowed_codes is not None:
        positive_codes = positives[:, 0] * n_genes + positives[:, 1]
        eligible = np.setdiff1d(allowed_codes, positive_codes, assume_unique=True)
        if not len(eligible):
            raise ValueError("No unobserved pairs remain inside the prior support.")
        codes = rng.choice(eligible, size=count, replace=True)
        return np.column_stack([codes // n_genes, codes % n_genes])
    if len(positive_set) >= n_peaks * n_genes:
        raise ValueError("No unobserved pairs remain in the feature pool.")
    negatives = []
    while len(negatives) < count:
        pair = (int(rng.integers(n_peaks)), int(rng.integers(n_genes)))
        if pair not in positive_set:
            negatives.append(pair)
    return np.asarray(negatives, dtype=np.int64)


def metrics(y_true, scores, threshold, inclusive=False):
    values = np.asarray(scores, dtype=np.float64) if inclusive else np.asarray(scores)
    predicted = values >= threshold if inclusive else values > threshold
    truth = np.asarray(y_true, dtype=bool)
    tp = int((truth & predicted).sum())
    fp = int((~truth & predicted).sum())
    fn = int((truth & ~predicted).sum())
    tn = int((~truth & ~predicted).sum())
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }
