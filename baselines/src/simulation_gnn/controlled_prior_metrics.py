"""Fixed-panel evaluation, independent of each method's changing prior support."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score


def auc(y, x):
    return float(roc_auc_score(y, x)) if np.unique(y).size == 2 else None


def mean(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else None


def continuity(values):
    a = np.asarray(values, dtype=float)
    distance = np.linalg.norm(a[:, None] - a[None, :], axis=-1)
    maximum = float(distance.max())
    return None if maximum == 0 else 1 - float(np.diag(distance, 1).mean()) / maximum


def evaluate(scores, truth, prior, mask, same, logits=None):
    scores, truth = np.asarray(scores, float), np.asarray(truth, bool)
    ranks = scores if logits is None else np.asarray(logits, float)
    if (
        scores.shape != (5, 150, 50)
        or truth.shape != scores.shape
        or ranks.shape != scores.shape
    ):
        raise ValueError("Expected5x150x50 scores/truth")
    if not np.isfinite(scores).all() or not np.isfinite(ranks).all():
        raise ValueError("Nonfinite predictions")
    if mask.shape != prior.shape or not mask.any() or (truth & ~mask).any():
        raise ValueError("Fixed evaluation panel omits a true edge")
    # Primary raw-score continuity remains in each method's native scale.
    # Rank continuity is additional scale-robust evidence, never a replacement field.
    values = scores[:, mask]
    normalized_rank = np.stack(
        [rankdata(row, method="average") / len(row) for row in values]
    )
    changes = truth[1:] != truth[:-1]
    delta = np.abs(np.diff(values, axis=0))
    panel_auc = [auc(y[mask], x[mask]) for y, x in zip(truth, ranks)]
    result = {
        "fixed_panel_auroc": mean(panel_auc),
        "fixed_panel_auprc": mean(
            [
                float(average_precision_score(y[mask], x[mask]))
                for y, x in zip(truth, ranks)
            ]
        ),
        "within_module_auroc": mean(
            [auc(y[mask & same], x[mask & same]) for y, x in zip(truth, ranks)]
        ),
        "within_prior_auroc": mean(
            [auc(y[prior], x[prior]) for y, x in zip(truth, ranks)]
        ),
        "full_auroc": mean([auc(y.ravel(), x.ravel()) for y, x in zip(truth, ranks)]),
        "mean_continuity": continuity(values),
        "rank_continuity": continuity(normalized_rank),
        "change_auroc": mean([auc(y[mask], x) for y, x in zip(changes, delta)]),
        "truth_continuity": continuity(truth[:, mask].astype(float)),
        "active_prior_coverage": float((truth & prior).sum() / truth.sum()),
        "max_tie_fraction": float(
            np.mean(
                [
                    np.unique(row, return_counts=True)[1].max() / len(row)
                    for row in values
                ]
            )
        ),
        "constant_over_time": bool(
            np.array_equal(values, np.repeat(values[:1], 5, axis=0))
        ),
        "per_time_auroc": panel_auc,
    }
    return result
