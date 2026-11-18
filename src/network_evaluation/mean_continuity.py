"""Temporal smoothness: 1 - mean adjacent distance / largest distance between stage networks."""

from __future__ import annotations

import numpy as np

VERSION = "mean_adjacent_frobenius_v1"
POLICY = (
    "1 - mean unsquared adjacent Frobenius distance / maximum all-pairs distance; "
    "fixed candidates and equally spaced formal times; constant=1 and flagged separately"
)


def mean_continuity(scores):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim < 2 or len(scores) < 3 or not scores[0].size:
        raise ValueError("Mean continuity requires at least three nonempty snapshots")
    if not np.isfinite(scores).all():
        raise ValueError("Mean continuity scores must be finite")
    flat = scores.reshape(len(scores), -1)
    diameter = max(np.linalg.norm(x - y) for x in flat for y in flat)
    if not np.isfinite(diameter):
        raise ValueError("Nonfinite trajectory diameter")
    if diameter == 0:
        return 1.0
    adjacent = np.linalg.norm(np.diff(flat, axis=0), axis=1)
    return float(1 - adjacent.mean() / diameter)
