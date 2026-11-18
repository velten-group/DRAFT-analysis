"""Time-aware continuity diagnostics, independent of model fitting and truth labels.

The primary score penalizes the largest adjacent jump relative to the trajectory
diameter. It measures continuity of predictions, not accuracy of their dynamics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

VERSION = "max_adjacent_jump_v2"
POLICY = (
    "unsquared Frobenius on a fixed edge set; continuity = 1 - min(1, "
    "mean(time gaps) * max(adjacent distance / time gap) / trajectory diameter); "
    "constant trajectories score 1 and are flagged; fewer than 3 times: undefined"
)
SCALAR_METRICS = ("temporal_continuity", "velocity_roughness", "max_jump_ratio")


def frobenius_distances(scores):
    """Return unsquared distances between time-first score arrays."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim < 2 or not len(scores) or not scores[0].size:
        raise ValueError("Expected nonempty scores with axes (time, edges...)")
    if not np.isfinite(scores).all():
        raise ValueError("Scores must be finite")
    flat = scores.reshape(len(scores), -1)
    distances = np.zeros((len(scores), len(scores)), dtype=np.float64)
    for i in range(len(scores)):
        for j in range(i + 1, len(scores)):
            distances[i, j] = distances[j, i] = np.linalg.norm(flat[i] - flat[j])
    if not np.isfinite(distances).all():
        raise ValueError("Score distances overflowed; use a common finite scale")
    return distances


def summarize_distances(distances, time_points=None):
    """Evaluate an ordered, unsquared Euclidean/Frobenius distance matrix.

    Stage labels alone imply equal spacing. Pass numeric physical times only
    when their intervals have a meaningful interpretation. The function never
    reorders samples or estimates missing intermediate networks.
    """
    distances = np.asarray(distances, dtype=np.float64)
    if (
        distances.ndim != 2
        or distances.shape[0] != distances.shape[1]
        or not len(distances)
        or not np.isfinite(distances).all()
        or (distances < 0).any()
    ):
        raise ValueError("Expected a finite, nonnegative square distance matrix")
    n = len(distances)
    diameter = float(distances.max())
    tolerance = diameter * 1e-8
    if not np.allclose(distances, distances.T, rtol=1e-8, atol=tolerance):
        raise ValueError("Distance matrix must be symmetric")
    if np.any(np.diag(distances) > tolerance):
        raise ValueError("Distance matrix must have a zero diagonal")
    times = (
        np.arange(n, dtype=float)
        if time_points is None
        else np.asarray(time_points, dtype=float)
    )
    if (
        times.shape != (n,)
        or not np.isfinite(times).all()
        or (np.diff(times) <= 0).any()
    ):
        raise ValueError("Time points must be finite, unique and strictly increasing")
    gaps = np.diff(times)
    adjacent = np.diag(distances, k=1)
    constant = diameter == 0
    relative = distances / diameter if not constant else distances.copy()
    # Normalizing gaps makes the diagnostic invariant to the units of time.
    relative_gaps = gaps / gaps.mean() if n > 1 else gaps
    speeds = np.diag(relative, k=1) / relative_gaps
    largest = int(np.argmax(speeds)) if n > 1 else None
    jump_ratio = float(speeds.max()) if n > 1 else 0.0
    roughness = float("nan")
    if n >= 3:
        denominator = float(np.square(speeds).sum())
        numerator = 0.0
        # ||v[t+1]-v[t]||^2 is determined by three pairwise distances.
        for i in range(n - 2):
            left, right = relative_gaps[i : i + 2]
            coefficients = np.array([1 / left, -(1 / left + 1 / right), 1 / right])
            block = np.square(relative[i : i + 3, i : i + 3])
            value = float(-0.5 * coefficients @ block @ coefficients)
            if value < -1e-6 * max(denominator, np.finfo(float).tiny):
                raise ValueError("Distances are inconsistent with Euclidean increments")
            numerator += max(0.0, value)
        roughness = numerator / denominator if denominator else 0.0
    return {
        "metric_version": VERSION,
        "temporal_continuity": float(1 - min(1.0, jump_ratio))
        if n >= 3
        else float("nan"),
        "velocity_roughness": roughness,
        "max_jump_ratio": jump_ratio,
        "constant_over_time": constant,
        "time_count": n,
        "time_points": times.tolist(),
        "time_spacing": "ordinal_equal" if time_points is None else "supplied_numeric",
        "trajectory_diameter": diameter,
        "adjacent_distances": adjacent.tolist(),
        "relative_jump_speeds": speeds.tolist(),
        "largest_jump_interval": [float(times[largest]), float(times[largest + 1])]
        if largest is not None
        else None,
    }


def evaluate_temporal(scores, time_points=None):
    return summarize_distances(frobenius_distances(scores), time_points)


def json_finite(value):
    """Keep undefined short-trajectory metrics valid JSON (null)."""
    if isinstance(value, dict):
        return {key: json_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_finite(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--distance-matrix",
        type=Path,
        required=True,
        help="Headerless tab-separated Frobenius distances in chronological order",
    )
    parser.add_argument(
        "--squared",
        action="store_true",
        help="Explicitly convert squared distances to unsquared Frobenius distances",
    )
    parser.add_argument(
        "--times",
        type=float,
        nargs="+",
        help="Optional physical times; default: equally spaced stages",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    source = args.distance_matrix.resolve()
    distances = np.loadtxt(source, delimiter="\t", ndmin=2)
    if args.squared:
        if (distances < 0).any():
            parser.error("Squared distances must be nonnegative")
        distances = np.sqrt(distances)
    report = summarize_distances(distances, args.times)
    report.update(
        input_file=str(source),
        input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        policy=POLICY,
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    target = output / "temporal_metrics.json"
    if target == source:
        parser.error("Output must not overwrite the input")
    target.write_text(
        json.dumps(json_finite(report), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(json_finite(report), indent=2, allow_nan=False))
    print(f"Saved: {target}")


if __name__ == "__main__":
    main()
