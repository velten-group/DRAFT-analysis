"""Paired prior interventions, with a fixed biological/RE universe.

Pure NumPy/pandas: external environments need not import DRAFT or Torch.
Simulator states and module labels are used only for construction/evaluation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "simulation_controlled_prior_v1"
INTERNAL = ("draft", "linear_gru", "pooled_linear", "pooled_gat")
OFFICIAL = ("scenicplus", "scmtni", "scdori", "dictys")
METHODS = INTERNAL + OFFICIAL
CASES = {
    "base": ("reference", 0),
    "structure_low": ("structure", -1),
    "structure_high": ("structure", 1),
    "size_12": ("prior_size", 12),
    "size_20": ("prior_size", 20),
    "error_20": ("prior_error", 20),
    "error_40": ("prior_error", 40),
    "turnover_0": ("turnover", 0),
    "turnover_5": ("turnover", 5),
}
AXES = {
    "structure": [("structure_low", -1), ("base", 0), ("structure_high", 1)],
    "prior_size": [("size_12", 12), ("base", 16), ("size_20", 20)],
    "prior_error": [("base", 0), ("error_20", 20), ("error_40", 40)],
    "turnover": [("turnover_0", 0), ("turnover_5", 5), ("base", 10)],
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    import os

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def frame(matrix, rows, columns, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(np.asarray(matrix, dtype=int), index=rows, columns=columns).to_csv(
        path, lineterminator="\n"
    )


def rebind(original, original_motif, owner, candidate):
    """Replace TF endpoints in vacated RE slots; persistent links keep RE IDs."""
    result = original_motif.copy()
    for g in range(len(candidate)):
        old = np.flatnonzero(original[g] & ~candidate[g])
        new = np.flatnonzero(candidate[g] & ~original[g])
        if len(old) != len(new):
            raise ValueError("Use rebind only for equal per-TG degree interventions")
        for before, after in zip(old, new):
            region = np.flatnonzero((owner == g) & result[:, before])
            if len(region) != 1:
                raise ValueError("Each public pair must occupy one declared RE slot")
            result[region[0], before], result[region[0], after] = False, True
    return result


def structure_swap(candidate, signal, same, evaluation, seed, direction, steps=150):
    """Change block association using only false edges; exact row/column degrees."""
    result = candidate.copy()
    rng = np.random.default_rng(np.random.SeedSequence([seed, 202, direction + 1]))
    desired = int((candidate & same).sum()) + direction * steps
    current = int((result & same).sum())
    for _ in range(400000):
        if direction * (desired - current) <= 0:
            break
        g, h = rng.choice(len(result), 2, replace=False)
        aa = np.flatnonzero(result[g] & ~signal[g] & ~evaluation[g])
        bb = np.flatnonzero(result[h] & ~signal[h] & ~evaluation[h])
        if not len(aa) or not len(bb):
            continue
        a = int(rng.choice(aa))
        b = int(rng.choice(bb))
        if (
            a == b
            or result[g, b]
            or result[h, a]
            or signal[g, b]
            or signal[h, a]
            or evaluation[g, b]
            or evaluation[h, a]
        ):
            continue
        delta = int(same[g, b]) + int(same[h, a]) - int(same[g, a]) - int(same[h, b])
        if direction * delta <= 0 or abs(delta) > abs(desired - current):
            continue
        result[g, a] = result[h, b] = False
        result[g, b] = result[h, a] = True
        current += delta
    if current != desired:
        raise ValueError(
            f"Cannot reach prespecified block count: {current} vs {desired}"
        )
    np.testing.assert_array_equal(result.sum(0), candidate.sum(0))
    np.testing.assert_array_equal(result.sum(1), candidate.sum(1))
    assert np.array_equal(result & signal, candidate & signal)
    np.testing.assert_array_equal(result & evaluation, candidate & evaluation)
    return result


def corrupt(candidate, signal, same, seed, fraction):
    """Swap two correct endpoints into two wrong links within the same module.

    Exact candidate TF/TG degrees, per-TG block counts and RE slot degrees survive.
    Fraction is of correct potential links, not of all currently inactive edges.
    """
    result = candidate.copy()
    rng = np.random.default_rng(np.random.SeedSequence([seed, 303]))
    wanted = round(float(fraction) * int(signal.sum()))
    if wanted % 2:
        raise ValueError("The fixed paired-switch design requires an even loss count")
    removed = 0
    for _ in range(500000):
        if removed == wanted:
            break
        g, h = rng.choice(len(result), 2, replace=False)
        aa, bb = (
            np.flatnonzero(result[g] & signal[g]),
            np.flatnonzero(result[h] & signal[h]),
        )
        if not len(aa) or not len(bb):
            continue
        a, b = int(rng.choice(aa)), int(rng.choice(bb))
        if (
            a == b
            or result[g, b]
            or result[h, a]
            or signal[g, b]
            or signal[h, a]
            or not same[g, b]
            or not same[h, a]
        ):
            continue
        result[g, a] = result[h, b] = False
        result[g, b] = result[h, a] = True
        removed += 2
    if removed != wanted:
        raise ValueError("Could not realize the predeclared corruption fraction")
    for axis in (0, 1):
        np.testing.assert_array_equal(result.sum(axis), candidate.sum(axis))
    np.testing.assert_array_equal((result & same).sum(1), (candidate & same).sum(1))
    return result


def build(seed):
    """50TF/150TG/3 modules;16 public candidates;6 active;fixed1500 REs."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 101]))
    tf_module, tg_module = (
        rng.permutation(np.arange(50) % 3),
        rng.permutation(np.arange(150) % 3),
    )
    same = tg_module[:, None] == tf_module[None, :]
    signal, small, base, physical = [np.zeros((150, 50), bool) for _ in range(4)]
    motifs = [np.zeros((1500, 50), bool) for _ in range(3)]
    owner = np.repeat(np.arange(150), 10)
    for g in range(150):
        inside = rng.choice(np.flatnonzero(same[g]), 15, replace=False)
        outside = rng.choice(np.flatnonzero(~same[g]), 5, replace=False)
        signal[g, inside[:9]] = True
        first = rng.permutation(np.r_[inside[:9], outside[:3]])
        middle = np.r_[inside[9:12], outside[3]]
        last = np.r_[inside[12:15], outside[4]]
        for matrix, values in (
            (small, first),
            (base, np.r_[first, middle]),
            (physical, np.r_[first, middle, last]),
        ):
            matrix[g, values] = True
        regions = (
            np.r_[np.arange(10), [0, 1], np.arange(2, 6), np.arange(6, 10)] + g * 10
        )
        tf_order = np.r_[first, middle, last]
        for motif, count in zip(motifs, (12, 16, 20)):
            motif[regions[:count], tf_order[:count]] = True
    rg = np.zeros((150, 1500), bool)
    rg[owner, np.arange(1500)] = True
    evaluation = signal.copy()
    for g in range(150):
        evaluation[
            g, rng.choice(np.flatnonzero(same[g] & ~signal[g]), 3, replace=False)
        ] = True
        evaluation[g, rng.choice(np.flatnonzero(~same[g]), 6, replace=False)] = True
    priors = {"base": base, "size_12": small, "size_20": physical}
    for name, direction in (("structure_low", -1), ("structure_high", 1)):
        priors[name] = structure_swap(base, signal, same, evaluation, seed, direction)
    for percent in (20, 40):
        priors[f"error_{percent}"] = corrupt(base, signal, same, seed, percent / 100)
    for percent in (0, 5):
        priors[f"turnover_{percent}"] = base.copy()
    public_motifs = {"size_12": motifs[0], "size_20": motifs[2], "base": motifs[1]}
    for name, matrix in priors.items():
        if name not in public_motifs:
            public_motifs[name] = rebind(base, motifs[1], owner, matrix)
        assert public_motifs[name].any(1).all()
        np.testing.assert_array_equal(
            rg.astype(int) @ public_motifs[name].astype(int) > 0, matrix
        )
    initial = np.zeros_like(signal)
    dynamics = np.random.default_rng(np.random.SeedSequence([seed, 811]))
    for g in range(150):
        initial[g, dynamics.choice(np.flatnonzero(signal[g]), 6, replace=False)] = True
    old = [list(dynamics.permutation(np.flatnonzero(initial[g]))) for g in range(150)]
    new = [
        list(dynamics.permutation(np.flatnonzero(signal[g] & ~initial[g])))
        for g in range(150)
    ]
    events = []
    for _ in range(360):
        g = int(dynamics.choice([i for i in range(150) if new[i]]))
        events.append((g, int(old[g].pop()), int(new[g].pop())))
    trajectories = {}
    for rate in (0, 5, 10):
        states = [initial.copy()]
        for t in range(1, 5):
            state = initial.copy()
            for g, a, b in events[: t * round(900 * rate / 100)]:
                state[g, a], state[g, b] = False, True
            states.append(state)
        trajectories[rate] = np.stack(states)
    return {
        "signal": signal,
        "priors": priors,
        "motifs": public_motifs,
        "physical": physical,
        "physical_motif": motifs[2],
        "re_tg": rg,
        "evaluation": evaluation,
        "same": same,
        "modules": np.r_[tf_module, tg_module],
        "trajectories": trajectories,
        "events": events,
    }


def rate_for(case):
    return int(CASES[case][1]) if case.startswith("turnover_") else 10


def write_biology(folder, design, rate):
    folder = Path(folder)
    genes = [str(i + 1) for i in range(200)]
    regions = [f"RE_{i + 1}" for i in range(1500)]
    frame(
        design["physical_motif"],
        regions,
        genes[:50],
        folder / "generator_only/potential_motif.csv",
    )
    frame(
        design["re_tg"].T,
        regions,
        genes[50:],
        folder / "generator_only/potential_distance.csv",
    )
    frame(
        design["physical_motif"],
        regions,
        genes[:50],
        folder / "multiome/tf_re_prior.csv",
    )
    frame(design["re_tg"], genes[50:], regions, folder / "multiome/re_tg_prior.csv")
    frame(design["physical"], genes[50:], genes[:50], folder / "candidate_prior.csv")
    for t, state in enumerate(design["trajectories"][rate]):
        full = np.zeros((200, 200), bool)
        full[50:, :50] = state
        frame(full, genes, genes, folder / f"grn_time_{t}.csv")
        frame(
            state,
            genes[50:],
            genes[:50],
            folder / f"generator_only/weights_time_{t}.csv",
        )
    per_step = round(900 * rate / 100)
    records = [
        (t, *event)
        for t in range(1, 5)
        for event in design["events"][(t - 1) * per_step : t * per_step]
    ]
    pd.DataFrame(
        records, columns=["time", "target_index", "old_tf_index", "new_tf_index"]
    ).to_csv(folder / "generator_only/transitions.tsv", sep="\t", index=False)
    pd.DataFrame({"gene": genes, "module": design["modules"]}).to_csv(
        folder / "gene_modules.tsv", sep="\t", index=False
    )


def write_public(folder, design, case):
    folder = Path(folder)
    genes, regions = (
        [str(i + 1) for i in range(200)],
        [f"RE_{i + 1}" for i in range(1500)],
    )
    frame(
        design["priors"][case], genes[50:], genes[:50], folder / "candidate_prior.csv"
    )
    frame(
        design["motifs"][case], regions, genes[:50], folder / "multiome/tf_re_prior.csv"
    )
    frame(design["re_tg"], genes[50:], regions, folder / "multiome/re_tg_prior.csv")
    frame(design["evaluation"], genes[50:], genes[:50], folder / "evaluation_mask.csv")
    frame(
        design["signal"],
        genes[50:],
        genes[:50],
        folder / "evaluation_only/potential_true.csv",
    )
