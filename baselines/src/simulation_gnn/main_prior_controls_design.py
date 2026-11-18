"""Prior-only interventions on the exact 9-inside/3-outside main biology.

Simulator labels enter construction and evaluation only, never fitting inputs.
"""

from __future__ import annotations

import itertools
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .controlled_prior_design import frame, rebind
    from .prior_m3_design import write_design
except ImportError:  # Native baseline environments must not import the Torch package.
    from controlled_prior_design import frame, rebind
    from prior_m3_design import write_design

VERSION = "simulation_main_prior_controls_v2"
CASES = {
    "base": ("reference", 0),
    "size_16": ("prior_size", 16),
    "size_20": ("prior_size", 20),
    "error_20": ("extra_corruption", 20),
    "error_40": ("extra_corruption", 40),
}
AXES = {
    "prior_size": [("base", 12), ("size_16", 16), ("size_20", 20)],
    "extra_corruption": [("base", 0), ("error_20", 20), ("error_40", 40)],
}
PARAMS = {
    "num_tf": 50,
    "num_genes": 200,
    "modules": 3,
    "times": 5,
    "active_per_tg": 6,
    "active_within": 6,
    "candidate_inside": 9,
    "candidate_outside": 3,
    "change_fraction": 0.1,
    "tfs_per_region": 2,
}


def matrix(path):
    return pd.read_csv(path, index_col=0).to_numpy(bool)


def corruption_path(base, truth, same, seed):
    """Nested swaps preserve TF/TG degrees, RE slots and exact per-time coverage.

    Remove 20/40% of *ever-active* relationships, stratified by the entire
    five-time activity pattern. Two edges in the same module and pattern swap
    TF endpoints. New endpoints must be never-active for those TGs.
    """
    ever = truth.any(0)
    pattern = np.sum(truth.astype(int) * (2 ** np.arange(5))[:, None, None], axis=0)
    patterns, counts = np.unique(pattern[ever], return_counts=True)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 20261112]))
    result, output = base.copy(), {}
    for percent in (20, 40):
        wanted = {int(p): int(n * percent // 100) for p, n in zip(patterns, counts)}
        if any(
            wanted[int(p)] * 100 != int(n) * percent or wanted[int(p)] % 2
            for p, n in zip(patterns, counts)
        ):
            raise ValueError(
                "Activity-pattern counts cannot realize paired exact corruption"
            )
        removed = {int(p): int(((~result) & (pattern == p)).sum()) for p in patterns}
        for _ in range(500000):
            pending = [p for p in wanted if removed[p] < wanted[p]]
            if not pending:
                break
            p = int(rng.choice(pending))
            edges = np.argwhere(result & (pattern == p))
            (g, a), (h, b) = edges[rng.choice(len(edges), 2, replace=False)]
            if (
                g == h
                or a == b
                or result[g, b]
                or result[h, a]
                or ever[g, b]
                or ever[h, a]
                or not same[g, b]
                or not same[h, a]
            ):
                continue
            result[g, a] = result[h, b] = False
            result[g, b] = result[h, a] = True
            removed[p] += 2
        if removed != wanted:
            raise ValueError(
                f"Cannot realize degree-preserving corruption: {removed} / {wanted}"
            )
        for axis in (0, 1):
            np.testing.assert_array_equal(result.sum(axis), base.sum(axis))
        np.testing.assert_array_equal((result & same).sum(1), (base & same).sum(1))
        np.testing.assert_array_equal(
            (truth & result).sum((1, 2)), np.full(5, 900 * (100 - percent) // 100)
        )
        output[f"error_{percent}"] = result.copy()
    return output


def build(seed, folder=None):
    """Read the main input, or reproduce its exact design without simulating assays."""
    if folder is None:
        with tempfile.TemporaryDirectory(prefix="main_prior_design_") as temporary:
            write_design(temporary, dict(PARAMS, seed=seed))
            return build(seed, Path(temporary))
    folder = Path(folder)
    base = matrix(folder / "candidate_prior.csv")
    motif = matrix(folder / "multiome/tf_re_prior.csv")
    rg = matrix(folder / "multiome/re_tg_prior.csv")
    modules = pd.read_csv(folder / "gene_modules.tsv", sep="\t").module.to_numpy()
    same = modules[50:, None] == modules[None, :50]
    truth = np.stack([matrix(folder / f"grn_time_{t}.csv")[50:, :50] for t in range(5)])
    if (
        base.shape != (150, 50)
        or motif.shape != (900, 50)
        or rg.shape != (150, 900)
        or not np.all(base.sum(1) == 12)
        or not np.all((base & same).sum(1) == 9)
        or not np.all(truth.sum(2) == 6)
        or (truth & ~base).any()
        or (truth & ~same).any()
    ):
        raise ValueError(
            "Expected the unchanged 6-active / 9-inside+3-outside main design"
        )
    if [(a & ~b).sum() for a, b in itertools.pairwise(truth)] != [90] * 4:
        raise ValueError("Expected 10% one-way replacement at every step")
    if (np.abs(np.diff(truth.astype(int), axis=0)).sum(0) > 1).any():
        raise ValueError("Main truth contains edge reversals")
    np.testing.assert_array_equal(rg.astype(int) @ motif.astype(int) > 0, base)
    if not np.all(rg.sum(0) == 1) or not np.all(rg.sum(1) == 6):
        raise ValueError("Expected six fixed REs per TG")
    owner = rg.argmax(0)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 20261113]))
    priors, motifs = {"base": base.copy()}, {"base": motif.copy()}
    p, m = base.copy(), motif.copy()
    for size in (16, 20):
        for g in range(150):
            added = np.r_[
                rng.choice(np.flatnonzero(same[g] & ~p[g]), 3, replace=False),
                rng.choice(np.flatnonzero(~same[g] & ~p[g]), 1, replace=False),
            ]
            regions = np.flatnonzero(owner == g)
            for tf in added:
                degree = m[regions].sum(1)
                region = rng.choice(regions[degree == degree.min()])
                p[g, tf] = m[region, tf] = True
        priors[f"size_{size}"], motifs[f"size_{size}"] = p.copy(), m.copy()
    priors.update(corruption_path(base, truth, same, seed))
    for name in ("error_20", "error_40"):
        motifs[name] = rebind(base, motif, owner, priors[name])
    for name in CASES:
        np.testing.assert_array_equal(
            rg.astype(int) @ motifs[name].astype(int) > 0, priors[name]
        )
    return {
        "priors": priors,
        "motifs": motifs,
        "re_tg": rg,
        "truth": truth,
        "same": same,
        "evaluation": base.copy(),
        "union": np.logical_or.reduce(list(priors.values())),
        "ever": truth.any(0),
    }


def write_public(folder, d, case):
    folder = Path(folder)
    genes = [str(i + 1) for i in range(200)]
    regions = [f"RE_{i + 1}" for i in range(900)]
    for a, rows, cols, name in (
        (d["priors"][case], genes[50:], genes[:50], "candidate_prior.csv"),
        (d["motifs"][case], regions, genes[:50], "multiome/tf_re_prior.csv"),
        (d["evaluation"], genes[50:], genes[:50], "evaluation_mask.csv"),
        (d["union"], genes[50:], genes[:50], "evaluation_only/union_mask.csv"),
        (d["ever"], genes[50:], genes[:50], "evaluation_only/ever_active.csv"),
    ):
        frame(a, rows, cols, folder / name)
