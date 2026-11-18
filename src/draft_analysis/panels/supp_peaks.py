"""Supplementary Figs. 10 and 11: Mondo profiles of DRAFT and its ablations; controls for the TF peak shifts.

Profiles come from the dense network exports of the checkpoints (fly stages network, seeds and controls):
TF network activity = sum of the selected outgoing TF-TG probabilities per stage (scripts/seed_replicates.py); peak
shift = network peak >= 3 stages from the expression peak (network smoothed with sigma 0.5 stage, expression raw stage
means, near-maximum 0.95, relative dynamic range >= 0.1).

10  Mondo centrality on the dense time grid (0.2-stage steps) of DRAFT (seed 2025) and the two ablations, smoothed and
    drawn as cubic curves, each scaled to its maximum (drosophila_gnn.analysis.mondo_supplement); mean Mondo expression;
    SCENIC+ = distinct Mondo targets in the published dAWA stage eRegulons (DAR = yes, DEG = yes).
11  a: observed TFs with a peak shift and 10,000 random re-pairings of network and expression profiles within each
       method's evaluable TFs (DRAFT from the checkpoint, SCENIC+ and
       scDoRI from their deposited profile table). b: Fisher tests between the five germline models. c: Spearman rho of
       network-peak stages between full DRAFT and models with another seed (prior controls, ablations).
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .. import paths
from ..fly import controls, numbering
from . import fig5, save

K = "supplementary"
REAL = ("2025", "2024", "2026", "2027", "2028")  # row order of the published table
SHUFFLED = ("tfu2025", "tfu2026", "shuf2025", "shuf2026", "shuflow2025")
ABLATION_MATRIX = ("2025", "2027", "2028", "nogru25", "nogru27", "nogru28", "lingru25", "lingru27", "lingru28")
PERMUTATIONS, PERMUTATION_SEED, FAR = 10_000, 2025, 3


def real_dense(seed: str):
    return paths.FLY / ("germ" if seed == "2025" else f"germ_seed{seed}") / "network_dense"


def dense(name: str):
    return real_dense(name) if name in REAL else controls.dense_dir(name)


class Profiles:
    """Network and expression profiles and the peak table of every model, computed once."""

    def __init__(self):
        self.sr = fig5._sr()
        self.tfs, self.aliases = self.sr.load_tfs(paths.data("fly", "raw", "TF_filtered.tsv"))
        self.expression = self.sr.expression_profiles(paths.DATA / "fly" / "raw", self.tfs)
        self._net, self._tab = {}, {}

    def network(self, name: str) -> np.ndarray:
        if name not in self._net:
            d = dense(name)
            if len(list(d.glob("TF_TG_dense_t*.tsv"))) != 15 * 5 - 4:  # 15 stages, 4 interpolated points between
                raise FileNotFoundError(f"{d}: dense network export incomplete; run the network/seeds/controls stages")
            self._net[name] = self.sr.network_profiles(d, self.tfs, self.aliases)
        return self._net[name]

    def table(self, name: str) -> pd.DataFrame:
        if name not in self._tab:
            self._tab[name] = self.sr.offsets_table(self.tfs, self.expression, self.network(name))
        return self._tab[name]


# ------------------------------------------------------------------------------------------------ Supp. Fig. 10
def mondo_centrality(directory, sigma: float = 0.5, n_interp: int = 4, n_stages: int = 15) -> pd.DataFrame:
    """Selected Mondo centrality at every dense time point, with explicit zeros."""
    from drosophila_gnn.analysis.continuous_activity import _clean_exported_tf_names, gaussian_smooth_1d
    from drosophila_gnn.analysis.network import _dense_tftg_filename, _make_dense_labels

    _, labels, _, _ = _make_dense_labels(n_stages, n_interp)
    rows = []
    for label in labels:
        path = directory / _dense_tftg_filename(label)
        value = 0.0
        if path.exists():
            frame = pd.read_csv(path, sep="\t")
            value = float(frame.loc[_clean_exported_tf_names(frame["TF"]).str.casefold().eq("mondo"), "prob"].sum())
        rows.append({"time": round(float(label) - 1, 8), "centrality_raw": value})
    table = pd.DataFrame(rows)
    table["centrality_smooth"] = gaussian_smooth_1d(table.centrality_raw.to_numpy(), table.time.to_numpy(), sigma)
    return table


def scenic_mondo_targets() -> np.ndarray:
    t = pd.read_csv(paths.data("fly", "external", "scenicplus_published_eregulons", "germline_stage_eregulons.tsv.gz"),
                    sep="\t")
    return t[t.TF.str.casefold().eq("mondo")].groupby("time_idx").Gene.nunique().reindex(range(15), fill_value=0) \
        .to_numpy()


def supp10(p: Profiles) -> None:
    from scipy.interpolate import PchipInterpolator

    from drosophila_gnn.analysis.mondo_supplement import _interpolate

    runs = {"gru": dense("2025"), "no_gru": dense("nogru25"), "linear_gru": dense("lingru25")}
    cen = {v: mondo_centrality(d) for v, d in runs.items()}
    times = cen["gru"].time.to_numpy(float)
    expr = p.expression[:, p.aliases["Mondo"]].astype(float)
    counts = scenic_mondo_targets()
    grid = np.linspace(times[0], times[-1], 300)
    curves = pd.DataFrame({"time": grid})
    for v, c in cen.items():
        curves[v] = _interpolate(times, c.centrality_smooth.to_numpy(), grid)
    curves["mean_expr"] = _interpolate(np.arange(15.0), expr, grid)
    curves["scenic_target_count"] = PchipInterpolator(np.arange(15.0), counts.astype(float))(grid)
    anchors = {**{v: c.centrality_smooth for v, c in cen.items()}, "mean_expr": expr, "scenic_target_count": counts}
    for col, a in anchors.items():
        m = max(float(curves[col].max()), float(np.max(a)))
        curves[col] = curves[col] / m if m > 0 else 0.0
    save(K, "supp10_Supp_Fig_10_0.tsv", curves.rename(columns={
        "time": "stage_coordinate (0 = GSC, 14 = 2° Sc_6)", "gru": "DRAFT (GAT + GRU)", "no_gru": "DRAFT without GRU",
        "linear_gru": "Linear + GRU (GAT ablation)", "mean_expr": "Mean Mondo expression",
        "scenic_target_count": "SCENIC+ stage-filtered targets"}))
    stage = pd.DataFrame({"stage": numbering.germline_labels()})
    for v, label in (("gru", "DRAFT (GAT + GRU)"), ("no_gru", "DRAFT without GRU"), ("linear_gru", "Linear + GRU")):
        c = cen[v]
        stage[f"{label}: centrality_raw"] = c.loc[np.isclose(c.time, c.time.round()), "centrality_raw"].to_numpy()
    stage["mean_expression"] = expr
    stage["SCENIC+ stage-filtered targets"] = counts
    save(K, "supp10_Supp_Fig_10_1.tsv", stage)


# ------------------------------------------------------------------------------------------------ Supp. Fig. 11
def _near(value) -> list[int]:
    import json

    return json.loads(value) if isinstance(value, str) else list(value)


def random_pairing(cohort: pd.DataFrame) -> tuple[int, int, np.ndarray]:
    """cohort: TF_id, network_near, expression_near, offset of the evaluable TFs. Returns n, observed count and the
    count of every permutation (TFs with a separation >= 3 stages)."""
    cohort = cohort.sort_values("TF_id")
    net = [np.asarray(v, int) for v in cohort.network_near]
    exp = [np.asarray(v, int) for v in cohort.expression_near]
    d = np.asarray([[np.abs(a[:, None] - b[None, :]).min() for b in exp] for a in net])
    if not np.array_equal(np.diag(d), cohort.offset.to_numpy()):
        raise ValueError("offsets disagree with the near-maximum stage sets")
    rng = np.random.default_rng(PERMUTATION_SEED)
    n, idx = len(cohort), np.arange(len(cohort))
    draws = np.empty((PERMUTATIONS, 3), dtype=int)
    for i in range(PERMUTATIONS):
        paired = d[idx, rng.permutation(n)]
        draws[i] = [(paired >= t).sum() for t in (2, 3, 4)]
    return n, int((np.diag(d) >= FAR).sum()), draws[:, 1]


def supp11a(p: Profiles) -> tuple[pd.DataFrame, pd.DataFrame]:
    from matplotlib.cbook import boxplot_stats

    t = p.table("2025")
    ev = t[t.evaluable]
    cohorts = {"DRAFT": pd.DataFrame({"TF_id": ev.TF_id, "network_near": ev.network_near,
                                      "expression_near": ev.expression_near, "offset": ev.conservative_offset})}
    saved = pd.read_csv(paths.data("fly", "external", "tf_peak_candidate_table", "all_models_peak_details.tsv"),
                        sep="\t", keep_default_na=False)
    saved = saved[saved.profile.eq("smoothed") & saved.status.eq("evaluable")]
    for model in ("SCENIC+", "scDoRI"):
        s = saved[saved.model.eq(model)]
        cohorts[model] = pd.DataFrame({"TF_id": s.TF_id, "network_near": s.network_near_max_stages.map(_near),
                                       "expression_near": s.expression_near_max_stages.map(_near),
                                       "offset": pd.to_numeric(s.conservative_abs_offset).astype(int)})
    rows, freq = [], []
    for model, cohort in cohorts.items():
        n, observed, counts = random_pairing(cohort)
        b = boxplot_stats(counts, whis=(2.5, 97.5))[0]
        rows.append({"model": model, "n_evaluable": n, "observed_count": observed, "permutations": PERMUTATIONS,
                     **{k: b[k] for k in ("whislo", "q1", "med", "q3", "whishi")}})
        v, c = np.unique(counts, return_counts=True)
        freq.append(pd.DataFrame({"method": model, "TFs_with_peak_shift (>=3 stages)": v,
                                  "number_of_random_re-pairings": c}))
    return pd.DataFrame(rows), pd.concat(freq, ignore_index=True)


def supp11(p: Profiles) -> None:
    box, freq = supp11a(p)
    seeds = sorted(REAL)
    pairs_b = pd.DataFrame([{"seed_a": int(a), "seed_b": int(b), **p.sr.pair_stats(p.table(a), p.table(b))}
                            for a, b in itertools.combinations(seeds, 2)])
    keys = [f"real:{s}" for s in REAL] + [f"shuffled:{s}" for s in SHUFFLED]
    pairs_c = pd.DataFrame([{"a": a, "b": b, **p.sr.pair_stats(p.table(a.split(":")[1]), p.table(b.split(":")[1]))}
                            for a, b in itertools.combinations(keys, 2)])
    matrix = pd.DataFrame(index=list(ABLATION_MATRIX), columns=list(ABLATION_MATRIX), dtype=object)
    for a, b in itertools.combinations(ABLATION_MATRIX, 2):
        rho = round(p.sr.pair_stats(p.table(a), p.table(b))["rho_network_peak"], 2)
        matrix.loc[a, b] = matrix.loc[b, a] = rho
    matrix = matrix.rename_axis("model").reset_index()
    summary = agreement_summary(pairs_c, matrix)
    save(K, "supp11_Supp_Fig_11_0.tsv", box)
    save(K, "supp11_Supp_Fig_11_1.tsv", pairs_b)
    save(K, "supp11_Supp_Fig_11_2.tsv", summary)
    save(K, "supp11_Supp_Fig_11_3.tsv", pairs_c[["a", "b", "n_both", "p_far_b_given_far_a", "rho_network_peak"]])
    save(K, "supp11_Supp_Fig_11_4.tsv", matrix)
    save(K, "supp11_Supp_Fig_11_5.tsv", freq)


def _seed_of(name: str) -> int:
    import re

    digits = re.search(r"(\d+)$", name).group(1)
    return int(digits) if len(digits) == 4 else 2000 + int(digits)


def agreement_summary(pairs: pd.DataFrame, matrix: pd.DataFrame) -> pd.DataFrame:
    """Rho of every different-seed pair with full DRAFT, per group."""
    a, b = pairs.a.str.split(":").str[0], pairs.b.str.split(":").str[0]
    other = pd.Series(np.where(a == "real", pairs.b, pairs.a))
    one_real = (a == "real") ^ (b == "real")
    different = np.array([_seed_of(x) != _seed_of(y) for x, y in zip(pairs.a, pairs.b)])
    tfu = one_real & different & other.str.match(r"shuffled:tfu\d{4}$").to_numpy()
    m = matrix.set_index("model")

    def ablation(prefix):
        return np.array([m.loc[f, v] for f in ("2025", "2027", "2028") for v in m.columns
                         if v.startswith(prefix) and _seed_of(v) != _seed_of(f)], dtype=float)

    groups = [("Full DRAFT, another seed", pairs.rho_network_peak[(a == "real") & (b == "real")].to_numpy()),
              ("TF–RE shuffle", pairs.rho_network_peak[tfu].to_numpy()),
              ("DRAFT w/o GRU", ablation("nogru")), ("Linear + GRU", ablation("lingru"))]
    return pd.DataFrame([{"comparison": label, "pairs": len(v), "mean_rho": round(v.mean(), 3),
                          "min_rho": round(v.min(), 3), "max_rho": round(v.max(), 3)} for label, v in groups])


def build() -> None:
    p = Profiles()
    supp10(p)
    supp11(p)
