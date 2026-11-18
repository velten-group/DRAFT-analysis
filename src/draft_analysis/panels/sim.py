"""Fig. 2 and Supplementary Figs. 1-4 (simulation): panel tables from the scores of stage simulation.

Every fit is scored again from its saved predictions (draft_analysis.simulation); the tables below aggregate those
scores as the published figures did: per stage or per run, mean over the model seeds of each dataset, then mean and
sample SD across the datasets (Supp. Fig. 2: one dataset, SD across model seeds).

2b, 2e  candidate AUROC per stage (main design, datasets 101-103 x model seeds 0-4; Random: uniform scores, seeds 0-4)
2c, 2d  candidate AUROC and temporal smoothness against the prior SNR (2, 3, 6 and 8 cross-module candidates per TG,
        model seeds 0-2), the random-network reference and the SNR of the fly and mouse data priors
2f      temporal smoothness of every method on the main design
S1      stage-by-stage distances divided by each run's largest distance, averaged over runs
S2      random (non-modular) network, dataset 101, model seeds 0-2
S3      prior size (on the main design's fixed 1,800 candidates), temporal change and inactive candidates
S4      branching lineage: AUROC per state, smoothness over the parent-child pairs, t-SNE of the RNA of dataset 101
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import simulation as sim
from . import save

NAME = sim.NAME
FIVE = ["draft", "scenicplus", "scdori", "scmtni", "dictys"]
FIVE_PLOT = ["draft", "dictys", "scdori", "scmtni", "scenicplus"]  # order of the plotted-value tables
ORDER = [*sim.INTERNAL, *sim.OFFICIAL]  # order of the fits in the individual-fit tables
STRUCTURE = {2: (sim.MAIN, "outside_candidates_2"), 3: (sim.MAIN, "main"), 6: (sim.MAIN, "outside_candidates_6"),
             8: (sim.MAIN, "outside_candidates_8")}
ROUNDED = (2, 8)  # Fig. 2c,d shows these points with 4 decimals
# Fig. 2c,d mouse reference line: prior SNR of the union of the mouse per-cluster TG x TF priors (clusters 1-5, 7, 8;
# 11,887 TGs x 423 TFs, 1,636,409 edges), computed with sim.prior_snr. The mouse prior networks are not deposited,
# so the value is given here.
MOUSE_PRIOR_SNR = 1.6402352796796498


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(sim.out(name), sep="\t", float_precision="round_trip")


def _sd(s: pd.Series) -> float:
    return s.std(ddof=1)


def _select(df: pd.DataFrame, study: str, case: str, seeds=None) -> pd.DataFrame:
    part = df[df.study.eq(study) & df.case.eq(case)]
    return part if seeds is None else part[part.seed.isin(seeds)]


def _rank(df: pd.DataFrame, column: str, order: list[str]) -> pd.Series:
    return df[column].map({m: i for i, m in enumerate(order)})


def _by_data(df: pd.DataFrame, keys: list[str], metric: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mean over model seeds within each dataset, then mean and sample SD across datasets."""
    per_data = df.groupby([*keys, "data"])[metric].mean().reset_index()
    agg = per_data.groupby(keys)[metric].agg(["mean", _sd]).reset_index()
    agg.columns = [*keys, "mean", "sd"]
    return per_data, agg


# --------------------------------------------------------------------------------------------- Fig. 2
def fig2_stage_curves(per_time: pd.DataFrame) -> None:
    pt = _select(per_time, sim.MAIN, "main")
    for panel, methods in (("b", ["draft", "scenicplus", "scdori", "scmtni", "dictys", "random"]),
                           ("e", ["draft", "pooled_gat", "linear_gru", "pooled_linear"])):
        runs = pt[pt.method.isin(methods)]
        per_data, summ = _by_data(runs, ["method", "time"], "auroc")
        table = summ.assign(_m=_rank(summ, "method", methods)).sort_values(["_m", "time"])
        table = pd.DataFrame({"method": table.method.map(NAME), "stage": table.time,
                              "candidate_AUROC_mean": table["mean"], "SD_across_datasets": table.sd})
        wide = per_data.pivot_table(index=["method", "time"], columns="data", values="auroc").reset_index()
        wide = wide.assign(_m=_rank(wide, "method", methods)).sort_values(["_m", "time"]).drop(columns="_m")
        wide.columns = ["method", "stage"] + [f"dataset_{c}" for c in wide.columns[2:]]
        wide["method"] = wide.method.map(NAME)
        raw = pd.concat([runs[runs.method.ne("random")], runs[runs.method.eq("random")]])
        raw = pd.DataFrame({"method": raw.method.map(NAME), "dataset": raw.data, "model_seed": raw.seed,
                            "stage": raw.time, "candidate_AUROC": raw.auroc})
        for i, df in enumerate([table, wide, raw]):
            save("main", f"fig2_Fig_2{panel}_{i}.tsv", df.reset_index(drop=True))


def fig2_structure(runs: pd.DataFrame, refs: pd.DataFrame) -> None:
    rows, data_rows, fit_rows = [], [], []
    for outside, (study, case) in STRUCTURE.items():
        part = _select(runs, study, case, seeds=(0, 1, 2))
        part = part[part.method.isin(FIVE)]
        ref = _select(refs, study, case)
        snr, truth = ref.prior_snr.mean(), ref.truth_continuity.mean()
        per_data = part.groupby(["method", "data"])[["candidate_auroc", "mean_continuity"]].mean().reset_index()
        for metric, panel in (("candidate_auroc", "c"), ("mean_continuity", "d")):
            agg = per_data.groupby("method")[metric].agg(["mean", _sd])
            for m in FIVE:
                mean, sd = agg.loc[m, "mean"], agg.loc[m, "_sd"]
                rows.append({"panel": panel, "outside_candidates_per_TG": outside, "prior_SNR": snr, "method": NAME[m],
                             "metric": "candidate AUROC" if panel == "c" else "temporal smoothness",
                             "mean": mean, "SD_across_datasets": sd,
                             "plotted_mean": round(mean, 4) if outside in ROUNDED else mean,
                             "plotted_SD": round(sd, 4) if outside in ROUNDED else sd})
        rows.append({"panel": "d", "outside_candidates_per_TG": outside, "prior_SNR": snr, "method": "Ground truth",
                     "metric": "temporal smoothness", "mean": truth, "SD_across_datasets": np.nan,
                     "plotted_mean": truth, "plotted_SD": np.nan})
        for r in per_data.itertuples():
            data_rows.append({"outside_candidates_per_TG": outside, "prior_SNR": snr, "method": NAME[r.method],
                              "dataset": r.data, "candidate_AUROC": r.candidate_auroc,
                              "temporal_smoothness": r.mean_continuity})
        for r in part.itertuples():
            fit_rows.append({"outside_candidates_per_TG": outside, "method": NAME[r.method], "dataset": r.data,
                             "model_seed": r.seed, "candidate_AUROC": r.candidate_auroc,
                             "temporal_smoothness": r.mean_continuity})
    random = _select(runs, sim.RANDOM_NETWORK, "random_6true3false")
    ref_rows = []
    for panel, metric in (("c", "candidate_auroc"), ("d", "mean_continuity")):
        agg = random.groupby("method")[metric].agg(["mean", _sd])
        ref_rows += [{"panel": panel, "method": NAME[m], "position": "nominal x = 1 (not a measured SNR)",
                      "mean": agg.loc[m, "mean"], "SD_across_3_model_seeds_dataset_101": agg.loc[m, "_sd"]}
                     for m in FIVE_PLOT]
    # fly: recomputed from the deposited fly priors; mouse: the recorded value (MOUSE_PRIOR_SNR)
    fly = _read("real_prior_snr.tsv").set_index("prior").loc["fly", "snr"]
    lines = pd.DataFrame({"reference": ["Fly data prior", "Mouse data prior"], "prior_SNR": [fly, MOUSE_PRIOR_SNR]})
    for i, df in enumerate([pd.DataFrame(rows), pd.DataFrame(ref_rows), lines, pd.DataFrame(data_rows),
                            pd.DataFrame(fit_rows)]):
        save("main", f"fig2_Fig_2c-d_{i}.tsv", df)


BARS = ["ground_truth", "draft", "dictys", "scdori", "scmtni", "scenicplus", "random", "pooled_gat", "linear_gru",
        "pooled_linear"]


def main_smoothness(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fits and references of the main design (fits first, then ground truth and Random), and the summary per method."""
    part = _select(runs, sim.MAIN, "main")
    src = pd.concat([part[~part.method.isin(["ground_truth", "random"])], part[part.method.isin(["ground_truth", "random"])]])
    per_data = src.groupby(["method", "data"]).mean_continuity.mean().unstack("data")
    agg = pd.DataFrame({"mean": per_data.mean(1), "sd": per_data.std(1, ddof=1)}).loc[BARS]
    return src, per_data.loc[BARS], agg


def fig2_smoothness(runs: pd.DataFrame) -> None:
    src, per_data, agg = main_smoothness(runs)
    group = {"ground_truth": "reference", "pooled_gat": "internal ablation", "linear_gru": "internal ablation",
             "pooled_linear": "internal ablation"}
    bars = pd.DataFrame({"method": [NAME[m] for m in BARS], "group": [group.get(m, "benchmark") for m in BARS],
                         "temporal_smoothness_mean": agg["mean"].to_numpy(), "SD_across_datasets": agg.sd.to_numpy()})
    wide = per_data.reset_index()
    wide.columns = ["method"] + [f"dataset_{c}" for c in wide.columns[1:]]
    wide["method"] = wide.method.map(NAME)
    raw = pd.DataFrame({"method": src.method.map(NAME), "dataset": src.data,
                        "model_seed (-1 = ground truth)": src.seed, "temporal_smoothness": src.mean_continuity})
    for i, df in enumerate([bars, wide, raw]):
        save("main", f"fig2_Fig_2f_{i}.tsv", df.reset_index(drop=True))


# --------------------------------------------------------------------------------------------- Supp. Figs. 1-4
GRID = ["ground_truth", "draft", "pooled_gat", "linear_gru", "pooled_linear", "dictys", "scdori", "scmtni", "scenicplus"]


def supp1(runs: pd.DataFrame, dist: pd.DataFrame) -> None:
    _, _, smooth = main_smoothness(runs)
    maxima = dist.groupby(["method", "data", "seed"]).raw_distance.max().rename("max_distance").reset_index()
    _, mx = _by_data(maxima, ["method"], "max_distance")
    mx = mx.set_index("method")
    summary = pd.DataFrame({"method": [NAME[m] for m in GRID], "smoothness": smooth.loc[GRID, "mean"].to_numpy(),
                            "max_distance": mx.loc[GRID, "mean"].to_numpy(),
                            "max_distance_sd_between_data": mx.loc[GRID, "sd"].fillna(0.0).to_numpy()})
    save("supplementary", "supp01_Supp_Fig_1_0.tsv", summary)
    for i, m in enumerate(GRID, start=1):
        mean = dist[dist.method.eq(m)].groupby(["time_i", "time_j"]).distance.mean().unstack("time_j")
        mean.columns = [f"T{c}" for c in mean.columns]
        mean.index = [f"T{c}" for c in mean.index]
        save("supplementary", f"supp01_Supp_Fig_1_{i}.tsv", mean.rename_axis("stage").reset_index())


def supp2(runs: pd.DataFrame, per_time: pd.DataFrame) -> None:
    pt = _select(per_time, sim.RANDOM_NETWORK, "random_6true3false")
    pt = pt[pt.method.ne("ground_truth")]
    methods = [*ORDER, "random"]
    pms = pt.groupby(["method", "time"]).auroc.agg(["mean", _sd]).reset_index()
    bench, abl = ["draft", "scenicplus", "scmtni", "scdori", "dictys", "random"], list(sim.INTERNAL)
    for i, keep in enumerate([bench, abl]):
        part = pms[pms.method.isin(keep)]
        part = part.assign(_m=_rank(part, "method", keep)).sort_values(["_m", "time"])
        save("supplementary", f"supp02_Supp_Fig_2_{i}.tsv", pd.DataFrame({
            "method": part.method.map(NAME), "stage": part.time, "candidate_AUROC_mean": part["mean"],
            "SD_across_model_seeds": part._sd}))
    r = _select(runs, sim.RANDOM_NETWORK, "random_6true3false")
    rows = []
    for m in methods:
        g = r[r.method.eq(m)]
        rows.append({"method": NAME[m], "runs": f"{len(g)}/3", "mean_continuity": g.mean_continuity.mean(),
                     "mean_continuity_sd": _sd(g.mean_continuity), "candidate_auroc": g.candidate_auroc.mean(),
                     "candidate_auroc_sd": _sd(g.candidate_auroc)})
    gt = r[r.method.eq("ground_truth")].iloc[0]
    rows.append({"method": "Ground truth", "runs": "1/1", "mean_continuity": gt.mean_continuity,
                 "mean_continuity_sd": 0.0, "candidate_auroc": 1.0, "candidate_auroc_sd": 0.0})
    save("supplementary", "supp02_Supp_Fig_2_2.tsv", pd.DataFrame(rows))
    raw = pt.assign(_m=_rank(pt, "method", methods)).sort_values(["_m", "seed", "time"])
    save("supplementary", "supp02_Supp_Fig_2_3.tsv", pd.DataFrame({
        "method": raw.method.map(NAME), "dataset": raw.data, "model_seed": raw.seed, "stage": raw.time,
        "candidate_AUROC": raw.auroc}))


SENSITIVITY = [("a", 12.0, sim.PRIOR_SIZE, "base"), ("a", 16.0, sim.PRIOR_SIZE, "size_16"),
               ("a", 20.0, sim.PRIOR_SIZE, "size_20"), ("b", 5.0, sim.MAIN, "turnover_5"),
               ("b", 10.0, sim.MAIN, "main"), ("b", 12.0, sim.MAIN, "turnover_12"), ("c", 50.0, sim.MAIN, "main"),
               ("c", 62.5, sim.MAIN, "false_edges_62"), ("c", 70.0, sim.MAIN, "false_edges_70")]


def supp3(runs: pd.DataFrame) -> None:
    plotted, data_rows, fit_rows = [], [], []
    for panel, x, study, case in SENSITIVITY:
        part = _select(runs, study, case, seeds=(0, 1, 2))
        part = part[part.method.isin(FIVE)]
        if study == sim.PRIOR_SIZE:  # listed by dataset and model seed, then method
            part = part.assign(_m=_rank(part, "method", ["draft", *sim.OFFICIAL])).sort_values(["data", "seed", "_m"])
        for metric, label in (("candidate_auroc", "candidate AUROC"), ("mean_continuity", "temporal smoothness")):
            per_data, agg = _by_data(part, ["method"], metric)
            for r in agg.itertuples():
                rounded = panel == "a"  # panel a shows values rounded to 4 decimals
                plotted.append({"panel": panel, "setting": x, "metric": label, "method": r.method,
                                "plotted_mean": round(r.mean, 4) if rounded else r.mean,
                                "plotted_SD_across_datasets": round(r.sd, 4) if rounded else r.sd,
                                "runs": int(part.method.eq(r.method).sum())})
            data_rows += [{"panel": panel, "setting": x, "metric": label, "method": NAME[r.method], "dataset": r.data,
                           "value": getattr(r, metric)} for r in per_data.itertuples()]
        fit_rows += [{"panel": panel, "setting": x, "method": NAME[r.method], "dataset": r.data, "model_seed": r.seed,
                      "candidate_AUROC": r.candidate_auroc, "temporal_smoothness": r.mean_continuity}
                     for r in part.itertuples()]
    plotted = pd.DataFrame(plotted)
    plotted = plotted.assign(_p=plotted.panel, _k=plotted.metric.eq("temporal smoothness"),
                             _m=_rank(plotted, "method", FIVE_PLOT)).sort_values(["_p", "_k", "_m", "setting"],
                                                                                kind="stable")
    plotted = plotted.drop(columns=["_p", "_k", "_m"]).assign(method=lambda d: d.method.map(NAME))
    for i, df in enumerate([plotted, pd.DataFrame(data_rows), pd.DataFrame(fit_rows)]):
        save("supplementary", f"supp03_Supp_Fig_3_{i}.tsv", df.reset_index(drop=True))


BRANCH_ORDER = ["draft", "pooled_gat", "linear_gru", "pooled_linear", "scmtni"]  # row order of the published table


def supp4(runs: pd.DataFrame, per_time: pd.DataFrame, refs: pd.DataFrame) -> None:
    pst = _select(per_time, sim.BRANCH, "branch").assign(state=lambda d: d.time.map(dict(enumerate(sim.STATES))))
    _, agg = _by_data(pst, ["method", "state"], "auroc")
    brun = _select(runs, sim.BRANCH, "branch")
    _, cont = _by_data(brun, ["method"], "mean_continuity")
    ref = _select(refs, sim.BRANCH, "branch")
    references = pd.DataFrame({"data": ref.data, "truth_smoothness": ref.truth_continuity,
                               **{f"edges_differing_{a}_{b}": ref[f"edges_differing_{a}_{b}"].astype(int)
                                  for a, b in (("A2", "B2"), ("A3", "B3"), ("A4", "B4"))}})
    tsne = _read("tsne_branch_101.tsv")
    brun = brun.assign(_m=_rank(brun, "method", BRANCH_ORDER)).sort_values(["data", "_m", "seed"])
    columns = ["candidate_auroc", "trunk_auroc", "branch_states_auroc", "same_module_auroc", "true_vs_outside_auroc",
               "change_auroc", "branch_auroc", "mean_continuity"]
    fits = brun[["method", "data", "seed", *columns]].assign(method=lambda d: d.method.map(NAME))
    pst = pst.assign(_m=_rank(pst, "method", BRANCH_ORDER)).sort_values(["data", "_m", "seed", "time"])
    states = pst[["method", "data", "seed", "state", "auroc"]].assign(method=lambda d: d.method.map(NAME))
    for i, df in enumerate([agg.assign(method=agg.method.map(NAME)), cont.assign(method=cont.method.map(NAME)),
                            references, tsne, fits, states]):
        save("supplementary", f"supp04_Supp_Fig_4_{i}.tsv", df.reset_index(drop=True))


def build() -> None:
    runs, per_time = _read("runs.tsv"), _read("per_time.tsv")
    dist, refs = _read("distances.tsv"), _read("references.tsv")
    fig2_stage_curves(per_time)
    fig2_structure(runs, refs)
    fig2_smoothness(runs)
    supp1(runs, dist)
    supp2(runs, per_time)
    supp3(runs)
    supp4(runs, per_time, refs)
