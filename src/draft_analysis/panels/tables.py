"""Supplementary Tables recomputed by the checkpoint line, in the format of source_data/tables/ (published values).

2  rank of each Fig. 4g TF in the frozen edge clusters of its stage (243-TF list, distinct target genes; as Supp. Fig. 8)
4  TFs whose network peak is >= 3 stages from their RNA peak (seed 2025) and that are hubs of an edge cluster active in
   their peak window (window = network peak +-1 stage; rank
   <= 10 among all non-TE TFs of the cluster by distinct targets in the window, then mean edge score, then name;
   >= 50% of the cluster's edge observations in the window; >= 50 observations), restricted to the 243-TF list;
   direction and offset in the five germline models; SCENIC+ offsets from the deposited peak table
5  Mondo-target ranks of kuz and mamo in DRAFT and its two ablations (stage mondo_ranks)
6  network and expression peaks of selected TFs in the five germline models (argmax, signed offsets)
7  stage pairing of the two lineages used for the soma -> germline and germline -> soma scores
8  literature ligand-receptor pairs listed and evaluated (FlyPhoneDB and network genes)
9  literature pairs whose DRAFT score peaks early (>= 3 of 5 germline models; soma model for somatic receivers)
The TF lists of Tables 2 and 6 and the literature pairs of Tables 8 and 9 are inputs (curated); every other cell is
recomputed. Tables 1, 3, 10-13 are text, literature or settings and are not recomputed.
"""
from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

from .. import config, paths, source_data
from ..fly import numbering
from . import fig4, save

K = "tables"
SEEDS = ("2024", "2025", "2026", "2027", "2028")
STAGE_ECS = {"GSC/Gb": (0, 9), "Sp2-16": (5, 6, 9), "1°Sc": (4, 7, 11), "2°Sc": (1, 2, 8)}
DIRECTION = {"hub_to_germline": "Hub → germline", "hub_to_soma": "Hub → soma",
             "soma_to_germline": "Soma → germline", "germline_to_soma": "Germline → soma"}
SOMA_AXIS = ["CySC", "eSCC-1", "eSCC-2", "eSCC-3", "lSCC-1", "lSCC-2", "lSCC-3", "SdA/SdB"]
SOMA_GROUP = {"Soma_0 (CySC)": "CySC", "Soma_1 (eSCC)": "eSCC-1", "Soma_2 (eSCC)": "eSCC-2",
              "Soma_3 (eSCC)": "eSCC-3", "Soma_4 (lSCC)": "lSCC-1", "Soma_5 (lSCC)": "lSCC-2",
              "Soma_6 (lSCC)": "lSCC-3", "Soma_7a (SdA)": "SdA", "Soma_7b (SdB)": "SdB"}
CCC = "ccc_unified"  # the alignment with SdA and SdB as parallel terminal groups (Fig. 6, Supp. Tables 7-9)


def published(name: str) -> pd.DataFrame:
    return source_data.read_table(source_data.frozen_dir(K) / name)


def gene(symbol: str) -> str:
    return symbol[:1].upper() + symbol[1:]


# ------------------------------------------------------------------------------------------------ Table 2
def ec_ranks() -> pd.DataFrame:
    """Rank of every TF of the 243-TF list within each frozen edge cluster (distinct targets; Supp. Fig. 8 order)."""
    clusters = pd.read_csv(numbering.frozen("edge_cluster", "edge_clusters_12k_topk.csv"))
    cl = clusters[~clusters.tf_name.astype(str).str.contains(fig4.TE)]
    stats = cl.groupby(["edge_cluster", "tf_idx", "tf_name"]).tg_idx.nunique().reset_index(name="degree")
    keep = pd.read_csv(numbering.frozen("tf_list", "TF_tsv_FlyBase_plus_exceptions.tsv"), sep="\t", dtype=str)
    keep = keep.included.astype(str).eq("True").to_numpy()
    stats = stats[keep[stats.tf_idx]].sort_values(["edge_cluster", "degree"], ascending=[True, False])
    stats["rank"] = stats.groupby("edge_cluster").cumcount() + 1
    return stats


def table02() -> None:
    ranks = ec_ranks().set_index(["edge_cluster", "tf_name"])["rank"]
    ref = published("table02_ranks.tsv")
    col = "Rank in edge clusters (243 TFs)"
    out = ref[["Stage (Fig. 4g)", "TF"]].copy()
    out[col] = ["; ".join(f"EC{c} #{ranks[(c, tf)]}" for c in STAGE_ECS[stage]
                          if (c, tf) in ranks.index and ranks[(c, tf)] <= 10)
                for stage, tf in zip(out["Stage (Fig. 4g)"], out.TF)]
    save(K, "table02_ranks.tsv", out)


# ------------------------------------------------------------------------------------------------ Tables 4 and 6
def _stage(i) -> str:
    return numbering.germline_labels()[int(i)]


def _offsets():
    from .supp_peaks import Profiles

    p = Profiles()
    return p, {s: p.table(s).set_index("TF_id") for s in SEEDS}


def hub_screen(candidates: pd.DataFrame, radius: int = 1, top_n: int = 10, min_fraction: float = 0.5,
               min_observations: int = 50) -> pd.DataFrame:
    """Peak-window edge-cluster hubs among the offset candidates (columns TF_id, tf_idx, network_peak)."""
    edges = pd.read_csv(numbering.frozen("edge_cluster", "edges_selected_topk.csv"))
    clusters = pd.read_csv(numbering.frozen("edge_cluster", "edge_clusters_12k_topk.csv"))
    edges = edges.merge(clusters[["tf_idx", "tg_idx", "edge_cluster"]], on=["tf_idx", "tg_idx"], how="left",
                        validate="many_to_one")
    assert edges.edge_cluster.notna().all()
    totals = edges.groupby("edge_cluster").size()
    last = len(numbering.germline_labels()) - 1
    rows, cache = [], {}
    for c in candidates.itertuples():
        center = int(c.network_peak)
        lo, hi = max(0, center - radius), min(last, center + radius)
        if center not in cache:
            inside = edges[edges.cell_cluster.between(lo, hi)]
            frac = inside.groupby("edge_cluster").size().reindex(totals.index, fill_value=0) / totals
            r = inside[~inside.tf_name.astype(str).str.contains(fig4.TE)].groupby(
                ["edge_cluster", "tf_idx", "tf_name"]).agg(n=("tg_idx", "nunique"), s=("score", "mean")).reset_index()
            r = r.sort_values(["edge_cluster", "n", "s", "tf_name"], ascending=[True, False, False, True],
                              kind="stable")
            r["rank"] = r.groupby("edge_cluster").cumcount() + 1
            r["fraction"] = r.edge_cluster.map(frac)
            r["total"] = r.edge_cluster.map(totals)
            cache[center] = r
        m = cache[center]
        m = m[m.tf_idx.eq(c.tf_idx) & m["rank"].le(top_n) & m.fraction.ge(min_fraction) & m.total.ge(min_observations)]
        if len(m):
            best = m.sort_values(["rank", "fraction", "edge_cluster"], ascending=[True, False, True]).iloc[0]
            rows.append({"TF_id": c.TF_id, "window_start": lo, "window_end": hi,
                         "edge_cluster": int(best.edge_cluster), "rank": int(best["rank"])})
    return pd.DataFrame(rows)


def _scenic_offsets() -> dict:
    d = pd.read_csv(paths.data("fly", "external", "tf_peak_candidate_table", "all_models_peak_details.tsv"), sep="\t",
                    keep_default_na=False)
    d = d[d.model.eq("SCENIC+") & d.profile.eq("smoothed")]
    label = {"no_reported_targets": "n.a. (no targets)", "not_in_published_tf_collection": "n.a. (no eRegulon)",
             "expression_zero_profile": "n.a. (no expression)"}
    return {r.TF_id: (("Yes" if float(r.conservative_abs_offset) >= 3 else "No") if r.status == "evaluable"
                      else label[r.status]) for r in d.itertuples()}


def table04_06() -> None:
    _, t = _offsets()
    t25 = t["2025"]
    cand = t25[t25.evaluable & t25.far.astype(bool)].reset_index()
    tfs = pd.read_csv(paths.data("fly", "raw", "TF_filtered.tsv"), sep="\t", dtype=str, keep_default_na=False)
    alias = dict(zip(tfs.gene_name, tfs.FBgn_ID))
    edges = pd.read_csv(numbering.frozen("edge_cluster", "edge_clusters_12k_topk.csv"), usecols=["tf_idx", "tf_name"])
    ids = edges.drop_duplicates().assign(TF_id=lambda e: e.tf_name.map(
        lambda s: alias.get(re.sub(r"\s*\(\d+\)$", "", str(s).strip()))))
    cand = cand.merge(ids[["TF_id", "tf_idx"]].dropna(), on="TF_id", how="left")
    cand["tf_idx"] = cand.tf_idx.fillna(-1).astype(int)
    hubs = hub_screen(cand)
    tf243 = set(pd.read_csv(numbering.frozen("tf_list", "tf_list_flybase_FB2026_03.tsv"), sep="\t", dtype=str).FBgn_ID)
    print(f"   Table 4: {len(cand)} TFs with a >= 3-stage offset, {len(hubs)} peak-window hubs, "
          f"{int(hubs.TF_id.isin(tf243).sum())} in the 243-TF list", flush=True)
    hubs = hubs[hubs.TF_id.isin(tf243)].merge(cand, on="TF_id")
    scenic = _scenic_offsets()

    def sign(s, tf):
        r = t[s].loc[tf]
        return int(np.sign(int(r.network_peak) - int(r.expression_peak)))

    main, per_seed = [], []
    for h in hubs.itertuples():
        d25 = sign("2025", h.TF_id)
        same = sum(sign(s, h.TF_id) == d25 for s in SEEDS)
        far = sum(bool(t[s].loc[h.TF_id].far) for s in SEEDS)
        main.append({"TF": h.TF, "FlyBase ID": h.TF_id, "Network centrality peak": _stage(h.network_peak),
                     "RNA expression peak": _stage(h.expression_peak), "Peak offset (stages)": int(h.conservative_offset),
                     "Network peak relative to RNA peak": "Earlier" if d25 < 0 else "Later",
                     "Peak window": f"{_stage(h.window_start)} to {_stage(h.window_end)}",
                     "Edge cluster": f"EC{h.edge_cluster}", "Rank in edge cluster": h.rank,
                     "Seeds with the same direction (of 5)": f"{same}/5",
                     "Seeds with an offset of ≥3 stages (of 5)": f"{far}/5",
                     "Offset also found with SCENIC+": scenic.get(h.TF_id, "n.a. (no eRegulon)"),
                     "_order": (int(h.network_peak), h.edge_cluster, h.rank)})
        row = {"TF": h.TF, "RNA expression peak": _stage(h.expression_peak)}
        for s in SEEDS:
            r, d = t[s].loc[h.TF_id], sign(s, h.TF_id)
            side = "earlier" if d < 0 else "later" if d > 0 else "same stage"
            row[f"Seed {s}"] = f"{_stage(r.network_peak)} ({side}{', ≥3' if bool(r.far) else ''})"
        row["Seeds with the same direction (of 5)"] = f"{same}/5"
        row["Seeds with an offset of ≥3 stages (of 5)"] = f"{far}/5"
        per_seed.append(row)
    order = sorted(range(len(main)), key=lambda i: main[i]["_order"])
    for m in main:
        m.pop("_order")
    save(K, "table04_offsets.tsv", pd.DataFrame([main[i] for i in order]))
    save(K, "table04_per_seed.tsv", pd.DataFrame([per_seed[i] for i in order]))

    by_name = {s: t[s].reset_index().set_index("TF") for s in SEEDS}
    rows = []
    for tf in published("table06_seed_offsets.tsv").TF:
        e = int(by_name["2025"].loc[tf].expression_peak)
        row, signs = {"TF": tf, "Expression peak": _stage(e)}, []
        for s in SEEDS:
            r = by_name[s].loc[tf]
            assert int(r.expression_peak) == e
            d = int(r.network_peak) - e
            signs.append(d)
            row[f"Seed {s}"] = f"{_stage(r.network_peak)} ({'+' if d > 0 else '−' if d < 0 else ''}{abs(d)})"
        row["Earlier / later seeds"] = f"{sum(d < 0 for d in signs)} / {sum(d > 0 for d in signs)}"
        rows.append(row)
    save(K, "table06_seed_offsets.tsv", pd.DataFrame(rows))


# ------------------------------------------------------------------------------------------------ Table 5
def table05() -> None:
    """DRAFT row: consensus rank of the Fig. 5c eRegulon (as published); ablation rows: fixed-universe stage ranks
    (stage mondo_ranks)."""
    fig5c = pd.read_csv(paths.PANELS["main"] / "fig5_Fig_5c_0.tsv", sep="\t").set_index("TG").consensus_rank
    ranks = pd.read_csv(paths.FLY / "mondo_ranks" / "ranks.tsv", sep="\t")
    rows = [{"Method": "DRAFT", "kuz rank": f"{int(fig5c['kuz']):,}", "mamo rank": f"{int(fig5c['mamo']):,}"}]
    for model in ("DRAFT w/o GRU", "Linear + GRU"):
        r = ranks[ranks.model.eq(model)].set_index("TG").aggregate_rank
        rows.append({"Method": model, "kuz rank": f"{int(r['kuz']):,}", "mamo rank": f"{int(r['mamo']):,}"})
    save(K, "table05_mondo_target_ranks.tsv", pd.DataFrame(rows))


# ------------------------------------------------------------------------------------------------ Table 7
def _weights(pos: float, axis: list[str]) -> tuple[str, str]:
    lo = math.floor(pos + 1e-9)
    frac = pos - lo
    if frac < 1e-9:
        mix = f"{axis[lo - 1]} (100%)"
    else:
        mix = f"{axis[lo - 1]} ({round(100 * (1 - frac))}%), {axis[lo]} ({round(100 * frac)}%)"
    return mix, axis[math.floor(pos + 0.5) - 1]


def table07() -> None:
    a = pd.read_csv(paths.FLY / CCC / "receptor_input_ungated_gate0.05_lig0.05" / "alignment_pairing.tsv", sep="\t")
    germ = numbering.germline_labels()
    g = a[a.axis.eq("germline stage")]
    rows = []
    for r in g.itertuples():
        mix, near = _weights(r.aligned_position, SOMA_AXIS)
        rows.append({"Germline stage": r.stage, "Aligned position on the somatic axis (1–8)": f"{r.aligned_position:.2f}",
                     "Somatic sender expression taken from": mix, "Nearest somatic time point": near})
    save(K, "table07_germline_receivers.tsv", pd.DataFrame(rows))
    rows = []
    for r in a[a.axis.eq("soma group")].itertuples():
        mix, near = _weights(r.aligned_position, germ)
        rows.append({"Somatic group": SOMA_GROUP[r.stage],
                     "Aligned position on the germline axis (1–15)": f"{r.aligned_position:.2f}",
                     "Germline sender expression taken from": mix, "Nearest germline stage": near})
    save(K, "table07_somatic_receivers.tsv", pd.DataFrame(rows))


# ------------------------------------------------------------------------------------------------ Tables 8 and 9
def _literature() -> pd.DataFrame:
    lit = pd.read_csv(config.CONFIGS / "fly" / "ccc_literature_pairs.tsv", sep="\t")
    ev = pd.read_csv(paths.FLY / CCC / "receptor_input_ungated_gate0.05_lig0.05" / "literature.tsv", sep="\t")
    primary = set(zip(ev.direction[ev.readout.eq("primary")], ev.pair[ev.readout.eq("primary")]))
    why = dict(zip(zip(ev.direction, ev.pair), ev.status))
    lit["pair"] = lit.ligand + "-" + lit.receptor
    lit["evaluated"] = [(d, p) in primary for d, p in zip(lit.direction, lit.pair)]
    lit["reason"] = [why.get((d, p), "") if not e else "" for d, p, e in zip(lit.direction, lit.pair, lit.evaluated)]
    return lit


def table08() -> None:
    reasons = {"not in flyphonedb": "Not in FlyPhoneDB", "gene not in the TG list": "Not in the network gene set"}
    rows = []
    for (d, pw), g in _literature().groupby(["direction", "pathway"], sort=False):
        rows.append({"Direction": DIRECTION[d], "Pathway": pw, "Pairs listed": len(g),
                     "Pairs evaluated": int(g.evaluated.sum()),
                     "Not evaluated because": "; ".join(dict.fromkeys(reasons[r] for r in g.reason if r))})
    save(K, "table08_evaluated.tsv", pd.DataFrame(rows))


def table09() -> None:
    calls = pd.read_csv(paths.FLY / CCC / "peak_shifts_gate0.05_lig0.05" / "calls.tsv", sep="\t", dtype={"seed": str})
    calls["ligand"], calls["receptor"] = calls.ligand.str.lower(), calls.receptor.str.lower()  # names as in FlyPhoneDB
    calls = calls.set_index(["direction", "ligand", "receptor", "readout", "seed"])

    def stage(label):
        return SOMA_GROUP.get(label, label)

    rows = []
    for r in _literature().itertuples():
        if not r.evaluated:
            continue
        key = (r.direction, r.ligand, r.receptor)
        expr = calls.loc[key + ("expression", "all")]
        if r.direction.endswith("_to_germline"):
            draft = [calls.loc[key + ("DRAFT_A", s)] for s in SEEDS]
            n = sum(c.status == "early" for c in draft)
            if n < 3:
                continue
            models, peak = f"{n}/5", calls.loc[key + ("DRAFT_A", "2025")].peak_stage
            fig6d = "Yes" if n == 5 and expr.status == "late" else ""
        else:
            draft = calls.loc[key + ("DRAFT_A", "soma")]
            if draft.status != "early":
                continue
            models, peak, fig6d = "1/1 (soma model)", draft.peak_stage, ""
        rows.append({"Direction": DIRECTION[r.direction], "Pathway": r.pathway,
                     "Ligand–receptor": f"{gene(r.ligand)}–{gene(r.receptor)}",
                     "DRAFT models with an early peak": models,
                     "DRAFT peak stage (seed-2025 model; soma model)": stage(peak),
                     "Also early with the LR expression product": "Yes" if expr.status == "early" else "No",
                     "LR expression product peak stage": stage(expr.peak_stage), "Shown in Fig. 6d": fig6d})
    out = pd.DataFrame(rows)
    order = list(DIRECTION.values())
    out = out.assign(_d=out.Direction.map(order.index)).sort_values(["_d", "Ligand–receptor"]).drop(columns="_d")
    save(K, "table09_draft_early_pairs.tsv", out.reset_index(drop=True))


def build() -> None:
    table02()
    table04_06()
    table05()
    table07()
    table08()
    table09()
