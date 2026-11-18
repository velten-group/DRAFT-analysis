"""Fig. 6 panel tables (cell-cell communication timing).

b  alignment check with an anatomical anchor: eya in cyst cells, kmg/aly/can in germ cells (stage means of the paired
   RNA/ATAC cells), the eya switch mapped onto germline stages by the frozen DTW path and by a linear map
c  AUROC of the earliness score for literature LR pairs (scripts/ccc_effect_sizes.py on the timing run)
d  strips of the literature pairs placed early by DRAFT and late by the LR expression product in all five germline models
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config, paths
from ..fly import ccc
from . import save

K = "main"
GERM_GENES = {"kmg": "on", "aly": "on", "can": "on"}
LATE_SPERMATOGONIA = "Sp-8"
METHODS = ["DRAFT", "CellChat", "LIANA+", "LR expression"]
C_DIRS = [("hub_to_germline", "Hub → germline", "germline"), ("hub_to_soma", "Hub → soma", "soma"),
          ("soma_to_germline", "Soma → germline", "germline")]
# (direction, ligands drawn as one row, receptor): DRAFT early and expression late in 5 of 5 germline models
ROWS = [("hub_to_germline", ["upd1", "upd2", "upd3"], "dome"), ("soma_to_germline", ["Wnt4"], "arr"),
        ("soma_to_germline", ["Wnt4"], "fz"), ("soma_to_germline", ["Wnt4"], "fz2"),
        ("soma_to_germline", ["Wnt4"], "fz3"), ("soma_to_germline", ["Wnt4"], "fz4"),
        ("soma_to_germline", ["gbb"], "sax")]
GROUP = {"hub_to_germline": "Hub → germline", "soma_to_germline": "Soma → germline"}
DISPLAY = {"upd1": "Upd1", "upd2": "Upd2", "upd3": "Upd3", "dpp": "Dpp", "gbb": "Gbb", "wg": "Wg", "dome": "Dome",
           "tkv": "Tkv", "arr": "Arr", "fz": "Fz", "fz2": "Fz2", "fz3": "Fz3", "fz4": "Fz4", "sax": "Sax"}


def _unified():
    sys.path.insert(0, str(ccc.SCRIPTS))
    import run_ccc_unified

    return run_ccc_unified


def scaled(profile: np.ndarray, direction: str) -> np.ndarray:
    lo, hi = profile.min(), profile.max()
    s = (profile - lo) / (hi - lo) if hi > lo else np.zeros_like(profile)
    return 1 - s if direction == "off" else s


def stretch(curve: np.ndarray) -> np.ndarray:
    """Min-max stretch of a max-scaled curve; gated stages (score 0) become NaN."""
    kept = curve > 0
    low = curve[kept].min()
    return np.where(kept, (curve - low) / (1 - low), np.nan)


def panel_b() -> None:
    import anndata as ad
    import scipy.sparse as sp

    u = _unified()
    cfg = config.load("ccc_unified")
    germ_labels = json.loads(Path(cfg["germ_stage_labels"]).read_text(encoding="utf-8"))["labels"]
    soma_meta = json.loads(Path(cfg["soma_stage_labels"]).read_text(encoding="utf-8"))
    soma_groups = soma_meta["groups"]
    nodes, _ = u.soma_time_points(cfg, soma_groups, soma_groups)
    germ_groups = [f"Germline_{i}" for i in range(len(germ_labels))]
    tg = pd.read_csv(cfg["data"]["tg_list"], sep="\t", dtype=str)
    sym2fb = dict(zip(tg["gene_name"], tg["FBgn_ID"]))
    genes = sorted(set(GERM_GENES) | {"eya"})
    rna = ad.read_h5ad(cfg["data"]["rna_h5ad"])
    atac_cells = {u.norm_id(c) for c in ad.read_h5ad(cfg["data"]["atac_h5ad"], backed="r").obs_names}
    rna_map = {u.norm_id(c): c for c in rna.obs_names}
    anno = pd.read_csv(cfg["data"]["annotation"], dtype=str)
    anno["norm"] = anno.iloc[:, 0].map(u.norm_id)
    anno = anno[anno["norm"].isin(atac_cells) & anno["norm"].isin(rna_map.keys())]
    sub = rna[anno["norm"].map(rna_map).to_numpy(), [sym2fb[g] for g in genes]]
    m = sub.layers["lognorm"]
    matrix = pd.DataFrame(m.toarray() if sp.issparse(m) else np.asarray(m), columns=genes)
    cell_labels = anno["GEX_latent_time_bins_annotated"].to_numpy()

    def stage_means(gene, groups):
        return np.array([matrix[gene].to_numpy()[cell_labels == g].mean() for g in groups])

    germ_stage = np.mean([scaled(stage_means(g, germ_groups), d) for g, d in GERM_GENES.items()], axis=0)
    soma_raw = u.to_soma_nodes(stage_means("eya", soma_groups), soma_groups, nodes)
    eya_nodes = scaled(soma_raw, "on")
    soma_label = dict(zip(soma_groups, soma_meta["labels"]))
    seen, node_names = {}, []
    for node in nodes:
        if len(node) > 1:
            node_names.append(str(cfg.get("soma_branch_label", "/".join(node))))
            continue
        lab = soma_label[node[0]]
        seen[lab] = seen.get(lab, 0) + 1
        node_names.append(lab if list(soma_label.values()).count(lab) == 1 else f"{lab}-{seen[lab]}")
    p = pd.read_csv(config.CONFIGS / "fly" / "ccc_frozen_dtw_soma_to_germ_parallel.tsv", sep="\t")
    n_germ, n_nodes = len(germ_labels), len(nodes)
    germ_at = {"DTW": lambda x: np.interp(x, p.t_soma_net, p.t_germ_warped),
               "Linear": lambda x: 1 + (np.asarray(x, dtype=float) - 1) * (n_germ - 1) / (n_nodes - 1)}
    k = int(np.argmax(np.diff(eya_nodes)))
    g_k = int(np.argmax(np.diff(germ_stage)))

    b_germ = pd.DataFrame({"germline_stage": germ_labels})
    for gene in GERM_GENES:
        b_germ[f"{gene}_mean_lognorm"] = stage_means(gene, germ_groups)
    b_germ["spermatocyte_program_scaled (plotted)"] = germ_stage
    b_soma = pd.DataFrame({"soma_time_point": node_names, "eya_mean_lognorm": soma_raw, "eya_scaled (plotted)": eya_nodes})
    events = [{"event": "eya switch in cyst cells (largest increase)", "between": f"{node_names[k]} and {node_names[k + 1]}",
               "germline_coordinate (1-based stage centres)": np.nan},
              {"event": "germ-cell switch (largest increase of kmg/aly/can)",
               "between": f"{germ_labels[g_k]} and {germ_labels[g_k + 1]}",
               "germline_coordinate (1-based stage centres)": g_k + 1.5},
              {"event": "expected eya-switch window starts (late spermatogonia)", "between": "",
               "germline_coordinate (1-based stage centres)": germ_labels.index(LATE_SPERMATOGONIA) + 0.5}]
    for key in ("DTW", "Linear"):
        events.append({"event": f"eya switch mapped onto germline stages by {key}", "between": "",
                       "germline_coordinate (1-based stage centres)": float(germ_at[key](k + 1.5))})
    save(K, "fig6_Fig_6b_0.tsv", b_germ)
    save(K, "fig6_Fig_6b_1.tsv", b_soma)
    save(K, "fig6_Fig_6b_2.tsv", pd.DataFrame(events))


def panel_c() -> None:
    u = _unified()
    run = paths.FLY / "ccc_unified" / ccc.RUN_NAME
    auroc = pd.read_csv(run / "auroc.tsv", sep="\t").set_index(["direction", "method"])
    soma_groups = json.loads((config.CONFIGS / "fly" / "soma_stage_labels.json").read_text(encoding="utf-8"))["groups"]
    n_stages = {"germline": 15, "soma": len(soma_groups)}
    rows = []
    for direction, title, receiver in C_DIRS:
        for method in METHODS:
            r = auroc.loc[(direction, method)]
            n = int(r.literature)
            rows.append({"direction": title, "method": method, "AUROC": r.auroc_balanced,
                         "literature_pairs_placed_early": int(r.found), "literature_pairs": n,
                         "expected_by_chance": n * (u.LAST_EARLY[receiver] + 1) / n_stages[receiver]})
    save(K, "fig6_Fig_6c_0.tsv", pd.DataFrame(rows))


def panel_d() -> None:
    u = _unified()
    labels = json.loads((config.CONFIGS / "fly" / "germline_stage_labels.json").read_text(encoding="utf-8"))["labels"]
    run = paths.FLY / "ccc_unified" / ccc.RUN_NAME
    curves = pd.read_csv(run / "curves.tsv", sep="\t")
    pairs = pd.read_csv(run / "pairs.tsv", sep="\t")
    calls = pd.read_csv(paths.FLY / "ccc_unified" / "peak_shifts_gate0.05_lig0.05" / "calls.tsv", sep="\t")
    draft_calls = calls[calls.readout.eq("DRAFT_A")]

    def pick(df, direction, ligand, receptor, readout):
        return df[(df.direction == direction) & (df.ligand == ligand) & (df.receptor == receptor) & (df.readout == readout)]

    rows, peaks = [], []
    for direction, ligands, receptor in ROWS:
        cur = {}
        for readout in ("primary", "expression"):
            per_ligand = [pick(curves, direction, lig, receptor, readout).sort_values("stage_index").score.to_numpy(float)
                          for lig in ligands]
            cur[readout] = [c / c.max() for c in per_ligand]
            assert all(np.allclose(c, cur[readout][0]) for c in cur[readout]), (direction, ligands, receptor)
        for lig in ligands:
            assert pick(pairs, direction, lig, receptor, "primary").status.iloc[0] == "early", (lig, receptor)
            assert pick(pairs, direction, lig, receptor, "expression").status.iloc[0] == "late", (lig, receptor)
            # selection rule of the panel: early in all five germline models (seeds 2024-2028)
            seeds = draft_calls[(draft_calls.direction == direction) & (draft_calls.ligand == lig)
                                & (draft_calls.receptor == receptor)]
            assert len(seeds) == 5 and seeds.status.eq("early").all(), (lig, receptor, seeds.status.tolist())
        name = DISPLAY.get(ligands[0], ligands[0])
        if len(ligands) > 1:
            name += "/" + "/".join(x[-1] for x in ligands[1:])
        label = f"{name}–{DISPLAY.get(receptor, receptor)}"
        draft, expr = stretch(cur["primary"][0]), stretch(cur["expression"][0])
        for i, st in enumerate(labels):
            rows.append({"direction": GROUP[direction], "pair": label, "stage": st,
                         "DRAFT_scaled (top strip; empty = receptor in <5% of cells)": draft[i],
                         "LR_expression_scaled (bottom strip)": expr[i]})
        peaks.append({"direction": GROUP[direction], "pair": label, "DRAFT_peak_stage": labels[int(np.nanargmax(draft))],
                      "LR_expression_peak_stage": labels[int(np.nanargmax(expr))]})
    last_early = u.LAST_EARLY["germline"]
    pk = pd.DataFrame(peaks)
    assert all(labels.index(p) <= last_early for p in pk.DRAFT_peak_stage)
    assert all(labels.index(p) > last_early for p in pk.LR_expression_peak_stage)
    save(K, "fig6_Fig_6d_0.tsv", pd.DataFrame(rows))
    save(K, "fig6_Fig_6d_1.tsv", pk)


def build() -> None:
    panel_b()
    panel_c()
    panel_d()
