"""Score ligand-receptor timing in four directions with the unified rule and evaluate the literature pairs.

Rule: ligand = mean expression in the sender cells, receptor = its regulatory input
in the receiver's full-score network (scripts/export_ccc_full_scores.py), LR score on the receiver's integer stages.
Pairs are FlyPhoneDB rows matched to the TG list by exact symbol; genes are then handled by FBgn.

    python scripts/run_ccc_unified.py --config configs/fly/ccc_unified.yaml

Writes to output_dir: pairs.tsv (stage call of every pair, direction and readout), curves.tsv, literature.tsv,
evaluation.tsv (literature pairs against the base rate of each direction), alignment_pairing.tsv, summary.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import yaml
from scipy.stats import false_discovery_control, fisher_exact

from drosophila_gnn.checkpoint import save_checkpoint_metadata
from drosophila_gnn.preprocess import norm_id

ROOT = Path(__file__).resolve().parents[1]
DIRECTIONS = {
    "hub_to_germline": ("hub", "germline"),
    "hub_to_soma": ("hub", "soma"),
    "soma_to_germline": ("soma", "germline"),
    "germline_to_soma": ("germline", "soma"),
}
LAST_EARLY = {"germline": 5, "soma": 3}  # Sp-16; eSCC-3 (Soma_3)
READOUTS = ["primary", "expression", "sens_50k_indegree", "sens_smoothed"]
MEAN_PERCENTILE_FILE = "receptor_input.tsv"


def path(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def soma_time_points(cfg: dict, soma_groups: list[str], soma_labels: list[str]) -> tuple[list[list[str]], list[str]]:
    """Soma time points of the alignment: one per soma group, except the parallel terminal branches listed in
    soma_parallel_branches (SdA and SdB), which share one time point named soma_branch_label."""
    branches = list(cfg.get("soma_parallel_branches") or [])
    nodes: list[list[str]] = []
    names: list[str] = []
    for group, label in zip(soma_groups, soma_labels):
        if group not in branches:
            nodes.append([group])
            names.append(label)
        elif group == next(g for g in soma_groups if g in branches):
            nodes.append(branches)
            names.append(str(cfg.get("soma_branch_label", "/".join(branches))))
    return nodes, names


def soma_alignment(cfg: dict, soma_groups: list[str], soma_labels: list[str], n_germ: int, file: str | None = None) -> dict:
    """Soma-germline alignment on the soma time points.

    soma_at_germ: soma time-point coordinate (1-based) of each germline stage; germ_at_soma: germline coordinate of each
    soma group (parallel branches share their time point); nodes/names: the soma time points.
    """
    nodes, names = soma_time_points(cfg, soma_groups, soma_labels)
    p = pd.read_csv(path(file or cfg["alignment"]), sep="\t")
    if not np.isclose(p["t_soma_net"].max(), len(nodes)):
        raise SystemExit(f"alignment {file or cfg['alignment']} spans {p['t_soma_net'].max():g} soma time points, the config defines "
                         f"{len(nodes)} (soma_parallel_branches: {cfg.get('soma_parallel_branches')})")
    group_node = np.array([next(i for i, node in enumerate(nodes) if g in node) + 1 for g in soma_groups], dtype=float)
    return {"nodes": nodes, "names": names, "group_node": group_node,
            "soma_at_germ": np.interp(np.arange(1, n_germ + 1), p["t_germ_warped"], p["t_soma_net"]),
            "germ_at_soma": np.interp(group_node, p["t_soma_net"], p["t_germ_warped"])}


def to_soma_nodes(values: np.ndarray, soma_groups: list[str], nodes: list[list[str]]) -> np.ndarray:
    """Per-soma-group values (first axis) on the soma time points: the mean over a time point's groups."""
    index = {g: i for i, g in enumerate(soma_groups)}
    values = np.asarray(values, dtype=float)
    return np.stack([values[[index[g] for g in node]].mean(axis=0) for node in nodes])


def group_expression(cfg: dict, fbgns: list[str], groups: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Mean lognorm expression and expressing fraction per group over the paired RNA/ATAC cells, as the models use."""
    rna = ad.read_h5ad(path(cfg["data"]["rna_h5ad"]))
    atac_cells = {norm_id(c) for c in ad.read_h5ad(path(cfg["data"]["atac_h5ad"]), backed="r").obs_names}
    rna_map = {norm_id(c): c for c in rna.obs_names}
    anno = pd.read_csv(path(cfg["data"]["annotation"]), dtype=str)
    anno["norm"] = anno.iloc[:, 0].map(norm_id)
    anno = anno[anno["norm"].isin(atac_cells) & anno["norm"].isin(rna_map.keys())]
    anno["cell"] = anno["norm"].map(rna_map)
    genes = [g for g in fbgns if g in rna.var_names]
    sub = rna[anno["cell"].to_numpy(), genes]
    matrix = sub.layers["lognorm"]
    matrix = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
    labels = anno["GEX_latent_time_bins_annotated"].to_numpy()
    mean = pd.DataFrame({g: matrix[labels == g].mean(axis=0) for g in groups}, index=genes)
    frac = pd.DataFrame({g: (matrix[labels == g] > 0).mean(axis=0) for g in groups}, index=genes)
    counts = {g: int((labels == g).sum()) for g in groups}
    return mean, frac, counts


def full_scores(directory: Path, receptor_file: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    receptor = pd.read_csv(directory / receptor_file, sep="\t")
    active = pd.read_csv(directory / "tg_active.tsv", sep="\t")
    stages = [c for c in receptor.columns if c[1:].isdigit() and c.startswith("t")]
    order = receptor["tg_id"].astype(str).tolist()
    return receptor.set_index("tg_id")[stages], active.set_index("tg_id")[stages], order


def native_scores(spec: dict, all_groups: list[str]) -> dict[str, pd.DataFrame]:
    """Sender x receiver group scores of an expression-based method (official CellChat or LIANA+ output) per LR key."""
    cols = [spec["sender"], spec["receiver"], spec["key"], spec["score"]]
    edges = pd.read_csv(path(spec["edges"]), sep="\t", usecols=cols)
    return {
        key: part.pivot_table(index=spec["sender"], columns=spec["receiver"], values=spec["score"], aggfunc="max")
        .reindex(index=all_groups, columns=all_groups).fillna(0.0)
        for key, part in edges.groupby(spec["key"])
    }


def published_indegree(directory: Path, order: list[str], n_stage: int) -> pd.DataFrame:
    values = np.zeros((len(order), n_stage))
    for t in range(n_stage):
        edges = pd.read_csv(directory / f"TF_TG_dense_t{t + 1}.tsv", sep="\t", usecols=["TG_idx", "prob"])
        sums = edges.groupby("TG_idx")["prob"].sum()
        values[sums.index.to_numpy(), t] = sums.to_numpy()
    return pd.DataFrame(values, index=order)


def smooth(curve: np.ndarray, sigma: float) -> np.ndarray:
    x = np.arange(len(curve), dtype=float)
    kernel = np.exp(-0.5 * ((x[:, None] - x[None, :]) / sigma) ** 2)
    return (kernel / kernel.sum(axis=1, keepdims=True)) @ curve


def call(curve: np.ndarray, receiver: str, labels: list[str], min_range: float) -> dict:
    curve = np.asarray(curve, dtype=float)
    if not np.isfinite(curve).all() or curve.max() <= 0:
        return {"status": "unscored"}
    dynamic = float((curve.max() - curve.min()) / curve.max())
    peak = int(np.argmax(curve))
    out = {"dynamic_range": dynamic, "peak_index": peak, "peak_stage": labels[peak], "total": float(curve.sum())}
    if dynamic < min_range:
        out["status"] = "untimed"
    else:
        out["status"] = "early" if peak <= LAST_EARLY[receiver] else "late"
    return out


def frozen_original(curves_path: Path, labels: list[str], min_range: float) -> dict[str, dict]:
    if not curves_path.exists():
        return {}
    curves = pd.read_csv(curves_path, sep="\t")
    curves = curves[curves["method"] == "DRAFT_program"]
    calls = {}
    for key, part in curves.groupby("interaction_name"):
        score = part["score"].to_numpy(dtype=float)
        position = part["germline_index"].to_numpy(dtype=float)
        if not np.isfinite(score).all() or score.max() <= 0:
            calls[key] = {"status": "unscored"}
            continue
        dynamic = float((score.max() - score.min()) / score.max())
        peak = float(position[int(np.argmax(score))])
        status = "untimed" if dynamic < min_range else ("early" if peak < LAST_EARLY["germline"] + 0.5 else "late")
        calls[key] = {"status": status, "dynamic_range": dynamic, "peak_index": peak, "peak_stage": labels[int(round(peak))]}
    return calls


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default="configs/fly/ccc_unified.yaml")
    parser.add_argument("--receptor-readout", help="receptor readout file of the full-score export (default: receptor_readout in the config)")
    parser.add_argument("--receptor-gate", type=float, help="receptor expressing-cell fraction below which a receiver stage scores 0 for every "
                        "readout (default: the export's active set, 11%%); use with receptor_input_ungated.tsv")
    args = parser.parse_args()
    cfg = yaml.safe_load(path(args.config).read_text(encoding="utf-8"))
    out_dir = path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    min_range = float(cfg["dynamic_range_min"])
    min_frac = float(cfg["ligand_min_frac"])

    labels = {
        "germline": json.loads(path(cfg["germ_stage_labels"]).read_text(encoding="utf-8"))["labels"],
        "soma": json.loads(path(cfg["soma_stage_labels"]).read_text(encoding="utf-8"))["labels"],
    }
    soma_groups = json.loads(path(cfg["soma_stage_labels"]).read_text(encoding="utf-8"))["groups"]
    labels["soma"] = [f"{group} ({label})" for group, label in zip(soma_groups, labels["soma"])]
    groups = {"hub": [cfg["hub_group"]], "soma": soma_groups, "germline": [f"Germline_{i}" for i in range(len(labels["germline"]))]}

    tg = pd.read_csv(path(cfg["data"]["tg_list"]), sep="\t", dtype=str)
    sym2fb = dict(zip(tg["gene_name"], tg["FBgn_ID"]))
    lr = pd.read_csv(path(cfg["data"]["ccc_csv"]), dtype=str)
    lr = lr[lr["resource"].str.contains(cfg["lr_resource"], case=False, na=False)]
    lr = lr[["source_genesymbol", "target_genesymbol"]].drop_duplicates().rename(columns={"source_genesymbol": "ligand", "target_genesymbol": "receptor"})
    resource_keys = {(a.lower(), b.lower()) for a, b in zip(lr.ligand, lr.receptor)}
    universe = lr[lr.ligand.isin(sym2fb) & lr.receptor.isin(sym2fb)].reset_index(drop=True)
    universe["l_fb"], universe["r_fb"] = universe.ligand.map(sym2fb), universe.receptor.map(sym2fb)
    print(f"{cfg['lr_resource']} pairs: {len(lr)}; both genes in the TG list: {len(universe)}")

    receptor_file = args.receptor_readout or cfg.get("receptor_readout", MEAN_PERCENTILE_FILE)
    # The receptor gate of the config applies to the config's readout; an explicit --receptor-readout uses the export's
    # 11% active set unless --receptor-gate is given.
    receptor_gate = args.receptor_gate if args.receptor_gate is not None else (None if args.receptor_readout else cfg.get("receptor_gate"))
    out_dir = out_dir / (Path(receptor_file).stem + (f"_gate{receptor_gate:g}" if receptor_gate is not None else "")
                         + (f"_lig{min_frac:g}" if abs(min_frac - 0.10) > 1e-9 else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    scores = {side: full_scores(path(cfg["full_scores"][side]), receptor_file) for side in ("germline", "soma")}
    readouts = list(READOUTS)
    if receptor_file != MEAN_PERCENTILE_FILE:
        mean_percentile = {side: full_scores(path(cfg["full_scores"][side]), MEAN_PERCENTILE_FILE)[0] for side in scores}
        readouts.append("sens_mean_percentile")
    print(f"primary receptor readout: {receptor_file}; outputs in {out_dir}")
    indegree = {side: published_indegree(path(cfg["published_networks"][side]), scores[side][2], scores[side][0].shape[1]) for side in scores}
    all_groups = groups["hub"] + groups["soma"] + groups["germline"]
    fbgns = sorted(set(universe.l_fb) | set(universe.r_fb))
    mean, frac, counts = group_expression(cfg, fbgns, all_groups)
    print("paired cells:", counts)

    # Soma senders enter on the soma time points (parallel branches as their mean) at their aligned position.
    align = soma_alignment(cfg, soma_groups, labels["soma"], len(groups["germline"]))
    soma_at_germ, germ_at_soma, soma_nodes = align["soma_at_germ"], align["germ_at_soma"], align["nodes"]
    pairing = pd.concat([
        pd.DataFrame({"axis": "germline stage", "stage": labels["germline"], "aligned_position": soma_at_germ,
                      "aligned_group": [align["names"][int(round(x)) - 1] for x in soma_at_germ]}),
        pd.DataFrame({"axis": "soma group", "stage": labels["soma"], "aligned_position": germ_at_soma,
                      "aligned_group": [labels["germline"][int(round(x)) - 1] for x in germ_at_soma]}),
    ])
    pairing.to_csv(out_dir / "alignment_pairing.tsv", sep="\t", index=False)

    def sender_values(values: np.ndarray, sender: str) -> np.ndarray:
        return to_soma_nodes(values, groups["soma"], soma_nodes) if sender == "soma" else values

    def ligand_term(fb: str, sender: str, receiver: str) -> np.ndarray:
        n = len(groups[receiver])
        if sender == "hub":
            return np.full(n, float(mean.loc[fb, groups["hub"][0]]))
        values = sender_values(mean.loc[fb, groups[sender]].to_numpy(dtype=float), sender)
        where = soma_at_germ if sender == "soma" else germ_at_soma
        return np.interp(where, np.arange(1, len(values) + 1), values)

    original = frozen_original(path(cfg["frozen_original_curves"]), labels["germline"], min_range)
    baselines = {name: native_scores(spec, all_groups) for name, spec in cfg.get("baselines", {}).items() if path(spec["edges"]).exists()}
    readouts += ["expression_ungated", *baselines]
    print("expression-based baselines:", ", ".join(baselines) or "none")

    def baseline_term(table: pd.DataFrame, sender: str, receiver: str) -> np.ndarray:
        """Native group-level score on the receiver's stages; trajectory senders interpolated along the alignment."""
        receivers = groups[receiver]
        if sender == "hub":
            return table.loc[groups["hub"][0], receivers].to_numpy(dtype=float)
        values = sender_values(table.loc[groups[sender], receivers].to_numpy(dtype=float), sender)
        where = soma_at_germ if sender == "soma" else germ_at_soma
        x = np.arange(1, values.shape[0] + 1)
        return np.array([np.interp(where[j], x, values[:, j]) for j in range(len(receivers))])
    rows, curve_rows = [], []
    for direction, (sender, receiver) in DIRECTIONS.items():
        primary, active, _ = scores[receiver]
        if receptor_gate is not None:
            # Receptor gate from the expressing fraction of the receiver groups, the same for every readout.
            active = (frac[groups[receiver]] >= receptor_gate).astype(float).set_axis(active.columns, axis=1)
        for item in universe.itertuples(index=False):
            base = {"direction": direction, "ligand": item.ligand, "receptor": item.receptor}
            if item.l_fb not in frac.index or frac.loc[item.l_fb, groups[sender]].max() < min_frac:
                rows.append({**base, "readout": "all", "status": "ligand below the sender fraction"})
                continue
            if item.r_fb not in active.index or active.loc[item.r_fb].max() < 1:
                rows.append({**base, "readout": "all", "status": "receptor never active"})
                continue
            ligand = ligand_term(item.l_fb, sender, receiver)
            receptor = {
                "primary": primary.loc[item.r_fb].to_numpy(dtype=float) * (active.loc[item.r_fb].to_numpy(dtype=float) if receptor_gate is not None else 1.0),
                "expression": mean.loc[item.r_fb, groups[receiver]].to_numpy(dtype=float),
                "sens_50k_indegree": indegree[receiver].loc[item.r_fb].to_numpy(dtype=float),
            }
            if "sens_mean_percentile" in readouts:
                receptor["sens_mean_percentile"] = mean_percentile[receiver].loc[item.r_fb].to_numpy(dtype=float)
            curves = {name: ligand * value for name, value in receptor.items()}
            curves["sens_smoothed"] = smooth(curves["primary"], float(cfg["smoothing_sigma"]))
            # Same receptor gate for every comparison: a stage where the receptor is below the active fraction scores 0.
            gate = active.loc[item.r_fb].to_numpy(dtype=float)
            curves["expression_ungated"] = curves["expression"]
            curves["expression"] = curves["expression"] * gate
            key = f"{item.ligand.lower()}__{item.receptor.lower()}"
            for name, native in baselines.items():
                curves[name] = baseline_term(native[key], sender, receiver) * gate if key in native else None
            for name in readouts:
                if curves[name] is None:
                    rows.append({**base, "readout": name, "status": f"not in {name}"})
                    continue
                rows.append({**base, "readout": name, **call(curves[name], receiver, labels[receiver], min_range)})
                curve_rows.append(pd.DataFrame({**base, "readout": name, "stage_index": range(len(ligand)), "stage": labels[receiver], "score": curves[name]}))
            if direction == "soma_to_germline":
                rows.append({**base, "readout": "sens_original", **original.get(key, {"status": "not in frozen analysis"})})
    pairs = pd.DataFrame(rows)
    pairs.to_csv(out_dir / "pairs.tsv", sep="\t", index=False)
    pd.concat(curve_rows, ignore_index=True).to_csv(out_dir / "curves.tsv", sep="\t", index=False)

    lit = pd.read_csv(path(cfg["literature_pairs"]), sep="\t", dtype=str)
    upper = {(a.lower(), b.lower()): (a, b) for a, b in zip(universe.ligand, universe.receptor)}
    lit_rows = []
    for row in lit.itertuples(index=False):
        key = (row.ligand.lower(), row.receptor.lower())
        base = {"direction": row.direction, "pathway": row.pathway, "pair": f"{row.ligand}-{row.receptor}", "strict_stage": row.strict_stage}
        if key not in resource_keys:
            lit_rows.append({**base, "readout": "all", "status": f"not in {cfg['lr_resource']}"})
            continue
        if key not in upper:
            lit_rows.append({**base, "readout": "all", "status": "gene not in the TG list"})
            continue
        ligand, receptor = upper[key]
        hits = pairs[(pairs.direction == row.direction) & (pairs.ligand == ligand) & (pairs.receptor == receptor)]
        for record in hits.to_dict("records"):
            lit_rows.append({**base, "readout": record["readout"], "status": record["status"], "peak_stage": record.get("peak_stage"),
                             "dynamic_range": record.get("dynamic_range")})
    literature = pd.DataFrame(lit_rows)
    literature.to_csv(out_dir / "literature.tsv", sep="\t", index=False)

    lit_keys = {(r.direction, r.ligand.lower(), r.receptor.lower()) for r in lit.itertuples(index=False)}
    pairs["is_literature"] = [(d, a.lower(), b.lower()) in lit_keys for d, a, b in zip(pairs.direction, pairs.ligand, pairs.receptor)]
    evaluation = []
    for direction, (sender, receiver) in DIRECTIONS.items():
        part = pairs[(pairs.direction == direction) & (pairs.readout != "all")]
        for readout in part.readout.unique():
            sub = part[part.readout == readout]
            rec = {"direction": direction, "readout": readout}
            for group, mask in (("lit", sub.is_literature), ("other", ~sub.is_literature)):
                g = sub[mask]
                timed = g[g.status.isin(["early", "late"])]
                rec.update({f"{group}_n": len(g), f"{group}_scored": int((g.status != "unscored").sum()), f"{group}_timed": len(timed),
                            f"{group}_early": int((timed.status == "early").sum())})
                if direction.startswith("hub"):
                    rec[f"{group}_contact_stage"] = int((timed.peak_index == 0).sum())
            table = [[rec["lit_early"], rec["lit_timed"] - rec["lit_early"]], [rec["other_early"], rec["other_timed"] - rec["other_early"]]]
            rec["fisher_p_lit_more_early"] = float(fisher_exact(table, alternative="greater")[1]) if rec["lit_timed"] and rec["other_timed"] else None
            evaluation.append(rec)
    evaluation = pd.DataFrame(evaluation)
    # Benjamini-Hochberg over every direction x readout test.
    tested = evaluation["fisher_p_lit_more_early"].notna()
    evaluation.loc[tested, "bh_q"] = false_discovery_control(evaluation.loc[tested, "fisher_p_lit_more_early"].astype(float), method="bh")
    evaluation.to_csv(out_dir / "evaluation.tsv", sep="\t", index=False)

    save_checkpoint_metadata(out_dir / "summary.json", {
        "config": cfg, "paired_cells": counts, "n_resource_pairs": len(lr), "n_universe": len(universe),
        "candidates_per_direction": {d: int(((pairs.direction == d) & (pairs.readout == "primary")).sum()) for d in DIRECTIONS},
    })
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 400)
    print("\nCandidates per direction (pass ligand and receptor rules):",
          {d: int(((pairs.direction == d) & (pairs.readout == "primary")).sum()) for d in DIRECTIONS})
    cols = ["direction", "readout", "lit_n", "lit_timed", "lit_early", "other_timed", "other_early", "fisher_p_lit_more_early", "bh_q"]
    cols += [c for c in ("lit_contact_stage", "other_contact_stage") if c in evaluation.columns]
    print(evaluation[cols].to_string(index=False))
    wide = literature[literature.readout.isin(["all", "primary", "expression", "CellChat", "LIANA", "sens_original"])]
    wide = wide.assign(call=wide.status + wide.peak_stage.fillna("").map(lambda s: f" ({s})" if s else ""))
    wide = wide.pivot_table(index=["direction", "pathway", "pair"], columns="readout", values="call", aggfunc="first").reset_index()
    print(wide.to_string(index=False))


if __name__ == "__main__":
    main()
