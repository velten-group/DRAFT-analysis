"""AUROC of the literature evaluation of LR timing.

For each direction and method, the earliness score (early maximum - late maximum) / curve maximum of the literature
pairs (positives) is compared with that of candidate pairs without literature support. Those pairs are unlabelled,
not known negatives; using them as negatives pulls the AUROC toward 0.5 without changing the order of methods. The
reported AUROC is the mean over balanced draws: each draw samples as many unlabelled pairs as there are positives,
the same pairs for every method. The AUROC against all unlabelled pairs and a 95% interval from a bootstrap over
receptors (pairs sharing a receptor are not independent) are written alongside. Pairs a method cannot score get 0.

    python scripts/ccc_effect_sizes.py [--run-dir result/ccc_unified/receptor_input_ungated_gate0.05_lig0.05]
"""

import argparse

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description="AUROC of the literature evaluation of LR timing")
parser.add_argument("--run-dir", default="result/ccc_unified/receptor_input_ungated_gate0.05_lig0.05")
parser.add_argument("--draws", type=int, default=1000, help="balanced draws of unlabelled pairs")
parser.add_argument("--bootstrap", type=int, default=2000, help="bootstrap replicates over receptors")
args = parser.parse_args()
RUN = args.run_dir
LAST_EARLY = {"germline": 5, "soma": 3}
METHODS = {"primary": "DRAFT", "CellChat": "CellChat", "LIANA": "LIANA+", "expression": "LR expression"}
ORDER = ["hub_to_germline", "hub_to_soma", "soma_to_germline", "germline_to_soma"]
rng = np.random.default_rng(2025)

lit = pd.read_csv("configs/fly/ccc_literature_pairs.tsv", sep="\t")
lit_keys = {(d, a.lower(), b.lower()) for d, a, b in zip(lit.direction, lit.ligand, lit.receptor)}
pairs = pd.read_csv(f"{RUN}/pairs.tsv", sep="\t")
curves = pd.read_csv(f"{RUN}/curves.tsv", sep="\t")


def scores(direction: str) -> pd.DataFrame:
    """Earliness score of every candidate pair of a direction, one column per method."""
    recv = "germline" if direction.endswith("germline") else "soma"
    split = LAST_EARLY[recv] + 1
    table = pairs[(pairs.direction == direction) & (pairs.readout == "primary")][["ligand", "receptor"]].reset_index(drop=True)
    for readout, name in METHODS.items():
        c = curves[(curves.direction == direction) & (curves.readout == readout)]
        values = {}
        for (lig, rec), g in c.groupby(["ligand", "receptor"]):
            s = g.sort_values("stage_index").score.to_numpy(dtype=float)
            values[(lig, rec)] = 0.0 if s.max() <= 0 else (s[:split].max() - s[split:].max()) / s.max()
        table[name] = [values.get((a, b), 0.0) for a, b in zip(table.ligand, table.receptor)]
        status = pairs[(pairs.direction == direction) & (pairs.readout == readout)].set_index(["ligand", "receptor"]).status
        table[f"{name}_early"] = [status.get((a, b)) == "early" for a, b in zip(table.ligand, table.receptor)]
    table["lit"] = [(direction, a.lower(), b.lower()) in lit_keys for a, b in zip(table.ligand, table.receptor)]
    return table


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    return ((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg))


rows = []
for direction in ORDER:
    t = scores(direction)
    pos_idx, neg_idx = np.flatnonzero(t.lit), np.flatnonzero(~t.lit)
    draws = [rng.choice(neg_idx, size=min(len(pos_idx), len(neg_idx)), replace=False) for _ in range(args.draws)] if len(pos_idx) > 1 else []
    receptors = t.receptor.unique()
    by_rec = {r: np.flatnonzero(t.receptor.to_numpy() == r) for r in receptors}
    boots = [np.concatenate([by_rec[r] for r in rng.choice(receptors, size=len(receptors), replace=True)]) for _ in range(args.bootstrap)]
    for name in METHODS.values():
        s = t[name].to_numpy()
        record = {"direction": direction, "method": name, "found": int(t.loc[t.lit, f"{name}_early"].sum()), "literature": len(pos_idx)}
        if draws:
            balanced = np.array([auroc(s[pos_idx], s[d]) for d in draws])
            full = []
            for idx in boots:
                lab = t.lit.to_numpy()[idx]
                if lab.any() and (~lab).any():
                    full.append(auroc(s[idx][lab], s[idx][~lab]))
            record.update({"auroc_balanced": balanced.mean(), "balanced_draw_lo": np.percentile(balanced, 2.5), "balanced_draw_hi": np.percentile(balanced, 97.5),
                           "auroc_all_unlabelled": auroc(s[pos_idx], s[neg_idx]),
                           "receptor_boot_lo": np.percentile(full, 2.5), "receptor_boot_hi": np.percentile(full, 97.5)})
        rows.append(record)
out = pd.DataFrame(rows)
out.to_csv(f"{RUN}/auroc.tsv", sep="\t", index=False)

pd.set_option("display.width", 250)
print(out.round(2).to_string(index=False))
print(f"wrote {RUN}/auroc.tsv")
