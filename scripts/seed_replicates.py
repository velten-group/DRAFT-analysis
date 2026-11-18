"""TF network and expression peaks in each germline model, and their offsets (Fig. 5a, Supp. Fig. 11, Supp. Tables 4, 6).

Reads the dense TF-TG exports (TF_TG_dense_t*.tsv) and the RNA stage means. A TF's network profile is the sum of its
selected outgoing TF-TG probabilities per stage, smoothed with sigma 0.5; a TF is far when its conservative peak offset
is >= 3 stages (near-max 0.95, profile range >= 0.1). pair_stats compares two models: one-sided Fisher test of far x
far over the TFs evaluable in both, and the Spearman correlation of the network peak stages.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, spearmanr

SIGMA, NEAR_MAX, MIN_RANGE, FAR = 0.5, 0.95, 0.1, 3


def norm_id(value):
    return re.sub(r"#s?0*([0-9]+)$", r"_sample\1", str(value).strip(), flags=re.IGNORECASE)


def smooth(values):
    """gaussian_smooth_1d for stages 0..14, edge padded."""
    radius = max(1, int(round(3 * SIGMA)))
    offsets = np.arange(-radius, radius + 1)
    weights = np.exp(-0.5 * (offsets / SIGMA) ** 2)
    weights /= weights.sum()
    return np.convolve(np.pad(np.asarray(values, float), (radius, radius), mode="edge"), weights, mode="valid")


def peak(values):
    values = np.asarray(values, float)
    if not np.isfinite(values).all() or (values < 0).any():
        return {"status": "missing_or_invalid_profile"}
    maximum = float(values.max())
    if maximum <= 0:
        return {"status": "zero_profile"}
    span = (maximum - values.min()) / maximum
    return {
        "status": "evaluable" if span >= MIN_RANGE else "near_flat_profile",
        "peak": int(np.argmax(values)),
        "near": np.flatnonzero(values >= maximum * NEAR_MAX).tolist(),
    }


def load_tfs(path):
    table = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    table = table[["gene_name", "FBgn_ID"]].apply(lambda column: column.str.strip())
    aliases = {}
    for i, row in enumerate(table.itertuples(index=False)):
        for value in row:
            aliases[value] = i
    return table.reset_index(drop=True), aliases


def expression_profiles(root, tfs):
    import anndata as ad

    anno = pd.read_csv(root / "cell_annotation.csv", dtype=str)
    rna = ad.read_h5ad(root / "A_rna_focus.h5ad", backed="r")
    atac = ad.read_h5ad(root / "B_atac_focus.h5ad", backed="r")
    try:
        rna_ids = pd.Index([norm_id(v) for v in rna.obs_names])
        atac_ids = pd.Index([norm_id(v) for v in atac.obs_names])
        anno["cell_norm"] = anno.iloc[:, 0].map(norm_id)
        cells = anno.loc[
            anno.GEX_latent_time_bins_annotated.str.match(r"^Germline_\d+$", na=False)
            & anno.cell_norm.isin(rna_ids.intersection(atac_ids))
        ].copy()
        cells["stage"] = cells.GEX_latent_time_bins_annotated.str.extract(r"(\d+)$")[0].astype(int)
        if cells.cell_norm.duplicated().any() or set(cells.stage) != set(range(15)):
            raise ValueError("Paired germline cells must map uniquely to all fifteen stages.")
        present = tfs.FBgn_ID.isin(rna.var_names).to_numpy()
        selected = rna.layers["lognorm"][rna_ids.get_indexer(cells.cell_norm)][
            :, rna.var_names.get_indexer(tfs.loc[present, "FBgn_ID"])
        ]
        values = np.full((15, len(tfs)), np.nan)
        for stage in range(15):
            part = selected[cells.stage.eq(stage).to_numpy()]
            values[stage, present] = np.asarray(part.mean(axis=0)).ravel()
        return values
    finally:
        rna.file.close()
        atac.file.close()


def network_profiles(directory, tfs, aliases):
    """Sum of selected outgoing TF-TG probabilities at the 15 original stages."""
    labeled = sorted(
        (float(re.fullmatch(r"TF_TG_dense_t(.+)\.tsv", p.name)[1]), p) for p in directory.glob("TF_TG_dense_t*.tsv")
    )
    if not labeled:
        raise FileNotFoundError(f"No dense export in {directory}.")
    limits = np.asarray([labeled[0][0], labeled[-1][0]])
    offset = -1 if np.allclose(limits, [1, 15]) else 0 if np.allclose(limits, [0, 14]) else None
    if offset is None:
        raise ValueError("Dense times must cover 0-14 or 1-15.")
    by_time = {round(t + offset, 6): p for t, p in labeled}
    profile = np.zeros((15, len(tfs)))
    for stage in range(15):
        frame = pd.read_csv(by_time[float(stage)], sep="\t", usecols=["TF", "TG", "prob"])
        if not np.isfinite(frame.prob).all() or not frame.prob.between(0, 1).all() or frame.duplicated(["TF", "TG"]).any():
            raise ValueError(f"Invalid dense scores at stage {stage} in {directory}.")
        index = frame.TF.astype(str).str.replace(r"^tf:", "", regex=True, case=False).str.strip().map(aliases)
        valid = index.notna()
        profile[stage] = np.bincount(index[valid].astype(int), weights=frame.loc[valid, "prob"], minlength=len(tfs))
    return profile


def offsets_table(tfs, expression, network):
    rows = []
    for i, tf in enumerate(tfs.itertuples(index=False)):
        e, n = peak(expression[:, i]), peak(smooth(network[:, i]))
        row = {"TF_id": tf.FBgn_ID, "TF": tf.gene_name, "evaluable": e["status"] == "evaluable" and n["status"] == "evaluable"}
        if row["evaluable"]:
            gap = np.abs(np.asarray(n["near"])[:, None] - np.asarray(e["near"])[None, :]).min()
            row.update(expression_peak=e["peak"], network_peak=n["peak"], conservative_offset=int(gap), far=bool(gap >= FAR),
                       network_near=n["near"], expression_near=e["near"])
        rows.append(row)
    return pd.DataFrame(rows)


def pair_stats(a, b):
    both = a.merge(b, on="TF_id", suffixes=("_a", "_b"))
    both = both[both.evaluable_a & both.evaluable_b]
    fa, fb = both.far_a.to_numpy(bool), both.far_b.to_numpy(bool)
    table = [[int((fa & fb).sum()), int((fa & ~fb).sum())], [int((~fa & fb).sum()), int((~fa & ~fb).sum())]]
    odds, p = fisher_exact(table, alternative="greater")
    return {
        "n_both": len(both), "far_a": int(fa.sum()), "far_b": int(fb.sum()), "far_both": table[0][0],
        "p_far_b_given_far_a": table[0][0] / max(fa.sum(), 1), "p_far_b_given_not_far_a": table[1][0] / max((~fa).sum(), 1),
        "odds_ratio": float(odds), "fisher_p_one_sided": float(p),
        "rho_network_peak": float(spearmanr(both.network_peak_a, both.network_peak_b).statistic),
        "rho_offset": float(spearmanr(both.conservative_offset_a, both.conservative_offset_b).statistic),
    }
