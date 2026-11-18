"""Prepare identical candidates, cells, and frozen GRN scores for comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmwrite

from drosophila_gnn.analysis import ccc
from drosophila_gnn.analysis.ccc_alignment_case import _activity, sha256
from drosophila_gnn.preprocess import norm_id
from drosophila_gnn.utils import load_yaml_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ccc_regulatory_benchmark.yaml")
    args = parser.parse_args()
    cfg = load_yaml_config(args.config)
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    settings = ccc._ccc_settings(load_yaml_config(cfg["ccc_config"]))
    soma = _activity(
        Path(settings["network_dir_soma"]),
        Path(cfg["tg_annotation"]),
        settings["sigma_time"],
    )
    germ = _activity(
        Path(settings["network_dir_germ"]),
        Path(cfg["tg_annotation"]),
        settings["sigma_time"],
    )
    candidates = ccc.load_lr_candidates(settings["ccc_csv"], soma, germ)
    candidates["interaction_name"] = candidates.ligand + "__" + candidates.receptor
    warp = pd.read_csv(Path(cfg["reference_dir"]) / settings["warp_dtw_tsv"], sep="\t")
    sg = np.interp(soma.columns.to_numpy(float), warp.t_soma_net, warp.t_germ_warped)
    aligned = ccc.interp_df_over_time(germ, germ.columns.to_numpy(float), sg)
    aligned.columns = soma.columns
    programs = ccc.compute_receptor_programs(
        aligned,
        sorted(candidates.receptor.unique()),
        settings["corr_thresh"],
        settings["top_k_prog"],
    )
    members = [
        {"receptor": r, "rank": i + 1, "gene": g}
        for r, genes in programs.items()
        for i, g in enumerate(genes)
    ]
    pd.DataFrame(members).to_csv(
        out / "fixed_program_members.tsv", sep="\t", index=False
    )
    candidates["has_program"] = candidates.receptor.isin(programs)
    rna = ad.read_h5ad(cfg["rna"], backed="r")
    atac = ad.read_h5ad(cfg["atac"], backed="r")
    atac_ids = set(map(norm_id, atac.obs_names))
    keep = np.array([norm_id(c) in atac_ids for c in rna.obs_names])
    annot = pd.read_csv(cfg["tg_annotation"], sep="\t").dropna(
        subset=["gene_name", "FBgn_ID"]
    )
    annot["gene"] = annot.gene_name.str.lower()
    grouped_ids = annot.groupby("gene").FBgn_ID.agg(lambda x: sorted(set(x)))
    ambiguous = set(grouped_ids[grouped_ids.map(len).gt(1)].index)
    mapping = {g: ids[0] for g, ids in grouped_ids.items() if len(ids) == 1}
    annot[annot.gene.isin(ambiguous)].to_csv(
        out / "casefold_ambiguous_gene_ids.tsv", sep="\t", index=False
    )
    all_genes = sorted(
        set(candidates.ligand)
        | set(candidates.receptor)
        | {g for genes in programs.values() for g in genes}
    )
    genes = [g for g in all_genes if mapping.get(g) in rna.var_names]
    for role in ("ligand", "receptor"):
        candidates[f"rna_{role}_available"] = candidates[role].isin(genes)
    candidates["ambiguous_gene_identity"] = candidates.ligand.isin(
        ambiguous
    ) | candidates.receptor.isin(ambiguous)
    candidates.to_csv(out / "candidate_coverage.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "gene": all_genes,
            "fbgn": [";".join(grouped_ids.get(g, [])) for g in all_genes],
            "ambiguous_casefold": [g in ambiguous for g in all_genes],
            "rna_available": [g in genes for g in all_genes],
        }
    ).to_csv(out / "gene_mapping.tsv", sep="\t", index=False)
    mat = rna.layers["lognorm"][
        :, rna.var_names.get_indexer([mapping[g] for g in genes])
    ][keep]
    mat = sparse.csr_matrix(mat)
    assert np.isfinite(mat.data).all() and (mat.data >= 0).all()
    obs = rna.obs.loc[keep, ["cell_type", "sample"]].copy()
    obs.insert(0, "cell_id", [norm_id(x) for x in obs.index])
    assert obs.cell_id.is_unique
    obs.to_csv(out / "cells.tsv", sep="\t", index=False)
    mmwrite(out / "expression_genes_by_cells.mtx", mat.T)
    pd.Series(genes).to_csv(out / "genes.tsv", sep="\t", index=False, header=False)
    native = pd.DataFrame(mat.toarray(), index=obs.index, columns=genes)
    native.insert(0, "cell_type", obs.cell_type.astype(str).to_numpy())
    native.insert(1, "sample", obs["sample"].astype(str).to_numpy())
    native.groupby("cell_type")[genes].mean().to_csv(
        out / "mean_expression.tsv", sep="\t"
    )
    native.groupby(["cell_type", "sample"])[genes].mean().to_csv(
        out / "sample_mean_expression.tsv", sep="\t"
    )
    obs.groupby(["cell_type", "sample"], observed=True).size().rename("n_cells").to_csv(
        out / "cell_counts.tsv", sep="\t"
    )
    soma.to_csv(out / "soma_centrality.tsv", sep="\t")
    germ.to_csv(out / "germ_centrality.tsv", sep="\t")
    pd.DataFrame({"soma_time": soma.columns.to_numpy(float), "germ_time": sg}).to_csv(
        out / "fixed_mapping.tsv", sep="\t", index=False
    )
    ligands = sorted(set(candidates.ligand) | set(candidates.receptor))
    db = candidates.loc[
        candidates.rna_ligand_available & candidates.rna_receptor_available,
        ["interaction_name", "ligand", "receptor"],
    ].copy()
    for column in ["agonist", "antagonist", "co_A_receptor", "co_I_receptor"]:
        db[column] = ""
    db["pathway_name"] = "unassigned"
    db["annotation"] = "unassigned"
    db.to_csv(out / "cellchat_fly_interactions.tsv", sep="\t", index=False)
    pd.Series([g for g in ligands if g in genes]).to_csv(
        out / "signaling_genes.tsv", header=False, index=False
    )
    files = [
        cfg["rna"],
        cfg["atac"],
        cfg["tg_annotation"],
        settings["ccc_csv"],
        args.config,
        str(Path(cfg["reference_dir"]) / settings["warp_dtw_tsv"]),
    ]
    manifest = {
        "config": cfg,
        "cells": len(obs),
        "candidate_pairs": len(candidates),
        "rna_covered_pairs": len(db),
        "receptor_programs": len(programs),
        "rna_genes": len(genes),
        "ambiguous_casefold_gene_groups": len(ambiguous),
        "ambiguous_lr_pairs": int(candidates.ambiguous_gene_identity.sum()),
        "ambiguity_policy": "Retain original GRN scores; exclude ambiguous gene IDs from CellChat and use identical unambiguous program subsets in E/C ablation.",
        "layer": "lognorm",
        "expression_means_include_zeros": True,
        "time_mapping_refitted": False,
        "lr_identity_is_prior": True,
        "stage_activity_labels_used_in_training": False,
        "inputs": [{"path": f, "sha256": sha256(Path(f))} for f in files],
    }
    (out / "preparation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in [
                    "cells",
                    "candidate_pairs",
                    "rna_covered_pairs",
                    "receptor_programs",
                    "rna_genes",
                ]
            }
        )
    )
    rna.file.close()
    atac.file.close()


if __name__ == "__main__":
    main()
