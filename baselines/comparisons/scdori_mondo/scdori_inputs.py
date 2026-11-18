"""Paired-count and shared-prior adapter; original inputs are never modified."""

from __future__ import annotations

import logging
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import yaml
from lineages import select_annotation
from scdori_native import (
    PIN,
    comparison_helpers,
    digest,
    load_native,
    read_json,
    safe_run_path,
    write_json,
    zero_evidence_repressors,
)

LOG = logging.getLogger(__name__)


def validate_counts(matrix, label: str):
    matrix = sp.csr_matrix(matrix, dtype=np.float32)
    values = matrix.data
    if (
        not np.isfinite(values).all()
        or (values < 0).any()
        or not np.allclose(values, np.round(values))
    ):
        raise ValueError(f"{label} must contain raw integer counts.")
    return matrix


def shared_matrices(
    root: Path, genes: list[str], tfs: list[str], peaks: list[str], cfg: dict
):
    helpers = comparison_helpers()
    tf = pd.read_csv(root / "TF.tsv", sep="\t", dtype=str)
    tg = pd.read_csv(root / "TG.tsv", sep="\t", dtype=str)
    mapping = {**helpers.symbol_map(tg), **helpers.symbol_map(tf)}
    pidx, gidx, tidx = [
        dict(zip(names, range(len(names)))) for names in [peaks, genes, tfs]
    ]
    region = helpers.region_mapper(peaks)
    motif = np.zeros((len(peaks), len(tfs)), dtype=np.float32)
    table = pd.read_csv(root / "TF_RE_top3pct_edges.tsv", sep="\t", dtype=str)
    for name, peak in zip(table.TF, table.RE):
        gid = mapping.get(re.sub(r"\s*\(\d+\)$", "", name))
        pid = region(peak)
        if gid in tidx and pid in pidx:
            motif[pidx[pid], tidx[gid]] = 1
    table = pd.read_csv(root / "RE_TG_100kb.tsv", sep="\t")
    gene_peak = np.zeros((len(genes), len(peaks)), dtype=np.float32)
    scale = cfg["prior"]["distance_scale"]
    for row in table.itertuples(index=False):
        gid = mapping.get(str(row.gene_name), str(row.gene_name))
        if gid not in gidx:
            continue
        pid = region(f"{row.chr_re}:{row.re_start}-{row.re_end}")
        if pid in pidx:
            distance = abs(float(row.distance))
            value = np.exp(-distance / scale)
            gene_peak[gidx[gid], pidx[pid]] = max(
                gene_peak[gidx[gid], pidx[pid]], value
            )
    # scDoRI reconstructs TF genes as RNA variables too. DRAFT's RE-TG file
    # excludes TFs, so add the same genomic cis rule for this input role only.
    coordinates = [re.fullmatch(r"([^:]+):(\d+)-(\d+)", p) for p in peaks]
    if not all(coordinates):
        raise ValueError("Peak identifiers must contain genomic coordinates.")
    chromosomes = np.asarray([m[1].removeprefix("chr") for m in coordinates])
    midpoints = np.asarray([(int(m[2]) + int(m[3])) / 2 for m in coordinates])
    tf_coords = tf.set_index("FBgn_ID")
    extra = 0
    for gid in tfs:
        row = tf_coords.loc[gid]
        start, end = float(row.start), float(row.end)
        distance = np.maximum(np.maximum(start - midpoints, midpoints - end), 0)
        keep = (chromosomes == str(row.chr).removeprefix("chr")) & (
            distance <= cfg["prior"]["tf_gene_cis_window"]
        )
        gene_peak[gidx[gid], keep] = np.exp(-distance[keep] / scale)
        extra += int(keep.sum())
    return (
        gene_peak,
        motif,
        {
            "tf_gene_cis_links_added": extra,
            "tf_re_pairs": int(motif.sum()),
            "gene_peak_pairs": int((gene_peak > 0).sum()),
        },
    )


def prepare_run(
    config: Path, data_root: Path | None, upstream: Path | None, seed: int | None
) -> Path:
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    cfg["binding_mode"] = "native_correlation"
    if seed is not None:
        cfg["seed"] = seed
    cfg["source_root"] = str((upstream or config.parent / cfg["upstream"]).resolve())
    native = load_native(Path(cfg["source_root"]))
    native["_core.utils"].set_seed(cfg["seed"])
    helpers = comparison_helpers()
    root = (data_root or config.parent / cfg["data_root"]).resolve()
    inputs = {
        name: root / name
        for name in [
            "A_rna_focus.h5ad",
            "B_atac_focus.h5ad",
            "cell_annotation.csv",
            "TF.tsv",
            "TG.tsv",
            "RE.txt",
            "TF_RE_top3pct_edges.tsv",
            "RE_TG_100kb.tsv",
        ]
    }
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    LOG.info("Matching RNA, ATAC and annotated lineage cells.")
    rna0, atac0 = [
        ad.read_h5ad(inputs[name]) for name in ["A_rna_focus.h5ad", "B_atac_focus.h5ad"]
    ]
    ann = pd.read_csv(inputs["cell_annotation.csv"])
    ann["cell_norm"] = ann.iloc[:, 0].map(helpers.normalize_cell)
    rn = pd.Index([helpers.normalize_cell(x) for x in rna0.obs_names])
    an = pd.Index([helpers.normalize_cell(x) for x in atac0.obs_names])
    if not rn.is_unique or not an.is_unique or ann.cell_norm.duplicated().any():
        raise ValueError("Normalized cell identifiers must be unique.")
    lineage = cfg.get("lineage", "germline")
    ann = select_annotation(
        ann.loc[ann.cell_norm.isin(rn) & ann.cell_norm.isin(an)], lineage
    )
    ann["pseudotime"] = pd.to_numeric(ann.GEX_latent_time)
    ann["cell_id"] = rna0.obs_names[rn.get_indexer(ann.cell_norm)].to_numpy()
    rna = rna0[ann.cell_id].copy()
    atac = atac0[atac0.obs_names[an.get_indexer(ann.cell_norm)]].copy()
    atac.obs_names = rna.obs_names.copy()
    del rna0, atac0
    rna.X, atac.X = [
        validate_counts(a.layers["counts"], name)
        for a, name in [(rna, "RNA"), (atac, "ATAC")]
    ]
    tf_table = pd.read_csv(inputs["TF.tsv"], sep="\t", dtype=str)
    tg_table = pd.read_csv(inputs["TG.tsv"], sep="\t", dtype=str)
    gene_set = set(tf_table.FBgn_ID) | set(tg_table.FBgn_ID)
    detected = rna.X.getnnz(axis=0) >= cfg["feature_selection"]["min_cells"]
    rna = rna[:, detected & rna.var_names.isin(gene_set)].copy()
    tfs = sorted(set(tf_table.FBgn_ID) & set(rna.var_names))
    if cfg["gene_id"] not in tfs:
        raise ValueError(
            "MONDO is absent/filtered; cannot treat feature omission as inactivity."
        )
    n_genes = cfg["feature_selection"]["total_genes"]
    if not cfg["feature_selection"]["all_tfs"] or n_genes <= len(tfs):
        raise ValueError(
            "This controlled design retains all expressed TFs and needs additional non-TF genes."
        )
    non_tf = rna[:, ~rna.var_names.isin(tfs)].copy()
    non_tf.uns.pop("log1p", None)  # original file's metadata described its former X
    sc.pp.normalize_total(non_tf)
    sc.pp.log1p(non_tf)
    sc.pp.highly_variable_genes(
        non_tf, n_top_genes=min(n_genes - len(tfs), non_tf.n_vars), subset=True
    )
    targets = sorted(non_tf.var_names)
    genes = [*targets, *tfs]
    rna = rna[:, genes].copy()
    rna.var["gene_type"] = ["HVG"] * len(targets) + ["TF"] * len(tfs)
    rna.var["highly_variable"] = (
        True  # use the newly selected feature set for metacell PCA
    )
    peaks_known = set(inputs["RE.txt"].read_text(encoding="utf-8").split())
    atac = atac[:, atac.var_names.isin(peaks_known)].copy()
    if cfg["atac_count_unit"] == "fragments":
        atac.X = atac.X * 2  # native (X+1)//2 will recover original counts exactly
    elif cfg["atac_count_unit"] != "insertions":
        raise ValueError("ATAC count unit must be fragments or insertions.")
    for obj in [rna, atac]:
        obj.obs["stage"] = ann.stage.to_numpy()
        obj.obs["pseudotime"] = ann.pseudotime.to_numpy()
        obj.obs["num_cells"] = 1
        if "sample" not in obj.obs:
            obj.obs["sample"] = (
                ann.cell_norm.str.extract(r"_sample(\d+)$")[0]
                .fillna("single")
                .to_numpy()
            )
    if np.any(np.asarray(rna.X[:, -len(tfs) :].sum(axis=1)).ravel() == 0):
        raise ValueError(
            "A cell has zero total TF counts; native TF normalization is undefined."
        )
    # RNA/ATAC counts are unchanged in value (except the explicitly recorded
    # count-unit adapter), but remove unnecessary layers/embeddings in new copies.
    rna = ad.AnnData(rna.X, obs=rna.obs.copy(), var=rna.var.copy())
    atac = ad.AnnData(atac.X, obs=atac.obs.copy(), var=atac.var.copy())
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = safe_run_path(
        config.parent
        / cfg["output_root"]
        / f"{'native_correlation' if lineage == 'germline' else f'native_correlation_{lineage}'}"
        f"_s{cfg['seed']}_{stamp}_{uuid.uuid4().hex[:8]}"
    )
    data = run / "data"
    data.mkdir(parents=True, exist_ok=False)
    LOG.info("Building shared gene-peak and binary motif-support matrices.")
    gene_peak, motif, details = shared_matrices(
        root, genes, tfs, list(atac.var_names), cfg
    )
    LOG.info(
        "Computing official RNA-based metacells and TF-peak correlation embeddings."
    )
    resolution = cfg["prior"]["metacell_resolution"]
    if resolution is None:
        # Explicit single-cell mode for software fixtures or a separately named run.
        rna_matrix = rna.X[:, -len(tfs) :].toarray()
        atac_matrix = (atac.X.toarray() + 1) // 2
        n_metacells = rna.n_obs
    else:
        rmeta, ameta = native["pp.metacells"].create_metacells(
            rna, atac, resolution=resolution, batch_key="sample"
        )
        rna_matrix = rmeta.X[:, -len(tfs) :]
        atac_matrix = ameta.X
        n_metacells = rmeta.n_obs
        rna.obs[["leiden"]].to_csv(data / "metacell_membership.tsv", sep="\t")
    act, rep = native["pp.correlation"].compute_in_silico_chipseq(
        atac_matrix,
        rna_matrix,
        pd.DataFrame(motif, columns=tfs, index=atac.var_names),
        percentile=cfg["prior"]["correlation_percentile"],
        n_bg=cfg["prior"]["background_peaks"],
    )
    if cfg["prior"]["zero_repressor_columns"] != "keep_zero":
        raise ValueError(
            "Zero-evidence repressor columns must remain zero in this adapter."
        )
    act, rep, corrected = zero_evidence_repressors(act, rep)
    # Float32 affine MinMax scaling can leave sub-epsilon residuals at zero.
    # Reject actual support violations, but restore exact prior zeros for rounding.
    tolerance = 1e-6
    if (np.abs(act[motif == 0]) > tolerance).any() or (
        np.abs(rep[motif == 0]) > tolerance
    ).any():
        raise ValueError(
            "Binding embeddings contain edges outside shared motif support."
        )
    act[np.abs(act) < tolerance] = 0
    rep[np.abs(rep) < tolerance] = 0
    act[motif == 0] = 0
    rep[motif == 0] = 0
    np.save(data / "gene_peak_distance_exp.npy", gene_peak)
    np.save(data / "insilico_chipseq_act.npy", act)
    np.save(data / "insilico_chipseq_rep.npy", rep)
    pd.DataFrame(
        {
            "tf_id": tfs,
            "motif_supported_peaks": motif.sum(axis=0).astype(int),
            "activator_peaks": (act > 0).sum(axis=0),
            "repressor_peaks": (rep < 0).sum(axis=0),
        }
    ).to_csv(data / "binding_support.tsv", sep="\t", index=False)
    # A single seeded random permutation, ordered prefix train loader, and suffix
    # validation loader preserve the native get_tf_expression cell correspondence.
    permutation = np.random.default_rng(cfg["split_seed"]).permutation(rna.n_obs)
    rna, atac = rna[permutation].copy(), atac[permutation].copy()
    rna.X, atac.X = rna.X.astype(np.float32), atac.X.astype(np.float32)
    rna.write_h5ad(data / "rna_processed.h5ad", compression="gzip")
    atac.write_h5ad(data / "atac_processed.h5ad", compression="gzip")
    files = [p for p in data.iterdir() if p.is_file()]
    manifest = {
        "design": "scDoRI core with shared DRAFT priors, all expressed TFs and HVG targets; no de-novo motif scanning",
        "upstream_commit": PIN,
        "configuration": cfg,
        "prepared_sha256": {str(p.relative_to(run)): digest(p) for p in files},
        "input_sources": {
            name: {"path": str(p), "sha256": digest(p)} for name, p in inputs.items()
        },
        "n_cells": rna.n_obs,
        "n_genes": rna.n_vars,
        "n_tfs": len(tfs),
        "n_targets": len(targets),
        "n_peaks": atac.n_vars,
        "n_metacells": n_metacells,
        "prior": details,
        "corrected_zero_repressor_tf_ids": [tfs[i] for i in corrected],
        "atac_unit_adapter": cfg["atac_count_unit"],
        "binding_roundoff_tolerance": tolerance,
        "training_cell_order": "one seeded random permutation; prefix 80% train, suffix 20% validation; no loader shuffling to preserve native topic/expression indexing",
        "readout": "unthresholded effective model MONDO->non-TF-target edge weighted out-degree; not native permutation-filtered regulon counts",
    }
    write_json(run / "manifest.json", manifest)
    report_preprocessing(run)
    print(
        f"Prepared: {run}\nCells={rna.n_obs}, genes={rna.n_vars}, TFs={len(tfs)}, targets={len(targets)}, peaks={atac.n_vars}, metacells={n_metacells}",
        flush=True,
    )
    return run


def load_run(run: Path) -> dict:
    run = safe_run_path(run)
    manifest = read_json(run / "manifest.json")
    for name, expected in manifest["prepared_sha256"].items():
        if digest(run / name) != expected:
            raise ValueError(f"Prepared data changed: {name}; use a fresh run.")
    return manifest


def report_preprocessing(run: Path) -> dict:
    manifest = load_run(run)
    gid = manifest["configuration"]["gene_id"]
    table = pd.read_csv(run / "data/binding_support.tsv", sep="\t")
    row = table.loc[table.tf_id.eq(gid)]
    if len(row) != 1:
        raise ValueError("MONDO must have exactly one binding-support record.")
    values = row.iloc[0]
    supported = bool(values.activator_peaks > 0 or values.repressor_peaks > 0)
    mode = manifest["configuration"].get("binding_mode", "native_correlation")
    report = {
        "gene_id": gid,
        "binding_mode": mode,
        "motif_supported_peaks": int(values.motif_supported_peaks),
        "activator_peaks": int(values.activator_peaks),
        "repressor_peaks": int(values.repressor_peaks),
        "can_estimate_mondo_edges": supported,
        "status": "binding_input_supported"
        if supported
        else "no_binding_evidence_after_preprocessing",
        "interpretation": "Untrained input diagnostic. With both MONDO binding columns zero and frozen, the native GRN branch cannot recover MONDO edges. This is not evidence about temporal localization or fully native scDoRI motif inference.",
    }
    path = run / "mondo_preprocessing_report.json"
    if path.exists() and read_json(path) != report:
        raise ValueError("Existing preprocessing report differs; do not overwrite it.")
    if not path.exists():
        write_json(path, report)
    print(
        f"MONDO input: mode={mode}, activator peaks={report['activator_peaks']}, repressor peaks={report['repressor_peaks']}, supported={supported}",
        flush=True,
    )
    return report


def derive_diagnostic(native_run: Path) -> Path:
    """Clone immutable prepared inputs and replace bindings for ALL TFs.

    No MONDO-specific rescue and no trained/default checkpoints are copied.
    Both signs are allowed by the unsigned motif prior; actual effects are learned.
    """
    manifest = load_run(native_run)
    cfg = manifest["configuration"].copy()
    if cfg.get("binding_mode", "native_correlation") != "native_correlation":
        raise ValueError(
            "Derive the diagnostic from a native-correlation prepared run."
        )
    tf_path = Path(manifest["input_sources"]["TF.tsv"]["path"])
    edges_path = Path(manifest["input_sources"]["TF_RE_top3pct_edges.tsv"]["path"])
    for key, path in [("TF.tsv", tf_path), ("TF_RE_top3pct_edges.tsv", edges_path)]:
        if digest(path) != manifest["input_sources"][key]["sha256"]:
            raise ValueError(f"Original prior changed: {path}")
    rna = ad.read_h5ad(native_run / "data/rna_processed.h5ad", backed="r")
    atac = ad.read_h5ad(native_run / "data/atac_processed.h5ad", backed="r")
    tfs = rna.var_names[rna.var.gene_type.eq("TF")].tolist()
    peaks = atac.var_names.tolist()
    rna.file.close()
    atac.file.close()
    helpers = comparison_helpers()
    mapping = helpers.symbol_map(pd.read_csv(tf_path, sep="\t", dtype=str))
    region = helpers.region_mapper(peaks)
    pidx, tidx = [dict(zip(names, range(len(names)))) for names in [peaks, tfs]]
    motif = np.zeros((len(peaks), len(tfs)), dtype=np.float32)
    edges = pd.read_csv(edges_path, sep="\t", dtype=str)
    for symbol, peak in zip(edges.TF, edges.RE):
        gid = mapping.get(re.sub(r"\s*\(\d+\)$", "", symbol))
        pid = region(peak)
        if gid in tidx and pid in pidx:
            motif[pidx[pid], tidx[gid]] = 1
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = safe_run_path(
        native_run.parent
        / f"shared_unsigned_s{cfg['seed']}_{stamp}_{uuid.uuid4().hex[:8]}"
    )
    (run / "data").mkdir(parents=True, exist_ok=False)
    for path in (native_run / "data").iterdir():
        if path.is_file() and path.name not in {
            "insilico_chipseq_act.npy",
            "insilico_chipseq_rep.npy",
            "binding_support.tsv",
        }:
            shutil.copy2(path, run / "data" / path.name)
    shutil.copy2(
        native_run / "data/binding_support.tsv",
        run / "data/native_correlation_binding_support.tsv",
    )
    np.save(run / "data/insilico_chipseq_act.npy", motif)
    np.save(run / "data/insilico_chipseq_rep.npy", -motif)
    pd.DataFrame(
        {
            "tf_id": tfs,
            "motif_supported_peaks": motif.sum(axis=0).astype(int),
            "activator_peaks": motif.sum(axis=0).astype(int),
            "repressor_peaks": motif.sum(axis=0).astype(int),
        }
    ).to_csv(run / "data/binding_support.tsv", sep="\t", index=False)
    cfg["binding_mode"] = "shared_unsigned"
    manifest["configuration"] = cfg
    manifest["design"] = (
        "DIAGNOSTIC: scDoRI core using unsigned shared TF-RE support in both sign branches; native correlation gating bypassed for every TF, not only MONDO"
    )
    manifest["source_native_run"] = str(native_run.resolve())
    manifest["source_native_manifest_sha256"] = digest(native_run / "manifest.json")
    manifest["native_correlation_zero_repressor_tf_ids"] = manifest.pop(
        "corrected_zero_repressor_tf_ids", []
    )
    manifest["prepared_sha256"] = {
        str(path.relative_to(run)): digest(path)
        for path in (run / "data").iterdir()
        if path.is_file()
    }
    write_json(run / "manifest.json", manifest)
    report_preprocessing(run)
    print(f"Diagnostic prepared: {run}", flush=True)
    return run
