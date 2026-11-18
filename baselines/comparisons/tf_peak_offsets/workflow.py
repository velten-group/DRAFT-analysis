"""All-TF network/expression peak offsets on the shared germline stage axis."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import uuid
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import issparse

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "comparisons/scdori_mondo"))
from scenic_hic import fetch_reference, load_reference

from drosophila_gnn.analysis.continuous_activity import gaussian_smooth_1d
from drosophila_gnn.preprocess import norm_id

MODELS = ["DRAFT", "SCENIC+", "scDoRI", "Dictys"]
COLORS = {
    "DRAFT": "#E5A000",
    "SCENIC+": "#4477AA",
    "scDoRI": "#333333",
    "Dictys": "#009E73",
}


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_tfs(path):
    table = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    if not {"gene_name", "FBgn_ID"}.issubset(table):
        raise ValueError("TF list must contain gene_name and FBgn_ID.")
    table = table[["gene_name", "FBgn_ID"]].apply(
        lambda column: column.astype(str).str.strip()
    )
    if table.eq("").any().any() or table.FBgn_ID.duplicated().any():
        raise ValueError("TF list contains empty identifiers or duplicate genes.")
    aliases = {}
    for i, row in enumerate(table.itertuples(index=False)):
        for value in row:
            key = value
            if key in aliases and aliases[key] != i:
                raise ValueError(f"Ambiguous TF alias: {value}.")
            aliases[key] = i
    return table.reset_index(drop=True), aliases


def expression_profiles(root, tfs):
    """Original lognorm means, including zeros, over paired annotated cells."""
    anno = pd.read_csv(root / "cell_annotation.csv", dtype=str)
    cell_col = next(
        (v for v in ["cell_id", "cell", "barcode", "Unnamed: 0"] if v in anno), None
    )
    stage_col = next(
        (
            v
            for v in ["GEX_latent_time_bins_annotated", "cell_type", "annotation"]
            if v in anno
        ),
        None,
    )
    if cell_col is None or stage_col is None:
        raise ValueError("Cannot identify annotated cells/stages.")
    rna = ad.read_h5ad(root / "A_rna_focus.h5ad", backed="r")
    atac = ad.read_h5ad(root / "B_atac_focus.h5ad", backed="r")
    try:
        if "lognorm" not in rna.layers or not rna.var_names.is_unique:
            raise ValueError(
                "RNA needs unique gene IDs and the original lognorm layer."
            )
        rna_ids = pd.Index([norm_id(v) for v in rna.obs_names])
        atac_ids = pd.Index([norm_id(v) for v in atac.obs_names])
        if not rna_ids.is_unique or not atac_ids.is_unique:
            raise ValueError("Paired cell IDs are ambiguous after normalization.")
        anno["cell_norm"] = anno[cell_col].map(norm_id)
        cells = anno.loc[
            anno[stage_col].str.match(r"^Germline_\d+$", case=False, na=False)
            & anno.cell_norm.isin(rna_ids.intersection(atac_ids))
        ].copy()
        cells["time_idx"] = cells[stage_col].str.extract(r"(\d+)$")[0].astype(int)
        if cells.cell_norm.duplicated().any() or set(cells.time_idx) != set(range(15)):
            raise ValueError(
                "Paired germline cells must map uniquely to all fifteen stages."
            )
        present = tfs.FBgn_ID.isin(rna.var_names).to_numpy()
        selected = rna.layers["lognorm"][rna_ids.get_indexer(cells.cell_norm)][
            :, rna.var_names.get_indexer(tfs.loc[present, "FBgn_ID"])
        ]
        if not np.isfinite(selected.data if issparse(selected) else selected).all():
            raise ValueError("Nonfinite lognorm expression.")
        values = np.full((15, len(tfs)), np.nan)
        for stage in range(15):
            values[stage, present] = np.asarray(
                selected[cells.time_idx.eq(stage).to_numpy()].mean(axis=0)
            ).ravel()
        counts = cells.groupby("time_idx").size().reindex(range(15)).to_numpy()
        return values, counts, cells[["cell_norm", "time_idx"]].reset_index(drop=True)
    finally:
        rna.file.close()
        atac.file.close()


def draft_profiles(directory, tfs, aliases):
    paths = list(directory.glob("TF_TG_dense_t*.tsv"))
    labeled = sorted(
        (float(re.fullmatch(r"TF_TG_dense_t(.+)\.tsv", p.name)[1]), p) for p in paths
    )
    if not labeled:
        raise FileNotFoundError(f"No complete DRAFT dense export in {directory}.")
    limits = np.asarray([labeled[0][0], labeled[-1][0]])
    offset = (
        -1
        if np.allclose(limits, [1, 15])
        else 0
        if np.allclose(limits, [0, 14])
        else None
    )
    if offset is None:
        raise ValueError("DRAFT dense times must cover either 0–14 or 1–15.")
    times = np.asarray([v + offset for v, _p in labeled])
    if len(np.unique(times)) != len(times):
        raise ValueError("Duplicate DRAFT dense times.")
    values = np.zeros((len(times), len(tfs)))
    hashes = {}
    for k, (_time, path) in enumerate(labeled):
        frame = pd.read_csv(path, sep="\t", usecols=["TF", "TG", "prob"])
        if not np.isfinite(frame.prob).all() or not frame.prob.between(0, 1).all():
            raise ValueError(f"Invalid DRAFT edge scores: {path}.")
        if frame.duplicated(["TF", "TG"]).any():
            raise ValueError(f"Duplicate selected DRAFT TF–TG edges: {path}.")
        indices = (
            frame.TF.astype(str)
            .str.replace(r"^tf:", "", regex=True, case=False)
            .str.strip()
            .map(aliases)
        )
        valid = indices.notna()
        values[k] = np.bincount(
            indices[valid].astype(int),
            weights=frame.loc[valid, "prob"],
            minlength=len(tfs),
        )
        hashes[path.name] = digest(path)
    anchors = []
    for stage in range(15):
        hit = np.flatnonzero(np.isclose(times, stage, rtol=0, atol=1e-8))
        if len(hit) != 1:
            raise ValueError(f"DRAFT is missing the original stage {stage}.")
        anchors.append(hit[0])
    report_path = directory / "dense_network_report.json"
    report = read_json(report_path)
    hashes[report_path.name] = digest(report_path)
    return (
        values[anchors],
        times,
        values,
        {
            "directory": str(directory),
            "time_offset": offset,
            "file_sha256": hashes,
            "export_report": report,
            "metric": "sum of selected outgoing TF–TG prob, matching the MONDO panel",
        },
    )


def export_missing_draft(root, directory, checkpoint=Path("auto")):
    """Use the existing exporter and manuscript settings when no export exists."""
    from compare_stage_hic import choose_draft_checkpoint

    from drosophila_gnn.analysis.germline_dynamics import ensure_dense_networks
    from drosophila_gnn.utils import load_yaml_config

    if any(directory.glob("TF_TG_dense_t*.tsv")):
        raise ValueError(
            "Existing DRAFT exports must be read/validated rather than overwritten."
        )
    config = load_yaml_config(PROJECT / "configs/germ_network.yaml")
    config["base_dir"] = str(PROJECT)
    for key, name in {
        "tf_list": "TF.tsv",
        "tg_list": "TG.tsv",
        "re_list": "RE.txt",
        "rna_h5ad": "A_rna_focus.h5ad",
        "atac_h5ad": "B_atac_focus.h5ad",
        "tf_re_prior": "TF_RE_top3pct_edges.tsv",
        "re_tg_prior": "RE_TG_100kb.tsv",
        "annotation": "cell_annotation.csv",
    }.items():
        config["data"][key] = str(root / name)
    config["reproduce"] = {
        "checkpoint_path": str(choose_draft_checkpoint(checkpoint, root))
    }
    config["network"]["tf_filter_file"] = str(root / "TF_filtered.tsv")
    settings = read_json(
        PROJECT / "docs/reference/mondo_seed2025/reference_manifest.json"
    )["network_settings"]
    config["network"].setdefault("dense", {}).update(settings)
    print(
        "No DRAFT dense export found; exporting the saved manuscript checkpoint, without training.",
        flush=True,
    )
    ensure_dense_networks(config, directory)


def scenic_profiles(directory, tfs, aliases):
    if directory == Path("auto"):
        directory = fetch_reference()
    table, report = load_reference(directory)
    values = np.zeros((15, len(tfs)))
    table["tf_idx"] = table.TF.str.strip().map(aliases)
    for (stage, tf), count in (
        table.loc[table.tf_idx.notna()]
        .groupby(["time_idx", "tf_idx"])
        .Gene.nunique()
        .items()
    ):
        values[int(stage), int(tf)] = count
    listed = set(report["tfs"])
    available = np.asarray(
        [
            row.gene_name in listed or row.FBgn_ID in listed
            for row in tfs.itertuples(index=False)
        ]
    )
    status = np.where(
        available, "no_reported_targets", "not_in_published_tf_collection"
    )
    return (
        values,
        status,
        {
            **report,
            "metric": "unique outgoing Gene count in the published DAR=yes/DEG=yes stage eRegulons; multiple regions counted once",
            "manifest_sha256": digest(directory / "reference_manifest.json"),
        },
    )


def choose_stage_export(run, networks):
    if networks is not None:
        candidates = [networks / "stage_report.json"]
    elif run is not None:
        candidates = list(run.glob("stage_networks_*/stage_report.json"))
    else:
        candidates = list(
            (PROJECT / "comparisons/scdori_mondo/runs").glob(
                "*/stage_networks_*/stage_report.json"
            )
        )
    candidates = [p for p in candidates if p.is_file()]
    candidates = [
        p
        for p in candidates
        if not read_json(p)["validation_only"]
        and read_json(p)["binding_mode"] == "native_correlation"
    ]
    if not candidates:
        raise FileNotFoundError(
            "No completed biological native-correlation scDoRI stage export. Run workflow.py stage-networks first, or use --methods DRAFT SCENIC+ for an explicit two-method run."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime_ns).parent.resolve()


def scdori_profiles(networks, root, tfs, aliases, cells, *, validation_only=False):
    report = read_json(networks / "stage_report.json")
    if (
        report["seed"] != 2025
        or report["binding_mode"] != "native_correlation"
        or bool(report["validation_only"]) != bool(validation_only)
    ):
        raise ValueError("Use the biological seed-2025 native-correlation scDoRI run.")
    for name, expected in report["files_sha256"].items():
        if digest(networks / name) != expected:
            raise ValueError(f"Changed scDoRI export: {name}.")
    for name, source in report["input_sources"].items():
        if digest(root / name) != source["sha256"]:
            raise ValueError(f"scDoRI/expression data differ: {name}.")
    observed = pd.read_csv(networks / "cells.tsv.gz", sep="\t")
    observed["cell_norm"] = observed.cell_id.map(norm_id)
    if observed.cell_norm.duplicated().any() or set(
        zip(observed.cell_norm, observed.time_idx)
    ) != set(zip(cells.cell_norm, cells.time_idx)):
        raise ValueError(
            "scDoRI and expression use different paired cell-stage cohorts."
        )
    run = networks.parent
    manifest = read_json(run / "manifest.json")
    support_path = run / "data/binding_support.tsv"
    if digest(support_path) != manifest["prepared_sha256"]["data/binding_support.tsv"]:
        raise ValueError("Changed scDoRI binding-support table.")
    support = pd.read_csv(support_path, sep="\t").set_index("tf_id")
    status = []
    for tf in tfs.FBgn_ID:
        if tf not in support.index:
            status.append("not_in_model_tf_pool")
        elif support.loc[tf, ["activator_peaks", "repressor_peaks"]].sum() == 0:
            status.append("no_tf_re_binding_support")
        else:
            status.append("zero_effective_weights")
    values = np.zeros((15, len(tfs)))
    for stage in range(15):
        frame = pd.read_csv(networks / f"t{stage:02d}_TF_TG.tsv.gz", sep="\t")
        if (
            frame.duplicated(["TF_id", "TG_id"]).any()
            or not np.isfinite(frame.path_score).all()
        ):
            raise ValueError("Invalid collapsed scDoRI TF–TG edges.")
        indices = frame.TF_id.map(aliases)
        valid = indices.notna()
        values[stage] = np.bincount(
            indices[valid].astype(int),
            weights=frame.loc[valid, "path_score"].abs(),
            minlength=len(tfs),
        )
    return (
        values,
        np.asarray(status),
        {
            **report,
            "directory": str(networks),
            "stage_report_sha256": digest(networks / "stage_report.json"),
            "binding_support_sha256": digest(support_path),
            "metric": "sum_TG abs(stage-mean signed TF–TG edge); RE path sums cancel before abs; differs from averaging per-cell absolute outdegrees",
        },
    )


def dictys_profiles(
    run, exported, root, tf_path, tfs, cells, sigma_stage, *, validation_only=False
):
    if exported is None:
        search_root = run or PROJECT / "comparisons/dictys_mondo/runs"
        pattern = (
            "tf_profiles_*/tf_profiles_report.json"
            if run
            else "shared_prior_s2025_*/tf_profiles_*/tf_profiles_report.json"
        )
        candidates = [
            p
            for p in search_root.glob(pattern)
            if not read_json(p)["validation_only"] and read_json(p)["seed"] == 2025
        ]
        if not candidates:
            raise FileNotFoundError(
                "No completed Dictys all-TF export. In dictys_mondo, run workflow.py tf-profiles --run /EXACT/RUN --tf-list /DATA/TF_filtered.tsv."
            )
        exported = max(candidates, key=lambda p: p.stat().st_mtime_ns).parent
    exported = exported.resolve()
    report = read_json(exported / "tf_profiles_report.json")
    if report["seed"] != 2025 or bool(report["validation_only"]) != bool(
        validation_only
    ):
        raise ValueError("Use the biological seed-2025 Dictys TF export.")
    if report["tf_list_sha256"] != digest(tf_path) or not np.isclose(
        report["sigma_stage"], sigma_stage
    ):
        raise ValueError(
            "Dictys TF list or smoothing sigma differs from this analysis."
        )
    for name, expected in report["files_sha256"].items():
        if digest(exported / name) != expected:
            raise ValueError(f"Changed Dictys TF export: {name}.")
    for name, source in report["input_sources"].items():
        if digest(root / name) != source["sha256"]:
            raise ValueError(f"Dictys and expression sources differ: {name}.")
    observed = pd.read_csv(exported / "cells.tsv.gz", sep="\t")
    if observed.cell_norm.duplicated().any() or set(
        zip(observed.cell_norm, observed.time_idx)
    ) != set(zip(cells.cell_norm, cells.time_idx)):
        raise ValueError("Dictys and expression cell-stage cohorts differ.")
    with np.load(exported / "stage_arrays.npz") as archive:
        if archive["tf_ids"].tolist() != tfs.FBgn_ID.tolist():
            raise ValueError("Dictys exported TF order differs.")
        raw, smooth, stages, reasons = [
            archive[key].copy()
            for key in ["raw", "smooth", "covered_stages", "reasons"]
        ]
    if (
        raw.shape != (15, len(tfs))
        or smooth.shape != raw.shape
        or not np.array_equal(stages, report["covered_stages"])
    ):
        raise ValueError("Invalid Dictys stage-array dimensions/coverage.")
    if (
        len(stages) < 3
        or not np.array_equal(stages, np.arange(stages[0], stages[-1] + 1))
        or stages[0] < 0
        or stages[-1] > 14
    ):
        raise ValueError(
            "Dictys common stage coverage must be a contiguous interval with at least three anchors."
        )
    if (
        np.isinf(raw).any()
        or np.isinf(smooth).any()
        or (raw[np.isfinite(raw)] < 0).any()
        or (smooth[np.isfinite(smooth)] < 0).any()
    ):
        raise ValueError("Invalid Dictys degree values.")
    outside = np.setdiff1d(np.arange(15), stages)
    if np.isfinite(raw[outside]).any() or np.isfinite(smooth[outside]).any():
        raise ValueError("Dictys export extrapolates outside its declared coverage.")
    return (
        raw,
        smooth,
        reasons,
        {
            **report,
            "directory": str(exported),
            "report_sha256": digest(exported / "tf_profiles_report.json"),
        },
        stages,
    )


def peak_details(values, min_range=0.1, near_max=0.95, *, stage_indices=None):
    values = np.asarray(values, dtype=float)
    stages = (
        np.arange(15) if stage_indices is None else np.asarray(stage_indices, dtype=int)
    )
    if (
        values.shape != stages.shape
        or len(stages) < 3
        or not np.isfinite(values).all()
        or (values < 0).any()
    ):
        return {"status": "missing_or_invalid_profile"}
    maximum = float(values.max())
    if maximum <= 0:
        return {"status": "zero_profile"}
    span = float((maximum - values.min()) / maximum)
    maxima = stages[np.isclose(values, maximum, rtol=1e-8, atol=0)].tolist()
    near = stages[values >= maximum * near_max].tolist()
    return {
        "status": "evaluable" if span >= min_range else "near_flat_profile",
        "peak_stage": int(stages[np.argmax(values)]),
        "maximal_stages": maxima,
        "near_max_stages": near,
        "maximum": maximum,
        "relative_dynamic_range": span,
        "endpoint_peak": any(v in [stages[0], stages[-1]] for v in maxima),
    }


def offset_record(
    expression,
    network,
    min_range=0.1,
    near_max=0.95,
    zero_reason="zero_profile",
    *,
    stage_indices=None,
):
    exp = peak_details(expression, min_range, near_max, stage_indices=stage_indices)
    net = peak_details(network, min_range, near_max, stage_indices=stage_indices)
    row = {f"expression_{k}": v for k, v in exp.items()}
    row.update({f"network_{k}": v for k, v in net.items()})
    row["status"] = "evaluable"
    if exp["status"] != "evaluable":
        row["status"] = "expression_" + exp["status"]
    elif net["status"] != "evaluable":
        row["status"] = (
            zero_reason
            if net["status"] == "zero_profile"
            else "network_" + net["status"]
        )
    if row["status"] == "evaluable":
        a, b = np.asarray(net["near_max_stages"]), np.asarray(exp["near_max_stages"])
        distance = a[:, None] - b[None, :]
        row.update(
            argmax_signed_offset=net["peak_stage"] - exp["peak_stage"],
            argmax_abs_offset=abs(net["peak_stage"] - exp["peak_stage"]),
            conservative_abs_offset=int(np.abs(distance).min()),
            direction="network_earlier"
            if distance.max() < 0
            else "network_later"
            if distance.min() > 0
            else "mixed_or_overlapping",
        )
    return row


def summarize(offsets, models, n_total, thresholds):
    records = []
    for profile in offsets.profile.unique():
        part = offsets.loc[offsets.profile.eq(profile)]
        valid = {
            model: set(
                part.loc[part.model.eq(model) & part.status.eq("evaluable"), "TF_id"]
            )
            for model in models
        }
        interior = {
            model: set(
                part.loc[
                    part.model.eq(model)
                    & part.status.eq("evaluable")
                    & ~part.expression_endpoint_peak.eq(True)
                    & ~part.network_endpoint_peak.eq(True),
                    "TF_id",
                ]
            )
            for model in models
        }
        shared = set.intersection(*(valid[model] for model in models))
        shared_interior = set.intersection(*(interior[model] for model in models))
        for scope in [
            "method_evaluable",
            "common_evaluable",
            "method_interior_peaks",
            "common_interior_peaks",
        ]:
            for model in models:
                data = part.loc[part.model.eq(model) & part.status.eq("evaluable")]
                if scope == "common_evaluable":
                    data = data.loc[data.TF_id.isin(shared)]
                elif scope == "method_interior_peaks":
                    data = data.loc[data.TF_id.isin(interior[model])]
                elif scope == "common_interior_peaks":
                    data = data.loc[data.TF_id.isin(shared_interior)]
                for threshold in thresholds:
                    far = data.conservative_abs_offset.ge(threshold)
                    records.append(
                        {
                            "profile": profile,
                            "scope": scope,
                            "model": model,
                            "offset_threshold_stages": threshold,
                            "n_tf_list": n_total,
                            "n_evaluable": len(data),
                            "n_far_conservative": int(far.sum()),
                            "fraction_far_conservative": float(far.mean())
                            if len(data)
                            else None,
                            "n_far_argmax": int(
                                data.argmax_abs_offset.ge(threshold).sum()
                            ),
                            "n_network_earlier_conservative": int(
                                (far & data.direction.eq("network_earlier")).sum()
                            ),
                            "n_network_later_conservative": int(
                                (far & data.direction.eq("network_later")).sum()
                            ),
                        }
                    )
    return pd.DataFrame(records)


def evaluate_offsets(
    tfs, expression, profiles, smooth_profiles, reasons, models, stages, args, domain
):
    rows = []
    for model in models:
        for i, tf in enumerate(tfs.itertuples(index=False)):
            for profile, values in [
                ("raw", profiles[model]),
                ("smoothed", smooth_profiles[model]),
            ]:
                record = offset_record(
                    expression[stages, i],
                    values[stages, i],
                    args.min_relative_range,
                    args.near_max_fraction,
                    reasons[model][i],
                    stage_indices=stages,
                )
                if (
                    model == "Dictys"
                    and record["status"] == "network_missing_or_invalid_profile"
                ):
                    record["status"] = (
                        reasons[model][i]
                        if reasons[model][i] != "zero_native_outdegree"
                        else "incomplete_dictys_stage_support"
                    )
                row = {
                    "TF_id": tf.FBgn_ID,
                    "TF_name": tf.gene_name,
                    "model": model,
                    "profile": profile,
                    "time_domain": domain,
                    "stage_min": int(stages[0]),
                    "stage_max": int(stages[-1]),
                    "n_stages": len(stages),
                    "network_n_valid_stages": int(np.isfinite(values[stages, i]).sum()),
                    **record,
                }
                rows.append(
                    {
                        k: json.dumps(v) if isinstance(v, list) else v
                        for k, v in row.items()
                    }
                )
    offsets = pd.DataFrame(rows)
    for column in [
        "argmax_signed_offset",
        "argmax_abs_offset",
        "conservative_abs_offset",
        "direction",
        "expression_endpoint_peak",
        "network_endpoint_peak",
    ]:
        if column not in offsets:
            offsets[column] = np.nan
    return offsets


def scoped_summary(offsets, models, n_total, thresholds):
    table = summarize(offsets, models, n_total, thresholds)
    for column in ["time_domain", "stage_min", "stage_max", "n_stages"]:
        table[column] = offsets[column].iloc[0]
    return table


def analyze(args):
    root = args.data_root.expanduser().resolve()
    tf_path = (args.tf_list or root / "TF_filtered.tsv").expanduser().resolve()
    tfs, aliases = load_tfs(tf_path)
    networks = (
        choose_stage_export(args.scdori_run, args.scdori_networks)
        if "scDoRI" in args.methods
        else None
    )
    print(f"Analyzing {len(tfs)} TFs from {tf_path}", flush=True)
    expression, n_cells, cells = expression_profiles(root, tfs)
    source = {}
    profiles, reasons = {}, {}
    dense = None
    if "DRAFT" in args.methods:
        if not any(args.draft_network_dir.glob("TF_TG_dense_t*.tsv")):
            export_missing_draft(
                root, args.draft_network_dir.resolve(), args.draft_checkpoint
            )
        profiles["DRAFT"], times, values, source["DRAFT"] = draft_profiles(
            args.draft_network_dir.resolve(), tfs, aliases
        )
        dense = (times, values)
        reasons["DRAFT"] = np.full(len(tfs), "no_selected_edges")
    if "SCENIC+" in args.methods:
        profiles["SCENIC+"], reasons["SCENIC+"], source["SCENIC+"] = scenic_profiles(
            args.scenic_reference, tfs, aliases
        )
    if networks is not None:
        profiles["scDoRI"], reasons["scDoRI"], source["scDoRI"] = scdori_profiles(
            networks, root, tfs, aliases, cells
        )
    stages = np.arange(15)
    domain = "full_trajectory"
    curve_rows, smooth_profiles = [], {}
    if "Dictys" in args.methods:
        (
            profiles["Dictys"],
            smooth_profiles["Dictys"],
            reasons["Dictys"],
            source["Dictys"],
            stages,
        ) = dictys_profiles(
            args.dictys_run,
            args.dictys_profiles,
            root,
            tf_path,
            tfs,
            cells,
            args.sigma_stage,
        )
        domain = "dictys_shared_coverage"
    for model in args.methods:
        raw = profiles[model]
        smooth = (
            smooth_profiles[model]
            if model == "Dictys"
            else np.stack(
                [
                    gaussian_smooth_1d(raw[:, i], np.arange(15), args.sigma_stage)
                    for i in range(len(tfs))
                ],
                axis=1,
            )
        )
        smooth_profiles[model] = smooth
        for i, tf in enumerate(tfs.itertuples(index=False)):
            for stage in range(15):
                curve_rows.append(
                    {
                        "TF_id": tf.FBgn_ID,
                        "TF_name": tf.gene_name,
                        "model": model,
                        "time_idx": stage,
                        "n_cells": int(n_cells[stage]),
                        "mean_expression": expression[stage, i],
                        "network_raw": raw[stage, i],
                        "network_smoothed": smooth[stage, i],
                        "in_primary_time_domain": stage in stages,
                    }
                )
    offsets = evaluate_offsets(
        tfs,
        expression,
        profiles,
        smooth_profiles,
        reasons,
        args.methods,
        stages,
        args,
        domain,
    )
    summary = scoped_summary(offsets, args.methods, len(tfs), args.thresholds)
    out = args.output_root.resolve() / f"seed2025_{uuid.uuid4().hex[:10]}"
    out.mkdir(parents=True, exist_ok=False)
    offsets.to_csv(out / "tf_peak_offsets.tsv", sep="\t", index=False)
    summary.to_csv(out / "offset_counts.tsv", sep="\t", index=False)
    if "Dictys" in args.methods:
        full_models = [model for model in args.methods if model != "Dictys"]
        if full_models:
            full_offsets = evaluate_offsets(
                tfs,
                expression,
                profiles,
                smooth_profiles,
                reasons,
                full_models,
                np.arange(15),
                args,
                "full_trajectory",
            )
            full_offsets.to_csv(
                out / "full_trajectory_tf_peak_offsets.tsv", sep="\t", index=False
            )
            scoped_summary(full_offsets, full_models, len(tfs), args.thresholds).to_csv(
                out / "full_trajectory_offset_counts.tsv", sep="\t", index=False
            )
    pd.DataFrame(curve_rows).to_csv(
        out / "tf_stage_profiles.tsv.gz", sep="\t", index=False
    )
    cells.to_csv(out / "paired_cells.tsv.gz", sep="\t", index=False)
    exclusions = (
        offsets.loc[offsets.profile.eq("smoothed")]
        .groupby(["model", "status"])
        .size()
        .rename("n_tf")
        .reset_index()
    )
    exclusions.to_csv(out / "tf_coverage.tsv", sep="\t", index=False)
    for threshold in args.thresholds:
        offsets.loc[
            offsets.status.eq("evaluable")
            & offsets.conservative_abs_offset.ge(threshold)
        ].sort_values(
            ["model", "conservative_abs_offset"], ascending=[True, False]
        ).to_csv(out / f"tf_far_ge{threshold:g}_stages.tsv", sep="\t", index=False)
    if dense is not None:
        times, values = dense
        dense_rows = []
        for i, tf in enumerate(tfs.itertuples(index=False)):
            smooth = gaussian_smooth_1d(values[:, i], times, args.sigma_stage)
            maximum = float(smooth.max())
            dense_rows.append(
                {
                    "TF_id": tf.FBgn_ID,
                    "TF_name": tf.gene_name,
                    "dense_network_peak_time": float(times[smooth.argmax()])
                    if maximum > 0
                    else None,
                    "dense_network_maximum": maximum,
                }
            )
        pd.DataFrame(dense_rows).to_csv(
            out / "draft_dense_peaks.tsv", sep="\t", index=False
        )
        mondo = np.flatnonzero(tfs.gene_name.str.casefold().eq("mondo"))
        if len(mondo):
            reference = pd.read_csv(
                PROJECT / "docs/reference/mondo_seed2025/mondo_dynamics_centrality.tsv",
                sep="\t",
            )
            source["DRAFT"]["mondo_manuscript_raw_profile_matches"] = bool(
                len(reference) == len(times)
                and np.allclose(reference.time, times)
                and np.allclose(
                    reference.centrality_raw, values[:, mondo[0]], rtol=1e-5, atol=1e-8
                )
            )
    report = {
        "methods": args.methods,
        "tf_list": str(tf_path),
        "tf_list_sha256": digest(tf_path),
        "n_tf_list": len(tfs),
        "paired_cells": len(cells),
        "stage_cell_counts": n_cells.tolist(),
        "sources": source,
        "data_sha256": {
            name: digest(root / name)
            for name in ["A_rna_focus.h5ad", "B_atac_focus.h5ad", "cell_annotation.csv"]
        },
        "settings": {
            "sigma_stage": args.sigma_stage,
            "min_relative_dynamic_range": args.min_relative_range,
            "near_max_fraction": args.near_max_fraction,
            "offset_thresholds_stages": args.thresholds,
        },
        "primary_time_domain": {
            "name": domain,
            "stage_indices": stages.tolist(),
            "interpretation": "all included methods and expression restricted to identical original stage anchors; endpoint maxima refer to the first/last analyzed anchor, not necessarily G0/G14",
        },
        "primary": "network profiles evaluated at identical original stage anchors, raw stage mean lognorm expression including zero cells; Dictys uses native signed-edge Gaussian interpolation within fitted-center coverage before absolute degree and receives no second smoothing; other methods retain existing stage-degree smoothing; any TF missing an analyzed stage is excluded",
        "offset": "network_peak - expression_peak; earliest argmax retained only as descriptive; conservative offset is the minimum distance between ALL network/expression near-maximum stages",
        "exclusions": "missing expression, zero expression/network profiles and relative dynamic range below the declared cutoff excluded; unsupported TFs retained in coverage/list denominator; common_evaluable intersects evaluable TFs across the methods actually run",
        "cautions": "stage units are ordinal, not elapsed hours; method-specific activity units, target pools and selection filters differ; offsets demonstrate temporal decoupling of model-derived profiles, not expression-independent causal TF action; single-seed descriptive counts with no biological-replicate uncertainty",
        "analysis_code_sha256": digest(Path(__file__)),
    }
    (out / "analysis_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    plot_results(
        summary,
        offsets,
        tfs,
        expression,
        smooth_profiles,
        args,
        out,
        stage_indices=stages,
    )
    print(
        summary.loc[summary.profile.eq("smoothed")].to_string(index=False), flush=True
    )
    print(f"TF peak offsets: {out}", flush=True)
    return out


def plot_results(
    summary, offsets, tfs, expression, profiles, args, out, *, stage_indices=None
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = read_json(PROJECT / "configs/germline_stage_labels.json")["labels"]
    stages = np.arange(15) if stage_indices is None else np.asarray(stage_indices)
    width = min(0.23, 0.8 / len(args.methods))
    with plt.rc_context({"pdf.fonttype": 42, "svg.fonttype": "none"}):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
        for ax, scope in zip(axes, ["method_evaluable", "common_evaluable"]):
            for i, model in enumerate(args.methods):
                selected = summary.loc[
                    summary.profile.eq("smoothed")
                    & summary.scope.eq(scope)
                    & summary.model.eq(model)
                ]
                x = np.arange(len(selected)) + (i - (len(args.methods) - 1) / 2) * width
                bars = ax.bar(
                    x,
                    selected.fraction_far_conservative.fillna(0),
                    width=width,
                    color=COLORS[model],
                    label=model,
                )
                for bar, row in zip(bars, selected.itertuples(index=False)):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.015,
                        f"{row.n_far_conservative}/{row.n_evaluable}"
                        if row.n_evaluable
                        else "NA",
                        ha="center",
                        fontsize=8,
                    )
            ax.set_xticks(
                range(len(args.thresholds)), [f"≥{v:g} stages" for v in args.thresholds]
            )
            ax.set(
                ylim=(0, 1.05),
                ylabel="Fraction with separated peak regions",
                title="Each method's evaluable TFs"
                if scope == "method_evaluable"
                else "TFs evaluable in every included method",
            )
            ax.legend(frameon=False)
        fig.suptitle(
            "Germline TF network–expression peak separation"
            + (
                f" (matched stage coverage {stages[0]}–{stages[-1]})"
                if len(stages) != 15
                else ""
            )
        )
        for suffix in ["png", "pdf", "svg"]:
            fig.savefig(out / f"peak_offset_counts.{suffix}", dpi=300)
        plt.close(fig)
        rank = offsets.loc[
            offsets.profile.eq("smoothed")
            & offsets.model.eq("DRAFT")
            & offsets.status.eq("evaluable")
        ].sort_values("conservative_abs_offset", ascending=False)
        indices = [int(tfs.index[tfs.FBgn_ID.eq(v)][0]) for v in rank.TF_id.head(12)]
        mondo = tfs.index[tfs.gene_name.str.casefold().eq("mondo")].tolist()
        if mondo:
            indices = list(dict.fromkeys(mondo + indices))[:12]
        if not indices:
            return
        fig, axes = plt.subplots(
            int(np.ceil(len(indices) / 3)),
            3,
            figsize=(13, 2.8 * int(np.ceil(len(indices) / 3))),
            constrained_layout=True,
            squeeze=False,
        )
        for ax, i in zip(axes.flat, indices):
            for model in args.methods:
                values = profiles[model][stages, i]
                if np.isfinite(values).any() and np.nanmax(values) > 0:
                    ax.plot(
                        stages,
                        values / np.nanmax(values),
                        label=model,
                        color=COLORS[model],
                        lw=1.7,
                    )
            values = expression[stages, i]
            if np.isfinite(values).all() and values.max() > 0:
                ax.plot(
                    stages,
                    values / values.max(),
                    label="Mean expression",
                    color="#795099",
                    linestyle="--",
                    marker=".",
                    lw=1.7,
                )
            ax.set(
                title=tfs.iloc[i].gene_name,
                ylim=(-0.03, 1.08),
                ylabel="Relative profile",
            )
            ax.set_xticks(
                stages,
                [labels[stage] for stage in stages],
                rotation=60,
                ha="right",
                fontsize=7,
            )
        for ax in list(axes.flat)[len(indices) :]:
            ax.set_visible(False)
        handles = {
            name: handle
            for ax in axes.flat
            for handle, name in zip(*ax.get_legend_handles_labels())
        }
        fig.legend(
            handles.values(),
            handles.keys(),
            loc="outside upper center",
            ncol=len(handles),
            frameon=False,
        )
        for suffix in ["png", "pdf", "svg"]:
            fig.savefig(out / f"tf_offset_examples.{suffix}", dpi=300)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT / "data/raw")
    parser.add_argument("--tf-list", type=Path)
    parser.add_argument("--methods", nargs="+", choices=MODELS, default=MODELS[:3])
    parser.add_argument(
        "--draft-network-dir", type=Path, default=PROJECT / "result/germ/network_dense"
    )
    parser.add_argument("--draft-checkpoint", type=Path, default=Path("auto"))
    parser.add_argument("--scenic-reference", type=Path, default=Path("auto"))
    parser.add_argument("--scdori-run", type=Path)
    parser.add_argument("--scdori-networks", type=Path)
    parser.add_argument("--dictys-run", type=Path)
    parser.add_argument("--dictys-profiles", type=Path)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[2, 3, 4])
    parser.add_argument("--sigma-stage", type=float, default=0.5)
    parser.add_argument("--min-relative-range", type=float, default=0.1)
    parser.add_argument("--near-max-fraction", type=float, default=0.95)
    parser.add_argument(
        "--output-root", type=Path, default=PROJECT / "result/germ/tf_peak_offsets"
    )
    args = parser.parse_args()
    if (
        len(set(args.methods)) != len(args.methods)
        or not np.isfinite(args.sigma_stage)
        or args.sigma_stage <= 0
        or not 0 <= args.min_relative_range <= 1
        or not 0 < args.near_max_fraction <= 1
        or not all(np.isfinite(v) and v > 0 for v in args.thresholds)
        or len(set(args.thresholds)) != len(args.thresholds)
    ):
        parser.error(
            "Methods/thresholds must be unique, sigma/offsets positive, dynamic range in [0,1], near-maximum fraction in (0,1]."
        )
    analyze(args)


if __name__ == "__main__":
    main()
