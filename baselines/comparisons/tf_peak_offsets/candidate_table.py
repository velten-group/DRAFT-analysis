"""Export TF peak locations and matched offset detection across saved models."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
sys.path.insert(0, str(HERE))
import workflow as wf

MODELS = [
    "DRAFT",
    "DRAFT w/o GRU",
    "Linear + GRU",
    "SCENIC+",
    "scDoRI",
    "Dictys",
    "separate GAT",
]
DEFAULT_MODELS = MODELS[:5]
SLUGS = dict(
    zip(
        MODELS,
        [
            "DRAFT",
            "no_GRU",
            "Linear_GRU",
            "SCENICplus",
            "scDoRI",
            "Dictys",
            "separate_GAT",
        ],
    )
)
NETWORK_FIELDS = (
    "active_thr",
    "min_score_keep",
    "top_k_global",
    "n_interp",
    "sigma_time",
    "rescue_gamma",
    "rescue_m_min",
    "rescue_m_max",
    "rescue_eps",
)


def choose_analysis(value, models):
    required = set(models) & set(wf.MODELS)
    candidates = (
        list(
            (PROJECT / "result/germ/tf_peak_offsets").glob(
                "seed2025_*/analysis_report.json"
            )
        )
        if str(value) == "latest"
        else [Path(value) / "analysis_report.json"]
    )
    valid = []
    for path in candidates:
        if not path.is_file():
            continue
        report = wf.read_json(path)
        if (
            required <= set(report["methods"])
            and not any(
                source.get("validation_only", False)
                for source in report["sources"].values()
            )
            and all(
                (path.parent / name).is_file()
                for name in ["tf_peak_offsets.tsv", "tf_stage_profiles.tsv.gz"]
            )
        ):
            valid.append(path)
    if not valid:
        raise FileNotFoundError(
            f"No completed biological peak analysis with {sorted(required)}. "
            "Run comparisons/tf_peak_offsets/workflow.py with these methods first."
        )
    return max(valid, key=lambda p: p.stat().st_mtime_ns).parent.resolve()


def saved_offsets(path, models, domain, stages, tfs):
    table = pd.read_csv(path, sep="\t", keep_default_na=False, na_values=[""])
    table = table.loc[table.model.isin(models)].copy()
    if (
        set(table.model) != set(models)
        or table.duplicated(["model", "profile", "TF_id"]).any()
    ):
        raise ValueError(f"Missing models or duplicate TFs in {path}.")
    for (model, profile), part in table.groupby(["model", "profile"]):
        if profile not in ["raw", "smoothed"] or set(part.TF_id) != set(tfs.FBgn_ID):
            raise ValueError(f"TF/profile mismatch for {model}: {path}.")
        names = part.set_index("TF_id").TF_name.reindex(tfs.FBgn_ID).tolist()
        if names != tfs.gene_name.tolist():
            raise ValueError("TF symbols differ from the curated TF list.")
    if len(table) != len(models) * len(tfs) * 2:
        raise ValueError(
            "Both raw and smoothed records are required for every TF/model."
        )
    if "time_domain" in table and not table.time_domain.eq(domain).all():
        raise ValueError("Cached peak table has a different time domain.")
    for key, expected in [
        ("stage_min", stages[0]),
        ("stage_max", stages[-1]),
        ("n_stages", len(stages)),
    ]:
        if key in table and not table[key].eq(expected).all():
            raise ValueError("Cached peak table has different stage coverage.")
    table["time_domain"] = domain
    table["stage_min"], table["stage_max"], table["n_stages"] = (
        stages[0],
        stages[-1],
        len(stages),
    )
    for value in table.loc[table.status.eq("evaluable"), "conservative_abs_offset"]:
        if not np.isfinite(float(value)) or float(value) < 0:
            raise ValueError("Invalid cached peak separation.")
    return table


def expression_from_cache(path, tfs, report):
    table = pd.read_csv(path, sep="\t", keep_default_na=False, na_values=[""])
    if table.duplicated(["TF_id", "model", "time_idx"]).any():
        raise ValueError("Duplicate cached TF stage profiles.")
    expressions = []
    for model in report["methods"]:
        part = table.loc[table.model.eq(model)]
        if len(part) != len(tfs) * 15 or set(part.TF_id) != set(tfs.FBgn_ID):
            raise ValueError(f"Incomplete cached profiles: {model}.")
        means = part.pivot(
            index="time_idx", columns="TF_id", values="mean_expression"
        ).reindex(index=range(15), columns=tfs.FBgn_ID)
        expressions.append(means.to_numpy(float))
        counts = (
            part.pivot(index="time_idx", columns="TF_id", values="n_cells")
            .reindex(index=range(15), columns=tfs.FBgn_ID)
            .to_numpy()
        )
        if not np.array_equal(
            counts,
            np.tile(np.asarray(report["stage_cell_counts"])[:, None], (1, len(tfs))),
        ):
            raise ValueError(
                "Cached stage cell counts differ from the analysis report."
            )
    if any(
        not np.allclose(expressions[0], values, equal_nan=True)
        for values in expressions[1:]
    ):
        raise ValueError("Methods use different cached expression profiles.")
    return expressions[0]


def ablation_profiles(root, models, tfs, aliases, report):
    arrays, sources = {}, {}
    reference = wf.read_json(
        PROJECT / "docs/reference/mondo_seed2025/reference_manifest.json"
    )
    settings = report["sources"]["DRAFT"]["export_report"]["settings"]
    if any(settings.get(k) != reference["network_settings"][k] for k in NETWORK_FIELDS):
        raise ValueError(
            "Saved DRAFT selection differs from the frozen MONDO settings."
        )
    expected_times = sorted(
        float(re.fullmatch(r"TF_TG_dense_t(.+)\.tsv", name)[1])
        + report["sources"]["DRAFT"]["time_offset"]
        for name in report["sources"]["DRAFT"]["file_sha256"]
        if name.startswith("TF_TG_dense_t")
    )
    for model, variant, gru, encoder in [
        ("DRAFT w/o GRU", "no_gru", False, "gat"),
        ("Linear + GRU", "linear_gru", True, "linear"),
        ("separate GAT", "separate_gat", False, "gat"),
    ]:
        if model not in models:
            continue
        directory = root / "seed_2025" / variant
        meta = wf.read_json(directory / "analysis_report.json")
        if (
            meta["seed"] != 2025
            or meta["variant"] != variant
            or bool(meta["use_gru"]) != gru
            or meta.get("spatial_encoder", "gat") != encoder
        ):
            raise ValueError(f"Wrong seed/architecture: {directory}.")
        if variant == "separate_gat" and (
            not meta.get("temporal_loss_disabled", False)
            or float(meta.get("lambda_smooth_used", -1)) != 0
            or any(p != -1 for p in meta.get("training_parent", [0]))
        ):
            raise ValueError(f"separate GAT retains temporal training: {directory}.")
        print(f"Reading saved {model} selected networks: {directory}", flush=True)
        values, times, _dense, source = wf.draft_profiles(
            directory / "network_dense", tfs, aliases
        )
        if len(times) != len(expected_times) or not np.allclose(
            times, expected_times, rtol=0, atol=1e-8
        ):
            raise ValueError("DRAFT/ablation dense time grids differ.")
        if any(
            source["export_report"]["settings"].get(k) != settings[k]
            for k in NETWORK_FIELDS
        ):
            raise ValueError("DRAFT/ablation network-selection settings differ.")
        smooth = np.stack(
            [
                wf.gaussian_smooth_1d(
                    values[:, i], np.arange(15), report["settings"]["sigma_stage"]
                )
                for i in range(len(tfs))
            ],
            axis=1,
        )
        arrays[model] = (values, smooth)
        sources[model] = {
            **source,
            "analysis_report_sha256": wf.digest(directory / "analysis_report.json"),
        }
    return arrays, sources


def stage_set(value):
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, (list, np.ndarray)):
        return [int(v) for v in value]
    return []


def annotate(table, threshold, labels):
    table = table.copy()
    for kind in ["expression", "network"]:
        table[f"{kind}_peak_label"] = table[f"{kind}_peak_stage"].map(
            lambda v: labels[int(v)] if pd.notna(v) else pd.NA
        )
        for field in ["maximal_stages", "near_max_stages"]:
            table[f"{kind}_{field}_labels"] = table[f"{kind}_{field}"].map(
                lambda v: "; ".join(labels[i] for i in stage_set(v)) or pd.NA
            )
    table["far_offset_detected"] = [
        "NA"
        if row.status != "evaluable"
        else "yes"
        if row.conservative_abs_offset >= threshold
        else "no"
        for row in table.itertuples(index=False)
    ]
    table["offset_threshold_stages"] = threshold
    return table


def wide_table(table, models, threshold, labels):
    table = annotate(table.loc[table.profile.eq("smoothed")], threshold, labels)
    reference = table.loc[table.model.eq("DRAFT")].set_index("TF_id")
    lookups = {m: table.loc[table.model.eq(m)].set_index("TF_id") for m in models}
    rows = []
    fields = [
        "network_peak_label",
        "network_peak_stage",
        "network_maximal_stages_labels",
        "network_near_max_stages_labels",
        "argmax_signed_offset",
        "argmax_abs_offset",
        "conservative_abs_offset",
        "direction",
        "far_offset_detected",
        "status",
        "network_endpoint_peak",
        "network_relative_dynamic_range",
    ]
    for tf, draft in reference.iterrows():
        row = {
            "TF": draft.TF_name,
            "TF_id": tf,
            "time_domain": draft.time_domain,
            "stage_range": f"{labels[int(draft.stage_min)]} to {labels[int(draft.stage_max)]}",
            "offset_threshold_stages": threshold,
        }
        for key in [
            "expression_peak_label",
            "expression_peak_stage",
            "expression_maximal_stages_labels",
            "expression_near_max_stages_labels",
            "expression_endpoint_peak",
        ]:
            row[key] = draft[key]
        for model in models:
            other = lookups[model].loc[tf]
            for key in ["expression_peak_stage", "expression_near_max_stages"]:
                a, b = draft[key], other[key]
                if not ((pd.isna(a) and pd.isna(b)) or a == b):
                    raise ValueError(
                        "Expression peak definitions differ across methods."
                    )
            slug = SLUGS[model]
            row.update({f"{slug}_{key}": other[key] for key in fields})
            if model == "DRAFT":
                continue
            comparison = "NA"
            distance = pd.NA
            if draft.status == other.status == "evaluable":
                a = np.asarray(stage_set(draft.network_near_max_stages))
                b = np.asarray(stage_set(other.network_near_max_stages))
                distance = int(np.abs(a[:, None] - b).min())
                if draft.far_offset_detected == "yes":
                    if other.far_offset_detected == "no":
                        comparison = "no"
                    elif draft.direction in [
                        "network_earlier",
                        "network_later",
                    ] and other.direction in ["network_earlier", "network_later"]:
                        comparison = (
                            "yes" if draft.direction == other.direction else "no"
                        )
            row[f"{slug}_same_direction_far_as_DRAFT"] = comparison
            row[f"{slug}_peak_region_distance_to_DRAFT"] = distance
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["DRAFT_conservative_abs_offset", "TF_id"],
        ascending=[False, True],
        na_position="last",
    )


def compact_candidates(candidates):
    """One row per DRAFT hit, with Y/N detection and separate detailed coverage."""
    result = candidates[
        [
            "TF",
            "TF_id",
            "DRAFT_network_peak_label",
            "expression_peak_label",
            "DRAFT_argmax_abs_offset",
            "DRAFT_conservative_abs_offset",
            "DRAFT_direction",
        ]
    ].rename(
        columns={
            "TF_id": "TF_ID",
            "DRAFT_network_peak_label": "DRAFT_network_peak",
            "expression_peak_label": "Expression_peak",
            "DRAFT_argmax_abs_offset": "Gap_stages",
            "DRAFT_conservative_abs_offset": "Conservative_gap_stages",
            "DRAFT_direction": "Network_timing",
        }
    )
    for key in ["Gap_stages", "Conservative_gap_stages"]:
        result[key] = result[key].astype("Int64")
    for model in MODELS[1:]:
        slug = SLUGS[model]
        field = f"{slug}_far_offset_detected"
        if field not in candidates:
            continue
        result[model] = ["Y" if value == "yes" else "N" for value in candidates[field]]
    result["Time_window"] = candidates.stage_range
    return result


def export(args):
    source = choose_analysis(args.analysis, args.methods)
    report_path = source / "analysis_report.json"
    report = wf.read_json(report_path)
    tf_path = (args.tf_list or Path(report["tf_list"])).expanduser().resolve()
    if wf.digest(tf_path) != report["tf_list_sha256"]:
        raise ValueError("TF list differs from the saved peak analysis.")
    tfs, aliases = wf.load_tfs(tf_path)
    if len(tfs) != report["n_tf_list"]:
        raise ValueError("Curated TF count differs from the analysis report.")
    expression = expression_from_cache(source / "tf_stage_profiles.tsv.gz", tfs, report)
    arrays, ablation_sources = ablation_profiles(
        args.ablation_root, args.methods, tfs, aliases, report
    )
    domain = report.get(
        "primary_time_domain",
        {"name": "full_trajectory", "stage_indices": list(range(15))},
    )
    stages = np.asarray(domain["stage_indices"], dtype=int)
    if (
        len(stages) < 3
        or stages[0] < 0
        or stages[-1] > 14
        or not np.array_equal(stages, np.arange(stages[0], stages[-1] + 1))
    ):
        raise ValueError(
            "Primary stage coverage must contain at least three contiguous anchors."
        )
    domains = [(domain["name"], stages, source / "tf_peak_offsets.tsv", args.methods)]
    if domain["name"] != "full_trajectory":
        domains.append(
            (
                "full_trajectory",
                np.arange(15),
                source / "full_trajectory_tf_peak_offsets.tsv",
                [m for m in args.methods if m != "Dictys"],
            )
        )
    main_name = (
        domain["name"]
        if getattr(args, "main_time_domain", "full_trajectory") == "source"
        else "full_trajectory"
    )
    if "Dictys" not in args.methods and main_name == "full_trajectory":
        domains = [entry for entry in domains if entry[0] == "full_trajectory"]
    labels_path = PROJECT / "configs/germline_stage_labels.json"
    labels = wf.read_json(labels_path)["labels"]
    out = args.output_root.resolve() / f"candidate_table_s2025_{uuid.uuid4().hex[:10]}"
    out.mkdir(parents=True, exist_ok=False)
    inputs = [report_path, tf_path, source / "tf_stage_profiles.tsv.gz", labels_path]
    counts, files = [], []
    for name, stages, path, models in domains:
        base_models = [m for m in models if m in wf.MODELS]
        records = [saved_offsets(path, base_models, name, stages, tfs)]
        inputs.append(path)
        settings = report["settings"]
        for model in models:
            if model not in arrays:
                continue
            records.append(
                wf.evaluate_offsets(
                    tfs,
                    expression,
                    {model: arrays[model][0]},
                    {model: arrays[model][1]},
                    {model: np.full(len(tfs), "no_selected_edges")},
                    [model],
                    stages,
                    argparse.Namespace(
                        min_relative_range=settings["min_relative_dynamic_range"],
                        near_max_fraction=settings["near_max_fraction"],
                    ),
                    name,
                )
            )
        long = pd.concat(records, ignore_index=True)
        # Validate the shared expression against cached paired-cell means, not just other methods.
        for record in long.loc[
            long.model.eq("DRAFT") & long.profile.eq("smoothed")
        ].itertuples(index=False):
            i = tfs.index[tfs.FBgn_ID.eq(record.TF_id)][0]
            expected = wf.peak_details(
                expression[stages, i],
                settings["min_relative_dynamic_range"],
                settings["near_max_fraction"],
                stage_indices=stages,
            )
            if expected["status"] != record.expression_status or expected.get(
                "peak_stage", None
            ) != (
                int(record.expression_peak_stage)
                if pd.notna(record.expression_peak_stage)
                else None
            ):
                raise ValueError(
                    "Cached peak table and paired-cell expression means differ."
                )
        wide = wide_table(long, models, args.threshold, labels)
        candidates = wide.loc[wide.DRAFT_far_offset_detected.eq("yes")]
        target = out / name
        target.mkdir()
        for filename, table in [
            ("all_TF_comparison.tsv", wide),
            ("DRAFT_far_candidates.tsv", candidates),
            ("all_models_peak_details.tsv", annotate(long, args.threshold, labels)),
            (
                "offset_counts.tsv",
                wf.scoped_summary(
                    long, models, len(tfs), sorted({2, 3, 4, args.threshold})
                ).assign(
                    validation_only=any(
                        item.get("validation_only", False)
                        for item in report["sources"].values()
                    )
                ),
            ),
        ]:
            table.to_csv(
                target / filename, sep="\t", index=False, na_rep="NA", encoding="utf-8"
            )
            files.append(target / filename)
        if getattr(args, "plot_counts", False):
            from plot_counts import render

            figure_dir = render(
                target / "offset_counts.tsv",
                target / "count_figures",
                threshold=args.threshold,
                models=models,
            )
            files.extend(p for p in figure_dir.iterdir() if p.is_file())
            raw_figure_dir = render(
                target / "offset_counts.tsv",
                target / "count_figures_raw",
                profile="raw",
                threshold=args.threshold,
                models=models,
            )
            files.extend(p for p in raw_figure_dir.iterdir() if p.is_file())
        candidates.to_csv(
            target / "DRAFT_far_candidates.csv",
            index=False,
            na_rep="NA",
            encoding="utf-8-sig",
        )
        files.append(target / "DRAFT_far_candidates.csv")
        compact = compact_candidates(candidates)
        filenames = [f"DRAFT_TF_peak_shifts_{name}"]
        if name == main_name:
            filenames.append("DRAFT_TF_peak_shifts")
            primary_compact = compact
        for filename in filenames:
            for suffix, sep in [("csv", ","), ("tsv", "\t")]:
                compact.to_csv(
                    out / f"{filename}.{suffix}",
                    sep=sep,
                    index=False,
                    na_rep="NA",
                    encoding="utf-8-sig",
                )
                files.append(out / f"{filename}.{suffix}")
        for model in models:
            part = long.loc[long.model.eq(model) & long.profile.eq("smoothed")]
            mask = part.status.eq("evaluable") & part.conservative_abs_offset.ge(
                args.threshold
            )
            counts.append(
                {
                    "time_domain": name,
                    "model": model,
                    "n_TF_list": len(tfs),
                    "n_evaluable": int(part.status.eq("evaluable").sum()),
                    "n_far": int(mask.sum()),
                    "n_DRAFT_far": len(candidates),
                    "n_also_far_among_DRAFT_far": int(
                        candidates[f"{SLUGS[model]}_far_offset_detected"]
                        .eq("yes")
                        .sum()
                    ),
                }
            )
        print(
            f"{name}: {len(candidates)} DRAFT candidates at >= {args.threshold:g} stages; {labels[stages[0]]} to {labels[stages[-1]]}",
            flush=True,
        )
    summary = pd.DataFrame(counts)
    summary.to_csv(out / "candidate_counts.tsv", sep="\t", index=False)
    files.append(out / "candidate_counts.tsv")
    (out / "table_report.json").write_text(
        json.dumps(
            {
                "seed": 2025,
                "validation_only": any(
                    source.get("validation_only", False)
                    for source in report["sources"].values()
                ),
                "analysis_source": str(source),
                "methods": args.methods,
                "main_time_domain": main_name,
                "n_TF_list": len(tfs),
                "threshold_stages": args.threshold,
                "peak_settings": report["settings"],
                "ablation_sources": ablation_sources,
                "inputs_sha256": {str(p): wf.digest(p) for p in inputs},
                "files_sha256": {str(p.relative_to(out)): wf.digest(p) for p in files},
                "code_sha256": {
                    p.name: wf.digest(p) for p in [Path(__file__), HERE / "workflow.py"]
                },
                "definitions": {
                    "main_file": "DRAFT_TF_peak_shifts.csv defaults to the full GSC-G14 trajectory; --main-time-domain source selects the source analysis window instead. Only methods evaluated in that domain appear as comparator columns, so Dictys is omitted from the full-trajectory main table. Y means also detected as far irrespective of direction. N includes unevaluable profiles and is not a tested-negative label. Full direction/coverage/status fields remain in per-domain details.",
                    "profiles": "Saved selected-network centrality profiles, not the unthresholded MONDO target-rank workflow. SCENIC+ is the published stage eRegulon target-count profile, not newly fitted TF2G.",
                    "signed_offset": "network peak stage minus mean RNA expression peak stage; negative means network earlier. Earliest tied argmax is descriptive only; all ties are retained.",
                    "far_offset_detected": "yes when evaluable and minimum distance between network/expression near-maximum sets is >= threshold (near-max fraction is inherited from source settings); no when evaluable but below threshold; NA when not evaluable.",
                    "same_direction_far_as_DRAFT": "yes requires both profiles far and the same unambiguous earlier/later direction; no for an evaluable method below threshold or opposite direction; NA for unevaluable data, mixed direction or a non-far DRAFT reference.",
                    "peak_region_distance_to_DRAFT": "Minimum stage distance between the two methods' network near-maximum sets; reported separately because same-direction offset does not guarantee the same network peak window.",
                    "domains": "All comparisons within a table share time coverage. Full-trajectory tables exclude Dictys; domain-specific maxima and candidate lists must not be mixed.",
                    "NA": "Missing, unsupported, all-zero, near-flat or incomplete profiles cannot supply an evaluable peak; status columns retain the reason. These are not negative detections.",
                    "scope": "Descriptive peak candidates; endpoint flags retained. Stage units are ordinal. No retraining or experimental-label tuning; method readouts and target pools differ.",
                    "count_plots": "offset_counts.tsv uses every TF in the input list, not only DRAFT-selected candidates, at conservative thresholds 2/3/4 and the requested primary threshold. Optional figures use the same per-domain records and include both ablations.",
                },
            },
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    print("\nDRAFT TF peak-shift list (all candidates):", flush=True)
    print(primary_compact.to_string(index=False), flush=True)
    print(f"\nOpen this file: {out / 'DRAFT_TF_peak_shifts.csv'}", flush=True)
    print(f"TF candidate tables: {out}", flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis",
        default="latest",
        help="Completed tf_peak_offsets directory, or latest matching requested methods.",
    )
    parser.add_argument("--methods", choices=MODELS, nargs="+", default=DEFAULT_MODELS)
    parser.add_argument(
        "--ablation-root", type=Path, default=PROJECT / "result/germ/mondo_gru_ablation"
    )
    parser.add_argument(
        "--tf-list",
        type=Path,
        help="Optional relocated TF_filtered.tsv; original hash must match.",
    )
    parser.add_argument("--threshold", type=float, default=3)
    parser.add_argument(
        "--plot-counts",
        action="store_true",
        help="Also plot per-method absolute counts and fraction panels at 2/3/4 stages.",
    )
    parser.add_argument(
        "--main-time-domain",
        choices=["full_trajectory", "source"],
        default="full_trajectory",
        help="Main list defaults to all 15 stages; source reproduces the matched source window.",
    )
    parser.add_argument("--cpus", type=int, default=8)
    parser.add_argument("--output-root", type=Path, default=HERE / "runs")
    args = parser.parse_args()
    if (
        "DRAFT" not in args.methods
        or len(set(args.methods)) != len(args.methods)
        or not 1 <= args.cpus <= 8
        or not np.isfinite(args.threshold)
        or args.threshold <= 0
    ):
        parser.error(
            "Require DRAFT, unique methods, 1-8 CPUs, and a positive finite threshold."
        )
    for key in [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]:
        os.environ[key] = str(args.cpus)
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=args.cpus):
        export(args)


if __name__ == "__main__":
    main()
