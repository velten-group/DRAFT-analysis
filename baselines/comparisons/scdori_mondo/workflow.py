"""Isolated scDoRI shared-prior training and MONDO temporal readout."""

from __future__ import annotations

import argparse
import dataclasses
import logging
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scdori_inputs import derive_diagnostic, load_run, prepare_run, report_preprocessing
from scdori_native import (
    HERE,
    PIN,
    check_outputs,
    comparison_helpers,
    digest,
    effective_mondo_edges,
    load_native,
    read_json,
    restore,
    safe_run_path,
    write_json,
)
from scipy.interpolate import PchipInterpolator
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader, TensorDataset

LOG = logging.getLogger(__name__)


def resolve_run(path: Path) -> Path:
    patterns = {
        "latest-diagnostic": ("shared_unsigned_s2025_*",),
        "latest-native": ("native_correlation_s2025_*", "shared_prior_s2025_*"),
    }
    if str(path) in patterns:
        mode = (
            "shared_unsigned"
            if str(path) == "latest-diagnostic"
            else "native_correlation"
        )
        candidates = []
        for pattern in patterns[str(path)]:
            for candidate in (HERE / "runs").glob(pattern):
                manifest_path = candidate / "manifest.json"
                if not manifest_path.is_file():
                    continue
                cfg = read_json(manifest_path)["configuration"]
                if (
                    cfg.get("seed") == 2025
                    and not cfg.get("validation_only")
                    and cfg.get("binding_mode", "native_correlation") == mode
                ):
                    candidates.append(candidate)
        if not candidates:
            raise FileNotFoundError(f"No prepared {path} run found.")
        path = max(candidates, key=lambda p: (p / "manifest.json").stat().st_mtime_ns)
        print(f"Selected run: {path}", flush=True)
    return safe_run_path(path)


def context(run: Path, device_name: str):
    manifest = load_run(run)
    cfg = manifest["configuration"]
    api = load_native(Path(cfg["source_root"]))
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Prepare on CPU and train on a CUDA machine."
        )
    device = torch.device(device_name)
    api["_core.utils"].set_seed(cfg["seed"])
    tc = api["_core.config"].TrainConfig()
    tc.data_dir, tc.output_subdir = run, "data"
    tc.random_seed = cfg["seed"]
    for key, value in cfg["training"].items():
        if not hasattr(tc, key):
            raise ValueError(f"Unknown native training setting: {key}")
        setattr(tc, key, value)
    rna, atac, distance, act, rep = api["_core.data_io"].load_scdori_inputs(tc)
    if not rna.obs_names.equals(atac.obs_names):
        raise ValueError("Paired cells or their order differ.")
    tf_indices = np.flatnonzero(rna.var.gene_type.eq("TF"))
    target_indices = np.flatnonzero(rna.var.gene_type.eq("HVG"))
    batches = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit_transform(
        rna.obs[[tc.batch_col]]
    )
    num_cells = np.ones((rna.n_obs, 1), dtype=np.float32)
    split = int(np.floor(rna.n_obs * 0.8))
    if split % tc.batch_size_cell == 1:
        raise ValueError(
            "Singleton training batch would break native BatchNorm. Use another batch size in a new run."
        )
    train_loader = DataLoader(
        TensorDataset(torch.arange(split)), batch_size=tc.batch_size_cell, shuffle=False
    )
    eval_loader = DataLoader(
        TensorDataset(torch.arange(split, rna.n_obs)),
        batch_size=tc.batch_size_cell,
        shuffle=False,
    )
    all_loader = DataLoader(
        TensorDataset(torch.arange(rna.n_obs)),
        batch_size=tc.batch_size_cell_prediction,
        shuffle=False,
    )
    model = (
        api["_core.models"]
        .scDoRI(
            device,
            rna.n_vars,
            atac.n_vars,
            len(tf_indices),
            tc.num_topics,
            batches.shape[1],
            tc.dim_encoder1,
            tc.dim_encoder2,
        )
        .to(device)
    )
    fixed = (distance > 0).to(torch.float32)
    api["_core.models"].initialize_scdori_parameters(
        model,
        distance.to(device),
        fixed.to(device),
        act.to(device),
        rep.to(device),
        phase="warmup",
    )
    return {
        "manifest": manifest,
        "cfg": cfg,
        "api": api,
        "tc": tc,
        "device": device,
        "model": model,
        "rna": rna,
        "atac": atac,
        "distance": distance,
        "fixed": fixed,
        "act": act,
        "rep": rep,
        "tf_indices": tf_indices,
        "targets": target_indices,
        "batches": batches,
        "num_cells": num_cells,
        "train_loader": train_loader,
        "eval_loader": eval_loader,
        "all_loader": all_loader,
    }


def native_args(ctx):
    return [
        ctx[k]
        for k in [
            "model",
            "device",
            "train_loader",
            "eval_loader",
            "rna",
            "atac",
            "num_cells",
            "tf_indices",
            "batches",
            "tc",
        ]
    ]


def train(run: Path, device_name: str, *, allow_missing_mondo: bool = False) -> None:
    support = report_preprocessing(run)
    if not support["can_estimate_mondo_edges"]:
        if not allow_missing_mondo:
            raise RuntimeError(
                "MONDO has no binding evidence after preprocessing. Full training cannot restore its frozen zero binding columns. To train the joint model for other TFs while retaining these inputs, pass --allow-missing-mondo. Keep mondo_preprocessing_report.json; the separately named shared_unsigned run is an alternative input diagnostic."
            )
        LOG.warning(
            "Continuing joint training for other TFs with unchanged binding inputs. "
            "MONDO has no binding support and its GRN edges remain zero."
        )
    ctx = context(run, device_name)
    api, tc, model = ctx["api"], ctx["tc"], ctx["model"]
    runtime = {
        "device": device_name,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "upstream_commit": PIN,
    }
    runtime_path = run / "runtime.json"
    if runtime_path.exists() and read_json(runtime_path) != runtime:
        raise ValueError(
            "Resume requires the same source, numerical environment and device."
        )
    if not runtime_path.exists():
        write_json(runtime_path, runtime)
    phase1_record = run / "phase1_complete.json"
    if phase1_record.exists():
        record = read_json(phase1_record)
        check_outputs(record, run)
        restore(model, run / record["checkpoint"], ctx["device"])
        print("Resume: completed scDoRI phase 1 retained.", flush=True)
    else:
        attempt = run / f"phase1_{uuid.uuid4().hex[:8]}"
        tc.weights_folder_scdori = str(attempt)
        api["_core.train_scdori"].train_scdori_phases(*native_args(ctx))
        checkpoint = attempt / "best_scdori_best_eval.pth"
        restore(model, checkpoint, ctx["device"])
        name = str(checkpoint.relative_to(run))
        write_json(
            phase1_record, {"checkpoint": name, "sha256": {name: digest(checkpoint)}}
        )
    phase2_record = run / "grn_complete.json"
    if phase2_record.exists():
        check_outputs(read_json(phase2_record), run)
        print("Resume: completed scDoRI GRN phase retained.", flush=True)
        return
    attempt = run / f"grn_{uuid.uuid4().hex[:8]}"
    tc.weights_folder_grn = str(attempt)
    api["_core.models"].initialize_scdori_parameters(
        model, ctx["distance"], ctx["fixed"], ctx["act"], ctx["rep"], phase="grn"
    )
    api["_core.train_grn"].train_model_grn(*native_args(ctx))
    checkpoint = attempt / "best_scdori_best_eval.pth"
    restore(model, checkpoint, ctx["device"])
    name = str(checkpoint.relative_to(run))
    settings = dataclasses.asdict(tc)
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in settings.items()}
    write_json(
        phase2_record,
        {
            "checkpoint": name,
            "sha256": {name: digest(checkpoint)},
            "native_training_config": settings,
        },
    )
    print(f"Training complete: {run}", flush=True)


def extract(run: Path, device_name: str):
    record_path = run / "grn_complete.json"
    if not record_path.is_file():
        raise RuntimeError(
            "GRN training is incomplete; no biological curve can be generated."
        )
    record = read_json(record_path)
    check_outputs(record, run)
    ctx = context(run, device_name)
    model, api = ctx["model"], ctx["api"]
    restore(model, run / record["checkpoint"], ctx["device"])
    model.eval()
    args = [
        ctx[k]
        for k in [
            "model",
            "device",
            "all_loader",
            "rna",
            "atac",
            "num_cells",
            "tf_indices",
            "batches",
        ]
    ]
    theta = api["_core.evaluation"].get_latent_topics(*args)
    if not np.isfinite(theta).all() or not np.allclose(theta.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Invalid topic weights.")
    # Match the training recipe: ordered train prefix corresponds to rows 0..split.
    topic_tf = api["_core.train_grn"].get_tf_expression(
        ctx["tc"].tf_expression_mode,
        model,
        ctx["device"],
        ctx["train_loader"],
        ctx["rna"],
        ctx["atac"],
        ctx["num_cells"],
        ctx["tf_indices"],
        ctx["batches"],
        ctx["tc"],
    )
    tfs = ctx["rna"].var_names[ctx["tf_indices"]].tolist()
    mondo = tfs.index(ctx["cfg"]["gene_id"])
    edges = effective_mondo_edges(model, topic_tf, mondo, ctx["targets"])
    values = []
    for start in range(0, len(theta), 512):
        # Mix signed topic networks, then take weighted out-degree. This retains
        # cancellation of opposite regulation; abs before mixing is different.
        values.extend(np.abs(theta[start : start + 512] @ edges).sum(axis=1).tolist())
    cells = pd.DataFrame(
        {
            "cell_id": ctx["rna"].obs_names,
            "time": ctx["rna"].obs.stage.to_numpy(),
            "pseudotime": ctx["rna"].obs.pseudotime.to_numpy(),
            "weighted_outdegree": values,
        }
    )
    stages = (
        cells.groupby("time")
        .weighted_outdegree.agg(mean_outdegree="mean", sd_cells="std", n_cells="size")
        .reset_index()
    )
    if not np.array_equal(stages.time, np.arange(15)):
        raise ValueError("Stage profile must cover G0-G14.")
    grid = np.linspace(0, 14, ctx["cfg"]["profile"]["points"])
    times = stages.time.to_numpy(dtype=float)
    sigma = ctx["cfg"]["profile"]["sigma_stage"]
    weights = np.exp(-0.5 * ((times[:, None] - times[None, :]) / sigma) ** 2)
    weights /= weights.sum(axis=1, keepdims=True)
    smooth = weights @ stages.mean_outdegree.to_numpy()
    display = PchipInterpolator(times, smooth)(grid)
    maximum = float(display.max())
    peak = int(np.argmax(display)) if maximum > 0 else None
    profile = pd.DataFrame(
        {
            "time": grid,
            "weighted_outdegree": display,
            "relative_profile": display / maximum if maximum > 0 else 0,
        }
    )
    support = pd.read_csv(run / "data/binding_support.tsv", sep="\t")
    report = {
        "seed": ctx["cfg"]["seed"],
        "binding_mode": ctx["cfg"].get("binding_mode", "native_correlation"),
        "metric": "sum_g abs(sum_k theta_ck * effective_MONDO_target_edge_kg), averaged over stage cells; non-TF HVG targets only",
        "filtering": "unthresholded model edges; native permutation significance filtering was not run",
        "smoothing": "Gaussian smoothing of equal-weight stage means (sigma_stage configured before training), then PCHIP; post hoc display only",
        "raw_peak_stage": int(stages.time.iloc[stages.mean_outdegree.argmax()])
        if stages.mean_outdegree.max() > 0
        else None,
        "smoothed_peak_time": float(grid[peak]) if peak is not None else None,
        "smoothed_peak_is_early": bool(grid[peak] <= ctx["cfg"]["profile"]["early_end"])
        if peak is not None
        else None,
        "normalization_maximum": maximum,
        "stage_readout_minimum": float(stages.mean_outdegree.min()),
        "stage_readout_maximum": float(stages.mean_outdegree.max()),
        "stage_relative_dynamic_range": float(
            (stages.mean_outdegree.max() - stages.mean_outdegree.min())
            / stages.mean_outdegree.max()
        )
        if stages.mean_outdegree.max() > 0
        else 0.0,
        "mondo_topic_tf_expression": topic_tf[:, mondo]
        .detach()
        .cpu()
        .numpy()
        .astype(float)
        .tolist(),
        "mondo_binding_support": support.loc[
            support.tf_id.eq(ctx["cfg"]["gene_id"])
        ].to_dict("records"),
        "zero_readout_caution": "a zero can result from motif/correlation support, topic TF-expression clamping, or learned weights; it is not proof of biological inactivity",
        "training": "topic-based, paired RNA/ATAC; no time/stage/order information enters model training",
        "comparison": "shared-prior input adapter, not the fully native de-novo motif pipeline or the published permutation-filtered activity score",
    }
    return cells, stages, profile, edges, report, ctx["cfg"]


def read_dictys_profile(path: Path | None, scdori_run: Path):
    if path is None:
        return None, None
    if path == Path("auto"):
        root = HERE.parent / "dictys_mondo/runs"
        candidates = sorted(
            root.glob("shared_prior_s2025_*/figures_*/mondo_dictys_profile.tsv"),
            key=lambda p: p.stat().st_mtime_ns,
            reverse=True,
        )
        if not candidates:
            print(
                "No completed Dictys profile yet; drawing scDoRI with existing five curves.",
                flush=True,
            )
            return None, None
        path = candidates[0]
    meta = read_json(path.parent / "profile_report.json")
    if meta.get("dictys_seed") != 2025:
        raise ValueError("Dictys comparison must use seed 2025.")
    dm = read_json(path.parent.parent / "manifest.json")
    sm = load_run(scdori_run)
    for filename, key in [
        ("A_rna_focus.h5ad", "rna"),
        ("B_atac_focus.h5ad", "atac"),
        ("cell_annotation.csv", "annotation"),
        ("TF_RE_top3pct_edges.tsv", "tf_re"),
        ("RE_TG_100kb.tsv", "re_tg"),
    ]:
        if (
            sm["input_sources"][filename]["sha256"]
            != dm["input_sources"][key]["sha256"]
        ):
            raise ValueError(f"Dictys/scDoRI source data differ: {filename}")
    table = pd.read_csv(path, sep="\t")
    if (
        not {"time", "relative_profile"}.issubset(table.columns)
        or not table.time.is_monotonic_increasing
        or not table.time.is_unique
    ):
        raise ValueError("Invalid saved Dictys profile.")
    finite = table.relative_profile.dropna()
    if not finite.between(0, 1.001).all() or not table.time.between(0, 14).all():
        raise ValueError(
            "Dictys profile must retain its normalized G0-G14 coordinates."
        )
    return table, {"path": str(path.resolve()), "sha256": digest(path)}


def plot(
    run: Path, device_name: str, existing: Path | None, dictys_path: Path | None = None
) -> Path:
    helpers = comparison_helpers()
    existing = helpers.resolve_comparison(existing)
    curves, sources = helpers.load_comparison(existing)
    cells, stages, profile, edges, report, cfg = extract(run, device_name)
    dictys, dictys_source = read_dictys_profile(dictys_path, run)
    out = run / f"figures_{uuid.uuid4().hex[:10]}"
    out.mkdir(exist_ok=False)
    cells.to_csv(out / "mondo_scdori_cells.tsv", sep="\t", index=False)
    stages.to_csv(out / "mondo_scdori_stages.tsv", sep="\t", index=False)
    profile.to_csv(out / "mondo_scdori_profile.tsv", sep="\t", index=False)
    curves.to_csv(out / "existing_comparison_curves.tsv", sep="\t", index=False)
    np.save(out / "mondo_topic_edges.npy", edges)
    report["comparison_sources"] = sources
    write_json(out / "profile_report.json", report)
    if dictys is not None:
        dictys.to_csv(out / "existing_dictys_profile.tsv", sep="\t", index=False)
        report["dictys_source"] = dictys_source
        write_json(out / "profile_report.json", report)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 12,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    ):
        fig, ax = plt.subplots(figsize=(11, 7))
        for key, (label, color, style) in helpers.STYLES.items():
            if key in curves:
                ax.plot(
                    curves.time, curves[key], color=color, ls=style, lw=2.5, label=label
                )
        ax.plot(
            profile.time,
            profile.relative_profile,
            color="#CC79A7",
            lw=2.8,
            label="scDoRI core (unsigned-prior diagnostic)"
            if cfg.get("binding_mode") == "shared_unsigned"
            else "scDoRI effective out-degree (shared prior)",
        )
        if dictys is not None:
            ax.plot(
                dictys.time,
                dictys.relative_profile,
                color="#333333",
                lw=2.6,
                label="Dictys weighted out-degree (shared prior)",
            )
        helpers.set_stage_axis(ax)
        ax.set(
            xlim=(-0.2, 14.2),
            ylim=(-0.025, 1.06),
            xlabel="Germline trajectory",
            ylabel="Relative profile (each maximum = 1)",
        )
        ax.spines[["top", "right"]].set_visible(False)
        title = (
            "SYNTHETIC SOFTWARE TEST: not a biological result"
            if cfg.get("validation_only")
            else f"MONDO temporal profiles (DRAFT seed 2025; scDoRI seed {cfg['seed']})"
        )
        fig.suptitle(title, y=0.97, fontsize=17)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.91),
            ncol=2,
            frameon=False,
        )
        fig.text(
            0.5,
            0.025,
            "scDoRI: shared prior; unthresholded effective edges; stage means; post hoc smoothing",
            ha="center",
            fontsize=9,
        )
        fig.subplots_adjust(
            left=0.1, right=0.97, bottom=0.24, top=0.68 if dictys is not None else 0.73
        )
        for suffix in ["png", "pdf", "svg"]:
            fig.savefig(
                out / f"mondo_with_scdori.{suffix}",
                dpi=cfg["profile"]["dpi"],
                facecolor="white",
            )
        plt.close(fig)
    (out / "legend.md").write_text(
        "scDoRI is trained on paired germline RNA–ATAC counts using a shared DRAFT prior adapter, all expressed candidate TFs and selected HVG targets. Official model and training functions are used. Binary TF–RE support replaces native FIMO scores; official correlation-based activator/repressor embeddings and RNA-based metacells are retained. Zero-evidence repressor columns are kept zero instead of the released scaler's constant −1 artifact. TF genes receive additional coordinate-based cis links required for scDoRI RNA reconstruction. Topic/expression cell correspondence is preserved by a seeded one-time cell permutation and an ordered training prefix. MONDO's effective model edges are mixed with each cell's topic weights; the sum of their absolute weights over selected non-TF target genes is averaged within each stage. Native permutation significance filtering is not run, so this readout is not the published filtered regulon/activity statistic. Stage means are Gaussian-smoothed at sigma=0.5 G units and connected by PCHIP. Smoothing is post hoc: training does not use pseudotime or stage. Curves are independently maximum-normalized and do not have shared activity units. Raw counts, support diagnostics and original stage means are preserved.\n",
        encoding="utf-8",
    )
    if cfg.get("binding_mode") == "shared_unsigned":
        with (out / "legend.md").open("a", encoding="utf-8") as stream:
            stream.write(
                "\nThis figure uses the UNSIGNED SHARED-PRIOR DIAGNOSTIC, not native correlation-filtered binding inputs. Every allowed TF–RE link is supplied to both the activator and repressor branches, and the model learns their effective weights. The override applies to every TF; no MONDO-specific peak or target is rescued. The default preprocessing report is preserved separately. Conclusions from this diagnostic apply to the model core under this input condition, not to fully native scDoRI.\n"
            )
    print(
        f"Figures: {out}\nRaw peak: {report['raw_peak_stage']}; smoothed peak: {report['smoothed_peak_time']}",
        flush=True,
    )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ["prepare", "prepare-both", "all"]:
        p = sub.add_parser(command)
        p.add_argument("--config", type=Path, default=HERE / "config.yaml")
        p.add_argument("--data-root", type=Path)
        p.add_argument("--upstream", type=Path)
        p.add_argument("--seed", type=int)
        if command == "all":
            p.add_argument("--device", default="cuda:0")
            p.add_argument("--existing-curves", type=Path, default=Path("auto"))
            p.add_argument("--dictys-profile", type=Path, default=Path("auto"))
    for command in ["train", "plot"]:
        p = sub.add_parser(command)
        p.add_argument("--run", type=Path, required=True)
        p.add_argument("--device", default="cuda:0")
        if command == "train":
            p.add_argument(
                "--allow-missing-mondo",
                action="store_true",
                help="Train the joint model for other TFs even when MONDO lacks binding support; preserve the prepared inputs.",
            )
        if command == "plot":
            p.add_argument("--existing-curves", type=Path, default=Path("auto"))
            p.add_argument("--dictys-profile", type=Path, default=Path("auto"))
    for command in ["derive-diagnostic", "preprocessing-report"]:
        p = sub.add_parser(command)
        p.add_argument("--run", type=Path, required=True)
    p = sub.add_parser("benchmark")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--data-root", type=Path)
    p.add_argument("--thresholds", nargs="+", type=float, default=[0.5, 0.95])
    p.add_argument("--negative-seed", type=int, default=42)
    p = sub.add_parser("derive-grn-patience")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--grn-val-patience", type=int, default=50)
    p = sub.add_parser("stage-networks")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    if args.command in ["prepare", "prepare-both", "all"]:
        if args.command == "all":
            h = comparison_helpers()
            args.existing_curves = h.resolve_comparison(args.existing_curves)
            h.load_comparison(args.existing_curves)
        run = prepare_run(
            args.config.resolve(), args.data_root, args.upstream, args.seed
        )
        if args.command == "prepare-both":
            diagnostic = derive_diagnostic(run)
            print(f"DEFAULT_RUN={run}\nDIAGNOSTIC_RUN={diagnostic}", flush=True)
        if args.command == "all":
            train(run, args.device)
            plot(run, args.device, args.existing_curves, args.dictys_profile)
    elif args.command == "train":
        train(
            resolve_run(args.run),
            args.device,
            allow_missing_mondo=args.allow_missing_mondo,
        )
    elif args.command == "derive-diagnostic":
        derive_diagnostic(resolve_run(args.run))
    elif args.command == "preprocessing-report":
        report_preprocessing(resolve_run(args.run))
    elif args.command == "benchmark":
        from benchmark import benchmark

        benchmark(
            resolve_run(args.run), args.data_root, args.thresholds, args.negative_seed
        )
    elif args.command == "derive-grn-patience":
        from grn_patience import derive_grn_patience

        derive_grn_patience(resolve_run(args.run), args.grn_val_patience)
    elif args.command == "stage-networks":
        from stage_networks import export_stage_networks

        export_stage_networks(
            resolve_run(args.run), args.device, context_loader=context
        )
    else:
        plot(
            resolve_run(args.run),
            args.device,
            args.existing_curves,
            args.dictys_profile,
        )


if __name__ == "__main__":
    main()
