"""Stage mixtures of native scDoRI RNA-decoder edges and exact GRN path contributions."""

from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lineages import stage_labels
from scdori_native import check_outputs, digest, read_json, restore, write_json


def topic_path_contributions(model, topic_tf, target_indices):
    """Decompose the unfiltered native GRN equations into additive TF–RE–TG paths."""
    with torch.no_grad():
        beta = torch.softmax(model.topic_peak_decoder, dim=1)
        gene_peak = model.gene_peak_factor_fixed * model.gene_peak_factor_learnt
        act_binding = model.tf_binding_matrix_activator
        rep_binding = model.tf_binding_matrix_repressor
        denominator = gene_peak.sum(dim=1) + 1e-7
        # Sparse eligible triples. Normalizations below still use ALL model genes/TFs.
        binding_support = ((act_binding != 0) | (rep_binding != 0)).cpu().numpy()
        target_support = (gene_peak[target_indices] != 0).T.cpu().numpy()
        triple_parts = []
        for peak in range(model.num_peaks):
            tf = np.flatnonzero(binding_support[peak])
            target = np.flatnonzero(target_support[peak])
            if len(tf) and len(target):
                triple_parts.append(
                    np.column_stack(
                        [
                            np.repeat(tf, len(target)),
                            np.full(len(tf) * len(target), peak),
                            np.tile(target, len(tf)),
                        ]
                    )
                )
        triples = (
            np.vstack(triple_parts).astype(np.int64)
            if triple_parts
            else np.empty((0, 3), dtype=np.int64)
        )
        tf, peak, target = [
            torch.as_tensor(v, device=gene_peak.device) for v in triples.T
        ]
        gene = torch.as_tensor(target_indices, device=gene_peak.device)[target]
        w = gene_peak[gene, peak] / denominator[gene]
        activation, repression = [], []
        for topic in range(model.num_topics):
            b = beta[topic]
            act = act_binding.T @ (b[:, None] * gene_peak.T)
            act = act / denominator[None, :]
            rep = rep_binding.T @ ((1 / (b[:, None] + 1e-20)) * gene_peak.T)
            rep = rep / denominator[None, :]
            a = topic_tf[topic, tf] * torch.relu(
                model.tf_gene_topic_activator_grn[topic, tf, gene]
            )
            a = a * act_binding[peak, tf] * b[peak] * w / (act.max() + 1e-15)
            r = topic_tf[topic, tf] * torch.relu(
                model.tf_gene_topic_repressor_grn[topic, tf, gene]
            )
            r = r * rep_binding[peak, tf] / (b[peak] + 1e-20) * w / (rep.min() + 1e-15)
            activation.append(a.cpu().numpy())
            repression.append(r.cpu().numpy())
        return triples, np.asarray(activation), np.asarray(repression)


def stage_means(theta, stages, n_stages=15):
    theta = np.asarray(theta, dtype=np.float64)
    if (
        theta.ndim != 2
        or not np.isfinite(theta).all()
        or (theta < 0).any()
        or not np.allclose(theta.sum(axis=1), 1, atol=1e-5)
    ):
        raise ValueError("Invalid per-cell topic mixture.")
    if not np.array_equal(np.unique(stages), np.arange(n_stages)):
        raise ValueError("Each stage must contain cells.")
    return np.stack(
        [theta[np.asarray(stages) == stage].mean(axis=0) for stage in range(n_stages)]
    )


def export_stage_networks(run: Path, device: str, context_loader=None) -> Path:
    if context_loader is None:
        from workflow import context

        context_loader = context
    complete = read_json(run / "grn_complete.json")
    check_outputs(complete, run)
    ctx = context_loader(run, device)
    model, api = ctx["model"], ctx["api"]
    restore(model, run / complete["checkpoint"], ctx["device"])
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
    stages = ctx["rna"].obs.stage.to_numpy()
    labels = stage_labels(ctx["cfg"].get("lineage", "germline"))
    mixture = stage_means(theta, stages, len(labels))
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
    triples, topic_act, topic_rep = topic_path_contributions(
        model, topic_tf, ctx["targets"]
    )
    stage_act, stage_rep = mixture @ topic_act, mixture @ topic_rep
    with torch.no_grad():
        beta = torch.softmax(model.topic_peak_decoder, dim=1)
        beta_min = beta.min(dim=0).values
        beta_max = beta.max(dim=0).values
        normalized = ((beta - beta_min) / (beta_max - beta_min + 1e-8)).cpu().numpy()
        gene_peak = (
            (model.gene_peak_factor_fixed * model.gene_peak_factor_learnt)[
                ctx["targets"]
            ]
            .cpu()
            .numpy()
        )
        fixed_support = model.gene_peak_factor_fixed[ctx["targets"]].cpu().numpy()
    peak_activity = mixture @ normalized
    gene, peak = np.nonzero(fixed_support)
    weights = gene_peak[gene, peak]
    re_tg_scores = (peak_activity[:, peak] * weights).astype(np.float32)
    targets = ctx["rna"].var_names[ctx["targets"]].tolist()
    peaks = ctx["atac"].var_names.tolist()
    tf_ids = ctx["rna"].var_names[ctx["tf_indices"]].tolist()
    source = ctx["manifest"]["input_sources"]
    symbols = {}
    for name in ["TF.tsv", "TG.tsv"]:
        path = Path(source[name]["path"])
        if digest(path) != source[name]["sha256"]:
            raise ValueError(f"Source identifiers changed: {path}")
        table = pd.read_csv(path, sep="\t", dtype=str)
        symbols.update(zip(table.FBgn_ID, table.gene_name))
    out = run / f"stage_networks_{uuid.uuid4().hex[:10]}"
    out.mkdir(exist_ok=False)
    archive = out / "stage_arrays.npz"
    np.savez_compressed(
        archive,
        genes=np.asarray(targets),
        peaks=np.asarray(peaks),
        pair_gene=gene,
        pair_peak=peak,
        global_gene_peak=weights,
        re_tg_scores=re_tg_scores,
        stage_topic_weights=mixture,
        stage_peak_activity=peak_activity,
    )
    cells = pd.DataFrame({"cell_id": ctx["rna"].obs_names, "time_idx": stages})
    cells.to_csv(out / "cells.tsv.gz", sep="\t", index=False)
    pd.DataFrame(mixture, index=np.arange(len(labels))).rename_axis("time_idx").to_csv(
        out / "stage_topics.tsv", sep="\t"
    )
    tf, p, g = triples.T
    triple_base = pd.DataFrame(
        {
            "TF_id": np.asarray(tf_ids)[tf],
            "TF_name": [symbols.get(tf_ids[i], tf_ids[i]) for i in tf],
            "RE": np.asarray(peaks)[p],
            "TG_id": np.asarray(targets)[g],
            "TG_name": [symbols.get(targets[i], targets[i]) for i in g],
        }
    )
    counts = []
    for stage in range(len(labels)):
        re_frame = pd.DataFrame(
            {
                "RE": np.asarray(peaks)[peak],
                "TG_id": np.asarray(targets)[gene],
                "TG_name": [symbols.get(targets[i], targets[i]) for i in gene],
                "score": re_tg_scores[stage],
            }
        )
        re_frame.insert(0, "time_idx", stage)
        re_frame.to_csv(out / f"t{stage:02d}_RE_TG.tsv.gz", sep="\t", index=False)
        frame = triple_base.copy()
        frame.insert(0, "time_idx", stage)
        frame["activation_contribution"] = stage_act[stage]
        frame["repression_contribution"] = stage_rep[stage]
        frame["path_score"] = stage_act[stage] - stage_rep[stage]
        frame["re_tg_score"] = peak_activity[stage, p] * gene_peak[g, p]
        frame.to_csv(out / f"t{stage:02d}_TF_RE_TG.tsv.gz", sep="\t", index=False)
        collapsed = (
            frame.groupby(["TF_id", "TF_name", "TG_id", "TG_name"], sort=False)[
                "path_score"
            ]
            .sum()
            .reset_index()
        )
        collapsed.insert(0, "time_idx", stage)
        collapsed.to_csv(out / f"t{stage:02d}_TF_TG.tsv.gz", sep="\t", index=False)
        counts.append(
            {
                "time_idx": stage,
                "stage_label": labels[stage],
                "n_cells": int((stages == stage).sum()),
                "n_eligible_triples": len(frame),
                "n_nonzero_paths": int(frame.path_score.ne(0).sum()),
            }
        )
    pd.DataFrame(counts).to_csv(out / "stage_summary.tsv", sep="\t", index=False)
    files = {p.name: digest(p) for p in out.iterdir() if p.is_file()}
    write_json(
        out / "stage_report.json",
        {
            "run": str(run.resolve()),
            "checkpoint": complete["checkpoint"],
            "checkpoint_sha256": complete["sha256"][complete["checkpoint"]],
            "manifest_sha256": digest(run / "manifest.json"),
            "input_sources": source,
            "validation_only": bool(ctx["cfg"].get("validation_only")),
            "seed": ctx["cfg"]["seed"],
            "stage_labels": labels,
            "binding_mode": ctx["cfg"].get("binding_mode", "native_correlation"),
            "device": device,
            "files_sha256": files,
            "stage_mixture": "arithmetic mean of native per-cell theta over the original annotated stage; no smoothing or time-based model retraining",
            "paths": "exact additive unfiltered native GRN path contributions, mixed BEFORE cancellation/absolute value; global normalizers use all model TFs/genes; selected non-TF targets only",
            "re_tg": "native RNA-decoder coefficient: gene_peak * stage_mean(theta) @ per-peak minmax(softmax(topic_peak)); before RNA BatchNorm/softmax, not a calibrated probability",
            "topic_tf": "official get_tf_expression applied to the restored GRN checkpoint and ordered training prefix; native expression clamping retained",
            "filtering": "eligible nonzero binding/gene-peak support; no Top-K, activity rescue or permutation significance filtering",
        },
    )
    print(f"Stage networks: {out}", flush=True)
    return out
