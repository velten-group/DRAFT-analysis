"""scDoRI release model/trainers with explicitly declared RNA+prior input adaptation."""

from __future__ import annotations

import dataclasses
import importlib
import logging
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
from common import read_json, source_file, write_json

HERE = Path(__file__).resolve().parent
PIN = "f02cf47de1e25d03f1409edf79c1690681b72c70"


def source_root(directory):
    return directory / "upstream/src/scdori/_core"


def software(directory):
    spec = read_json(directory / "upstream_api.json")
    preprocessing = read_json(directory / "preprocessing_api.json")
    if spec["commit"] != PIN:
        raise ValueError("Unexpected scDoRI source pin")
    if preprocessing["commit"] != PIN:
        raise ValueError("Unexpected scDoRI preprocessing source pin")
    for name, sha in spec["sha256_lf"].items():
        source_file(source_root(directory) / name, sha)
    return {
        "repository": "bioFAM/scDoRI",
        "release": "v1.0.0",
        "commit": PIN,
        "sha256_lf": spec["sha256_lf"],
        "optional_correlation_preprocessing": preprocessing,
    }


def native_api():
    # Private package avoids importing the unused motif/FIMO preprocessing stack.
    prefix = "_simulation_scdori_native"
    location = HERE / "upstream/src/scdori"
    for name, path in [(prefix, location), (prefix + "._core", location / "_core")]:
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(path)]
            sys.modules[name] = module
    return {
        name: importlib.import_module(prefix + "._core." + name)
        for name in (
            "config",
            "models",
            "utils",
            "evaluation",
            "train_scdori",
            "train_grn",
        )
    }


def virtual_features(prior, ntf):
    """One virtual feature per gene, identity feature-gene mask, unsigned TF prior."""
    ngenes = ntf + prior.shape[0]
    identity = np.eye(ngenes, dtype=np.float32)
    binding = np.zeros((ngenes, ntf), dtype=np.float32)
    binding[ntf:] = prior
    return identity, binding, -binding


def multiome_features(data):
    """Observed REs and public priors; no synthetic TF-gene labels or signs."""
    distance = np.zeros((len(data.genes), len(data.regions)), dtype=np.float32)
    distance[len(data.tfs) :] = data.re_tg
    binding = data.tf_re.astype(np.float32)
    return distance, binding, -binding


def ordered_split(data, seed):
    """Stratified by time, then randomize within a train prefix and validation suffix.

    Native get_tf_expression uses prefix row positions rather than loader IDs;
    an ordered prefix loader is required to retain its intended cell alignment.
    Time is used only for the split and final readout, never as a model covariate.
    """
    rng = np.random.default_rng(seed)
    train, val, offset = [], [], 0
    for counts in data.counts:
        order = rng.permutation(counts.shape[1]) + offset
        split = int(0.8 * len(order))
        train.extend(order[:split])
        val.extend(order[split:])
        offset += len(order)
    return np.concatenate([rng.permutation(train), rng.permutation(val)]), len(train)


def effective_edges(model, topic_tf):
    """Full signed topic networks from native forward equations (before RNA BN/softmax)."""
    import torch

    networks = []
    with torch.no_grad():
        q = torch.softmax(model.topic_peak_decoder, dim=1)
        gene_peak = (model.gene_peak_factor_fixed * model.gene_peak_factor_learnt).T
        denominator = gene_peak.sum(dim=0, keepdim=True) + 1e-7
        for topic in range(model.num_topics):
            act = (
                model.tf_binding_matrix_activator.T @ (q[topic, :, None] * gene_peak)
            ) / denominator
            rep = (
                model.tf_binding_matrix_repressor.T
                @ ((1 / (q[topic, :, None] + 1e-20)) * gene_peak)
            ) / denominator
            act = (
                act
                / (act.max() + 1e-15)
                * torch.relu(model.tf_gene_topic_activator_grn[topic])
            )
            rep = (
                rep
                / (rep.min() + 1e-15)
                * (-1)
                * torch.relu(model.tf_gene_topic_repressor_grn[topic])
            )
            networks.append(((act + rep) * topic_tf[topic, :, None]).cpu().numpy())
    return np.stack(networks)


def restore(model, path, device):
    import torch

    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise RuntimeError("Nonfinite official scDoRI checkpoint")


def fit(data, settings, run, seed, cpus, device, smoke):
    """Keep per-run native epoch/validation logs without editing official trainers."""
    run.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("_simulation_scdori_native")
    previous = logger.level
    handler = logging.FileHandler(
        run / "native_training.log", mode="w", encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        return _fit(data, settings, run, seed, cpus, device, smoke)
    finally:
        logger.removeHandler(handler)
        handler.close()
        logger.setLevel(previous)


def _fit(data, settings, run, seed, cpus, device, smoke):
    import anndata as ad
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(cpus)
    api = native_api()
    api["utils"].set_seed(seed)
    tc = api["config"].TrainConfig()
    for key, value in settings["training"].items():
        if not hasattr(tc, key):
            raise ValueError(f"Unknown official TrainConfig attribute: {key}")
        setattr(tc, key, value)
    multiome = hasattr(data, "atac")
    if not multiome:
        # RNA-only inputs: no ATAC loss terms; multiome inputs keep the native ATAC weights.
        tc.weight_atac_phase1 = tc.weight_atac_phase2 = tc.weight_atac_grn = 0.0
    elif min(tc.weight_atac_phase1, tc.weight_atac_phase2, tc.weight_atac_grn) <= 0:
        raise ValueError("Multiome scDoRI requires native positive ATAC loss weights")
    tc.random_seed = seed
    if smoke:
        tc.epoch_warmup_1, tc.max_scdori_epochs, tc.max_grn_epochs = 1, 2, 2
        tc.dim_encoder1, tc.dim_encoder2, tc.num_topics = 16, 8, 3
    order, split = ordered_split(data, seed)
    values = np.concatenate(
        [frame.to_numpy(dtype=np.float32).T for frame in data.counts]
    )
    times = np.concatenate(
        [np.full(frame.shape[1], time) for time, frame in zip(data.times, data.counts)]
    )[order]
    cell_ids = [
        f"T{time}:{cell}"
        for time, frame in zip(data.times, data.counts)
        for cell in frame.columns
    ]
    values = values[order]
    if (values[:, : len(data.tfs)].sum(axis=1) == 0).any():
        raise ValueError(
            "Official TF-expression normalization needs a nonzero TF library per cell"
        )
    if split < 2 or len(order) - split < 2 or split % tc.batch_size_cell == 1:
        raise ValueError(
            "Singleton scDoRI BatchNorm batch; choose another baseline batch_size_cell"
        )
    obs = pd.DataFrame(index=pd.Index(np.array(cell_ids)[order]))
    rna = ad.AnnData(values, obs=obs, var=pd.DataFrame(index=list(data.genes)))
    accessibility = (
        np.concatenate([frame.to_numpy(dtype=np.float32).T for frame in data.atac])[
            order
        ]
        if multiome
        else np.zeros_like(values)
    )
    atac = ad.AnnData(
        accessibility,
        obs=obs.copy(),
        var=pd.DataFrame(
            index=list(data.regions)
            if multiome
            else [f"virtual:{gene}" for gene in data.genes]
        ),
    )
    train = DataLoader(
        TensorDataset(torch.arange(split)), batch_size=tc.batch_size_cell, shuffle=False
    )
    val = DataLoader(
        TensorDataset(torch.arange(split, len(order))),
        batch_size=tc.batch_size_cell,
        shuffle=False,
    )
    all_cells = DataLoader(
        TensorDataset(torch.arange(len(order))),
        batch_size=tc.batch_size_cell_prediction,
        shuffle=False,
    )
    dev = torch.device(device)
    model = (
        api["models"]
        .scDoRI(
            dev,
            len(data.genes),
            atac.n_vars,
            len(data.tfs),
            tc.num_topics,
            1,
            tc.dim_encoder1,
            tc.dim_encoder2,
        )
        .to(dev)
    )
    identity, activator, repressor = [
        torch.tensor(a, device=dev)
        for a in (
            multiome_features(data)
            if multiome
            else virtual_features(data.prior, len(data.tfs))
        )
    ]
    preprocessing = settings.get("preprocessing", {"binding": "public_unsigned"})
    if preprocessing["binding"] == "observed_correlation":
        if not multiome:
            raise ValueError("Signed observed binding requires paired ATAC")
        from common import load_module

        prepare = load_module(
            "_simulation_scdori_signed_prior", HERE / "signed_prior.py"
        )
        act, rep = prepare.signed_binding(
            data,
            values[:split, : len(data.tfs)],
            accessibility[:split],
            preprocessing,
            run,
        )
        activator = torch.tensor(act, device=dev)
        repressor = torch.tensor(rep, device=dev)
    elif preprocessing != {"binding": "public_unsigned"}:
        raise ValueError("Unknown scDoRI binding preprocessing")
    initialize = api["models"].initialize_scdori_parameters
    initialize(model, identity, identity, activator, repressor, phase="warmup")
    batches = np.ones((len(order), 1), dtype=np.float32)
    numbers = np.ones((len(order), 1), dtype=np.float32)
    tf_idx = np.arange(len(data.tfs))
    args = [model, dev, train, val, rna, atac, numbers, tf_idx, batches, tc]
    tc.weights_folder_scdori, tc.weights_folder_grn = (
        str(run / "warmup"),
        str(run / "grn"),
    )
    write_json(
        run / "native_training_config.json",
        dataclasses.asdict(tc) | {"data_dir": str(tc.data_dir)},
    )
    pd.DataFrame(
        {
            "cell_id": obs.index,
            "time": times,
            "split": ["train"] * split + ["validation"] * (len(order) - split),
        }
    ).to_csv(run / "cell_order.tsv", sep="\t", index=False)
    api["train_scdori"].train_scdori_phases(*args)
    restore(model, run / "warmup/best_scdori_best_eval.pth", dev)
    initialize(model, identity, identity, activator, repressor, phase="grn")
    api["train_grn"].train_model_grn(*args)
    restore(model, run / "grn/best_scdori_best_eval.pth", dev)
    model.eval()
    theta = api["evaluation"].get_latent_topics(
        model, dev, all_cells, rna, atac, numbers, tf_idx, batches
    )
    topic_tf = api["train_grn"].get_tf_expression(
        tc.tf_expression_mode,
        model,
        dev,
        train,
        rna,
        atac,
        numbers,
        tf_idx,
        batches,
        tc,
    )
    if not np.isfinite(theta).all() or not np.allclose(theta.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Invalid native cell-topic weights")
    edges = effective_edges(model, topic_tf)[:, :, len(data.tfs) :]
    np.savez_compressed(
        run / "topic_networks.npz",
        signed_edges=edges,
        theta=theta,
        topic_tf=topic_tf.detach().cpu().numpy(),
    )
    signed = np.stack(
        [
            np.einsum("k,kfg->gf", theta[times == time].mean(axis=0), edges)
            for time in data.times
        ]
    )
    if not np.isfinite(signed).all() or np.any(
        signed[:, ~data.prior.astype(bool)] != 0
    ):
        raise ValueError("Invalid effective network or edges outside fixed prior")
    np.save(run / "signed_time_networks.npy", signed)
    return abs(signed), {
        "label": "scDoRI observed multiome (official model and trainers)"
        if multiome
        else "scDoRI RNA+prior adaptation (official model and trainers)",
        "virtual_features": None
        if multiome
        else "one per RNA gene, identity feature-gene scaffold; all-zero ATAC placeholders, NOT measured ATAC",
        "modalities": "paired observed RNA and ATAC; public RE IDs and fixed TF-RE/RE-TG priors"
        if multiome
        else "RNA with virtual features",
        "loss": "official native losses and early stopping; native ATAC weights enabled in all phases"
        if multiome
        else "official NB TF/RNA/GRN losses and native regularization/early stopping; ATAC weights zero in all phases",
        "prior": "observed training-cell signed TF-RE correlations through the pinned native routine, restricted to full public support; binary RE-TG mask"
        if preprocessing["binding"] == "observed_correlation"
        else "unsigned public TF-RE support in both sign branches; binary RE-TG links initialize and mask gene-peak parameters; no invented links to TF genes"
        if multiome
        else "same unsigned fixed union in activator and negative repressor branches; no known regulatory signs supplied",
        "score": "abs(mean-cell-theta mixture of signed effective topic TF-TG coefficients including native topic-TF expression); no permutation filtering",
        "scope": "joint RNA topic training across times; time only used for stratified validation split and readout",
        "native_trainers": ["train_scdori_phases", "train_model_grn"],
        "binding_preprocessing": preprocessing,
        "validation_only": smoke,
    }
