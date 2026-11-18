"""Framework-free public multiome inputs and metrics for the simulation benchmark."""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

if __package__:
    from . import common
else:
    import common

PROTOCOL = "prior_only_multiome"
POLICIES = {
    "scenicplus": "Official RNA TF2G component; no edge-prior or ATAC parameter; not a complete eRegulon pipeline",
    "scmtni": "RNA and the same projected TF-RE-TG candidate support as a native soft prior; native time chain and stability selection",
    "scdori": "Paired observed RNA/ATAC and the same TF-RE/RE-TG priors; native ATAC losses enabled; unsigned TF-RE prior in both sign branches",
    "dictys": "RNA and the same projected candidate support as a native allowed-edge mask; native per-time QC and OU inference",
}
sys.path.insert(0, str(common.PROJECT / "src"))
from network_evaluation.temporal import evaluate_temporal


@dataclass(frozen=True)
class MultiomeInputs(common.Inputs):
    regions: tuple[str, ...]
    atac: tuple[pd.DataFrame, ...]
    tf_re: np.ndarray
    re_tg: np.ndarray


def enabled(config):
    return config.get("benchmark", {}).get("input_protocol") == PROTOCOL


def generation_identity(cfg, seed):
    """Verify declared generators by source hashes, without importing trainers."""
    declared = cfg["simulate"].get("generator")
    if declared == "prior_sensitivity":
        # Import the specification by file: simulation_gnn.__init__ imports
        # Torch, which is deliberately absent from some official environments.
        specification = common.load_module(
            "_simulation_sensitivity_design",
            common.PROJECT / "src/simulation_gnn/sensitivity_design.py",
        )
        return specification.generation_identity(cfg, seed, common.PROJECT)
    if declared == "prior_module_mixing":
        paths = [
            common.PROJECT / "src/simulation_gnn" / name
            for name in (
                "prior_module_mixing.py",
                "prior_all_within_design.py",
                "R/prior_binary.R",
                "R/positive_kinetics.R",
                "R/prior_trial_context.R",
                "prior_trajectory_modules.py",
                "prior_binary.py",
            )
        ]
        return {
            "version": "prior_module_mixing_trajectory_v1",
            "params": dict(cfg["simulate"]["params"], seed=seed),
            "code": {
                p.relative_to(common.PROJECT).as_posix(): common.digest(p)
                for p in paths
            },
        }
    if declared == "prior_all_within":
        paths = [
            common.PROJECT / "src/simulation_gnn" / name
            for name in (
                "prior_all_within.py",
                "prior_all_within_design.py",
                "R/prior_binary.R",
                "R/positive_kinetics.R",
                "R/prior_trial_context.R",
                "prior_trajectory_modules.py",
                "prior_binary.py",
            )
        ]
        return {
            "version": "prior_binary_all_within_trajectory_v1",
            "params": dict(cfg["simulate"]["params"], seed=seed),
            "code": {
                p.relative_to(common.PROJECT).as_posix(): common.digest(p)
                for p in paths
            },
        }
    if declared == "prior_random":
        paths = [
            common.PROJECT / "src/simulation_gnn" / name
            for name in (
                "prior_random.py",
                "prior_random_design.py",
                "R/prior_binary.R",
                "R/positive_kinetics.R",
                "R/prior_trial_context.R",
                "prior_trajectory_modules.py",
            )
        ]
        return {
            "version": "prior_binary_random_trajectory_v1",
            "params": dict(cfg["simulate"]["params"], seed=seed),
            "code": {
                p.relative_to(common.PROJECT).as_posix(): common.digest(p)
                for p in paths
            },
        }
    if declared == "prior_binary":
        paths = [
            common.PROJECT / "src/simulation_gnn" / name
            for name in (
                "prior_binary.py",
                "prior_binary_design.py",
                "prior_trajectory_modules.py",
                "R/prior_binary.R",
                "R/positive_kinetics.R",
                "R/prior_trial_context.R",
            )
        ]
        return {
            "version": "prior_binary_modular_trajectory_v2",
            "params": dict(cfg["simulate"]["params"], seed=seed),
            "code": {
                p.relative_to(common.PROJECT).as_posix(): common.digest(p)
                for p in paths
            },
        }
    trajectories = {
        "prior_trajectory": (
            "prior_trajectory.py",
            "prior_developmental_trajectory_v1",
            ("prior_signal_design.py", "prior_trial_design.py"),
        ),
        "prior_trajectory_regions": (
            "prior_trajectory_regions.py",
            "prior_developmental_single_tf_region_v1",
            ("prior_trial_design.py",),
        ),
        "prior_trajectory_modules": (
            "prior_trajectory_modules.py",
            "prior_developmental_strong_modules_v1",
            ("prior_trajectory_module_design.py",),
        ),
        "prior_noise": (
            "prior_noise.py",
            "prior_developmental_strong_modules_prior_noise_v1",
            ("prior_trajectory_modules.py", "prior_trajectory_module_design.py"),
        ),
        "prior_no_extra": (
            "prior_no_extra.py",
            "prior_developmental_strong_modules_no_extra_v1",
            ("prior_trajectory_modules.py", "prior_trajectory_module_design.py"),
        ),
    }
    if declared in trajectories:
        entry, version, designs = trajectories[declared]
        names = [
            entry,
            *designs,
            "R/prior_trajectory.R",
            "R/prior_trial_context.R",
        ]
        paths = [common.PROJECT / "src/simulation_gnn" / name for name in names]
        return {
            "version": version,
            "params": dict(cfg["simulate"]["params"], seed=seed),
            "code": {
                p.relative_to(common.PROJECT).as_posix(): common.digest(p)
                for p in paths
            },
        }
    if declared is not None:
        raise ValueError(f"Unknown declared simulation generator: {declared}")
    signal = "atac_effect" in cfg["simulate"]["params"]
    names = (
        (
            "prior_signal.py",
            "prior_signal_design.py",
            "prior_trial_design.py",
            "R/prior_signal.R",
            "R/prior_trial_context.R",
        )
        if signal
        else ("R/prior_trial.R", "R/prior_trial_context.R", "prior_trial_design.py")
    )
    paths = [common.PROJECT / "src/simulation_gnn" / name for name in names]
    return {
        "version": "prior_signal_snapshot_v1" if signal else "prior_only_snapshot_v5",
        "params": dict(cfg["simulate"]["params"], seed=seed),
        "code": {
            p.relative_to(common.PROJECT).as_posix()
            if signal
            else p.name: common.digest(p)
            for p in paths
        },
    }


def reference(config):
    path = common.PROJECT / config["benchmark"]["reference_config"]
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if (
        config["draft"]["task"] != "prior_reconstruction"
        or any(
            config["draft"][k] != cfg["draft"][k]
            for k in ("num_tf", "num_tg", "time_points")
        )
        or config["datasets"]["data_dir"] != cfg["simulate"]["data_dir"]
        or config["datasets"]["seeds"] != cfg["multiome"]["data_seeds"]
        or config["datasets"]["model_seeds"] != cfg["multiome"]["model_seeds"]
    ):
        raise ValueError("Baseline axes, seeds or task differ from the benchmark reference")
    return cfg


def matrix(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        import csv

        header = next(csv.reader(handle))
    if len(header[1:]) != len(set(header[1:])) or any(not x for x in header[1:]):
        raise ValueError(f"Invalid matrix column IDs: {path}")
    frame = pd.read_csv(path, index_col=0)
    if frame.index.isna().any() or frame.index.has_duplicates:
        raise ValueError(f"Invalid matrix row IDs: {path}")
    frame.index, frame.columns = frame.index.astype(str), frame.columns.astype(str)
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError(f"Nonfinite matrix: {path}")
    return frame


def load_inputs(directory, config):
    """Hash truth files for provenance; never parse truth or latent states here."""
    directory = Path(directory)
    cfg = reference(config)
    marker = common.read_json(directory / "generation_complete.json")
    seed = int(directory.name.removeprefix("seed_"))
    expected = generation_identity(cfg, seed)
    files = marker["files"]
    if (
        not matching_generation_identity(
            marker["identity"],
            expected,
            allow_line_endings=config.get("benchmark", {}).get(
                "allow_generator_line_ending_equivalence", False
            ),
        )
        or not files
        or any(common.digest(directory / name) != sha for name, sha in files.items())
    ):
        raise ValueError(f"Generation identity/checksum changed: {directory}")
    ntf, ntg = (int(cfg["draft"][k]) for k in ("num_tf", "num_tg"))
    genes = tuple(str(i + 1) for i in range(ntf + ntg))
    tfs, targets = genes[:ntf], genes[ntf:]
    tr = matrix(directory / "multiome/tf_re_prior.csv")
    rg = matrix(directory / "multiome/re_tg_prior.csv")
    prior = matrix(directory / "candidate_prior.csv")
    regions = tuple(tr.index)
    if (
        tuple(tr.columns) != tfs
        or tuple(rg.index) != targets
        or tuple(rg.columns) != regions
        or tuple(prior.index) != targets
        or tuple(prior.columns) != tfs
    ):
        raise ValueError("Multiome prior axes differ")
    for frame in (tr, rg, prior):
        if not np.isin(frame.to_numpy(), [0, 1]).all():
            raise ValueError("Multiome priors must be binary")
    tr, rg, prior = (x.to_numpy(dtype=bool) for x in (tr, rg, prior))
    if not np.array_equal(rg.astype(int) @ tr.astype(int) > 0, prior):
        raise ValueError("Candidate support differs from the TF-RE-TG paths")
    counts, atac = [], []
    times = tuple(cfg["draft"]["time_points"])
    for time in times:
        rna = common.load_counts(directory / f"counts_time_{time}.csv", genes)
        acc = matrix(directory / f"multiome/atac_time_{time}.csv")
        values = acc.to_numpy(dtype=float)
        if (
            tuple(acc.index) != regions
            or not acc.columns.equals(rna.columns)
            or (values < 0).any()
            or not np.equal(values, np.floor(values)).all()
            or (values.sum(0) == 0).any()
        ):
            raise ValueError(f"Invalid ATAC counts or RNA/ATAC alignment at T{time}")
        counts.append(rna)
        atac.append(acc)
    for array in (tr, rg, prior):
        array.setflags(write=False)
    sources = dict(
        files,
        **{
            "generation_complete.json": common.digest(
                directory / "generation_complete.json"
            )
        },
    )
    # Require every actual training input to have a generation checksum.
    required = [
        "candidate_prior.csv",
        "multiome/tf_re_prior.csv",
        "multiome/re_tg_prior.csv",
    ]
    required += [
        name
        for t in times
        for name in (f"counts_time_{t}.csv", f"multiome/atac_time_{t}.csv")
    ]
    if not set(required).issubset(files):
        raise ValueError("Generation receipt omits a public training input")
    return MultiomeInputs(
        times, genes, tfs, targets, tuple(counts), prior, regions, tuple(atac), tr, rg
    ), sources


def matching_generation_identity(recorded, expected, *, allow_line_endings=False):
    """Optional cross-platform receipt check; accept only LF/CRLF source changes.

    Data hashes and the recorded identity are never rewritten. All parameters,
    versions and source-path sets must match exactly. Opt-in is restricted to
    file-backed source hashes; symbolic identities remain strict.
    """
    if recorded == expected:
        return True
    if not allow_line_endings:
        return False
    if {k: v for k, v in recorded.items() if k != "code"} != {
        k: v for k, v in expected.items() if k != "code"
    } or set(recorded.get("code", {})) != set(expected.get("code", {})):
        return False
    for name, current in expected["code"].items():
        if recorded["code"][name] == current:
            continue
        path = common.PROJECT / name
        if not path.is_file():
            return False
        raw = path.read_bytes().replace(b"\r\n", b"\n")
        variants = (raw, raw.replace(b"\n", b"\r\n"))
        if recorded["code"][name] not in {
            hashlib.sha256(value).hexdigest() for value in variants
        }:
            return False
    return True


def evaluate(scores, truth, data, same, k, *, rank_scores=None):
    """Same candidates, AUC, top-k and continuity definition as prior_trial."""
    scores = np.asarray(scores, dtype=float)
    rank_scores = (
        scores if rank_scores is None else np.asarray(rank_scores, dtype=float)
    )
    truth = np.asarray(truth, dtype=bool)
    candidate = data.prior.astype(bool)
    if (
        scores.shape != truth.shape
        or rank_scores.shape != scores.shape
        or scores.shape != (len(data.times), len(data.targets), len(data.tfs))
        or not np.isfinite(scores).all()
        or not np.isfinite(rank_scores).all()
        or (scores < 0).any()
        or (truth & ~candidate).any()
        or not 1 <= k <= candidate.sum()
    ):
        raise ValueError(
            "Invalid prediction axes, finite scores or candidate support"
        )

    def average(values):
        values = [v for v in values if v is not None]
        return float(np.mean(values)) if values else None

    local = candidate & same
    priority = np.random.default_rng(20261022).random(int(candidate.sum()))
    selected = np.zeros((len(data.times), int(candidate.sum())), dtype=bool)
    for t, row in enumerate(rank_scores[:, candidate]):
        selected[t, np.lexsort((priority, -row))[:k]] = True
    temporal = evaluate_temporal(scores[:, candidate], data.times)
    values = {
        "candidate_auroc": average(
            [common.auc(y[candidate], x[candidate]) for y, x in zip(truth, rank_scores)]
        ),
        "same_module_auroc": average(
            [common.auc(y[local], x[local]) for y, x in zip(truth, rank_scores)]
        )
        if local.any()
        else None,
        "f1_at_k": float(
            np.mean(
                2 * (selected & truth[:, candidate]).sum(1) / (k + truth.sum((1, 2)))
            )
        ),
        "change_auroc": average(
            [
                common.auc((a != b)[candidate], np.abs(q - p)[candidate])
                for a, b, p, q in zip(truth[:-1], truth[1:], scores[:-1], scores[1:])
            ]
        ),
        "ground_truth_continuity": evaluate_temporal(truth[:, candidate], data.times)[
            "temporal_continuity"
        ],
        "max_tie_fraction": float(
            np.mean(
                [
                    np.unique(x, return_counts=True)[1].max() / len(x)
                    for x in rank_scores[:, candidate]
                ]
            )
        ),
        "zero_fraction": float(np.mean(rank_scores[:, candidate] == 0)),
        "one_fraction": float(np.mean(rank_scores[:, candidate] == 1)),
    }
    values.update(
        {
            key: temporal[key]
            for key in (
                "temporal_continuity",
                "velocity_roughness",
                "constant_over_time",
                "largest_jump_interval",
            )
        }
    )
    return values


def truth_and_modules(directory, data):
    truth = common.load_truth(directory, data.times, len(data.tfs), len(data.targets))
    table = pd.read_csv(
        Path(directory) / "gene_modules.tsv", sep="\t", dtype={"gene": str}
    )
    # Generator table uses explicit gene IDs; no label is passed to a fit adapter.
    modules = table.set_index("gene")["module"]
    same = (
        modules.loc[list(data.targets)].to_numpy()[:, None]
        == modules.loc[list(data.tfs)].to_numpy()[None, :]
    )
    return truth, same
