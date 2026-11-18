"""Framework-free simulation IO and evaluation. Does not import DRAFT."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
METHODS = ("scenicplus", "scmtni", "scdori", "dictys")
LOGGER = logging.getLogger(__name__)


def resolve_data_root(config, *, data_root=None, data_project=None):
    """Resolve read-only input paths independently of this code checkout."""
    if data_root is not None:
        root = Path(data_root).expanduser()
        return (root if root.is_absolute() else PROJECT / root).resolve()
    root = Path(config["datasets"]["data_dir"]).expanduser()
    if root.is_absolute():
        return root.resolve()
    base = Path(
        data_project or os.environ.get("SIMULATION_DATA_PROJECT") or PROJECT
    ).expanduser()
    if not base.is_absolute():
        base = PROJECT / base
    return (base / root).resolve()


def check_dataset_paths(config, root, *, data_seeds=None):
    """Check every selected dataset before starting any model or creating runs."""
    times = config["draft"]["time_points"]
    seeds = list(config["datasets"]["seeds"] if data_seeds is None else data_seeds)
    if not seeds or not times:
        raise ValueError("Data seeds and time points must not be empty")
    missing = [
        root / f"seed_{seed}/{kind}_time_{time}.csv"
        for seed in seeds
        for time in times
        for kind in ("counts", "grn")
        if not (root / f"seed_{seed}/{kind}_time_{time}.csv").is_file()
    ]
    if missing:
        details = "\n".join(str(path) for path in missing[:12])
        if len(missing) > 12:
            details += f"\n... and {len(missing) - 12} more missing files"
        raise FileNotFoundError(
            f"Simulation input check failed: {len(missing)} missing files under {root}\n"
            f"{details}\nSet SIMULATION_DATA_PROJECT to the project containing the existing data/ "
            "directory, or use run.py --data-root for one dataset root. Inputs will not be generated automatically."
        )
    expected = config.get("simulate", {}).get("params", {}).get("num_changing_edges")
    for seed in seeds:
        marker = root / f"seed_{seed}/simulation_params.json"
        if expected is not None and marker.is_file():
            actual = read_json(marker).get("num_changing_edges")
            if actual is not None and float(actual) != float(expected):
                raise ValueError(
                    f"Simulation condition mismatch: {marker}: num_changing_edges={actual}, "
                    f"but config expects {expected}. Choose the matching dataset; no fallback is applied."
                )
    return seeds


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def load_module(name, path):
    import sys

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def source_file(path, expected):
    actual = hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    if actual != expected:
        raise ValueError(f"Pinned upstream source changed: {path}")


@dataclass(frozen=True)
class Inputs:
    # No per-time truth or truth-file paths are supplied to an adapter.
    times: tuple[int, ...]
    genes: tuple[str, ...]
    tfs: tuple[str, ...]
    targets: tuple[str, ...]
    counts: tuple[pd.DataFrame, ...]  # gene x cell, raw integer counts
    prior: np.ndarray  # TG x TF, one fixed candidate graph for all times


def load_truth(directory, times, ntf, ntg):
    genes = [str(i) for i in range(1, ntf + ntg + 1)]
    output = []
    for time in times:
        path = Path(directory) / f"grn_time_{time}.csv"
        frame = pd.read_csv(path)
        if {"regulated.gene", "regulator.gene"}.issubset(frame):
            matrix = np.zeros((ntg, ntf), dtype=np.int8)
            for target, tf in frame[["regulated.gene", "regulator.gene"]].itertuples(
                index=False, name=None
            ):
                target, tf = int(target), int(tf)
                if ntf < target <= ntf + ntg and 1 <= tf <= ntf:
                    matrix[target - ntf - 1, tf - 1] = 1
        else:
            frame = pd.read_csv(path, index_col=0)
            frame.index, frame.columns = (
                frame.index.astype(str),
                frame.columns.astype(str),
            )
            if frame.index.has_duplicates or frame.columns.has_duplicates:
                raise ValueError(f"Duplicate GRN axes: {path}")
            if frame.shape == (ntf + ntg, ntf + ntg) or frame.shape == (ntg, ntf):
                values = frame.loc[genes[ntf:], genes[:ntf]].to_numpy(dtype=float)
            else:
                raise ValueError(f"Unexpected GRN shape: {path}: {frame.shape}")
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite GRN: {path}")
            matrix = (values > 0).astype(np.int8)
        output.append(matrix)
    return np.stack(output)


def load_counts(path, genes):
    """Select configured genes by ID; keep all cells and ignore background rows."""
    # Check the raw header: pandas otherwise silently renames duplicate columns.
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), [])
    cells = pd.Index(header[1:])
    duplicate_cells = cells[cells.duplicated()].unique().tolist()
    if cells.empty or duplicate_cells or any(not cell.strip() for cell in cells):
        raise ValueError(
            f"Invalid cell IDs: {path}: cells={len(cells)}, duplicate_cells={duplicate_cells[:8]}"
        )
    frame = pd.read_csv(path, index_col=0)
    missing_labels = frame.index.isna().any()
    frame.index, frame.columns = frame.index.astype(str), frame.columns.astype(str)
    duplicate_genes = frame.index[frame.index.duplicated()].unique().tolist()
    missing = [gene for gene in genes if gene not in frame.index]
    if missing_labels or duplicate_genes or missing:
        raise ValueError(
            f"Invalid gene IDs: {path}: shape={frame.shape}, "
            f"missing_required={missing[:12]}, duplicate_genes={duplicate_genes[:8]}, "
            f"row_head={frame.index[:8].tolist()}, row_tail={frame.index[-8:].tolist()}. "
            "Required gene IDs must be present and unique; additional background genes are allowed."
        )
    if len(frame) > len(genes):
        LOGGER.info(
            "Selecting %d configured genes from %d rows; ignoring %d background rows; keeping all %d cells: %s",
            len(genes),
            len(frame),
            len(frame) - len(genes),
            frame.shape[1],
            path,
        )
    frame = frame.loc[list(genes)]
    array = frame.to_numpy(dtype=float)
    if (
        not np.isfinite(array).all()
        or (array < 0).any()
        or not np.equal(array, np.floor(array)).all()
    ):
        raise ValueError(f"Expected nonnegative raw integer RNA counts: {path}")
    if array.shape[1] < 4 or (array.sum(axis=0) == 0).any():
        raise ValueError(f"Too few cells or empty RNA libraries: {path}")
    return frame


def load_candidate_prior(path, ntf, ntg):
    """Load the fixed TG x TF candidate graph with the same axis contract as DRAFT."""
    path = Path(path)
    frame = pd.read_csv(path, index_col=0)
    frame.index, frame.columns = frame.index.astype(str), frame.columns.astype(str)
    expected_targets = [str(i) for i in range(ntf + 1, ntf + ntg + 1)]
    expected_tfs = [str(i) for i in range(1, ntf + 1)]
    if frame.index.tolist() != expected_targets:
        raise ValueError(f"Candidate prior has unexpected TG IDs: {path}")
    if frame.columns.tolist() != expected_tfs:
        raise ValueError(f"Candidate prior has unexpected TF IDs: {path}")
    values = frame.to_numpy()
    if not np.isin(values, [0, 1]).all():
        raise ValueError(f"Candidate prior must be binary: {path}")
    return values.astype(np.int8)


def candidate_edge_split(candidate, *, train_fraction, seed):
    """Reproduce DRAFT's fixed per-TG candidate train/evaluation split."""
    candidate = np.asarray(candidate, dtype=bool)
    if candidate.ndim != 2 or not candidate.any():
        raise ValueError("Candidate prior must be a nonempty TG x TF matrix")
    if not 0 < train_fraction < 1:
        raise ValueError("candidate_train_fraction must be between zero and one")
    rng = np.random.default_rng(seed)
    train = np.zeros_like(candidate, dtype=bool)
    row_edges, row_counts = [], []
    for tg in range(candidate.shape[0]):
        edges = np.flatnonzero(candidate[tg])
        if len(edges) < 2:
            raise ValueError("Every TG needs at least two candidate TFs for a split")
        row_edges.append(rng.permutation(edges))
        row_counts.append(
            min(max(int(np.floor(train_fraction * len(edges))), 1), len(edges) - 1)
        )
    desired = round(train_fraction * int(candidate.sum()))
    delta = desired - sum(row_counts)
    while delta:
        direction = 1 if delta > 0 else -1
        eligible = [
            tg
            for tg, (count, edges) in enumerate(zip(row_counts, row_edges))
            if (direction > 0 and count < len(edges) - 1)
            or (direction < 0 and count > 1)
        ]
        if not eligible:
            break
        for tg in rng.permutation(eligible)[: abs(delta)]:
            row_counts[int(tg)] += direction
            delta -= direction
            if delta == 0:
                break
    if delta:
        raise ValueError("Could not allocate the requested candidate train fraction")
    for tg, (edges, count) in enumerate(zip(row_edges, row_counts)):
        train[tg, edges[:count]] = True
    return train, candidate & ~train


def load_inputs(directory, times, ntf, ntg):
    directory = Path(directory)
    genes = tuple(str(i) for i in range(1, ntf + ntg + 1))
    counts = []
    sources = {
        f"{kind}_time_{time}.csv": digest(directory / f"{kind}_time_{time}.csv")
        for time in times
        for kind in ("counts", "grn")
    }
    marker = directory / "simulation_params.json"
    if marker.is_file():
        sources[marker.name] = digest(marker)
    candidate_path = directory / "candidate_prior.csv"
    if candidate_path.is_file():
        sources[candidate_path.name] = digest(candidate_path)
    for time in times:
        path = directory / f"counts_time_{time}.csv"
        counts.append(load_counts(path, genes))
    truth = load_truth(directory, times, ntf, ntg)
    for name, sha in sources.items():
        if digest(directory / name) != sha:
            raise ValueError(
                f"Simulation source changed while reading: {directory / name}"
            )
    prior = (
        load_candidate_prior(candidate_path, ntf, ntg)
        if candidate_path.is_file()
        else truth.any(axis=0).astype(np.int8)
    )
    if candidate_path.is_file() and np.any(truth.astype(bool) & ~prior.astype(bool)):
        raise ValueError("A true edge falls outside candidate_prior.csv")
    if not prior.any():
        raise ValueError("Empty fixed prior.")
    prior.setflags(write=False)
    return Inputs(
        tuple(times), genes, genes[:ntf], genes[ntf:], tuple(counts), prior
    ), sources


def log_expression(counts, target_sum):
    values = counts.to_numpy(dtype=float).T
    return pd.DataFrame(
        np.log1p(values / values.sum(axis=1, keepdims=True) * target_sum),
        index=counts.columns,
        columns=counts.index,
    )


def write_inputs(data, folder):
    folder.mkdir(parents=True, exist_ok=True)
    for time, counts in zip(data.times, data.counts):
        counts.to_csv(folder / f"counts_time_{time}.tsv.gz", sep="\t")
    pd.DataFrame(data.prior, index=data.targets, columns=data.tfs).to_csv(
        folder / "fixed_union_prior.tsv", sep="\t"
    )
    write_json(
        folder / "axes.json",
        {
            "times": data.times,
            "genes": data.genes,
            "tfs": data.tfs,
            "targets": data.targets,
        },
    )


def balanced_indices(y, seed):
    # Same fresh RNG, choice, concatenation and shuffle as simulation_gnn.workflow.
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if len(pos) == 0:
        return np.arange(len(y))
    idx = np.concatenate((pos, rng.choice(neg, min(len(neg), len(pos)), replace=False)))
    rng.shuffle(idx)
    return idx


def auc(y, score):
    # Mann-Whitney AUROC with average ranks for ties (no sklearn runtime needed).
    from scipy.stats import rankdata

    pos = y == 1
    npos, nneg = pos.sum(), (~pos).sum()
    if not npos or not nneg:
        return None
    return float((rankdata(score)[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def evaluate(
    data,
    truth,
    scores,
    seed,
    *,
    evaluation_mask=None,
    smoothness_metric="frobenius",
):
    expected = (len(data.times), len(data.targets), len(data.tfs))
    if scores.shape != expected or not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError(
            f"Invalid score tensor: expected {expected}, got {scores.shape}"
        )
    if evaluation_mask is not None:
        evaluation_mask = np.asarray(evaluation_mask, dtype=bool)
        if evaluation_mask.shape != data.prior.shape or np.any(
            evaluation_mask & ~data.prior.astype(bool)
        ):
            raise ValueError("Evaluation mask must be a subset of the fixed prior")
    per_time, in_prior, heldout = {}, {}, {}
    for time, y, score in zip(data.times, truth, scores):
        idx = balanced_indices(y.ravel(), seed)
        per_time[str(time)] = auc(y.ravel()[idx], score.ravel()[idx])
        mask = data.prior.astype(bool)
        in_prior[str(time)] = auc(y[mask], score[mask])
        if evaluation_mask is not None:
            heldout[str(time)] = auc(y[evaluation_mask], score[evaluation_mask])
    if smoothness_metric not in {"frobenius", "squared_frobenius"}:
        raise ValueError("smoothness_metric must be frobenius or squared_frobenius")
    distances = np.array(
        [
            [
                np.square(a - b).sum()
                if smoothness_metric == "squared_frobenius"
                else np.linalg.norm(a - b)
                for b in scores
            ]
            for a in scores
        ]
    )
    distances /= distances.max() if distances.max() else 1
    values = [v for v in per_time.values() if v is not None]
    dynamic_values = [v for v in in_prior.values() if v is not None]
    heldout_values = [v for v in heldout.values() if v is not None]
    return {
        "auroc": per_time,
        "auroc_mean": float(np.mean(values)) if values else None,
        "auroc_within_union": in_prior,
        "auroc_within_union_mean": float(np.mean(dynamic_values))
        if dynamic_values
        else None,
        "auroc_heldout_candidate": heldout if evaluation_mask is not None else None,
        "auroc_heldout_candidate_mean": (
            float(np.mean(heldout_values)) if heldout_values else None
        ),
        "smoothness": float(1 - distances.mean()),
        "smoothness_metric": smoothness_metric,
        "evaluation": (
            "TG x TF flatten; fixed held-out candidate edges matching DRAFT's split"
            if evaluation_mask is not None
            else "TG x TF flatten; all positives plus 1:1 sampled negatives, same model-seed RNG as DRAFT; within-union AUROC uses all union candidates"
        ),
    }


def completed(folder, identity):
    path = folder / "complete.json"
    if not path.exists():
        return False
    record = read_json(path)
    if record["identity"] != identity:
        raise ValueError(f"Run identity changed: {folder}")
    for name, expected in record["outputs"].items():
        if digest(folder / name) != expected:
            raise ValueError(f"Completed output changed: {folder / name}")
    return True


def mark_complete(folder, identity):
    outputs = {
        str(p.relative_to(folder)): digest(p)
        for p in sorted(folder.rglob("*"))
        if p.is_file()
        and p.name not in {"complete.json", ".running.lock"}
        and not p.name.endswith(".part")
    }
    write_json(folder / "complete.json", {"identity": identity, "outputs": outputs})


def acquire_lock(folder):
    path = folder / ".running.lock"
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, f"pid={os.getpid()}\n".encode())
    os.close(fd)
    return path
