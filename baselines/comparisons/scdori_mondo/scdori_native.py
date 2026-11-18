"""Load verified upstream functions without importing unused FIMO dependencies."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
PIN = "f02cf47de1e25d03f1409edf79c1690681b72c70"
PREFIX = "_mondo_scdori_upstream"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_run_path(path: Path) -> Path:
    path = path.resolve()
    if not path.is_relative_to(HERE / "runs"):
        raise ValueError(f"Output must stay inside {HERE / 'runs'}: {path}")
    return path


def load_native(source: Path) -> dict:
    source = source.resolve()
    reference = read_json(HERE / "upstream_api.json")
    for name, expected in reference["sha256_lf"].items():
        path = source / name
        actual = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if actual != expected:
            raise RuntimeError(f"scDoRI source differs from release v1.0.0 at {name}.")
    for name, location in [
        (PREFIX, source),
        (PREFIX + "._core", source / "_core"),
        (PREFIX + ".pp", source / "pp"),
    ]:
        if name in sys.modules and sys.modules[name].__path__ != [str(location)]:
            raise RuntimeError("Use a fresh process to change upstream source.")
        if name not in sys.modules:
            namespace = types.ModuleType(name)
            namespace.__path__ = [str(location)]
            sys.modules[name] = namespace
    modules = {
        name: importlib.import_module(PREFIX + "." + name)
        for name in [
            "_core.config",
            "_core.models",
            "_core.data_io",
            "_core.evaluation",
            "_core.utils",
            "_core.train_scdori",
            "_core.train_grn",
            "pp.correlation",
            "pp.metacells",
        ]
    }
    return modules


def comparison_helpers():
    path = PROJECT / "comparisons/dictys_mondo/workflow.py"
    spec = importlib.util.spec_from_file_location("_mondo_existing_comparison", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_outputs(record: dict, base: Path) -> None:
    for name, expected in record["sha256"].items():
        if digest(base / name) != expected:
            raise ValueError(f"Saved output changed: {base / name}")


def restore(model, path: Path, device) -> None:
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise RuntimeError("Checkpoint contains nonfinite model parameters.")


def zero_evidence_repressors(
    activator, repressor
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Correct upstream MinMaxScaler's all-zero -> all-minus-one boundary case."""
    bad = np.flatnonzero(np.all(repressor == -1, axis=0))
    activator, repressor = activator.astype(np.float32), repressor.astype(np.float32)
    repressor[:, bad] = 0
    return activator, repressor, bad.tolist()


def effective_mondo_edges(
    model, topic_tf, mondo_index: int, target_indices: np.ndarray
) -> np.ndarray:
    """Extract unthresholded effective edges from the exact model.forward GRN equations.

    Normalization maxima/minima are calculated across ALL TFs/genes per topic,
    not just MONDO. Native permutation filtering is deliberately not substituted.
    """
    result = []
    with torch.no_grad():
        topic_peak = torch.softmax(model.topic_peak_decoder, dim=1)
        gene_peak = (model.gene_peak_factor_fixed * model.gene_peak_factor_learnt).T
        denominator = gene_peak.sum(dim=0, keepdim=True) + 1e-7
        for topic in range(model.num_topics):
            act = model.tf_binding_matrix_activator.T @ (
                topic_peak[topic, :, None] * gene_peak
            )
            act = act / denominator
            rep = model.tf_binding_matrix_repressor.T @ (
                (1 / (topic_peak[topic, :, None] + 1e-20)) * gene_peak
            )
            rep = rep / denominator
            act_row = act[mondo_index] / (act.max() + 1e-15)
            rep_row = rep[mondo_index] / (rep.min() + 1e-15)
            row = act_row * torch.relu(
                model.tf_gene_topic_activator_grn[topic, mondo_index]
            )
            row = row - rep_row * torch.relu(
                model.tf_gene_topic_repressor_grn[topic, mondo_index]
            )
            row = row * topic_tf[topic, mondo_index]
            result.append(row[target_indices].detach().cpu().numpy())
    edges = np.asarray(result, dtype=np.float32)
    if not np.isfinite(edges).all():
        raise RuntimeError("Nonfinite effective MONDO edges.")
    return edges
