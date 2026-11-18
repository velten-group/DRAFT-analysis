"""Portable mouse inputs and scores; no external model framework imports."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
COMPARISONS = HERE.parent
CLUSTERS = [4, 8, 7, 3, 2, 1, 5]
PARENTS = {4: None, 8: 4, 7: 8, 3: 7, 2: 3, 1: 8, 5: 1}
IDENTIFIER_POLICY = "literal_mouse_symbols; cluster_suffix_removed; Ensembl_version_only; case_insensitive_matching"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def inside(path: Path, root: Path) -> Path:
    path, root = path.resolve(), root.resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Outputs must stay inside {root}: {path}")
    return path


def new_run(root: Path, prefix: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = root / f"{prefix}_{stamp}_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def gene_name(value: str) -> str:
    value = re.sub(r"_cluster\d+$", "", str(value).strip(), flags=re.IGNORECASE)
    # Epb4.1 and Tex19.1 are mouse gene symbols, not identifier versions.
    return re.sub(r"^(ENS[A-Z]*G\d+)\.\d+$", r"\1", value, flags=re.IGNORECASE)


def gene_key(value: str) -> str:
    return gene_name(value).upper()


def gene_index(names: list[str]) -> dict[str, int]:
    keys = [gene_key(name) for name in names]
    if len(set(keys)) != len(keys):
        raise ValueError("Ambiguous gene identifiers after suffix/case normalization.")
    return {name: i for i, name in enumerate(keys)}


def load_expression(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", index_col=0)
    frame.index = frame.index.map(gene_key)
    if frame.index.has_duplicates:
        frame = frame.groupby(level=0, sort=True).max()  # original notebook rule
    if frame.columns.has_duplicates or frame.empty:
        raise ValueError(f"Empty expression or duplicate cell identifiers: {path}")
    values = frame.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"Expression must be finite and nonnegative: {path}")
    return frame.astype(np.float32)


def validate_counts(frame: pd.DataFrame) -> None:
    values = frame.to_numpy()
    if not np.allclose(values, np.rint(values), atol=1e-5, rtol=0):
        raise ValueError(
            "Dictys requires original integer RNA counts. Supply --counts-root; "
            "do not round, exponentiate, or substitute normalized expression."
        )


def load_network(path: Path, threshold: float, header: str = "auto") -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", header=None)
    if frame.shape[1] < 2:
        raise ValueError(f"Expected TF/TG columns: {path}")
    first = tuple(gene_key(x) for x in frame.iloc[0, :2])
    has_header = first in {("TF", "TG"), ("TF", "TARGET"), ("SOURCE", "TARGET")}
    if header == "present" or (header == "auto" and has_header):
        frame = frame.iloc[1:].copy()
    if threshold > 0:
        if frame.shape[1] < 3:
            raise ValueError(
                f"A score column is required for threshold {threshold}: {path}"
            )
        frame = frame[pd.to_numeric(frame.iloc[:, 2], errors="raise") > threshold]
    result = frame.iloc[:, :2].copy()
    result.columns = ["TF", "TG"]
    result["TF"] = result.TF.map(gene_key)
    result["TG"] = result.TG.map(gene_key)
    return result.drop_duplicates().reset_index(drop=True)


def read_gold(path: Path, header: str = "auto") -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", header=None, dtype=str)
    if frame.shape[1] != 2 or frame.isna().any().any():
        raise ValueError("Gold must have exactly two nonempty TF/TG columns.")
    first = tuple(gene_key(x) for x in frame.iloc[0])
    has_header = first in {("TF", "TG"), ("TF", "TARGET"), ("SOURCE", "TARGET")}
    if header == "present" or (header == "auto" and has_header):
        frame = frame.iloc[1:].copy()
    frame.columns = ["TF", "TG"]
    frame["TF"] = frame.TF.map(gene_key)
    frame["TG"] = frame.TG.map(gene_key)
    return frame.drop_duplicates().reset_index(drop=True)


def edge_codes(frame: pd.DataFrame, tfs: list[str], tgs: list[str]) -> np.ndarray:
    tf_idx, tg_idx = gene_index(tfs), gene_index(tgs)
    i, j = frame.TF.map(tf_idx), frame.TG.map(tg_idx)
    valid = i.notna() & j.notna()
    return np.unique(
        i[valid].to_numpy(dtype=np.int64) * len(tgs) + j[valid].to_numpy(dtype=np.int64)
    )


def resolve_prepared(path: Path) -> Path:
    if path == Path("latest"):
        candidates = [
            p
            for p in (HERE / "runs").glob("shared_prior_*")
            if (p / "manifest.json").exists()
            and not read_json(p / "manifest.json").get("validation_only")
        ]
        if not candidates:
            raise FileNotFoundError(
                "No prepared biological mouse run; run prepare first."
            )
        path = max(candidates, key=lambda p: (p / "manifest.json").stat().st_mtime_ns)
    path = inside(path, HERE / "runs")
    verify_manifest(path)
    return path


def verify_manifest(path: Path) -> dict:
    manifest = read_json(path / "manifest.json")
    if (
        not manifest.get("validation_only")
        and manifest.get("identifier_policy") != IDENTIFIER_POLICY
    ):
        raise ValueError(
            "Prepared run uses the old gene-dot rule. Prepare a new run so mouse symbols such as Epb4.1 and Tex19.1 are preserved."
        )
    for name, expected in manifest["files_sha256"].items():
        if digest(path / name) != expected:
            raise ValueError(f"Prepared input changed: {path / name}")
    return manifest


def mark_complete(folder: Path, metadata: dict) -> None:
    metadata = dict(metadata)
    metadata["files_sha256"] = {
        path.name: digest(path)
        for path in folder.iterdir()
        if path.is_file() and path.name != "complete.json"
    }
    write_json(folder / "complete.json", metadata)


def verify_complete(folder: Path) -> dict:
    record = read_json(folder / "complete.json")
    for name, expected in record["files_sha256"].items():
        if digest(folder / name) != expected:
            raise ValueError(f"Completed output changed: {folder / name}")
    return record


def model_run(model: str, prepared: Path, configuration: dict) -> Path:
    root = COMPARISONS / model / "runs"
    output = inside(root / prepared.name, root)
    output.mkdir(parents=True, exist_ok=True)
    runtime = {
        "prepared": str(prepared),
        "manifest_sha256": digest(prepared / "manifest.json"),
        "configuration": configuration,
    }
    path = output / "runtime.json"
    if path.exists() and read_json(path) != runtime:
        raise ValueError(
            "Resume settings/software differ. Prepare a new run to change them."
        )
    if not path.exists():
        write_json(path, runtime)
    return output


def validate_model_run(model: str, prepared: Path, path: Path | None) -> Path:
    root = COMPARISONS / model / "runs"
    path = inside(path or root / prepared.name, root)
    runtime = read_json(path / "runtime.json")
    if runtime["manifest_sha256"] != digest(prepared / "manifest.json"):
        raise ValueError(f"Model does not use the selected prepared inputs: {path}")
    return path


def project_edges(
    edges: pd.DataFrame, codes: np.ndarray, tfs: list[str], tgs: list[str]
) -> np.ndarray:
    """TF/TG names, never a guessed matrix orientation, determine score alignment."""
    required = {"TF", "TG", "score"}
    if not required.issubset(edges.columns):
        raise ValueError(f"Portable edges need {required}.")
    edges = edges.copy()
    edges["TF"] = edges.TF.map(gene_key)
    edges["TG"] = edges.TG.map(gene_key)
    if not np.isfinite(edges.score.to_numpy()).all() or (edges.score < 0).any():
        raise ValueError("Evaluation scores must be finite and nonnegative.")
    tf_idx, tg_idx = gene_index(tfs), gene_index(tgs)
    i, j = edges.TF.map(tf_idx), edges.TG.map(tg_idx)
    valid = i.notna() & j.notna()
    mapped = (
        pd.DataFrame(
            {
                "code": i[valid].to_numpy(dtype=np.int64) * len(tgs)
                + j[valid].to_numpy(dtype=np.int64),
                "score": edges.loc[valid, "score"].to_numpy(),
            }
        )
        .groupby("code")
        .score.max()
    )
    return mapped.reindex(codes, fill_value=0).to_numpy(dtype=np.float64)
