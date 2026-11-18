"""Cell-cluster order and gene-name keys shared by the mouse scripts."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

CLUSTERS = [4, 8, 7, 3, 2, 1, 5]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


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
