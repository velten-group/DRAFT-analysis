"""Read cells of the original mouse notebooks (original_notebooks/) and check them against their SHA-256."""

from __future__ import annotations

import hashlib
import json

TOPIC_SHA = "53333f797f7556641429c8af381bb2c5e5dfe4418842289bdeb8c2d47ef6a002"
ENRICH_SHA = "bbd16599ad31c7942baa6363886cc93f806a5fc1818e42f7b5e898024d9e8689"


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def cell_source(path, index, expected=None):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][index]["source"])
    actual = hashlib.sha256(source.encode()).hexdigest()
    if expected is not None and actual != expected:
        raise ValueError(f"Notebook cell differs from the expected source: {path}, cell {index}")
    return source
