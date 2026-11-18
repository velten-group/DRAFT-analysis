"""Settings, gold standards, node lists and the official scMTNI preparation script of the mouse AUPR (Fig. 3b).

Gene symbols of the method outputs are matched to the dataset's symbols ignoring case.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re

import numpy as np
import pandas as pd
from common import CLUSTERS, digest, gene_index, gene_key

COMMIT = "c4d4850629f3529cc7a8b4dae54d91daf0702809"
SOURCE_HASHES = {
    "prepareAUClistfile_intersect.py": "39543dfba0351af1facf5e6b466cc43262d769a61426fb6eaac70059ca4ce1d9",
    "auc.jar": "d193a28ec1297a8b16d858223e66f4fec7eaf3ec65e33e61dfbc0ec43e748c50",
}
GOLDS = {
    "ChIP": ("mESC_chipunion.txt", "8629656f0958fa713a5473760d1a9d33d341f24fe6efe78bf243de7ccb59473d"),
    "Perturb": ("mESC_KDUnion.txt", "1b24baf64ed1991c87d721b6cef807e951b6f787e25d20f0509b6462b5d7aece"),
    "Perturb+ChIP": ("mESC_chipunion_KDUnion_intersect.txt", "80da24f556684ef088501262f34296ea0f3efdb24c1aa84ce04b10b4da97963d"),
}
PROTOCOL = {
    "schema": "mouse_official_aupr_v1",
    "metric": "AUPR_official_auc_jar",
    "draft_output": "full_checkpoint_scores",
    "source_commit": COMMIT,
    "source_sha256": SOURCE_HASHES,
    "source_text_hash_policy": "canonical UTF-8 LF; matches the pinned Git blob across Windows/Linux",
    "gold_sha256": {name: value[1] for name, value in GOLDS.items()},
    "node_scope": "per-cluster archived regulators/genes; original symbol case; remove _clusterN suffix",
    "evaluation_universe": "gold-covered TF x TG; remove self-pairs; no prior-edge restriction",
    "missing_scores": 0,
    "balanced_negative_sampling": False,
    "score_serialization": "%g, exactly as official preparation script",
    "summary_unit": "7 cell-cluster contexts; not training seeds or biological replicates",
}


def clean_symbol(value):
    return re.sub(r"_cluster\d+$", "", str(value).strip())


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_digest(path):
    data = path.read_bytes()
    if path.suffix == ".py":
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def load_nodes(data_dir):
    nodes = {}
    for cid in CLUSTERS:
        nodes[cid] = {}
        for key, suffix in [("regulators", "allregulators"), ("targets", "allGenes")]:
            path = data_dir / f"cluster{cid}_{suffix}.txt"
            names = [clean_symbol(x) for x in path.read_text().splitlines() if x.strip()]
            if len(names) != len(set(names)) or any("\t" in g for g in names):
                raise ValueError(f"Malformed/duplicate one-symbol-per-line file: {path}")
            nodes[cid][key] = names
    return nodes


def load_gold_tables(gold_dir):
    tables = {}
    for name, (filename, expected) in GOLDS.items():
        path = gold_dir / filename
        if digest(path) != expected:
            raise ValueError(f"Official gold checksum mismatch: {name}")
        table = pd.read_csv(path, sep="\t", header=None, names=["TF", "TG"], dtype=str)
        if table.isna().any().any():
            raise ValueError(f"Malformed official gold: {name}")
        tables[name] = table
    return tables


def evaluation_scope(nodes, tables, protocol=None):
    records = []
    signature = {"protocol": PROTOCOL if protocol is None else protocol, "nodes": {}}
    for cid in CLUSTERS:
        regs, genes = set(nodes[cid]["regulators"]), set(nodes[cid]["targets"])
        signature["nodes"][f"C{cid}"] = {
            "regulators": sorted(regs), "targets": sorted(genes),
        }
        for name, table in tables.items():
            gold = table.loc[table.TF.isin(regs) & table.TG.isin(genes) & table.TF.ne(table.TG)]
            tf_set, tg_set = set(gold.TF), set(gold.TG)
            if len(tf_set) <= 1:
                raise ValueError("This frozen mouse protocol requires multiple covered TFs per context.")
            records.append({"gold_standard": name, "cluster": f"C{cid}",
                            "n_positive": len(gold),
                            "n_pairs": len(tf_set) * len(tg_set) - len(tf_set & tg_set),
                            "filtered_gold_sha256": content_hash(sorted(map(tuple, gold.values)))})
    return content_hash(signature), pd.DataFrame(records)


def load_official_preparer(source_dir, destination):
    for name, expected in SOURCE_HASHES.items():
        if source_digest(source_dir / name) != expected:
            raise ValueError(f"Official source checksum mismatch: {name}")
    source = (source_dir / "prepareAUClistfile_intersect.py").read_text(encoding="utf-8")
    if source.count("sep='\\n'") != 2:
        raise ValueError("Unexpected official one-column file reader.")
    destination.write_text(source.replace("sep='\\n'", "sep='\\t'"), encoding="utf-8", newline="\n")
    spec = importlib.util.spec_from_file_location("pinned_official_aupr_preparer", destination)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def canonical_edges(frame, tfs, tgs):
    """Native scores on the model's TF and TG symbols (matched ignoring case); the maximum for duplicate pairs."""
    if not {"TF", "TG", "score"} <= set(frame.columns):
        raise ValueError("Native edge table must have TF, TG and score columns.")
    tf_lookup = {gene_key(x): x for x in tfs}
    tg_lookup = {gene_key(x): x for x in tgs}
    gene_index(tfs)
    gene_index(tgs)
    scores = pd.to_numeric(frame.score, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("Native magnitudes must be finite and nonnegative; do not silently transform scores.")
    out = pd.DataFrame({"TF": frame.TF.map(gene_key).map(tf_lookup),
                        "TG": frame.TG.map(gene_key).map(tg_lookup), "score": scores})
    unmapped = int(out[["TF", "TG"]].isna().any(axis=1).sum())
    out = out.dropna(subset=["TF", "TG"])
    out = out.groupby(["TF", "TG"], sort=False, as_index=False).score.max()
    return out, {"native_rows": len(frame), "unmapped_rows_outside_fixed_axes": unmapped,
                 "canonical_unique_rows": len(out),
                 "score_rule": "native nonnegative magnitude; normalized duplicate pair maximum; no threshold"}
