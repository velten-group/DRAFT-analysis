"""Fixed-edge-count random network of the DRAFT notebook, without model fitting or inference."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scdori_mondo"))
sys.path.insert(0, str(HERE.parents[1] / "src"))

from compare_stage_hic import digest, read_json
from hic_common import map_positives, region_key

from drosophila_gnn.preprocess import build_re_overlap_index, fix_dash, map_re_to_known


def sample_random_edges(k, n_re, n_tg, rng, batch=200_000):
    """Notebook's batched unique Cartesian-product draw and uniform trimming."""
    if k < 0 or n_re < 1 or n_tg < 1 or k > n_re * n_tg:
        raise ValueError("Invalid random edge budget or Cartesian-product dimensions.")
    if k == 0:
        return np.empty(0, dtype=np.int64)
    selected = set()
    while len(selected) < k:
        count = max(batch, int((k - len(selected)) * 1.5))
        re_idx = rng.integers(0, n_re, size=count, dtype=np.int64)
        tg_idx = rng.integers(0, n_tg, size=count, dtype=np.int64)
        selected.update(np.unique(re_idx * n_tg + tg_idx).tolist())
    codes = np.fromiter(selected, dtype=np.int64)
    if len(codes) > k:
        codes = rng.choice(codes, size=k, replace=False)
    return np.sort(codes)


def set_metrics(selected, positives):
    """Full predicted/positive edge sets: no sampled negatives or fitted cutoff."""
    hit = int(np.intersect1d(selected, positives, assume_unique=True).size)
    k, h = len(selected), len(positives)
    return {
        "n_selected": k,
        "n_full_pool_hic_positives": h,
        "n_hits": hit,
        "precision": hit / k if k else 0.0,
        "recall": hit / h if h else 0.0,
        "F1": 2 * hit / (k + h) if k + h else 0.0,
    }


def original_prior_codes(root, peaks, genes):
    """Match DRAFT's input graph RE–TG mapping/deduplication, before labels."""
    pi, gi = {p: j for j, p in enumerate(peaks)}, {g: j for j, g in enumerate(genes)}
    by_chr = build_re_overlap_index(peaks)
    table = pd.read_csv(root / "RE_TG_100kb.tsv", sep="\t", dtype=str)
    re_col = next((c for c in table if c.lower() in ["re", "peak", "region"]), None)
    tg_col = next((c for c in table if c.lower() in ["tg", "gene", "target"]), None)
    pairs = set()
    if re_col and tg_col:
        for region, gene in zip(table[re_col], table[tg_col]):
            name = fix_dash(str(region))
            mapped = name if name in pi else map_re_to_known(name, pi, by_chr)
            if mapped in pi and gene in gi:
                pairs.add(pi[mapped] * len(genes) + gi[gene])
    else:
        for row in table.itertuples(index=False, name=None):
            gene, name = str(row[0]), f"{row[4]}:{int(row[5])}-{int(row[6])}"
            if gene in gi and name in pi:
                pairs.add(pi[name] * len(genes) + gi[gene])
    return np.asarray(sorted(pairs), dtype=np.int64)


def prepare_context(cache, root=None):
    """Recover the original graph node pool/count from metadata and prior tables."""
    cache = Path(cache).resolve()
    frozen = read_json(cache / "cache_report.json")
    signature = frozen["signature"]
    comparison = Path(frozen["comparison"])
    if (
        digest(comparison / "comparison_report.json")
        != signature["parent_report_sha256"]
    ):
        raise ValueError("Original paired comparison report changed.")
    parent = read_json(comparison / "comparison_report.json")
    networks = Path(parent["stage_networks"])
    if digest(networks / "stage_report.json") != signature["stage_report_sha256"]:
        raise ValueError("Original stage report changed.")
    stage = read_json(networks / "stage_report.json")
    root = (
        Path(root or Path(stage["input_sources"]["TG.tsv"]["path"]).parent)
        .expanduser()
        .resolve()
    )
    sources = {}
    for name in ["TG.tsv", "RE.txt", "RE_TG_100kb.tsv", "A_rna_focus.h5ad"]:
        sha = digest(root / name)
        if sha != stage["input_sources"][name]["sha256"]:
            raise ValueError(f"Original Random graph input changed: {name}")
        sources[name] = {"path": str(root / name), "sha256": sha}
    for name in [
        "src/drosophila_gnn/preprocess.py",
        "src/drosophila_gnn/graph_builder.py",
    ]:
        if digest(HERE.parents[1] / name) != signature["draft_scoring_sources"][name]:
            raise ValueError(
                "Original DRAFT graph construction changed since score caching."
            )
    for assay in ["spg", "spc"]:
        path = root / f"{assay}_hic.tsv"
        if digest(path) != signature["truth_sources"][assay]["sha256"]:
            raise ValueError(f"Original Hi-C source changed: {assay}")
        sources[path.name] = {"path": str(path), "sha256": digest(path)}
    native = Path(frozen["scenic_run"])
    if digest(native / "r2g_report.json") != signature["native_report_sha256"]:
        raise ValueError("Native feature provenance changed.")
    native_report = read_json(native / "r2g_report.json")
    if digest(native / "manifest.json") != native_report["manifest_sha256"]:
        raise ValueError("Native manifest changed.")
    if (
        digest(native / "features.json")
        != read_json(native / "manifest.json")["files_sha256"]["features.json"]
    ):
        raise ValueError("Frozen common feature identifiers changed.")
    provenance = {
        "cache_report_sha256": digest(cache / "cache_report.json"),
        "sources": sources,
        "features_sha256": digest(native / "features.json"),
        "code_sha256": digest(Path(__file__)),
    }
    key = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()[
        :12
    ]
    out = cache / f"random_control_context_{key}"
    if (out / "context_report.json").exists():
        load_context(out, cache)
        return out
    tg = pd.read_csv(root / "TG.tsv", sep="\t", dtype=str)
    fb = next((c for c in tg if c.lower().startswith("fbgn")), None)
    if fb is None:
        raise ValueError("Original Random requires the project's FBgn target list.")
    # Read only the H5AD variable index; backed AnnData can still load layers.
    with h5py.File(root / "A_rna_focus.h5ad", "r") as rna:
        index_key = rna["var"].attrs["_index"]
        if isinstance(index_key, bytes):
            index_key = index_key.decode("utf-8")
        index = rna["var"][index_key]
        rna_ids = set(index.asstr()[()].tolist())
    genes = [g for g in pd.unique(tg[fb].dropna().astype(str)) if g in rna_ids]
    peaks = list(
        dict.fromkeys(
            line.strip()
            for line in (root / "RE.txt").read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    )
    if not genes or not peaks:
        raise ValueError("Original RE/TG graph pool is empty.")
    prior = original_prior_codes(root, peaks, genes)
    # DRAFT assigns this same unfiltered RE–TG prior to every stage graph.
    budgets = np.full(15, len(prior), dtype=np.int64)
    full_pi = {region_key(p): j for j, p in enumerate(peaks)}
    full_gi = {g: j for j, g in enumerate(genes)}
    if len(full_pi) != len(peaks):
        raise ValueError("Ambiguous full graph peak identifiers.")
    features = read_json(native / "features.json")
    with np.load(cache / "scores.npz", allow_pickle=False) as score:
        common_codes, common_n = score["pair_codes"], int(score["n_genes"])
        if common_n != len(features["genes"]):
            raise ValueError("Common score/feature dimensions differ.")
        map_p = np.asarray([full_pi[region_key(p)] for p in features["peaks"]])
        map_g = np.asarray([full_gi[g] for g in features["genes"]])
        candidate_codes = (
            map_p[common_codes // common_n] * len(genes)
            + map_g[common_codes % common_n]
        )
        if len(np.unique(candidate_codes)) != len(candidate_codes):
            raise ValueError(
                "Common candidates do not map to unique original graph node pairs."
            )
        # The DRAFT graph parser and baseline adapter can map the SAME source
        # file differently (e.g. literal FBgn IDs versus gene-symbol aliases).
        # Random is drawn from the full node Cartesian product, so evaluation
        # pairs need valid endpoints, not membership in the DRAFT prior mask.
        in_original_prior = np.isin(candidate_codes, prior)
        differences = pd.DataFrame(
            {
                "common_candidate_idx": np.flatnonzero(~in_original_prior),
                "full_pair_code": candidate_codes[~in_original_prior],
                "RE": np.asarray(peaks)[
                    candidate_codes[~in_original_prior] // len(genes)
                ],
                "TG_id": np.asarray(genes)[
                    candidate_codes[~in_original_prior] % len(genes)
                ],
            }
        )
        overlap = {
            "n_original_draft_prior_edges": len(prior),
            "n_common_candidates": len(candidate_codes),
            "n_common_candidates_in_original_prior": int(in_original_prior.sum()),
            "n_common_candidates_outside_original_prior": int(
                (~in_original_prior).sum()
            ),
            "policy": "retain all common evaluation candidates; original prior determines only the random-control edge budget; no prior intersection or model-score changes",
        }
    gold = {}
    for assay in ["spg", "spc"]:
        positives, _ = map_positives(root / f"{assay}_hic.tsv", peaks, genes, tg)
        gold[f"gold_{assay}"] = positives[:, 0] * len(genes) + positives[:, 1]
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        out / "context.npz",
        n_re=len(peaks),
        n_tg=len(genes),
        budgets=budgets,
        candidate_codes=candidate_codes,
        candidate_in_original_prior=in_original_prior,
        **gold,
    )
    differences.to_csv(out / "prior_mapping_differences.tsv.gz", sep="\t", index=False)
    (out / "full_features.json").write_text(
        json.dumps({"genes": genes, "peaks": peaks}), encoding="utf-8"
    )
    (out / "context_report.json").write_text(
        json.dumps(
            {
                "provenance": provenance,
                "cache_report_sha256": provenance["cache_report_sha256"],
                "validation_only": frozen["validation_only"],
                "pool": "original DRAFT RE.txt × TG.tsv genes present in RNA var_names; before common baseline intersection or within-prior restriction",
                "budget": "original graph's mapped unique RE–TG prior edge count, identical across its 15 stage graphs; not fitted using Hi-C labels",
                "prior_overlap": overlap,
                "files_sha256": {
                    name: digest(out / name)
                    for name in [
                        "context.npz",
                        "full_features.json",
                        "prior_mapping_differences.tsv.gz",
                    ]
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"Random-control context: original prior budget={len(prior)}; "
        f"common candidates={len(candidate_codes)}; "
        f"outside original graph prior={overlap['n_common_candidates_outside_original_prior']} "
        "(retained; see prior_mapping_differences.tsv.gz)",
        flush=True,
    )
    return out


def load_context(path, cache):
    path, cache = Path(path), Path(cache)
    report = read_json(path / "context_report.json")
    if report["cache_report_sha256"] != digest(cache / "cache_report.json"):
        raise ValueError("Random-control context belongs to a different score cache.")
    if (
        report["validation_only"]
        != read_json(cache / "cache_report.json")["validation_only"]
    ):
        raise ValueError("Random-control/model biological scopes differ.")
    for name, sha in report["files_sha256"].items():
        if digest(path / name) != sha:
            raise ValueError(f"Random-control context changed: {name}")
    with np.load(path / "context.npz", allow_pickle=False) as arrays:
        return {name: arrays[name] for name in arrays.files}


def draw_stages(context, seed):
    rng = np.random.default_rng(seed)
    # Draw all 15 in order, including GSC/Gb, to match the notebook RNG sequence.
    return [
        sample_random_edges(int(k), int(context["n_re"]), int(context["n_tg"]), rng)
        for k in context["budgets"]
    ]
