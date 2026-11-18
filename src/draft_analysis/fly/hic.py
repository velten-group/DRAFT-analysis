"""Fig. 4b scores recomputed: DRAFT from the germline checkpoint, SCENIC+ per-stage R2G from its fitted tables.

The candidate universe (120,111 RE-TG pairs of the shared prior), the scDoRI stage coefficients, the Random edges and
the balanced Hi-C pairs are deposited inputs (data/fly/external/hic_within_prior). This stage scores the candidates:
DRAFT with the sigmoid decoder of the reloaded checkpoint, SCENIC+ with the native GBM importance of each stage fit,
mapped through the SCENIC+ feature index (features.json); a candidate absent from a stage fit scores zero. The DRAFT
scores are compared with the published scores.

Output: results/fly/germ/hic_within_prior/scores.npz (pair_codes, draft, scenicplus_per_stage) and check.json.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config, paths

N_STAGES = 15


def _hic() -> Path:
    return paths.data("fly", "external", "hic_within_prior")


def cache_dir() -> Path:
    return _hic() / "score_cache"


def evaluation_dir() -> Path:
    return _hic() / "stage_evaluation"


def out_dir() -> Path:
    return paths.FLY / "germ" / "hic_within_prior"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def region_key(value: str) -> str:
    """Peak identifier in the normalised form of the Hi-C evaluation."""
    text = str(value).strip().replace(" ", "").replace("_", "-").replace("..", "-")
    text = text.replace("–", "-").replace("—", "-")
    return re.sub(r"^chr", "", text, flags=re.IGNORECASE)


def features() -> dict:
    """Genes and peaks of the SCENIC+ run; candidate code = peak index * n_genes + gene index."""
    native = paths.data("fly", "external", "scenicplus_native_r2g")
    return json.loads((native / "features.json").read_text(encoding="utf-8"))


def scenic_stage_scores(codes: np.ndarray, feats: dict) -> np.ndarray:
    """(stage, candidate) native R2G importance of the per-stage SCENIC+ fits (as baselines/comparisons/hic_within_prior/stage_r2g.py)."""
    files = sorted((_hic() / "scenicplus_stage_r2g").glob("t[0-9][0-9]_region_to_gene.tsv.gz"))
    genes = {name: j for j, name in enumerate(feats["genes"])}
    peaks = {name: j for j, name in enumerate(feats["peaks"])}
    scores = np.zeros((N_STAGES, len(codes)), dtype=np.float64)
    if [int(f.name[1:3]) for f in files] != list(range(N_STAGES)):
        raise ValueError("per-stage R2G must cover all fifteen stages")
    for path in files:
        name = path.name
        table = pd.read_csv(path, sep="\t")
        pair = table.region.map(peaks).to_numpy() * len(genes) + table.target.map(genes).to_numpy()
        pos = np.searchsorted(codes, pair)
        inside = pos < len(codes)
        inside[inside] &= codes[pos[inside]] == pair[inside]
        if not inside.all():
            raise ValueError("per-stage R2G scores a pair outside the candidates")
        scores[int(name[1:3]), pos] = table.importance.to_numpy()
    return scores


def draft_scores(codes: np.ndarray, feats: dict) -> np.ndarray:
    """(stage, candidate) RE-TG probabilities of the germline checkpoint (as baselines/comparisons/hic_within_prior/score_cache.py)."""
    import torch

    from drosophila_gnn.analysis.repro import score_pairs_over_time_sigmoid
    from drosophila_gnn.infer import reload_germ_from_checkpoint

    cfg = config.load("germ_reproduce", {"reproduce": {"checkpoint_path": str(config.checkpoint("germ"))}})
    res = reload_germ_from_checkpoint(cfg)
    if res["germline_types"] != [f"Germline_{t}" for t in range(N_STAGES)]:
        raise ValueError("DRAFT stage order differs")
    genes, peaks = feats["genes"], feats["peaks"]
    tg = pd.read_csv(paths.data("fly", "raw", "TG.tsv"), sep="\t", dtype=str).fillna("")
    aliases = dict(zip(tg.gene_name, tg.FBgn_ID))
    aliases.update({g: g for g in tg.FBgn_ID})
    dp = {region_key(str(p)): j for j, p in enumerate(res["maps"]["re"])}
    dg = {aliases.get(str(g), str(g)): j for j, g in enumerate(res["maps"]["tg"])}
    if len(dp) != len(res["maps"]["re"]) or len(dg) != len(res["maps"]["tg"]):
        raise ValueError("ambiguous DRAFT node identifiers")
    p, g = codes // len(genes), codes % len(genes)
    pairs = torch.as_tensor(np.column_stack([[dp[region_key(peaks[j])] for j in p], [dg[genes[j]] for j in g]]),
                            dtype=torch.long, device=res["out_re"].device)
    return np.stack(score_pairs_over_time_sigmoid(res["out_re"], res["out_tg"], pairs))


def run() -> dict:
    with np.load(cache_dir() / "scores.npz") as z:
        codes, n_genes, cached = z["pair_codes"], int(z["n_genes"]), z["draft"]
    feats = features()
    if len(feats["genes"]) != n_genes:
        raise ValueError("feature index and score cache disagree on the gene count")
    draft = draft_scores(codes, feats)
    scenic = scenic_stage_scores(codes, feats)
    out = out_dir()
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "scores.npz", pair_codes=codes, draft=draft, scenicplus_per_stage=scenic)
    diff = np.abs(draft.astype(np.float64) - cached.astype(np.float64))
    check = {"n_candidates": len(codes), "draft_vs_published_scores": {
        "identical": bool(np.array_equal(draft, cached)), "n_differing": int((diff > 0).sum()),
        "max_abs_diff": float(diff.max())},
        "scenicplus_nonzero_share_by_stage": [float((s > 0).mean()) for s in scenic]}
    (out / "check.json").write_text(json.dumps(check, indent=2), encoding="utf-8")
    print(f"   DRAFT vs published scores: identical={check['draft_vs_published_scores']['identical']}, "
          f"max |diff| {check['draft_vs_published_scores']['max_abs_diff']:.2e}", flush=True)
    return check
