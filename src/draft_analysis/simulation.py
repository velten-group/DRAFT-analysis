"""Simulation benchmark (Fig. 2, Supplementary Figs. 1-4), scored again from the saved predictions of every fit.

No model is trained or loaded. Each fit's final TF-TG scores (its relations.npz) are scored
against the ground-truth GRNs of its simulated dataset with the metrics of the manuscript. Inputs, under
data/simulation/:

    datasets/<dataset>/seed_<data>/        one simulated dataset: RNA counts (counts_time_<t>.csv), ATAC and the
                                           TF-RE / RE-TG priors (multiome/), the candidate TF-TG prior
                                           (candidate_prior.csv), the ground-truth GRN of each stage (grn_time_<t>.csv)
                                           and the gene modules; the prior-size controls add their fixed evaluation
                                           panel (evaluation_mask.csv), the branching lineage its tree (tree.tsv)
    fits/<study>/<case>/<method>/data<data>_seed<seed>.npz   the final scores of one fit (`scores`; the internal
                                           models also save `logits`)
    fits.tsv                               one row per fit: study, case, method, data, seed, dataset, file, sha256

Outputs in results/checkpoint/simulation/: runs.tsv (one row per fit and per Random control), per_time.tsv (AUROC per
stage), distances.tsv (stage-by-stage Frobenius distances of the main design), references.tsv (prior SNR and
ground-truth smoothness of each dataset), real_prior_snr.tsv (fly data prior), tsne_branch_101.tsv.
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from sklearn.metrics import roc_auc_score

from . import paths

INTERNAL = ("draft", "linear_gru", "pooled_linear", "pooled_gat")
OFFICIAL = ("scenicplus", "scmtni", "scdori", "dictys")
NAME = {"draft": "DRAFT", "dictys": "Dictys", "scdori": "scDoRI", "scmtni": "scMTNI", "scenicplus": "SCENIC+",
        "random": "Random", "ground_truth": "Ground truth", "pooled_gat": "Shared GAT", "linear_gru": "LR+GRU",
        "pooled_linear": "Shared LR"}
MAIN = "linear"  # three-module linear design and its sensitivity cases
RANDOM_NETWORK = "random_network"
PRIOR_SIZE = "prior_size"
BRANCH = "branching"
# Random control: independent uniform scores on the candidate pairs, numpy default_rng(model seed)
RANDOM_SEEDS = {MAIN: (0, 1, 2, 3, 4), RANDOM_NETWORK: (0, 1, 2)}
# branching lineage: states in topological order and the parent of each
STATES = ("T0", "T1", "A2", "A3", "A4", "B2", "B3", "B4")
PARENTS = (-1, 0, 1, 2, 3, 1, 5, 6)
BRANCH_PAIRS = ((2, 5), (3, 6), (4, 7))  # A_k and B_k at the same depth
TRUNK = (0, 1)


def root() -> Path:
    return paths.DATA / "simulation"


def out(*parts: str) -> Path:
    return paths.result("simulation", *parts)


# --------------------------------------------------------------------------------------------- inputs
def _matrix(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    frame.index, frame.columns = frame.index.astype(str), frame.columns.astype(str)
    return frame


class Dataset:
    """Candidate panel, ground truth and module membership of one simulated dataset (TG x TF matrices)."""

    def __init__(self, folder: Path):
        prior = _matrix(folder / "candidate_prior.csv")
        self.tfs, self.targets = list(prior.columns), list(prior.index)
        self.candidate = prior.to_numpy().astype(bool)
        times = len(list(folder.glob("grn_time_*.csv")))
        self.truth = np.stack([_matrix(folder / f"grn_time_{t}.csv").loc[self.targets, self.tfs].to_numpy() > 0
                               for t in range(times)])
        modules = pd.read_csv(folder / "gene_modules.tsv", sep="\t", dtype={"gene": str}).set_index("gene").module
        self.same = modules.loc[self.targets].to_numpy()[:, None] == modules.loc[self.tfs].to_numpy()[None, :]
        # the prior-size controls are all scored on the main design's 1,800 candidates (their evaluation panel)
        panel = folder / "evaluation_mask.csv"
        self.panel = (_matrix(panel).loc[self.targets, self.tfs].to_numpy().astype(bool) if panel.exists()
                      else self.candidate)
        self.folder = folder


def load_fit(path: Path, dataset: Dataset, sha256: str) -> tuple[np.ndarray, np.ndarray]:
    """(scores, ranking) of one fit; the internal models are ranked by their logits, the others by their scores."""
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError(f"{path}: SHA-256 differs from fits.tsv")
    with np.load(io.BytesIO(data), allow_pickle=False) as saved:
        # the prior-size fits store no axes: they are in the order of the dataset's candidate_prior.csv
        if "tfs" in saved.files and (list(saved["tfs"].astype(str)) != dataset.tfs
                                     or list(saved["targets"].astype(str)) != dataset.targets):
            raise ValueError(f"{path}: TF/TG axes differ from the dataset")
        scores = np.asarray(saved["scores"], dtype=np.float64)
        ranking = np.asarray(saved["logits"], dtype=np.float64) if "logits" in saved.files else scores
    if scores.shape != dataset.truth.shape or not np.isfinite(scores).all() or not np.isfinite(ranking).all():
        raise ValueError(f"{path}: scores do not match the dataset")
    return scores, ranking


# --------------------------------------------------------------------------------------------- metrics
def distances(values: np.ndarray) -> np.ndarray:
    """Frobenius distances between the stage networks (rows of `values`)."""
    flat = np.asarray(values, dtype=np.float64).reshape(len(values), -1)
    return np.linalg.norm(flat[:, None] - flat[None], axis=2)


def smoothness(values: np.ndarray, pairs: list[tuple[int, int]] | None = None) -> float:
    """Temporal smoothness: 1 - mean distance between adjacent stages (or the given parent-child pairs) / largest
    distance between any two stages."""
    flat = np.asarray(values, dtype=np.float64).reshape(len(values), -1)
    diameter = max(np.linalg.norm(x - y) for x in flat for y in flat)
    if diameter == 0:
        return float("nan")
    if pairs is None:
        adjacent = np.linalg.norm(np.diff(flat, axis=0), axis=1)
    else:
        adjacent = np.array([np.linalg.norm(flat[c] - flat[p]) for p, c in pairs])
    return float(1 - adjacent.mean() / diameter)


def _auc(y: np.ndarray, x: np.ndarray) -> float:
    """AUROC, or NaN when only one class is present (e.g. no cross-module candidates in the random network)."""
    return float(roc_auc_score(y, x)) if 0 < y.sum() < y.size else float("nan")


def score_chain(scores: np.ndarray, ranking: np.ndarray, d: Dataset) -> tuple[list[float], dict]:
    """Candidate AUROC per stage on the evaluation panel and the run-level metrics of a linear time course."""
    mask, truth = d.panel, d.truth
    per_time = [float(roc_auc_score(y[mask], x[mask])) for y, x in zip(truth, ranking)]
    inside = mask & d.same
    outside = [mask & (y | ~d.same) for y in truth]  # true edges against the cross-module candidates only
    changed = truth[1:] != truth[:-1]
    delta = np.abs(np.diff(ranking, axis=0))
    usable = [t for t in range(len(changed)) if 0 < changed[t][mask].sum() < mask.sum()]
    return per_time, {
        "candidate_auroc": float(np.mean(per_time)),
        "same_module_auroc": float(np.mean([_auc(y[inside], x[inside]) for y, x in zip(truth, ranking)])),
        "true_vs_outside_auroc": float(np.mean([_auc(y[m], x[m]) for y, x, m in zip(truth, ranking, outside)])),
        "change_auroc": (float(np.mean([_auc(changed[t][mask], delta[t][mask]) for t in usable]))
                         if usable else float("nan")),
        "mean_continuity": smoothness(scores[:, mask]),
    }


def tree_pairs() -> list[tuple[int, int]]:
    return [(parent, child) for child, parent in enumerate(PARENTS) if parent >= 0]


def score_tree(scores: np.ndarray, ranking: np.ndarray, d: Dataset) -> tuple[list[float], dict]:
    """The branching lineage: AUROC per state, smoothness and change AUROC over the seven parent-child pairs, and
    branch AUROC (among candidates active in exactly one of A_k and B_k, does logit(A_k) - logit(B_k) tell which)."""
    mask, truth = d.candidate, d.truth
    per_state = np.array([roc_auc_score(y[mask], x[mask]) for y, x in zip(truth, ranking)])
    inside = mask & d.same
    outside = [mask & (y | ~d.same) for y in truth]
    change = [roc_auc_score((truth[c] != truth[p])[mask], np.abs(ranking[c] - ranking[p])[mask])
              for p, c in tree_pairs()]
    branch = []
    for a, b in BRANCH_PAIRS:
        differ = mask & (truth[a] != truth[b])
        branch.append(roc_auc_score(truth[a][differ], (ranking[a] - ranking[b])[differ]))
    return [float(v) for v in per_state], {
        "candidate_auroc": float(per_state.mean()),
        "trunk_auroc": float(per_state[list(TRUNK)].mean()),
        "branch_states_auroc": float(np.delete(per_state, TRUNK).mean()),
        "same_module_auroc": float(np.mean([roc_auc_score(y[inside], x[inside]) for y, x in zip(truth, ranking)])),
        "true_vs_outside_auroc": float(np.mean([roc_auc_score(y[m], x[m]) for y, x, m in zip(truth, ranking, outside)])),
        "change_auroc": float(np.mean(change)),
        "branch_auroc": float(np.mean(branch)),
        "mean_continuity": smoothness(scores[:, mask], tree_pairs()),
    }


# --------------------------------------------------------------------------------------------- prior SNR
def _second_singular_value(matrix: sp.csr_matrix) -> float:
    row, col = np.asarray(matrix.sum(1)).ravel(), np.asarray(matrix.sum(0)).ravel()
    normalised = sp.diags(1 / np.sqrt(row.clip(1))) @ matrix @ sp.diags(1 / np.sqrt(col.clip(1)))
    return float(np.sort(svds(normalised.astype(np.float64), k=3, random_state=0, return_singular_vectors=False))[-2])


def _degree_shuffle(matrix: sp.csr_matrix, seed: int, passes: int = 20) -> sp.csr_matrix:
    """Curveball randomisation: two random rows repeatedly trade the columns they do not share."""
    matrix = sp.csr_matrix(matrix)
    rng = np.random.default_rng(seed)
    rows = [set(matrix.indices[matrix.indptr[i]: matrix.indptr[i + 1]].tolist()) for i in range(matrix.shape[0])]
    for i, j in rng.integers(0, matrix.shape[0], (passes * matrix.shape[0], 2)).tolist():
        only_i, only_j = rows[i] - rows[j], rows[j] - rows[i]
        if i == j or not only_i or not only_j:
            continue
        pool = rng.permutation(sorted(only_i | only_j)).tolist()
        rows[i] = (rows[i] - only_i).union(pool[: len(only_i)])
        rows[j] = (rows[j] - only_j).union(pool[len(only_i):])
    indices = np.repeat(np.arange(matrix.shape[0]), [len(row) for row in rows])
    columns = np.fromiter((c for row in rows for c in sorted(row)), dtype=int, count=indices.size)
    return sp.csr_matrix((np.ones(indices.size, np.float32), (indices, columns)), shape=matrix.shape)


def prior_snr(candidate) -> dict:
    """Prior SNR, the measure of modularity: second singular value of the degree-normalised binary TG x TF candidate
    matrix divided by the largest one of three degree-preserving shuffles. Reads the prior only."""
    matrix = sp.csr_matrix(candidate.astype(np.float32))
    signal = _second_singular_value(matrix)
    ceiling = max(_second_singular_value(_degree_shuffle(matrix, seed)) for seed in range(3))
    return {"second_singular_value": signal, "shuffle_ceiling": ceiling, "snr": signal / ceiling}


def fly_prior() -> sp.csr_matrix:
    """The fly TG x TF candidate prior (TF -> RE motif top 3%, RE -> TG within 100 kb) on the TFs of TF_filtered.tsv,
    as in the Fig. 2c,d reference line."""
    import anndata as ad

    from drosophila_gnn.preprocess import build_re_overlap_index, map_re_to_known

    raw = paths.data("fly", "raw")
    tf, tg = pd.read_csv(raw / "TF.tsv", sep="\t"), pd.read_csv(raw / "TG.tsv", sep="\t")
    to_id = dict(zip(pd.concat([tf.gene_name, tg.gene_name]), pd.concat([tf.FBgn_ID, tg.FBgn_ID])))
    peaks = pd.Index(ad.read_h5ad(raw / "B_atac_focus.h5ad", backed="r").var_names)
    genes = pd.Index(ad.read_h5ad(raw / "A_rna_focus.h5ad", backed="r").var_names)
    tf_re = pd.read_csv(raw / "TF_RE_top3pct_edges.tsv", sep="\t", dtype=str)
    re_tg = pd.read_csv(raw / "RE_TG_100kb.tsv", sep="\t", dtype={"chr_re": str})
    re_tg["RE"] = re_tg.chr_re + ":" + re_tg.re_start.astype(str) + "-" + re_tg.re_end.astype(str)
    index, by_chr = {name: i for i, name in enumerate(peaks)}, build_re_overlap_index(list(peaks))
    tf_re = tf_re.assign(tf=tf_re.TF.map(to_id), RE=tf_re.RE.map(lambda v: map_re_to_known(v, index, by_chr)))
    tf_re = tf_re.dropna(subset=["tf", "RE"])
    tf_re = tf_re[tf_re.tf.isin(genes) & tf_re.RE.isin(peaks)].drop_duplicates(["tf", "RE"])
    re_tg = re_tg.assign(tg=re_tg.gene_name.map(to_id)).dropna(subset=["tg"])
    re_tg = re_tg[re_tg.tg.isin(genes) & re_tg.RE.isin(peaks)].drop_duplicates(["tg", "RE"])
    pairs = tf_re[["tf", "RE"]].merge(re_tg[["tg", "RE"]], on="RE")[["tg", "tf"]].drop_duplicates()
    rows, cols = pd.Index(sorted(set(pairs.tg))), pd.Index(sorted(set(pairs.tf)))
    full = sp.csr_matrix((np.ones(len(pairs), np.float32), (rows.get_indexer(pairs.tg), cols.get_indexer(pairs.tf))),
                         shape=(len(rows), len(cols)))
    keep = cols.isin(pd.read_csv(raw / "TF_filtered.tsv", sep="\t", dtype=str).FBgn_ID)
    restricted = full[:, keep].tocsr()
    return restricted[np.asarray(restricted.sum(1)).ravel() > 0].tocsr()


# --------------------------------------------------------------------------------------------- t-SNE
def branch_tsne(data_seed: int = 101) -> pd.DataFrame:
    """RNA t-SNE of the branching dataset (Supp. Fig. 4a): all 1,600 cells embedded jointly; the state is used only
    for colours. 200 genes, library size 1e4 and log1p, 100 most variable genes, gene scaling, PCA to 50 components,
    t-SNE (perplexity 70, early exaggeration 8, learning rate 200, PCA initialisation, seed 16)."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    folder = root() / "datasets" / BRANCH / f"seed_{data_seed}"
    genes = [str(i) for i in range(1, 201)]
    blocks, obs = [], []
    for t, name in enumerate(STATES):
        counts = _matrix(folder / f"counts_time_{t}.csv")
        blocks.append(counts.loc[genes].to_numpy(float).T)
        obs.extend({"state": name, "time": t, "cell": cell} for cell in counts.columns)
    rna = np.concatenate(blocks)
    normalized = np.log1p(rna / np.maximum(rna.sum(1, keepdims=True), 1) * 1e4)
    variance = normalized.var(0, ddof=1)
    selected = np.argsort(-variance, kind="stable")[: min(100, int((variance > 0).sum()))]
    features = normalized[:, selected]
    features = (features - features.mean(0)) / features.std(0, ddof=1)
    pcs = PCA(n_components=min(50, *features.shape), svd_solver="full").fit_transform(features)
    xy = TSNE(n_components=2, perplexity=70.0, init="pca", learning_rate=200.0, early_exaggeration=8.0,
              random_state=16, method="barnes_hut", angle=0.5, n_jobs=1, max_iter=1000).fit_transform(pcs)
    return pd.DataFrame(obs).assign(tsne_1=xy[:, 0], tsne_2=xy[:, 1])


# --------------------------------------------------------------------------------------------- stage
def fits() -> pd.DataFrame:
    return pd.read_csv(root() / "fits.tsv", sep="\t", dtype={"data": int, "seed": int})


def run() -> None:
    """Score every fit of fits.tsv and the Random controls; write the tables of results/checkpoint/simulation."""
    table = fits()
    runs, per_time, matrices, refs = [], [], [], []
    for (study, case, dataset, data), group in table.groupby(["study", "case", "dataset", "data"], sort=False):
        d = Dataset(root() / "datasets" / dataset / f"seed_{data}")
        branch = study == BRANCH
        key = {"study": study, "case": case, "data": data}
        if not branch:
            refs.append({**key, "prior_snr": prior_snr(d.candidate)["snr"],
                         "truth_continuity": smoothness(d.truth[:, d.panel].astype(float))})
        else:
            refs.append({**key, "prior_snr": float("nan"),
                         "truth_continuity": smoothness(d.truth[:, d.candidate].astype(float), tree_pairs()),
                         **{f"edges_differing_{STATES[a]}_{STATES[b]}": int((d.candidate & (d.truth[a] != d.truth[b])).sum())
                            for a, b in BRANCH_PAIRS}})
        for r in group.itertuples():
            scores, ranking = load_fit(root() / r.file, d, r.sha256)
            times, row = (score_tree if branch else score_chain)(scores, ranking, d)
            runs.append({**key, "method": r.method, "seed": r.seed, **row})
            per_time.extend({**key, "method": r.method, "seed": r.seed, "time": t, "auroc": a}
                            for t, a in enumerate(times))
            if study == MAIN and case == "main":
                raw = distances(scores[:, d.candidate])
                matrices.extend({"method": r.method, "data": data, "seed": r.seed, "time_i": i, "time_j": j,
                                 "raw_distance": raw[i, j], "distance": raw[i, j] / raw.max()}
                                for i in range(len(raw)) for j in range(len(raw)))
        # references scored like a fit: the ground truth itself (seed -1) and the Random control
        seeds = RANDOM_SEEDS.get(study, ()) if case in ("main", "random_6true3false") else ()
        mask = d.candidate
        for method, seed in [("ground_truth", -1)] * bool(seeds) + [("random", s) for s in seeds]:
            values = (d.truth[:, mask].astype(float) if seed == -1
                      else np.random.default_rng(seed).random((len(d.truth), int(mask.sum()))))
            aucs = [float(roc_auc_score(y[mask], x)) for y, x in zip(d.truth, values)]
            runs.append({**key, "method": method, "seed": seed, "candidate_auroc": float(np.mean(aucs)),
                         "mean_continuity": smoothness(values)})
            per_time.extend({**key, "method": method, "seed": seed, "time": t, "auroc": a} for t, a in enumerate(aucs))
            if study == MAIN:
                raw = distances(values)
                matrices.extend({"method": method, "data": data, "seed": seed, "time_i": i, "time_j": j,
                                 "raw_distance": raw[i, j], "distance": raw[i, j] / raw.max()}
                                for i in range(len(raw)) for j in range(len(raw)))
        print(f"   {study} {case} data {data}: {len(group)} fits", flush=True)
    for name, rows in (("runs", runs), ("per_time", per_time), ("distances", matrices), ("references", refs)):
        pd.DataFrame(rows).to_csv(out(f"{name}.tsv"), sep="\t", index=False, lineterminator="\n")
    fly = fly_prior()
    pd.DataFrame([{"prior": "fly", "TGs": fly.shape[0], "TFs": fly.shape[1], "edges": fly.nnz, **prior_snr(fly)}]).to_csv(
        out("real_prior_snr.tsv"), sep="\t", index=False, lineterminator="\n")
    branch_tsne(101).to_csv(out("tsne_branch_101.tsv"), sep="\t", index=False, lineterminator="\n")
