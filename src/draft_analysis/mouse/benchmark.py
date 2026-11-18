"""Fig. 3b: official AUPR and balanced AUROC of the five methods against three mESC gold standards, per cell cluster.

For each cluster and method the predicted TF-TG scores are written in the official list format:
DRAFT = the checkpoint's full saved score matrix on the gold-covered nodes; scMTNI+Prior = the published consensus
confidences; Dictys and SCENIC = their native edge scores (deposited); Random = the fixed-budget uniform
edge set (seed 12345 per cluster). The official scMTNI preparation script and auc.jar (Java) give the AUPR at natural
prevalence. The balanced AUROC samples all positives
and as many negatives on the same covered universe (seed 2025 per cluster), from the official filtered snapshots
(evaluate_official_balanced_auc.py). Nothing is trained.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import re
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pandas as pd

from .. import paths
from . import CLUSTERS, native, out, scripts_on_path

METHODS = ["DRAFT", "Dictys", "scMTNI+Prior", "SCENIC", "Random"]


def java() -> str:
    exe = os.environ.get("DRAFT_JAVA") or shutil.which("java")
    if not exe:
        raise FileNotFoundError("Java is needed for the official auc.jar; put java on PATH or set DRAFT_JAVA")
    return exe


def predictions(method: str, cid: int, tfs, tgs, regulators, targets, checkpoint, random_inputs) -> pd.DataFrame:
    from evaluate import random_network
    from official_aupr_protocol import canonical_edges
    from predictions import checkpoint_prediction, consensus_prediction

    if method == "DRAFT":
        return checkpoint_prediction(checkpoint["sims_best"][CLUSTERS.index(cid)], tfs, tgs, regulators, targets)
    if method == "scMTNI+Prior":
        return consensus_prediction(paths.data("mouse", "raw", "scMTNI_plus_prior", f"cluster{cid}_consensus_edges.txt"),
                                    tfs, tgs, regulators, targets)
    if method == "Random":
        spec = random_inputs["clusters"][f"C{cid}"]
        _, chosen, _ = random_network(np.arange(spec["n_edges"]), tfs, tgs, spec["genes"], random_inputs["seed"], cid)
        return pd.DataFrame({"TF": np.asarray(tfs)[chosen // len(tgs)], "TG": np.asarray(tgs)[chosen % len(tgs)],
                             "score": np.ones(len(chosen))})
    folder = {"Dictys": "dictys", "SCENIC": "scenic"}[method]
    frame, _ = canonical_edges(pd.read_csv(native(folder) / f"C{cid}" / "edges.tsv.gz", sep="\t"), tfs, tgs)
    return frame


def aupr() -> pd.DataFrame:
    """Official AUPR for every gold, cluster and method; keeps the filtered snapshots for the AUROC."""
    scripts_on_path()
    from official_aupr_protocol import GOLDS, evaluation_scope, load_gold_tables, load_nodes, load_official_preparer
    from predictions import load_checkpoint

    res = paths.data("mouse", "official_aupr")
    features = json.loads((native("features") / "features.json").read_text(encoding="utf-8"))
    tfs, tgs = features["TF"], features["TG"]
    random_inputs = json.loads(paths.data("mouse", "random_distance_inputs.json").read_text(encoding="utf-8"))
    if random_inputs["TF"] != tfs or random_inputs["TG"] != tgs:
        raise ValueError("Random inputs and the prepared feature axes differ")
    checkpoint = load_checkpoint(paths.data("mouse", "checkpoint", "grn_best.pt"), tfs, tgs)
    nodes, tables = load_nodes(res / "nodes"), load_gold_tables(res / "gold")
    gold_paths = {gold: res / "gold" / name for gold, (name, _) in GOLDS.items()}
    signature, scope = evaluation_scope(nodes, tables)
    root = out("benchmark", "aupr")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    official = load_official_preparer(res / "source", root / "official_preparer_compat.py")
    jar, exe = str((res / "source" / "auc.jar").resolve()), java()
    records = []
    for cid in CLUSTERS:
        croot = root / f"C{cid}"
        croot.mkdir()
        reg_path, target_path = croot / "regulators.txt", croot / "targets.txt"
        reg_path.write_text("\n".join(nodes[cid]["regulators"]) + "\n", encoding="utf-8")
        target_path.write_text("\n".join(nodes[cid]["targets"]) + "\n", encoding="utf-8")
        covered = {g: t.loc[t.TF.isin(nodes[cid]["regulators"]) & t.TG.isin(nodes[cid]["targets"]) & t.TF.ne(t.TG)]
                   for g, t in tables.items()}
        regulators = set().union(*(set(t.TF) for t in covered.values()))
        targets = set().union(*(set(t.TG) for t in covered.values()))
        for method in METHODS:
            frame = predictions(method, cid, tfs, tgs, regulators, targets, checkpoint, random_inputs)
            prediction = croot / f"{method}_predictions.tsv"
            frame[["TF", "TG", "score"]].to_csv(prediction, sep="\t", header=False, index=False, float_format="%.17g")
            for gold, gold_path in gold_paths.items():
                o = croot / method / gold.replace("+", "_")
                o.mkdir(parents=True)
                c = covered[gold]
                pred = prediction
                if not (frame.TF.isin(set(c.TF)) & frame.TG.isin(set(c.TG))).any():
                    pred = o / "empty_output_zero_score_adapter.tsv"  # all scores stay zero, as in the original
                    pred.write_text(f"{c.iloc[0].TF}\t{c.iloc[0].TG}\t0\n")
                log = io.StringIO()
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    official(SimpleNamespace(truefile=str(gold_path.resolve()), predfile=str(pred.resolve()),
                                             outdir=str(o.resolve()) + "/", prefix="evaluation",
                                             regulators=str(reg_path.resolve()), targets=str(target_path.resolve())))
                result = subprocess.run([exe, "-Xmx2g", "-jar", jar, str((o / "evaluation_list.txt").resolve()), "list"],
                                        capture_output=True, text=True, check=True)
                (o / "auc_jar_log.txt").write_text(result.stdout + result.stderr, encoding="utf-8")
                m = re.search(r"Area Under the Curve for Precision - Recall is ([\deE.+-]+)", result.stdout)
                truth = pd.read_csv(o / "evaluation_list.txt", sep="\t", header=None, usecols=[1]).iloc[:, 0]
                exp = scope.loc[(scope.gold_standard == gold) & (scope.cluster == f"C{cid}")].iloc[0]
                if m is None or len(truth) != exp.n_pairs or int(truth.sum()) != exp.n_positive:
                    raise ValueError(f"official evaluation failed or used another universe: C{cid} {gold} {method}")
                records.append({"gold_standard": gold, "cluster": f"C{cid}", "model": method, "AUPR": float(m.group(1)),
                                "n_positive": int(truth.sum()), "n_pairs": len(truth)})
            print(f"   C{cid} {method}: AUPR done", flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(root / "metrics_by_cluster.tsv", sep="\t", index=False, float_format="%.17g")
    scope.to_csv(root / "evaluation_scope.tsv", sep="\t", index=False)
    (root / "protocol_sha256.txt").write_text(signature + "\n", encoding="utf-8")
    return frame


def auroc() -> pd.DataFrame:
    """Balanced 1:1 AUROC on the test half, from the official filtered snapshots of every method."""
    scripts_on_path()
    from evaluate_official_balanced_auc import make_sample, project_scores
    from official_aupr_protocol import load_gold_tables
    from sklearn.metrics import roc_auc_score

    root = out("benchmark", "aupr")
    golds = list(load_gold_tables(paths.data("mouse", "official_aupr", "gold")))
    rows = []
    for cid in CLUSTERS:
        for gold in golds:
            sample = None
            for method in METHODS:
                folder = root / f"C{cid}" / method / gold.replace("+", "_")
                if sample is None:
                    truth = pd.read_csv(folder / "gold_filtrered.txt", sep="\t", header=None, names=["TF", "TG"],
                                        dtype=str)
                    sample = make_sample(truth, cid)
                pred = pd.read_csv(folder / "predicted_filtrered.txt", sep="\t", header=None, names=["TF", "TG", "score"])
                scores = project_scores(pred, sample)
                rows.append({"gold_standard": gold, "cluster": f"C{cid}", "model": method,
                             "AUROC": float(roc_auc_score(sample["test_labels"], scores[sample["test_codes"]])),
                             "n_positive": int(sample["test_labels"].sum()),
                             "n_negative": int((sample["test_labels"] == 0).sum())})
    frame = pd.DataFrame(rows)
    frame.to_csv(out("benchmark", "auroc_metrics_by_cluster.tsv"), sep="\t", index=False, float_format="%.17g")
    return frame


def run() -> None:
    aupr()
    auroc()
