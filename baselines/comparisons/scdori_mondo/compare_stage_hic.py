"""DRAFT, scDoRI and published SCENIC+ Hi-C F1 on identical stage-wise pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from hic_common import map_positives, metrics, negative_pairs, region_key
from scenic_hic import fetch_reference, load_reference, map_stage_edges
from sklearn.metrics import precision_recall_curve

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def grouped_calibration_split(pairs, labels, n_genes, seed=2025):
    """Split unique physical pairs; replacement duplicates never cross folds."""
    codes = pairs[:, 0] * n_genes + pairs[:, 1]
    calibration = np.zeros(len(pairs), dtype=bool)
    rng = np.random.default_rng(seed)
    for label in [0, 1]:
        groups = np.unique(codes[np.asarray(labels) == label])
        if len(groups) < 2:
            raise ValueError(
                "Need at least two unique pairs in each class for calibration/test splitting."
            )
        chosen = rng.choice(groups, size=len(groups) // 2, replace=False)
        calibration |= np.isin(codes, chosen)
    return calibration


def calibration_threshold(labels, scores):
    """Maximize calibration F1; no test labels and no zero-support predictions."""
    scores = np.asarray(scores, dtype=np.float64)
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = np.divide(
        2 * precision[:-1] * recall[:-1],
        precision[:-1] + recall[:-1],
        out=np.zeros_like(thresholds),
        where=(precision[:-1] + recall[:-1]) > 0,
    )
    eligible = np.flatnonzero(thresholds > 0)
    if not len(eligible):
        return max(1.0, float(scores.max()) + 1.0)
    if f1[eligible].max() == 0:
        return max(1.0, float(scores.max()) + 1.0)
    best = eligible[np.isclose(f1[eligible], f1[eligible].max(), rtol=0, atol=1e-12)]
    return float(thresholds[best[-1]])


def raw_draft_scores(out_re, out_tg, peak_indices, gene_indices):
    from drosophila_gnn.analysis.repro import score_pairs_over_time_sigmoid

    pairs = torch.as_tensor(
        np.column_stack([peak_indices, gene_indices]),
        dtype=torch.long,
        device=out_re.device,
    )
    return np.stack(score_pairs_over_time_sigmoid(out_re, out_tg, pairs))


def choose_draft_checkpoint(value: Path, root: Path):
    if value != Path("auto"):
        path = value.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    data_project = root.parent.parent
    candidates = [
        PROJECT / "docs/germ.pt",
        PROJECT.parent / "germ.pt",
        data_project / "germ.pt",
        data_project / "docs/germ.pt",
    ]
    for parent in [PROJECT, data_project]:
        candidates.extend((parent / "result/germ/model").glob("*.pt"))
    expected = read_json(
        PROJECT / "docs/reference/mondo_seed2025/reference_manifest.json"
    )["source_checkpoint_sha256"]
    for path in dict.fromkeys(candidates):
        if path.is_file() and digest(path) == expected:
            return path.resolve()
    raise FileNotFoundError(
        "Cannot locate the verified manuscript DRAFT checkpoint. Pass --draft-checkpoint /absolute/path/to/germ.pt."
    )


def load_draft(root, checkpoint):
    sys.path.insert(0, str(PROJECT / "src"))
    from drosophila_gnn.infer import reload_germ_from_checkpoint
    from drosophila_gnn.utils import load_yaml_config

    config = load_yaml_config(PROJECT / "configs/germ_reproduce.yaml")
    config["base_dir"] = str(PROJECT)
    for key, name in {
        "tf_list": "TF.tsv",
        "tg_list": "TG.tsv",
        "re_list": "RE.txt",
        "rna_h5ad": "A_rna_focus.h5ad",
        "atac_h5ad": "B_atac_focus.h5ad",
        "tf_re_prior": "TF_RE_top3pct_edges.tsv",
        "re_tg_prior": "RE_TG_100kb.tsv",
        "annotation": "cell_annotation.csv",
    }.items():
        config["data"][key] = str(root / name)
    config["reproduce"]["checkpoint_path"] = str(checkpoint)
    result = reload_germ_from_checkpoint(config)
    hp = result["hp"]
    if not hp.get("use_gru", True) or hp.get("spatial_encoder", "gat") != "gat":
        raise ValueError("Default comparison requires full DRAFT (GAT + GRU).")
    return result


def compare(
    networks: Path,
    root: Path | None,
    draft_checkpoint=Path("auto"),
    draft_loader=None,
    scenic_reference=Path("auto"),
):
    networks = networks.resolve()
    if not networks.is_relative_to(HERE / "runs"):
        raise ValueError("Stage networks must be inside the isolated runs directory.")
    meta = read_json(networks / "stage_report.json")
    for name, expected in meta["files_sha256"].items():
        if digest(networks / name) != expected:
            raise ValueError(f"Stage export changed: {name}")
    root = (
        (root or Path(meta["input_sources"]["TG.tsv"]["path"]).parent)
        .expanduser()
        .resolve()
    )
    for name, source in meta["input_sources"].items():
        if digest(root / name) != source["sha256"]:
            raise ValueError(f"DRAFT/scDoRI input sources differ: {name}")
    scenic_reference = Path(scenic_reference)
    if scenic_reference == Path("auto"):
        scenic_reference = fetch_reference()
    scenic_table, scenic_meta = load_reference(scenic_reference)
    if scenic_meta.get("validation_only", False) and not meta["validation_only"]:
        raise ValueError(
            "Synthetic SCENIC+ reference cannot enter a biological comparison."
        )
    checkpoint = choose_draft_checkpoint(draft_checkpoint, root)
    # Load both models' unlabeled scores/feature pools before reading Hi-C labels.
    draft = (draft_loader or load_draft)(root, checkpoint)
    if draft["checkpoint"].get("seed") != meta["seed"]:
        raise ValueError("DRAFT and scDoRI seeds differ.")
    types = draft["germline_types"]
    if types != [f"Germline_{i}" for i in range(15)] or draft["out_re"].shape[1] != 15:
        raise ValueError("DRAFT stage order must exactly match Germline_0–14.")
    anno = draft["artifacts"]["common"]["anno"]
    observed = anno.loc[anno.cell_type.isin(types), ["cell_id", "cell_type"]].copy()
    observed["time_idx"] = observed.cell_type.str.extract(r"(\d+)$")[0].astype(int)
    cells = pd.read_csv(networks / "cells.tsv.gz", sep="\t")
    if set(zip(cells.cell_id, cells.time_idx)) != set(
        zip(observed.cell_id, observed.time_idx)
    ):
        raise ValueError(
            "Paired germline cell cohorts/stages differ between the methods."
        )
    tg = pd.read_csv(root / "TG.tsv", sep="\t", dtype=str).fillna("")
    aliases = dict(zip(tg.gene_name, tg.FBgn_ID))
    aliases.update({g: g for g in tg.FBgn_ID})
    dp = {region_key(str(p)): i for i, p in enumerate(draft["maps"]["re"])}
    dg = {aliases.get(str(g), str(g)): i for i, g in enumerate(draft["maps"]["tg"])}
    if len(dp) != len(draft["maps"]["re"]) or len(dg) != len(draft["maps"]["tg"]):
        raise ValueError("Ambiguous DRAFT node identifiers.")
    arrays = np.load(networks / "stage_arrays.npz")
    stage_genes, stage_peaks = arrays["genes"].tolist(), arrays["peaks"].tolist()
    genes = [g for g in stage_genes if g in dg]
    peaks = [p for p in stage_peaks if region_key(p) in dp]
    if not genes or not peaks:
        raise ValueError("No shared RE/TG feature pool.")
    # Do not restrict validation to endpoints of SCENIC+'s predicted edges.
    scenic_edges, scenic_audits, scenic_region_mapping = map_stage_edges(
        scenic_table, peaks, genes, tg, return_region_mapping=True
    )
    gi = {g: i for i, g in enumerate(genes)}
    pi = {region_key(p): i for i, p in enumerate(peaks)}
    pair_gene, pair_peak = arrays["pair_gene"], arrays["pair_peak"]
    sparse_lookup = {
        (pi[region_key(stage_peaks[p])], gi[stage_genes[g]]): i
        for i, (g, p) in enumerate(zip(pair_gene, pair_peak))
        if stage_genes[g] in gi and region_key(stage_peaks[p]) in pi
    }
    allowed = np.sort(
        np.asarray([p * len(genes) + g for p, g in sparse_lookup], dtype=np.int64)
    )
    rows, audits, pair_tables, scored_tables, truth_sources = [], [], {}, {}, {}
    for assay, times in [("spg", range(2, 6)), ("spc", range(6, 15))]:
        path = root / f"{assay}_hic.tsv"
        truth_sources[assay] = {"path": str(path), "sha256": digest(path)}
        if truth_sources[assay]["sha256"] in {
            v["sha256"] for v in meta["input_sources"].values()
        }:
            raise ValueError("Hi-C labels duplicate a preparation input.")
        positives, mapped = map_positives(path, peaks, genes, tg)
        reachable = np.asarray([tuple(pair) in sparse_lookup for pair in positives])
        for stage in times:
            scenic_audits[stage].update(
                dataset=assay,
                n_positive_reported=sum(
                    tuple(pair) in scenic_edges[stage] for pair in positives
                ),
                n_positive_unreported=sum(
                    tuple(pair) not in scenic_edges[stage] for pair in positives
                ),
            )
        audits.append(
            {
                "dataset": assay,
                **mapped,
                "n_common_genes": len(genes),
                "n_common_peaks": len(peaks),
                "n_positive_inside_prior": int(reachable.sum()),
                "n_positive_outside_prior": int((~reachable).sum()),
            }
        )
        for universe in ["all_features", "within_prior"]:
            selected = positives if universe == "all_features" else positives[reachable]
            if len(selected) < 2:
                continue
            negatives = negative_pairs(
                selected,
                len(peaks),
                len(genes),
                42,
                allowed if universe == "within_prior" else None,
            )
            pairs = np.vstack([selected, negatives])
            labels = np.r_[
                np.ones(len(selected), dtype=int), np.zeros(len(negatives), dtype=int)
            ]
            calibration = grouped_calibration_split(pairs, labels, len(genes))
            p, g = pairs.T
            native = np.zeros((15, len(pairs)), dtype=np.float64)
            positions = np.asarray(
                [sparse_lookup.get(tuple(pair), -1) for pair in pairs]
            )
            present = positions >= 0
            native[:, present] = arrays["re_tg_scores"][:, positions[present]]
            baseline = raw_draft_scores(
                draft["out_re"],
                draft["out_tg"],
                np.asarray([dp[region_key(peaks[i])] for i in p]),
                np.asarray([dg[genes[i]] for i in g]),
            )
            scenic = np.asarray(
                [
                    [lookup.get(tuple(pair), 0.0) for pair in pairs]
                    for lookup in scenic_edges
                ],
                dtype=np.float64,
            )
            if not all(
                np.isfinite(values).all() for values in [native, baseline, scenic]
            ):
                raise ValueError("Nonfinite stage scores.")
            pair_tables[f"pairs_{assay}_{universe}.tsv.gz"] = pd.DataFrame(
                {
                    "pair_idx": np.arange(len(pairs)),
                    "RE": np.asarray(peaks)[p],
                    "TG_id": np.asarray(genes)[g],
                    "label": labels,
                    "fold": np.where(calibration, "calibration", "test"),
                    "in_prior": present,
                }
            )
            for stage in times:
                common = {
                    "dataset": assay,
                    "time_idx": stage,
                    "stage_label": meta["stage_labels"][stage],
                    "universe": universe,
                }
                scored_tables[f"scores_{assay}_{universe}_t{stage:02d}.tsv.gz"] = (
                    pd.DataFrame(
                        {
                            "pair_idx": np.arange(len(pairs)),
                            "scdori_score": native[stage],
                            "draft_score": baseline[stage],
                            "scenicplus_score": scenic[stage],
                            "scenicplus_reported": [
                                tuple(pair) in scenic_edges[stage] for pair in pairs
                            ],
                        }
                    )
                )
                for label, values in [
                    ("scDoRI", native[stage]),
                    ("DRAFT", baseline[stage]),
                    ("SCENIC+", scenic[stage]),
                ]:
                    threshold = calibration_threshold(
                        labels[calibration], values[calibration]
                    )
                    for protocol, mask, cutoff in [
                        ("fixed_0.5_all_pairs", np.ones(len(pairs), dtype=bool), 0.5),
                        ("calibrated_disjoint_test", ~calibration, threshold),
                    ]:
                        rows.append(
                            {
                                **common,
                                "model": label,
                                "protocol": protocol,
                                "threshold": cutoff,
                                "n_positive": int(labels[mask].sum()),
                                "n_negative": int((labels[mask] == 0).sum()),
                                **metrics(
                                    labels[mask], values[mask], cutoff, inclusive=True
                                ),
                            }
                        )
    if not rows:
        raise ValueError("Too few shared Hi-C pairs for a paired stage comparison.")
    out = networks / f"comparison_hic_{uuid.uuid4().hex[:10]}"
    out.mkdir(exist_ok=False)
    (out / "evaluation_features.json").write_text(
        json.dumps({"genes": genes, "peaks": peaks}), encoding="utf-8"
    )
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "hic_f1_by_stage.tsv", sep="\t", index=False)
    comparison = (
        summary.pivot(
            index=["dataset", "time_idx", "stage_label", "universe", "protocol"],
            columns="model",
            values="f1",
        )
        .reindex(columns=["DRAFT", "scDoRI", "SCENIC+"])
        .reset_index()
    )
    comparison.to_csv(out / "hic_f1_comparison.tsv", sep="\t", index=False)
    pd.DataFrame(audits).to_csv(out / "hic_coverage.tsv", sep="\t", index=False)
    pd.DataFrame(scenic_audits).to_csv(
        out / "scenic_coverage.tsv", sep="\t", index=False
    )
    scenic_region_mapping.to_csv(
        out / "scenic_region_mapping.tsv", sep="\t", index=False
    )
    for name, table in {**pair_tables, **scored_tables}.items():
        table.to_csv(out / name, sep="\t", index=False)
    report = {
        "stage_networks": str(networks),
        "stage_report_sha256": digest(networks / "stage_report.json"),
        "draft_checkpoint": str(checkpoint),
        "draft_checkpoint_sha256": digest(checkpoint),
        "sources": truth_sources,
        "n_common_genes": len(genes),
        "n_common_peaks": len(peaks),
        "evaluation_features_sha256": digest(out / "evaluation_features.json"),
        "models": ["DRAFT", "scDoRI", "SCENIC+"],
        "comparison_code_sha256": digest(Path(__file__)),
        "scenic_adapter_code_sha256": digest(HERE / "scenic_hic.py"),
        "scenic_reference": scenic_meta,
        "scenic_reference_manifest_sha256": digest(
            scenic_reference / "reference_manifest.json"
        ),
        "scenic_region_mapping": "exact coordinates preferred, otherwise the unique maximum genomic overlap covering >=50% of the source SCENIC+ interval; half-open coordinates; ties and insufficient overlaps omitted and audited; mapping fixed before reading Hi-C labels; many source regions mapping to one RE–TG use maximum R2G_importance",
        "validation_only": bool(meta["validation_only"]),
        "sampling_seed": 42,
        "calibration_seed": 2025,
        "fixed_protocol": "same source Hi-C labels, identical 1:1 sampled pairs, original stages and >=0.5 rule for all three methods; scDoRI uses native stage-conditioned RNA-decoder coefficients, DRAFT uses original sigmoid RE/TG embedding dot product, SCENIC+ uses published R2G_importance (maximum across duplicate TF paths)",
        "calibrated_protocol": "same physical-pair-disjoint 50/50 stratified calibration/test split for all three methods; per-stage/per-method positive threshold maximizes calibration F1 only; highest threshold breaks ties; zero-support links cannot become positive",
        "candidate_protocol": "all_features primary: DRAFT/scDoRI node intersection retained when adding SCENIC+, whose unreported edges score zero; never intersect predicted-edge endpoints; within_prior conditional on the same scDoRI support for all three methods; no additional Top-K or smoothing before F1",
        "cautions": "score units remain method-specific; per-stage scDoRI is a native-topic mixture after training, not a recurrent model; SCENIC+ is the original study's DAR=yes/DEG=yes filtered network, not a retrained model with verified identical cells, priors or seed; unobserved sampled links are proxy negatives; this adapted comparison is not full native end-to-end benchmarking",
    }
    (out / "comparison_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    plot_f1(summary, out, meta)
    print(comparison.to_string(index=False), flush=True)
    print(f"Three-method comparison: {out}", flush=True)
    return out


def plot_f1(table, out, meta):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context({"pdf.fonttype": 42, "svg.fonttype": "none"}):
        for protocol in table.protocol.unique():
            fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
            for ax, universe in zip(axes, ["all_features", "within_prior"]):
                for name, color in [
                    ("DRAFT", "#E5A000"),
                    ("scDoRI", "#333333"),
                    ("SCENIC+", "#4477AA"),
                ]:
                    data = table.loc[
                        table.protocol.eq(protocol)
                        & table.universe.eq(universe)
                        & table.model.eq(name)
                    ].sort_values("time_idx")
                    labeled = False
                    for assay in ["spg", "spc"]:
                        segment = data.loc[data.dataset.eq(assay)]
                        if segment.empty:
                            continue
                        ax.plot(
                            segment.time_idx,
                            segment.f1,
                            color=color,
                            marker="^" if name == "SCENIC+" else "o",
                            markerfacecolor="white" if name == "SCENIC+" else color,
                            linestyle="--" if name == "SCENIC+" else "-",
                            label=(
                                meta.get("scenic_label", name)
                                if name == "SCENIC+"
                                else name
                            )
                            if not labeled
                            else None,
                        )
                        labeled = True
                ax.set_xticks(
                    range(15), meta["stage_labels"], rotation=60, ha="right", fontsize=9
                )
                ax.set(
                    xlim=(-0.4, 14.4),
                    ylim=(-0.025, 1.025),
                    ylabel="Hi-C RE–TG F1",
                    title="Common RE–TG pool"
                    if universe == "all_features"
                    else "Pairs within shared prior",
                )
                ax.legend(frameon=False)
            fig.suptitle(
                "SYNTHETIC SOFTWARE TEST"
                if meta["validation_only"]
                else "Germline Hi-C RE–TG prediction (calibrated test F1)"
                if protocol == "calibrated_disjoint_test"
                else "Germline Hi-C RE–TG prediction (F1 at threshold 0.5)"
            )
            for suffix in ["png", "pdf", "svg"]:
                fig.savefig(out / f"hic_f1_{protocol}.{suffix}", dpi=300)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--networks", type=Path, default=Path("latest"))
    parser.add_argument(
        "--run",
        type=Path,
        help="Run whose newest completed stage export is selected by --networks latest.",
    )
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--draft-checkpoint", type=Path, default=Path("auto"))
    parser.add_argument(
        "--scenic-reference",
        type=Path,
        default=Path("auto"),
        help="Completed dAWA extraction, or auto to download/resume and reuse the cached all-TF stage networks.",
    )
    args = parser.parse_args()
    if args.networks == Path("latest"):
        if args.run is None:
            parser.error(
                "Pass --run for latest stage networks, or give --networks an exact export path."
            )
        run = args.run.expanduser().resolve()
        if not run.is_relative_to(HERE / "runs"):
            parser.error("Run must be inside comparisons/scdori_mondo/runs.")
        reports = list(run.glob("stage_networks_*/stage_report.json"))
        if not reports:
            parser.error(
                "No completed stage export found. Run workflow.py stage-networks first."
            )
        args.networks = max(reports, key=lambda p: p.stat().st_mtime_ns).parent
        print(f"Selected stage networks: {args.networks}", flush=True)
    compare(
        args.networks,
        args.data_root,
        args.draft_checkpoint,
        scenic_reference=args.scenic_reference,
    )


if __name__ == "__main__":
    main()
