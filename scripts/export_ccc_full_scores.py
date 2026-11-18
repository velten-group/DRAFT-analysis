"""Export unthresholded TF-TG scores at the integer stages for the LR timing rule (configs/fly/ccc_unified.yaml).

For every integer stage of the germline (15) or soma (9) model, every pair of an active, whitelisted TF and an
active TG gets the cosine of their L2-normalised embeddings and its percentile among all such pairs at that stage.
No Gaussian resampling, no cosine floor, no edge budget. The published networks are only read, to check that the
integer-stage cosines are the ones they were cut from.

    python scripts/export_ccc_full_scores.py --lineage germ --config <germ network config> --checkpoint germ_seed2025.pt
    python scripts/export_ccc_full_scores.py --lineage soma --config <soma network config> --checkpoint soma.pt --artifact soma_graphs.pt

Outputs in --output-dir (default result/ccc_full_scores/<lineage>):
  receptor_input.tsv        per TG and stage, readout A: mean percentile over the active TFs (0 where the TG is inactive)
  receptor_input_top10pct_tfs.tsv       readout B: mean of the top 10% percentiles over the active TFs
  receptor_input_top5pct_pairs_frac.tsv readout C: share of active TFs whose pair is in the stage's top 5% of pairs
  receptor_input_ungated.tsv            readout A without the receptor activity gate (inactive TGs scored as well)
  tg_active.tsv             per TG and stage: 1 if expressed in >= active_thr of the stage's cells
  effector_percentiles.tsv  cosine and percentile from each effector TF to every active TG, per stage
  stage_summary.tsv         active counts, cosine quantiles and the check against the published network
  manifest.json             checkpoint hash, settings and effector-TF status
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

from drosophila_gnn.analysis.network import (
    _dense_active_masks,
    _dense_network_settings,
    _dense_tf_whitelist,
    _l2_normalize_rows,
    _load_tf_filter,
    _load_tg_meta,
    _network_settings,
    _pick_reference_names,
    _resolve_relative_path,
)
from drosophila_gnn.checkpoint import save_checkpoint_metadata
from drosophila_gnn.infer import (
    reload_germ_from_checkpoint,
    reload_soma_from_checkpoint,
)
from drosophila_gnn.io import InputPaths, strip_copy_suffix
from drosophila_gnn.utils import ensure_directory, load_yaml_config, resolve_config_path

EFFECTOR_TFS = ["pan", "Mad", "Med", "ci", "Su(H)", "foxo", "cic"]
TOP_TF_SHARE = 0.10
TOP_PAIR_SHARE = 0.05
READOUT_FILES = {
    "mean": "receptor_input.tsv",
    "top10pct_tfs": "receptor_input_top10pct_tfs.tsv",
    "top5pct_pairs_frac": "receptor_input_top5pct_pairs_frac.tsv",
    "mean_ungated": "receptor_input_ungated.tsv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--lineage", choices=["germ", "soma"], required=True)
    parser.add_argument("--config", help="network config (default configs/<lineage>_network.yaml)")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--artifact", help="soma graph artifact (soma_graphs.pt); soma only")
    parser.add_argument("--output-dir")
    parser.add_argument("--published-dir", help="published dense network to check against (default network.dense.output_dir)")
    parser.add_argument("--effector-tfs", nargs="*", default=EFFECTOR_TFS)
    args = parser.parse_args()

    config = load_yaml_config(args.config or f"configs/{args.lineage}_network.yaml")
    checkpoint = Path(args.checkpoint).resolve()
    config.setdefault("reproduce", {})["checkpoint_path"] = str(checkpoint)
    if args.lineage == "soma":
        if not args.artifact:
            parser.error("--artifact is required for the soma model")
        config["reproduce"]["artifact_path"] = str(Path(args.artifact).resolve())
        inference = reload_soma_from_checkpoint(config)
        graphs = inference["graphs_somatic"]
    else:
        inference = reload_germ_from_checkpoint(config)
        graphs = inference["graphs_germline"]

    paths = InputPaths.from_config(config)
    tf_filter = _load_tf_filter(_resolve_relative_path(paths.base_dir, _network_settings(config)["tf_filter_file"]))
    refs = _pick_reference_names(
        maps=inference.get("maps"),
        paths=paths,
        num_tf=int(inference["out_tf"].size(0)),
        num_tg=int(inference["out_tg"].size(0)),
        num_re=int(inference["out_re"].size(0)),
        tf_filter=tf_filter,
        tg_meta_bundle=_load_tg_meta(paths.tg_list),
    )
    dense = _dense_network_settings(config)
    active_thr = float(dense["active_thr"])
    published = Path(args.published_dir).resolve() if args.published_dir else resolve_config_path(config, dense["output_dir"])
    output = ensure_directory(Path(args.output_dir) if args.output_dir else Path(f"result/ccc_full_scores/{args.lineage}"))

    whitelist = _dense_tf_whitelist(refs["tf_ids"], refs["tf_names"], tf_filter).numpy()
    tf_active, tg_active, _ = _dense_active_masks(graphs, active_thr)
    tf_active, tg_active = tf_active.numpy(), tg_active.numpy()
    out_tf = inference["out_tf"].detach().cpu().float()
    out_tg = inference["out_tg"].detach().cpu().float()
    n_stage = int(out_tf.size(1))
    tf_key = [strip_copy_suffix(str(name)).lower() for name in refs["tf_names"]]
    effectors = {name: [i for i, key in enumerate(tf_key) if key == name.lower()] for name in args.effector_tfs}

    # Receptor readouts of the rule: A mean percentile, B mean of the top 10% of TFs, C share of TFs in the stage's top 5% of pairs.
    readouts = {name: np.zeros((len(refs["tg_names"]), n_stage)) for name in READOUT_FILES}
    effector_rows: list[pd.DataFrame] = []
    effector_status: dict[str, dict[str, str]] = {name: {} for name in effectors}
    summary: list[dict] = []
    for t in range(n_stage):
        tf_idx = np.flatnonzero(tf_active[t] & whitelist)
        tg_idx = np.flatnonzero(tg_active[t])
        tf_vec = _l2_normalize_rows(out_tf[tf_idx, t, :])
        tg_vec = _l2_normalize_rows(out_tg[tg_idx, t, :])
        cos = (tf_vec @ tg_vec.T).numpy().astype(np.float64)
        pct = rankdata(cos, method="average").reshape(cos.shape) / cos.size
        k = int(np.ceil(TOP_TF_SHARE * tf_idx.size))
        readouts["mean"][tg_idx, t] = pct.mean(axis=0)
        readouts["top10pct_tfs"][tg_idx, t] = np.sort(pct, axis=0)[-k:].mean(axis=0)
        readouts["top5pct_pairs_frac"][tg_idx, t] = (pct > 1.0 - TOP_PAIR_SHARE).mean(axis=0)
        # Readout A without the receptor activity gate: every TG, inactive ones included, is ranked against the same
        # reference distribution (all active TF x active TG pairs of the stage).
        reference = np.sort(cos, axis=None)
        all_vec = _l2_normalize_rows(out_tg[:, t, :])
        cos_all = (tf_vec @ all_vec.T).numpy().astype(np.float64)
        readouts["mean_ungated"][:, t] = (np.searchsorted(reference, cos_all, side="right") / reference.size).mean(axis=0)

        label = t + 1
        row = {"stage": label, "n_tf_active": int(tf_idx.size), "n_tg_active": int(tg_idx.size), "n_pairs": int(cos.size)}
        row.update({f"cos_q{int(q * 100):02d}": float(np.quantile(cos, q)) for q in (0.0, 0.05, 0.5, 0.95, 1.0)})
        top_k = int(dense["top_k_global"])
        order = np.argsort(-cos, axis=None, kind="stable")[:top_k]
        row["cos_at_top_k"] = float(cos.ravel()[order[-1]])
        published_file = published / f"TF_TG_dense_t{label}.tsv"
        if published_file.exists():
            pub = pd.read_csv(published_file, sep="\t", usecols=["TF_idx", "TG_idx", "score_cos"])
            rows_i = pub["TF_idx"].map(pd.Series(np.arange(tf_idx.size), index=tf_idx))
            cols_j = pub["TG_idx"].map(pd.Series(np.arange(tg_idx.size), index=tg_idx))
            found = (rows_i.notna() & cols_j.notna()).to_numpy()
            mine = np.full(len(pub), np.nan)
            mine[found] = cos[rows_i[found].astype(int).to_numpy(), cols_j[found].astype(int).to_numpy()]
            top = set(zip(tf_idx[order // cos.shape[1]].tolist(), tg_idx[order % cos.shape[1]].tolist()))
            row["published_edges"] = len(pub)
            row["published_edges_found"] = int(found.sum())
            row["published_max_abs_cos_diff"] = float(np.nanmax(np.abs(mine - pub["score_cos"].to_numpy())))
            row["published_edges_in_own_top_k"] = sum(pair in top for pair in zip(pub["TF_idx"].tolist(), pub["TG_idx"].tolist()))
        summary.append(row)

        for name, hits in effectors.items():
            local = [k for k, i in enumerate(tf_idx) if i in hits]
            if not hits:
                effector_status[name][str(label)] = "not in TF list"
            elif not local:
                effector_status[name][str(label)] = "inactive or not whitelisted"
            else:
                effector_status[name][str(label)] = "scored"
                k = local[0]
                effector_rows.append(pd.DataFrame({
                    "tf": name, "stage": label, "gene": [refs["tg_names"][j] for j in tg_idx], "tg_id": [refs["tg_ids"][j] for j in tg_idx],
                    "cosine": cos[k], "percentile": pct[k],
                }))
        print(f"stage {label}: {tf_idx.size} TFs x {tg_idx.size} TGs, cosine {row['cos_q00']:.3f}-{row['cos_q100']:.3f}"
              + (f", published edges in own top {top_k}: {row.get('published_edges_in_own_top_k')}/{row.get('published_edges')}" if "published_edges" in row else ""),
              flush=True)

    stages = [f"t{t + 1}" for t in range(n_stage)]
    ids = {"gene": refs["tg_names"], "tg_id": refs["tg_ids"]}
    for name, file_name in READOUT_FILES.items():
        pd.concat([pd.DataFrame(ids), pd.DataFrame(readouts[name], columns=stages)], axis=1).to_csv(output / file_name, sep="\t", index=False)
    pd.concat([pd.DataFrame(ids), pd.DataFrame(tg_active.T.astype(int), columns=stages)], axis=1).to_csv(output / "tg_active.tsv", sep="\t", index=False)
    if effector_rows:
        pd.concat(effector_rows, ignore_index=True).to_csv(output / "effector_percentiles.tsv", sep="\t", index=False)
    pd.DataFrame(summary).to_csv(output / "stage_summary.tsv", sep="\t", index=False)
    save_checkpoint_metadata(output / "manifest.json", {
        "lineage": args.lineage,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "artifact": args.artifact,
        "active_thr": active_thr,
        "n_stages": n_stage,
        "score": "percentile of the cosine of L2-normalised integer-stage TF and TG embeddings among all active whitelisted TF x active TG pairs of the stage",
        "receptor_readouts": {
            "receptor_input.tsv": "A: mean percentile over the active TFs",
            "receptor_input_top10pct_tfs.tsv": f"B: mean of the top {TOP_TF_SHARE:.0%} percentiles over the active TFs (k = ceil share x TFs)",
            "receptor_input_top5pct_pairs_frac.tsv": f"C: share of active TFs whose pair is in the top {TOP_PAIR_SHARE:.0%} of the stage's pairs",
            "receptor_input_ungated.tsv": "A without the activity gate: every TG ranked against the stage's active pairs, mean over the active TFs",
            "inactive": "0 where the TG is inactive",
        },
        "published_network_checked": str(published),
        "effector_status": effector_status,
        "versions": {"python": sys.version, "torch": torch.__version__},
    })
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
