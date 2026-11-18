"""Derive a soma-germline alignment in which the parallel terminal soma branches share one time point.

The sequential alignment places every soma group in sequence, so the parallel branches SdA
(Soma_7a) and SdB (Soma_7b) sit one after the other (germline 14.0 and 15.0 on the DTW path). The soma groups are
parallel terminal branches of cyst cells enclosing spermatocytes (configs/fly/soma_stage_labels.json), so here they form
one time point (soma_branch_label, entered by scripts/run_ccc_unified.py as the mean of the branches). The path is kept
unchanged up to the last soma group before the branches; the shared time point is paired with the last germline stage,
and the stretch between them is linear.

    python scripts/make_ccc_parallel_alignment.py configs/fly/ccc_frozen_dtw_soma_to_germ.tsv configs/fly/ccc_frozen_dtw_soma_to_germ_parallel.tsv
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import yaml
from run_ccc_unified import path, soma_time_points

STEP = 0.2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source", help="sequential alignment (t_soma_net over all soma groups, t_germ_warped)")
    parser.add_argument("target")
    parser.add_argument("--config", default="configs/fly/ccc_unified.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(path(args.config).read_text(encoding="utf-8"))
    meta = json.loads(path(cfg["soma_stage_labels"]).read_text(encoding="utf-8"))
    groups = meta["groups"]
    n_germ = len(json.loads(path(cfg["germ_stage_labels"]).read_text(encoding="utf-8"))["labels"])
    branches = list(cfg.get("soma_parallel_branches") or [])
    if not branches or groups[-len(branches):] != branches:
        raise SystemExit(f"soma_parallel_branches {branches} must be the terminal soma groups {groups}")
    nodes, _ = soma_time_points(cfg, groups, groups)
    last_linear = len(nodes) - 1  # 1-based soma position of the last group before the branches
    source = pd.read_csv(path(args.source), sep="\t")
    if not np.isclose(source.t_soma_net.max(), len(groups)):
        raise SystemExit(f"{args.source} is not a sequential path over the {len(groups)} soma groups")
    kept = source[source.t_soma_net <= last_linear + 1e-9]
    g_last = float(np.interp(last_linear, source.t_soma_net, source.t_germ_warped))
    tail = np.round(np.arange(last_linear + STEP, len(nodes) + 1e-9, STEP), 6)
    extra = pd.DataFrame({"t_soma_net": tail, "t_germ_warped": g_last + (tail - last_linear) * (n_germ - g_last)})
    out = pd.concat([kept, extra], ignore_index=True)
    out.to_csv(path(args.target), sep="\t", index=False)
    print(f"{args.target}: soma time points 1-{len(nodes)}; {nodes[-1]} paired with germline {n_germ}; "
          f"last sequential group at germline {g_last:.2f}")


if __name__ == "__main__":
    main()
