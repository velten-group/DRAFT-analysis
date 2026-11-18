"""Three-module main design: fixed inside and outside candidates per TG, with one-way replacement of true edges.

A replaced edge never returns and an added edge is never removed. The inside/outside mixture of true edges stays fixed.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def write_design(folder, params):
    folder = Path(folder)
    audit, inputs = folder / "generator_only", folder / "multiome"
    audit.mkdir(parents=True, exist_ok=True)
    inputs.mkdir(parents=True, exist_ok=True)
    ntf = int(params["num_tf"])
    ntg = int(params["num_genes"]) - ntf
    active, active_in = int(params["active_per_tg"]), int(params["active_within"])
    cand_in, cand_out = int(params["candidate_inside"]), int(params["candidate_outside"])
    times, modules = int(params["times"]), int(params["modules"])
    rng = np.random.default_rng(int(params["seed"]))
    dynamics = np.random.default_rng(np.random.SeedSequence([int(params["seed"]), 811]))
    tf_module = rng.permutation(np.arange(ntf) % modules)
    tg_module = rng.permutation(np.arange(ntg) % modules)
    same = tg_module[:, None] == tf_module[None, :]
    candidate, initial = np.zeros_like(same), np.zeros_like(same)
    for g in range(ntg):
        for local, count, number in ((True, cand_in, active_in), (False, cand_out, active - active_in)):
            chosen = rng.choice(np.flatnonzero(same[g] == local), count, replace=False)
            candidate[g, chosen] = True
            initial[g, dynamics.choice(chosen, number, replace=False)] = True
    changes = round(float(params["change_fraction"]) * ntg * active)
    per_stratum = {True: round(changes * active_in / active)}
    per_stratum[False] = changes - per_stratum[True]
    states, events = [initial], []
    for t in range(1, times):
        previous = states[-1]
        state = previous.copy()
        for local, count in per_stratum.items():
            allowed = candidate & (same == local)
            old = [np.flatnonzero(previous[g] & initial[g] & allowed[g]) for g in range(ntg)]
            new = [np.flatnonzero(~previous[g] & ~initial[g] & allowed[g]) for g in range(ntg)]
            capacity = np.array([min(len(a), len(b)) for a, b in zip(old, new)])
            slots = np.repeat(np.arange(ntg), capacity)
            if count > len(slots):
                raise ValueError("Change rate exceeds one-way capacity in a stratum")
            allocation = np.bincount(slots[dynamics.choice(slots.size, count, replace=False)], minlength=ntg)
            for g in np.flatnonzero(allocation):
                n = int(allocation[g])
                removed = dynamics.choice(old[g], n, replace=False)
                added = dynamics.choice(new[g], n, replace=False)
                state[g, removed], state[g, added] = False, True
                events.extend((t, int(g), int(a), int(b)) for a, b in zip(removed, added))
        states.append(state)
    states = np.stack(states)
    if not np.all(states.sum(2) == active) or not np.all((states & same).sum(2) == active_in):
        raise AssertionError("Active degrees/module mixture drifted")
    mapper = np.random.default_rng(np.random.SeedSequence([int(params["seed"]), 913]))
    motif, distance = [], []
    for g in range(ntg):
        chosen = mapper.permutation(np.flatnonzero(candidate[g]))
        for chunk in np.array_split(chosen, len(chosen) // int(params["tfs_per_region"])):
            tr, rg = np.zeros(ntf, bool), np.zeros(ntg, bool)
            tr[chunk], rg[g] = True, True
            motif.append(tr)
            distance.append(rg)
    motif, distance = np.array(motif), np.array(distance)
    genes = [str(i + 1) for i in range(ntf + ntg)]
    tfs, targets = genes[:ntf], genes[ntf:]
    regions = [f"RE_{i + 1}" for i in range(len(motif))]
    for matrix, index, columns, name in (
        (motif, regions, tfs, audit / "potential_motif.csv"),
        (distance, regions, targets, audit / "potential_distance.csv"),
        (motif, regions, tfs, inputs / "tf_re_prior.csv"),
        (distance.T, targets, regions, inputs / "re_tg_prior.csv"),
        (candidate, targets, tfs, folder / "candidate_prior.csv"),
    ):
        pd.DataFrame(matrix.astype(int), index=index, columns=columns).to_csv(name, lineterminator="\n")
    np.testing.assert_array_equal(distance.T.astype(int) @ motif.astype(int) > 0, candidate)
    activity = []
    for t, state in enumerate(states):
        pd.DataFrame(state.astype(int), index=targets, columns=tfs).to_csv(audit / f"weights_time_{t}.csv", lineterminator="\n")
        full = np.zeros((ntf + ntg, ntf + ntg), dtype=int)
        full[ntf:, :ntf] = state
        pd.DataFrame(full, index=genes, columns=genes).to_csv(folder / f"grn_time_{t}.csv", lineterminator="\n")
        activity.append(np.sum((distance @ state.astype(float)) * motif, axis=1) / motif.sum(1))
    pd.DataFrame(np.array(activity).T, index=regions).to_csv(audit / "region_activity.csv", lineterminator="\n")
    pd.DataFrame({"gene": genes, "module": np.r_[tf_module, tg_module]}).to_csv(
        folder / "gene_modules.tsv", sep="\t", index=False, lineterminator="\n"
    )
    pd.DataFrame(events, columns=["time", "target_index", "old_tf_index", "new_tf_index"]).to_csv(
        audit / "transitions.tsv", sep="\t", index=False, lineterminator="\n"
    )
    return states, candidate, motif, distance
