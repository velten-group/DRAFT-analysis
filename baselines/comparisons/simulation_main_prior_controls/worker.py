"""Official adapters on the five main-prior interventions."""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/simulation_gnn"))
sys.path.insert(0, str(ROOT / "comparisons/simulation_benchmark"))
import common
import multiome_inputs
from controlled_prior_design import OFFICIAL, digest, write_json
from main_prior_controls_design import CASES, VERSION
from main_prior_controls_metrics import evaluate


@contextlib.contextmanager
def lock(path):
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def load(folder):
    """Only public input contents enter this object; audit files are hashed only."""
    receipt = json.loads(
        (folder / "generation_complete.json").read_text(encoding="utf-8")
    )
    for name, sha in receipt["files"].items():
        if digest(folder / name) != sha:
            raise ValueError(f"Input checksum changed: {folder / name}")
    genes = tuple(str(i + 1) for i in range(200))
    tr = multiome_inputs.matrix(folder / "multiome/tf_re_prior.csv")
    rg = multiome_inputs.matrix(folder / "multiome/re_tg_prior.csv")
    p = multiome_inputs.matrix(folder / "candidate_prior.csv")
    if (
        tuple(tr.columns) != genes[:50]
        or tuple(rg.index) != genes[50:]
        or tuple(rg.columns) != tuple(tr.index)
        or tuple(p.index) != genes[50:]
        or tuple(p.columns) != genes[:50]
    ):
        raise ValueError("Public axes differ")
    arrays = [a.to_numpy() for a in (tr, rg, p)]
    if any(not np.isin(a, [0, 1]).all() for a in arrays):
        raise ValueError("Public priors are not binary")
    tr_array, rg_array, p_array = [a.astype(bool) for a in arrays]
    np.testing.assert_array_equal(
        rg_array.astype(int) @ tr_array.astype(int) > 0, p_array
    )
    counts = tuple(
        common.load_counts(folder / f"counts_time_{t}.csv", genes) for t in range(5)
    )
    atac = tuple(
        multiome_inputs.matrix(folder / f"multiome/atac_time_{t}.csv") for t in range(5)
    )
    for x, a in zip(counts, atac):
        values = a.to_numpy()
        if (
            tuple(a.index) != tuple(tr.index)
            or not a.columns.equals(x.columns)
            or (values < 0).any()
            or not np.equal(values, np.floor(values)).all()
        ):
            raise ValueError("ATAC/RNA cell alignment or counts differ")
    for a in (tr_array, rg_array, p_array):
        a.setflags(write=False)
    return multiome_inputs.MultiomeInputs(
        tuple(range(5)),
        genes,
        genes[:50],
        genes[50:],
        counts,
        p_array,
        tuple(tr.index),
        atac,
        tr_array,
        rg_array,
    ), receipt["files"]


def run(args):
    version = VERSION + ("_smoke" if args.smoke else "")
    folder = ROOT / "data" / version / args.case / f"seed_{args.data}"
    data, files = load(folder)
    method_dir = ROOT / "comparisons" / f"simulation_{args.method}"
    if args.method == "scmtni":
        settings_path = Path(
            os.environ.get(
                "SCMTNI_SETTINGS",
                ROOT
                / "publication/simulation/configuration/scmtni_selected_settings.json",
            )
        )
    else:
        settings_path = method_dir / (
            "settings_patience50.json" if args.method == "scdori" else "settings.json"
        )
    settings = common.read_json(settings_path)
    adapter = common.load_module(
        "_controlled_official_adapter", method_dir / "adapter.py"
    )
    software = adapter.software(method_dir)
    software["packages"] = {
        d.metadata["Name"]: d.version
        for d in importlib.metadata.distributions()
        if d.metadata["Name"]
    }
    software["python"] = sys.version
    code = {
        str(p.relative_to(ROOT)): digest(p)
        for p in sorted(
            [
                *method_dir.glob("*.py"),
                *Path(common.__file__).parent.glob("*.py"),
                Path(__file__),
                Path(__file__).parents[2]
                / "src/simulation_gnn/controlled_prior_metrics.py",
                ROOT / "src/simulation_gnn/main_prior_controls_design.py",
                ROOT / "src/simulation_gnn/main_prior_controls_metrics.py",
            ]
        )
    }
    identity = {
        "version": version,
        "case": args.case,
        "method": args.method,
        "data": args.data,
        "seed": args.seed,
        "sources": files,
        "settings": settings,
        "software": software,
        "adapter_code": code,
        "cpus": args.cpus,
        "device": args.device,
        "smoke": args.smoke,
        "input_policy": multiome_inputs.POLICIES[args.method],
        "evaluation": "fixed 1800 main pairs plus fixed all-condition union;no additional clamping",
    }
    run_dir = (
        method_dir
        / "runs"
        / version
        / args.case
        / f"seed_{args.data}"
        / f"model_{args.seed}"
        / args.method
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    with lock(run_dir / ".fit.lock"):
        complete = run_dir / "complete.json"
        if complete.is_file():
            saved = common.read_json(complete)
            if (
                saved["identity"] != identity
                or digest(run_dir / "relations.npz") != saved["prediction_sha256"]
            ):
                raise ValueError(f"Completed official fit identity changed: {run_dir}")
            print("[REUSE] " + str(run_dir), flush=True)
            return
        manifest = run_dir / "manifest.json"
        if manifest.is_file() and common.read_json(manifest) != identity:
            raise ValueError(
                "Interrupted official fit has a different identity;use a new version"
            )
        write_json(manifest, identity)
        # SCENIC+ TF2G has no edge-prior input. Fit once per RNA/seed/protocol,
        # then rescore the identical tensor in all prior-only interventions.
        native_identity = dict(identity)
        if args.method == "scenicplus":
            for k in ("case", "sources", "evaluation"):
                native_identity.pop(k)
            native_identity["rna_sources"] = {
                n: s for n, s in files.items() if n.startswith("counts_time_")
            }
            native_root = (
                method_dir
                / "runs"
                / version
                / "_shared_rna"
                / common.fingerprint(native_identity)
            )
        else:
            native_root = run_dir / "native"
        native_root.mkdir(parents=True, exist_ok=True)
        with lock(native_root / ".native.lock"):
            native_marker = native_root / "complete.json"
            reused = native_marker.is_file()
            if reused:
                saved = common.read_json(native_marker)
                if (
                    saved["identity"] != native_identity
                    or digest(native_root / "scores.npy") != saved["sha256"]
                ):
                    raise ValueError("Native shared cache identity changed")
                scores = np.load(native_root / "scores.npy", allow_pickle=False)
                provenance = saved["provenance"]
                print("[NATIVE_REUSE] " + str(native_root), flush=True)
            else:
                scores, provenance = adapter.fit(
                    data,
                    settings,
                    native_root,
                    args.seed,
                    args.cpus,
                    args.device,
                    args.smoke,
                )
                scores = np.asarray(scores, dtype=float)
                if scores.shape != (5, 150, 50) or not np.isfinite(scores).all():
                    raise ValueError(
                        "Native output is incomplete/nonfinite;not marking cache complete"
                    )
                np.save(native_root / "scores.npy", scores)
                write_json(
                    native_marker,
                    {
                        "identity": native_identity,
                        "provenance": provenance,
                        "sha256": digest(native_root / "scores.npy"),
                    },
                )
        # Truth and evaluation labels are read after the native fitting call.
        truth, same = multiome_inputs.truth_and_modules(folder, data)
        mask = multiome_inputs.matrix(folder / "evaluation_mask.csv").to_numpy(bool)
        union = multiome_inputs.matrix(folder / "evaluation_only/union_mask.csv").to_numpy(
            bool
        )
        metrics = evaluate(scores, truth, data.prior, mask, same, union_mask=union)
        for name, sha in files.items():
            if digest(folder / name) != sha:
                raise ValueError("Input changed during native fit")
        np.savez_compressed(run_dir / "relations.npz", scores=scores)
        write_json(
            complete,
            {
                "identity": identity,
                "metrics": metrics,
                "provenance": provenance,
                "native_reused": reused,
                "prediction_sha256": digest(run_dir / "relations.npz"),
            },
        )
        print(
            "[RESULT] "
            + json.dumps(dict(method=args.method, case=args.case, **metrics)),
            flush=True,
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", choices=CASES, required=True)
    p.add_argument("--data", type=int, choices=(101, 102, 103), required=True)
    p.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    p.add_argument("--method", choices=OFFICIAL, required=True)
    p.add_argument("--cpus", type=int, default=2)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.cpus < 1:
        p.error("cpus must be positive")
    run(args)


if __name__ == "__main__":
    main()
