"""Package the accepted native mouse scores and provenance; standard library only."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import re
import tarfile
from datetime import datetime, timezone
from pathlib import Path

RUN = "shared_prior_s2025_20261014T084844Z_ecdf2a70"
CLUSTERS = (4, 8, 7, 3, 2, 1, 5)
GOLD_HASHES = {"ChIP": "8629656f0958fa713a5473760d1a9d33d341f24fe6efe78bf243de7ccb59473d",
               "Perturb": "1b24baf64ed1991c87d721b6cef807e951b6f787e25d20f0509b6462b5d7aece",
               "Perturb+ChIP": "80da24f556684ef088501262f34296ea0f3efdb24c1aa84ce04b10b4da97963d"}
SCOPE_HASH = "16465c206e4ad58ec81bf1845b6c97c512d4df62e59c43d6b5b031207ee67f0f"
REPORT_FILES = ("evaluation_report.json", "metrics_by_cluster.tsv", "metrics_summary.tsv",
                "evaluation_scope.tsv", "sample_identity.tsv", "score_coverage.tsv")


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for data in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(data)
    return value.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def collect(root, run):
    if not re.fullmatch(r"[A-Za-z0-9._-]+", run) or run in {".", ".."}:
        raise ValueError("The run must be a directory name, not a path.")
    prepared = root / "comparisons/mouse_benchmark/runs" / run
    manifest_path = prepared / "manifest.json"
    manifest = read_json(manifest_path)
    if (manifest.get("validation_only") is not False or len(manifest.get("clusters", [])) != 7
            or set(manifest["clusters"]) != set(CLUSTERS)):
        raise ValueError("Expected a real, complete seven-cluster prepared run.")
    manifest_hash = digest(manifest_path)
    features_path = prepared / "features.json"
    features = read_json(features_path)
    for key in ["TF", "TG"]:
        if not features.get(key) or len(features[key]) != len(set(features[key])):
            raise ValueError(f"Missing/duplicate prepared {key} axes.")
    approved = root / "docs/mouse/current/source_data/random_distance_inputs.json"
    if approved.exists():
        reference = read_json(approved)
        if any(features[key] != reference[key] for key in ["TF", "TG"]):
            raise ValueError("Prepared axes differ from the adopted DRAFT checkpoint axes.")
    files, native_hashes, omitted = {}, {}, {}

    def add(path, expected=None):
        if not path.is_file():
            raise FileNotFoundError(f"Required input missing: {path}")
        sha = digest(path)
        if expected is not None and sha != expected:
            raise ValueError(f"Recorded file hash differs: {path}")
        relative = path.relative_to(root).as_posix()
        files[relative] = {"path": relative, "sha256": sha, "bytes": path.stat().st_size}
        return sha

    add(manifest_path)
    add(features_path, manifest.get("files_sha256", {}).get("features.json"))
    if (prepared / "input_summary.tsv").is_file():
        add(prepared / "input_summary.tsv")
    for cid in CLUSTERS:
        path = prepared / f"C{cid}/regulators.txt"
        if path.is_file():
            add(path, manifest.get("files_sha256", {}).get(f"C{cid}/regulators.txt"))
    for method, folder in [("Dictys", "mouse_dictys"), ("SCENIC", "mouse_scenic")]:
        directory = root / "comparisons" / folder / "runs" / run
        runtime = directory / "runtime.json"
        if read_json(runtime).get("manifest_sha256") != manifest_hash:
            raise ValueError(f"{method} runtime refers to different prepared inputs.")
        add(runtime)
        native_hashes[method] = {}
        for cid in CLUSTERS:
            stage = directory / f"C{cid}"
            complete_path = stage / "complete.json"
            complete = read_json(complete_path)
            if (complete.get("validation_only") is not False or complete.get("model") != method
                    or complete.get("cluster") != cid
                    or complete.get("input_manifest_sha256") != manifest_hash):
                raise ValueError(f"Wrong, synthetic or mismatched completed output: {stage}")
            expected = complete.get("files_sha256", {}).get("edges.tsv.gz")
            if not expected:
                raise ValueError(f"Native edge hash absent from completion metadata: {stage}")
            edge_path = stage / "edges.tsv.gz"
            sha = add(edge_path, expected)
            with gzip.open(edge_path, "rt", encoding="utf-8", newline="") as handle:
                if not {"TF", "TG", "score"}.issubset(next(csv.reader(handle, delimiter="\t"))):
                    raise ValueError(f"Invalid native score-table columns: {edge_path}")
            add(complete_path)
            native_hashes[method][f"C{cid}"] = sha
            omitted[f"{method}/C{cid}"] = sorted(set(complete["files_sha256"]) - {"edges.tsv.gz"})

    reports = sorted((root / "result").glob("mouse*/evaluation_report.json"))
    reports += sorted(prepared.glob("evaluation*/evaluation_report.json"))
    selected, skipped = [], []
    for path in reports:
        report = read_json(path)
        if (report.get("status") != "passed" or report.get("validation_only") is not False
                or report.get("n_clusters") != 7
                or not {"Dictys", "SCENIC"}.issubset(report.get("methods", []))):
            continue
        protocol = report.get("protocol", {})
        if protocol.get("gold_sha256") != GOLD_HASHES:
            continue
        reported_hashes = {key.replace("\\", "/"): value for key, value in report.get("input_sha256", {}).items()}
        expected = [row for row in files.values() if row["path"].endswith("/edges.tsv.gz")]
        if not all(any((name == row["path"] or name.endswith("/" + row["path"])) and value == row["sha256"]
                       for name, value in reported_hashes.items()) for row in expected):
            skipped.append(path.relative_to(root).as_posix())
            continue
        selected.append(path.parent)
    aupr_directories = {str(p.resolve()) for p in selected}
    for path in reports:
        report = read_json(path)
        if (report.get("status") == "passed" and report.get("validation_only") is False
                and report.get("n_clusters") == 7 and report.get("source_scope_sha256") == SCOPE_HASH
                and {"Dictys", "SCENIC"}.issubset(report.get("methods", []))
                and aupr_directories.intersection(report.get("source_directories", []))):
            selected.append(path.parent)
    for directory in sorted(set(selected)):
        expected = read_json(directory / "evaluation_report.json").get("outputs_sha256", {})
        for name in REPORT_FILES:
            if (directory / name).is_file():
                add(directory / name, expected.get(name))
    record = {"schema": "mouse_native_raw_support_v1", "run": run, "clusters": list(CLUSTERS),
              "methods": ["Dictys", "SCENIC"], "native_edge_files": 14,
              "shape_each_network": [len(features["TF"]), len(features["TG"])],
              "prepared_manifest_sha256": manifest_hash, "features_sha256": digest(features_path),
              "native_edge_sha256": native_hashes,
              "evaluation_directories": [p.relative_to(root).as_posix() for p in sorted(set(selected))],
              "reports_skipped_due_to_native_hash_mismatch": skipped,
              "omitted_other_native_outputs": omitted,
              "source_files_modified": False, "training_or_evaluation_run": False,
              "subset_note": "Prediction/provenance subset only; original complete.json is unmodified and lists additional native outputs not bundled.",
              "files": [files[key] for key in sorted(files)]}
    return record


class HashingReader:
    def __init__(self, handle):
        self.handle, self.hasher = handle, hashlib.sha256()

    def read(self, size=-1):
        value = self.handle.read(size)
        self.hasher.update(value)
        return value


def package(root, run, output):
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    record = collect(root, run)  # All required files are checked before creating output.
    print(f"Validated 14 native score files and {len(record['files'])} source files; packing {sum(r['bytes'] for r in record['files']) / 1024**2:.1f} MiB.", flush=True)
    record["created_utc"] = datetime.now(timezone.utc).isoformat()
    output.mkdir(parents=True)
    prefix = f"mouse_native_raw_{run}"
    archive = output / f"{prefix}.tar.gz"
    partial = output / f"{prefix}.tar.gz.partial"
    with tarfile.open(partial, "w:gz", compresslevel=1, dereference=True) as bundle:
        for row in record["files"]:
            source = root / row["path"]
            info = bundle.gettarinfo(str(source), arcname=f"{prefix}/{row['path']}")
            with source.open("rb") as handle:
                reader = HashingReader(handle)
                bundle.addfile(info, reader)
                if reader.hasher.hexdigest() != row["sha256"]:
                    raise ValueError(f"Source changed while packaging: {source}")
        raw = (json.dumps(record, indent=2) + "\n").encode()
        info = tarfile.TarInfo(f"{prefix}/PACKAGE_MANIFEST.json")
        info.size = len(raw)
        bundle.addfile(info, io.BytesIO(raw))
    partial.rename(archive)
    (output / f"{archive.name}.sha256").write_text(f"{digest(archive)}  {archive.name}\n", encoding="ascii")
    (output / "PACKAGE_MANIFEST.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return archive, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--run", default=RUN)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.project_root.resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output_dir or root / "result" / f"mouse_native_raw_export_{stamp}").resolve()
    archive, record = package(root, args.run, output)
    print(f"ARCHIVE: {archive}\nCHECKSUM: {archive}.sha256")
    print(f"Native score files: 14; total source files: {len(record['files'])}; archive size: {archive.stat().st_size / 1024**2:.1f} MiB")
    print(f"Matched evaluation reports: {len(record['evaluation_directories'])}; no training/evaluation rerun.")
    if not record["evaluation_directories"]:
        print("NOTE: No matching official evaluation reports found; native scores and provenance are included for local recalculation.")


if __name__ == "__main__":
    main()
