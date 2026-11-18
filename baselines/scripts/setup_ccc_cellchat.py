"""Fetch pinned upstream CellChat inference sources and local Windows R binaries.

All downloads and R packages stay inside the project cache. No system R library
or proxy configuration is modified. CellChat inference is sourced verbatim;
the full visualization package and its unrelated dependencies are not installed.
"""

from __future__ import annotations

import concurrent.futures
import gzip
import hashlib
import json
import re
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache/ccc_cellchat"
COMMIT = "75253cd0c9e68410e6e721a6d3a0419a1d7e358f"
REPOSITORY = "https://cloud.r-project.org/bin/windows/contrib/4.6"
BASE = {
    "R",
    "base",
    "boot",
    "class",
    "cluster",
    "codetools",
    "compiler",
    "datasets",
    "foreign",
    "graphics",
    "grDevices",
    "grid",
    "KernSmooth",
    "lattice",
    "MASS",
    "Matrix",
    "methods",
    "mgcv",
    "nlme",
    "nnet",
    "parallel",
    "rpart",
    "spatial",
    "splines",
    "stats",
    "stats4",
    "survival",
    "tcltk",
    "tools",
    "translations",
    "utils",
}


def fetch(url: str, path: Path) -> bytes:
    if path.exists():
        return path.read_bytes()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=120) as response:
        content = response.read()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return content


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    entries = {}
    content = gzip.decompress(
        fetch(f"{REPOSITORY}/PACKAGES.gz", CACHE / "PACKAGES.gz")
    ).decode()
    for block in content.split("\n\n"):
        fields = {}
        key = None
        for line in block.splitlines():
            if line.startswith((" ", "\t")) and key:
                fields[key] += " " + line.strip()
            elif ": " in line:
                key, value = line.split(": ", 1)
                fields[key] = value
        if "Package" in fields:
            entries[fields["Package"]] = fields
    wanted = set()

    def include(name: str) -> None:
        if name in wanted or name in BASE:
            return
        fields = entries[name]
        wanted.add(name)
        for category in ("Depends", "Imports"):
            for dep in fields.get(category, "").split(","):
                dep = re.sub(r"\s*\(.*", "", dep).strip()
                if dep:
                    include(dep)

    for name in ("future", "future.apply", "pbapply", "collapse", "dplyr", "jsonlite"):
        include(name)

    def provision(name: str) -> dict:
        fields = entries[name]
        url = f"{REPOSITORY}/{name}_{fields['Version']}.zip"
        path = CACHE / "binaries" / url.rsplit("/", 1)[-1]
        binary = fetch(url, path)
        if fields.get("MD5sum"):
            assert hashlib.md5(binary).hexdigest() == fields["MD5sum"], name
        library = CACHE / "R-library"
        library.mkdir(exist_ok=True)
        with zipfile.ZipFile(path) as archive:
            for member in archive.namelist():
                assert (library / member).resolve().is_relative_to(library.resolve())
            archive.extractall(library)
        print(f"Provisioned {name} {fields['Version']}", flush=True)
        return {
            "package": name,
            "version": fields["Version"],
            "url": url,
            "sha256": hashlib.sha256(binary).hexdigest(),
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        packages = list(executor.map(provision, sorted(wanted)))
    sources = []
    (CACHE / "CellChat").mkdir(parents=True, exist_ok=True)
    (CACHE / "CellChat/source_commit.txt").write_text(COMMIT + "\n", encoding="utf-8")
    for filename in (
        "DESCRIPTION",
        "LICENSE",
        "R/CellChat_class.R",
        "R/modeling.R",
        "R/utilities.R",
        "R/database.R",
    ):
        url = f"https://raw.githubusercontent.com/jinworks/CellChat/{COMMIT}/{filename}"
        binary = fetch(url, CACHE / "CellChat" / filename)
        sources.append(
            {"file": filename, "url": url, "sha256": hashlib.sha256(binary).hexdigest()}
        )
    (CACHE / "source_manifest.json").write_text(
        json.dumps(
            {
                "commit": COMMIT,
                "mode": "verbatim upstream R inference sources; not a full CellChat package installation",
                "packages": packages,
                "sources": sources,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
