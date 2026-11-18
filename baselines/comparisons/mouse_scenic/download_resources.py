"""Resume official mm10/mc9nr gene-ranking resources inside this baseline only."""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = "https://resources.aertslab.org/cistarget/databases/mus_musculus/mm10/refseq_r80/mc9nr/gene_based/"
DATABASES = [
    "mm10__refseq-r80__10kb_up_and_down_tss.mc9nr.genes_vs_motifs.rankings.feather",
    "mm10__refseq-r80__500bp_up_and_100bp_down_tss.mc9nr.genes_vs_motifs.rankings.feather",
]
ANNOTATIONS = "motifs-v9-nr.mgi-m0.001-o0.0.tbl"


def sha1(path: Path) -> str:
    result = hashlib.sha1()
    with path.open("rb") as handle:
        for data in iter(lambda: handle.read(2**20), b""):
            result.update(data)
    return result.hexdigest()


def download(url: str, path: Path) -> None:
    if path.exists():
        print(f"Already present: {path}", flush=True)
        return
    partial = path.with_suffix(path.suffix + ".partial")
    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(
        url, headers={"Range": f"bytes={offset}-"} if offset else {}
    )
    print(f"Downloading official resource: {url}", flush=True)
    with urllib.request.urlopen(request, timeout=120) as response:
        append = offset > 0 and response.status == 206
        with partial.open("ab" if append else "wb") as handle:
            while data := response.read(2**20):
                handle.write(data)
    partial.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "resources/mm10_mc9nr")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((HERE / "resources").resolve()):
        parser.error(
            "Resource outputs must stay in comparisons/mouse_scenic/resources/."
        )
    output.mkdir(parents=True, exist_ok=True)
    for name in DATABASES:
        download(BASE + name + ".sha1sum.txt", output / (name + ".sha1sum.txt"))
        expected = (output / (name + ".sha1sum.txt")).read_text().split()[0]
        download(BASE + name, output / name)
        if sha1(output / name) != expected:
            raise ValueError(
                f"Official SHA1 mismatch: {output / name}; do not use this database."
            )
    download(
        "https://resources.aertslab.org/cistarget/motif2tf/" + ANNOTATIONS,
        output / ANNOTATIONS,
    )
    print(f"SCENIC resources: {output}", flush=True)


if __name__ == "__main__":
    main()
