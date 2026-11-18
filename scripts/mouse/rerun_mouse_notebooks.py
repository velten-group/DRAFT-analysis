"""Rerun the original mouse notebooks (original_notebooks/, cells checked by SHA-256) on grn_best.pt.

Network export from the checkpoint's saved scores, 12 edge clusters and the 10-topic LDA. Only paths and a Matplotlib
call removed in newer versions are adapted; GO enrichment uses the network's targets as background instead of the
notebooks' online Enrichr call.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"
# The topic cell fits the LDA with n_jobs=-1, which splits the documents over the joblib workers; the published topics
# come from 4 workers (another count gives other topics). Needs at least 4 logical CPUs.
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
os.environ["MPLBACKEND"] = "Agg"

from notebook_cells import (
    ENRICH_SHA,
    TOPIC_SHA,
    cell_source,
    digest,
    write_json,
)
from notebook_runner import (
    EDGE_HASHES,
    MODEL_HASHES,
    NotebookRenderer,
    replot_edge,
    replot_model,
)

EXPORT_SHA = "410145d7c615dda1185431749e287851b1d4c43361a7eca2663757ba92cfab57"
CLUSTERS = [4, 8, 7, 3, 2, 1, 5]


def write_index(output):
    figures = json.loads((output / "figures.json").read_text(encoding="utf-8"))
    lines = [
        "# Mouse notebooks rerun from grn_best.pt",
        "",
        "- [mouse_notebooks.ipynb](mouse_notebooks.ipynb): the executed cells with their output",
        "- [provenance.json](provenance.json): SHA-256 of the inputs and cells, settings",
        "- [topic_proportions.tsv](topic_proportions.tsv): normalised weights of the ten topics",
        "",
        "| # | Part | PNG | SVG |",
        "| --- | --- | --- | --- |",
    ]
    lines.extend(
        f"| {item['index']} | {item['section']} | [PNG]({item['stem']}.png) | [SVG]({item['stem']}.svg) |"
        for item in figures
    )
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def offline_enrichment_nodes(body):
    """The enrichment cell without its online Enrichr function."""
    result = []
    for node in body:
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "run_enrichment_per_topic"
        ):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            fn = node.value.func
            if isinstance(fn, ast.Name) and fn.id in {
                "run_enrichment_per_topic",
                "print",
            }:
                continue
        result.append(node)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", type=Path, default=Path(__file__).resolve().parent / "original_notebooks"
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="grn_best.pt")
    parser.add_argument("--data-dir", type=Path, required=True, help="folder of the cluster<N>.table files")
    parser.add_argument("--go-library", type=Path, required=True, help="GO_Biological_Process_2021.gmt")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.work_dir = args.run_dir
    for path in (args.run_dir, args.output_dir):
        if path.exists() and any(path.iterdir()):
            parser.error(f"Select a new empty directory; preserving {path}")
    archive = args.source_root
    checkpoint_hash = digest(args.checkpoint)
    if checkpoint_hash != "0a7599ba08e8d2fa7572212d1927dca81b508fad42a2aa81f91b3a23da6da725":
        raise ValueError("grn_best.pt differs from the deposited checkpoint (SHA-256)")
    export_source = cell_source(archive / "model.ipynb", 3, EXPORT_SHA)
    model_source = cell_source(archive / "model.ipynb", 0, MODEL_HASHES[0])
    topic_source = cell_source(archive / "topic.ipynb", 0, TOPIC_SHA)
    enrich_source = cell_source(archive / "topic.ipynb", 1, ENRICH_SHA)
    for index, sha in enumerate(EDGE_HASHES):
        cell_source(archive / "edge_cluster.ipynb", index, sha)
    args.run_dir.mkdir(parents=True)
    args.output_dir.mkdir(parents=True)
    started = datetime.now(timezone.utc).isoformat()
    inputs = [
        args.checkpoint,
        args.go_library,
        *[args.data_dir / f"cluster{cid}.table" for cid in CLUSTERS],
    ]
    provenance = {
        "started_utc": started,
        "inputs_sha256": {str(p.resolve()): digest(p) for p in inputs},
        "checkpoint_sha256": checkpoint_hash,
        "source_cell_sha256": {
            "export": EXPORT_SHA,
            "edge": EDGE_HASHES,
            "topic": TOPIC_SHA,
            "enrichment": ENRICH_SHA,
        },
        "fresh_network_export": True,
        "fresh_edge_fit": True,
        "fresh_topic_fit": True,
        "reused_fit_artifacts": False,
        "neural_training_or_forward": False,
        "online_calls": False,
        "adaptations": [
            "local input/output paths",
            "removed Matplotlib get_cmap API",
            "local network-background ORA replaces online Enrichr",
        ],
        "python": platform.python_version(),
    }
    write_json(args.run_dir / "manifest.json", provenance)
    renderer = NotebookRenderer(args.output_dir)
    renderer.markdown(
        "# Mouse notebooks rerun from grn_best.pt\n\n"
        "Network export from the checkpoint's saved scores, 12 edge clusters (k-means) and the 10-topic LDA, all "
        "fitted in this run. GO enrichment uses the network's targets as background instead of the online Enrichr call."
    )
    renderer.code(
        "import numpy as np\nimport pandas as pd\nimport torch\n"
        "import matplotlib.pyplot as plt\nfrom types import SimpleNamespace\n"
        "from pathlib import Path\nimport sys\n"
        f"sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})\n"
        "from notebook_runner import local_topic_go\n"
        f"checkpoint = torch.load({str(args.checkpoint.resolve())!r}, map_location='cpu', weights_only=False)\n"
        "meta = checkpoint['meta']\nTF_list = meta['TF_list']\nTG_list = meta['TG_list']\n"
        "topo = list(meta['topo'])\nassert topo == [4, 8, 7, 3, 2, 1, 5]\n"
        "sims_best = [np.asarray(s) for s in checkpoint['sims_best']]\n"
        f"cfg = SimpleNamespace(base_dir={str(args.data_dir.resolve())!r})\n"
    )
    renderer.original(
        model_source,
        archive / "model.ipynb",
        0,
        MODEL_HASHES[0],
        select=lambda body: [
            n
            for n in body
            if isinstance(n, ast.FunctionDef) and n.name == "clean_suffix"
        ],
    )
    renderer.section = "network_export"
    print("Fresh original network export", flush=True)
    renderer.original(
        export_source,
        archive / "model.ipynb",
        3,
        EXPORT_SHA,
        paths={"NETWORK_DIR": str((args.run_dir / "network").resolve())},
    )
    renderer.code("del checkpoint, sims_best\n")
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        replot_model(args, renderer)
        print("Fresh original 12-EC fit and TF calculations", flush=True)
        replot_edge(args, renderer)
    topic_dir = args.run_dir / "topic10"
    renderer.section = "topic"
    renderer.markdown(
        "## Fresh original LDA fit\n\n10 topics, batch, seed 42, 200 iterations; original float64 count matrix."
    )
    renderer.code("plt.rcdefaults()\nplt.rcParams['svg.fonttype'] = 'none'\n")
    print("Starting fresh original 10-topic LDA fit (no cached W/H)", flush=True)
    renderer.original(
        topic_source,
        archive / "topic.ipynb",
        0,
        TOPIC_SHA,
        paths={
            "NETWORK_DIR": str((args.run_dir / "network").resolve()),
            "OUTPUT_DIR": str(topic_dir.resolve()),
        },
    )
    ns = renderer.namespace
    h = ns["H"]
    import pandas as pd

    pd.DataFrame(
        h / h.sum(axis=1, keepdims=True),
        index=[f"T{i}" for i in range(10)],
        columns=[f"C{i}" for i in ns["CLUSTERS"]],
    ).to_csv(topic_dir / "topic_proportions.tsv", sep="\t")
    matrix = ns["C_sparse"]
    import hashlib

    import numpy as np
    import sklearn

    provenance.update(
        {
            "network_sha256": {
                p.name: digest(p)
                for p in sorted((args.run_dir / "network").glob("C*_network.txt"))
            },
            "lda_parameters": ns["lda"].get_params(),
            "matrix_shape": list(matrix.shape),
            "matrix_dtype": str(matrix.dtype),
            "matrix_sha256": {
                key: hashlib.sha256(getattr(matrix, key).tobytes()).hexdigest()
                for key in ("data", "indices", "indptr")
            },
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "lda_artifact_sha256": digest(topic_dir / "lda_artifacts.pkl"),
        }
    )
    print("Fresh LDA fit finished; computing local GO", flush=True)
    renderer.original(
        enrich_source,
        archive / "topic.ipynb",
        1,
        ENRICH_SHA,
        paths={"OUTPUT_DIR": str(topic_dir.resolve())},
        select=offline_enrichment_nodes,
    )
    renderer.code(
        f"enrich_dict = local_topic_go(art, {str(args.go_library.resolve())!r}, {str((topic_dir / 'local_go').resolve())!r})\n"
        "draw_all_topics_enrichment_grid(enrich_dict, W.shape[1], 'topic10_local_go.png', 'GO BP2021 | local network background')\n"
    )
    renderer.save()
    shutil.copyfile(
        topic_dir / "topic_proportions.tsv", args.output_dir / "topic_proportions.tsv"
    )
    provenance["finished_utc"] = datetime.now(timezone.utc).isoformat()
    provenance["n_figures"] = len(renderer.figures)
    write_json(args.run_dir / "manifest.json", provenance)
    write_json(args.output_dir / "provenance.json", provenance)
    write_index(args.output_dir)
    print(
        json.dumps(
            {
                "completed": str(args.output_dir.resolve()),
                "n_figures": len(renderer.figures),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
