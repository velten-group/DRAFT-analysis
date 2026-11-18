"""Run cells of the original mouse notebooks and collect them, with their output and figures, into a notebook."""

from __future__ import annotations

import ast
import base64
import contextlib
import io
import json
import platform
from pathlib import Path

from notebook_cells import cell_source

EDGE_HASHES = [
    "d00537a4965144af3c51eb35ea7460585869c4d1463d9a9c5c74ec7e5fce570c",
    "bdc09067abe6cff4f3f2611ca0dd45a8fc7dbfa3ddba65640c3a3b2eb3db9cbf",
    "73a7d6a425d54318f78e6d3747cd33589ccd246ed57f2836b3152b3054fb9e9e",
    "5477d08665784d63a00ed4860b590d3ac6ff900764dfdef6caa9c2fe5f09bda7",
    "9d0c258e10259a87191a8ce5a3e24bf2bf4ab94bea49dcc91ba98a697c32604b",
    "c134f6f022623793e644cfa345d18eee599af82c8c8df66db4ce7f4ef0d7eb42",
]
MODEL_HASHES = [
    "72c45f1bce5b3731f832c92ba0d1cf370b821c490fbe43d0e8af3c458187029e",
    "160b0ad636f940ebd3275ee9d321e364496d89de863c445fb81bf735014e77c3",
    "74cb7a327e6ed83e243933c6674b225d14b6c055b574414fde570fa43f1c3f91",
]


class PlotCompatibility(ast.NodeTransformer):
    """Set the cells' path variables for this run and replace the Matplotlib colormap accessor removed in newer versions."""

    def __init__(self, paths):
        self.paths = paths

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in self.paths:
                node.value = ast.copy_location(
                    ast.Constant(self.paths[target.id]), node.value
                )
        return self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module == "matplotlib.cm" and [n.name for n in node.names] == [
            "get_cmap"
        ]:
            return ast.parse(
                "from matplotlib import colormaps\nget_cmap = colormaps.get_cmap"
            ).body
        return node

    def visit_Attribute(self, node):
        node = self.generic_visit(node)
        if (
            node.attr == "get_cmap"
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "cm"
        ):
            node.value.attr = "colormaps"
        return node


def local_topic_go(art, library_path, output):
    """GO over-representation of each topic's top-500 edge targets against all network targets (replaces the online Enrichr call)."""
    import numpy as np
    import pandas as pd
    from enrich_mouse_topics import background_ora, parse_library

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    library = parse_library(library_path)
    background = list(art["tg_idx2name"].values())
    indices = np.argsort(-art["W"], axis=0)[:500]
    edges = list(art["edge_keys"])
    tables, summary = {}, []
    for topic in range(art["W"].shape[1]):
        targets = sorted({int(edges[i][1]) for i in indices[:, topic]})
        genes = [art["tg_idx2name"][i] for i in targets]
        table = background_ora(genes, background, library)
        table.to_csv(output / f"T{topic}_local_go.tsv", sep="\t", index=False)
        tables[topic] = table
        summary.append(
            {
                "topic": f"T{topic}",
                "n_query": len(genes),
                "n_fdr05": int((table["Adjusted P-value"] < 0.05).sum()),
            }
        )
    pd.DataFrame(summary).to_csv(output / "summary.tsv", sep="\t", index=False)
    print(
        f"Local GO: {len(tables)} topics, {len(background)} network targets; no uploads."
    )
    return tables


class NotebookRenderer:
    def __init__(self, output):
        import matplotlib.pyplot as plt

        plt.switch_backend("Agg")
        self.output = output
        self.cells, self.figures = [], []
        self.namespace = {"__name__": "__main__"}
        self.plt = plt
        self.original_show = plt.show
        self.active_outputs = []
        self.section, self.cell_index = "setup", 0

    def markdown(self, text):
        self.cells.append(
            {
                "cell_type": "markdown",
                "id": f"cell-{len(self.cells):03d}",
                "metadata": {},
                "source": text.splitlines(keepends=True),
            }
        )

    def capture(self, *_args, **_kwargs):
        for number in self.plt.get_fignums():
            fig = self.plt.figure(number)
            index = len(self.figures) + 1
            stem = f"{index:02d}_{self.section}_cell{self.cell_index}"
            image_buffer = io.BytesIO()
            fig.savefig(
                image_buffer,
                format="png",
                dpi=150,
                bbox_inches="tight",
                facecolor="white",
            )
            image_bytes = image_buffer.getvalue()
            (self.output / f"{stem}.png").write_bytes(image_bytes)
            svg_path = self.output / f"{stem}.svg"
            fig.savefig(
                svg_path,
                bbox_inches="tight",
                metadata={"Date": None},
            )
            svg_path.write_text(
                "\n".join(
                    line.rstrip()
                    for line in svg_path.read_text(encoding="utf-8").splitlines()
                )
                + "\n",
                encoding="utf-8",
            )
            titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
            self.figures.append(
                {
                    "index": index,
                    "section": self.section,
                    "cell": self.cell_index,
                    "stem": stem,
                    "titles": titles,
                    "axes": len(fig.axes),
                }
            )
            self.active_outputs.append(
                {
                    "output_type": "display_data",
                    "metadata": {},
                    "data": {
                        "image/png": base64.b64encode(image_bytes).decode(),
                        "text/plain": f"<Replotted figure {index}: {self.section}>",
                    },
                }
            )
            self.plt.close(fig)

    def code(self, source, *, metadata=None):
        self.cell_index += 1
        self.active_outputs = []
        stdout, stderr = io.StringIO(), io.StringIO()
        self.plt.show = self.capture
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exec(  # noqa: S102
                    compile(source, f"replot_cell_{self.cell_index}", "exec"),
                    self.namespace,
                )
        finally:
            self.plt.show = self.original_show
        for name, stream in [("stdout", stdout), ("stderr", stderr)]:
            if stream.getvalue():
                self.active_outputs.append(
                    {
                        "output_type": "stream",
                        "name": name,
                        "text": stream.getvalue().splitlines(keepends=True),
                    }
                )
        self.cells.append(
            {
                "cell_type": "code",
                "id": f"cell-{len(self.cells):03d}",
                "metadata": metadata or {},
                "source": source.splitlines(keepends=True),
                "execution_count": self.cell_index,
                "outputs": self.active_outputs,
            }
        )
        print(
            f"Rendered {self.section} cell {self.cell_index}; total figures: {len(self.figures)}",
            flush=True,
        )

    def original(self, source, path, index, sha, *, paths=None, select=None):
        tree = ast.parse(source)
        if select is not None:
            tree.body = select(tree.body)
        tree = PlotCompatibility(paths or {}).visit(tree)
        ast.fix_missing_locations(tree)
        self.code(
            ast.unparse(tree) + "\n",
            metadata={
                "source_cell": {
                    "notebook": str(path),
                    "cell_zero_based": index,
                    "source_sha256": sha,
                }
            },
        )

    def save(self):
        notebook = {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {
                    "name": "python",
                    "version": platform.python_version(),
                    "file_extension": ".py",
                },
            },
            "cells": self.cells,
        }
        (self.output / "mouse_notebooks.ipynb").write_text(
            json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
        )
        (self.output / "figures.json").write_text(
            json.dumps(self.figures, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def replot_model(args, renderer):
    path = args.source_root / "model.ipynb"
    sources = [cell_source(path, i, sha) for i, sha in enumerate(MODEL_HASHES)]
    renderer.section = "model"
    renderer.markdown(
        "## DRAFT checkpoint distance plots\n\nOriginal distance function and plotting blocks; saved scores only, no model training or forward inference."
    )
    renderer.code(
        "from typing import List, Tuple\nimport numpy as np\nimport pandas as pd\nimport torch\nimport matplotlib.pyplot as plt\nplt.rcdefaults()\nplt.rcParams['svg.fonttype'] = 'none'\n"
    )
    renderer.original(
        sources[0],
        path,
        0,
        MODEL_HASHES[0],
        select=lambda body: [
            n
            for n in body
            if isinstance(n, ast.FunctionDef) and n.name == "compute_smoothness_matrix"
        ],
    )
    renderer.code(
        f"checkpoint = torch.load({str(args.checkpoint.resolve())!r}, map_location='cpu', weights_only=False)\nmeta = checkpoint['meta']\nsims_best = [np.asarray(s) for s in checkpoint['sims_best']]\ntopo = list(meta['topo'])\nparent_cid = meta['parent_cid']\ncid2tid = {{c: i for i, c in enumerate(topo)}}\ncluster_labels = [f'C{{c}}' for c in topo]\nCMAP = 'OrRd'\n"
    )
    tree = ast.parse(sources[1])
    start = next(
        i
        for i, n in enumerate(tree.body)
        if isinstance(n, ast.FunctionDef) and n.name == "plot_heatmap_with_labels"
    )
    renderer.original(
        sources[1], path, 1, MODEL_HASHES[1], select=lambda body: body[start:]
    )
    renderer.original(sources[2], path, 2, MODEL_HASHES[2])
    renderer.code(
        f"pd.DataFrame(diffM_all, index=cluster_labels, columns=cluster_labels).to_csv({str((args.work_dir / 'model_distance.tsv').resolve())!r}, sep='\\t')\nprint(f'Overall smoothness: {{smoothness_all:.9f}}')\ndel checkpoint, sims_best\n"
    )


def replot_edge(args, renderer):
    path = args.source_root / "edge_cluster.ipynb"
    output = args.work_dir / "edge"
    renderer.section = "edge"
    renderer.markdown(
        "## Edge clustering and TF panels\n\nOriginal cells 0–5 in order; 12 modules, seed 5465 and original cell-cluster order. Only output paths and the removed Matplotlib colormap API are adapted."
    )
    renderer.code("plt.rcdefaults()\nplt.rcParams['svg.fonttype'] = 'none'\n")
    for i, sha in enumerate(EDGE_HASHES):
        renderer.original(
            cell_source(path, i, sha),
            path,
            i,
            sha,
            paths={
                "NETWORK_DIR": str((args.run_dir / "network").resolve()),
                "OUTPUT_DIR": str(output.resolve()),
            },
        )
