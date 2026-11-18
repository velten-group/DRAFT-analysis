"""Source Data: panel tables -> one workbook per figure, and comparisons with the frozen tables.

A panel table is a TSV with the values drawn in one figure panel (or the data behind them). ``source_data/sheets_*.json``
lists, per workbook sheet, the title, the notes and the tables written to it, in order. The same format is used for the
frozen tables in ``source_data/`` and for the tables the pipeline writes to ``results/panels/``, so any regenerated table
can be compared with the published one and built into a workbook in the same way.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths

KINDS = ("main", "supplementary")  # figures: one Source Data workbook per figure
TABLES = "tables"  # recomputed cells of the Supplementary Tables (sheets_tables.json; no workbooks)
# Fig. 5b and 5d-f are experimental measurements; their sheets are copied unchanged from the authors' workbook.
EXPERIMENTAL_XLSX = paths.SOURCE_DATA / "experimental" / "Source_Data_Figure_5_experimental.xlsx"
EXPERIMENTAL = [  # (sheet in the authors' workbook, sheet name, title, notes, inserted after this sheet)
    ("GSC_number", "Fig. 5b", "Fig. 5b | Number of GSCs per testis",
     "GSCs per testis scored by 3D segmentation as Vasa-positive germ cells in contact with the Fas3-positive hub; "
     "control = nos-GAL4::VP16 > UAS-mCherry RNAi, Mondo RNAi = nos-GAL4::VP16 > UAS-Mondo RNAi. One row per testis "
     "(n = 11 control and 14 Mondo RNAi testes, 2 independent experiments).", "Fig. 5a"),
    ("smFISH_measurements", "Fig. 5d-f", "Fig. 5d-f | smFISH intensity per GSC (Mondo, kuz, mamo)",
     "One row per segmented GSC: image, GSC area and smFISH mean, median and integrated intensity "
     "(Mondo: d; kuz: e; mamo: f).", "Fig. 5c"),
    ("Statistics", "Fig. 5 statistics", "Fig. 5b,d-f | Group sizes, means and two-sided Mann-Whitney U tests",
     "As reported in the figure legend.", "Fig. 5d-f"),
]


def sheets(kind: str) -> list[dict]:
    assert kind in (*KINDS, TABLES), kind
    return json.loads((paths.SOURCE_DATA / f"sheets_{kind}.json").read_text(encoding="utf-8"))


def frozen_dir(kind: str) -> Path:
    return paths.SOURCE_DATA / kind


def read_table(path: Path) -> pd.DataFrame:
    # round_trip: parse every number exactly as written (the default parser can be off by one unit in the last place)
    return pd.read_csv(path, sep="\t", keep_default_na=True, float_precision="round_trip")


def tables(kind: str) -> list[str]:
    """All table file names of one kind, in workbook order."""
    return [b["tsv"] for s in sheets(kind) for b in s["blocks"]]


def table_name(kind: str, figure: int, sheet: str, block: int) -> str:
    """File name of a table, e.g. table_name('main', 4, 'Fig. 4c', 0) -> 'fig4_Fig_4c_0.tsv'."""
    for s in sheets(kind):
        if s["figure"] == figure and s["sheet"] == sheet:
            return s["blocks"][block]["tsv"]
    raise KeyError((kind, figure, sheet))


# --------------------------------------------------------------------------------------------- comparison
@dataclass
class TableDiff:
    name: str
    ok: bool
    max_abs_diff: float = 0.0
    problems: list[str] = field(default_factory=list)


_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _text_close(a: str, b: str, tol: float) -> float | None:
    """Largest difference between the numbers written in two texts with the same wording, or None."""
    if _NUMBER.sub("#", a) != _NUMBER.sub("#", b):
        return None
    x, y = [float(v) for v in _NUMBER.findall(a)], [float(v) for v in _NUMBER.findall(b)]
    diff = max((abs(p - q) for p, q in zip(x, y)), default=0.0)
    return diff if diff <= tol + 1e-12 else None


def compare_table(new: pd.DataFrame, ref: pd.DataFrame, name: str, atol: float = 1e-9, rtol: float = 1e-9,
                  text_atol: float = 0.0) -> TableDiff:
    """Same columns, same number of rows, numbers equal within atol + rtol*|ref|, all other cells identical (or, with
    text_atol > 0, the same wording with every written number within text_atol)."""
    d = TableDiff(name, True)
    if list(new.columns) != list(ref.columns):
        d.ok = False
        d.problems.append(f"columns differ: {list(new.columns)} vs {list(ref.columns)}")
        return d
    if len(new) != len(ref):
        d.ok = False
        d.problems.append(f"{len(new)} rows vs {len(ref)}")
        return d
    for c in ref.columns:
        a, b = new[c], ref[c]
        num_a, num_b = pd.api.types.is_numeric_dtype(a), pd.api.types.is_numeric_dtype(b)
        if num_a and num_b:
            x, y = a.to_numpy(float), b.to_numpy(float)
            nan_mismatch = np.isnan(x) != np.isnan(y)
            if nan_mismatch.any():
                d.ok = False
                d.problems.append(f"{c}: missing values differ in {int(nan_mismatch.sum())} rows")
            m = ~(np.isnan(x) | np.isnan(y))
            if m.any():
                err = np.abs(x[m] - y[m])
                d.max_abs_diff = max(d.max_abs_diff, float(err.max()))
                bad = err > atol + rtol * np.abs(y[m])
                if bad.any():
                    d.ok = False
                    d.problems.append(f"{c}: {int(bad.sum())} values differ (max {float(err.max()):.3g})")
        else:
            sa, sb = a.astype(str).where(a.notna(), ""), b.astype(str).where(b.notna(), "")
            bad = sa.to_numpy() != sb.to_numpy()
            if bad.any() and text_atol > 0:
                for i in np.flatnonzero(bad):
                    diff = _text_close(sa.iloc[i], sb.iloc[i], text_atol)
                    if diff is not None:
                        bad[i] = False
                        d.max_abs_diff = max(d.max_abs_diff, diff)
            if bad.any():
                d.ok = False
                i = int(np.flatnonzero(bad)[0])
                d.problems.append(f"{c}: {int(bad.sum())} cells differ, first row {i}: {sa.iloc[i]!r} vs {sb.iloc[i]!r}")
    return d


def compare_dir(kind: str, new_dir: Path, names: list[str] | None = None, atol: float = 1e-9,
                rtol: float = 1e-9, text_atol: float = 0.0) -> list[TableDiff]:
    """Compare regenerated tables in new_dir with the frozen ones. Tables not (yet) regenerated are skipped."""
    out = []
    for name in names or tables(kind):
        f = new_dir / name
        if f.exists():
            out.append(compare_table(read_table(f), read_table(frozen_dir(kind) / name), name, atol, rtol, text_atol))
    return out


# --------------------------------------------------------------------------------------------- workbooks
def _value(v):
    if isinstance(v, float) and math.isnan(v):
        return None
    return v.item() if hasattr(v, "item") else v


def build_workbooks(kind: str, panels_dir: Path, out_dir: Path) -> list[Path]:
    """Write Source_Data_Figure_<n>.xlsx (main) or Source_Data_Supplementary_Figure_<nn>.xlsx, one sheet per panel;
    for the tables, Supplementary_Table_<nn>.xlsx with the recomputed columns of each published table."""
    import openpyxl
    from openpyxl.styles import Alignment, Border, Font, Side

    bold, title_font, sub_font = Font(bold=True), Font(bold=True, size=12), Font(bold=True, italic=True)
    line = Border(bottom=Side(style="thin"))
    supp, table = kind == "supplementary", kind == TABLES
    out_dir.mkdir(parents=True, exist_ok=True)

    def header(ws, title, notes):
        ws.cell(1, 1, title).font = title_font
        ws.cell(2, 1, notes).alignment = Alignment(wrap_text=False, vertical="top")
        return 4

    def widths(ws):
        for col in ws.iter_cols(min_row=4):
            size = max((len(str(c.value)) for c in col[:300] if c.value is not None), default=8)
            ws.column_dimensions[col[0].column_letter].width = min(max(size + 2, 9), 45)

    def write_sheet(wb, spec):
        ws = wb.create_sheet(spec["sheet"])
        row = header(ws, spec["title"], spec["notes"])
        for block in spec["blocks"]:
            df = read_table(panels_dir / block["tsv"])
            assert len(df) == block["rows"], (block, len(df))
            if block["subtitle"]:
                ws.cell(row, 1, block["subtitle"]).font = sub_font
                row += 1
            for j, name in enumerate(df.columns, 1):
                c = ws.cell(row, j, str(name))
                c.font, c.border = bold, line
            row += 1
            for rec in df.itertuples(index=False):
                for j, v in enumerate(rec, 1):
                    ws.cell(row, j, _value(v))
                row += 1
            row += 1
        widths(ws)

    def copy_experimental(wb, src, name, title, notes):
        ws = wb.create_sheet(name)
        start = header(ws, title, notes)
        for r, values in enumerate(src.iter_rows(values_only=True), start):
            for j, v in enumerate(values, 1):
                if v is not None:
                    ws.cell(r, j, v)
        for j in range(1, src.max_column + 1):
            ws.cell(start, j).font = bold
        widths(ws)
        return start

    specs_all = sheets(kind)
    written = []
    for fig in sorted({s["figure"] for s in specs_all}):
        wb = openpyxl.Workbook()
        contents = wb.active
        contents.title = "Contents"
        specs = sorted((s for s in specs_all if s["figure"] == fig), key=lambda s: s["sheet"])
        order = [(s["sheet"], s["title"]) for s in specs]
        copies = []
        if fig == 5 and kind == "main":
            orig = openpyxl.load_workbook(EXPERIMENTAL_XLSX, data_only=False)
            assert orig.sheetnames == [e[0] for e in EXPERIMENTAL], orig.sheetnames
            for _, name, title, _, after in EXPERIMENTAL:
                order.insert([o[0] for o in order].index(after) + 1, (name, title))
        for name, _ in order:
            spec = next((s for s in specs if s["sheet"] == name), None)
            if spec:
                write_sheet(wb, spec)
            else:
                src_name, _, title, notes, _ = next(e for e in EXPERIMENTAL if e[1] == name)
                copies.append((orig[src_name], name, copy_experimental(wb, orig[src_name], name, title, notes)))
        if table:
            contents.cell(1, 1, f"Supplementary Table {fig}").font = title_font
            contents.cell(2, 1, "Columns of the published table that the checkpoint line recomputes.")
        else:
            contents.cell(1, 1, f"Source Data for {'Supplementary ' if supp else ''}Figure {fig}").font = title_font
            contents.cell(2, 1, "One sheet per panel; each sheet starts with a short description. Schematic and image "
                                "panels have no source data.")
        contents.cell(4, 1, "Sheet").font = bold
        contents.cell(4, 2, "Content").font = bold
        for i, (name, title) in enumerate(order, 5):
            contents.cell(i, 1, name)
            contents.cell(i, 2, title)
        contents.column_dimensions["A"].width = 18
        contents.column_dimensions["B"].width = 110
        path = out_dir / (f"Supplementary_Table_{fig:02d}.xlsx" if table else
                          f"Source_Data_Supplementary_Figure_{fig:02d}.xlsx" if supp else f"Source_Data_Figure_{fig}.xlsx")
        wb.save(path)
        back = openpyxl.load_workbook(path)
        assert back.sheetnames == ["Contents"] + [o[0] for o in order], back.sheetnames
        for src, name, start in copies:  # experimental sheets: identical cell values
            for r, values in enumerate(src.iter_rows(values_only=True), start):
                for j, v in enumerate(values, 1):
                    assert back[name].cell(r, j).value == v, (name, r, j)
        written.append(path)
    return written


def compare_workbooks(a: Path, b: Path) -> list[str]:
    """Cell-by-cell comparison of two workbooks (values only). Returns a list of differences (empty = identical)."""
    import openpyxl

    wa, wb_ = openpyxl.load_workbook(a, data_only=True), openpyxl.load_workbook(b, data_only=True)
    if wa.sheetnames != wb_.sheetnames:
        return [f"sheets differ: {wa.sheetnames} vs {wb_.sheetnames}"]
    diffs = []
    for name in wa.sheetnames:
        ra = list(wa[name].iter_rows(values_only=True))
        rb = list(wb_[name].iter_rows(values_only=True))
        if len(ra) != len(rb):
            diffs.append(f"{name}: {len(ra)} rows vs {len(rb)}")
            continue
        for i, (x, y) in enumerate(zip(ra, rb), 1):
            if tuple(x) != tuple(y):
                diffs.append(f"{name}!row {i} differs")
                break
    return diffs
