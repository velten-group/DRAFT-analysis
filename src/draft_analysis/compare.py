"""Compare the panel tables of the checkpoint line with the published Source Data tables.

Default tolerance: |difference| <= 1e-9 + 1e-9 * |published| for numbers, exact equality for everything else. The
tables below allow for float32 rounding that differs between machines, for near-tied ranks that this rounding can
swap, and for tied GO scores at the top-10 cut of Supp. Fig. 6; the reason is written into the report. ``text_atol``
applies to numbers written inside text cells (e.g. "0.28 (-0.24-0.78)"): same wording, every number within text_atol.
"""
from __future__ import annotations

from pathlib import Path

from . import paths, source_data

DEFAULT = {"atol": 1e-9, "rtol": 1e-9, "text_atol": 0.0, "reason": "default"}
_FLOAT32 = "float32 rounding differs between machines"
_TIE = "a near-tied pair can swap order between machines"
_BOUNDARY = _FLOAT32 + "; one edge at the top-k boundary can change"
TOLERANCE = {
    **{f"fig3_Fig_3c_{i}.tsv": {"atol": 1e-4, "rtol": 1e-3, "reason": _FLOAT32 + " (network distances 50-160)"}
       for i in range(3)},
    "fig4_Fig_4b_0.tsv": {"atol": 1e-5, "reason": _FLOAT32 + " (DRAFT RE-TG scores)"},
    **{f"fig4_Fig_4c_{i}.tsv": {"atol": 1e-4, "reason": _FLOAT32} for i in range(5)},
    "fig4_Fig_4d_0.tsv": {"atol": 1e-4, "reason": _FLOAT32},
    "fig5_Fig_5a_0.tsv": {"rtol": 1e-6, "reason": _FLOAT32},
    "fig5_Fig_5c_0.tsv": {"atol": 1e-7, "reason": _FLOAT32},
    "fig6_Fig_6c_0.tsv": {"atol": 0.01, "reason": "AUROC: " + _TIE},
    "fig6_Fig_6d_0.tsv": {"atol": 1e-4, "reason": _FLOAT32 + "; early/late calls identical"},
    "supp06_Supp_Fig_6_0.tsv": {"tied_terms": True, "reason": "top 10 GO terms: another term with the same tied "
                                "adjusted p-value may be shown; scores and untied rows identical"},
    "supp10_Supp_Fig_10_0.tsv": {"atol": 1e-4, "reason": _BOUNDARY},
    "supp10_Supp_Fig_10_1.tsv": {"rtol": 1e-3, "reason": _BOUNDARY},
    "supp13_Supp_Fig_13_0.tsv": {"text_atol": 0.01, "reason": "Spearman range: " + _TIE},
    "supp13_Supp_Fig_13_2.tsv": {"text_atol": 0.01, "reason": "Spearman range: " + _TIE},
    "supp14_Supp_Fig_14_0.tsv": {"atol": 0.01, "reason": "AUROC: " + _TIE},
}


def tolerance(name: str) -> dict:
    return {**DEFAULT, **TOLERANCE.get(name, {})}


def go_score(adjusted_p):
    """-log10 adjusted p, the score by which the topic notebook ranks GO terms."""
    import numpy as np

    return -np.log10(np.maximum(np.asarray(adjusted_p, dtype=float), 1e-300))


def tied_go_terms(new, ref, full: dict | None = None) -> list[str]:
    """Supp. Fig. 6 (top 10 GO terms per topic): the scores must be identical per topic and every row whose score is
    unique in the topic's full GO table identical; a row whose score is tied may show another term of that score.
    full: topic -> full GO table (default: data/mouse/frozen_numbering/topic10/local_go)."""
    import numpy as np
    import pandas as pd

    if list(new.columns) != list(ref.columns) or len(new) != len(ref):
        return [f"shape or columns differ: {new.shape} vs {ref.shape}"]
    problems = []
    for topic in ref.topic.unique():
        a, b = new[new.topic == topic].reset_index(drop=True), ref[ref.topic == topic].reset_index(drop=True)
        if len(a) != len(b) or not np.allclose(a.score, b.score, rtol=1e-9, atol=1e-12):
            problems.append(f"{topic}: scores differ")
            continue
        if not a.panel_title.equals(b.panel_title):
            problems.append(f"{topic}: panel title differs")
        table = (full or {}).get(topic)
        if table is None:
            table = pd.read_csv(paths.data("mouse", "frozen_numbering", "topic10", "local_go", f"{topic}_local_go.tsv"),
                                sep="\t")
        score = pd.Series(go_score(table["Adjusted P-value"]), index=table.Term).round(12)
        counts = score.value_counts()
        for i in range(len(b)):
            s = round(float(b.score[i]), 12)
            if counts.get(s, 0) > 1:
                if a.Term[i] not in score.index or score[a.Term[i]] != s:
                    problems.append(f"{topic} row {i}: {a.Term[i]!r} does not have the tied score")
            else:
                d = source_data.compare_table(a.iloc[[i]].reset_index(drop=True), b.iloc[[i]].reset_index(drop=True),
                                              "row")
                if not d.ok:
                    problems.append(f"{topic} row {i} (untied): " + "; ".join(d.problems))
    return problems


def run(kind: str = "main", folder: Path | None = None, verbose: bool = True) -> list[source_data.TableDiff]:
    folder = folder or paths.PANELS[kind]
    results = []
    for name in source_data.tables(kind):
        if not (folder / name).exists():
            continue
        t = tolerance(name)
        found = source_data.compare_dir(kind, folder, [name], atol=t["atol"], rtol=t["rtol"], text_atol=t["text_atol"])
        if t.get("tied_terms") and not found[0].ok:  # exact match not reached: accept other terms of tied scores
            found[0].problems = tied_go_terms(source_data.read_table(folder / name),
                                              source_data.read_table(source_data.frozen_dir(kind) / name))
            found[0].ok = not found[0].problems
        results.extend(found)
    if verbose:
        missing = [n for n in source_data.tables(kind) if not (folder / n).exists()]
        for d in results:
            note = f" (tolerance: {TOLERANCE[d.name]['reason']})" if d.name in TOLERANCE else ""
            print(f"{'OK  ' if d.ok else 'FAIL'} {d.name}  max|diff| {d.max_abs_diff:.2g}{note}"
                  + ("" if d.ok else "  " + "; ".join(d.problems)))
        if kind == source_data.TABLES:
            print(f"Supplementary Tables: {sum(d.ok for d in results)}/{len(results)} recomputed tables match the "
                  f"published values" + (f"; {len(missing)} not built" if missing else ""))
        else:
            print(f"{kind}: {sum(d.ok for d in results)}/{len(results)} tables computed by the checkpoint line match "
                  f"the Source Data" + (f"; the other {len(missing)} of {len(source_data.tables(kind))} are in the "
                                        f"quick line only" if missing else ""))
    return results


def write_report(results: list[source_data.TableDiff], out: Path) -> Path:
    """One row per compared table: match, largest absolute difference, tolerance used and why."""
    import pandas as pd

    rows = []
    for d in results:
        t = tolerance(d.name)
        rows.append({"table": d.name, "match": d.ok, "max_abs_diff": d.max_abs_diff, "atol": t["atol"],
                     "rtol": t["rtol"], "text_atol": t["text_atol"], "tolerance_reason": t["reason"],
                     "problems": "; ".join(d.problems)})
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, sep="\t", index=False, lineterminator="\n")
    print(out)
    return out
