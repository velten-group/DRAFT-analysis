"""Plot saved TF peak-offset counts, including ablations, without inference."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import uuid
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

PROJECT = Path(__file__).resolve().parents[2]
MODELS = [
    "DRAFT",
    "DRAFT w/o GRU",
    "Linear + GRU",
    "SCENIC+",
    "scDoRI",
    "Dictys",
    "separate GAT",
]
COLORS = {
    "DRAFT": "#E5A000",
    "DRAFT w/o GRU": "#88CCEE",
    "separate GAT": "#88CCEE",
    "Linear + GRU": "#228833",
    "SCENIC+": "#4477AA",
    "scDoRI": "#333333",
    "Dictys": "#009E73",
}
SCOPES = [
    "method_evaluable",
    "common_evaluable",
    "method_interior_peaks",
    "common_interior_peaks",
]


def model_prefix(models):
    return {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}.get(
        len(models), str(len(models))
    )


def read_counts(path, profile="smoothed"):
    raw = path.read_text(encoding="utf-8-sig")
    lines = raw.splitlines()
    header = next(
        (
            line
            for line in lines
            if line.strip().startswith("profile") and "n_far_conservative" in line
        ),
        None,
    )
    if header is None:
        raise ValueError("No peak-offset count table found.")
    body = [
        line
        for line in lines
        if line.split() and line.split()[0] in ["raw", "smoothed"]
    ]
    table = pd.read_csv(
        io.StringIO("\n".join([header, *body])), sep="\t" if "\t" in header else r"\s+"
    )
    table = table.loc[table.profile.eq(profile)].copy()
    required = [
        "n_tf_list",
        "n_evaluable",
        "n_far_conservative",
        "offset_threshold_stages",
    ]
    if table.empty or not set(required + ["model", "scope"]).issubset(table):
        raise ValueError(f"Missing {profile} count rows or required columns.")
    for column in required:
        table[column] = pd.to_numeric(table[column], errors="raise")
        if not np.isfinite(table[column]).all() or (table[column] < 0).any():
            raise ValueError(f"Invalid count/threshold: {column}.")
        if column != "offset_threshold_stages" and not (table[column] % 1 == 0).all():
            raise ValueError(f"Noninteger count: {column}.")
    if table.duplicated(["scope", "model", "offset_threshold_stages"]).any():
        raise ValueError("Repeated model/scope/threshold rows.")
    if (table.n_far_conservative > table.n_evaluable).any() or (
        table.n_evaluable > table.n_tf_list
    ).any():
        raise ValueError("Invalid numerator/denominator ordering.")
    if table.n_tf_list.nunique() != 1:
        raise ValueError("TF-list denominators differ.")
    if not table.groupby(["scope", "model"]).n_evaluable.nunique().eq(1).all():
        raise ValueError("Evaluable denominators change across thresholds.")
    for scope in ["common_evaluable", "common_interior_peaks"]:
        if table.loc[table.scope.eq(scope), "n_evaluable"].nunique() > 1:
            raise ValueError(f"Shared-cohort denominators differ: {scope}.")
    # Recalculate from integer counts rather than rounded terminal fractions.
    table["percentage"] = (
        100 * table.n_far_conservative / table.n_evaluable.replace(0, np.nan)
    )
    return table


def select_counts(value, profile, models=None):
    if value != Path("latest"):
        return value.expanduser().resolve()
    candidates = []
    paths = list(
        (PROJECT / "result/germ/tf_peak_offsets").glob("seed2025_*/offset_counts.tsv")
    )
    paths.extend(
        (PROJECT / "comparisons/tf_peak_offsets/runs").glob(
            "candidate_table_s2025_*/full_trajectory/offset_counts.tsv"
        )
    )
    for path in paths:
        report_path = path.parent / "analysis_report.json"
        if not report_path.is_file():
            report_path = path.parent.parent / "table_report.json"
        if not report_path.is_file():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("validation_only", False) or any(
            item.get("validation_only", False)
            for item in report.get("sources", {}).values()
        ):
            continue
        present = set(read_counts(path, profile).model)
        if (
            present == set(models)
            if models
            else len(present) >= 3 and present.issubset(MODELS)
        ):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(
            "No completed count table with the requested methods. Pass --counts /EXACT/offset_counts.tsv."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime_ns)


def label_bar(ax, bar, numerator, denominator):
    label = f"{numerator:g}/{denominator:g}" if denominator else "NA"
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1.3,
        label,
        ha="center",
        va="bottom",
        fontsize=9,
    )


def grouped_panel(ax, table, scope, title, models):
    data = table.loc[table.scope.eq(scope)]
    if scope.startswith("common_") and data.n_evaluable.eq(0).all():
        ax.text(
            0.5,
            0.5,
            "No TFs evaluable in every included method",
            transform=ax.transAxes,
            ha="center",
            fontsize=10,
        )
        ax.set(title=title, ylabel="TFs with separated peak regions (%)", ylim=(0, 112))
        ax.set_xticks([])
        ax.spines[["top", "right"]].set_visible(False)
        return
    thresholds = sorted(data.offset_threshold_stages.unique())
    width = min(0.25, 0.8 / len(models))
    for i, model in enumerate(models):
        color = COLORS[model]
        part = (
            data.loc[data.model.eq(model)]
            .set_index("offset_threshold_stages")
            .reindex(thresholds)
        )
        if part.n_evaluable.isna().any():
            raise ValueError(f"Incomplete thresholds: {scope}, {model}.")
        x = np.arange(len(thresholds)) + (i - (len(models) - 1) / 2) * width
        bars = ax.bar(
            x, part.percentage.fillna(0), width=width, color=color, label=model
        )
        for bar, row in zip(bars, part.itertuples()):
            label_bar(ax, bar, row.n_far_conservative, row.n_evaluable)
    ax.set_xticks(
        range(len(thresholds)), [f"≥{value:g} stages" for value in thresholds]
    )
    ax.set(title=title, ylabel="TFs with separated peak regions (%)", ylim=(0, 112))
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.spines[["top", "right"]].set_visible(False)


def render_absolute_counts(table, models, out):
    """Show the absolute number of candidate TFs, without denominator scaling."""
    data = table.loc[table.scope.eq("method_evaluable")].copy()
    thresholds = sorted(data.offset_threshold_stages.unique())
    if not thresholds:
        raise ValueError("No individual-method candidate counts to plot.")
    maximum = float(data.n_far_conservative.max())
    pad = max(0.4, maximum * 0.025)
    width = min(0.25, 0.8 / len(models))
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    ):
        fig, ax = plt.subplots(
            figsize=(11.5 if len(models) >= 5 else 10, 5.6), constrained_layout=True
        )
        for i, model in enumerate(models):
            part = (
                data.loc[data.model.eq(model)]
                .set_index("offset_threshold_stages")
                .reindex(thresholds)
            )
            if part.n_far_conservative.isna().any():
                raise ValueError(f"Incomplete absolute counts: {model}.")
            x = np.arange(len(thresholds)) + (i - (len(models) - 1) / 2) * width
            bars = ax.bar(
                x,
                part.n_far_conservative,
                width=width,
                color=COLORS[model],
                label=model,
            )
            for bar, count in zip(bars, part.n_far_conservative):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + pad,
                    f"{int(count)}",
                    ha="center",
                    va="bottom",
                    fontsize=11,
                )
        ax.set_xticks(
            range(len(thresholds)), [f"≥{value:g} stages" for value in thresholds]
        )
        ax.set(
            ylabel="Number of TFs with separated peak regions",
            title="TF candidates with network–expression peak separation",
            ylim=(0, max(5, maximum * 1.3 + pad)),
        )
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.spines[["top", "right"]].set_visible(False)
        validation_only = (
            "validation_only" in table and table.validation_only.eq(True).any()
        )
        ax.legend(
            frameon=False,
            loc="upper right",
            ncol=3 if len(models) >= 5 else 2,
            title="SYNTHETIC SOFTWARE TEST" if validation_only else None,
        )
        if (
            "time_domain" in table
            and table.time_domain.eq("dictys_shared_coverage").all()
        ):
            fig.supxlabel(
                f"Matched stage coverage {int(table.stage_min.iloc[0])}–{int(table.stage_max.iloc[0])}",
                fontsize=10,
            )
        prefix = model_prefix(models)
        for extension in ["png", "pdf", "svg"]:
            fig.savefig(
                out / f"{prefix}_method_peak_offsets_absolute.{extension}", dpi=300
            )
        plt.close(fig)
    data[["model", "offset_threshold_stages", "n_far_conservative"]].to_csv(
        out / "absolute_candidate_counts.tsv", sep="\t", index=False
    )


def render(path, out, profile="smoothed", threshold=3, models=None):
    table = read_counts(path, profile)
    models = models or [model for model in MODELS if model in set(table.model)]
    if (
        set(table.model) != set(models)
        or not set(SCOPES).issubset(table.scope)
        or not models
    ):
        raise ValueError(
            "All four comparison scopes must contain exactly the requested methods."
        )
    primary = table.loc[table.offset_threshold_stages.eq(threshold)]
    if len(primary) != len(SCOPES) * len(models):
        raise ValueError(
            f"Threshold {threshold:g} is not available in all four scopes."
        )
    out.mkdir(parents=True, exist_ok=False)
    summary = []
    for scope in SCOPES:
        data = primary.loc[primary.scope.eq(scope)].set_index("model").reindex(models)
        summary.append(
            pd.DataFrame(
                {
                    "scope": scope,
                    "model": models,
                    "n_evaluable": data.n_evaluable.to_numpy(),
                    "n_far": data.n_far_conservative.to_numpy(),
                    "percentage": data.percentage.to_numpy(),
                }
            )
        )
    pd.concat(summary, ignore_index=True).to_csv(
        out / "primary_counts.tsv", sep="\t", index=False
    )
    markdown = [
        "| "
        + " | ".join([f"Cohort (separation ≥{threshold:g} stages)", *models])
        + " |",
        "| " + " | ".join(["---"] * (len(models) + 1)) + " |",
    ]
    names = {
        "method_evaluable": "Each method's evaluable TFs",
        "common_evaluable": "Same TFs across all included methods",
        "method_interior_peaks": "Each method, endpoint maxima excluded",
        "common_interior_peaks": "Same TFs, endpoint maxima excluded",
    }
    for scope in SCOPES:
        part = primary.loc[primary.scope.eq(scope)].set_index("model")
        cells = [
            f"{part.loc[model, 'n_far_conservative']:g}/{part.loc[model, 'n_evaluable']:g} ({part.loc[model, 'percentage']:.2f}%)"
            if part.loc[model, "n_evaluable"]
            else "NA (n=0)"
            for model in models
        ]
        markdown.append("| " + " | ".join([names[scope], *cells]) + " |")
    (out / "primary_counts.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    render_absolute_counts(table, models, out)
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    ):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
        validation_only = (
            "validation_only" in table and table.validation_only.eq(True).any()
        )
        ax = axes[0, 0]
        coverage = (
            primary.loc[primary.scope.eq("method_evaluable")]
            .set_index("model")
            .reindex(models)
        )
        bars = ax.bar(
            models,
            100 * coverage.n_evaluable / coverage.n_tf_list,
            color=[COLORS[model] for model in models],
            width=0.6,
        )
        for bar, row in zip(bars, coverage.itertuples()):
            label_bar(ax, bar, row.n_evaluable, row.n_tf_list)
        ax.set(
            title="A  TFs with evaluable expression/network peaks",
            ylabel="Coverage of the TF list (%)",
            ylim=(0, 112),
        )
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.spines[["top", "right"]].set_visible(False)
        grouped_panel(
            axes[0, 1],
            table,
            "method_evaluable",
            "B  Each method's evaluable TFs",
            models,
        )
        grouped_panel(
            axes[1, 0],
            table.loc[table.offset_threshold_stages.eq(threshold)],
            "method_interior_peaks",
            "C  Endpoint maxima excluded",
            models,
        )
        common_n = int(
            primary.loc[primary.scope.eq("common_evaluable"), "n_evaluable"].iloc[0]
        )
        grouped_panel(
            axes[1, 1],
            table,
            "common_evaluable",
            f"D  Shared cohort (n={common_n}; descriptive)",
            models,
        )
        handles, labels = axes[0, 1].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="outside upper center",
            ncol=min(3, len(models)) if len(models) >= 5 else len(models),
            frameon=False,
            title="SYNTHETIC SOFTWARE TEST" if validation_only else None,
        )
        domain_label = ""
        if (
            "time_domain" in table
            and table.time_domain.eq("dictys_shared_coverage").all()
        ):
            if table.stage_min.nunique() != 1 or table.stage_max.nunique() != 1:
                raise ValueError("The plotted methods have different time coverage.")
            domain_label = f"Matched stage coverage {int(table.stage_min.iloc[0])}–{int(table.stage_max.iloc[0])}; "
        fig.supxlabel(
            domain_label
            + "peak separation is measured in ordered stages; labels show counts/denominators",
            fontsize=10,
        )
        for extension in ["png", "pdf", "svg"]:
            prefix = model_prefix(models)
            fig.savefig(out / f"{prefix}_method_peak_offsets.{extension}", dpi=300)
        plt.close(fig)
    (out / "plot_report.json").write_text(
        json.dumps(
            {
                "input": str(path),
                "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "profile": profile,
                "primary_threshold_stages": threshold,
                "models": models,
                "validation_only": bool(validation_only),
                "time_domain": table.time_domain.iloc[0]
                if "time_domain" in table
                else "full_trajectory",
                "stage_min": int(table.stage_min.iloc[0])
                if "stage_min" in table
                else 0,
                "stage_max": int(table.stage_max.iloc[0])
                if "stage_max" in table
                else 14,
                "shared_evaluable_tf_count": common_n,
                "absolute_count_plot": "n_far_conservative among each method's evaluable TFs; integer counts, without normalization by n_evaluable or n_tf_list",
                "interpretation": "Different eligibility/target pools; descriptive temporal separation, not expression-independent causality or prediction accuracy; no uncertainty estimated from aggregate counts.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\n".join(markdown), flush=True)
    print(f"Tables and figures: {out}", flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", type=Path, default=Path("latest"))
    parser.add_argument("--profile", choices=["raw", "smoothed"], default="smoothed")
    parser.add_argument("--models", nargs="+", choices=MODELS)
    parser.add_argument("--threshold", type=float, default=3)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    path = select_counts(args.counts, args.profile, args.models)
    out = (
        (args.output_dir or path.parent / f"count_figures_{uuid.uuid4().hex[:10]}")
        .expanduser()
        .resolve()
    )
    render(path, out, args.profile, args.threshold, args.models)


if __name__ == "__main__":
    main()
