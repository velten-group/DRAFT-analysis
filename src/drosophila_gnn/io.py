"""Input path helpers for the Drosophila lineage workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any


@dataclass(frozen=True)
class InputPaths:
    base_dir: Path
    tf_list: Path
    tg_list: Path
    re_list: Path
    rna_h5ad: Path
    atac_h5ad: Path
    re_tg_prior: Path
    tf_re_prior: Path
    annotation: Path

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "InputPaths":
        data = config.get("data", {})
        base_dir = Path(config.get("base_dir", ".")).expanduser().resolve()

        def _resolve(key: str, default: str) -> Path:
            target = (base_dir / data.get(key, default)).resolve()
            if target.exists():
                return target
            copy = resolve_copy_suffix_path(target)
            return copy.resolve() if copy is not None else target

        return cls(
            base_dir=base_dir,
            tf_list=_resolve("tf_list", "TF.tsv"),
            tg_list=_resolve("tg_list", "TG.tsv"),
            re_list=_resolve("re_list", "RE.txt"),
            rna_h5ad=_resolve("rna_h5ad", "A_rna_focus.h5ad"),
            atac_h5ad=_resolve("atac_h5ad", "B_atac_focus.h5ad"),
            re_tg_prior=_resolve("re_tg_prior", "RE_TG_100kb.tsv"),
            tf_re_prior=_resolve("tf_re_prior", "TF_RE_top3pct_edges.tsv"),
            annotation=_resolve("annotation", "cell_annotation.csv"),
        )


def strip_copy_suffix(value: str) -> str:
    return re.sub(r"\s*\(\d+\)$", "", str(value).strip())


def resolve_copy_suffix_path(path: str | Path) -> Path | None:
    target = Path(path).expanduser()
    parent = target.parent
    if not parent.exists():
        return None

    suffix = target.suffix
    stem = strip_copy_suffix(target.stem)
    pattern = re.compile(rf"^{re.escape(stem)}(?: ?\(\d+\))?{re.escape(suffix)}$")

    for candidate in sorted(parent.iterdir(), key=lambda item: item.name):
        if candidate.is_file() and pattern.match(candidate.name):
            return candidate.resolve()
    return None


def save_text_report(path: str | Path, content: str) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return output_path


def summarize_input_paths(paths: InputPaths) -> dict[str, str]:
    return {
        "base_dir": str(paths.base_dir),
        "tf_list": str(paths.tf_list),
        "tg_list": str(paths.tg_list),
        "re_list": str(paths.re_list),
        "rna_h5ad": str(paths.rna_h5ad),
        "atac_h5ad": str(paths.atac_h5ad),
        "re_tg_prior": str(paths.re_tg_prior),
        "tf_re_prior": str(paths.tf_re_prior),
        "annotation": str(paths.annotation),
    }
