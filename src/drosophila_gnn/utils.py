"""Small shared helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - dependency is declared in pyproject
    yaml = None


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def config_base_dir(config: dict[str, Any]) -> Path:
    return Path(str(config.get("base_dir", "."))).expanduser().resolve()


def resolve_config_path(config_or_base: dict[str, Any] | str | Path, path: str | Path) -> Path:
    if isinstance(config_or_base, dict):
        base_dir = config_base_dir(config_or_base)
    else:
        base_dir = Path(config_or_base).expanduser().resolve()
    target = Path(path).expanduser()
    if not target.is_absolute():
        target = base_dir / target
    return target.resolve()


def ensure_directory(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to load config files.")
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Config at {path} must be a mapping.")
    return data


def emit_progress(status: str, task: str, detail: str | None = None) -> None:
    line = f"[{status}] {task}"
    if detail:
        line = f"{line} | {detail}"
    print(line)
