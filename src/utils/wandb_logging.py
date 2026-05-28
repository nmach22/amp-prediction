"""Weights & Biases logging helpers for experiment scripts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


DEFAULT_WANDB_CONFIG_PATH = Path("config/wandb.yml")
_API_KEY_PLACEHOLDERS = {
    "",
    "YOUR_WANDB_API_KEY",
    "paste-your-api-key-here",
    "${WANDB_API_KEY}",
}


def flatten_split_metrics(
    metrics_by_split: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Flatten nested split metrics to wandb-friendly keys."""
    return {
        f"{split}/{name}": value
        for split, split_metrics in metrics_by_split.items()
        for name, value in split_metrics.items()
    }


def flatten_metric_history_row(row: dict[str, Any]) -> dict[str, Any]:
    """Flatten one metric-history row to W&B keys."""
    split = row["split"]
    flattened = {
        "step": row["step"],
        "epoch": row["step"],
        "training_step": row["step"],
    }
    if "num_estimators" in row:
        flattened["num_estimators"] = row["num_estimators"]

    for name, value in row["metrics"].items():
        flattened[f"{split}/{name}"] = value
    return flattened


def group_metric_history(
    metric_history: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Group split metric rows into one W&B payload per training step."""
    grouped: dict[int, dict[str, Any]] = {}
    for row in metric_history:
        step = int(row["step"])
        payload = grouped.setdefault(
            step,
            {
                "step": step,
                "epoch": step,
                "training_step": step,
            },
        )
        if "num_estimators" in row:
            payload["num_estimators"] = row["num_estimators"]
        payload.update(flatten_metric_history_row(row))
    return [grouped[step] for step in sorted(grouped)]


def log_wandb_run(
    *,
    project: str,
    run_name: str,
    config: dict[str, Any],
    metrics_by_split: dict[str, dict[str, float]],
    metric_history: list[dict[str, Any]] | None = None,
    mode: str | None = None,
    entity: str | None = None,
    tags: list[str] | None = None,
    api_key: str | None = None,
    figures_dir: str | Path | None = None,
) -> None:
    """Log only run config, scalar metrics, and generated figures to W&B."""
    try:
        import wandb
    except ImportError as exc:
        raise SystemExit(
            "Weights & Biases logging requires `wandb`. "
            "Install dependencies from env.yml or run `pip install wandb`."
        ) from exc

    figure_path = Path(figures_dir) if figures_dir is not None else None

    resolved_api_key = _resolve_api_key(api_key)
    if resolved_api_key:
        os.environ["WANDB_API_KEY"] = resolved_api_key

    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        config=config,
        mode=mode,
        tags=tags,
    )
    try:
        if metric_history:
            for payload in group_metric_history(metric_history):
                wandb.log(payload, step=payload["step"])
        else:
            wandb.log(flatten_split_metrics(metrics_by_split))

        if figure_path is not None and figure_path.exists():
            for image_path in sorted(figure_path.glob("*.png")):
                wandb.log({image_path.stem: wandb.Image(str(image_path))})
    finally:
        run.finish()


def load_wandb_config(path: str | Path = DEFAULT_WANDB_CONFIG_PATH) -> dict[str, Any]:
    """Load optional local W&B config."""
    config_path = Path(path)
    if not config_path.exists():
        return {}

    with open(config_path) as f:
        raw_config = f.read()

    try:
        import yaml
    except ImportError:
        data = _load_simple_wandb_config(raw_config)
    else:
        data = yaml.safe_load(raw_config) or {}
    if not isinstance(data, dict):
        raise ValueError(f"W&B config must be a YAML mapping: {config_path}")
    return data


def _load_simple_wandb_config(raw_config: str) -> dict[str, Any]:
    """Parse the simple checked-in W&B config format without PyYAML."""
    data: dict[str, Any] = {}
    active_list_key: str | None = None

    for raw_line in raw_config.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue

        stripped = line.strip()
        if stripped.startswith("- ") and active_list_key:
            data[active_list_key].append(_parse_scalar(stripped[2:].strip()))
            continue

        active_list_key = None
        if ":" not in stripped:
            continue

        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value == "":
            data[key] = []
            active_list_key = key
        else:
            data[key] = _parse_scalar(value)

    return data


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"", "null", "none", "~"}:
        return None
    return value.strip("\"'")


def resolve_wandb_settings(
    *,
    config_path: str | Path,
    default_project: str,
    cli_project: str | None,
    cli_mode: str | None,
    cli_disabled: bool,
) -> dict[str, Any]:
    """Merge local W&B config with CLI overrides."""
    cfg = load_wandb_config(config_path)
    enabled = bool(cfg.get("enabled", True)) and not cli_disabled
    project = cli_project or cfg.get("project") or default_project
    mode = cli_mode or cfg.get("mode")
    tags = cfg.get("tags") or []

    return {
        "enabled": enabled,
        "project": project,
        "entity": cfg.get("entity") or None,
        "mode": mode,
        "tags": tags,
        "api_key": cfg.get("api_key") or None,
        "config_path": str(config_path),
    }


def _resolve_api_key(api_key: str | None) -> str | None:
    if api_key is None:
        return None
    api_key = str(api_key).strip()
    if api_key.startswith("${") and api_key.endswith("}"):
        return os.environ.get(api_key[2:-1])
    if api_key in _API_KEY_PLACEHOLDERS:
        return None
    return api_key
