"""YAML -> :mod:`fpso.config.schema` with inheritance and validation.

Arm configs are written as thin diffs on top of a shared base file via an
``extends:`` key, so the only lines in ``configs/arms/regime_hmm.yaml`` are the
lines that make that arm different from the baseline. That keeps the ablation
matrix auditable by eye.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from fpso.config.schema import (
    DataConfig,
    ExperimentConfig,
    FPSOParams,
    RegimeConfig,
    RegimeLabel,
    ScheduleConfig,
)

_SECTION_TYPES = {
    "base_params": FPSOParams,
    "regime": RegimeConfig,
    "schedule": ScheduleConfig,
    "data": DataConfig,
}

_TUPLE_FIELDS = {"seeds", "transaction_cost_rates", "features"}


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment config, resolving ``extends:`` chains first."""
    merged = _load_merged_mapping(Path(path))
    return _build_experiment(merged, source=Path(path))


def _load_merged_mapping(path: Path) -> dict[str, Any]:
    """Read `path` and deep-merge it onto whatever it extends."""
    with path.open() as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level.")

    parent_ref = raw.pop("extends", None)
    if parent_ref is None:
        return raw

    parent_path = (path.parent / parent_ref).resolve()
    if not parent_path.exists():
        raise FileNotFoundError(f"{path} extends missing config {parent_path}.")
    return _deep_merge(_load_merged_mapping(parent_path), raw)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge `override` onto `base`; scalars and lists are replaced."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _build_experiment(mapping: Mapping[str, Any], source: Path) -> ExperimentConfig:
    """Instantiate the config tree, rejecting unknown keys at every level."""
    payload: dict[str, Any] = {}
    for key, value in mapping.items():
        if key in _SECTION_TYPES:
            payload[key] = _build_section(_SECTION_TYPES[key], value, f"{source}:{key}")
        else:
            payload[key] = value

    payload.setdefault("name", source.stem)
    _reject_unknown(ExperimentConfig, payload, str(source))
    payload = {k: _coerce(k, v) for k, v in payload.items()}
    config = ExperimentConfig(**payload)
    _validate(config)
    return config


def _build_section(section_type: type, value: Any, where: str):
    """Build one nested config dataclass from its YAML mapping."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{where} must be a mapping, got {type(value).__name__}.")
    _reject_unknown(section_type, value, where)
    return section_type(**{k: _coerce(k, v) for k, v in value.items()})


def _reject_unknown(dataclass_type: type, payload: Mapping[str, Any], where: str) -> None:
    """Fail on keys the dataclass does not define, so typos never pass silently."""
    if not is_dataclass(dataclass_type):  # pragma: no cover - defensive
        raise TypeError(f"{dataclass_type} is not a dataclass.")
    valid = {f.name for f in fields(dataclass_type)}
    unknown = set(payload) - valid
    if unknown:
        raise ValueError(
            f"Unknown key(s) {sorted(unknown)} in {where}. Valid keys: {sorted(valid)}"
        )


def _coerce(key: str, value: Any) -> Any:
    """Freeze sequence-valued fields into tuples so configs stay hashable."""
    if key in _TUPLE_FIELDS and isinstance(value, list):
        return tuple(value)
    return value


def _validate(config: ExperimentConfig) -> None:
    """Catch configuration mistakes that would otherwise surface mid-backtest."""
    params = config.base_params
    if params.num_particles < 2:
        raise ValueError("base_params.num_particles must be at least 2.")
    if params.max_assets < 1:
        raise ValueError("base_params.max_assets must be at least 1.")
    if not 0.0 < params.max_weight <= 1.0:
        raise ValueError("base_params.max_weight must lie in (0, 1].")
    if params.max_assets * params.max_weight < 1.0:
        raise ValueError(
            f"Infeasible constraints: max_assets ({params.max_assets}) * max_weight "
            f"({params.max_weight}) = {params.max_assets * params.max_weight:.3f} < 1, "
            "so no fully-invested portfolio satisfies both bounds."
        )
    if not 0.0 <= params.pso_blend <= 1.0:
        raise ValueError("base_params.pso_blend must lie in [0, 1].")

    valid_labels = {label.name for label in RegimeLabel}
    for label_name, overrides in config.regime.overrides.items():
        if label_name not in valid_labels:
            raise ValueError(
                f"regime.overrides key '{label_name}' is not a RegimeLabel "
                f"({sorted(valid_labels)})."
            )
        # Raises on unknown override fields.
        params.with_overrides(overrides)

    if config.schedule.frequency not in {"monthly", "annual"}:
        raise ValueError("schedule.frequency must be 'monthly' or 'annual'.")
    if config.schedule.start >= config.schedule.end:
        raise ValueError("schedule.start must precede schedule.end.")
    if config.data.panel_start > config.schedule.start:
        raise ValueError(
            "data.panel_start must be on or before schedule.start so that the "
            "first rebalance has a full trailing estimation window."
        )
    if not config.seeds:
        raise ValueError("seeds must contain at least one entry.")
