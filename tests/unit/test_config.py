"""Config loading, inheritance, and the validation that catches bad experiments early.

A misconfigured arm is the most expensive kind of bug here: it runs to
completion, produces plausible numbers, and is only caught during review. The
loader therefore rejects unknown keys everywhere instead of ignoring them.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from fpso.config.loader import load_experiment_config
from fpso.config.schema import FPSOParams

ARM_DIR = Path("configs/arms")


@pytest.mark.parametrize("arm_path", sorted(ARM_DIR.glob("*.yaml")), ids=lambda p: p.stem)
def test_every_shipped_arm_config_loads(arm_path):
    """All committed arm configs must load and validate."""
    config = load_experiment_config(arm_path)
    assert config.name == arm_path.stem
    assert config.seeds


def test_arms_in_the_matrix_share_a_data_configuration():
    """Cross-arm comparisons are meaningless if the arms saw different panels."""
    configs = [load_experiment_config(p) for p in sorted(ARM_DIR.glob("*.yaml"))]
    signatures = {
        (c.data.source, c.data.panel_start, c.data.panel_end, c.data.universe_size,
         c.data.universe_rule)
        for c in configs
    }
    assert len(signatures) == 1, f"Arms disagree on data config: {signatures}"


def test_regime_arms_share_one_parameter_map():
    """Arms 2-6 must differ only in detector, or the comparison is confounded."""
    regime_arms = [
        load_experiment_config(ARM_DIR / f"{name}.yaml")
        for name in ("regime_hmm", "regime_gmm", "regime_volq", "regime_shuffled",
                     "regime_oracle")
    ]
    maps = {repr(sorted(c.regime.overrides.items())) for c in regime_arms}
    assert len(maps) == 1, "Regime arms do not share a single parameter map."

    detectors = {c.regime.detector for c in regime_arms}
    assert len(detectors) == len(regime_arms), "Regime arms must use distinct detectors."


def test_calibrated_arm_differs_only_in_its_parameter_map():
    """The calibration sensitivity arm must isolate the parameter map.

    If it also changed the detector, the schedule or the data, the comparison
    against the a-priori headline would answer a different question than the one
    the paper asks.
    """
    a_priori = load_experiment_config(ARM_DIR / "regime_hmm.yaml")
    calibrated = load_experiment_config(ARM_DIR / "regime_hmm_calibrated.yaml")

    assert calibrated.regime.overrides != a_priori.regime.overrides
    assert calibrated.regime.detector == a_priori.regime.detector
    assert calibrated.schedule == a_priori.schedule
    assert calibrated.data == a_priori.data
    assert calibrated.base_params == a_priori.base_params
    assert calibrated.seeds == a_priori.seeds


def test_static_arm_has_no_overrides():
    config = load_experiment_config(ARM_DIR / "static.yaml")
    assert not config.is_regime_conditioned
    assert config.regime.overrides == {}


def test_extends_merges_parent_and_child(tmp_path):
    (tmp_path / "parent.yaml").write_text(
        "name: parent\nbase_params:\n  alpha: 0.2\n  gamma: 1.0\nseeds: [0, 1]\n"
    )
    (tmp_path / "child.yaml").write_text(
        "extends: parent.yaml\nname: child\nbase_params:\n  alpha: 0.5\n"
    )
    config = load_experiment_config(tmp_path / "child.yaml")

    assert config.name == "child"
    assert config.base_params.alpha == 0.5, "Child override was not applied."
    assert config.base_params.gamma == 1.0, "Parent value was not inherited."


def test_unknown_top_level_key_is_rejected(tmp_path):
    (tmp_path / "bad.yaml").write_text("name: bad\nnum_seeds: 30\n")
    with pytest.raises(ValueError, match="Unknown key"):
        load_experiment_config(tmp_path / "bad.yaml")


def test_unknown_nested_key_is_rejected(tmp_path):
    (tmp_path / "bad.yaml").write_text("name: bad\nregime:\n  detecter: hmm\n")
    with pytest.raises(ValueError, match="Unknown key"):
        load_experiment_config(tmp_path / "bad.yaml")


def test_infeasible_constraints_are_rejected(tmp_path):
    """K * u < 1 would make every rebalance raise; catch it at load time instead."""
    (tmp_path / "bad.yaml").write_text(
        "name: bad\nbase_params:\n  max_assets: 5\n  max_weight: 0.1\n"
    )
    with pytest.raises(ValueError, match="Infeasible constraints"):
        load_experiment_config(tmp_path / "bad.yaml")


def test_unknown_regime_label_is_rejected(tmp_path):
    (tmp_path / "bad.yaml").write_text(
        "name: bad\nregime:\n  overrides:\n    PANIC:\n      alpha: 0.5\n"
    )
    with pytest.raises(ValueError, match="not a RegimeLabel"):
        load_experiment_config(tmp_path / "bad.yaml")


def test_panel_must_start_before_the_schedule(tmp_path):
    (tmp_path / "bad.yaml").write_text(
        'name: bad\nschedule:\n  start: "2005-01-01"\n'
        'data:\n  panel_start: "2010-01-01"\n'
    )
    with pytest.raises(ValueError, match="panel_start must be on or before"):
        load_experiment_config(tmp_path / "bad.yaml")


def test_params_are_immutable():
    """Frozen params guarantee a manifest describes the run that produced it."""
    params = FPSOParams()
    with pytest.raises(dataclasses.FrozenInstanceError):
        params.alpha = 0.9  # type: ignore[misc]


def test_with_overrides_returns_a_new_object():
    base = FPSOParams(alpha=0.2)
    modified = base.with_overrides({"alpha": 0.5})
    assert base.alpha == 0.2
    assert modified.alpha == 0.5
    assert modified.gamma == base.gamma
