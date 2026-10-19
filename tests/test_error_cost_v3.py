import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('cost_v3', Path(__file__).resolve().parents[1]/'scripts/analyze_error_cost_v3.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_count_loss_equals_prevalence_weighted_loss():
    for cost in [1, 2, 5]:
        before = mod.weighted_risk(77/96, 3/96, .5, cost)
        after = mod.weighted_risk(72/96, 4/96, .5, cost)
        assert (before-after)*192 == pytest.approx(5*cost-1)


def test_prevalence_changes_break_even_and_invalid_inputs_rejected():
    before = mod.weighted_risk(77/96, 3/96, .9, 1)
    after = mod.weighted_risk(72/96, 4/96, .9, 1)
    assert after > before
    with pytest.raises(ValueError):
        mod.weighted_risk(.5, .1, 1.1, 1)
