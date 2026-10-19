"""Ensure mutually exclusive pipeline causes cannot conceal selection/recall errors."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('diagnostic_v3', Path(__file__).parents[1]/'scripts/diagnose_instance_v3.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_pipeline_failure_attribution():
    f = module.decompose
    assert f([], None, False, False)['reason'] == 'proposal_missing'
    assert f([.8, .2], 1, True, False)['reason'] == 'wrong_selection'
    assert f([.8], 0, False, False)['reason'] == 'visual_rejection'
    assert f([.8], 0, True, False)['reason'] == 'text_rejection'
    assert f([.5], 0, True, True)['reason'] == 'success'
