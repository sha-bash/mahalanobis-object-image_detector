import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    'freeze_experiment', Path(__file__).parents[1] / 'scripts/freeze_experiment.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_checkpoint_refuses_changed_threshold_without_replacing_original(tmp_path):
    thresholds = tmp_path / 'thresholds.json'
    thresholds.write_text('{"threshold": 0.4}')
    checkpoint = module.freeze_inputs(tmp_path, ['thresholds.json'])
    original = checkpoint.read_bytes()
    assert module.freeze_inputs(tmp_path, ['thresholds.json']) == checkpoint
    thresholds.write_text('{"threshold": 0.7}')
    with pytest.raises(RuntimeError, match='Frozen inputs changed'):
        module.freeze_inputs(tmp_path, ['thresholds.json'])
    assert checkpoint.read_bytes() == original


def test_missing_input_does_not_create_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError):
        module.freeze_inputs(tmp_path, ['missing.json'])
    assert not (tmp_path / 'pretest_checkpoint.json').exists()
