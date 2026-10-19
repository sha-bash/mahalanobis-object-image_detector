import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('heldout_summary', Path(__file__).resolve().parents[1]/'scripts/summarize_heldout.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_sign_test_keeps_ties_visible_and_has_correct_small_sample_tail():
    assert module.sign_test([0]*8) == dict(positive=0, negative=0, ties=8, one_sided_p=1., conditional_nonzero_p=1.)
    assert module.sign_test([1]*8)['one_sided_p'] == pytest.approx(1/256)
    assert module.sign_test([1, -1, 0])['one_sided_p'] == pytest.approx(.875)
    assert module.sign_test([1, -1, 0])['conditional_nonzero_p'] == pytest.approx(.75)
    assert module.sign_test([-1]*8)['one_sided_p'] == 1.
