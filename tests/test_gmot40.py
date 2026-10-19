"""Critical source-format and preregistered sample selection checks."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('prepare_gmot40', Path(__file__).resolve().parents[1]/'scripts/prepare_gmot40.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_zero_based_identity_and_xywh_conversion():
    rows = module.parse_labels('0,3,10,20,40,20,1,-1,-1,-1\r\r\n')
    assert rows == [dict(frame=0,identity=3,bbox_xyxy=[10,20,50,40])]
    image=np.arange(60*70*3).reshape(60,70,3)
    np.testing.assert_array_equal(module.crop(image,rows[0]['bbox_xyxy']),image[20:40,10:50])


def test_select_lowest_eligible_id_not_largest_or_longest():
    rows=[dict(frame=f,identity=i,bbox_xyxy=[0,0,w,25]) for i,w,n in [(0,39,100),(1,50,29),(2,40,30),(3,80,100)] for f in range(n)]
    assert module.select_target(list(reversed(rows))) == 2


def test_duplicate_identity_frame_rejected():
    with pytest.raises(ValueError, match='Duplicate'):
        module.parse_labels('0,1,0,0,40,20\n0,1,5,5,40,20')
