import sys
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from instance_experiment_v4 import select_policy, gate_decision


def test_strict_visual_control_obeys_same_recall_floor():
    v=np.array([.1,.3,.2,.4]); y=np.array([True,True,False,False]); g=np.array(['a']*4)
    r=select_policy(v,np.zeros(4),np.ones(4,bool),y,g,.4,.98,False)
    assert r['counts']==dict(tp=2,fp=1,fn=0,tn=1)
    assert r['config']['visual_threshold']==.3
    assert r['loss']==5


def test_unhelpful_text_is_disabled_and_never_adds_acceptances():
    v=np.array([.1,.3,.2,.4]); y=np.array([True,True,False,False]); g=np.array(['a']*4)
    r=select_policy(v,np.zeros(4),np.ones(4,bool),y,g,.4,1.,True)
    assert not r['config']['enabled']
    accepted,_=gate_decision(v,np.zeros(4),np.ones(4,bool),r['config'])
    assert np.all(~accepted | (v<=.4))


def test_missing_all_descriptions_keeps_visual_policy():
    v=np.array([.1,.3,.2,.4]); y=np.array([True,True,False,False]); g=np.array(['a']*4)
    r=select_policy(v,np.full(4,1e30),np.zeros(4,bool),y,g,.4,1.,True)
    assert not r['config']['enabled']


def test_mask_box_is_half_open_and_absence_is_none():
    from davis_experiment_v4 import mask_bbox
    mask=np.array([[0,2,2],[0,2,0]])
    assert mask_bbox(mask,2)==[1,0,3,2]
    assert mask_bbox(mask,1) is None
