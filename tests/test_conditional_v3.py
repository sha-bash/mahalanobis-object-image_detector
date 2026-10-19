"""Critical invariants of the new split and optional semantic rejection policy."""
import importlib.util
from pathlib import Path
import sys
import numpy as np

scripts=Path(__file__).parents[1]/'scripts'
sys.path.insert(0,str(scripts))
spec=importlib.util.spec_from_file_location('instance_experiment_v3',scripts/'instance_experiment_v3.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_pair_construction_never_crosses_split():
    rows=[dict(split=s,category='same_class',identity=f'{s}_{i}',role=role,key=f'{s}_{i}_{role}')
          for s in ['train','validation','test'] for i in range(2) for role in ['reference','candidate']]
    ps=module.pairs(rows,'test')
    assert len(ps)==4 and sum(p['same'] for p in ps)==2
    assert all(rows[p['a']]['split']==rows[p['b']]['split']=='test' for p in ps)


def test_gate_preserves_clear_matches_and_missing_text():
    config=dict(enabled=True,visual_threshold=.5,ambiguity_threshold=.3,text_threshold=2.)
    accepted,requested=module.gate_decision(np.array([.1,.4,.4,.8]),np.array([9.,9.,1e30,0.]),np.array([True,True,False,True]),config)
    assert accepted.tolist()==[True,False,True,False]
    assert requested.tolist()==[False,True,True,False]
    config['enabled']=False
    accepted,requested=module.gate_decision(np.array([.1,.4,.4,.8]),np.zeros(4),np.ones(4,bool),config)
    assert accepted.tolist()==[True,True,True,False] and not requested.any()
