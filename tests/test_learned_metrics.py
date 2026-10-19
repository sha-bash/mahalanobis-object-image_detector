import numpy as np
import pytest

from moid.learned_metrics import covariance_for_identities, best_f1_threshold, calibrate_recall, macro_rates


def test_covariance_never_uses_test_identities():
    rows=[dict(split='train',identity='a'),dict(split='train',identity='a'),dict(split='test',identity='b')]
    with pytest.raises(ValueError,match='training identities'):
        covariance_for_identities(np.eye(3),rows,[0,1,2])


def test_recall_budget_avoids_all_refusal_and_uses_macro():
    distances=np.array([.1,.2,.7,.8,.9,.95])
    labels=np.array([1,0,1,1,0,0],bool)
    clips=np.array(['a','a','b','b','b','b'])
    tau=calibrate_recall(distances,labels,clips,.75)
    assert macro_rates(distances,labels,clips,tau)['tpr']>=.75
    assert tau==.8  # Equal FPR: prefer higher recall, as preregistered.
    assert np.isfinite(best_f1_threshold(distances,labels,clips))


def test_real_upstream_projector_roundtrip_and_singleton(tmp_path):
    torch=pytest.importorskip('torch')
    from mcd.projection.projector import DeepMahalanobisProjector
    from mcd.projection.runtime import save_projector,load_projector,project
    torch.manual_seed(3)
    model=DeepMahalanobisProjector(input_dim=8).eval()
    x=np.random.default_rng(0).normal(size=(7,8))
    expected=project(model,x)
    config=tmp_path/'projector.json'
    save_projector(model,config)
    loaded=load_projector(config)
    np.testing.assert_allclose(project(loaded,x,batch_size=1),expected,rtol=1e-5,atol=1e-7)
    assert loaded.architecture_dict()==model.architecture_dict()


def test_missing_descriptions_cannot_be_accepted_by_threshold_search():
    from moid.learned_metrics import refusal_aware_ap
    distances=np.array([.1,1e30,.8,1e30])
    labels=np.array([True,True,False,False])
    clips=np.array(['a']*4)
    with pytest.raises(ValueError,match='infeasible'):
        calibrate_recall(distances,labels,clips,.9)
    assert calibrate_recall(distances,labels,clips,.5)==.1
    assert refusal_aware_ap(labels,distances)==.5
