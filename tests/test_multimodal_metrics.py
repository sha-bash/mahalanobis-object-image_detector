import numpy as np
import pytest
from moid.multimodal_metrics import fit_metric,choose_threshold,rates,distance_matrix,normalize


def test_pca_and_covariance_do_not_depend_on_test_vectors():
    rng=np.random.default_rng(2027)
    a=rng.normal(size=(20,16))
    b=a.copy()
    b[12:]=rng.normal(size=(8,16))*100
    _,pa=fit_metric(a,range(12),[(i,i+1) for i in range(0,12,2)])
    _,pb=fit_metric(b,range(12),[(i,i+1) for i in range(0,12,2)])
    for key in ['mean','components','diagonal','covariance']:
        np.testing.assert_allclose(pa[key],pb[key],atol=1e-12)
    with pytest.raises(ValueError,match='leakage'):
        fit_metric(a,range(12),[(0,13),(2,3)])


def test_threshold_respects_whole_tied_groups_and_validation_recall():
    scores=[.1,.2,.2,.5]
    labels=[True,True,False,False]
    assert choose_threshold(scores,labels)==.2
    assert rates(scores,labels,.2)['fpr']==.5


def test_isotropic_mahalanobis_equals_cosine_transform():
    x=normalize(np.random.default_rng(1).normal(size=(6,8)))
    cosine=distance_matrix(x)
    mahal=distance_matrix(x,'mahalanobis',np.eye(8)*.01)
    np.testing.assert_allclose(mahal**2,200*cosine,atol=1e-10)
