import importlib.util
from pathlib import Path
import cv2
import numpy as np
import pytest
from moid.instance import run
from tests.test_instance import reference_image
from tests.test_semantic_video import verifier


def test_reacquisition_needs_new_semantic_confirmation(tmp_path):
    ref=reference_image();image=tmp_path/'r.png';cv2.imwrite(str(image),ref)
    video=tmp_path/'v.avi';writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),5,(640,480))
    for present in [True,True,False,False,False,True,True]:
        frame=np.zeros((480,640,3),np.uint8)
        if present:frame[100:250,200:380]=ref
        writer.write(frame)
    writer.release()
    result=run(image,video,tmp_path/'result',verifier=lambda r:verifier(['van','van','sedan']))
    assert [r['found'] for r in result['frames']]==[True,True,False,False,False,False,False]
    assert [c['frame'] for c in result['semantic_checks']]==[0,5]
    assert result['semantic_checks'][1]['accepted'] is False
    assert 'bbox_xyxy' in result['semantic_checks'][1]


def test_low_positive_fps_does_not_divide_by_zero(tmp_path):
    ref=reference_image();image=tmp_path/'r.png';cv2.imwrite(str(image),ref)
    video=tmp_path/'slow.avi';writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),0.1,(640,480))
    frame=np.zeros((480,640,3),np.uint8);frame[100:250,200:380]=ref
    writer.write(frame);writer.release()
    assert run(image,video,tmp_path/'result',baseline=True)['processed_frames']==1


def test_optimized_diagonal_distance_equals_dense_and_cosine():
    gate=verifier(['van','sedan'])
    d=gate.verify(reference_image())['distance']
    assert d*d==pytest.approx(200)
    rng=np.random.default_rng(2027)
    for _ in range(10):
        a,b=rng.normal(size=(2,8));a/=np.linalg.norm(a);b/=np.linalg.norm(b)
        var=rng.uniform(0.001,1,8);diff=a-b
        assert np.sqrt(np.sum(diff*diff/var))==pytest.approx(np.sqrt(diff@np.linalg.inv(np.diag(var))@diff))
        assert np.sum(diff*diff)/0.01==pytest.approx(200*(1-a@b))


def test_pair_calibration_uses_no_test_scores_for_threshold():
    path=Path(__file__).resolve().parents[1]/'scripts/compare_pair_metrics.py'
    spec=importlib.util.spec_from_file_location('pair_metrics',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    pairs=[]
    for split in ['train','validation','test']:
        for n,same in enumerate([True,True,False]):
            pairs.append(dict(split=split,identities=[split+str(n)],videos=[split+str(n)],same=same,
                              reference=[1.,0.],candidate=[1.,0.1] if same else [0.,1.]))
    before=module.compare(pairs)
    pairs[-1]['candidate']=[1.,0.001]
    after=module.compare(pairs)
    for metric in before['metrics']:
        assert before['metrics'][metric]['threshold']==after['metrics'][metric]['threshold']
    pairs[-1]['identities']=['train0']
    with pytest.raises(ValueError,match='leakage'):module.compare(pairs)
