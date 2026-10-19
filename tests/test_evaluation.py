import copy
import pytest
from moid.evaluation import evaluate, iou, validate_manifest
from moid.instance import clipped_box
from moid.experiment_stats import paired_cluster_interval


def test_half_open_boxes_and_anisotropic_scaling():
    assert iou([0,0,10,10],[10,0,20,10]) == 0
    result=dict(processed_frames=1,output_size=[100,40],bbox_coordinates='output pixels',
                frames=[dict(frame=0,found=True,bbox_xyxy=[10,4,20,8])])
    m=evaluate(result,[dict(frame=0,status='present',size=[1000,500],bbox_xyxy=[100,50,200,100])])
    assert m['mean_iou_present'] == 1
    assert m['absence_fpr'] is None


def test_misses_false_acceptance_and_ignored_frames():
    result=dict(processed_frames=5,output_size=[100,100],bbox_coordinates='output pixels',frames=[
        dict(frame=0,found=False),dict(frame=1,found=True,bbox_xyxy=[50,50,60,60]),
        dict(frame=2,found=True,bbox_xyxy=[0,0,10,10]),dict(frame=3,found=False),dict(frame=4,found=False)])
    gt=[dict(frame=0,status='present',size=[100,100],bbox_xyxy=[0,0,10,10]),
        dict(frame=1,status='present',size=[100,100],bbox_xyxy=[0,0,10,10],
             distractors_complete=True,distractor_boxes=[[50,50,60,60]]),
        dict(frame=2,status='absent',size=[100,100]),dict(frame=3,status='ignore')]
    m=evaluate(result,gt)
    assert (m['tp'],m['fp'],m['fn'],m['wrong_instance']) == (0,2,2,1)
    assert m['recall']==0 and m['absence_fpr']==1 and m['unlabelled']==1
    assert m['mean_iou_present']==0
    result['frames'].pop()
    with pytest.raises(ValueError,match='cover'): evaluate(result,gt)


def test_losses_recovery_and_reappearance():
    found=[True,False,True,False,False,True]
    r=dict(processed_frames=6,output_size=[10,10],bbox_coordinates='output pixels',frames=[
        dict(frame=f,found=b,**({'bbox_xyxy':[0,0,10,10]} if b else {})) for f,b in enumerate(found)])
    gt=[dict(frame=f,status='absent' if f==3 else 'present',size=[10,10],bbox_xyxy=[0,0,10,10]) for f in range(6)]
    m=evaluate(r,gt)
    assert m['losses']==1 and m['recoveries']==1
    assert m['acquisition_delays_frames']==[0,1]


def test_offscreen_and_nonfinite_candidate_rejected():
    assert clipped_box([[-20,0],[-10,20]], [100,100]) is None
    assert clipped_box([[110,0],[120,20]], [100,100]) is None
    assert clipped_box([[0,0],[float('nan'),20]], [100,100]) is None
    assert clipped_box([[-5,1.5],[4.2,20.1]], [100,100]) == [0,1,5,21]


def test_split_leakage_including_reference_video():
    first=dict(id='a',split='train',reference_protocol='independent_capture',identities=['car1'],
               video_id='v1',reference_video_id='r1',scene_ids=['s1'],cluster_id='c1')
    other=dict(id='b',split='test',reference_protocol='independent_capture',identities=['car2'],
               video_id='v2',reference_video_id='r2',scene_ids=['s2'],cluster_id='c2')
    assert validate_manifest({'sequences':[first,other]})
    for key,value in [('identities',['car1']),('reference_video_id','v1'),('scene_ids',['s1']),('cluster_id','c1')]:
        altered=copy.deepcopy(other);altered[key]=value
        with pytest.raises(ValueError,match='leakage'):validate_manifest({'sequences':[first,altered]})


def test_bootstrap_counts_clusters_not_frames():
    records=[dict(cluster_id='a',a=.2,b=.4)]*1000
    assert paired_cluster_interval(records)['ci95'] is None
    records.append(dict(cluster_id='b',a=.8,b=.4))
    result=paired_cluster_interval(records,repetitions=1000)
    assert result['clusters']==2
    assert result['mean_difference']==pytest.approx(-.1)
