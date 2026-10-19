"""Verify lossless frame alignment, score recomputation and paired decisions."""
import json
import math
from pathlib import Path
import sys
import zipfile

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from moid.evaluation import evaluate


def equal_metrics(actual, expected):
    """Allow only floating-point roundoff across the two Python environments."""
    if isinstance(actual, dict):
        return actual.keys()==expected.keys() and all(equal_metrics(actual[k],expected[k]) for k in actual)
    if isinstance(actual, list):
        return len(actual)==len(expected) and all(equal_metrics(a,b) for a,b in zip(actual,expected))
    if isinstance(actual, float):
        return math.isclose(actual,expected,rel_tol=1e-12,abs_tol=1e-12)
    return actual==expected


def main():
    manifest=json.loads((ROOT/'experiments/gmot40_manifest.json').read_text())
    records=[]
    with zipfile.ZipFile(ROOT/'experiments/local/gmot40/GenericMOT_JPEG_Sequence.zip') as source:
        names=source.namelist()
        for seq in manifest['sequences']:
            clip=seq['id']
            file=next(n for n in names if f'/{clip}/img1/000001.jpg' in n)
            original=cv2.imdecode(np.frombuffer(source.read(file),np.uint8),cv2.IMREAD_COLOR)
            cap=cv2.VideoCapture(str(ROOT/seq['video']))
            ok,decoded=cap.read()
            count=1
            while cap.read()[0]:
                count+=1
            cap.release()
            assert ok and np.array_equal(original,decoded),'Frame 1 must decode identically'
            annotations=json.loads((ROOT/seq['annotations']).read_text())['frames']
            assert count==len(annotations)
            result=dict(clip=clip,decoded_frames=count,first_frame_exact_source_jpeg=True,methods=[])
            predictions=[]
            for method in ['geometry','semantic']:
                path=ROOT/'experiments/runs/gmot40'/clip/method
                meta=json.loads((path/'provenance.json').read_text())
                if meta['status']!='completed':
                    result['methods'].append(dict(method=method,status=meta['status'],error=meta['error']))
                    continue
                pred=json.loads((path/'pipeline/results.json').read_text())
                saved=json.loads((path/'metrics.json').read_text())
                assert equal_metrics(evaluate(pred,annotations),saved)
                predictions.append([(r['found'],r.get('bbox_xyxy')) for r in pred['frames']])
                assert pred['processed_frames']==count
                result['methods'].append(dict(method=method,status='completed',metrics_recomputed=True))
            result['paired_decisions_and_boxes_identical']=predictions[0]==predictions[1] if len(predictions)==2 else None
            records.append(result)
    (ROOT/'experiments/evidence/gmot40/verification.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print(json.dumps(records,indent=2))


if __name__=='__main__':
    main()
