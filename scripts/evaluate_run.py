"""Evaluate one result JSON against explicit present/absent/ignore annotations."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from moid.evaluation import evaluate,coco_target_annotations


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('result',type=Path);p.add_argument('annotations',type=Path)
    p.add_argument('--legacy-coco',action='store_true');p.add_argument('--iou',type=float,default=.5)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    result=json.loads(args.result.read_text(encoding='utf-8'))
    annotations=json.loads(args.annotations.read_text(encoding='utf-8'))
    gt=coco_target_annotations(annotations) if args.legacy_coco else annotations['frames']
    metrics=evaluate(result,gt,args.iou)
    args.out.write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in metrics.items() if k!='frames'},indent=2))


if __name__=='__main__':main()
