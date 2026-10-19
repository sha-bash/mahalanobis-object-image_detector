"""New fixed seven-category GMOT expansion; resumable sequential feature stages."""
import argparse
import os
from pathlib import Path
import zipfile

import multimodal_data as shared
from prepare_gmot40 import crop, parse_labels

ROOT=shared.ROOT
BASE=ROOT/'experiments/local/mlp_v2'
EVIDENCE=ROOT/'experiments/evidence/mlp_v2'
PROTOCOL=ROOT/'experiments/MLP_PROTOCOL.md'
CATEGORIES=['ball','balloon','bird','fish','insect','person','stock']


def prepare():
    import cv2
    import numpy as np
    BASE.mkdir(parents=True,exist_ok=True)
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    rows=[]
    with zipfile.ZipFile(ROOT/'experiments/local/gmot40/track_label.zip') as labels, zipfile.ZipFile(ROOT/'experiments/local/gmot40/GenericMOT_JPEG_Sequence.zip') as images:
        files={n.split('/')[-3]+'/'+Path(n).stem:n for n in images.namelist() if '/img1/' in n and n.endswith('.jpg')}
        for category in CATEGORIES:
            for number in range(4):
                clip=f'{category}-{number}'
                anns=parse_labels(labels.read(f'track_label/{clip}.txt').decode())
                valid=[r for r in anns if r['frame']<300 and r['bbox_xyxy'][2]-r['bbox_xyxy'][0]>=20 and r['bbox_xyxy'][3]-r['bbox_xyxy'][1]>=20]
                byid={i:sorted([r for r in valid if r['identity']==i],key=lambda r:r['frame']) for i in sorted({r['identity'] for r in valid})}
                ids=[i for i,rs in byid.items() if len(rs)>=61][:4]
                if len(ids)!=4:
                    raise ValueError(f'Ineligible clip {clip}; do not replace based on scores')
                for identity in ids:
                    for position in [0,10,20,40,60]:
                        ann=byid[identity][position]
                        raw=images.read(files[f'{clip}/{ann["frame"]:06d}'])
                        frame=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
                        path=BASE/f'{clip}_{identity}_{position}.png'
                        cv2.imwrite(str(path),crop(frame,ann['bbox_xyxy']))
                        rows.append(dict(key=path.stem,clip=clip,category=category,identity=f'{clip}:{identity}',source_frame=ann['frame'],position=position,role='reference' if position<=20 else 'candidate',split='train' if number==0 else 'validation' if number==1 else 'test',bbox=ann['bbox_xyxy'],path=path.relative_to(ROOT).as_posix(),sha256=shared.sha(path)))
    shared.save(EVIDENCE/'manifest.json',dict(protocol_sha256=shared.sha(PROTOCOL),crops=rows))
    print('Prepared',len(rows),'crops',flush=True)


if __name__=='__main__':
    os.environ['HF_HUB_OFFLINE']='1'
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','describe','features'])
    args=parser.parse_args()
    shared.BASE,shared.EVIDENCE,shared.PROTOCOL=BASE,EVIDENCE,PROTOCOL
    shared.KEEP_ALIVE='5m'
    shared.FINALIZE_FAILURES=True
    if args.stage=='prepare':
        prepare()
    else:
        getattr(shared,args.stage)()
