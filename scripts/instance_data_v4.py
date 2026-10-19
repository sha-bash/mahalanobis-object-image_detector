"""Prepare a new ID-disjoint dataset before scoring; reuse pinned encoders."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import zlib
import numpy as np
from instance_data_v3 import sha, save, ROOT

LOCAL=ROOT/'experiments/local/instance_v4'
OUT=ROOT/'experiments/evidence/instance_v4'
PROTOCOL=ROOT/'experiments/INSTANCE_V4_PROTOCOL.md'


def prepare():
    from rgbd_range_v3 import index, member
    by_id={}
    for info in index():
        m=re.fullmatch(r'(.+_\d+)_(\d+)_(\d+)_crop.png',Path(info.filename).name)
        if m:
            identity,video,frame=m.groups()
            by_id.setdefault(identity,{}).setdefault(int(video),[]).append((int(frame),info))
    used={r['identity'] for r in json.loads((ROOT/'experiments/evidence/instance_v3/manifest.json').read_text())['crops']}
    reserve={'bowl','cap','cereal_box','coffee_mug','soda_can','flashlight'}
    groups={}
    for identity in by_id:
        cat=identity.rsplit('_',1)[0]
        if identity not in used and cat not in reserve:
            groups.setdefault(cat,[]).append(identity)
    available=sorted(c for c,ids in groups.items() if len(ids)>=4)
    assert len(available)>=20
    categories=[available[i] for i in np.linspace(0,len(available)-1,20,dtype=int)]
    selected=[]
    for category in categories:
        ids=sorted(groups[category],key=lambda i:int(i.rsplit('_',1)[1]))[:4]
        for j,identity in enumerate(ids):
            videos=sorted(by_id[identity]); assert len(videos)>=2
            ref=min(by_id[identity][videos[0]],key=lambda x:x[0])
            candidates=sorted(by_id[identity][videos[1]],key=lambda x:x[0]); assert len(candidates)>=4
            frames=[(videos[0],ref)]+[(videos[1],candidates[int((len(candidates)-1)*q)]) for q in [0,1/3,2/3,1]]
            for pos,(video,(frame,info)) in enumerate(frames):
                key=Path(info.filename).stem
                selected.append((info,dict(key=key,identity=identity,category=category,clip=category,
                    split='validation' if j<2 else 'test',role='reference' if pos==0 else 'candidate',
                    position=0 if pos==0 else 20+20*pos,video=video,source_frame=frame,
                    path=(LOCAL/(key+'.png')).relative_to(ROOT).as_posix(),source_member=info.filename,crc32=info.CRC)))
    LOCAL.mkdir(parents=True,exist_ok=True); OUT.mkdir(parents=True,exist_ok=True)
    def fetch(item):
        info,row=item; path=ROOT/row['path']
        if not path.exists(): path.write_bytes(member(info))
        assert zlib.crc32(path.read_bytes())==info.CRC
        row['sha256']=sha(path); return row
    rows=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for row in pool.map(fetch,selected):
            rows.append(row)
            if len(rows)%40==0: print('Prepared',len(rows),'of',len(selected),flush=True)
    assert len(rows)==400 and not used & {r['identity'] for r in rows}
    save(OUT/'manifest.json',dict(protocol_sha256=sha(PROTOCOL),source='Washington RGB-D',categories=categories,crops=rows))
    print(categories,flush=True)


def describe():
    import multimodal_data as shared
    shared.BASE=LOCAL; shared.EVIDENCE=OUT; shared.PROTOCOL=PROTOCOL; shared.KEEP_ALIVE='5m'
    shared.FINALIZE_FAILURES=False; shared.describe()
    if json.loads((LOCAL/'captions.json').read_text())['status']!='completed':
        shared.FINALIZE_FAILURES=True; shared.describe()


def features():
    import multimodal_data as shared
    shared.BASE=LOCAL; shared.EVIDENCE=OUT; shared.PROTOCOL=PROTOCOL; shared.features()


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('stage',choices=['prepare','describe','features'])
    globals()[p.parse_args().stage]()
