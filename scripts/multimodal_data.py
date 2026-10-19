"""Frozen GMOT crop preparation, independent captions and local feature extraction."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from prepare_gmot40 import parse_labels, eligible, crop

BASE=ROOT/'experiments/local/multimodal_v1'
EVIDENCE=ROOT/'experiments/evidence/multimodal_v1'
PROTOCOL=ROOT/'experiments/MULTIMODAL_PROTOCOL.md'
KEEP_ALIVE='0'
FINALIZE_FAILURES=False
HUB=Path.home()/'.cache/huggingface/hub'
DINO=HUB/'models--facebook--dinov2-small/snapshots/ed25f3a31f01632728cabb09d1542f84ab7b0056'
SBERT=HUB/'models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41'


def sha(path):
    with open(path,'rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def save(path,payload):
    path.write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding='utf-8')


def prepare():
    import cv2
    import numpy as np
    BASE.mkdir(parents=True,exist_ok=False)
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    rows=[]
    with zipfile.ZipFile(ROOT/'experiments/local/gmot40/track_label.zip') as labels, zipfile.ZipFile(ROOT/'experiments/local/gmot40/GenericMOT_JPEG_Sequence.zip') as images:
        files={n.split('/')[-3]+'/'+Path(n).stem:n for n in images.namelist() if '/img1/' in n and n.endswith('.jpg')}
        for clip in ['car-0','car-1','car-2','car-3']:
            anns=parse_labels(labels.read(f'track_label/{clip}.txt').decode())
            byid={i:sorted([r for r in anns if r['identity']==i and r['frame']<300 and eligible(r)],key=lambda r:r['frame']) for i in sorted({r['identity'] for r in anns})}
            ids=[i for i,rs in byid.items() if len(rs)>=61][:6]
            if len(ids)!=6:
                raise ValueError(f'Insufficient eligible IDs in {clip}; do not silently change protocol')
            for identity in ids:
                for position in [0,10,20,40,60]:
                    ann=byid[identity][position]
                    raw=images.read(files[f'{clip}/{ann["frame"]:06d}'])
                    frame=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
                    path=BASE/f'{clip}_{identity}_{position}.png'
                    cv2.imwrite(str(path),crop(frame,ann['bbox_xyxy']))
                    rows.append(dict(key=path.stem,clip=clip,identity=f'{clip}:{identity}',source_frame=ann['frame'],
                                     position=position,role='reference' if position<=20 else 'candidate',bbox=ann['bbox_xyxy'],
                                     path=path.relative_to(ROOT).as_posix(),sha256=sha(path)))
    save(EVIDENCE/'manifest.json',dict(protocol_sha256=sha(PROTOCOL),crops=rows))
    print('Prepared',len(rows),'crops',flush=True)


def describe():
    import cv2
    from run import local_server,request_json
    from moid.semantic import IndependentCaptioner
    cv2.setNumThreads(4)
    rows=json.loads((EVIDENCE/'manifest.json').read_text(encoding='utf-8'))['crops']
    out=BASE/'captions.json'
    data=json.loads(out.read_text(encoding='utf-8')) if out.exists() else dict(records=[],failures=[],status='started',
        protocol_sha256=sha(PROTOCOL),source_sha256=sha(Path(__file__)),
        commit=subprocess.check_output(['git','-c',f'safe.directory={ROOT.as_posix()}','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        python=sys.version,versions={d.metadata['Name']:d.version for d in importlib.metadata.distributions()})
    done={r['key'] for r in data['records']}
    host='http://127.0.0.1:11435'
    start=time.perf_counter()
    previous=data.get('wall_sec',0)
    with local_server(host,'qwen3.5:0.8b'):
        data['models']=request_json(host+'/api/tags')
        captioner=IndependentCaptioner('qwen3.5:0.8b',host,BASE/'caption_cache',scene_context='',keep_alive=KEEP_ALIVE)
        data.setdefault('runtime_sessions',[]).append(dict(keep_alive=KEEP_ALIVE,starting_completed=len(done)))
        for i,row in enumerate(rows):
            if row['key'] in done:
                continue
            assert sha(ROOT/row['path'])==row['sha256']
            t=time.perf_counter()
            try:
                description=captioner.describe(cv2.imread(str(ROOT/row['path'])))
                data['records'].append(dict(key=row['key'],elapsed_sec=time.perf_counter()-t,**description))
                print(i+1,len(rows),row['key'],description['caption'][:90],flush=True)
            except Exception as exc:
                data['failures'].append(dict(key=row['key'],error=f'{type(exc).__name__}: {exc}'))
                print('FAILED',row['key'],str(exc),flush=True)
            data['wall_sec']=previous+time.perf_counter()-start
            save(out,data)
        if KEEP_ALIVE!='0':
            import urllib.request
            request=urllib.request.Request(host+'/api/generate',data=json.dumps(dict(model='qwen3.5:0.8b',keep_alive=0)).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(request,timeout=60) as response:
                response.read()
    data['status']='completed' if len(data['records'])==len(rows) else 'incomplete'
    if FINALIZE_FAILURES and data['status']=='incomplete':
        completed={r['key'] for r in data['records']}
        for row in rows:
            if row['key'] not in completed:
                data['records'].append(dict(key=row['key'],caption='Description unavailable',valid_description=False,elapsed_sec=0,cache_hit=False,fields={}))
        data['status']='completed'
        data['failure_policy']='Missing descriptions permanently refuse; placeholders excluded from text metric fitting'
    save(out,data)
    save(EVIDENCE/'captions.json',data)


def features():
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoModel
    from transformers.models.bit.image_processing_pil_bit import BitImageProcessorPil
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    torch.manual_seed(2027)
    rows=json.loads((EVIDENCE/'manifest.json').read_text(encoding='utf-8'))['crops']
    captions=json.loads((BASE/'captions.json').read_text(encoding='utf-8'))
    descriptions={r['key']:r['caption'] for r in captions['records']}
    if len(descriptions)!=len(rows):
        raise ValueError('Missing captions; report failure, do not subset evaluation')
    start=time.perf_counter()
    processor=BitImageProcessorPil.from_pretrained(str(DINO),local_files_only=True)
    model=AutoModel.from_pretrained(str(DINO),local_files_only=True).eval()
    vectors=[]
    for i in range(0,len(rows),8):
        batch=[Image.open(ROOT/r['path']).convert('RGB') for r in rows[i:i+8]]
        with torch.inference_mode():
            features=model(**processor(images=batch,return_tensors='pt')).last_hidden_state[:,0]
            vectors.extend(torch.nn.functional.normalize(features,dim=1).numpy())
    visual=np.array(vectors)
    timing=dict(visual_sec=time.perf_counter()-start,preprocessor='BitImageProcessorPil; pinned model preprocessor_config.json',
                initial_attempt_failure='AutoImageProcessor required unavailable torchvision; no features produced; switched to official PIL backend before evaluation')
    del model,processor
    import gc
    gc.collect()
    start=time.perf_counter()
    from sentence_transformers import SentenceTransformer
    textmodel=SentenceTransformer(str(SBERT),device='cpu',local_files_only=True)
    text=textmodel.encode([descriptions[r['key']] for r in rows],normalize_embeddings=True,batch_size=32)
    timing['text_sec']=time.perf_counter()-start
    del textmodel
    sift=cv2.SIFT_create(nfeatures=2000,contrastThreshold=.02)
    descriptors=[]
    for row in rows:
        im=cv2.imread(str(ROOT/row['path']),cv2.IMREAD_GRAYSCALE)
        descriptors.append(sift.detectAndCompute(im,None)[1])
    sims=np.zeros((len(rows),len(rows)))
    matcher=cv2.BFMatcher()
    start=time.perf_counter()
    for i,a in enumerate(descriptors):
        if a is None:
            continue
        for j,b in enumerate(descriptors):
            if b is None or len(b)<2:
                continue
            matches=matcher.knnMatch(a,b,k=2)
            sims[i,j]=sum(len(m)==2 and m[0].distance<.75*m[1].distance for m in matches)/len(a)
    timing['sift_pair_matrix_sec']=time.perf_counter()-start
    valid_by_key={r['key']:r.get('valid_description',True) for r in captions['records']}
    valid_text=np.array([valid_by_key[r['key']] for r in rows])
    np.savez(BASE/'features.npz',visual=visual,text=text,sift=sims,valid_text=valid_text)
    np.savez(EVIDENCE/'features.npz',visual=visual,text=text,sift=sims,valid_text=valid_text)
    timing.update(dino=str(DINO),sbert=str(SBERT),features_sha256=sha(BASE/'features.npz'),
                  dino_files={p.name:sha(p) for p in DINO.iterdir() if p.is_file()},
                  sbert_files={str(p.relative_to(SBERT)):sha(p) for p in SBERT.rglob('*') if p.is_file()})
    save(EVIDENCE/'feature_provenance.json',timing)
    print(timing['visual_sec'],timing['text_sec'],flush=True)


if __name__=='__main__':
    os.environ['HF_HUB_OFFLINE']='1'
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','describe','features'])
    args=p.parse_args()
    globals()[args.stage]()
