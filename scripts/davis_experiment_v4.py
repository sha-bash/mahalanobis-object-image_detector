"""Preregistered full-image transfer to 20 DAVIS sequences and two continuous demos."""
import argparse
import io
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
from PIL import Image
from instance_data_v4 import ROOT, save, sha

sys.path.insert(0,str(ROOT/'src'))
LOCAL=ROOT/'experiments/local/davis_v4/prepared'
OUT=ROOT/'experiments/evidence/davis_v4'
MAIN=ROOT/'experiments/evidence/instance_v4'
MODELS=ROOT/'experiments/local/instance_v4'
PROTOCOL=ROOT/'experiments/INSTANCE_V4_VIDEO_PROTOCOL.md'


def read(name): return json.loads((OUT/name).read_text(encoding='utf-8'))


def mask_bbox(mask, identity):
    yy,xx=np.where(mask==identity)
    return [int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1] if len(xx) else None


def prepare():
    LOCAL.mkdir(parents=True,exist_ok=True); OUT.mkdir(parents=True,exist_ok=True)
    refs=[]; rows=[]; inventory=[]
    with zipfile.ZipFile(ROOT/'experiments/local/davis_v4/trainval.zip') as z:
        split=next(n for n in z.namelist() if n.endswith('ImageSets/2017/val.txt'))
        sequences=sorted(z.read(split).decode().split())[:20]; assert len(sequences)==20
        for seq in sequences:
            images=sorted(n for n in z.namelist() if f'JPEGImages/480p/{seq}/' in n and n.endswith('.jpg'))
            masks={Path(n).stem:n for n in z.namelist() if f'Annotations/480p/{seq}/' in n and n.endswith('.png')}
            first=Path(images[0]).stem; mask=np.array(Image.open(io.BytesIO(z.read(masks[first]))))
            ids=sorted(int(i) for i in np.unique(mask) if i not in (0,255))[:2]; assert ids
            im=Image.open(io.BytesIO(z.read(images[0]))).convert('RGB'); seqrefs={}
            for identity in ids:
                key=f'davis_ref_{seq}_{identity}'; path=LOCAL/(key+'.png'); box=mask_bbox(mask,identity)
                im.crop(box).save(path)
                row=dict(key=key,identity=f'{seq}:{identity}',path=path.relative_to(ROOT).as_posix(),sha256=sha(path),source_image=images[0],bbox=box)
                refs.append(row); seqrefs[identity]=row
            main_positions=set(np.linspace(1,len(images)-1,min(8,len(images)-1),dtype=int).tolist())
            positions=range(1,len(images)) if seq in sequences[:2] else sorted(main_positions)
            for pos in positions:
                member=images[pos]; frame=Path(member).stem; key=f'{seq}_{frame}'; path=LOCAL/(key+'.jpg')
                path.write_bytes(z.read(member)); mask=np.array(Image.open(io.BytesIO(z.read(masks[frame]))))
                boxes={int(i):mask_bbox(mask,int(i)) for i in np.unique(mask) if i not in (0,255)}
                for identity in ids:
                    target=boxes.get(identity); ref=seqrefs[identity]
                    rows.append(dict(key=f'{key}__{identity}',scene=seq,frame_key=key,frame=int(frame),identity=ref['identity'],
                        image=path.relative_to(ROOT).as_posix(),image_sha256=sha(path),reference=ref['path'],reference_key=ref['key'],
                        bbox=target,present=target is not None,distractors=[b for i,b in boxes.items() if i!=identity],
                        main_eval=pos in main_positions,continuous_eval=seq in sequences[:2],source_mask=masks[frame]))
            inventory.append(dict(sequence=seq,frames=len(images),query_ids=ids,main_positions=sorted(main_positions),continuous=seq in sequences[:2]))
    save(OUT/'manifest.json',dict(protocol_sha256=sha(PROTOCOL),refs=refs,frames=rows,sequences=inventory,
        reference_protocol='First-frame RGB crop; frame excluded from evaluation; no first-frame coordinate supplied to detector',
        absence_definition='No visible mask pixels, not proof of physical absence'))
    print(dict(sequences=len(inventory),queries=len(refs),main_decisions=sum(r['main_eval'] for r in rows),all_decisions=len(rows),
               main_absent=sum(r['main_eval'] and not r['present'] for r in rows),unique_frames=len({r['frame_key'] for r in rows})),flush=True)


def proposals():
    import instance_scenes_v3 as shared
    shared.LOCAL=LOCAL; shared.OUT=OUT; shared.PROTOCOL=PROTOCOL
    shared.proposals()


def policy():
    frozen=json.loads((MAIN/'frozen_thresholds.json').read_text())
    return frozen['settings']['0.02']['policies']


def visual():
    import torch
    from transformers import AutoModel
    from transformers.models.bit.image_processing_pil_bit import BitImageProcessorPil
    from multimodal_data import DINO
    from moid.multimodal_metrics import normalize
    torch.set_num_threads(4)
    manifest=read('manifest.json'); proposals=read('proposals.json'); samples=manifest['refs']+proposals['crops']
    processor=BitImageProcessorPil.from_pretrained(str(DINO),local_files_only=True)
    model=AutoModel.from_pretrained(str(DINO),local_files_only=True).eval(); vectors=[]
    for start in range(0,len(samples),8):
        ims=[Image.open(ROOT/r['path']).convert('RGB') for r in samples[start:start+8]]
        with torch.inference_mode():
            vectors.extend(model(**processor(images=ims,return_tensors='pt')).last_hidden_state[:,0].numpy())
    del model,processor
    x=normalize(np.array(vectors)); refs={r['identity']:i for i,r in enumerate(manifest['refs'])}; offset=len(refs)
    distances=1-x[:offset]@x[offset:].T
    prop={r['frame_key']:r['boxes'] for r in proposals['records']}; gate=policy()['gate_text_s2027_full']['config']
    selections=[]; requested={}
    for row in manifest['frames']:
        boxes=prop[row['frame_key']]; a=refs[row['identity']]
        selected=min(boxes,key=lambda b:distances[a,b['crop_index']]) if boxes else None
        distance=float(distances[a,selected['crop_index']]) if selected else 1e30
        needs=selected is not None and gate['enabled'] and gate['ambiguity_threshold']<=distance<=gate['visual_threshold']
        cand_key=None
        if needs:
            candidate=proposals['crops'][selected['crop_index']]; ref=manifest['refs'][a]
            requested[candidate['key']]=candidate; requested[ref['key']]=ref; cand_key=candidate['key']
        selections.append(dict(key=row['key'],bbox=selected['bbox'] if selected else None,distance=distance,
            requested=bool(needs),candidate_key=cand_key,reference_key=row['reference_key']))
    save(OUT/'selections.json',dict(records=selections,thresholds_sha256=sha(MAIN/'frozen_thresholds.json')))
    (OUT/'captions').mkdir(exist_ok=True); save(OUT/'captions/manifest.json',dict(crops=list(requested.values())))
    np.savez(OUT/'visual_features.npz',visual=x)
    print('Text crops required:',len(requested),'candidate crops:',len(proposals['crops']),flush=True)


def describe():
    import instance_scenes_v3 as shared
    shared.LOCAL=LOCAL; shared.OUT=OUT; shared.PROTOCOL=PROTOCOL
    shared.describe()


def aggregate(rs):
    c={k:sum(r[k] for r in rs) for k in ['tp','fp','fn','tn','wrong_instance']}
    present=[r for r in rs if r['present']]; absent=[r for r in rs if not r['present']]
    return dict(**c,decisions=len(rs),present=len(present),absent=len(absent),
        recall=c['tp']/len(present) if present else None,
        precision=c['tp']/(c['tp']+c['fp']) if c['tp']+c['fp'] else 0.,
        absent_fpr=sum(r['fp'] for r in absent)/len(absent) if absent else None,
        mean_iou=float(np.mean([r['iou'] for r in present])) if present else None,
        proposal_recall=sum(r['proposal_available'] for r in present)/len(present) if present else None,
        risk=(5*c['fp']+c['fn'])/len(rs),text_requested_fraction=sum(r['text_requested'] for r in rs)/len(rs))


def evaluate():
    from moid.evaluation import iou
    from moid.multimodal_metrics import normalize
    manifest=read('manifest.json'); selections=read('selections.json')
    assert selections['thresholds_sha256']==sha(MAIN/'frozen_thresholds.json')
    captions=read('captions/captions.json')['records']; valid=[r for r in captions if r.get('valid_description',True)]; vectors={}
    if valid:
        import torch
        from sentence_transformers import SentenceTransformer
        from multimodal_data import SBERT
        from mcd.projection.runtime import load_projector,project
        torch.set_num_threads(4)
        model=SentenceTransformer(str(SBERT),device='cpu',local_files_only=True)
        x=model.encode([r['caption'] for r in valid],normalize_embeddings=True,batch_size=32); del model
        z=normalize(project(load_projector(MODELS/'text_s2027.json'),x)); vectors=dict(zip([r['key'] for r in valid],z))
    inverse=np.linalg.inv(np.array(json.loads((MAIN/'covariance.json').read_text())['text_s2027_full']))
    prop={r['frame_key']:r['boxes'] for r in read('proposals.json')['records']}; policies=policy(); output=[]
    for row,s in zip(manifest['frames'],selections['records']):
        assert row['key']==s['key']; td=None
        if s['requested'] and s['candidate_key'] in vectors and s['reference_key'] in vectors:
            d=vectors[s['reference_key']]-vectors[s['candidate_key']];td=float(np.sqrt(max(0,d@inverse@d)))
        for name in ['visual_baseline','visual_strict','gate_text_s2027_full']:
            cfg=policies[name]['config']; bbox=s['bbox'] if s['distance']<=cfg['visual_threshold'] else None
            if name.startswith('gate_') and s['requested'] and td is not None and td>cfg['text_threshold']: bbox=None
            overlap=iou(bbox,row['bbox']) if bbox is not None and row['present'] else 0.
            tp=int(row['present'] and bbox is not None and overlap>=.5); fp=int(bbox is not None and not tp)
            output.append(dict(key=row['key'],scene=row['scene'],frame=row['frame'],identity=row['identity'],method=name,
                main_eval=row['main_eval'],continuous_eval=row['continuous_eval'],present=row['present'],bbox=bbox,iou=overlap,
                tp=tp,fp=fp,fn=int(row['present'] and not tp),tn=int(not row['present'] and bbox is None),
                wrong_instance=int(fp and any(iou(bbox,b)>=.5 for b in row['distractors'])),
                proposal_available=bool(row['present'] and any(iou(b['bbox'],row['bbox'])>=.5 for b in prop[row['frame_key']])),
                text_requested=bool(name.startswith('gate_') and s['requested']),text_distance=td))
    summaries=[]; perseq=[]; episodes=[]
    for name in ['visual_baseline','visual_strict','gate_text_s2027_full']:
        for subset in ['main_eval','continuous_eval']:
            rs=[r for r in output if r['method']==name and r[subset]]
            seqrows=[]
            for seq in sorted({r['scene'] for r in rs}):
                record=dict(method=name,subset=subset,scene=seq,**aggregate([r for r in rs if r['scene']==seq])); perseq.append(record);seqrows.append(record)
            macro={key:float(np.mean([r[key] for r in seqrows if r[key] is not None])) if any(r[key] is not None for r in seqrows) else None
                   for key in ['recall','precision','absent_fpr','mean_iou','proposal_recall','risk']}
            summaries.append(dict(method=name,subset=subset,**aggregate(rs),macro=macro,sequences=len(seqrows)))
        for identity in sorted({r['identity'] for r in output if r['continuous_eval']}):
            rs=sorted([r for r in output if r['identity']==identity and r['method']==name and r['continuous_eval']],key=lambda r:r['frame'])
            lost=sum(a['tp'] and b['present'] and not b['tp'] for a,b in zip(rs,rs[1:]))
            restored=sum(not a['tp'] and b['tp'] for a,b in zip(rs,rs[1:]))
            fp_episodes=sum(r['fp'] and (i==0 or not rs[i-1]['fp']) for i,r in enumerate(rs))
            return_delays=[]
            for i in range(1,len(rs)):
                if not rs[i-1]['present'] and rs[i]['present']:
                    run=[]
                    for r in rs[i:]:
                        if not r['present']: break
                        run.append(r)
                    first=next((r for r in run if r['tp']),None)
                    return_delays.append(first['frame']-rs[i]['frame'] if first else None)
            episodes.append(dict(method=name,identity=identity,losses=lost,recoveries=restored,fp_episodes=fp_episodes,return_delays_frames=return_delays))
    for name,data in [('predictions.json',output),('summary.json',summaries),('per_sequence.json',perseq),('episodes.json',episodes)]: save(OUT/name,data)
    print(json.dumps([r for r in summaries if r['subset']=='main_eval'],indent=2),flush=True)


if __name__=='__main__':
    import os
    os.environ['YOLO_CONFIG_DIR']=str(LOCAL/'ultralytics');os.environ['YOLO_AUTOINSTALL']='False';os.environ['HF_HUB_OFFLINE']='1'
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','proposals','visual','describe','evaluate'])
    globals()[p.parse_args().stage]()
