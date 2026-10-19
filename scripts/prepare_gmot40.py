"""Prepare the preregistered four-car-clip pilot from official GMOT-40 archives."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from moid.evaluation import validate_manifest


def sha(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def parse_labels(text):
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        values = [float(x) for x in line.split(',')]
        frame, identity, x, y, w, h = values[:6]
        if min(w, h) <= 0:
            raise ValueError('Nonpositive annotation size')
        rows.append(dict(frame=int(frame), identity=int(identity),
                         bbox_xyxy=[x, y, x+w, y+h]))
    if len({(r['frame'], r['identity']) for r in rows}) != len(rows):
        raise ValueError('Duplicate identity within frame')
    return rows


def eligible(row):
    x1, y1, x2, y2 = row['bbox_xyxy']
    return x2-x1 >= 40 and y2-y1 >= 20


def select_target(rows):
    return next((r['identity'] for r in sorted(rows, key=lambda r: r['identity'])
                 if r['frame'] == 0 and eligible(r)
                 and sum(x['identity'] == r['identity'] and x['frame'] < 300 for x in rows) >= 30), None)


def crop(image, bbox):
    x1, y1 = np.floor(bbox[:2]).astype(int)
    x2, y2 = np.ceil(bbox[2:]).astype(int)
    return image[max(0,y1):min(image.shape[0],y2), max(0,x1):min(image.shape[1],x2)]


def main():
    base = ROOT/'experiments/local/gmot40'
    out = base/'prepared'
    out.mkdir(exist_ok=False)
    manifest = dict(version='gmot40-pilot-v1', iou_threshold=.5, sequences=[])
    inventory = dict(source='https://spritea.github.io/GMOT40/download.html',
                     license='CC BY-NC-SA 4.0', protocol_sha256=sha(ROOT/'experiments/GMOT40_PROTOCOL.md'),
                     archives={p.name: dict(bytes=p.stat().st_size, sha256=sha(p)) for p in base.glob('*.zip')},
                     clips=[])
    panel = []
    with zipfile.ZipFile(base/'track_label.zip') as labels, zipfile.ZipFile(base/'GenericMOT_JPEG_Sequence.zip') as images:
        for clip in ['car-0', 'car-1', 'car-2', 'car-3']:
            text = labels.read(f'track_label/{clip}.txt').decode()
            rows = parse_labels(text)
            target = select_target(rows)
            record = dict(clip=clip, target=target, annotation_sha256=hashlib.sha256(text.encode()).hexdigest())
            inventory['clips'].append(record)
            if target is None:
                record['status'] = 'ineligible'
                continue
            files = sorted(n for n in images.namelist() if f'/{clip}/img1/' in n and n.endswith('.jpg'))[:300]
            if [int(Path(n).stem) for n in files] != list(range(len(files))):
                raise ValueError('Frames are not contiguous from zero')
            directory = out/clip
            directory.mkdir()
            frame_rows = {f: [r for r in rows if r['frame'] == f] for f in range(len(files))}
            first = cv2.imdecode(np.frombuffer(images.read(files[0]), np.uint8), cv2.IMREAD_COLOR)
            height, width = first.shape[:2]
            ref_row = next(r for r in frame_rows[0] if r['identity'] == target)
            ref = crop(first, ref_row['bbox_xyxy'])
            cv2.imwrite(str(directory/'reference.png'), ref)
            positive = [r for r in rows if r['identity'] == target and 30 <= r['frame'] < len(files) and eligible(r)]
            positive.sort(key=lambda r: r['frame'])
            selected = sorted({positive[min(len(positive)-1, int(len(positive)*q))]['frame'] for q in [1/3,2/3]}) if positive else []
            writer = cv2.VideoWriter(str(directory/'video.avi'), cv2.VideoWriter_fourcc(*'FFV1'), 30, (width,height))
            if not writer.isOpened():
                raise RuntimeError('Lossless FFV1 writer unavailable')
            annotations = []
            contact = []
            try:
                for source_frame, name in enumerate(files):
                    image = cv2.imdecode(np.frombuffer(images.read(name),np.uint8),cv2.IMREAD_COLOR)
                    if image is None or image.shape[:2] != (height,width):
                        raise ValueError('Invalid frame')
                    rs = frame_rows[source_frame]
                    target_row = next((r for r in rs if r['identity']==target),None)
                    if source_frame > 0:
                        writer.write(image)
                        ann = dict(frame=source_frame-1,source_frame=source_frame,status='present' if target_row else 'ignore',
                                   size=[width,height], distractors_complete=False,
                                   distractor_boxes=[r['bbox_xyxy'] for r in rs if r['identity'] != target])
                        if target_row:
                            ann['bbox_xyxy'] = target_row['bbox_xyxy']
                        annotations.append(ann)
                    if source_frame in selected:
                        candidates = [('positive',target_row)]
                        others = sorted((r for r in rs if r['identity'] != target and eligible(r)),key=lambda r:r['identity'])
                        if others:
                            candidates.append(('negative',others[0]))
                        for kind, row in candidates:
                            path = directory/f'{kind}_{source_frame}.png'
                            cv2.imwrite(str(path),crop(image,row['bbox_xyxy']))
                            panel.append(dict(clip=clip,target_identity=target,candidate_identity=row['identity'],
                                source_frame=source_frame,same_identity=kind=='positive',bbox_xyxy=row['bbox_xyxy'],
                                path=path.relative_to(ROOT).as_posix(),sha256=sha(path)))
                    if source_frame in [0, len(files)//2, len(files)-1]:
                        thumb=image.copy()
                        if target_row:
                            x1,y1,x2,y2=map(int,target_row['bbox_xyxy'])
                            cv2.rectangle(thumb,(x1,y1),(x2,y2),(0,255,0),3)
                        thumb=cv2.resize(thumb,(480,270))
                        cv2.putText(thumb,f'{clip} id={target} frame={source_frame}',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,255),1)
                        contact.append(thumb)
            finally:
                writer.release()
            cv2.imwrite(str(directory/'contact.jpg'),np.hstack(contact))
            (directory/'annotations.json').write_text(json.dumps(dict(frames=annotations)),encoding='utf-8')
            prefix = directory.relative_to(ROOT).as_posix()
            manifest['sequences'].append(dict(id=clip,split='test',identities=[f'{clip}:{target}'],video_id=clip,
                reference_video_id=clip,scene_ids=[clip],cluster_id=clip,reference_protocol='same_scene',
                video=prefix+'/video.avi',reference=prefix+'/reference.png',ref_box=[0,0,ref.shape[1],ref.shape[0]],
                annotations=prefix+'/annotations.json',scene_context='',rights='GMOT-40 CC BY-NC-SA 4.0'))
            record.update(status='prepared',input_frames=len(files),evaluated_present=sum(a['status']=='present' for a in annotations),
                          ignored=sum(a['status']=='ignore' for a in annotations),size=[width,height],
                          source_frame_offset=1,encoded_fps=30,fps_note='fixed processing clock, original fps not supplied',
                          video_sha256=sha(directory/'video.avi'),reference_sha256=sha(directory/'reference.png'))
    validate_manifest(manifest)
    (ROOT/'experiments/gmot40_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    evidence=ROOT/'experiments/evidence/gmot40'
    evidence.mkdir(exist_ok=True)
    (evidence/'inventory.json').write_text(json.dumps(inventory,indent=2),encoding='utf-8')
    (evidence/'panel.json').write_text(json.dumps(panel,indent=2),encoding='utf-8')
    print(json.dumps(inventory,indent=2))


if __name__ == '__main__':
    main()
