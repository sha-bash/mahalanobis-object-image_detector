"""Prepare only the eight clips specified in HELDOUT_PROTOCOL.md."""
import json
from pathlib import Path
import sys
import zipfile

import cv2
import numpy as np

from prepare_gmot40 import parse_labels, select_target, crop, sha

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from moid.evaluation import validate_manifest


def main():
    base = ROOT/'experiments/local/heldout_v1'
    base.mkdir(parents=True, exist_ok=False)
    evidence = ROOT/'experiments/evidence/heldout_v1'
    evidence.mkdir(parents=True, exist_ok=True)
    manifest = dict(version='heldout-v1', iou_threshold=.5, sequences=[])
    inventory = dict(protocol_sha256=sha(ROOT/'experiments/HELDOUT_PROTOCOL.md'), clips=[])
    archives = ROOT/'experiments/local/gmot40'
    with zipfile.ZipFile(archives/'track_label.zip') as labels, zipfile.ZipFile(archives/'GenericMOT_JPEG_Sequence.zip') as images:
        for clip in [f'{kind}-{i}' for kind in ['airplane', 'boat'] for i in range(4)]:
            rows = parse_labels(labels.read(f'track_label/{clip}.txt').decode())
            target = select_target(rows)
            record = dict(clip=clip, target=target)
            inventory['clips'].append(record)
            if target is None:
                record['status'] = 'ineligible'
                continue
            files = sorted(n for n in images.namelist() if f'/{clip}/img1/' in n and n.endswith('.jpg'))[:300]
            assert [int(Path(n).stem) for n in files] == list(range(len(files)))
            directory = base/clip
            directory.mkdir()
            first = cv2.imdecode(np.frombuffer(images.read(files[0]), np.uint8), cv2.IMREAD_COLOR)
            height, width = first.shape[:2]
            clipped = 0
            for r in rows:
                original = r['bbox_xyxy']
                r['bbox_xyxy'] = [max(0, original[0]), max(0, original[1]), min(width, original[2]), min(height, original[3])]
                clipped += r['bbox_xyxy'] != original
                assert r['bbox_xyxy'][2] > r['bbox_xyxy'][0] and r['bbox_xyxy'][3] > r['bbox_xyxy'][1]
            ref = crop(first, next(r['bbox_xyxy'] for r in rows if r['frame'] == 0 and r['identity'] == target))
            cv2.imwrite(str(directory/'reference.png'), ref)
            writer = cv2.VideoWriter(str(directory/'video.avi'), cv2.VideoWriter_fourcc(*'FFV1'), 30, (width, height))
            assert writer.isOpened()
            annotations, contact = [], []
            try:
                for f, name in enumerate(files):
                    image = cv2.imdecode(np.frombuffer(images.read(name), np.uint8), cv2.IMREAD_COLOR)
                    rs = [r for r in rows if r['frame'] == f]
                    chosen = next((r for r in rs if r['identity'] == target), None)
                    if f:
                        writer.write(image)
                        ann = dict(frame=f-1, source_frame=f, status='present' if chosen else 'ignore',
                                   size=[width, height], distractors_complete=False,
                                   distractor_boxes=[r['bbox_xyxy'] for r in rs if r['identity'] != target])
                        if chosen:
                            ann['bbox_xyxy'] = chosen['bbox_xyxy']
                        annotations.append(ann)
                    if f in [0, len(files)//2, len(files)-1]:
                        if chosen:
                            x1, y1, x2, y2 = map(int, chosen['bbox_xyxy'])
                            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        thumb = cv2.resize(image, (480, 270))
                        cv2.putText(thumb, f'{clip} id={target} frame={f}', (5, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1)
                        contact.append(thumb)
            finally:
                writer.release()
            cv2.imwrite(str(directory/'contact.jpg'), np.hstack(contact))
            (directory/'annotations.json').write_text(json.dumps(dict(frames=annotations)), encoding='utf-8')
            prefix = directory.relative_to(ROOT).as_posix()
            manifest['sequences'].append(dict(id=clip, split='test', identities=[f'{clip}:{target}'], video_id=clip,
                reference_video_id=clip, scene_ids=[clip], cluster_id=clip, reference_protocol='same_scene',
                video=prefix+'/video.avi', reference=prefix+'/reference.png', ref_box=[0, 0, ref.shape[1], ref.shape[0]],
                annotations=prefix+'/annotations.json', scene_context='', rights='GMOT-40 CC BY-NC-SA 4.0'))
            record.update(status='prepared', frames=len(annotations), present=sum(r['status'] == 'present' for r in annotations),
                          clipped_annotation_boxes=clipped, size=[width, height],
                          hashes={p.name: sha(p) for p in directory.iterdir() if p.suffix in {'.json', '.avi', '.png'}})
            print(clip, target, record['present'], '/', record['frames'], flush=True)
    validate_manifest(manifest)
    (evidence/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (evidence/'inventory.json').write_text(json.dumps(inventory, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
