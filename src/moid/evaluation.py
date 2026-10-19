"""Single-reference, single-output evaluation. No unlabelled frame is negative."""
from __future__ import annotations

import math
import re
from pathlib import Path


def box(value):
    if len(value) != 4 or not all(math.isfinite(v) for v in value):
        raise ValueError("Box must have four finite coordinates")
    x1, y1, x2, y2 = value
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Box must have positive area")
    return list(value)


def iou(a, b):
    a, b = box(a), box(b)
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1]))
    return intersection / ((a[2]-a[0])*(a[3]-a[1]) +
                           (b[2]-b[0])*(b[3]-b[1]) - intersection)


def ratio(a, b):
    return a / b if b else None


def coco_target_annotations(coco):
    """Legacy adapter: only supplied target boxes are labelled; never infer absence."""
    images = {im['id']: im for im in coco['images']}
    rows = []
    seen = set()
    for ann in coco['annotations']:
        im = images[ann['image_id']]
        match = re.fullmatch(r'frame_(\d+)', Path(im['file_name']).stem)
        if not match:
            raise ValueError('Expected frame_N filenames')
        frame = int(match[1])
        if frame in seen:
            raise ValueError('Legacy adapter requires exactly one target per frame')
        seen.add(frame)
        x, y, w, h = ann['bbox']
        rows.append(dict(frame=frame, status='present', bbox_xyxy=box([x,y,x+w,y+h]),
                         size=[im['width'], im['height']], distractors_complete=False))
    return sorted(rows, key=lambda r: r['frame'])


def evaluate(result, annotations, threshold=0.5):
    if not 0 < threshold <= 1:
        raise ValueError('IoU threshold must be in (0,1]')
    predictions = {r['frame']: r for r in result['frames']}
    if any(type(r['frame']) is not int or type(r['found']) is not bool for r in result['frames']):
        raise ValueError('Integer frame and Boolean found required')
    if len(predictions) != len(result['frames']):
        raise ValueError('Duplicate prediction frame')
    if sorted(predictions) != list(range(result['processed_frames'])):
        raise ValueError('Predictions must cover every decoded frame, including refusals')
    if result['bbox_coordinates'] != 'output pixels':
        raise ValueError('Explicit output pixel coordinates required')
    ow, oh = result['output_size']
    if min(ow, oh) <= 0:
        raise ValueError('Invalid output size')
    counts = dict(present=0, absent=0, ignored=0, tp=0, fp=0, fn=0,
                  false_on_absent=0, wrong_instance=0, distractor_evaluable=0,
                  losses=0, recoveries=0, presence_episodes=0, episodes_without_acquisition=0)
    details, seen, delays = [], set(), []
    last_frame, last_success, onset, acquired = None, None, None, False
    for ann in sorted(annotations, key=lambda a: a['frame']):
        f, status = ann['frame'], ann['status']
        if type(f) is not int:
            raise ValueError('Integer annotation frame required')
        if f in seen or f not in predictions:
            raise ValueError('Duplicate annotation or annotation outside predictions')
        seen.add(f)
        if status not in {'present', 'absent', 'ignore'}:
            raise ValueError('Unknown annotation status')
        pred = predictions[f]
        p = box(pred['bbox_xyxy']) if pred['found'] else None
        if p and not (0 <= p[0] < p[2] <= ow and 0 <= p[1] < p[3] <= oh):
            raise ValueError('Prediction outside output image')
        if status == 'ignore':
            if onset is not None and not acquired:
                counts['episodes_without_acquisition'] += 1
            counts['ignored'] += 1
            last_success, onset = None, None
            continue
        sw, sh = ann['size']
        if min(sw, sh) <= 0:
            raise ValueError('Invalid annotation size')
        def scaled(b):
            b = box(b)
            if not (0 <= b[0] < b[2] <= sw and 0 <= b[1] < b[3] <= sh):
                raise ValueError('Annotation outside source image')
            return [v*s for v,s in zip(b, [ow/sw,oh/sh,ow/sw,oh/sh])]
        target = scaled(ann['bbox_xyxy']) if status == 'present' else None
        overlap = iou(p, target) if p and target else 0.0
        success = status == 'present' and overlap >= threshold
        counts[status] += 1
        counts['tp'] += int(success)
        counts['fp'] += int(p is not None and not success)
        counts['fn'] += int(status == 'present' and not success)
        counts['false_on_absent'] += int(status == 'absent' and p is not None)
        complete = ann.get('distractors_complete', False)
        counts['distractor_evaluable'] += int(complete)
        wrong = bool(p and not success and any(iou(p, scaled(b)) >= threshold
                     for b in ann.get('distractor_boxes', [])))
        counts['wrong_instance'] += int(wrong)
        continuous = last_frame is not None and f == last_frame + 1
        if status == 'present':
            if not continuous or onset is None:
                if onset is not None and not acquired:
                    counts['episodes_without_acquisition'] += 1
                counts['presence_episodes'] += 1
                onset, acquired, last_success = f, False, None
            if last_success is True and not success:
                counts['losses'] += 1
            if success and not acquired:
                delays.append(f-onset)
                acquired = True
            elif success and last_success is False:
                counts['recoveries'] += 1
            last_success = success
        else:
            if onset is not None and not acquired:
                counts['episodes_without_acquisition'] += 1
            onset, last_success = None, None
        last_frame = f
        details.append(dict(frame=f, status=status, iou=overlap, success=success,
                            found=p is not None, wrong_instance=wrong))
    if onset is not None and not acquired:
        counts['episodes_without_acquisition'] += 1
    present_scores = [r['iou'] for r in details if r['status'] == 'present']
    return dict(**counts, precision=ratio(counts['tp'], counts['tp']+counts['fp']),
                recall=ratio(counts['tp'], counts['present']),
                absence_fpr=ratio(counts['false_on_absent'], counts['absent']),
                mean_iou_present=ratio(sum(present_scores),len(present_scores)),
                acquisition_delays_frames=delays, iou_threshold=threshold,
                evaluated=len(details), unlabelled=len(predictions)-len(seen),
                frames=details)


def validate_manifest(manifest):
    """Reject identity/video/scene leakage, including reference origin videos."""
    owners = {}
    ids = set()
    for seq in manifest['sequences']:
        if seq['id'] in ids:
            raise ValueError('Duplicate sequence ID')
        ids.add(seq['id'])
        split = seq['split']
        if split not in {'train', 'validation', 'test', 'debug'}:
            raise ValueError('Unknown split')
        if seq['reference_protocol'] not in {'same_scene', 'independent_capture'}:
            raise ValueError('Explicit reference protocol required')
        if not seq.get('cluster_id'):
            raise ValueError('Independent cluster ID required')
        for key, values in [('identity', seq['identities']),
                            ('video', [seq['video_id'], seq['reference_video_id']]),
                            ('scene', seq['scene_ids']), ('cluster', [seq['cluster_id']])]:
            for value in values:
                token = (key, value)
                if token in owners and owners[token] != split:
                    raise ValueError(f'Split leakage: {token}')
                owners[token] = split
    return True
