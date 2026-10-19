"""Prepare the frozen cross-video RGB-D crop protocol and reuse sequential encoders."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
LOCAL = ROOT/'experiments/local/instance_v3'
OUT = ROOT/'experiments/evidence/instance_v3'
PROTOCOL = ROOT/'experiments/INSTANCE_V3_PROTOCOL.md'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def prepare():
    from rgbd_range_v3 import index, member
    infos = index()
    by_id = {}
    for info in infos:
        match = re.fullmatch(r'(.+_\d+)_(\d+)_(\d+)_crop.png', Path(info.filename).name)
        if match:
            identity, video, frame = match.groups()
            by_id.setdefault(identity, {}).setdefault(int(video), []).append((int(frame), info))
    categories = sorted({i.rsplit('_', 1)[0] for i in by_id})
    excluded = {'bowl','cap','cereal_box','coffee_mug','soda_can','flashlight'}
    categories = [c for c in categories if c not in excluded and sum(i.rsplit('_',1)[0] == c for i in by_id) >= 6][:12]
    assert len(categories) == 12
    LOCAL.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    selected = []
    for category in categories:
        ids = sorted([i for i in by_id if i.rsplit('_',1)[0] == category], key=lambda i:int(i.rsplit('_',1)[1]))[:6]
        for j, identity in enumerate(ids):
            videos = sorted(by_id[identity])
            assert len(videos) >= 2
            ref = min(by_id[identity][videos[0]], key=lambda x:x[0])
            candidates = sorted(by_id[identity][videos[1]], key=lambda x:x[0])
            assert len(candidates) >= 4
            records = [(videos[0], ref)] + [(videos[1], candidates[int((len(candidates)-1)*q)]) for q in [0,1/3,2/3,1]]
            for position, (video, (frame, info)) in enumerate(records):
                key = Path(info.filename).stem
                selected.append((info, dict(key=key, identity=identity, category=category, clip=category,
                    split=['train','validation','test'][j//2], video=video, source_frame=frame,
                    role='reference' if position == 0 else 'candidate', position=0 if position == 0 else 20+position*20,
                    path=(LOCAL/(key+'.png')).relative_to(ROOT).as_posix(), source_member=info.filename, crc32=info.CRC)))
    def fetch_one(item):
        info, row = item
        path = ROOT/row['path']
        if not path.exists():
            path.write_bytes(member(info))
        import zlib
        assert zlib.crc32(path.read_bytes()) == info.CRC
        row['sha256'] = sha(path)
        return row
    rows = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for row in pool.map(fetch_one, selected):
            rows.append(row)
            if len(rows)%30 == 0:
                print('Downloaded', len(rows), '/', len(selected), flush=True)
    assert len(rows) == 360 and len({r['identity'] for r in rows}) == 72
    save(OUT/'manifest.json', dict(protocol_sha256=sha(PROTOCOL), source='Washington RGB-D Object Dataset', crops=rows))
    print('Prepared', len(rows), 'crops; categories:', categories, flush=True)


def describe():
    import multimodal_data as shared
    shared.BASE=LOCAL; shared.EVIDENCE=OUT; shared.PROTOCOL=PROTOCOL; shared.KEEP_ALIVE='5m'
    shared.FINALIZE_FAILURES=False
    shared.describe()
    # Only a single repeat for missing descriptions, then preserve them as refusals.
    if json.loads((LOCAL/'captions.json').read_text())['status'] != 'completed':
        shared.FINALIZE_FAILURES=True
        shared.describe()


def features():
    import multimodal_data as shared
    shared.BASE=LOCAL; shared.EVIDENCE=OUT; shared.PROTOCOL=PROTOCOL
    shared.features()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','describe','features'])
    globals()[parser.parse_args().stage]()
