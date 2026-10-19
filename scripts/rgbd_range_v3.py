"""Read selected members of the official ZIP by HTTP Range, verifying ZIP CRC."""
import io
import json
from pathlib import Path
import struct
import time
import zipfile
import zlib
import requests

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT/'experiments/local/rgbd_v3'
URL = 'https://rgbd-dataset.cs.washington.edu/dataset/rgbd-dataset_eval/rgbd-dataset_eval.zip'


def fetch(start, end):
    path = LOCAL/'ranges'/f'{start}_{end}.bin'
    if path.exists() and path.stat().st_size == end-start+1:
        return path.read_bytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            r = requests.get(URL, headers={'Range':f'bytes={start}-{end}'}, timeout=(30, 120))
            r.raise_for_status()
            if r.status_code != 206 or not r.headers.get('Content-Range', '').startswith(f'bytes {start}-{end}/') or len(r.content) != end-start+1:
                raise RuntimeError('Invalid byte-range response')
            partial=path.with_suffix('.part')
            partial.write_bytes(r.content)
            partial.replace(path)
            return r.content
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(1)


def index():
    # Official server Content-Range total and EOCD validated before implementing this reader.
    size = 673456874
    tail = fetch(size-65536, size-1)
    end = tail.rfind(b'PK\x05\x06')
    signature, disk, cd_disk, on_disk, entries, length, offset, comment = struct.unpack('<4s4H2LH', tail[end:end+22])
    assert disk == cd_disk == 0 and entries == on_disk
    central = fetch(offset, offset+length-1)
    # Rebase central offsets for a metadata-only in-memory ZIP; local offsets in ZipInfo remain original.
    rebased = struct.pack('<4s4H2LH', signature, 0, 0, entries, entries, length, 0, 0)
    with zipfile.ZipFile(io.BytesIO(central+rebased)) as z:
        return z.infolist()


def member(info):
    extra_slack = 1024
    raw = fetch(info.header_offset, info.header_offset+30+len(info.filename.encode())+extra_slack+info.compress_size-1)
    header = struct.unpack('<4s5H3L2H', raw[:30])
    assert header[0] == b'PK\x03\x04'
    start = 30+header[-2]+header[-1]
    if start+info.compress_size > len(raw):
        raw = fetch(info.header_offset, info.header_offset+start+info.compress_size-1)
    compressed = raw[start:start+info.compress_size]
    data = zlib.decompress(compressed, -15) if info.compress_type == zipfile.ZIP_DEFLATED else compressed
    assert len(data) == info.file_size and zlib.crc32(data) == info.CRC
    return data


if __name__ == '__main__':
    infos = index()
    names = [i.filename for i in infos if i.filename.endswith('_crop.png')]
    categories = {}
    for name in names:
        identity = Path(name).name.rsplit('_', 3)[0]
        category = identity.rsplit('_', 1)[0]
        categories.setdefault(category, set()).add(identity)
    result = {k:sorted(v) for k,v in sorted(categories.items())}
    out = ROOT/'experiments/evidence/instance_v3/rgbd_inventory.json'
    out.write_text(json.dumps(dict(source=URL, rgb_images=len(names), categories=result, first_members=names[:10]), indent=2), encoding='utf-8')
    print(json.dumps(dict(images=len(names), counts={k:len(v) for k,v in result.items()}, examples=names[:5]), indent=2), flush=True)
