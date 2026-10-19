"""Fetch official research-only Washington RGB-D archives with resumable checkpoints."""
import hashlib
import json
from pathlib import Path
import time
import requests

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'experiments/local/rgbd_v3'
EVIDENCE = ROOT/'experiments/evidence/instance_v3'
URLS = {
    f'{name}.tar':f'https://rgbd-dataset.cs.washington.edu/dataset/rgbd-scenes/{name}.tar'
    for name in ['desk_1', 'desk_2', 'meeting_small_1', 'table_small_1', 'background']
}


def main():
    BASE.mkdir(parents=True, exist_ok=True); EVIDENCE.mkdir(parents=True, exist_ok=True)
    inventory = json.loads((EVIDENCE/'rgbd_downloads.json').read_text()) if (EVIDENCE/'rgbd_downloads.json').exists() else {}
    for name, url in URLS.items():
        target = BASE/name
        if target.exists() and name in inventory and target.stat().st_size == inventory[name]['bytes']:
            continue
        partial = target.with_suffix(target.suffix+'.part')
        offset = partial.stat().st_size if partial.exists() else 0
        with requests.get(url, headers={'Range':f'bytes={offset}-'} if offset else {}, stream=True, timeout=(30, 90)) as response:
            response.raise_for_status()
            append = offset > 0 and response.status_code == 206
            if append and not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):
                raise RuntimeError('Server range mismatch')
            total = offset if append else 0
            started = time.time(); last = started
            with partial.open('ab' if append else 'wb') as file:
                for chunk in response.iter_content(1024*1024):
                    file.write(chunk); total += len(chunk)
                    if time.time()-last > 20:
                        print(name, round(total/1024**2, 1), 'MiB', flush=True); last = time.time()
            expected = response.headers.get('Content-Length')
            if expected and total != int(expected)+(offset if append else 0):
                raise RuntimeError('Incomplete response')
        partial.replace(target)
        with target.open('rb') as file:
            digest = hashlib.file_digest(file, 'sha256').hexdigest()
        inventory[name] = dict(url=url, bytes=total, sha256=digest, download_sec=time.time()-started,
                               use='non-commercial research/educational use only',
                               terms='https://rgbd-dataset.cs.washington.edu/dataset.html')
        (EVIDENCE/'rgbd_downloads.json').write_text(json.dumps(inventory, indent=2), encoding='utf-8')
        print(name, 'complete', flush=True)


if __name__ == '__main__':
    main()
