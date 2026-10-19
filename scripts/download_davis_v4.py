"""Resumable download of the official DAVIS trainval archive; no redistribution."""
import hashlib
import json
from pathlib import Path
import time
import requests

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiments/local/davis_v4'
OUT=ROOT/'experiments/evidence/davis_v4'
URL='https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip'


def main():
    BASE.mkdir(parents=True,exist_ok=True); OUT.mkdir(parents=True,exist_ok=True)
    target=BASE/'trainval.zip'; partial=target.with_suffix('.part')
    if target.exists() and (OUT/'download.json').exists(): return
    offset=partial.stat().st_size if partial.exists() else 0; start=time.perf_counter(); last=start
    with requests.get(URL,headers={'Range':f'bytes={offset}-'} if offset else {},stream=True,timeout=(30,90)) as r:
        r.raise_for_status(); append=offset>0 and r.status_code==206
        if append and not r.headers.get('Content-Range','').startswith(f'bytes {offset}-'): raise ValueError('Invalid range')
        count=offset if append else 0
        with partial.open('ab' if append else 'wb') as f:
            for chunk in r.iter_content(1024*1024):
                f.write(chunk);count+=len(chunk)
                if time.perf_counter()-last>20:
                    print(round(count/1024**2,1),'MiB downloaded',flush=True);last=time.perf_counter()
        if r.headers.get('Content-Length'):
            assert count==int(r.headers['Content-Length'])+(offset if append else 0)
    partial.replace(target)
    with target.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
    (OUT/'download.json').write_text(json.dumps(dict(url=URL,bytes=count,sha256=digest,seconds=time.perf_counter()-start,
        source='https://davischallenge.org/davis2017/code.html',
        terms='Dataset metadata CC BY-NC 4.0; new annotations CC BY 4.0. Research use; raw media kept local.',
        terms_source='https://raw.githubusercontent.com/davisvideochallenge/davisvideochallenge.github.io/master/index.html'),indent=2),encoding='utf-8')
    print('Official DAVIS archive downloaded',flush=True)


if __name__=='__main__': main()
