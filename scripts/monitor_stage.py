"""Run a resumable stage, recording elapsed time and sampled process-tree RSS."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import psutil

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--record',required=True)
    p.add_argument('command',nargs=argparse.REMAINDER)
    args=p.parse_args()
    output=Path(args.record)
    output.parent.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    peak=0
    with output.with_suffix('.log').open('w',encoding='utf-8') as log:
        environment=os.environ.copy()
        environment.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONHASHSEED='2027')
        child=subprocess.Popen(args.command,stdout=log,stderr=subprocess.STDOUT,env=environment)
        process=psutil.Process(child.pid)
        while child.poll() is None:
            try:
                rss=0
                for proc in [process]+process.children(recursive=True):
                    try:
                        rss+=proc.memory_info().rss
                    except (psutil.NoSuchProcess,psutil.AccessDenied):
                        pass
                peak=max(peak,rss)
            except psutil.NoSuchProcess:
                pass
            time.sleep(.2)
    record=dict(command=args.command,returncode=child.returncode,wall_sec=time.perf_counter()-start,peak_tree_rss_bytes=peak,rss_note='Sum of sampled process RSS, may double-count shared pages; not total host memory')
    output.write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record),flush=True)
    raise SystemExit(child.returncode)
