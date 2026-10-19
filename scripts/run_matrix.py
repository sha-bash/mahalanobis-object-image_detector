"""Sequential runs with sampled peak memory of the child process tree."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import psutil

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--python',required=True)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--variants',nargs='+',default=['legacy_geometry','foreground_geometry','foreground_semantic','template'])
p.add_argument('--cache-only',action='store_true')
p.add_argument('--fresh-captions',action='store_true')
p.add_argument('--managed-server',action='store_true')
p.add_argument('--manifest',type=Path)
p.add_argument('--sequence',default='demo_van')
p.add_argument('--sbert-model')
args=p.parse_args()
args.out.mkdir(parents=True,exist_ok=False)
for variant in args.variants:
    out=args.out/variant
    command=[args.python,str(ROOT/'scripts/run_experiment.py'),'--variant',variant,'--out',str(out)]
    if args.cache_only:command.append('--cache-only')
    if args.fresh_captions:command.append('--fresh-captions')
    if args.managed_server:command.append('--managed-server')
    if args.manifest:command.extend(['--manifest',str(args.manifest)])
    command.extend(['--sequence',args.sequence])
    if args.sbert_model:command.extend(['--sbert-model',args.sbert_model])
    peak=0; started=time.perf_counter()
    with (args.out/(variant+'.log')).open('w',encoding='utf-8') as log:
        child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        proc=psutil.Process(child.pid)
        while child.poll() is None:
            try:
                processes=[proc]+proc.children(recursive=True)
                rss=0
                for process in processes:
                    try:rss+=process.memory_info().rss
                    except (psutil.NoSuchProcess,psutil.AccessDenied):pass
                peak=max(peak,rss)
            except psutil.NoSuchProcess:pass
            time.sleep(0.1)
    record=dict(command=command,returncode=child.returncode,wall_sec=time.perf_counter()-started,
                peak_tree_rss_sampled_bytes=peak,sample_interval_sec=0.1,
                excludes='pre-existing external Ollama server only; managed child server included',ram_bytes=psutil.virtual_memory().total,
                cpu=__import__('platform').processor())
    (args.out/(variant+'_monitor.json')).write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(variant,child.returncode,round(record['wall_sec'],2),flush=True)
