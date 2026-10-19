"""Run the frozen oracle-crop panel; never confuse it with online search quality."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'src'))
os.environ['PYTHONPATH']=str(ROOT/'src')+os.pathsep+os.environ.get('PYTHONPATH','')


def main():
    import cv2
    from audit_evidence import sha
    from run import local_server, request_json
    from moid.semantic import SemanticVerifier
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--sbert-model',required=True)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    panel=json.loads((ROOT/'experiments/evidence/gmot40/panel.json').read_text())
    manifest=json.loads((ROOT/'experiments/gmot40_manifest.json').read_text())
    cv2.setNumThreads(4)
    cv2.setRNGSeed(2027)
    host='http://127.0.0.1:11435'
    payload=dict(protocol_sha256=sha(ROOT/'experiments/GMOT40_PROTOCOL.md'),
        panel_sha256=sha(ROOT/'experiments/evidence/gmot40/panel.json'),
        commit=subprocess.check_output(['git','-c',f'safe.directory={ROOT.as_posix()}','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_sha256={str(f.relative_to(ROOT)):sha(f) for f in [Path(__file__),ROOT/'src/moid/semantic.py']},
        python=sys.version,versions={d.metadata['Name']:d.version for d in importlib.metadata.distributions()},
        threshold=6,scene_context='',records=[],references=[],failures=[],status='started')
    started=time.perf_counter()
    def save():
        payload['wall_sec']=time.perf_counter()-started
        (args.out/'results.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    save()
    with local_server(host,'qwen3.5:0.8b'):
        payload['ollama_tags']=request_json(host+'/api/tags')
        payload['ollama_version']=request_json(host+'/api/version')
        for seq in manifest['sequences']:
            try:
                ref=cv2.imread(str(ROOT/seq['reference']))
                gate=SemanticVerifier(ref,host=host,threshold=6,cache_dir=args.out/'cache',
                                      sbert_model=args.sbert_model,scene_context='')
                payload['references'].append(dict(clip=seq['id'],sha256=sha(ROOT/seq['reference']),metadata=gate.metadata))
                save()
                for candidate in [r for r in panel if r['clip']==seq['id']]:
                    if sha(ROOT/candidate['path']) != candidate['sha256']:
                        raise ValueError('Panel input changed')
                    t=time.perf_counter()
                    decision=gate.verify(cv2.imread(str(ROOT/candidate['path'])))
                    payload['records'].append(dict(**candidate,**decision,elapsed_sec=time.perf_counter()-t))
                    save()
                    print(seq['id'],candidate['source_frame'],candidate['same_identity'],decision['distance'],decision['accepted'],flush=True)
            except Exception as exc:
                payload['failures'].append(dict(clip=seq['id'],error=f'{type(exc).__name__}: {exc}'))
                save()
                print(seq['id'],type(exc).__name__,str(exc),flush=True)
    payload['status']='completed_with_failures' if payload['failures'] else 'completed'
    save()


if __name__=='__main__':
    main()
