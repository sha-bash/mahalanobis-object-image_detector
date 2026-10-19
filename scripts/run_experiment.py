"""Run a fixed debug comparison with input/code hashes and explicit provenance."""
import argparse
import contextlib
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
os.environ['PYTHONPATH']=str(ROOT/'src')+os.pathsep+os.environ.get('PYTHONPATH','')
from moid.evaluation import evaluate, coco_target_annotations, validate_manifest
from audit_evidence import sha


def git(*args):
    return subprocess.check_output(['git','-c',f'safe.directory={ROOT.as_posix()}',*args],cwd=ROOT).decode('utf-8')


def template_run(reference, video, out, ref_box, width, threshold):
    """External OpenCV TM_CCOEFF_NORMED, fixed reference scale; no video box."""
    import cv2
    import numpy as np
    started=time.perf_counter()
    ref=cv2.imread(str(reference)); x1,y1,x2,y2=ref_box
    ref=cv2.cvtColor(ref[y1:y2,x1:x2],cv2.COLOR_BGR2GRAY)
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened(): raise ValueError('Cannot open video')
    sw,sh=int(cap.get(3)),int(cap.get(4)); scale=min(1,width/sw)
    size=(int(sw*scale)//2*2,int(sh*scale)//2*2)
    fps=cap.get(cv2.CAP_PROP_FPS); frames=[]
    try:
        while True:
            ok,frame=cap.read()
            if not ok: break
            gray=cv2.cvtColor(cv2.resize(frame,size,interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2GRAY)
            scores=cv2.matchTemplate(gray,ref,cv2.TM_CCOEFF_NORMED)
            _,score,_,point=cv2.minMaxLoc(scores)
            found=bool(np.isfinite(score) and score >= threshold)
            row=dict(frame=len(frames), found=found, score=score)
            if found: row['bbox_xyxy']=[point[0],point[1],point[0]+ref.shape[1],point[1]+ref.shape[0]]
            frames.append(row)
    finally: cap.release()
    return dict(method='OpenCV TM_CCOEFF_NORMED fixed scale',processed_frames=len(frames),
                found_frames=sum(r['found'] for r in frames),output_size=list(size),fps=fps,
                bbox_coordinates='output pixels',frames=frames,elapsed_sec=time.perf_counter()-started,
                threshold=threshold, export_video=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant',choices=['legacy_geometry','legacy_semantic','geometry','semantic','foreground_geometry','foreground_semantic','template'],required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--manifest',type=Path,default=ROOT/'experiments/protocol.json')
    p.add_argument('--sequence',default='demo_van')
    p.add_argument('--width',type=int,default=1920)
    p.add_argument('--seed',type=int,default=2027)
    p.add_argument('--tau',type=float,default=6)
    p.add_argument('--sbert-model',default='all-MiniLM-L6-v2',help='Prefer a pinned local snapshot path for new studies')
    p.add_argument('--template-threshold',type=float,default=0.8)
    p.add_argument('--host',default='http://127.0.0.1:11435')
    p.add_argument('--cache-only',action='store_true',help='Fail on VLM cache miss; embeddings also must exist')
    p.add_argument('--fresh-captions',action='store_true',help='Use a new per-run VLM cache; embedding cache remains enabled')
    p.add_argument('--managed-server',action='store_true',help='Start project-local Ollama and stop only the service started here')
    args=p.parse_args()
    if args.cache_only and args.fresh_captions:p.error('Conflicting cache policies')
    manifest=json.loads(args.manifest.read_text(encoding='utf-8'));validate_manifest(manifest)
    seq=next(s for s in manifest['sequences'] if s['id']==args.sequence)
    scene_context=seq.get('scene_context','Outdoor aerial drone footage of objects on the ground')
    args.out.mkdir(parents=True,exist_ok=False)
    import cv2
    cv2.setNumThreads(4);cv2.setRNGSeed(args.seed)
    code={str(f.relative_to(ROOT)):sha(f) for base in ['src','scripts'] for f in (ROOT/base).rglob('*.py')}
    for relative in code:
        destination=args.out/'source_snapshot'/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/relative,destination)
    meta=dict(variant=args.variant, sequence=seq['id'],split=seq['split'],seed=args.seed,
              config=vars(args).copy(),commit=git('rev-parse','HEAD').strip(),dirty=git('status','--short'),
              code_sha256=code,protocol_sha256=sha(args.manifest),python=sys.version,
              platform=platform.platform(),cpu=platform.processor(),logical_cpus=os.cpu_count(),
              versions={d.metadata['Name']:d.version for d in importlib.metadata.distributions()},
              inputs={seq[k]:sha(ROOT/seq[k]) for k in ['reference','video','annotations']},
              status='started',cache_policy='cache_only' if args.cache_only else 'normal_cache')
    meta['fresh_captions']=args.fresh_captions
    meta['model_manifests']={str(f.relative_to(ROOT)):dict(sha256=sha(f),content=json.loads(f.read_text()))
                            for f in (ROOT/'models/ollama-models/manifests').rglob('*') if f.is_file()}
    hf=Path.home()/'.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/refs/main'
    meta['local_sbert_main_revision']=hf.read_text().strip() if hf.exists() else None
    meta['cached_embedding_original_revision']='unknown; original cache stores no revision'
    def save_meta():
        (args.out/'provenance.json').write_text(json.dumps(meta,indent=2,default=str),encoding='utf-8')
    save_meta()
    started=time.perf_counter()
    stack=contextlib.ExitStack()
    try:
        if args.variant.startswith('legacy'):
            source=git('show','5552bc2:src/moid/instance.py')
            path=args.out/'legacy_instance.py';path.write_text(source,encoding='utf-8')
            spec=importlib.util.spec_from_file_location('legacy_instance',path)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            # Also preserve the original semantic implementation for the baseline.
            source=git('show','5552bc2:src/moid/semantic.py')
            sempath=args.out/'legacy_semantic.py';sempath.write_text(source,encoding='utf-8')
            spec=importlib.util.spec_from_file_location('legacy_semantic',sempath)
            sem=importlib.util.module_from_spec(spec);spec.loader.exec_module(sem)
            meta['legacy_source_sha256']=sha(path);meta['legacy_semantic_sha256']=sha(sempath)
        else:
            import moid.instance as module
            import moid.semantic as sem
        verifier=None
        semantic=args.variant.endswith('semantic')
        if semantic:
            if args.managed_server:
                sys.path.insert(0,str(ROOT))
                from run import local_server, request_json
                stack.enter_context(local_server(args.host,'qwen3.5:0.8b'))
                meta['ollama_tags']=request_json(args.host+'/api/tags')
                meta['ollama_version']=request_json(args.host+'/api/version')
            caption_cache=args.out/'fresh_captions' if args.fresh_captions else ROOT/'.cache/semantic/captions'
            def factory(ref):
                captioner=sem.IndependentCaptioner('qwen3.5:0.8b',args.host,caption_cache,
                    scene_context=scene_context)
                if args.cache_only:
                    def unavailable(**kw): raise RuntimeError('VLM cache miss: fresh inference not permitted in cache-only run')
                    captioner.chat=unavailable
                    # IsolatedSBERT's worker must never start for a cache-only replay.
                    class CachedEmbeddings(sem.IsolatedSBERT):
                        def embed(self,texts):
                            import numpy as np
                            vectors=[]
                            for text in texts:
                                key=hashlib.sha256(json.dumps([self.model,'l2-normalized-v1',text]).encode()).hexdigest()
                                vectors.append(np.load(self.cache_dir/(key+'.npy'),allow_pickle=False))
                            return np.stack(vectors)
                    embedder=CachedEmbeddings(model=args.sbert_model,cache_dir=ROOT/'.cache/semantic/embeddings')
                else: embedder=None
                return sem.SemanticVerifier(ref,host=args.host,threshold=args.tau,captioner=captioner,sbert_model=args.sbert_model,
                    embedder=embedder,cache_dir=ROOT/'.cache/semantic',scene_context=scene_context)
            verifier=factory
        if args.variant=='template':
            result=template_run(ROOT/seq['reference'],ROOT/seq['video'],args.out,seq['ref_box'],args.width,args.template_threshold)
            (args.out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        else:
            kwargs={} if args.variant.startswith('legacy') else dict(seed=args.seed,foreground_only=args.variant.startswith('foreground'))
            result=module.run(ROOT/seq['reference'],ROOT/seq['video'],args.out/'pipeline',
                ref_box=seq['ref_box'],width=args.width,baseline=not semantic,verifier=verifier,**kwargs)
        annotations=json.loads((ROOT/seq['annotations']).read_text())
        gt=coco_target_annotations(annotations) if seq.get('annotation_format')=='legacy_single_target_coco' else annotations['frames']
        metrics=evaluate(result,gt,manifest['iou_threshold'])
        (args.out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
        meta.update(status='completed',pipeline_elapsed_sec=result['elapsed_sec'])
    except Exception as exc:
        meta.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        stack.close()
        meta['runner_elapsed_sec']=time.perf_counter()-started
        try:
            import psutil
            meta['peak_memory_bytes']=psutil.Process().memory_info().peak_wset
            meta['ram_bytes']=psutil.virtual_memory().total
        except (ImportError,AttributeError):
            meta['peak_memory_bytes']=None
        meta['memory_scope']='main process only; excludes SBERT and Ollama; null if unavailable'
        save_meta()


if __name__=='__main__': main()
