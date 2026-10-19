"""Frozen video ablation; sequential subprocesses and complete run provenance."""
import argparse
import contextlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
os.environ['PYTHONPATH'] = str(ROOT/'src')
from prepare_gmot40 import sha

EVIDENCE = ROOT/'experiments/evidence/heldout_v1'
RUNS = ROOT/'experiments/runs/heldout_v1'
VARIANTS = ['geometry', 'isotropic', 'text_cosine', 'text_full']


def save(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')


def single(sequence, variant):
    import cv2
    from moid.instance import run
    from moid.evaluation import evaluate
    from moid.semantic import IndependentCaptioner, SemanticVerifier, IsolatedSBERT
    from moid.frozen_verifier import FrozenTextVerifier
    from run import local_server, request_json

    cv2.setNumThreads(4)
    out = RUNS/sequence/variant
    out.mkdir(parents=True, exist_ok=False)
    seq = next(s for s in json.loads((EVIDENCE/'manifest.json').read_text())['sequences'] if s['id'] == sequence)
    annotations = json.loads((ROOT/seq['annotations']).read_text())['frames']
    source = ROOT/'experiments/evidence/multimodal_v1'
    params = next(p for p in json.loads((source/'parameters.json').read_text()) if p['train'] == 'car-0' and p['validation'] == 'car-1' and p['modality'] == 'text')
    old = json.loads((source/'results.json').read_text())['rows']
    method = 'text_pca_cos' if variant == 'text_cosine' else 'text_full'
    tau = next(r['threshold'] for r in old if r['train'] == 'car-0' and r['validation'] == 'car-1' and r['method'] == method and r['k'] == 1)
    sbert = Path.home()/'.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
    meta = dict(sequence=sequence, variant=variant, seed=2027, width=1920, verify_every_sec=10,
                commit=subprocess.check_output(['git', '-c', f'safe.directory={ROOT.as_posix()}', 'rev-parse', 'HEAD'], text=True).strip(),
                python=sys.version, cpu=platform.processor(), versions={d.metadata['Name']:d.version for d in importlib.metadata.distributions()},
                code_sha256={str(p.relative_to(ROOT)):sha(p) for d in ['src', 'scripts'] for p in (ROOT/d).rglob('*.py')},
                input_sha256={seq[k]:sha(ROOT/seq[k]) for k in ['video', 'reference', 'annotations']},
                protocol_sha256=sha(ROOT/'experiments/HELDOUT_PROTOCOL.md'),
                frozen_parameters_sha256=sha(source/'parameters.json'), status='started',
                timing_note='Shared fresh-experiment caption/embedding cache; variant order G,I,C,M; not cold-time comparison')
    start = time.perf_counter()
    save(out/'provenance.json', meta)
    verifier = None
    try:
        with contextlib.ExitStack() as stack:
            if variant != 'geometry':
                host = 'http://127.0.0.1:11435'
                stack.enter_context(local_server(host, 'qwen3.5:0.8b'))
                meta['ollama_tags'] = request_json(host+'/api/tags')
                meta['ollama_version'] = request_json(host+'/api/version')
                cache = ROOT/'experiments/local/heldout_v1/cache'
                captioner = IndependentCaptioner('qwen3.5:0.8b', host, cache/'captions', scene_context='')
                embedder = IsolatedSBERT(str(sbert), cache/'embeddings')

                def factory(ref):
                    kwargs = dict(captioner=captioner, embedder=embedder, sbert_model=str(sbert), scene_context='')
                    if variant == 'isotropic':
                        return SemanticVerifier(ref, threshold=6, **kwargs)
                    return FrozenTextVerifier(ref, parameters=params, metric='cosine' if variant == 'text_cosine' else 'mahalanobis', threshold=tau, **kwargs)
                verifier = factory
            try:
                result = run(ROOT/seq['reference'], ROOT/seq['video'], out/'pipeline', ref_box=seq['ref_box'],
                             width=1920, seed=2027, baseline=variant == 'geometry', verifier=verifier)
                result['method'] = variant
                result['distance_metric'] = dict(geometry=None, isotropic='isotropic_mahalanobis', text_cosine='PCA_cosine', text_full='PCA_full_mahalanobis')[variant]
                meta['status'] = 'completed'
            except ValueError as exc:
                if str(exc) != 'Reference has too few distinctive details for geometric matching':
                    raise
                width, height = annotations[0]['size']
                scale = min(1, 1920/width)
                result = dict(method=variant, processed_frames=len(annotations), frames=[dict(frame=i, found=False) for i in range(len(annotations))],
                              bbox_coordinates='output pixels', output_size=[int(width*scale)//2*2, int(height*scale)//2*2],
                              semantic_checks=[], elapsed_sec=time.perf_counter()-start, operational_failure=str(exc))
                meta.update(status='operational_refusal', error=str(exc))
            save(out/'results.json', result)
            save(out/'metrics.json', evaluate(result, annotations, .5))
    except Exception as exc:
        meta.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        meta['runner_elapsed_sec'] = time.perf_counter()-start
        save(out/'provenance.json', meta)


def all_runs(python):
    import psutil
    RUNS.mkdir(parents=True, exist_ok=False)
    seqs = json.loads((EVIDENCE/'manifest.json').read_text())['sequences']
    records = []
    for seq in seqs:
        for variant in VARIANTS:
            log = RUNS/f'{seq["id"]}_{variant}.log'
            start, peak = time.perf_counter(), 0
            with log.open('w', encoding='utf-8') as stream:
                child = subprocess.Popen([python, str(Path(__file__)), '--sequence', seq['id'], '--variant', variant], stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT)
                process = psutil.Process(child.pid)
                while child.poll() is None:
                    try:
                        rss = 0
                        for p in [process]+process.children(recursive=True):
                            try:
                                rss += p.memory_info().rss
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                pass
                        peak = max(peak, rss)
                    except psutil.NoSuchProcess:
                        pass
                    time.sleep(.2)
            record = dict(sequence=seq['id'], variant=variant, returncode=child.returncode,
                          wall_sec=time.perf_counter()-start, peak_tree_rss_bytes=peak)
            records.append(record)
            save(EVIDENCE/'execution.json', records)
            print(json.dumps(record), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--sequence')
    parser.add_argument('--variant', choices=VARIANTS)
    args = parser.parse_args()
    if args.all:
        all_runs(args.python)
    elif args.sequence and args.variant:
        single(args.sequence, args.variant)
    else:
        parser.error('Use --all or --sequence and --variant')
