"""Sequential checkpointed CPU experiment stages; never overlap large models."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
PY=ROOT/'.venv/Scripts/python.exe'
AUDIT=ROOT/'.audit-venv/Scripts/python.exe'
EVIDENCE=ROOT/'experiments/evidence/mlp_v2'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--await-captions',action='store_true')
    args=p.parse_args()
    if args.await_captions:
        while True:
            path=EVIDENCE/'captions.json'
            if path.exists():
                try:
                    if json.loads(path.read_text(encoding='utf-8')).get('status')=='completed':
                        break
                except json.JSONDecodeError:
                    pass
            time.sleep(2)
    stages=[('features',['scripts/mlp_data.py','features']),('train',['scripts/learned_experiment.py','train']),('calibrate',['scripts/learned_experiment.py','calibrate']),('evaluate',['scripts/learned_experiment.py','evaluate']),('plots',['scripts/plot_learned.py']),('full_proposals',['scripts/full_image_mlp.py','proposals']),('full_visual',['scripts/full_image_mlp.py','visual']),('full_describe',['scripts/full_image_mlp.py','describe']),('full_text',['scripts/full_image_mlp.py','text']),('full_evaluate',['scripts/full_image_mlp.py','evaluate'])]
    for name,command in stages:
        record=EVIDENCE/f'{name}_execution.json'
        if record.exists() and json.loads(record.read_text()).get('returncode')==0:
            print('Already completed:',name,flush=True)
            continue
        print('Starting:',name,flush=True)
        result=subprocess.run([str(AUDIT),'scripts/monitor_stage.py','--record',str(record),str(PY),*command],cwd=ROOT)
        if result.returncode:
            print('Stage failed; saved log:',record.with_suffix('.log'),flush=True)
            return result.returncode
    return 0


if __name__=='__main__':
    sys.exit(main())
