"""Sequential, checkpointed stages; frozen choices checkpointed locally before new scoring."""
import argparse
import json
from pathlib import Path
import subprocess
from freeze_experiment import freeze_inputs

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/evidence/instance_v4'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--force',action='store_true',help='Recompute all stages despite saved execution records; use a separate reproduction checkout.')
    args=parser.parse_args()
    assert json.loads((OUT/'caption_execution.json').read_text())['returncode']==0
    for name,script,stage in [('features','instance_data_v4.py','features'),('train','instance_experiment_v4.py','train'),
                               ('calibrate','instance_experiment_v4.py','calibrate'),('evaluate','instance_experiment_v4.py','evaluate')]:
        path=OUT/(name+'_execution.json')
        if not args.force and path.exists() and json.loads(path.read_text())['returncode']==0: continue
        if name=='evaluate':
            freeze_inputs(OUT, ['frozen_thresholds.json','training_provenance.json'])
        subprocess.run([str(ROOT/'.audit-venv/Scripts/python.exe'),'scripts/monitor_stage.py','--record',str(path),
                        str(ROOT/'.venv/Scripts/python.exe'),'scripts/'+script,stage],cwd=ROOT,check=True)


if __name__=='__main__': main()
