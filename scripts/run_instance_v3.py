"""Resume sequential CPU stages, refuse test evaluation until thresholds are frozen."""
import json
from pathlib import Path
import subprocess
import sys
from freeze_experiment import freeze_inputs

ROOT=Path(__file__).resolve().parents[1]
PYTHON=ROOT/'.venv/Scripts/python.exe'
MONITOR=ROOT/'.audit-venv/Scripts/python.exe'
OUT=ROOT/'experiments/evidence/instance_v3'
STAGES=[
    ('features','instance_data_v3.py','features'),
    ('train','instance_experiment_v3.py','train'),
    ('calibrate','instance_experiment_v3.py','calibrate'),
    ('evaluate','instance_experiment_v3.py','evaluate'),
    ('scene_proposals','instance_scenes_v3.py','proposals'),
    ('scene_visual','instance_scenes_v3.py','visual'),
    ('scene_describe','instance_scenes_v3.py','describe'),
    ('scene_evaluate','instance_scenes_v3.py','evaluate'),
    ('plots','plot_instance_v3.py',None),
]


def main():
    caption=json.loads((OUT/'caption_execution.json').read_text())
    if caption['returncode']!=0:
        raise RuntimeError('Caption stage must complete successfully first')
    for name,script,stage in STAGES:
        record=OUT/(name+'_execution.json')
        if record.exists() and json.loads(record.read_text())['returncode']==0:
            continue
        if name=='evaluate':
            freeze_inputs(OUT, ['frozen_thresholds.json','training_provenance.json'])
        command=[str(MONITOR),'scripts/monitor_stage.py','--record',str(record),str(PYTHON),'scripts/'+script]
        if stage: command.append(stage)
        print('Stage:',name,flush=True)
        subprocess.run(command,cwd=ROOT,check=True)


if __name__=='__main__':
    try: main()
    except Exception as exc:
        print(type(exc).__name__,str(exc),file=sys.stderr,flush=True)
        raise
