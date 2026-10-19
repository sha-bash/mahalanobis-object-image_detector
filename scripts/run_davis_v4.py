"""Run CPU video stages after the main models and thresholds have completed."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/evidence/davis_v4'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--force',action='store_true',help='Recompute all stages despite saved execution records; use a separate reproduction checkout.')
    args=parser.parse_args()
    assert (ROOT/'experiments/evidence/instance_v4/test_summary.json').exists()
    assert (OUT/'manifest.json').exists()
    for stage in ['proposals','visual','describe','evaluate']:
        record=OUT/(stage+'_execution.json')
        if not args.force and record.exists() and json.loads(record.read_text())['returncode']==0: continue
        subprocess.run([str(ROOT/'.audit-venv/Scripts/python.exe'),'scripts/monitor_stage.py','--record',str(record),
                        str(ROOT/'.venv/Scripts/python.exe'),'scripts/davis_experiment_v4.py',stage],cwd=ROOT,check=True)


if __name__=='__main__': main()
