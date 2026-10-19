"""Full-image IoU and gate effects; no target-absence claims."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiments/evidence/mlp_full_image'
LABELS={'SIFT':'SIFT/RANSAC','visual_cos':'YOLO + DINO','visual_cos+text':'YOLO + DINO + текст','visual_mlp_s2027_cos':'YOLO + сиамский MLP','visual_mlp_s2027_cos+text':'YOLO + сиамский MLP + текст'}


def main():
    rows=json.loads((BASE/'predictions.json').read_text())
    rows=[r for r in rows if r['split']=='test']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig,axs=plt.subplots(1,2,figsize=(11,4.3))
    thresholds=np.linspace(.05,1,20)
    for name,label in LABELS.items():
        selected=[r for r in rows if r['method']==name]
        rates=[np.mean([r['bbox'] is not None and r['iou']>=tau for r in selected]) for tau in thresholds]
        axs[0].plot(thresholds,rates,label=label,marker='.',ms=3)
    axs[0].axvline(.5,ls=':',color='black')
    axs[0].set(xlabel='Порог IoU',ylabel='Доля успешных кадров',title='Локализация на 28 целых кадрах')
    axs[0].legend(fontsize=7)
    summary=json.loads((BASE/'summary.json').read_text())
    values=[next(r for r in summary if r['split']=='test' and r['method']==m) for m in LABELS]
    x=np.arange(len(values))
    axs[1].bar(x-.17,[v['tp'] for v in values],width=.34,label='Правильные рамки')
    axs[1].bar(x+.17,[v['fp'] for v in values],width=.34,label='Ошибочные рамки')
    axs[1].set_xticks(x,list(LABELS.values()),rotation=40,ha='right',fontsize=8)
    axs[1].set_ylabel('Количество'); axs[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(BASE/'iou_gate.png',dpi=180,bbox_inches='tight'); plt.close(fig)
    from PIL import Image,ImageDraw
    manifest={r['key']:r for r in json.loads((BASE/'manifest.json').read_text())['frames']}
    errors=[r for r in rows if r['method']=='visual_cos' and r['fp']]
    # Deterministic first four failures, not hand-selected to favour a method.
    panel=Image.new('RGB',(960,280*min(4,len(errors))),'white')
    for number,r in enumerate(errors[:4]):
        frame=manifest[r['key']]; im=Image.open(ROOT/frame['image']).convert('RGB')
        drawing=ImageDraw.Draw(im)
        drawing.rectangle(frame['bbox'],outline='lime',width=3); drawing.rectangle(r['bbox'],outline='red',width=3)
        im.thumbnail((760,250)); panel.paste(im,(200,number*280+25))
        ref=Image.open(ROOT/frame['reference']).convert('RGB'); ref.thumbnail((170,240)); panel.paste(ref,(10,number*280+25))
        ImageDraw.Draw(panel).text((10,number*280+3),r['key'],fill='black')
    if errors:
        panel.save(BASE/'localization_errors.png')
    cropbase=ROOT/'experiments/evidence/mlp_v2'
    comparison=json.loads((cropbase/'test_summary.json').read_text())
    names=['visual_cos','text_mlp_s2027_full','fusion_s2027']
    selected=[next(r for r in comparison if r['method']==n and r['k']==1) for n in names]
    fig,ax=plt.subplots(figsize=(6.4,2.5))
    y=np.arange(3)
    ax.barh(y-.17,[100*r['tpr'] for r in selected],height=.32,label='Полнота',color='#357ca5')
    ax.barh(y+.17,[100*r['fpr'] for r in selected],height=.32,label='Принятие чужого ID',color='#bb6137')
    for i,r in enumerate(selected):
        ax.text(100*r['tpr']+1,i-.17,f'{100*r["tpr"]:.1f}',va='center',fontsize=9)
        ax.text(100*r['fpr']+1,i+.17,f'{100*r["fpr"]:.1f}',va='center',fontsize=9)
    ax.set_yticks(y,['DINOv2 + косинус','Текстовый MLP + М.','Совместная метрика'],fontsize=9)
    ax.set(xlim=(0,100),xlabel='Проценты на тесте'); ax.invert_yaxis(); ax.legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,1.3),ncol=2)
    fig.tight_layout(); fig.savefig(cropbase/'paper_comparison.png',dpi=240,bbox_inches='tight'); plt.close(fig)
    print('Saved full-image IoU plot',flush=True)


if __name__=='__main__': main()
