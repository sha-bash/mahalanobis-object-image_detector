"""Descriptive reanalysis of fixed predictions, without model or threshold changes."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiments/evidence/instance_v4'
OUT=ROOT/'output/report_revision_20261020'
GATE='gate_text_s2027_full'
NAMES=['visual_baseline','visual_strict',GATE]

def read(path):return json.loads(path.read_text(encoding='utf-8'))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
    summary={r['method']:r for r in read(BASE/'test_summary.json') if r['delta']==.02}
    categories=read(BASE/'test_per_category.json')
    lookup={(r['category'],r['method']):r for r in categories if r['delta']==.02}
    cats=sorted({r['category'] for r in categories})
    result={'status':'descriptive reanalysis of disclosed test; no new independent observations','comparisons':{}}
    for control in NAMES[:2]:
        b=summary[control];g=summary[GATE]
        gain=np.array([lookup[(c,control)]['risk']-lookup[(c,GATE)]['risk'] for c in cats])
        indices=np.random.default_rng(2027).integers(0,len(cats),(10000,len(cats)))
        result['comparisons'][control]={'fp_relative_reduction':(b['fp']-g['fp'])/b['fp'],
            'tp_relative_reduction':(b['tp']-g['tp'])/b['tp'],
            'cost_relative_reduction':(b['risk']-g['risk'])/b['risk'],
            'risk_reduction':float(gain.mean()),'risk_ci95':np.quantile(gain[indices].mean(axis=1),[.025,.975]).tolist()}
    fig,ax=plt.subplots(figsize=(10,2));ax.axis('off')
    labels=['Эталон и кандидат\n(изображения)','Визуальное сходство\nDINOv2','Описание каждого\nизображения\nQwen','Векторы описаний\nMiniLM и\nобученный проектор','Сравнение векторов\nМахаланобис\nПринять или отказать']
    for i,label in enumerate(labels):
        x=i*.205
        ax.add_patch(FancyBboxPatch((x,.32),.175,.57,boxstyle='round,pad=.008',facecolor='#eef3f7',edgecolor='#3e5c73'))
        ax.text(x+.0875,.605,label,ha='center',va='center',fontsize=11)
        if i<4:ax.annotate('',xy=(x+.2,.60),xytext=(x+.18,.60),arrowprops={'arrowstyle':'->','color':'#3e5c73'})
    ax.text(.5,.09,'Текстовая проверка вызывается только для неоднозначных визуально принятых кандидатов',ha='center',fontsize=10)
    ax.set_xlim(-.02,1.01);ax.set_ylim(0,1)
    fig.tight_layout();fig.savefig(OUT/'pipeline.png',dpi=190,bbox_inches='tight');plt.close(fig)
    manifest=read(BASE/'manifest.json')['crops']
    predictions=read(BASE/'test_predictions.json')
    b={(r['a'],r['b']):r for r in predictions if r['delta']==.02 and r['method']=='visual_baseline'}
    g=[r for r in predictions if r['delta']==.02 and r['method']==GATE]
    corrected=next(r for r in g if not r['same'] and b[r['a'],r['b']]['accepted'] and not r['accepted'])
    lost=next(r for r in g if r['same'] and b[r['a'],r['b']]['accepted'] and not r['accepted'])
    examples=[corrected,lost]
    labels=['Другой экземпляр\nЛожное принятие\nустранено','Тот же экземпляр\nПравильное совпадение\nпотеряно']
    fig,axes=plt.subplots(2,3,figsize=(8,3.6),gridspec_kw={'width_ratios':[1,1,1.8]})
    for i,(r,label) in enumerate(zip(examples,labels)):
        for j,index in enumerate([r['a'],r['b']]):
            axes[i,j].imshow(Image.open(ROOT/manifest[index]['path']));axes[i,j].axis('off')
            ru={'cell_phone':'Телефон','hand_towel':'Полотенце'}[r['category']]
            identity=ru+' '+manifest[index]['identity'].rsplit('_',1)[-1]
            axes[i,j].set_title(('Эталон' if j==0 else 'Кандидат')+'\n'+identity,fontsize=11)
        axes[i,2].axis('off')
        axes[i,2].text(0,.94,label,fontsize=12,va='top')
        axes[i,2].text(0,.05,'Без текста: принятие\nС текстом: '+('принятие' if r['accepted'] else 'отказ'),fontsize=11)
    fig.tight_layout();fig.savefig(OUT/'examples.png',dpi=180,bbox_inches='tight');plt.close(fig)
    result['example_selection']='First matching cases in saved pair order: removed negative, removed positive; illustrative, not representative frequency estimates.'
    result['examples']=examples
    video={r['method']:r for r in read(ROOT/'experiments/evidence/davis_v4/summary.json') if r['subset']=='main_eval'}
    fig,axes=plt.subplots(1,2,figsize=(10,3.25))
    for ax,data,title in zip(axes,[summary,video],['Сопоставление изображений','Поиск на видеокадрах']):
        x=np.arange(3)
        for shift,key,label,color in [(-.18,'fp','Ложные принятия','#c66045'),(.18,'tp','Правильные совпадения','#427798')]:
            bars=ax.bar(x+shift,[data[n][key] for n in NAMES],.35,label=label,color=color)
            ax.bar_label(bars,padding=3,fontsize=10)
        ax.set_xticks(x,['Без текста','Строгий\nбез текста','С текстом']);ax.set_ylabel('Число решений');ax.set_title(title,fontsize=11);ax.set_ylim(0,175)
    handles,legend_labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,legend_labels,loc='upper center',bbox_to_anchor=(.5,1.09),ncol=2,fontsize=9,frameon=False)
    fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=190,bbox_inches='tight');plt.close(fig)
    result['video_fp_relative_reduction_vs_original']=(video['visual_baseline']['fp']-video[GATE]['fp'])/video['visual_baseline']['fp']
    (OUT/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result['comparisons'],ensure_ascii=False))

if __name__=='__main__':main()
