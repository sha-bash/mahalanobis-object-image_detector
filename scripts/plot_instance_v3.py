"""Russian scientific plots from frozen v3 decisions; no parameter changes."""
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
BASE=ROOT/'experiments/evidence/instance_v3'
LABELS={'sift':'SIFT','visual_cos':'DINOv2 + косинус','visual_hard_cos':'Визуальный MLP + косинус',
    'visual_random_cos':'MLP случайных троек + косинус','visual_random_full':'MLP случайных троек + М.',
    'visual_hard_full':'Визуальный MLP + М.','visual_hard_diagonal':'Визуальный MLP + диаг. М.',
    'text_cos':'Текст + косинус','text_raw_full':'Текст + М.', 'text_hard_cos':'Текстовый MLP + косинус',
    'text_hard_full':'Текстовый MLP + М.','text_hard_diagonal':'Текстовый MLP + диаг. М.',
    'conditional_text':'Условная текстовая проверка'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12})


def load(name): return json.loads((BASE/name).read_text(encoding='utf-8'))


def finish(fig,name):
    fig.tight_layout(); fig.savefig(BASE/(name+'.png'),dpi=170,bbox_inches='tight'); plt.close(fig)


def main():
    from sklearn.metrics import roc_curve,precision_recall_curve
    summary=load('test_summary.json'); frozen=load('frozen_thresholds.json'); records=load('test_predictions.json')
    (BASE/'labels.json').write_text(json.dumps(LABELS,ensure_ascii=False,indent=2),encoding='utf-8')
    primary=frozen['selected_visual']; names=list(dict.fromkeys(['visual_cos',primary,'text_hard_full','conditional_text']))
    selected=[next(r for r in summary if r['method']==n) for n in names]
    fig,ax=plt.subplots(figsize=(10,4.6)); y=np.arange(len(names))
    ax.barh(y-.18,[r['tpr']*100 for r in selected],height=.35,label='Полнота')
    ax.barh(y+.18,[r['fpr']*100 for r in selected],height=.35,label='Принятие чужого ID')
    ax.set_yticks(y,[LABELS[n] for n in names]); ax.set(xlabel='Проценты на новом тесте',xlim=(0,105))
    ax.axvline(frozen['recall_budget']*100,color='black',ls=':',label='Ориентир полноты')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,1.2),ncol=3,fontsize=10)
    finish(fig,'operating_points')
    fig,axs=plt.subplots(1,2,figsize=(11,4.8))
    # The two-threshold conditional policy is a fixed operating point, not a one-dimensional ROC curve.
    for name in list(dict.fromkeys(['visual_cos',primary,'text_cos','text_raw_full','text_hard_full'])):
        rows=[r for r in records if r['method']==name]; truth=[r['same'] for r in rows]; score=-np.array([r['distance'] for r in rows])
        fpr,tpr,tau=roc_curve(truth,score); p,r,pt=precision_recall_curve(truth,score)
        keep=tau>-1e29; valid=np.r_[pt>-1e29,True]
        axs[0].plot(fpr[keep],tpr[keep],label=LABELS[name]); axs[1].plot(r[valid],p[valid],label=LABELS[name])
    cond=next(r for r in summary if r['method']=='conditional_text')
    axs[0].scatter([cond['fpr']],[cond['tpr']],marker='*',s=130,color='black',label='Условный фильтр: фиксированная точка')
    # Curves pool all pairs; their operating point must also use micro precision.
    micro_precision=cond['tp']/(cond['tp']+cond['fp']) if cond['tp']+cond['fp'] else 0.
    axs[1].scatter([cond['tpr']],[micro_precision],marker='*',s=130,color='black')
    axs[0].set(xlabel='Принятие чужого ID',ylabel='Полнота',title='ROC'); axs[1].set(xlabel='Полнота',ylabel='Точность',title='Точность и полнота по всем парам')
    axs[0].legend(fontsize=8,loc='lower right'); finish(fig,'roc_pr')
    fig,axs=plt.subplots(1,2,figsize=(8.5,3.6))
    for ax,name in zip(axs,[primary,'conditional_text']):
        r=next(r for r in summary if r['method']==name); matrix=np.array([[r['tn'],r['fp']],[r['fn'],r['tp']]])
        ax.imshow(matrix,cmap='Blues'); ax.set_xticks([0,1],['Отказ','Принятие']); ax.set_yticks([0,1],['Другой','Цель'])
        ax.set_title(LABELS[name],fontsize=11); ax.set_xlabel('Решение'); ax.set_ylabel('Разметка')
        for (a,b),value in np.ndenumerate(matrix): ax.text(b,a,str(value),ha='center',va='center',color='white' if value>matrix.max()/2 else 'black')
    finish(fig,'confusion')
    fig,axs=plt.subplots(1,3,figsize=(11,3.6))
    for ax,name in zip(axs,['text_cos','text_raw_full','text_hard_full']):
        for label,title in [(True,'Тот же экземпляр'),(False,'Другой экземпляр')]:
            distances=[r['distance'] for r in records if r['method']==name and r['same']==label and r['distance']<1e29]
            ax.hist(distances,bins=15,density=True,alpha=.5,label=title)
        ax.set(title=LABELS[name],xlabel='Расстояние',ylabel='Плотность')
    axs[0].legend(fontsize=8); finish(fig,'distances')
    fig,ax=plt.subplots(figsize=(8,3.7))
    for modality,label in [('visual','Визуальный MLP'),('text','Текстовый MLP')]:
        history=load(modality+'_training.json')['epochs']; ax.plot([r['epoch'] for r in history],[r['loss'] for r in history],label=label)
    ax.set(xlabel='Эпоха',ylabel='Потеря на трудных обучающих тройках'); ax.legend(); finish(fig,'training')
    fig,ax=plt.subplots(figsize=(11,3.7)); cats=load('test_per_category.json')
    a=[r for r in cats if r['method']==primary]; b=[r for r in cats if r['method']=='conditional_text']
    categories={'ball':'Мяч','bell_pepper':'Перец','dry_battery':'Батарейка','food_bag':'Пакет продуктов','food_box':'Коробка продуктов','food_can':'Консервы','food_jar':'Банка продуктов','garlic':'Чеснок','glue_stick':'Клей','instant_noodles':'Лапша','lemon':'Лимон','marker':'Маркер'}
    ax.bar([categories[r['category']] for r in a],[(x['fpr']-y['fpr'])*100 for x,y in zip(a,b)])
    ax.set(ylabel='Снижение ложных принятий, п.п.'); ax.tick_params(axis='x',rotation=45); finish(fig,'per_category')
    scenes=json.loads((ROOT/'experiments/evidence/instance_scenes_v3/predictions.json').read_text())
    fig,axs=plt.subplots(1,2,figsize=(10,3.8))
    for name,label in [('visual','Визуальный метод'),('conditional_text','С условной проверкой')]:
        rs=[r for r in scenes if r['method']==name and r['present']]; thresholds=np.linspace(.05,1,20)
        axs[0].plot(thresholds,[np.mean([r['iou']>=t for r in rs]) for t in thresholds],label=label)
    axs[0].set(xlabel='Порог IoU',ylabel='Полнота локализации'); axs[0].legend(fontsize=9)
    result=json.loads((ROOT/'experiments/evidence/instance_scenes_v3/summary.json').read_text())
    for i,key in enumerate(['tp','fp','fn','tn']):
        axs[1].bar(np.arange(2)+(i-1.5)*.18,[r[key] for r in result],width=.18,label=key.upper())
    axs[1].set_xticks([0,1],['Визуальный','Условный текст']); axs[1].set_ylabel('Число решений'); axs[1].legend(ncol=4,fontsize=9)
    finish(fig,'scene_results')
    fig,ax=plt.subplots(figsize=(10,1.1)); ax.axis('off')
    ax.text(.5,.5,r'$d_M(a,b)=\sqrt{(a-b)^{\mathsf{T}}\Sigma^{-1}(a-b)},\quad\Sigma=\widehat\Sigma_{LW}+10^{-4}I$',ha='center',va='center',fontsize=16)
    finish(fig,'mahalanobis_formula')
    stages=[('caption','Описания 360 вырезок'),('features','Извлечение признаков'),('train','Обучение трёх MLP'),('calibrate','Выбор порогов'),('evaluate','Оценка пар'),('scene_proposals','Рамки сцен'),('scene_visual','Сопоставление сцен'),('scene_describe','Описания сцен')]
    fig,ax=plt.subplots(figsize=(9,4))
    ax.barh([label for _,label in stages],[load(name+'_execution.json')['wall_sec'] for name,_ in stages])
    ax.set(xlabel='Полное время стадии на CPU, с',title='Затраты выполненного эксперимента')
    finish(fig,'stage_latency')
    print('Russian figures saved',flush=True)


if __name__=='__main__': main()
