"""Scientific figures of frozen v4 results, with Russian labels."""
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
BASE=ROOT/'experiments/evidence/instance_v4'; VIDEO=ROOT/'experiments/evidence/davis_v4'
LABELS={'visual_baseline':'Исходный визуальный','visual_strict':'Строгий визуальный','gate_text_s2027_full':'Текстовый MLP + Махаланобис',
        'gate_text_raw_cos':'Текст + косинус','gate_text_raw_full':'Текст + Махаланобис',
        'gate_text_s2027_cos':'Текстовый MLP + косинус','gate_text_s2028_cos':'MLP 2028 + косинус','gate_text_s2028_full':'MLP 2028 + Махаланобис',
        'gate_text_s2029_cos':'MLP 2029 + косинус','gate_text_s2029_full':'MLP 2029 + Махаланобис'}
PRIMARY=['visual_baseline','visual_strict','gate_text_s2027_full']


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def finish(fig,name):
    fig.tight_layout();fig.savefig(BASE/(name+'.png'),dpi=170,bbox_inches='tight');plt.close(fig)


def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
    rows=read(BASE/'test_summary.json'); mainrows={r['method']:r for r in rows if r['delta']==.02}
    (BASE/'labels.json').write_text(json.dumps(LABELS,ensure_ascii=False,indent=2),encoding='utf-8')
    fig,axs=plt.subplots(1,2,figsize=(11,4.3))
    y=np.arange(3)
    for i,(key,label) in enumerate([('tpr','Полнота'),('fpr','Принятие другого экземпляра')]):
        axs[0].barh(y+(i-.5)*.3,[100*mainrows[n][key] for n in PRIMARY],height=.3,label=label)
    axs[0].set_yticks(y,[LABELS[n] for n in PRIMARY]);axs[0].set_xlabel('Проценты');axs[0].legend(fontsize=8)
    axs[1].bar(y,[5*mainrows[n]['fp']+mainrows[n]['fn'] for n in PRIMARY]);axs[1].set_xticks(y,['Исходный','Строгий','С текстом'])
    axs[1].set_ylabel('Взвешенная сумма ошибок 5FP + FN');finish(fig,'primary_results')
    fig,axs=plt.subplots(1,3,figsize=(10.5,3.5))
    for ax,name in zip(axs,PRIMARY):
        r=mainrows[name]; m=np.array([[r['tn'],r['fp']],[r['fn'],r['tp']]])
        ax.imshow(m,cmap='Blues');ax.set_xticks([0,1],['Отказ','Принятие']);ax.set_yticks([0,1],['Другой','Цель']);ax.set_title(LABELS[name],fontsize=9)
        for (i,j),value in np.ndenumerate(m):ax.text(j,i,str(value),ha='center',va='center',color='white' if value>m.max()/2 else 'black')
    finish(fig,'confusion')
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for name in PRIMARY:
        rr=sorted([r for r in rows if r['method']==name],key=lambda r:r['delta'])
        axs[0].plot([r['delta']*100 for r in rr],[r['fpr']*100 for r in rr],'o-',label=LABELS[name])
        axs[1].plot([r['delta']*100 for r in rr],[r['tpr']*100 for r in rr],'o-')
    axs[0].set(xlabel='Допуск потери полноты на валидации, п.п.',ylabel='FPR на новом тесте, %');axs[0].legend(fontsize=8)
    axs[1].set(xlabel='Допуск потери полноты на валидации, п.п.',ylabel='Полнота на новом тесте, %');finish(fig,'recall_tolerance')
    fig,ax=plt.subplots(figsize=(8,3.8))
    for seed in [2027,2028,2029]:
        h=read(BASE/f'training_s{seed}.json')['epochs'];ax.plot([r['epoch'] for r in h],[r['loss'] for r in h],label=f'Инициализация {seed}')
    ax.set(xlabel='Эпоха',ylabel='Обучающая потеря');ax.legend();finish(fig,'training')
    from instance_experiment_v3 import pairs,vectors
    manifest=read(BASE/'manifest.json')['crops']; pairs_test=pairs(manifest,'test');matrix=np.load(ROOT/'experiments/local/instance_v4/distances.npz')
    from sklearn.metrics import roc_curve,precision_recall_curve
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for key,label in [('visual_cos','Визуальный косинус'),('text_raw_cos','Текстовый косинус'),('text_s2027_full','MLP текста + Махаланобис')]:
        v,y,_=vectors(pairs_test,matrix[key]);fpr,tpr,tau=roc_curve(y,-v);p,r,pt=precision_recall_curve(y,-v)
        keep=tau>-1e29; valid=np.r_[pt>-1e29,True]
        axs[0].plot(fpr[keep],tpr[keep],label=label);axs[1].plot(r[valid],p[valid],label=label)
    for name in PRIMARY:
        r=mainrows[name];axs[0].scatter(r['fpr'],r['tpr'],marker='x',s=45)
        axs[1].scatter(r['tpr'],r['tp']/max(1,r['tp']+r['fp']),marker='x',s=45)
    axs[0].set(xlabel='FPR',ylabel='Полнота',title='ROC');axs[1].set(xlabel='Полнота',ylabel='Точность по всем парам',title='PR');axs[0].legend(fontsize=8);finish(fig,'roc_pr')
    fig,axs=plt.subplots(1,2,figsize=(9,3.6))
    for ax,key,label in zip(axs,['text_raw_full','text_s2027_full'],['Исходный текст','Текст после MLP']):
        v,y,_=vectors(pairs_test,matrix[key])
        for truth,title in [(True,'Тот же экземпляр'),(False,'Другой экземпляр')]:ax.hist(v[(y==truth)&(v<1e29)],bins=15,density=True,alpha=.5,label=title)
        ax.set(title=label,xlabel='Расстояние Махаланобиса',ylabel='Плотность')
    axs[0].legend(fontsize=8);finish(fig,'distances')
    vr=read(VIDEO/'summary.json');vs={r['method']:r for r in vr if r['subset']=='main_eval'}
    fig,axs=plt.subplots(1,2,figsize=(10,3.8))
    for ax,key,title in zip(axs,['recall','mean_iou'],['Полнота локализации','Средний IoU с учётом пропусков']):
        ax.bar(np.arange(3),[vs[n]['macro'][key] for n in PRIMARY]);ax.set_xticks(np.arange(3),['Исходный','Строгий','С текстом']);ax.set_ylabel(title);ax.set_ylim(0,1)
    finish(fig,'video_results')
    stages=[(BASE/'caption_execution.json','Описания RGB-D'),(BASE/'features_execution.json','Признаки RGB-D'),
            (BASE/'train_execution.json','Три MLP'),(BASE/'calibrate_execution.json','Выбор порогов'),
            (VIDEO/'proposals_execution.json','Кандидаты видео'),(VIDEO/'visual_execution.json','Признаки видео'),(VIDEO/'describe_execution.json','Описания видео')]
    fig,ax=plt.subplots(figsize=(9,4));ax.barh([label for _,label in stages],[read(path)['wall_sec'] for path,_ in stages]);ax.set_xlabel('Время стадии на CPU, с');finish(fig,'latency')
    # t-SNE is a descriptive projection, never a selection or significance criterion.
    from sklearn.manifold import TSNE
    features=np.load(BASE/'features.npz');idx=[i for i,r in enumerate(manifest) if r['split']=='test'];categories=sorted({manifest[i]['category'] for i in idx})
    fig,axs=plt.subplots(1,2,figsize=(9,4))
    for ax,key,title in zip(axs,['visual','text'],['Визуальные признаки','Текстовые признаки']):
        selected=[i for i in idx if key!='text' or features['valid_text'][i]]
        color=[categories.index(manifest[i]['category']) for i in selected]
        if len(selected)<=15:
            ax.text(.1,.5,'Недостаточно валидных описаний');continue
        z=TSNE(n_components=2,perplexity=15,max_iter=750,random_state=2027,init='pca',learning_rate='auto').fit_transform(features[key][selected])
        ax.scatter(z[:,0],z[:,1],c=color,cmap='tab20',s=17);ax.set(title=title,xlabel='Координата t-SNE 1',ylabel='Координата t-SNE 2')
    fig.suptitle('Цвет обозначает категорию; это не оценка качества',fontsize=10);finish(fig,'tsne')
    percat=read(BASE/'test_per_category.json')
    cats=sorted({r['category'] for r in percat})
    ru={'apple':'Яблоко','banana':'Банан','cell_phone':'Телефон','comb':'Расчёска','food_can':'Консервы','food_cup':'Стаканчик',
        'hand_towel':'Полотенце','keyboard':'Клавиатура','lightbulb':'Лампочка','lime':'Лайм','onion':'Лук','orange':'Апельсин',
        'plate':'Тарелка','pliers':'Плоскогубцы','rubber_eraser':'Ластик','scissors':'Ножницы','sponge':'Губка','stapler':'Степлер',
        'toothbrush':'Зубная щётка','water_bottle':'Бутылка'}
    lookup={(r['category'],r['method']):r for r in percat if r['delta']==.02}
    gains=[lookup[(c,'visual_strict')]['risk']-lookup[(c,'gate_text_s2027_full')]['risk'] for c in cats]
    fig,ax=plt.subplots(figsize=(11,4));ax.bar([ru[c] for c in cats],gains);ax.axhline(0,color='black',lw=.8)
    ax.set_ylabel('Снижение риска относительно строгого контроля');ax.tick_params(axis='x',rotation=60);finish(fig,'per_category')
    fig,ax=plt.subplots(figsize=(7,3.8));sensitivity=read(BASE/'cost_sensitivity.json')
    for name in PRIMARY:
        rr=sorted([r for r in sensitivity if r['method']==name and r['prevalence']==.5],key=lambda r:r['fp_cost'])
        ax.plot([r['fp_cost'] for r in rr],[r['risk'] for r in rr],'o-',label=LABELS[name])
    ax.set(xlabel='Относительная цена ложного принятия',ylabel='Риск на пару при доле цели 50%');ax.legend(fontsize=8);finish(fig,'cost_sensitivity')
    print('Russian v4 figures saved',flush=True)


if __name__=='__main__': main()
