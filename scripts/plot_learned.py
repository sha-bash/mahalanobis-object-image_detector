"""Russian scientific figures from saved, frozen MLP experiment outputs."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
BASE=ROOT/'experiments/evidence/mlp_v2'
LABELS={'sift':'SIFT','visual_cos':'Визуальный косинус','text_cos':'Текстовый косинус','text_raw_diagonal':'Текстовый диагональный М.','text_raw_full':'Текстовый полный М.','text_mlp_s2027_cos':'MLP текста + косинус','text_mlp_s2027_diagonal':'MLP текста + диаг. М.','text_mlp_s2027_full':'MLP текста + полный М.','visual_mlp_s2027_cos':'Сиамский MLP + косинус','visual_mlp_s2027_full':'Сиамский MLP + полный М.','fusion_s2027':'Совместная метрика'}
MAIN=['sift','visual_cos','text_cos','text_raw_full','text_mlp_s2027_cos','text_mlp_s2027_full','visual_mlp_s2027_cos','fusion_s2027']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'axes.spines.top':False,'axes.spines.right':False})


def read(name):
    return json.loads((BASE/name).read_text(encoding='utf-8'))


def finish(fig,name):
    fig.tight_layout()
    fig.savefig(BASE/f'{name}.png',dpi=180,bbox_inches='tight')
    fig.savefig(BASE/f'{name}.svg',bbox_inches='tight')
    plt.close(fig)


def main():
    from sklearn.metrics import roc_curve,precision_recall_curve
    summary=read('test_summary.json'); pairs=read('test_pairs.json')
    fig,axs=plt.subplots(1,2,figsize=(11,5.2))
    for name in MAIN:
        rows=[r for r in pairs if r['method']==name and r['k']==1]
        y=np.array([r['same'] for r in rows]); s=-np.array([r['distance'] for r in rows])
        fpr,tpr,rt=roc_curve(y,s); precision,recall,pt=precision_recall_curve(y,s)
        keep=rt>-1e29; kp=np.r_[pt>-1e29,True]
        axs[0].plot(fpr[keep],tpr[keep],label=LABELS[name]); axs[1].plot(recall[kp],precision[kp],label=LABELS[name])
    axs[0].plot([0,1],[0,1],'k:',lw=.7)
    axs[0].set(xlabel='Доля принятых чужих экземпляров',ylabel='Полнота',title='ROC по парам на тесте')
    axs[1].set(xlabel='Полнота',ylabel='Точность',title='Кривая точность-полнота')
    axs[1].legend(fontsize=10,loc='best')
    finish(fig,'roc_pr')
    fig,axs=plt.subplots(1,2,figsize=(11,4.4))
    x=np.arange(len(MAIN))
    for k,shift in [(1,-.18),(3,.18)]:
        rows=[next(r for r in summary if r['method']==m and r['k']==k) for m in MAIN]
        for ax,key in zip(axs,['tpr','fpr']):
            ax.bar(x+shift,[100*r[key] for r in rows],width=.36,label=f'{k} эталон(а)')
    for ax,key in zip(axs,['tpr','fpr']):
        ax.set_xticks(x,[LABELS[m] for m in MAIN],rotation=45,ha='right',fontsize=10)
        ax.set_ylabel('Проценты'); ax.set_title('Полнота' if key=='tpr' else 'Принятие чужого экземпляра'); ax.legend()
    budget=read('frozen_thresholds.json')['budgets']['1']['recall']
    axs[0].axhline(100*budget,ls=':',color='black',label='Ориентир валидации K=1')
    finish(fig,'operating_points')
    fig,axs=plt.subplots(1,3,figsize=(11,3.6))
    for ax,name in zip(axs,['visual_cos','text_mlp_s2027_full','fusion_s2027']):
        r=next(r for r in summary if r['method']==name and r['k']==1)
        matrix=np.array([[r['tn'],r['fp']],[r['fn'],r['tp']]])
        ax.imshow(matrix,cmap='Blues')
        for (i,j),n in np.ndenumerate(matrix):
            ax.text(j,i,str(n),ha='center',va='center',color='white' if n>matrix.max()*.6 else 'black')
        ax.set_xticks([0,1],['Отказ','Принятие']); ax.set_yticks([0,1],['Другой','Цель'])
        ax.set_title(LABELS[name]); ax.set_xlabel('Решение'); ax.set_ylabel('Разметка')
    finish(fig,'confusion')
    fig,axs=plt.subplots(1,3,figsize=(11,3.8))
    for ax,name in zip(axs,['text_cos','text_raw_full','text_mlp_s2027_full']):
        rows=[r for r in pairs if r['method']==name and r['k']==1]
        for truth,label in [(True,'Тот же экземпляр'),(False,'Другой экземпляр')]:
            ax.hist([r['distance'] for r in rows if r['same']==truth and r['distance']<1e29],bins=20,density=True,alpha=.5,label=label)
        threshold=next(r['threshold'] for r in summary if r['method']==name and r['k']==1)
        ax.axvline(threshold,color='black',ls=':',label='Порог валидации')
        ax.set(xlabel='Расстояние',ylabel='Плотность',title=LABELS[name])
    axs[0].legend(fontsize=10); finish(fig,'distance_distributions')
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    videos=read('test_per_video.json')
    clips=sorted({r['clip'] for r in videos})
    for name in ['visual_cos','text_mlp_s2027_full','fusion_s2027']:
        rows=[next(r for r in videos if r['method']==name and r['k']==1 and r['clip']==c) for c in clips]
        for ax,key in zip(axs,['tpr','fpr']):
            ax.plot(clips,[r[key] for r in rows],'-o',ms=3,label=LABELS[name])
    for ax,key in zip(axs,['tpr','fpr']):
        ax.tick_params(axis='x',rotation=70); ax.set_ylabel('Полнота' if key=='tpr' else 'Доля ложных принятий'); ax.legend(fontsize=10)
    finish(fig,'per_video')
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for modality,ax in zip(['text','visual'],axs):
        for seed in [2027,2028,2029]:
            h=read(f'{modality}_mlp_s{seed}_training.json')['epochs']
            ax.plot([r['epoch'] for r in h],[r['loss'] for r in h],label=f'Инициализация {seed}')
        ax.set(xlabel='Эпоха',ylabel='Триплетная потеря',title='Текстовый MLP' if modality=='text' else 'Визуальный MLP'); ax.legend(fontsize=10)
    finish(fig,'training')
    fig,ax=plt.subplots(figsize=(8,4))
    for name in MAIN[1:]:
        rows=[r for r in pairs if r['method']==name and r['k']==1]
        y=np.array([r['same'] for r in rows]); s=-np.array([r['distance'] for r in rows])
        p,r,thresholds=precision_recall_curve(y,s)
        keep=np.r_[thresholds>-1e29,True]; p,r=p[keep],r[keep]
        f1=2*p*r/np.maximum(p+r,1e-12)
        ax.plot(r,f1,label=LABELS[name])
    ax.set(xlabel='Полнота при изменении порога',ylabel='F1',title='Кривые на тесте только для анализа без перенастройки')
    ax.legend(fontsize=10); finish(fig,'f1_curve')
    from sklearn.manifold import TSNE
    manifest=read('manifest.json')['crops']; ids=[i for i,r in enumerate(manifest) if r['split']=='test']
    features=np.load(BASE/'features.npz'); projected=np.load(BASE/'projections.npz')
    categories=sorted({r['category'] for r in manifest})
    fig,axs=plt.subplots(1,2,figsize=(11,4.5))
    for ax,x,title in [(axs[0],features['text'],'Исходные текстовые признаки'),(axs[1],projected['text_mlp_s2027'],'После обученного MLP')]:
        y=TSNE(n_components=2,perplexity=20,random_state=2027,init='pca',learning_rate='auto').fit_transform(x[ids])
        for category in categories:
            mask=np.array([manifest[i]['category']==category for i in ids])
            labels_ru={'ball':'Мячи','balloon':'Воздушные шары','bird':'Птицы','fish':'Рыбы','insect':'Насекомые','person':'Люди','stock':'Животные'}
            ax.scatter(y[mask,0],y[mask,1],s=12,alpha=.6,label=labels_ru[category])
        ax.set(title=title,xlabel='Координата t-SNE 1',ylabel='Координата t-SNE 2')
    axs[1].legend(fontsize=10); finish(fig,'tsne')
    timing=read('feature_provenance.json'); captions=read('captions.json')
    fig,ax=plt.subplots(figsize=(8,3.5))
    labels=['VLM описание','DINOv2 признаки','SBERT признаки']
    values=[sum(r['elapsed_sec'] for r in captions['records'])/560,timing['visual_sec']/560,timing['text_sec']/560]
    ax.barh(labels,values); ax.set_xscale('log'); ax.set_xlabel('Секунды на изображение среднее по стадии')
    for i,v in enumerate(values): ax.text(v,i,f' {v:.3f}',va='center')
    finish(fig,'latency')
    equations=[r'$z(x)=f_{\theta}(e(x))/\|f_{\theta}(e(x))\|_2$',r'$\mathcal{L}=\max(0,\|f(a)-f(p)\|_2-\|f(a)-f(n)\|_2+1)$',r'$d_C(a,b)=1-a^{\mathsf{T}}b,\qquad d_M(a,b)=\sqrt{(a-b)^{\mathsf{T}}\Sigma^{-1}(a-b)}$',r'$d_F=\frac{1}{2}\left(\frac{d_V}{s_V}+\frac{d_T}{s_T}\right)$',r'$R=\frac{TP}{TP+FN},\quad P=\frac{TP}{TP+FP},\quad FPR=\frac{FP}{FP+TN}$']
    for i,equation in enumerate(equations):
        fig,ax=plt.subplots(figsize=(9,.55)); ax.axis('off'); ax.text(.5,.5,equation,ha='center',va='center',fontsize=15)
        finish(fig,f'formula_{i+1}')
    print('Saved nine Russian figure sets',flush=True)


if __name__=='__main__':
    main()
