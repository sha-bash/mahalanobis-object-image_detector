"""Publication figures and equations from saved experimental records."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.mathtext import math_to_image
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiments/evidence/multimodal_v1'


def main():
    data=json.loads((BASE/'results.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(9,3.8),sharey=True)
    names=['raw_cos','pca_cos','diag','full']
    for ax,k in zip(axes,[1,3]):
        for j,(mod,color) in enumerate([('visual','#2775a3'),('text','#bd6b25'),('fusion','#428253')]):
            values=[next(r['balanced_accuracy'] for r in data['macro'] if r['scope']=='real' and r['method']==mod+'_'+m and r['k']==k) for m in names]
            ax.bar(np.arange(4)+(j-1)*.24,values,width=.23,label={'visual':'Изображение','text':'Текст','fusion':'Изображение + текст'}[mod],color=color)
        ax.axhline(.5,color='#777777',linestyle=':',linewidth=.8)
        ax.set_xticks(range(4),['Косинус','PCA + косинус','Диаг. Мах.','Полный Мах.'])
        ax.set(title=f'Референсов: {k}',ylim=(0,1.03),xlabel='Проекция признаков и метрика')
    axes[0].set_ylabel('Сбалансированная точность, среднее 3 сцен')
    axes[1].legend(loc='lower right',fontsize=8)
    fig.tight_layout(rect=[0,.08,1,1])
    fig.text(.5,.015,'Фиксированные разбиения; пороги по валидации; вырезки из разметки; без доверительных интервалов.',ha='center',fontsize=7)
    fig.savefig(BASE/'balanced_accuracy.png',dpi=220)
    plt.close(fig)
    methods=['visual_raw_cos','text_raw_cos','fusion_raw_cos','visual_full','text_full','fusion_full']
    fig,axes=plt.subplots(1,2,figsize=(9,3.8),sharex=True,sharey=True)
    for ax,k in zip(axes,[1,3]):
        for name,color in zip(methods,['#2775a3','#bd6b25','#428253','#063d6a','#743000','#154925']):
            rs=[r for r in data['rows'] if r['test']!='car-3' and r['method']==name and r['k']==k]
            ax.scatter([r['fpr'] for r in rs],[r['tpr'] for r in rs],s=24,alpha=.65,color=color)
            mean=next(r for r in data['macro'] if r['scope']=='real' and r['method']==name and r['k']==k)
            mod=name.split('_')[0]
            label={'visual':'Изображение','text':'Текст','fusion':'Совместно'}[mod]+(' / косинус' if name.endswith('cos') else ' / Махаланобис')
            ax.scatter(mean['fpr'],mean['tpr'],marker='X',s=85,color=color,label=label)
        ax.axhline(.7,color='gray',linestyle=':',linewidth=.8)
        ax.set(title=f'Референсов: {k}',xlabel='Доля принятых чужих пар',xlim=(-.03,1.03),ylim=(-.03,1.05))
    axes[0].set_ylabel('Доля принятых правильных пар')
    axes[1].legend(fontsize=6,loc='lower right')
    fig.tight_layout(rect=[0,.08,1,1])
    fig.text(.5,.015,'Круги — сцены; кресты — среднее по сценам. Рабочие пороги выбраны на других, валидационных сценах.',ha='center',fontsize=7)
    fig.savefig(BASE/'operating_points.png',dpi=220)
    plt.close(fig)
    equations=[
        r'$v=\frac{f_V(x)}{\|f_V(x)\|_2},\quad t=\frac{f_T(g(x))}{\|f_T(g(x))\|_2},\quad z=\frac{1}{\sqrt{2}}[v;t]$',
        r'$u(x)=\frac{P_{\rm train}(z(x)-\mu_{\rm train})}{\|P_{\rm train}(z(x)-\mu_{\rm train})\|_2}$',
        r'$\widehat{\Sigma}=(1-\lambda)S+\lambda\,\frac{\mathrm{tr}(S)}{d}I+10^{-4}I,\quad d\leq8$',
        r'$D(q,c)=\min_{1\leq j\leq K}\sqrt{(u(r_j)-u(c))^T\widehat{\Sigma}^{-1}(u(r_j)-u(c))}$',
        r'$\mathrm{TPR}=\frac{TP}{TP+FN},\quad\mathrm{FPR}=\frac{FP}{FP+TN},\quad\mathrm{BA}=\frac{\mathrm{TPR}+1-\mathrm{FPR}}{2}$',
        r'$d^2_{0.01I}(a,b)=200(1-\cos(a,b)),\qquad \|a\|_2=\|b\|_2=1$'
    ]
    for i,eq in enumerate(equations,1):
        math_to_image(eq,str(BASE/f'equation_{i}.png'),dpi=240,format='png')
    captions=json.loads((BASE/'captions.json').read_text(encoding='utf-8'))
    manifest=json.loads((BASE/'manifest.json').read_text())['crops']
    metadata={r['key']:r for r in manifest}
    groups={}
    for row in captions['records']:
        groups.setdefault(row['caption'].casefold(),[]).append(row['key'])
    collisions=[dict(caption=text,keys=keys,identities=sorted({metadata[k]['identity'] for k in keys})) for text,keys in groups.items() if len({metadata[k]['identity'] for k in keys})>1]
    (BASE/'caption_collisions.json').write_text(json.dumps(collisions,indent=2,ensure_ascii=False),encoding='utf-8')


if __name__=='__main__':
    main()
