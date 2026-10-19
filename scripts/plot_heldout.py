"""Russian publication figures for the frozen held-out video evaluation."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parents[1]/'experiments/evidence/heldout_v1'
LABELS = {'geometry':'Без текста', 'isotropic':'Изотропный Махаланобис',
          'text_cosine':'Косинус (PCA)', 'text_full':'Полный Махаланобис (PCA)'}
COLORS = ['#495867', '#bf7930', '#267ca3', '#43805a']


def main():
    summary = json.loads((BASE/'summary.json').read_text())
    frames = json.loads((BASE/'frames.json').read_text())
    scenes = list(dict.fromkeys(r['sequence'] for r in summary['rows']))
    names = [s.replace('airplane-', 'Авиация ').replace('boat-', 'Суда ') for s in scenes]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for ax, metric, title in zip(axes, ['recall', 'mean_iou_present'],
                                ['Полнота при IoU ≥ 0,5', 'Средний IoU с учётом отказов']):
        for j, (variant, label) in enumerate(LABELS.items()):
            values = [next(r[metric] for r in summary['rows'] if r['variant'] == variant and r['sequence'] == s) for s in scenes]
            ax.bar(np.arange(len(scenes))+(j-1.5)*.19, values, width=.18, label=label, color=COLORS[j])
        ax.set(ylim=(0, 1.05), ylabel=title)
        ax.grid(axis='y', alpha=.2)
    axes[0].legend(loc='upper center', bbox_to_anchor=(.5, 1.23), ncol=2, fontsize=9)
    axes[1].set_xticks(range(len(scenes)), names)
    axes[1].set_xlabel('Новые видеопоследовательности; один фиксированный эталон на видео')
    fig.tight_layout(rect=[0, .03, 1, 1])
    fig.text(.5, .012, 'Пропуск даёт IoU = 0. Пороги перенесены со старых данных; тестовые параметры не подбирались.', ha='center', fontsize=8)
    fig.savefig(BASE/'localization_ru.png', dpi=210)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    thresholds = np.arange(.1, 1, .1)
    for j, (variant, label) in enumerate(LABELS.items()):
        values = []
        for tau in thresholds:
            per_scene = []
            for s in scenes:
                scores = [r['iou'] for r in frames if r['variant'] == variant and r['sequence'] == s and r['status'] == 'present']
                per_scene.append(np.mean(np.array(scores) >= tau))
            values.append(np.mean(per_scene))
        ax.plot(thresholds, values, marker=['o', 's', '^', 'x'][j], linestyle=['-', '--', ':', '-.'][j],
                color=COLORS[j], label=label, markersize=5)
    ax.axvline(.5, color='gray', linewidth=.8)
    ax.set(xlabel='Порог IoU для правильной локализации', ylabel='Полнота, среднее по восьми видео', ylim=(0, 1.02), xlim=(.08, .92))
    ax.legend(fontsize=9)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(BASE/'success_iou_ru.png', dpi=210)
    plt.close(fig)


if __name__ == '__main__':
    main()
