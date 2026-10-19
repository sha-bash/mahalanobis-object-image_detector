"""Build an analysis notebook with independently computed saved-data outputs."""
import contextlib
import io
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    directory=ROOT/'experiments/notebooks'; directory.mkdir(exist_ok=True)
    cells=[]
    def markdown(text): cells.append(dict(cell_type='markdown',metadata={},source=text.splitlines(True)))
    namespace={}
    def code(text):
        capture=io.StringIO()
        with contextlib.redirect_stdout(capture): exec(compile(text,'notebook-cell','exec'),namespace)
        outputs=[dict(output_type='stream',name='stdout',text=capture.getvalue().splitlines(True))] if capture.getvalue() else []
        cells.append(dict(cell_type='code',metadata={},source=text.splitlines(True),execution_count=sum(c['cell_type']=='code' for c in cells)+1,outputs=outputs))
    markdown('# Анализ обучаемой текстовой метрики\n\nНоутбук читает сохранённые результаты; не переобучает модели и не подбирает пороги по тесту. Ячейки с выводом выполнены последовательно при сборке. Графики получены из тех же файлов. Полный протокол находится в `experiments/MLP_PROTOCOL.md`.')
    code("import json\nfrom pathlib import Path\nroot=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'experiments/evidence/mlp_v2/test_summary.json').exists())\nbase=root/'experiments/evidence/mlp_v2'\nsummary=json.loads((base/'test_summary.json').read_text())\nfrozen=json.loads((base/'frozen_thresholds.json').read_text())\nprint('Ориентиры полноты:', {k:v['recall'] for k,v in frozen['budgets'].items()})")
    markdown('## Сопоставление экземпляров\n\nЕдиница ошибки - пара вырезок. FPR означает принятие другого ID, а не ложную рамку при отсутствии цели.')
    code("methods=['sift','visual_cos','text_cos','text_raw_full','text_mlp_s2027_cos','text_mlp_s2027_full','visual_mlp_s2027_cos','fusion_s2027']\nfor r in summary:\n    if r['k']==1 and r['method'] in methods:\n        print(r['method'], 'R=',round(r['tpr'],4),'FPR=',round(r['fpr'],4),'F1=',round(r['f1'],4))")
    markdown('![ROC и PR](../evidence/mlp_v2/roc_pr.png)\n\n![Рабочие точки](../evidence/mlp_v2/operating_points.png)\n\n![Матрицы ошибок](../evidence/mlp_v2/confusion.png)')
    code("stats=json.loads((base/'statistics.json').read_text())\nprint(json.dumps(stats,ensure_ascii=False,indent=2))")
    markdown('## Расстояния и обучение\n\nМахаланобис и косинус сравниваются на одних признаках. t-SNE служит иллюстрацией, не статистическим подтверждением.\n\n![Расстояния](../evidence/mlp_v2/distance_distributions.png)\n\n![Обучение](../evidence/mlp_v2/training.png)\n\n![t-SNE](../evidence/mlp_v2/tsne.png)')
    code("full=json.loads((root/'experiments/evidence/mlp_full_image/summary.json').read_text())\nfor r in full:\n    if r['split']=='test': print(r)")
    markdown('## Полные изображения и ограничения\n\nЗдесь TP требует IoU ≥0,5, пропуск учитывается. Все выбранные кадры содержат цель. Проверка отсутствия и длительного сопровождения новым MLP не проведена. Полные предсказания: `mlp_full_image/predictions.json`.\n\n## Проверка артефактов')
    code("verification=json.loads((base/'verification.json').read_text())\nprint(json.dumps(verification,ensure_ascii=False,indent=2))")
    notebook=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python',version='3.11')),nbformat=4,nbformat_minor=4)
    path=directory/'mlp_analysis.ipynb'
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=2),encoding='utf-8')
    print(path)


if __name__=='__main__': main()
