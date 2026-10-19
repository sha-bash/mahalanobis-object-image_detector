"""Create an executed notebook from existing measurements, without additional model calls."""
import contextlib
import io
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    cells=[]; namespace={}
    def markdown(text): cells.append(dict(cell_type='markdown',metadata={},source=text.splitlines(True)))
    def code(text):
        output=io.StringIO()
        with contextlib.redirect_stdout(output): exec(compile(text,'saved-results-cell','exec'),namespace)
        cells.append(dict(cell_type='code',metadata={},source=text.splitlines(True),execution_count=1+sum(r['cell_type']=='code' for r in cells),
            outputs=[dict(output_type='stream',name='stdout',text=output.getvalue().splitlines(True))]))
    markdown('# Диагностика и новая проверка поиска экземпляра\n\nЯчейки выполнены при сборке по сохранённым решениям. Модели здесь не обучаются, тестовые пороги не меняются. Основной протокол: `INSTANCE_V3_PROTOCOL.md`.')
    code("import json\nfrom pathlib import Path\nroot=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'experiments/evidence/instance_v3/test_summary.json').exists())\nbase=root/'experiments/evidence/instance_v3'\nread=lambda name:json.loads((base/name).read_text(encoding='utf-8'))\nfor r in read('diagnostics.json')['summary']:\n    if r['split']=='test': print(r)")
    markdown('![Причины прежних ошибок](../evidence/instance_v3/error_decomposition.png)\n\nПрежний тест теперь относится к разработке. Новые оценки ниже получены на Washington RGB-D.')
    code("frozen=read('frozen_thresholds.json')\nprint('Ориентир полноты:',frozen['recall_budget'])\nprint('Выбранный визуальный метод:',frozen['selected_visual'])\nprint('Условная политика:',frozen['gate'])\nfor r in read('test_summary.json'): print(r)")
    markdown('![Рабочие точки](../evidence/instance_v3/operating_points.png)\n\n![ROC и PR](../evidence/instance_v3/roc_pr.png)\n\n![Матрицы ошибок](../evidence/instance_v3/confusion.png)')
    code("print(json.dumps(read('statistics.json'),ensure_ascii=False,indent=2))")
    markdown('![Расстояния](../evidence/instance_v3/distances.png)\n\n![Обучение](../evidence/instance_v3/training.png)\n\n![Эффекты по категориям](../evidence/instance_v3/per_category.png)')
    code("scene=root/'experiments/evidence/instance_scenes_v3'\nfor r in json.loads((scene/'summary.json').read_text()): print(r)\nprint(read('verification.json'))")
    markdown('![Локализация и отказы](../evidence/instance_v3/scene_results.png)\n\nСценовая часть: 4 ID, 2 сцены, 48 решений на 30 изображениях. Отсутствие относится к видимой аннотации. Это описательная проверка переноса, не достаточное основание общего статистического вывода. Пропуски имеют IoU=0. Новизна и ограничения подробно изложены в отчёте.')
    notebook=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python',version='3.11')),nbformat=4,nbformat_minor=4)
    path=ROOT/'experiments/notebooks/instance_v3_analysis.ipynb'
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=2),encoding='utf-8'); print(path)


if __name__=='__main__': main()
