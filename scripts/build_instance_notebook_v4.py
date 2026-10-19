"""Write a read-only analysis notebook with outputs executed from saved measurements."""
import contextlib
import io
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    sources=[
        "from pathlib import Path\nimport json\nROOT=Path.cwd()\nif not (ROOT/'experiments').exists(): ROOT=ROOT.parent.parent\nBASE=ROOT/'experiments/evidence/instance_v4'\ndef read(path): return json.loads(path.read_text(encoding='utf-8'))\nprint(read(BASE/'statistics.json'))",
        "for row in read(BASE/'test_summary.json'):\n    if row['delta']==0.02:\n        print(row)",
        "for row in read(ROOT/'experiments/evidence/davis_v4/summary.json'):\n    if row['subset']=='main_eval':\n        print(row)",
        "print(read(BASE/'verification.json'))",
    ]
    cells=[dict(cell_type='markdown',metadata={},source=['# Новый тест при цене ложного принятия 5:1\n','Все пороги выбраны на validation до test. Допуск потери полноты 2 п.п.\n','Главный контроль — ужесточённый визуальный порог. Повторный запуск ячеек только читает результаты.'])]
    namespace={}
    for i,source in enumerate(sources,1):
        stream=io.StringIO()
        with contextlib.redirect_stdout(stream):exec(compile(source,f'cell_{i}','exec'),namespace)
        cells.append(dict(cell_type='code',metadata={},execution_count=i,source=source.splitlines(keepends=True),
                          outputs=[dict(output_type='stream',name='stdout',text=stream.getvalue().splitlines(keepends=True))]))
    data=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python',version='3.11')),nbformat=4,nbformat_minor=5)
    path=ROOT/'experiments/notebooks/instance_v4_analysis.ipynb';path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');print(path)


if __name__=='__main__':main()
