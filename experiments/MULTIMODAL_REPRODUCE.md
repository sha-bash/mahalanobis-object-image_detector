# Воспроизведение мультимодального опыта

Протокол: `MULTIMODAL_PROTOCOL.md`, frozen commit81a900e. Данные — официальные
архивы GMOT-40 из `GMOT40_REPRODUCE.md`; исходные пиксели не включены в Git.
Используются `.venv` для ML и `.audit-venv` для графиков. Версии из первого
снимка остаются в `requirements/experiment-win-py311.lock.txt`; новые фактические
версии сохранены в `evidence/multimodal_v1/captions.json`.

```powershell
.audit-venv/Scripts/python.exe scripts/multimodal_data.py prepare
.venv/Scripts/python.exe scripts/multimodal_data.py describe
.venv/Scripts/python.exe scripts/multimodal_data.py features
.venv/Scripts/python.exe scripts/evaluate_multimodal.py
.venv/Scripts/python.exe scripts/verify_multimodal_artifacts.py
.audit-venv/Scripts/python.exe scripts/plot_multimodal.py
.venv/Scripts/python.exe -m pytest -q
```

`prepare` отказывается перезаписывать существующий local/multimodal_v1. Для
нового опыта используйте отдельный checkout или новый путь BASE/EVIDENCE в
скрипте, зафиксировав это в журнале. `describe` умеет продолжить прерванный
запуск: завершённые crops пропускаются; VLM cache хранит только валидные описания.
Это продолжение того же опыта, а не новый независимый повтор.

Загруженные модели используют фиксированные локальные snapshots в стандартном
Hugging Face cache. DINOv2: ed25f3a31f01632728cabb09d1542f84ab7b0056; SBERT:
1110a243fdf4706b3f48f1d95db1a4f5529b4d41. Для другого компьютера скорректировать
только корень cache, сохраняя ревизии. VLM — проектный qwen3.5:0.8b, параметры
в IndependentCaptioner (temperature0, num_ctx2048, num_predict256, CPU4threads).
Пустой scene context, одинаковый prompt для reference/candidate. Шаблон prompt
сохранён в исходниках; модели и входы проверяются по digest/SHA256.

Результаты можно пересчитать без запуска VLM: сохранённый features.npz содержит
120 визуальных и120 текстовых векторов и SIFT score matrix. `evaluate_multimodal.py`
заново обучает PCA/ковариацию только на train fold и выбирает порог только на
validation; `scores.json` содержит все pair scores. `parameters.json` содержит
каждую проекцию и covariance. В игре отдельный fold; реальный macro включает3сцены.

Top1 при равных минимальных score — математическое ожидание случайного выбора
среди tied-кандидатов (доля верных в группе). Это уточнение введено в реализации
до вычисления результатов, чтобы порядок ID не создавал преимущества.

PDF создаются скриптом `build_research_pdfs.py` в bundled document runtime,
с ReportLab и системными Times New Roman. Путь к runtime определяется инструментом
загрузки зависимостей Codex. Для обычного локального запуска нужны reportlab,
pillow, pypdf; рисунки заранее создаются `plot_multimodal.py`. Проверка страниц —
Poppler pdftoppm, проверка количества — pypdf. Выход: output/research_20261010.
Это PDF-вёрстка; соответствие официальному бинарному Word-шаблону не заявляется.
