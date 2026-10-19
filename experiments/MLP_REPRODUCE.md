# Воспроизведение расширения MLP

Рабочая ветка `codex/itnt-2027-audit`. Старые статьи, исходное видео, исходная
реализация и предыдущие эксперименты сохранены. Новые данные и веса локальны,
их нельзя получить только клонированием Git. Новые результаты не заменяют старые.

## Входы

- Официальные архивы GMOT-40 в `experiments/local/gmot40/`:
  `GenericMOT_JPEG_Sequence.zip` SHA256
  `243d3c4c9c22f0f244df9e5fd13a57d192d2a8d194fee4614ff4e329601ca507`
  и `track_label.zip`; оригинальные URL и контрольные суммы
  `experiments/evidence/gmot40/inventory.json`. Условия CC BY-NC-SA 4.0.
- Локальные DINOv2-small revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`
  и all-MiniLM-L6-v2 revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
  Код читает стандартный Hugging Face cache. Веса/конфиги хешируются в provenance.
- Ollama и Qwen3.5:0.8b: локальная установка через `run.py`, CPU, порт11435.
  Digest модели фиксируется в captions.json. Нельзя заменять модель по одному имени
  без проверки digest при воспроизведении.
- YOLOv8n: `yolov8n.pt` в корне, SHA256 сохраняется в full-image proposals.json.
  Вес не коммитится. Предобучение COCO, ограничения категорий явно учитываются.

## Окружение

`requirements/mlp-cpu-lock.txt` фиксирует реально установленную ML-среду Windows
Python3.11.8. Используйте отдельную среду и CPU index для torch/torchvision:

```powershell
py -3.11 -m venv .venv-mlp
.venv-mlp/Scripts/python.exe -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements/mlp-cpu-lock.txt
.venv-mlp/Scripts/python.exe -m pip install --no-deps -e .
.venv-mlp/Scripts/python.exe -m pip check
```

Снимок пакетов не гарантирует побитовую переносимость на другую ОС/CPU. Нужны
те же веса, версии и входные контрольные суммы. В предоставленном запуске
`.venv` выполняет ML, `.audit-venv` запускает монитор с psutil. Путь Python в
run_mlp_suite.py задаёт эти среды; для другой машины измените только пути сред.

## Последовательность

```powershell
.venv/Scripts/python.exe scripts/mlp_data.py prepare
.venv/Scripts/python.exe scripts/mlp_data.py describe
.venv/Scripts/python.exe scripts/full_image_mlp.py prepare
.audit-venv/Scripts/python.exe scripts/run_mlp_suite.py
```

Описание сохраняется после каждого изображения и возобновляется той же командой.
Suite последовательно выполняет features, train, calibrate, evaluate, plots,
полные кадры proposals, visual, describe, text, evaluate. Завершённые стадии с
returncode0 пропускаются. Для намеренного нового опыта создавайте новый каталог
и протокол; не переписывайте frozen_thresholds.json после просмотра теста.
Для точечной диагностики доступны те же команды отдельно:

```powershell
.venv/Scripts/python.exe scripts/learned_experiment.py train
.venv/Scripts/python.exe scripts/learned_experiment.py calibrate
.venv/Scripts/python.exe scripts/learned_experiment.py evaluate
.venv/Scripts/python.exe scripts/plot_learned.py
.venv/Scripts/python.exe -m pytest -q
```

Состояние, ошибки и ресурсы: `experiments/evidence/mlp_v2/*_execution.json`,
одноимённые `.log` (логи локальны). Артефакты данных: manifest/captions/features,
ковариации, frozen_thresholds, test_pairs, test_per_video, test_summary, statistics.
Веса шести MLP в `experiments/local/mlp_v2/*mlp*.pt` и конфигурации JSON рядом;
SHA256 в *_training.json. Проектор загружается через `mcd.projection.runtime`.
Для общей фабрики `projector_path` указывает JSON, а не старый NPZ/PCA.

## Проверка сохранённых результатов и сборка материалов

Следующие команды не запускают VLM повторно:

```powershell
.venv/Scripts/python.exe scripts/verify_learned.py
.venv/Scripts/python.exe scripts/analyze_learned_errors.py
.venv/Scripts/python.exe scripts/plot_learned.py
.venv/Scripts/python.exe scripts/plot_full_mlp.py
.venv/Scripts/python.exe scripts/build_learned_notebook.py
```

Для `scripts/build_learned_report.py` нужен Python с `python-docx`, `reportlab`,
`pypdf`, `Pillow`. В данном запуске использована комплектная среда документов
Codex; на своей машине установите эти пакеты в отдельную среду и выполните
`python scripts/build_learned_report.py`. PDF и DOCX сохраняются в
`output/research_20261011_mlp/`. Для визуальной проверки DOCX нужны LibreOffice
и Poppler. Ноутбук читает сохранённые результаты, без повторного обучения.

В первоначальной сборке графика исправлен синтаксис математической подписи;
повторный запуск затронул только рисунки. При сохранении PNG из DOCX возникла
временная блокировка имени файла Windows; готовые страницы переименованы и
проверены. Эти операции не меняли предсказания и рабочие пороги.

## Интерпретация результатов

test_summary относится к готовым размеченным вырезкам: пары одного/разных ID.
Только `experiments/evidence/mlp_full_image/predictions.json` оценивает рамки на
целых изображениях и IoU. Кадры присутствия не измеряют FPR отсутствия цели.
Предыдущие video ablations находятся в heldout_v1; новый MLP не был использован
в тех прогонах. Их нельзя переименовать в результаты нового метода.

Оригинальная лицензия авторского MLP сохранена отдельно в
`src/mcd/projection/LICENSE.upstream`. Код и модели Ultralytics имеют собственные
условия; пакеты устанавливаются отдельно, веса не распространяются этим Git.
