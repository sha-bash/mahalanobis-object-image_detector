# Воспроизведение

Все команды ниже выполняются из корня репозитория в PowerShell. Каждому запуску
нужен **новый** каталог `--out`: существующие результаты не перезаписываются.
Базовый код извлекается из Git `5552bc2`, копируется в каталог запуска и исполняется
как отдельный модуль. Для improved используются текущие `src` с hashes/provenance.
Новые запуски сохраняют также снимок исходников. Метрики не требуют VLM или GPU.

## Окружения

Фактические ML-запуски: исходная `.venv`, Python 3.11.8, CPU. Полный снимок
зависимостей без локальной ссылки на проект — `requirements/experiment-win-py311.lock.txt`.
Для установки в новое окружение (не менять существующее):

```powershell
py -3.11 -m venv .venv-repro
.venv-repro/Scripts/python.exe -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements/experiment-win-py311.lock.txt
.venv-repro/Scripts/python.exe -m pip install --no-deps -e .
```

Точное восстановление нового окружения из этого snapshot ещё не проверено;
это lock конкретной Windows-машины, не универсальный cross-platform lock.
CPU PyTorch требует официального CPU wheel index. Приложенные веса не распространяются.
Для мониторинга и графиков использована отдельная `.audit-venv` Python 3.12.14:

```powershell
py -3.12 -m venv .audit-venv
.audit-venv/Scripts/python.exe -m pip install -r requirements/audit-tools.lock.txt
```

В среде владельца `.venv/Scripts/python.exe` уже работоспособен. В окружении агента
потребовался запуск с разрешением доступа к пользовательскому базовому Python.

## Восстановление данных

Первый query использует `data/search/video_test.mp4`, `data/refs/frame_000000.jpg`,
`annotations/instances.json`; эталонный crop `[41,34,229,207]`. SHA256 см. evidence.
Для второго query сохранить CSV из предоставленной папки Drive в
`experiments/local/drive/` с исходными именами:
`ground_truth_manual.csv`, `DAY_DASIAMRPN_GAZEL.csv`, `DAY_DASIAMRPN_66.csv`.
Идентификаторы файлов и размеры — `evidence/drive_inventory.json`. Импорт connector
сохранял весь CSV-текст с нормализацией CRLF→LF; хеши относятся к локальному LF-файлу.

```powershell
.audit-venv/Scripts/python.exe scripts/prepare_drive_experiment.py
.audit-venv/Scripts/python.exe scripts/audit_evidence.py
```

Первый скрипт создаёт query crop из первого кадра, явную покадровую разметку и
`drive_protocol.json`. CSV и crop остаются вне Git. Не использовать контрольные
JPEG с нарисованными рамками как вход алгоритма. Само имя папки `train` не задаёт
научное разбиение: весь текущий материал — один debug-cluster.

## Полные запуски

```powershell
.audit-venv/Scripts/python.exe scripts/run_matrix.py --python .venv/Scripts/python.exe --out experiments/runs/repro_geometry --variants legacy_geometry foreground_geometry template
.audit-venv/Scripts/python.exe scripts/run_matrix.py --python .venv/Scripts/python.exe --out experiments/runs/repro_semantic --variants legacy_semantic --fresh-captions --managed-server
.audit-venv/Scripts/python.exe scripts/run_matrix.py --python .venv/Scripts/python.exe --out experiments/runs/repro_66 --variants geometry semantic foreground_geometry template --manifest experiments/drive_protocol.json --sequence demo_66 --fresh-captions --managed-server
```

`--managed-server` использует проектный `models/ollama/ollama.exe` и
`models/ollama-models` на 127.0.0.1:11435; останавливает только созданный им сервер.
`--fresh-captions` создаёт новый каталог описаний: VLM действительно исполняется.
Embedding cache может использоваться; status каждого описания сохраняется.
`--cache-only` вместо fresh разрешает только уже сохранённые описания/векторы
и явно падает на cache miss. Это отдельный **cached** эксперимент, его время
не измеряет стоимость полного нового VLM-вывода.
Для каждого метода сохраняются `provenance.json`, `metrics.json`, pipeline results,
monitor JSON, консольный log; ошибка остаётся как failed, не исчезает из отчёта.
Пик памяти — максимум суммы RSS потомков с шагом 0.1 с, не точный hardware peak.
Если Ollama уже работал вне дерева, его память не входит в этот показатель.

Один запуск без monitor также доступен через `scripts/run_experiment.py`.
Для будущих опытов передавать `--sbert-model` как локальный snapshot с конкретной
revision. Сохранённые старые embedding-кэши revision не содержат: нельзя
приписывать им текущий revision задним числом. Манифесты Ollama и digests
фиксируются в provenance; для опубликованного пакета хранить также SHA256 весов.

## Проверка семантики отдельно

```powershell
.venv/Scripts/python.exe scripts/probe_semantic_identity.py --out experiments/runs/semantic_probe_repeat
.venv/Scripts/python.exe scripts/compare_pair_metrics.py experiments/local/pairs.json --out experiments/evidence/pair_metrics.json
```

Первый — диагностическая панель заранее заданных кадров 0/180/360 двух объектов,
с GT-crop; это не end-to-end показатель. Второй требует отдельно подготовленные
пары train/validation/test с нормальными provenance: пример схемы в docstring.
Реальные независимые train/validation/test пары пока отсутствуют; не запускать
оценку ковариации на трёх позитивных описаниях исходной записи. Числа результата
по синтетическим тестам библиотеки не являются научным экспериментом.

## Оценка, таблицы, графики и проверки

```powershell
.audit-venv/Scripts/python.exe scripts/summarize_experiments.py
.audit-venv/Scripts/python.exe scripts/evaluate_run.py experiments/runs/query66/geometry/pipeline/results.json experiments/local/drive/annotations_66.json --out experiments/local/recomputed_66.json
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check src tests scripts run.py
.venv/Scripts/python.exe -m build
```

Результаты и графики строятся из JSON, не из вручную набранных значений. Скрипт
summary разделяет query и не строит CI для одного независимого video-cluster.
После изменения порога online gate требуется **новый полный прогон**: replay
трёх исходных score не воспроизводит потерю и повторное обнаружение.
`moid.experiment_stats.paired_cluster_interval` готов для парных кластерных CI,
когда появится несколько независимых сцен. Отдельно анализировать два протокола
эталона, не объединять разные query одного видео в независимые наблюдения.
