# mahalanobis-object-image_detector

Few-shot и zero-shot поиск объектов: **референсные фото** задают идентичность, VLM описывает видимые атрибуты по-английски, Sentence-BERT строит эмбеддинги, скоринг — расстояние Махаланобиса (`mcd`).

Репозиторий **не изменяет** [mahalanobis-concept-drift](https://github.com/sha-bash/mahalanobis-concept-drift). Урезанная копия ядра лежит в `src/mcd`.

## Идея

Искомый объект — не захардкоженная марка, а то, что видно на эталонах (автомобиль, дерево, упаковка и т.д.). Опционально пользователь уточняет описание в терминале; LLM может задать короткие вопросы по полям `unknown`. Поиск идёт по второй папке кадров. Отчёт пишется в `reports/`.

## Установка

Python 3.11+.

```bash
pip install -e ".[dev]"
pip install -e ".[gigachat]"   # GigaChat Vision / LLM
pip install -e ".[ocr]"        # EasyOCR (опционально)
pip install -e ".[dino]"       # Grounding DINO (опционально)
pip install -e ".[video,visual]" # OpenCV + CLIP для обработки видео
```

Скопируйте `.env.example` в `.env` и задайте `GIGACHAT_CREDENTIALS`. CLI подхватывает `.env` через `python-dotenv` (extra `[gigachat]`).

## Данные демо

- `data/refs/` — эталоны (крупные кадры Lada Vesta)
- `data/search/` — кадры поиска (в том числе Веста в сцене: `photo_4`, `photo_6`) и `labels.json`
- `reports/` — выходы прогонов (не коммитятся)

## CLI

### Подкоманды

| Команда | Назначение |
|---------|-----------|
| `moid search` | Интерактивный или пакетный поиск объектов по изображениям |
| `moid search-video` | Интерактивный поиск объекта во всех видео папки |
| `moid few-shot` | Поиск объекта на одном целевом изображении |
| `moid zero-shot` | Поиск по текстовому запросу среди набора изображений |

### `moid search` — поиск по изображениям

Интерактивный режим (этапы 0–9 в терминале):

```bash
moid search --config configs/default.yaml
```

Enter подставляет `data/refs` и `data/search`. После подписей референсов: «желаете дополнить описание? yes/no».

Без вопросов (CI / скрипты):

```bash
moid search --refs data/refs --search data/search --non-interactive --no-refine --out reports --config configs/default.yaml
moid search --stub --refs tests/fixtures --search tests/fixtures --non-interactive --no-refine
```

| Флаг | Назначение |
|------|-----------|
| `--refs` | Папка с референсными фото (эталонные объекты) |
| `--search` | Папка с фото для поиска |
| `--out` | Директория для отчётов и overlay-изображений |
| `--config` | Файл YAML-конфигурации |
| `--stub` | Использовать stub VLM/LLM (без GigaChat, читает sidecar `.txt`) |
| `--no-refine` | Пропустить уточнение описания |
| `--non-interactive` | Пакетный режим без вопросов |

### `moid search-video` — поиск в видео

Интерактивно запрашивает папку референсов, предлагает дополнить профиль объекта,
запрашивает папку видео и частоту проверки в кадрах/с. Enter использует значения
`paths.refs`, `paths.videos` и `video.sample_fps` из конфигурации.

```bash
moid search-video --config configs/default.yaml
```

Без вопросов:

```bash
moid search-video --refs data/refs --video data/videos --sample-fps 2 \
  --out reports --non-interactive --no-refine --config configs/default.yaml
```

| Флаг | Назначение |
|------|-----------|
| `--video` | Папка с видео или один видеофайл |
| `--refs` | Папка с референсными фото |
| `--out` | Родительская директория запусков |
| `--config` | Файл YAML-конфигурации |
| `--sample-fps` | Сколько кадров в секунду проверять |
| `--stub` | Использовать stub VLM/LLM |
| `--no-refine` | Не запрашивать дополнение описания |
| `--non-interactive` | Использовать флаги и значения по умолчанию |

**Как работает:**
1. VLM и визуальный энкодер формируют профиль по референсам.
2. Все `mp4`, `avi`, `mov`, `mkv`, `webm` из папки обходятся по имени.
3. Кадры выбираются по времени с `sample_fps`, регионы фильтруются CLIP и проверяются VLM + Mahalanobis.
4. Каждый запуск создаёт `results.json`, LLM-отчёт `report.md`, все положительные кадры в `overlays/` и лучшие кадры в `best/`.

Команды `moid video` и `moid search-video-i` сохранены как совместимые aliases.

### `moid few-shot` — одно изображение

```bash
moid few-shot --refs data/refs --target data/search/photo_4_2026-09-01_18-53-11.jpg --config configs/default.yaml --out result.json
```

### `moid zero-shot` — текстовый запрос

```bash
moid zero-shot --query "красный грузовик" --images data/search --stub --mode pairwise
```

`--stub` читает sidecar `.txt` рядом с файлом. Для кропов сетки нужен живой VLM.

## Конфиг (`configs/default.yaml`)

| Ключ | По умолчанию | Комментарий |
|-----|---------|--------|
| `detector.sbert_model` | `all-MiniLM-L6-v2` | Можно `all-mpnet-base-v2` (768-d; лучше с проектором) |
| `detector.projector_path` | `null` | `.npz` / `.pt` линейный MLP |
| `detector.threshold` | `max_margin` | Также `chi2`, `fixed`, `quantile`, `negative_quantile` |
| `detector.threshold_margin` / `threshold_floor` | 0.8 / 0.0 | Для `max_margin` |
| `detector.negative_quantile` | 0.05 | Калибровка по негативам |
| `regions.backend` | `grid` | `dino` или `hybrid` (DINO → сетка) |
| `decision.target_match_gate` | true | `no` отбрасывает регион; `uncertain` — более строгий порог |
| `decision.min_positive_regions` | 1 | Сколько принятых боксов после NMS нужно для кадра |
| `ocr.backend` | `none` | `stub` / `easyocr`; только если профиль — транспорт с plate |
| `hints.text` | `""` | Необязательная подсказка, не замена референсов |
| `paths.refs` / `videos` / `reports` | `data/refs` … | Дефолты видео-интерактива |
| `video.sample_fps` | `1.0` | Проверяемых кадров в секунду |
| `visual.model_name` | `openai/clip-vit-base-patch32` | Энкодер префильтрации регионов |
| `adapters.vlm` / `llm` | `gigachat` в YAML | В dataclass по умолчанию `stub` |

Калибровка порога на негативных подписях: `moid.calibration.calibrate_threshold(detector, negative_captions, quantile=..., floor=...)`.

Проектор: `python scripts/train_projector.py --captions captions.txt --out projector.npz --dim 64`.

## Python API

```python
from moid.config import load_config
from moid.pipeline import run_few_shot, run_zero_shot
from moid.factory import build_vlm, build_embedder

cfg = load_config("configs/default.yaml")
result = run_few_shot("data/refs", "data/search/photo_4_2026-09-01_18-53-11.jpg",
                      vlm=build_vlm(cfg, stub=True), embedder=build_embedder(cfg), config=cfg)
print(result.frame_positive, result.detections)
```

`include_best_if_none_accepted` рисует лучший бокс, но **не** ставит `frame_positive`.

## Порядок запуска

### Полный пайплайн обработки

```
Шаг 1: Подготовка данных
  ├── data/refs/   → референсные фото (эталонный объект)
  └── data/search/ → кадры для поиска (фото или видео)

Шаг 2: Обработка изображений
  moid search --refs data/refs --search data/search \
    --non-interactive --no-refine --out reports \
    --config configs/default.yaml

Шаг 3: Обработка видео (если есть)
  moid search-video --video data/videos \
    --refs data/refs --out reports --non-interactive \
    --config configs/default.yaml

Шаг 4: Результаты
  ├── reports/           → отчёты по изображениям
  │   └── overlays/      → overlay с bounding boxes
  └── reports/<run>/       → results.json, report.md, overlays/, best/
```

### Пошаговое описание

1. **Референсы** — загрузите в `data/refs/` фото искомого объекта (крупные кадры, чётко видимый объект)
2. **Поиск** — загрузите в `data/search/` кадры/видео для анализа
3. **Обработка** — запустите `moid search` для изображений и `moid search-video` для видео
4. **Отчёты** — результаты в `reports/` (JSON + overlay-изображения с bounding boxes)

## Ограничения

- Сетка режет объекты; для транспорта/крупных предметов лучше `regions.backend: hybrid` при установленном `[dino]`.
- Ковариация в высокой размерности при n≤10 сильно регуляризована; проектор или `diagonal` предпочтительнее `full`.
- Видео не трекает объект между кадрами: каждый выбранный кадр анализируется независимо.
