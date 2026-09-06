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
```

Скопируйте `.env.example` в `.env` и задайте `GIGACHAT_CREDENTIALS`. CLI подхватывает `.env` через `python-dotenv` (extra `[gigachat]`).

## Данные демо

- `data/refs/` — эталоны (крупные кадры Lada Vesta)
- `data/search/` — кадры поиска (в том числе Веста в сцене: `photo_4`, `photo_6`) и `labels.json`
- `reports/` — выходы прогонов (не коммитятся)

## CLI

Интерактивный поиск (этапы 0–9 в терминале):

```bash
moid search --config configs/default.yaml
```

Enter подставляет `data/refs` и `data/search`. После подписей референсов: «желаете дополнить описание? yes/no».

Без вопросов (CI / скрипты):

```bash
moid search --refs data/refs --search data/search --non-interactive --no-refine --out reports --config configs/default.yaml
moid search --stub --refs tests/fixtures --search tests/fixtures --non-interactive --no-refine
```

Одно изображение (прежний API):

```bash
moid few-shot --refs data/refs --target data/search/photo_4_2026-09-01_18-53-11.jpg --config configs/default.yaml --out result.json
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
| `paths.refs` / `search` / `reports` | `data/refs` … | Дефолты интерактива |
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

## Ограничения

- Сетка режет объекты; для транспорта/крупных предметов лучше `regions.backend: hybrid` при установленном `[dino]`.
- Ковариация в высокой размерности при n≤10 сильно регуляризована; проектор или `diagonal` предпочтительнее `full`.
