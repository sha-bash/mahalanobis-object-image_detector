# Воспроизведение отложенной проверки

Протокол `HELDOUT_PROTOCOL.md` заморожен commit e36a8f8. Уточнения в
`HELDOUT_AMENDMENTS.md`; параметры не менялись по новым результатам.
Официальные архивы GMOT-40 и SHA256 указаны в `GMOT40_REPRODUCE.md` и
`evidence/gmot40/inventory.json`. Новые клипы выбираются без подбора по качеству.

```powershell
.audit-venv/Scripts/python.exe scripts/prepare_heldout.py
.audit-venv/Scripts/python.exe scripts/run_heldout.py --all --python .venv/Scripts/python.exe
.audit-venv/Scripts/python.exe scripts/summarize_heldout.py
.audit-venv/Scripts/python.exe scripts/plot_heldout.py
.venv/Scripts/python.exe -m pytest -q
```

Не перезаписывать старые каталоги: prepare и --all намеренно отказываются
работать при существующем каталоге назначения. Для независимого повтора
использовать отдельный checkout с исходными архивами/моделями либо явно
новые пути BASE/RUNS/EVIDENCE и записать их в журнал. Один незавершённый
вариант можно запустить через --sequence и --variant после выяснения причины,
сохранив исходный неудачный каталог; успешные результаты не заменять.

Диспетчер требует psutil и выполняется из audit-окружения; сами алгоритмы
работают в ML-окружении. Все версии, хэши входов, код, digest VLM, время и
память дерева процессов фиксируются в provenance.json / execution.json.
Свежий кэш этого опыта используется совместно для трёх семантических вариантов.
Это контролирует одинаковость текстов, но делает времена зависимыми от порядка.

G — геометрия; I — исходный изотропный Махаланобис, порог 6;
C — cosine на PCA8/L2; M — полная ковариация на тех же PCA8/L2.
Параметры и пороги C/M берутся из сохранённого старого опыта, car-0/car-1.
Для машины с другим именем пользователя автоматически меняется только
корень стандартного Hugging Face cache, ревизия SBERT фиксирована.

PDF: сначала `scripts/plot_multimodal.py` и `scripts/build_research_pdfs.py`
пересоздают прежние документы с русскими подписями графиков; затем
`scripts/build_heldout_pdfs.py` добавляет новый раздел и создаёт новые тезисы.
PDF-скрипты требуют ReportLab, Pillow и pypdf; в Codex используются из bundled
document runtime. Итог: `output/research_20261010_heldout/`.
Рендерить Poppler pdftoppm и проверять все новые страницы, количество страниц
тезисов обязательно равно двум. Word-шаблон конференции отдельно не проверен.

Исходные пиксели, видео, веса и кэш находятся только в игнорируемых каталогах.
В Git сохраняются компактные предсказания, метрики, описания, параметры,
графики и журналы происхождения в `evidence/heldout_v1`.
