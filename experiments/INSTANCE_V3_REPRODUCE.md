# Воспроизведение следующего этапа

Ветка `codex/instance-diagnostics-v3`. Протокол зафиксирован в `241ebb5`,
реализация и состав данных — в `03184ba`, до новых оценок. Перед чтением test scores последовательный запуск фиксирует SHA-256 порогов
и сведений об обучении в локальном pretest_checkpoint.json; Git не изменяется.
Старая ветка, отчёты v2 и незакоммиченные файлы пользователя сохранены.

Среда: прежний `requirements/mlp-cpu-lock.txt`, Python 3.11 CPU. Скрипты запуска
используют `.venv` для моделей и `.audit-venv` с psutil для мониторинга; пути можно
изменить в `run_instance_v3.py`. NumPy, SciPy, requests, Pillow, matplotlib,
torch, transformers, sentence-transformers и ultralytics уже входят в среду.
Нужны локальные ревизии DINO/MiniLM и Qwen3.5:0.8b, зафиксированные в
`MLP_REPRODUCE.md`. Тяжёлые модели запускаются строго по очереди.

```powershell
.venv/Scripts/python.exe scripts/diagnose_instance_v3.py
.venv/Scripts/python.exe scripts/rgbd_range_v3.py
.venv/Scripts/python.exe scripts/download_rgbd_v3.py
.venv/Scripts/python.exe scripts/instance_data_v3.py prepare
.venv/Scripts/python.exe scripts/instance_scenes_v3.py prepare
.audit-venv/Scripts/python.exe scripts/monitor_stage.py --record experiments/evidence/instance_v3/caption_execution.json .venv/Scripts/python.exe scripts/instance_data_v3.py describe
.audit-venv/Scripts/python.exe scripts/run_instance_v3.py
.venv/Scripts/python.exe scripts/verify_instance_v3.py
.venv/Scripts/python.exe scripts/build_instance_notebook_v3.py
.venv/Scripts/python.exe scripts/plot_instance_v3.py
.venv/Scripts/python.exe -m pytest -q
```

Загрузчик ZIP читает центральный каталог и выбранные записи по HTTP Range;
запись проверяется по CRC и SHA256. Полный ZIP не нужен. Его недокачанный файл
`.part`, оставшийся от первоначальной попытки полной загрузки, не используется.
`background.tar` сохранён для дальнейшей работы и не включён в текущие оценки.
Если зеркало меняет архив (размер, CRC, имена), нужно остановиться и оформить
новую версию данных, а не молча продолжать прежний протокол.

После `prepare` manifest фиксирует каждый ID, split, видео, кадр и хеш снимка.
Описания сохраняются после каждого изображения. Один повтор допускается только
для неудачного описания; его не заменяют более удачным по тестовой оценке.
При повторном запуске suite пропускает стадии с returncode0. Не удаляйте пороги
для попытки «улучшить» тот же тест. Для следующего опыта нужен новый каталог.

Веса: `experiments/local/instance_v3/*_hard.pt` и контроль `visual_random.pt`,
параметры загрузки JSON рядом.
Снимки, архивы, кэш VLM и веса остаются локальными. В Git сохраняются численные
результаты, признаки, описания, журналы завершения стадий и конфигурации.
Условия Washington RGB-D — non-commercial research/educational use only.

Готовые численные результаты и графики находятся в `experiments/evidence/instance_v3/`,
сценовые предсказания — в `experiments/evidence/instance_scenes_v3/`.
Единица основного sign test и bootstrap — категория, всего 12; сцены анализируются
описательно. 48 сценовых решений используют всего 30 изображений, 4 ID и 2 сцены.

Для документов используйте отдельный Python с python-docx, reportlab, pypdf,
Pillow (в данном запуске — комплектную среду документов Codex):

```powershell
python scripts/build_instance_report_v3.py
```

Он читает только завершённые результаты. Новые PDF/DOCX сохраняются отдельно
в `output/research_instance_v3/`. Для проверки Word использованы LibreOffice и
Poppler. Двухстраничные тезисы — научный черновик; соответствие всем требованиям
шаблона ИТНТ отдельно не удостоверено. Вклад известных методов, адаптации и
нового экспериментального вывода в отчёте разграничен.
