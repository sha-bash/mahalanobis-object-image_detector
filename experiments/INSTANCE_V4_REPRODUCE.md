# Воспроизведение серии v4 из чистого checkout

В Git сохранены протоколы и входные манифесты. Готовых описаний VLM, признаков,
порогов, весов, предсказаний и журналов нет. Ниже восстановлены все зависимости
серии v4; прежних весов MLP не требуется. Используйте отдельный checkout для
нового воспроизведения, чтобы не перезаписать локальные результаты.

## Окружение и модели

Windows x64, Python 3.11, CPU, около 8 ГБ RAM. Стадии выполняются последовательно.
Первичная загрузка требует интернета, свободного диска и соблюдения лицензий.

```powershell
py -3.11 -m venv .venv
.venv/Scripts/python.exe -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements/mlp-cpu-lock.txt
.venv/Scripts/python.exe -m pip install --no-deps -e .
py -3.11 -m venv .audit-venv
.audit-venv/Scripts/python.exe -m pip install psutil==7.1.0
.\run.ps1 -SetupOnly
.venv/Scripts/python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/dinov2-small', revision='ed25f3a31f01632728cabb09d1542f84ab7b0056'); snapshot_download('sentence-transformers/all-MiniLM-L6-v2', revision='1110a243fdf4706b3f48f1d95db1a4f5529b4d41')"
.venv/Scripts/python.exe -m pip check
```

Скрипты используют стандартный Hugging Face cache пользователя и Qwen3.5:0.8b
через локальную Ollama. Digest VLM записывается новым запуском. Совпадение имени
модели не гарантирует побитовое совпадение старой генерации. YOLOv8n получает
`yolov8n.pt` через Ultralytics при первом вызове; вес остаётся вне Git.

## Восстановление обучающих данных и признаков

Скачайте официальные архивы GMOT-40 в `experiments/local/gmot40/` по
[инструкции данных](INPUT_DATA.md). Файлы должны называться
`GenericMOT_JPEG_Sequence.zip` и `track_label.zip`. Сверьте SHA-256 с
`evidence/gmot40/inventory.json`.

```powershell
.venv/Scripts/python.exe scripts/mlp_data.py prepare
.venv/Scripts/python.exe scripts/mlp_data.py describe
.venv/Scripts/python.exe scripts/mlp_data.py features
.venv/Scripts/python.exe scripts/instance_data_v3.py prepare
.venv/Scripts/python.exe scripts/instance_data_v3.py describe
.venv/Scripts/python.exe scripts/instance_data_v3.py features
```

Эти команды восстанавливают `features.npz` и локальные изображения серий mlp_v2
и instance_v3. Итоговая серия использует их как обучающие данные; старые
validation/test экземпляры этих серий не являются новым тестом v4.
Подготовка RGB-D требует доступности HTTP Range на сервере авторов; ошибки
загрузки нельзя обходить молчаливой заменой объектов.

## Новые изображения и оценка

```powershell
.venv/Scripts/python.exe scripts/instance_data_v4.py prepare
.audit-venv/Scripts/python.exe scripts/monitor_stage.py --record experiments/evidence/instance_v4/caption_execution.json .venv/Scripts/python.exe scripts/instance_data_v4.py describe
.audit-venv/Scripts/python.exe scripts/run_instance_v4.py
```

Последовательный запуск вычисляет признаки, обучает три проектора, выбирает
пороги и оценивает тест. Перед тестом записывается `pretest_checkpoint.json`
с SHA-256 порогов и сведений об обучении. Повторный запуск проверяет неизменность
этих файлов. Контрольная точка хранится локально, в Git ничего не добавляется.
Успешные стадии пропускаются. `--force` пересчитывает стадии, но не разрешает
изменять уже зафиксированный эксперимент. Для другой конфигурации нужен
отдельный каталог/checkout и новый протокол.

## Перенос на видео

```powershell
.venv/Scripts/python.exe scripts/download_davis_v4.py
.venv/Scripts/python.exe scripts/davis_experiment_v4.py prepare
.audit-venv/Scripts/python.exe scripts/run_davis_v4.py
.venv/Scripts/python.exe scripts/verify_instance_v4.py
.venv/Scripts/python.exe scripts/plot_instance_v4.py
.venv/Scripts/python.exe -m pytest -q
```

Результаты, графики и журналы появятся в игнорируемой части
`experiments/evidence/`; веса — в `experiments/local/instance_v4/`.
Скрипты построения отчётов и ноутбуков сохранены в `scripts/`, но требуют
завершённых расчётов и отдельных издательских зависимостей.

Чистая установка и полный новый прогон моделей при уборке репозитория не
выполнялись: проверялись код, тесты, упаковка и отсутствие результатов в Git.
Воспроизведение не делает уже раскрытый тест независимым. Для новых научных
выводов нужен заранее зафиксированный новый тест; условия исходных протоколов
сохранены в INSTANCE_V4_PROTOCOL.md и INSTANCE_V4_VIDEO_PROTOCOL.md.
