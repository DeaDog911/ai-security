# Промпт для запуска плана в Claude Code

Запускать из корня проекта: `cd C:\Users\deado\IdeaProjects\ai-security` → `claude`, затем вставить текст ниже.

---

Выполни практическую работу №1 (с предварительным этапом 0) по курсу «Безопасность и доверенность систем ИИ» в этом репозитории.

## Контекст
Сначала прочитай `CLAUDE.md`, затем `docs/context/PROGRESS.md`, `docs/context/DECISIONS.md`, `docs/context/PLAN_PR1.md` и `docs/context/PROJECT.md`. План в `PLAN_PR1.md` уже утверждён — реализуй его, не пересматривая решения из `DECISIONS.md`. Если решение действительно нужно поменять (например, что-то не работает на этом железе), остановись и спроси меня. Требования к ПР1 — раздел 2 `docs/practice_ai.docx`, правила bootstrap — `docs/boot.docx`. Вводную презентацию для предобработки не используй.

## Окружение (Windows, AMD RX 7700 XT, 32 ГБ RAM)
1. Если папки `venv` нет: `powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1`. Скрипт создаёт окружение командой `python -m virtualenv --python python312 venv` (Python 3.12 обязателен: колёса PyTorch+ROCm от AMD есть только под cp312; системный Python — 3.13). Если 3.12 нет — `winget install Python.Python.3.12`.
2. Все команды Python выполняй через `venv\Scripts\python.exe`.
3. Запусти `venv\Scripts\python.exe src\check_device.py` и покажи мне итог. Если GPU (ROCm, виден как `torch.cuda`) не работает — попробуй разобраться (драйвер, версия Windows, сборка torch); не получается — сообщи и продолжай на CPU (`setup_env.ps1 -Cpu -Recreate`), уменьшив модель для 5-fold OOF.

## Этап 0 — baseline-модель (по PLAN_PR1.md)
- `src/stats.py`: `wilson_ci`, `bootstrap_ci` (BCa через `scipy.stats.bootstrap`, fallback percentile, B=2000, seed 42), `paired_bootstrap_ci`, `stratified_bootstrap_ci`; покрыть короткими тестами на синтетике.
- `src/data.py`: загрузка через `config.require_dataset()`; парсинг денег и чисел (разделитель тысяч NBSP `\xa0`, десятичная запятая, `$`), дат, `POINT (lon lat)`; ключ участка Book-Page-Parcel; фильтр Valuation > 0; удаление дублей без учёта ID; хронологический split 70/15/15 по `Issue Date` → `artifacts/splits.parquet`.
- `src/features.py`: препроцессор, который обучается только на train (исключить ID, PII, признаки из будущего — Status, Status Date — и столбцы с пропусками > 99%; числовые: log1p, клиппинг, индикаторы пропуска, стандартизация; категориальные: редкие → OTHER, пропуски → MISSING; текст: TF-IDF → SVD(64)). Сохраняется в `artifacts/`.
- `src/model.py`: FT-Transformer (feature tokenizer + [CLS] + pre-LN блоки + регрессионная голова, текст отдельным токеном).
- `src/train_baseline.py`: AdamW, MSE на log1p, early stopping по val RMSLE, AMP на GPU; сохранить веса, `config.json` (seed, гиперпараметры, split, версии библиотек, устройство), кривые обучения; на test посчитать RMSLE, MdAPE и долю прогнозов в пределах ±25% с bootstrap 95% CI.
- Сначала сделай быстрый smoke-прогон на подвыборке (несколько тысяч строк, 1–2 эпохи). Полное обучение запускай в фоне, с логом в файл, и отслеживай прогресс.
- **Остановка 1:** покажи метрики baseline и кривые обучения; перед ПР1 дождись моего «ок». После фиксации baseline не меняй.

## Этап 1 — ПР1 (по PLAN_PR1.md)
- `src/pr1_data_quality.py` → `reports/pr1/` (CSV-таблицы и PNG-графики): описание датасета, полнота, валидность, уникальность (включая near-duplicates через MinHash), train-test leakage, баланс, качество меток, shift (MMD с permutation p-value, C2ST с ROC-AUC и CI, KS-тест цели), аномалии.
- Качество меток: OOF-прогнозы FT-Transformer (5-fold на train, гиперпараметры baseline); метка подозрительна при |robust z| > 3,5, где z = 0,6745·(r − median)/MAD, r = log1p(y) − ŷ. Дополнительно: доля |r| > ln 10, `cleanlab.regression` score, частоты служебных значений (например, $501), 20 худших примеров для ручного разбора. OOF-прогнозы кешируй в `artifacts/`, чтобы не пересчитывать.
- Для всех долей — CI Уилсона, для остального — bootstrap (B=2000). Итоговая таблица в формате методички: свойство | метрика | значение | 95% CI | интерпретация.
- `notebooks/pr1_data_quality.ipynb` (ядро «Python (ai-security)»): загружает готовые результаты из `reports/pr1/` и `artifacts/` (тяжёлые вычисления не повторяет), показывает таблицы и графики, интерпретирует их; в конце — вывод о том, какие свойства данных повлияют на ПР2–ПР4. Выполни ноутбук целиком (`jupyter nbconvert --execute --inplace`), чтобы выводы были сохранены.

## Правила
- Язык кода (комментарии, docstring), отчётов и общения — русский.
- seed = 42 везде; LightGBM и прочие бустинги не используем.
- Тяжёлые шаги должны быть идемпотентными: если результат уже есть в `artifacts/`, не пересчитывать без флага `--force`.
- После каждого законченного шага отмечай пункт в `PLAN_PR1.md` и обновляй `PROGRESS.md`; новые решения записывай в `DECISIONS.md` с датой.
- Коммить в git логическими шагами (датасет, `venv`, `artifacts/` с весами — в `.gitignore`).
- В конце кратко опиши, что получилось, и приложи главные цифры.
