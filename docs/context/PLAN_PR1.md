# План: baseline-модель и ПР1

Статус: план согласован 2026-10-30. Датасет заменён (см. DECISIONS.md). Этап 0 завершён, baseline зафиксирован 2026-10-30. ПР1 выполнена 2026-10-31.

## Этап 0. Baseline-модель (нужна для ПР1–ПР4)
- [x] `scripts/setup_env.ps1` + `requirements.txt` + `README.md`: создание venv (virtualenv, Python 3.12), PyTorch ROCm/CPU.
- [x] `src/config.py`: пути, URL датасета, seed, B.
- [x] `src/check_device.py`: версии, GPU, тесты matmul/SDPA/autocast, замер скорости обучения. Проверен: RX 7700 XT работает через ROCm, x128 к CPU.
- [x] `src/stats.py`: `wilson_ci`, `bootstrap_ci` (BCa через scipy, fallback percentile, B = 2000, seed), `paired_bootstrap_ci`, `stratified_bootstrap_ci` — общие для ПР1–ПР4.
- [x] `src/data.py`: загрузка двух CSV, парсинг чисел и дат; ключ участка APN; отбор трёх типов Bldg.
- [x] Отбор строк для обучения: Valuation > 0; дубли без учёта ID удаляются до разбиения.
- [x] Split 70/15/15 хронологически по `ISSUE_DATE`. Пересечение участков между частями не запрещаем (в реальной работе участок может встретиться повторно), но измеряем в ПР1 как leakage. Индексы сохраняются в `artifacts/splits.parquet`.
- [x] `src/features.py`: препроцессор (fit только на train).
  - Исключаем ID, PII, признаки из будущего, столбцы с пропусками > 99%.
  - Числовые: площадь, этажи, квартиры, широта/долгота, год/месяц/день недели выдачи, срок до окончания лицензии; log1p для скошенных, клиппинг по квантилям train, индикаторы пропуска, стандартизация.
  - Категориальные: тип/подтип/категория, офис, зона, ZIP, census tract, тип лицензии, штат подрядчика, подрядчик; редкие (частота < порога) → `OTHER`, пропуск → `MISSING`.
  - Текст: TF-IDF (слова + символьные n-граммы) → TruncatedSVD(64) → один токен.
- [x] `src/model.py`: FT-Transformer — Feature Tokenizer (числа: w·x+b, категории: embedding, текст: linear) + [CLS] → N блоков (pre-LN, MHSA, FFN, dropout) → голова по [CLS].
- [x] `src/train_baseline.py`: AdamW, MSE на log1p, ранняя остановка по val RMSLE; сохранение весов, препроцессора, `config.json` (seed, гиперпараметры, split, версии библиотек), кривых обучения.
- [x] Оценка на test: RMSLE, MdAPE, доля в пределах ±25%, всё с bootstrap 95% CI (B = 2000).

## Этап 1. ПР1: профиль качества (`src/pr1_data_quality.py` → `reports/pr1/`)
- [x] Описание: источник, размер, модальности, число признаков, распределение цели, split.
- [x] Полнота: missing rate по столбцам и по объектам (используемые признаки), CI Уилсона; структурные пропуски по PERMIT_TYPE.
- [x] Валидность: VALUATION ≤ 0 или пусто; площадь ≤ 0 у Bldg-New/Addition; HEIGHT ≤ 0 или неправдоподобно большая; DU_CHANGED > 2 для «1 or 2 Family»; ISSUE_DATE < SUBMITTED_DATE; STATUS_DATE < ISSUE_DATE; координаты вне bbox Лос-Анджелеса; пустое или очень короткое описание; ZONE/CT с несколькими значениями.
- [x] Уникальность: точные дубли, дубли без ID, повторы PERMIT_NBR; близкие дубли (тот же участок + Jaccard текста по MinHash ≥ 0,9 + близкая стоимость).
- [x] Leakage: точные и почти точные совпадения признаков между train/val/test; общие участки; список исключённых признаков из будущего.
- [x] Баланс: распределение log-цели по бинам (imbalance ratio, энтропия, доля хвоста); баланс подгрупп PERMIT_TYPE / SUB_TYPE по частям.
- [x] Качество меток: out-of-fold прогнозы FT-Transformer (5-fold CV на train, те же гиперпараметры, что у baseline) → остатки в log-шкале → подозрительная метка при |robust z| > 3,5; доля с CI Уилсона. Дополнительно: доля |r| > ln(10), `cleanlab.regression` label quality score, частоты служебных значений ($501 и т. п.), ручной разбор 20 худших. Bootstrap для ПР1 — по объектам датасета (обычный или стратифицированный по PERMIT_TYPE).
- [x] Shift: MMD (RBF, медианная эвристика, подвыборки ≈ 5k, permutation p-value) по признакам и эмбеддингам [CLS]; C2ST (логистическая регрессия или MLP: train vs test) с ROC-AUC и bootstrap CI; KS-тест для цели; динамика по годам.
- [x] Аномалии: IsolationForest по признакам + выбросы log-цели (robust z-score); доля с CI Уилсона.
- [x] Итоговая таблица в формате методички и вывод о влиянии на ПР2–ПР4.
- [x] `notebooks/pr1_data_quality.ipynb`: чтение результатов, графики, интерпретация.
- [x] Дополнительно: C2ST по группам признаков (`shift_features`) и факты о причинах сдвига (`drift_facts`).
