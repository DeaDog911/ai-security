# ai-security — контекст для новых сессий

Курс «Безопасность и доверенность систем ИИ», цикл ПР1–ПР4 на одном датасете и одной модели.
Команда: Андрусенко К.А., Синёв В.Е., Шестакова М.А., гр. 1379.

## С чего начинать сессию
1. Прочитать `docs/context/PROGRESS.md`: где остановились и что делать дальше.
2. Прочитать `docs/context/DECISIONS.md`: принятые решения. Не пересматривать их без согласия пользователя.
3. Если нужны детали задания или данных, смотреть `docs/context/PROJECT.md` и `docs/context/PLAN_PR1.md`.

## В конце сессии
- Обновить `PROGRESS.md`: что сделано, что дальше, открытые вопросы.
- Новые решения записать в `DECISIONS.md` (с датой).

## Правила работы
- Язык общения и документации: русский.
- Python + PyTorch. LightGBM и другие бустинги не используем (решение пользователя).
- Обучение запускает пользователь на своём компьютере: Windows, AMD RX 7700 XT, 32 ГБ RAM. PyTorch с ROCm (GPU виден как `torch.cuda`); код выбирает `cuda`, если доступно, иначе CPU. Код пишется в облачной сессии, кладётся в проект, запускает пользователь.
- Перед реализацией нового этапа план утверждает пользователь.
- Baseline-модель после фиксации не меняется: она используется во всех ПР1–ПР4.
- Все оценки сопровождаются 95% CI: для долей интервал Уилсона, для остального bootstrap с B = 2000 (BCa, запасной вариант — percentile; парный для ПР2/ПР3, стратифицированный для ПР4). Модель при bootstrap не переобучается. Подробности — `docs/boot.docx` и раздел Bootstrap в `PROJECT.md`.
- seed = 42 везде.

## Окружение
- `venv` в корне проекта, Python 3.12. Создаётся скриптом `scripts/setup_env.ps1` командой `python -m virtualenv --python python312 venv` (требование пользователя: virtualenv, папка `venv`). Ключи: `-Cpu` — без GPU, `-Recreate` — пересоздать, `-PythonSpec` — другая версия. Системный Python пользователя — 3.13; 3.12 нужен для колёс ROCm.
- PyTorch 2.9.1 + ROCm 7.2.1 (колёса AMD с repo.radeon.com), прочие пакеты — `requirements.txt`. Ядро Jupyter: «Python (ai-security)».
- Проверка: `python src\check_device.py`.
- Пути и константы (датасет, seed, B) — `src/config.py`. Датасет — два набора LADBS (с 30.10.2026): https://data.lacity.org/City-Infrastructure-Service-Requests/Building-and-Safety-Building-Permits-Issued-Betwee/dyxf-7hc4/about_data и https://data.lacity.org/City-Infrastructure-Service-Requests/Building-and-Safety-Building-Permits-Issued-from-2/pi9x-tg5x/about_data
- В облачной сессии пакеты не ставить: код пишется там, а запускается у пользователя.

## Структура (целевая)
```
LA_BLDG_PERMITS_*_20261030.csv  исходный датасет, 2 файла (не изменять; ссылки — README.md)
README.md                       установка окружения и источник данных
scripts/setup_env.ps1           создание venv
docs/practice_ai.docx           методичка ПР1–ПР4
docs/boot.docx                  пояснение по bootstrap
docs/context/                   файлы контекста между сессиями
presentations/                  вводная презентация (предобработку из неё не берём)
src/                            код: data, features, model, stats, train, pr1
notebooks/                      Jupyter-ноутбуки с результатами
artifacts/                      модель, препроцессор, config.json, сплиты
reports/pr1/                    таблицы CSV и графики ПР1
requirements.txt
```
