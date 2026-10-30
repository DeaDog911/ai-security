# ai-security

Цикл практических работ ПР1–ПР4 по курсу «Безопасность и доверенность систем ИИ»: регрессия оценочной стоимости строительства (FT-Transformer, PyTorch) на данных разрешений Лос-Анджелеса.

## Данные
Файл `LA_BUILD_PERMITS_20261030.csv` лежит в корне проекта (выгрузка от 30.10.2026, 375 150 строк).
Если файла нет, его можно скачать со страницы набора данных на портале открытых данных Лос-Анджелеса (Export → CSV):
https://data.lacity.org/A-Prosperous-City/LA-BUILD-PERMITS/xnhu-aczu/about_data
Сохраните файл в корень проекта под тем же именем. Новая выгрузка будет другой версией данных, и результаты могут отличаться.

## Окружение (Windows)
Нужны Python 3.12; для GPU AMD — Windows 11 и драйвер AMD Adrenalin 26.2.2 или новее.

```powershell
# из корня проекта
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1        # PyTorch + ROCm (AMD GPU)
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1 -Cpu   # только CPU
.\venv\Scripts\Activate.ps1
python src\check_device.py
```

Скрипт создаёт окружение `venv` командой `python -m virtualenv --python python312 venv` (нужен установленный Python 3.12), ставит PyTorch 2.9.1 (ROCm 7.2.1 от AMD или CPU-сборку) и зависимости из `requirements.txt`, регистрирует ядро Jupyter «Python (ai-security)» и запускает проверку устройства.
Если Python 3.12 нет: `winget install Python.Python.3.12`.

Источник инструкции по ROCm: https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/install-pytorch.html

## Структура
См. `CLAUDE.md`; контекст работы и принятые решения — `docs/context/`.
