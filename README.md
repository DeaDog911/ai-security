# ai-security

Цикл практических работ ПР1–ПР4 по курсу «Безопасность и доверенность систем ИИ»: регрессия оценочной стоимости строительства (FT-Transformer, PyTorch) на данных разрешений Лос-Анджелеса.

## Данные
Два набора Департамента строительства и безопасности Лос-Анджелеса (LADBS) с одинаковой схемой, выгрузка 30.10.2026 (данные обновлены 25.10.2026). Файлы лежат в корне проекта:

| Файл | Набор | Строк |
|---|---|---|
| `LA_BLDG_PERMITS_2010_2019_20261030.csv` | [Building Permits Issued Between 2010 and 2019 (N)](https://data.lacity.org/City-Infrastructure-Service-Requests/Building-and-Safety-Building-Permits-Issued-Betwee/dyxf-7hc4/about_data) | 533 366 |
| `LA_BLDG_PERMITS_2020_PRESENT_20261030.csv` | [Building Permits Issued from 2020 to Present (N)](https://data.lacity.org/City-Infrastructure-Service-Requests/Building-and-Safety-Building-Permits-Issued-from-2/pi9x-tg5x/about_data) | 420 507 |

Если файлов нет, скачайте CSV (Export → CSV на странице набора или напрямую `https://data.lacity.org/api/views/<id>/rows.csv?accessType=DOWNLOAD`, где `<id>` — `dyxf-7hc4` и `pi9x-tg5x`) и сохраните под этими именами. Наборы обновляются, поэтому новая выгрузка будет другой версией данных и результаты могут отличаться.

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
