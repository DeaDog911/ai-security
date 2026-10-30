"""Общие пути и константы проекта."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Два набора LADBS с одинаковой схемой; выгрузка 30.10.2026 (данные обновлены 25.10.2026).
DATASETS = {
    "dyxf-7hc4": {
        "file": "LA_BLDG_PERMITS_2010_2019_20261030.csv",
        "name": "Building and Safety - Building Permits Issued Between 2010 and 2019 (N)",
        "url": "https://data.lacity.org/City-Infrastructure-Service-Requests/"
               "Building-and-Safety-Building-Permits-Issued-Betwee/dyxf-7hc4/about_data",
    },
    "pi9x-tg5x": {
        "file": "LA_BLDG_PERMITS_2020_PRESENT_20261030.csv",
        "name": "Building and Safety - Building Permits Issued from 2020 to Present (N)",
        "url": "https://data.lacity.org/City-Infrastructure-Service-Requests/"
               "Building-and-Safety-Building-Permits-Issued-from-2/pi9x-tg5x/about_data",
    },
}
DATASET_PATHS = {k: PROJECT_ROOT / v["file"] for k, v in DATASETS.items()}
EXPORT_URL = "https://data.lacity.org/api/views/{id}/rows.csv?accessType=DOWNLOAD"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"

SEED = 42
BOOTSTRAP_B = 2000


def require_dataset() -> dict[str, Path]:
    """Пути к обоим файлам датасета или объяснение, где их взять."""
    missing = [k for k, p in DATASET_PATHS.items() if not p.exists()]
    if missing:
        lines = [f"  {DATASETS[k]['file']}: {EXPORT_URL.format(id=k)}\n    (страница: {DATASETS[k]['url']})"
                 for k in missing]
        raise FileNotFoundError(
            "Не найдены файлы датасета. Скачайте CSV (Export → CSV) и сохраните в корень проекта:\n"
            + "\n".join(lines)
            + "\nУчтите: наборы обновляются; новая выгрузка будет другой версией данных, "
              "результаты ПР1–ПР4 зафиксированы для выгрузки от 30.10.2026."
        )
    return dict(DATASET_PATHS)
