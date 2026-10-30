"""Общие пути и константы проекта."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_FILE = "LA_BUILD_PERMITS_20261030.csv"
DATASET_PATH = PROJECT_ROOT / DATASET_FILE
DATASET_URL = "https://data.lacity.org/A-Prosperous-City/LA-BUILD-PERMITS/xnhu-aczu/about_data"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"

SEED = 42
BOOTSTRAP_B = 2000


def require_dataset(path: Path = DATASET_PATH) -> Path:
    """Возвращает путь к датасету или объясняет, где его взять."""
    if not path.exists():
        raise FileNotFoundError(
            f"Не найден датасет {path}.\n"
            f"Скачайте LA BUILD PERMITS (Export → CSV) со страницы {DATASET_URL}\n"
            f"и сохраните в корень проекта под именем {DATASET_FILE}.\n"
            "Учтите: новая выгрузка будет другой версией данных; результаты ПР1–ПР4 "
            "зафиксированы для выгрузки от 30.10.2026."
        )
    return path
