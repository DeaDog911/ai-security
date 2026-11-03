"""Загрузка и парсинг разрешений LADBS (2010–2019 и 2020–н. в.), отбор строк и хронологический split.

Запуск из корня проекта (идемпотентно, пересчёт — флаг --force):
    venv\\Scripts\\python src\\data.py

Результаты:
    artifacts/parsed.parquet   все строки обоих наборов (все типы): исходные столбцы (строки) + распарсенные поля
    artifacts/splits.parquet   row_id → split (train/val/test) для строк, допущенных к обучению
    artifacts/split_info.json  размеры частей, границы дат, число отброшенных строк
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from config import ARTIFACTS_DIR, DATASETS, require_dataset

PARSED_PATH = ARTIFACTS_DIR / "parsed.parquet"
SPLITS_PATH = ARTIFACTS_DIR / "splits.parquet"
SPLIT_INFO_PATH = ARTIFACTS_DIR / "split_info.json"

SPLIT_FRACTIONS = (0.70, 0.15, 0.15)
PERMIT_TYPES = ["Bldg-New", "Bldg-Addition", "Bldg-Alter/Repair"]

# Идентификаторы и технические поля: не признаки и не учитываются при поиске дублей.
ID_COLS = ["PERMIT_NBR", "REFRESH_TIME"]
NUMBER_COLS = {
    "VALUATION": "valuation",
    "SQUARE_FOOTAGE": "square_footage",
    "HEIGHT": "height",
    "DU_CHANGED": "du_changed",
    "ADU_CHANGED": "adu_changed",
    "JUNIOR_ADU": "junior_adu",
    "LAT": "lat",
    "LON": "lon",
}
DATE_COLS = {
    "ISSUE_DATE": "issue_date",
    "SUBMITTED_DATE": "submitted_date",
    "STATUS_DATE": "status_date",
    "COFO_DATE": "cofo_date",
}
DATE_FORMAT = "%m/%d/%Y"
TARGET = "valuation"
SERVICE_VALUES = (501.0, 301.0)  # служебные оценки (ПР1: $501 — 7,7% меток)
SERVICE_MAX = 10.0               # оценки ≤ $10 ($1 и т. п.) — тоже служебные


def service_mask(valuation: np.ndarray) -> np.ndarray:
    """R2: служебные значения VALUATION {$501, $301} ∪ {≤ $10} (эксперимент label_cleaning, принят 2026-11-03)."""
    v = np.asarray(valuation, dtype=float)
    return np.isin(np.round(v, 6), SERVICE_VALUES) | (v <= SERVICE_MAX + 1e-9)


def load_raw_csv() -> pd.DataFrame:
    """Оба CSV как строки (пустая ячейка → NaN), с колонками source и row_id."""
    parts = []
    for key, path in require_dataset().items():
        d = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""], encoding="utf-8")
        d.insert(0, "source", key)
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    df.insert(0, "row_id", np.arange(len(df), dtype=np.int64))
    return df


def parse(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет к исходным столбцам распарсенные поля (snake_case)."""
    out = df.copy()
    for col, name in NUMBER_COLS.items():
        out[name] = pd.to_numeric(df[col], errors="coerce").astype(float)
    for col, name in DATE_COLS.items():
        out[name] = pd.to_datetime(df[col], format=DATE_FORMAT, errors="coerce")
    out["parcel"] = df["APN"].astype("string").str.strip()
    out["work_desc"] = df["WORK_DESC"].fillna("").astype(str)
    return out


def load_parsed(force: bool = False) -> pd.DataFrame:
    """Распарсенный датасет (кеш artifacts/parsed.parquet)."""
    if PARSED_PATH.exists() and not force:
        return pd.read_parquet(PARSED_PATH)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    df = parse(load_raw_csv())
    df.to_parquet(PARSED_PATH, index=False)
    return df


def raw_columns(df: pd.DataFrame) -> list[str]:
    """Исходные столбцы CSV (без служебных и распарсенных полей)."""
    derived = {"row_id", "source", "parcel", "work_desc", *NUMBER_COLS.values(), *DATE_COLS.values()}
    return [c for c in df.columns if c not in derived]


def select_training_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Три типа Bldg → Valuation > 0 → удаление дублей без учёта ID (остаётся первое вхождение)."""
    info = {"n_raw": int(len(df)), "n_raw_by_source": df["source"].value_counts().to_dict()}
    d = df[df["PERMIT_TYPE"].isin(PERMIT_TYPES)]
    info["n_permit_types"] = int(len(d))
    info["n_valuation_nan"] = int(d[TARGET].isna().sum())
    info["n_valuation_le0"] = int((d[TARGET] <= 0).sum())
    d = d[d[TARGET].notna() & (d[TARGET] > 0)]
    content = [c for c in raw_columns(df) if c not in ID_COLS]
    dup = d.duplicated(subset=content, keep="first")
    info["n_dup_without_id_removed"] = int(dup.sum())
    d = d[~dup]
    info["n_clean"] = int(len(d))
    return d, info


def chronological_split(issue_date: pd.Series, fractions=SPLIT_FRACTIONS) -> tuple[pd.Series, dict]:
    """70/15/15 по дате выдачи. Граница — дата: все разрешения одного дня попадают в одну часть."""
    dates = issue_date.sort_values(kind="stable")
    n = len(dates)
    d1 = dates.iloc[int(round(fractions[0] * n)) - 1]
    d2 = dates.iloc[int(round((fractions[0] + fractions[1]) * n)) - 1]
    split = pd.Series(np.where(issue_date <= d1, "train", np.where(issue_date <= d2, "val", "test")),
                      index=issue_date.index)
    info = {"train_end": str(d1.date()), "val_end": str(d2.date())}
    for part in ("train", "val", "test"):
        m = split == part
        info[f"n_{part}"] = int(m.sum())
        info[f"{part}_dates"] = [str(issue_date[m].min().date()), str(issue_date[m].max().date())]
    return split, info


def build_splits(force: bool = False) -> pd.DataFrame:
    if SPLITS_PATH.exists() and SPLIT_INFO_PATH.exists() and not force:
        return pd.read_parquet(SPLITS_PATH)
    df = load_parsed(force=force)
    clean, info = select_training_rows(df)
    if clean["issue_date"].isna().any():
        raise ValueError("у строк для обучения не распарсилась ISSUE_DATE")
    split, sinfo = chronological_split(clean["issue_date"])
    info.update(sinfo)
    info["datasets"] = {k: v["file"] for k, v in DATASETS.items()}
    info["permit_types"] = PERMIT_TYPES
    info["fractions"] = list(SPLIT_FRACTIONS)
    info["rule"] = ("PERMIT_TYPE ∈ {Bldg-New, Bldg-Addition, Bldg-Alter/Repair}; VALUATION > 0; дубли без учёта "
                    "ID (PERMIT_NBR, REFRESH_TIME) удалены до разбиения; split хронологический по ISSUE_DATE, "
                    "граница по дате")
    out = pd.DataFrame({"row_id": clean["row_id"].to_numpy(), "split": split.to_numpy()})
    out.to_parquet(SPLITS_PATH, index=False)
    SPLIT_INFO_PATH.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def load_dataset() -> pd.DataFrame:
    """Строки для обучения с колонкой split (в порядке row_id)."""
    df = load_parsed()
    splits = build_splits()
    return df.merge(splits, on="row_id", how="inner", validate="one_to_one")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="пересчитать кеш parsed.parquet и split")
    args = ap.parse_args()
    load_parsed(force=args.force)
    build_splits(force=args.force)
    print(SPLIT_INFO_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
