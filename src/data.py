"""Загрузка и парсинг LA BUILD PERMITS, отбор строк для обучения и хронологический split.

Запуск из корня проекта (идемпотентно, пересчёт — флаг --force):
    venv\\Scripts\\python src\\data.py

Результаты:
    artifacts/parsed.parquet   все 375 150 строк: исходные столбцы (строки) + распарсенные поля
    artifacts/splits.parquet   row_id → split (train/val/test) для строк, допущенных к обучению
    artifacts/split_info.json  размеры частей, границы дат, число отброшенных строк
"""
from __future__ import annotations

import argparse
import json
import re

import numpy as np
import pandas as pd

from config import ARTIFACTS_DIR, DATASET_FILE, require_dataset

PARSED_PATH = ARTIFACTS_DIR / "parsed.parquet"
SPLITS_PATH = ARTIFACTS_DIR / "splits.parquet"
SPLIT_INFO_PATH = ARTIFACTS_DIR / "split_info.json"

SPLIT_FRACTIONS = (0.70, 0.15, 0.15)

# Идентификаторы разрешения: не признаки и не учитываются при поиске дублей.
ID_COLS = ["PCIS Permit #", "Reference # (Old Permit #)"]
# Числа с разделителем тысяч NBSP и десятичной запятой.
NUMBER_COLS = {
    "Floor Area-L.A. Zoning Code Definition": "floor_area_zoning",
    "Floor Area-L.A. Building Code Definition": "floor_area_building",
    "# of Residential Dwelling Units": "units",
    "# of Stories": "stories",
}
DATE_COLS = {
    "Issue Date": ("issue_date", "%m/%d/%Y"),
    "Status Date": ("status_date", "%m/%d/%Y %I:%M:%S %p"),
    "License Expiration Date": ("license_exp_date", "%m/%d/%Y"),
}
TARGET = "valuation"

_POINT_RE = re.compile(r"POINT\s*\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)")


def parse_number(s: pd.Series) -> pd.Series:
    """'$8\\xa0893\\xa0000,00' → 8893000.0; '2\\xa0454' → 2454.0; не число → NaN."""
    t = (
        s.astype("string")
        .str.replace("$", "", regex=False)
        .str.replace("\xa0", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    return pd.to_numeric(t, errors="coerce").astype(float)


def parse_point(s: pd.Series) -> tuple[pd.Series, pd.Series]:
    """'POINT (lon lat)' → (lon, lat)."""
    m = s.astype("string").str.extract(_POINT_RE)
    return pd.to_numeric(m[0], errors="coerce"), pd.to_numeric(m[1], errors="coerce")


def parcel_key(df: pd.DataFrame) -> pd.Series:
    """Ключ участка Assessor Book-Page-Parcel (NaN, если какая-то часть пропущена)."""
    parts = [df[c].astype("string").str.strip() for c in ("Assessor Book", "Assessor Page", "Assessor Parcel")]
    key = parts[0] + "-" + parts[1] + "-" + parts[2]
    return key.where(parts[0].notna() & parts[1].notna() & parts[2].notna())


def load_raw_csv() -> pd.DataFrame:
    """Исходный CSV как строки. Пустая ячейка → NaN; строки вроде 'NA' не трогаем."""
    path = require_dataset()
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""], encoding="utf-8")
    df.insert(0, "row_id", np.arange(len(df), dtype=np.int64))
    return df


def parse(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет к исходным столбцам распарсенные поля (snake_case)."""
    out = df.copy()
    out[TARGET] = parse_number(df["Valuation"])
    for col, name in NUMBER_COLS.items():
        out[name] = parse_number(df[col])
    for col, (name, fmt) in DATE_COLS.items():
        out[name] = pd.to_datetime(df[col], format=fmt, errors="coerce")
    out["lon"], out["lat"] = parse_point(df["Latitude/Longitude"])
    out["parcel"] = parcel_key(df)
    out["work_desc"] = df["Work Description"].fillna("").astype(str)
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
    """Исходные столбцы CSV (без row_id и распарсенных полей)."""
    derived = {"row_id", TARGET, "lon", "lat", "parcel", "work_desc",
               *NUMBER_COLS.values(), *(v[0] for v in DATE_COLS.values())}
    return [c for c in df.columns if c not in derived]


def select_training_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Valuation > 0, затем удаление дублей без учёта ID (остаётся первое вхождение)."""
    info = {"n_raw": int(len(df))}
    valid = df[TARGET].notna() & (df[TARGET] > 0)
    info["n_valuation_nan"] = int(df[TARGET].isna().sum())
    info["n_valuation_le0"] = int((df[TARGET] <= 0).sum())
    d = df[valid]
    content = [c for c in raw_columns(df) if c not in ID_COLS]
    dup = d.duplicated(subset=content, keep="first")
    info["n_dup_without_id_removed"] = int(dup.sum())
    d = d[~dup]
    info["n_clean"] = int(len(d))
    return d, info


def chronological_split(issue_date: pd.Series, fractions=SPLIT_FRACTIONS) -> tuple[pd.Series, dict]:
    """70/15/15 по Issue Date. Граница — дата: все разрешения одного дня попадают в одну часть."""
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
        raise ValueError("у строк для обучения не распарсилась Issue Date")
    split, sinfo = chronological_split(clean["issue_date"])
    info.update(sinfo)
    info["dataset"] = DATASET_FILE
    info["fractions"] = list(SPLIT_FRACTIONS)
    info["rule"] = ("Valuation > 0; дубли без учёта ID (PCIS Permit #, Reference #) удалены до разбиения; "
                    "split хронологический по Issue Date, граница по дате")
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
