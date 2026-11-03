"""Сверка CSV двух каталогов отчёта (например, reports/pr1 и копии reports/pr1_c96 до пересчёта).

Для каждого общего файла: совпадает бит-в-бит / совпадает с точностью до rtol / различается
(столбцы и число отличающихся ячеек). Файлы только в одном каталоге перечисляются отдельно.

Запуск из корня проекта:
    venv\\Scripts\\python scripts\\compare_reports.py reports\\pr1 reports\\pr1_c96
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def compare(a: pd.DataFrame, b: pd.DataFrame, rtol: float) -> str:
    if list(a.columns) != list(b.columns):
        return f"столбцы: +{sorted(set(a.columns) - set(b.columns))} −{sorted(set(b.columns) - set(a.columns))}"
    if a.shape != b.shape:
        return f"размер {a.shape} против {b.shape}"
    if a.equals(b):
        return "="
    diff = []
    for col in a.columns:
        x, y = a[col], b[col]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            bad = ~np.isclose(x.to_numpy(float), y.to_numpy(float), rtol=rtol, atol=0, equal_nan=True)
        else:
            bad = ~((x == y) | (x.isna() & y.isna())).to_numpy()
        if bad.any():
            diff.append(f"{col}: {int(bad.sum())}")
    return "≈" if not diff else "; ".join(diff)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("new")
    ap.add_argument("old")
    ap.add_argument("--rtol", type=float, default=1e-9)
    args = ap.parse_args()
    new, old = Path(args.new), Path(args.old)
    fn, fo = {p.name for p in new.glob("*.csv")}, {p.name for p in old.glob("*.csv")}
    for name in sorted(fn & fo):
        res = compare(pd.read_csv(new / name), pd.read_csv(old / name), args.rtol)
        print(f"{'OK ' if res in ('=', '≈') else 'DIFF'} {name}: {res}")
    for name in sorted(fn - fo):
        print(f"NEW  {name}")
    for name in sorted(fo - fn):
        print(f"GONE {name}")


if __name__ == "__main__":
    main()
