"""Сверка чисел в markdown-ячейках ноутбука с таблицами отчёта.

Ноутбук ПР не пересчитывает таблицы, а текст интерпретаций пишется вручную — после пересчёта
таблиц числа в тексте легко устаревают. Скрипт извлекает все числа из markdown-ячеек и ищет
каждое среди значений таблиц, показанных code-ячейками непосредственно перед этой ячейкой
(аргументы show("…")); если таких нет — таблиц всего раздела («## …»); плюс итоговая summary.csv
с учётом округления до показанного числа знаков и масштаба (доля → %, тыс., млн). Сравнение
со всеми CSV сразу почти бесполезно: короткие числа случайно совпадают с чем-нибудь.
Раздел без show(...) (выводы) сверяется со всеми CSV. Несовпавшие числа выводятся с контекстом — их нужно проверить
вручную: часть законно не берётся из CSV (пороги, параметры методов, даты).

Ограничение: совпадение не доказывает, что число взято из нужной строки таблицы.

Запуск из корня проекта:
    venv\\Scripts\\python scripts\\check_notebook_numbers.py notebooks\\pr1_data_quality.ipynb reports\\pr1
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# число: необязательный знак, группы разрядов через пробел/NBSP/узкий NBSP, дробная часть через , или .
NUM_RE = re.compile(r"(?<![\w.,])[-−+]?\d{1,3}(?:[   ]\d{3})+(?:[.,]\d+)?(?![\w])|(?<![\w.,])[-−+]?\d+(?:[.,]\d+)?(?![\w])")
SCALES = (1, 100, 1e-3, 1e-6, 1e-9)  # как есть, доля → %, тыс., млн, млрд


def parse(tok: str) -> tuple[float, int]:
    t = re.sub(r"[   ]", "", tok).replace("−", "-").replace(",", ".")
    dec = len(t.split(".")[1]) if "." in t else 0
    return float(t), dec


def csv_values(files) -> np.ndarray:
    vals = []
    for f in files:
        df = pd.read_csv(f, encoding="utf-8-sig", dtype=str)
        for col in df.columns:
            s = df[col].dropna().astype(str)
            # числа внутри текстовых ячеек тоже (например, «p = 0,094» в интерпретации)
            for cell in s:
                for tok in NUM_RE.findall(cell):
                    vals.append(parse(tok)[0])
                try:
                    vals.append(float(cell))
                except ValueError:
                    pass
    v = np.unique(np.abs(np.array(vals, dtype=float)))
    return v[np.isfinite(v)]


def matches(x: float, dec: int, values: np.ndarray) -> bool:
    x = abs(x)
    tol = 0.5 * 10.0 ** (-dec) + 1e-12
    for sc in SCALES:
        if np.any(np.abs(values * sc - x) <= tol):
            return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("notebook", type=Path)
    ap.add_argument("reports", type=Path)
    ap.add_argument("--min-value", type=float, default=0, help="не проверять числа меньше (по модулю)")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    nb = json.loads(args.notebook.read_text(encoding="utf-8"))
    # Для markdown-ячейки берутся таблицы code-ячеек, стоящих непосредственно перед ней (после
    # предыдущей markdown-ячейки); если их нет — таблицы всего раздела («## …»); если нет и их — все CSV.
    shown = lambda text: set(re.findall(r"show\(\s*[\"']([\w-]+)[\"']", text))  # noqa: E731
    sections, cur = [], {"cells": [], "tables": set()}
    local, prev_tables = {}, set()
    for i, cell in enumerate(nb["cells"]):
        text = "".join(cell["source"])
        if cell["cell_type"] == "markdown":
            if re.match(r"^##\s", text):
                sections.append(cur)
                cur = {"cells": [], "tables": set()}
            local[i] = prev_tables
            prev_tables = set()
        else:
            prev_tables = prev_tables | shown(text)
            cur["tables"] |= shown(text)
        cur["cells"].append(i)
    sections.append(cur)
    section_of = {i: sec["tables"] for sec in sections for i in sec["cells"]}
    all_csv = sorted(args.reports.glob("*.csv"))
    cache = {}

    def values_for(i: int) -> np.ndarray:
        for tables in (local.get(i, set()), section_of.get(i, set())):
            files = tuple(sorted(args.reports / f"{t}.csv" for t in tables if (args.reports / f"{t}.csv").exists()))
            if files:
                files = files + (args.reports / "summary.csv",)
                break
        else:
            files = tuple(all_csv)
        if files not in cache:
            cache[files] = csv_values(files)
        return cache[files]

    total, bad = 0, 0
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "markdown":
            continue
        values = values_for(i)
        text = "".join(cell["source"])
        for m in NUM_RE.finditer(text):
            x, dec = parse(m.group())
            if abs(x) < args.min_value:
                continue
            total += 1
            if not matches(x, dec, values):
                bad += 1
                ctx = text[max(0, m.start() - 50):m.end() + 30].replace("\n", " ")
                print(f"ячейка {i:2d}: {m.group():>10}  …{ctx}…")
    print(f"\nчисел в markdown: {total}; без совпадения в CSV: {bad}")


if __name__ == "__main__":
    main()
