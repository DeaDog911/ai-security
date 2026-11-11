"""Сборка презентации ПР2 «Устойчивость к шуму и естественным искажениям» (.pptx) из reports/pr2.

Числа на слайдах берутся из CSV отчёта ПР2, поэтому после пересчёта ПР2 достаточно запустить скрипт заново.
Вручную заданы только формулировки (типы шума, выводы, ограничения).
План и соответствие пунктам методички — docs/context/PLAN_PR2_PRESENTATION.md.
Оформление и проверка переполнения текста — общие с ПР1 (scripts/build_pr1_deck.py).

Запуск из корня проекта:
    venv\\Scripts\\python scripts\\build_pr2_deck.py
"""
from __future__ import annotations

import argparse
import re
import sys
import tempfile
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_pr1_deck import (  # noqa: E402
    BLUE, GREY, LIGHT, MID, NAVY, PALE, SH, SW, WHITE,
    check_overflow, ci, flow, header, new_slide, note, num, picture, rich, stat, text_box,
)

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports" / "pr2"
OUT = ROOT / "presentations" / "ПР2 Устойчивость к шуму.pptx"
N_MAIN = 11
TYPES = ["T1", "T2", "X1", "X2"]
POP = "test без служебных сумм ($501, $301, ≤ $10)"


# ---------------------------------------------------------------- данные и форматирование

def csv(name: str) -> pd.DataFrame:
    return pd.read_csv(REP / f"{name}.csv", encoding="utf-8-sig")


def sg(x: float, d: int = 3) -> str:
    """Число со знаком: «+0,231», «−0,027»."""
    return ("+" if x > 0 else "") + num(x, d)


def lvl(s: float) -> str:
    return num(s, 1)


class Data:
    def __init__(self) -> None:
        b = csv("baseline").set_index("metric")
        self.rmsle, self.mdape = b.loc["RMSLE"], b.loc["MdAPE, %"]
        self.w25 = b.loc["доля в ±25%"]
        self.n = int(self.rmsle["n"])
        self.B = int(self.rmsle["B"])
        self.matches = bool(b["matches_baseline"].all())

        self.curves = csv("curves")
        self.grid = csv("noise_grid")
        self.rauc = csv("rauc").set_index("type")
        self.levels = sorted(x for x in self.curves["level"].unique() if x > 0)
        self.s_max = max(self.levels)
        self.r = int(self.grid["r"].iloc[0])
        self.names = self.rauc["name"].to_dict()
        self.order = list(self.rauc.sort_values("rauc_rel", ascending=False).index)  # 1 — самый критичный
        self.group = csv("by_group")
        self.feat = csv("by_feature")
        self.bias = csv("bias")

    def point(self, t: str, s: float) -> pd.Series:
        c = self.curves
        rows = c[(c["type"] == t) & (c["level"].round(3) == round(s, 3))]
        assert len(rows) == 1, (t, s)
        return rows.iloc[0]

    def first_level(self, t: str, col: str) -> float | None:
        c = self.curves
        rows = c[(c["type"] == t) & (c["level"] > 0) & c[col].astype(bool)].sort_values("level")
        return float(rows["level"].iloc[0]) if len(rows) else None

    def grid_row(self, t: str, s: float) -> pd.Series:
        g = self.grid
        return g[(g["type"] == t) & (g["level"].round(3) == round(s, 3))].iloc[0]

    def grp(self, t: str, s: float, grouping: str, group: str) -> pd.Series:
        g = self.group
        rows = g[(g["type"] == t) & (g["level"].round(3) == round(s, 3)) & (g["grouping"] == grouping)
                 & (g["group"] == group)]
        assert len(rows) == 1, (t, s, grouping, group)
        return rows.iloc[0]

    def ft(self, t: str, group: str) -> pd.Series:
        f = self.feat
        return f[(f["type"] == t) & (f["group"] == group)].iloc[0]

    def bs(self, t: str, s: float, grouping: str, group: str) -> pd.Series:
        b = self.bias
        rows = b[(b["type"] == t) & (b["level"].round(3) == round(s, 3)) & (b["grouping"] == grouping)
                 & (b["group"] == group)]
        assert len(rows) == 1, (t, s, grouping, group)
        return rows.iloc[0]


# ---------------------------------------------------------------- таблица с многострочными ячейками

def mtable(slide, x, y, w, rows: list[list[str]], col_w: list[float], size: float = 11, row_h: float = 0.4,
           head_size: float | None = None, bold_first_col: bool = False, sub_size: float | None = None,
           head_h: float | None = None):
    """Как table() из ПР1, но «\\n» в ячейке начинает новый абзац; второй и следующие абзацы — мельче и серым
    (sub_size), если он задан. Префикс «!» у абзаца — выделить его жирным."""
    nr, nc = len(rows), len(rows[0])
    hh = head_h or row_h
    gf = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(hh + row_h * (nr - 1)))
    t = gf.table
    total = sum(col_w)
    for j, cw in enumerate(col_w):
        t.columns[j].width = Inches(w * cw / total)
    for i in range(nr):
        t.rows[i].height = Inches(hh if i == 0 else row_h)
        for j in range(nc):
            cell = t.cell(i, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = NAVY if i == 0 else (WHITE if i % 2 else PALE)
            cell.margin_left = cell.margin_right = Inches(0.06)
            cell.margin_top = cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame
            tf.word_wrap = True
            for k, part in enumerate(rows[i][j].split("\n")):
                p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
                bold = part.startswith("!")
                part = part.lstrip("!")
                if i == 0:
                    rich(p, part, head_size or size, WHITE, bold=True)
                elif k > 0 and sub_size:
                    rich(p, part, sub_size, NAVY if bold else GREY, bold=bold)
                else:
                    rich(p, part, size, NAVY, bold=bold or (bold_first_col and j == 0))
    return t


# ---------------------------------------------------------------- слайды

def s01_title(prs, d: Data) -> None:
    s = new_slide(prs)
    bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(SW), Inches(SH))
    bg.fill.solid()
    bg.fill.fore_color.rgb = BLUE
    bg.line.fill.background()
    text_box(s, 0.9, 1.3, 11.5, 0.5, ["ПРАКТИЧЕСКАЯ РАБОТА 2 · УСТОЙЧИВОСТЬ К ШУМУ"], size=16, color=LIGHT, bold=True)
    text_box(s, 0.9, 1.9, 11.5, 1.9, ["Исследование устойчивости глубокой модели к шуму и естественным искажениям"],
             size=34, color=WHITE, bold=True)
    text_box(s, 0.9, 3.85, 11.5, 0.9,
             ["Разрешения на строительство г. Лос-Анджелес (LADBS) → оценочная стоимость работ; "
              "модель FT-Transformer (таблица + текст)"], size=18, color=LIGHT)
    text_box(s, 0.9, 5.3, 11.5, 0.4, ["КОМАНДА"], size=14, color=LIGHT, bold=True)
    text_box(s, 0.9, 5.7, 11.5, 0.9, ["Андрусенко К.А. · Синёв В.Е. · Шестакова М.А.", "гр. 1379 · ноябрь 2026"],
             size=18, color=WHITE)


def s02_setup(prs, d: Data) -> None:
    s = new_slide(prs)
    header(s, "Постановка эксперимента и baseline", None, f"2 / {N_MAIN}")
    text_box(s, 0.45, 1.25, 7.4, 3.6, [
        "**Что исследуем:** устойчивость к случайным искажениям входа, не подобранным под модель "
        "(corruption robustness, не состязательная).",
        "**Модель:** baseline FT-Transformer из ПР1; веса и препроцессор не меняются, переобучения нет.",
        f"**Данные:** {POP}, n = {num(d.n)}; хронологически последние 15% разрешений.",
        "**Основная метрика:** RMSLE — **меньше лучше**; вспомогательные — MdAPE и доля прогнозов в ±25% от факта.",
    ], size=15, bullet=True, space=10)
    text_box(s, 0.45, 4.95, 7.4, 0.35, ["**Где применяется шум** — как ошибка во входных данных:"], size=13)
    flow(s, 0.45, 5.35, 7.4, 0.95, ["исходные столбцы test", "искажение (seed)", "препроцессор train",
                                     "модель → RMSLE"], size=11.5, gap=0.28)
    note(s, 0.45, 6.4, 7.4, 0.55, "Клиппинг по квантилям train, замена редких категорий на OTHER и эмбеддинг "
                                 "«пропуск» — часть модели: их сглаживающее действие входит в измеряемую устойчивость.")

    r = d.rmsle
    stat(s, 8.2, 1.25, 4.7, 1.5, num(r["value"], 4), "M(0) — RMSLE на неизменённом test",
         f"95% CI {ci(r['ci_low'], r['ci_high'], 4)}, {r['ci_method']}, B = {d.B}", vsize=30)
    m = d.mdape
    stat(s, 8.2, 2.95, 2.25, 1.3, num(m["value"], 1) + "%", "MdAPE",
         f"95% CI {ci(m['ci_low'], m['ci_high'], 1)}, {m['ci_method']}", vsize=22)
    w = d.w25
    stat(s, 10.6, 2.95, 2.3, 1.3, num(w["value"] * 100, 1) + "%", "прогнозов в ±25% от факта",
         f"95% CI {ci(w['ci_low'], w['ci_high'], 1, 100)}, Уилсон", vsize=22)
    text_box(s, 8.2, 4.5, 4.7, 1.8, [
        "**Проверка:** M(0), посчитанная кодом ПР2, совпала с метриками baseline "
        + ("бит-в-бит" if d.matches else "НЕ совпала") + "; при s = 0 прогноз после всего пути искажения "
        "равен прогнозу baseline.",
        "seed = 42; CI метрик — bootstrap по объектам.",
    ], size=12, color=GREY, space=6)


TYPE_ROWS = {
    "T1": ("таблица", "значение → пусто (MCAR), по каждой ячейке числовых и категориальных признаков, кроме типа "
                      "разрешения и дат; координаты — парой", "доля стёртых ячеек",
           "пропусков в данных 23,7%; площадь ≤ 0 у 16% заполненных"),
    "T2": ("таблица", "непустая категория → случайное значение с частотами train", "доля заменённых категорий",
           "23,4% районов CNC test нет в train, новые census tract"),
    "X1": ("текст", "в слове одна опечатка: перестановка, удаление, вставка или соседняя клавиша", "доля слов ≥ 3 букв "
                    "с опечаткой", "незнакомых модели слов 0,56% → 1,36%"),
    "X2": ("текст", "слово описания удаляется", "доля удалённых слов", "длинные описания обрезаются на 96 словах"),
}


def s03_noise(prs, d: Data) -> None:
    s = new_slide(prs)
    header(s, "Модель шума", None, f"3 / {N_MAIN}")
    rows = [["Тип", "Мод.", "Искажение", "s — это", "Связь с данными (ПР1)", f"Изменено при s = {lvl(d.s_max)}"]]
    for t in TYPES:
        mod, what, smean, link = TYPE_ROWS[t]
        g = d.grid_row(t, d.s_max)
        fact = f"{num(g['frac_changed'] * 100, 1)}%"
        if t == "T1":
            fact += "\nуже пустые не меняются"
        elif t == "T2":
            fact += "\nзамена может совпасть с исходным"
        elif t == "X1":
            fact += f"\n{num(g['frac_changed_of_words'] * 100, 1)}% всех слов"
        rows.append([f"{t}\n{d.names[t]}", mod, what, smean, link, fact])
    mtable(s, 0.45, 1.2, 12.4, rows, [1.55, 0.85, 3.6, 1.75, 2.65, 2.0], size=11, row_h=0.84, sub_size=9.5,
           bold_first_col=True, head_h=0.42)

    lv = "; ".join(lvl(x) for x in [0.0] + d.levels)
    y = 5.55
    stat(s, 0.45, y, 4.0, 1.2, f"s ∈ {{{lv}}}", "общая шкала интенсивности для всех типов",
         f"{len(d.levels)} ненулевых уровней; s₀ = 0 — чистый test", vsize=17)
    stat(s, 4.6, y, 2.6, 1.2, f"r = {d.r}", "реализаций шума в каждой точке",
         "seed = 42 + 1000t + 100k + j", vsize=22)
    stat(s, 7.35, y, 2.6, 1.2, num(len(TYPES) * len(d.levels) * d.r), "прогонов инференса",
         f"{len(TYPES)} типа × {len(d.levels)} уровней × {d.r}", vsize=22)
    text_box(s, 10.1, y, 2.8, 1.2, [
        "По два типа на модальность. Слова короче 3 букв и числа опечатками не портятся; длина описания "
        "пересчитывается."], size=10.5, color=GREY)


def s04_method(prs, d: Data) -> None:
    s = new_slide(prs)
    header(s, "Метрики и статистика", None, f"4 / {N_MAIN}")
    text_box(s, 0.45, 1.15, 6.0, 0.35, ["**Метрики** (направление зафиксировано до запусков)"], size=14, color=BLUE)
    rows = [["Величина", "Формула", "Ухудшение"],
            ["Robustness curve", "R(s) = M(Dₛ), M — RMSLE", "рост"],
            ["Изменение", "ΔM(s) = M(0) − M(s)", "ΔM < 0"],
            ["Относительное", "R_rel(s) = M(s) / M(0)", "R_rel > 1"],
            ["Площадь", "R-AUC = 1/(sₘₐₓ − s₀) · ∫ R(s) ds,\nтрапеции по s₀ = 0 … sₘₐₓ = " + lvl(d.s_max),
             "больше"],
            ["Для сравнения типов", "R-AUC_rel = R-AUC / M(0)", "> 1"]]
    mtable(s, 0.45, 1.55, 6.0, rows, [1.6, 3.2, 1.2], size=11.5, row_h=0.5, bold_first_col=True)
    text_box(s, 0.45, 4.85, 6.0, 1.6, [
        "R-AUC — средняя RMSLE на диапазоне шума, в единицах RMSLE; у идеально устойчивой модели R-AUC_rel = 1.",
        "**Значимость:** статистическая — 95% CI ΔM не содержит 0; практическая — |ΔM| ≥ 0,02 "
        "(вдвое больше разброса RMSLE между seed обучения ≈ 0,01).",
    ], size=12, space=8)

    text_box(s, 6.85, 1.15, 6.0, 0.35, ["**95% CI: парный bootstrap по объектам**"], size=14, color=BLUE)
    steps = [
        f"для каждого объекта — квадрат лог-ошибки, усреднённый по r = {d.r} реализациям",
        "перевыборка объектов: одни и те же индексы для чистого test и всех (тип, s)",
        "на перевыборке — вся кривая R(s), ΔM(s) и R-AUC",
        f"B = {d.B} → percentile-интервалы; M(0) — BCa",
    ]
    for i, st in enumerate(steps):
        y = 1.6 + i * 0.95
        text_box(s, 6.85, y, 0.55, 0.75, [str(i + 1)], size=18, bold=True, color=WHITE, align=PP_ALIGN.CENTER,
                 anchor=MSO_ANCHOR.MIDDLE, shape=MSO_SHAPE.OVAL, fill=MID, margin=0)
        text_box(s, 7.55, y, 5.35, 0.75, [st], size=12.5, anchor=MSO_ANCHOR.MIDDLE,
                 shape=MSO_SHAPE.ROUNDED_RECTANGLE, fill=LIGHT, margin=0.1)
    note(s, 6.85, 5.5, 6.0, 1.0, "Парность убирает из ΔM разброс сложности объектов: CI разности намного уже CI "
                                 "самих M(s). Модель при bootstrap не переобучается. Разброс между реализациями "
                                 "шума показан отдельно (слайд 5).")


def s05_curves(prs, d: Data, tmp: Path) -> None:
    s = new_slide(prs)
    header(s, "Robustness curves", None, f"5 / {N_MAIN}")
    # надзаголовок PNG содержит внутреннее название совокупности — обрезаем, подписываем сами
    picture(s, REP / "curves_rmsle.png", 0.35, 1.15, 12.6, 4.35, crop=(0, 0.075, 1, 1), tmp=tmp)
    note(s, 0.45, 6.78, 11.0, 0.3, f"Test без служебных сумм, n = {num(d.n)}; точка — RMSLE по {d.r} реализациям; "
                                   f"коридор — 95% CI (парный bootstrap, B = {d.B}); вертикали — уровень по ПР1.")
    t1, t2 = d.point("T1", 0.1), d.point("T1", d.s_max)
    x1, x2 = d.point("X1", d.s_max), d.point("X2", d.s_max)
    c2 = d.point("T2", 0.2)
    c1 = d.point("T1", 0.2)
    facts = [
        f"**Пропуски (T1)** — самая крутая кривая: ΔM уже при s = 0,1 = {sg(t1['delta'], 3)} "
        f"{ci(t1['delta_low'], t1['delta_high'], 3)}; при s = {lvl(d.s_max)} RMSLE {num(t2['rmsle'], 3)}.",
        f"**Около естественного уровня** (s = 0,2): T1 {num(c1['rmsle'], 3)}, T2 {num(c2['rmsle'], 3)} "
        f"против {num(d.rmsle['value'], 3)} на чистом test.",
        f"**Текст:** опечатки и удаление слов идут вместе — при s = {lvl(d.s_max)} RMSLE {num(x1['rmsle'], 3)} "
        f"и {num(x2['rmsle'], 3)}.",
        f"**Случайность шума** мала: std RMSLE между реализациями ≤ {num(d.curves['rmsle_real_std'].max(), 4)}.",
    ]
    for i, f in enumerate(facts):
        text_box(s, 0.45 + (i % 2) * 6.3, 5.55 + (i // 2) * 0.6, 6.1, 0.58, [f], size=12)


def s06_levels(prs, d: Data) -> None:
    s = new_slide(prs)
    header(s, "Качество по уровням шума", None, f"6 / {N_MAIN}")
    rows = [["s"] + [f"{t}: {d.names[t]}" for t in TYPES]]
    for lv_ in d.levels:
        row = [lvl(lv_)]
        for t in TYPES:
            p = d.point(t, lv_)
            dl = f"ΔM {sg(p['delta'], 3)} {ci(p['delta_low'], p['delta_high'], 3)}"
            row.append(f"{num(p['rmsle'], 3)} {ci(p['rmsle_low'], p['rmsle_high'], 3)}\n"
                       + ("!" if p["practical"] else "") + dl)
        rows.append(row)
    mtable(s, 0.45, 1.2, 12.4, rows, [0.6, 2.95, 2.95, 2.95, 2.95], size=13, row_h=0.82, head_size=12,
           sub_size=11.5, bold_first_col=True)
    note(s, 0.45, 6.25, 12.4, 0.7,
         f"Первая строка ячейки — RMSLE и 95% CI; вторая — ΔM = M(0) − M(s) и 95% CI (парный bootstrap percentile, "
         f"B = {d.B}). **Жирным** — практически значимое ухудшение |ΔM| ≥ 0,02; во всех точках CI ΔM не содержит 0. "
         f"M(0) = {num(d.rmsle['value'], 3)}.")


def s07_rauc(prs, d: Data) -> None:
    s = new_slide(prs)
    header(s, "Площадь под кривой устойчивости", None, f"7 / {N_MAIN}")
    rows = [["Место", "Тип", "R-AUC\n95% CI", "R-AUC_rel\n95% CI", f"RMSLE при s = {lvl(d.s_max)}",
             f"ΔM при s = {lvl(d.s_max)}\n95% CI", "Значимо\nс s", "|ΔM| ≥ 0,02\nс s"]]
    for i, t in enumerate(d.order, 1):
        r = d.rauc.loc[t]
        p = d.point(t, d.s_max)
        rows.append([str(i), f"{t}\n{d.names[t]} ({r['modality']})",
                     f"{num(r['rauc'], 4)}\n{ci(r['rauc_low'], r['rauc_high'], 4)}",
                     f"{num(r['rauc_rel'], 4)}\n{ci(r['rauc_rel_low'], r['rauc_rel_high'], 4)}",
                     num(p["rmsle"], 3),
                     f"{sg(p['delta'], 3)}\n{ci(p['delta_low'], p['delta_high'], 3)}",
                     lvl(d.first_level(t, "significant")), lvl(d.first_level(t, "practical"))])
    mtable(s, 0.45, 1.25, 12.4, rows, [0.7, 2.6, 1.7, 1.7, 1.4, 1.75, 1.05, 1.5], size=14, row_h=0.85,
           head_size=12, sub_size=11)
    stat(s, 0.45, 5.6, 3.4, 1.1, num(d.rmsle["value"], 4), "M(0) — R-AUC идеально устойчивой модели",
         f"95% CI {ci(d.rmsle['ci_low'], d.rmsle['ci_high'], 4)}, BCa", vsize=20)
    note(s, 4.1, 5.6, 8.8, 1.1,
         f"R-AUC — средняя RMSLE на s ∈ [0; {lvl(d.s_max)}] (трапеции, {len(d.levels) + 1} точек), меньше — лучше; "
         f"место — по R-AUC_rel. CI — парный bootstrap по объектам, кривая восстанавливается на каждой из B = {d.B} "
         "перевыборок, percentile. CI X1 и X2 перекрываются; парно эти типы не сравнивались.")


def s08_groups(prs, d: Data) -> None:
    s = new_slide(prs)
    s_ = 0.4
    header(s, "Подгруппы", None, f"8 / {N_MAIN}")
    text_box(s, 0.45, 1.12, 12.4, 0.4, [f"ΔM при s = {lvl(s_)} и 95% CI; вторая строка — R_rel = M(s) / M(0)"],
             size=13)
    groups = [("PERMIT_TYPE", "Bldg-New", "Новое строительство"),
              ("PERMIT_TYPE", "Bldg-Addition", "Пристройки"),
              ("PERMIT_TYPE", "Bldg-Alter/Repair", "Ремонт и перестройка"),
              ("Площадь", "есть площадь (> 0)", "Площадь указана"),
              ("Площадь", "нет площади", "Площади нет")]
    rows = [["Подгруппа", "n", "RMSLE\nчистый test"] + [f"{t}: {d.names[t]}" for t in TYPES]]
    for gg, g, lab in groups:
        r0 = d.grp("T1", s_, gg, g)
        row = [lab, num(r0["n"]), f"{num(r0['rmsle0'], 3)}\n{ci(r0['rmsle0_low'], r0['rmsle0_high'], 3)}"]
        worst = min(TYPES, key=lambda t: d.grp(t, s_, gg, g)["delta"])
        for t in TYPES:
            r = d.grp(t, s_, gg, g)
            mark = "!" if t == worst else ""
            row.append(f"{mark}{sg(r['delta'], 3)} {ci(r['delta_low'], r['delta_high'], 3)}\n"
                       f"R_rel {num(r['r_rel'], 3)}")
        rows.append(row)
    mtable(s, 0.45, 1.55, 12.4, rows, [2.2, 0.9, 1.35, 1.99, 1.99, 1.99, 1.99], size=12, row_h=0.76,
           head_size=11.5, sub_size=10.5, bold_first_col=True)
    new, alt = d.grp("T1", s_, "PERMIT_TYPE", "Bldg-New"), d.grp("X2", s_, "PERMIT_TYPE", "Bldg-New")
    rep = d.grp("X2", s_, "PERMIT_TYPE", "Bldg-Alter/Repair")
    text_box(s, 0.45, 6.15, 12.4, 0.8, [
        f"**Жирным** — самый сильный тип шума в подгруппе. Новое строительство теряет больше всех от пропусков "
        f"(R_rel {num(new['r_rel'], 2)}) и почти не замечает удаления слов ({sg(alt['delta'], 3)}); у ремонта и "
        f"перестройки текстовый шум и порча категорий бьют сильнее (удаление слов {sg(rep['delta'], 3)}): "
        "описание работ для них — главный источник информации."], size=11.5, color=GREY)


def s09_features(prs, d: Data, tmp: Path) -> None:
    s = new_slide(prs)
    header(s, "Вклад признаков", None, f"9 / {N_MAIN}")
    text_box(s, 0.45, 1.12, 12.4, 0.4, [
        "При s = 0,3 портится **одна группа признаков**, остальное чистое; серым — порча всех признаков типа"], size=13)
    picture(s, REP / "by_feature.png", 0.45, 1.55, 12.4, 4.25)
    a, b, c = d.ft("T1", "Подтип и назначение"), d.ft("T1", "Площадь"), d.ft("T2", "Business Unit")
    geo = d.ft("T1", "Районы (CD, CNC, CPA, APC)")
    co = d.ft("T1", "Координаты")
    text_box(s, 0.45, 5.85, 12.4, 1.15, [
        (f"Пропуски: сильнее всего — подтип и назначение {sg(a['delta'], 4)} {ci(a['delta_low'], a['delta_high'], 4)}"
         f" и площадь {sg(b['delta'], 4)} {ci(b['delta_low'], b['delta_high'], 4)}; порча категорий — "
         f"Business Unit {sg(c['delta'], 4)} {ci(c['delta_low'], c['delta_high'], 4)}.", {"bullet": True}),
        (f"География почти не важна: координаты {sg(co['delta'], 4)} {ci(co['delta_low'], co['delta_high'], 4)}, "
         f"районы {sg(geo['delta'], 4)} {ci(geo['delta_low'], geo['delta_high'], 4)}. ΔM и 95% CI — парный "
         f"bootstrap percentile, B = {d.B}, r = {d.r}.", {"bullet": True}),
    ], size=12, space=4)


def s10_bias(prs, d: Data, tmp: Path) -> None:
    s = new_slide(prs)
    header(s, "Смещение прогноза под шумом", None, f"10 / {N_MAIN}")
    picture(s, REP / "bias_curves.png", 0.35, 1.2, 5.6, 3.6, crop=(0, 0, 0.5, 1), tmp=tmp)
    note(s, 0.45, 4.85, 5.5, 0.5, "Средний остаток log1p(прогноз) − log1p(факт); > 0 — завышение. Коридор — 95% CI.")
    sm = d.s_max
    qs = [g for g in d.bias[d.bias["grouping"] == "Квартиль VALUATION"]["group"].unique()]
    rows = [["Стоимость", "baseline"] + [f"{t}, s = {lvl(sm)}" for t in TYPES]]
    for q in qs + ["все"]:
        gg = "Все" if q == "все" else "Квартиль VALUATION"
        b0 = d.bs("T1", 0.0, gg, q)
        lab = re.sub(r"\d{4,}", lambda m: num(int(m.group())), q.split(": ")[-1]).replace("≤", "до")
        row = ["Все объекты" if q == "все" else lab,
               f"{sg(b0['bias'], 2)}\n{ci(b0['bias_low'], b0['bias_high'], 2)}"]
        for t in TYPES:
            b = d.bs(t, sm, gg, q)
            row.append(f"{sg(b['bias'], 2)}\n{ci(b['bias_low'], b['bias_high'], 2)}")
        rows.append(row)
    mtable(s, 6.2, 1.2, 6.7, rows, [1.55, 1.05, 1.03, 1.03, 1.03, 1.03], size=12, row_h=0.6, head_size=10.5,
           sub_size=9.5, bold_first_col=True)
    tr0 = d.grp("X2", 0.4, "Обрезка текста", "обрезан (> 96 слов)")
    text_box(s, 0.45, 5.4, 12.45, 1.6, [
        ("Baseline прижимает прогноз к среднему: дешёвые работы завышает, дорогие занижает. Табличный шум это "
         "усиливает — меньше информации, прогноз ближе к среднему.", {"bullet": True}),
        ("Удаление слов сдвигает все прогнозы вниз и частично гасит завышение: отсюда «улучшение» MdAPE и RMSLE на "
         f"обрезанных описаниях (ΔM {sg(tr0['delta'], 3)} {ci(tr0['delta_low'], tr0['delta_high'], 3)} при s = 0,4) — "
         "компенсация смещения, а не устойчивость. Вывод делаем по RMSLE.", {"bullet": True}),
    ], size=12, space=5)
    note(s, 6.2, 4.85, 6.7, 0.5, f"Таблица — средний остаток и 95% CI (парный percentile bootstrap внутри группы, "
                                 f"B = {d.B}); квартили стоимости test.")


def s11_conclusions(prs, d: Data) -> None:
    s = new_slide(prs)
    header(s, "Выводы и ограничения", None, f"11 / {N_MAIN}")
    cols = [
        ("Выводы", [
            "Самый критичный тип — пропуски во входных признаках: ухудшение практически значимо уже при малой доле; "
            "сильнее всего страдает новое строительство.",
            "Опечатки и удаление слов — второе место, между собой близки; уязвимее всего ремонт, где описание — "
            "главный источник информации.",
            "Порча категорий наименее критична; важны тип работ и Business Unit, география — нет.",
            "Шум смещает прогноз, поэтому вспомогательные метрики могут «улучшаться»; судить по RMSLE и по квартилям "
            "стоимости.",
            "На практике: проверять заполненность площади, подтипа и назначения; без них помечать прогноз как "
            "ненадёжный.",
        ]),
        ("Ограничения", [
            "Числовые признаки испорчены только пропусками: шум измерений и округление не исследовались.",
            "Шум MCAR и равновероятные опечатки — упрощение: реальные пропуски структурные, опечатки неравномерны.",
            "Клиппинг, OTHER и эмбеддинг «пропуск» — часть модели; их сглаживание входит в оценку.",
            "Профиль признаков — при одном уровне, подгруппы — при двух.",
            "Только один период (test); на других периодах устойчивость не проверялась.",
        ]),
    ]
    w = (SW - 0.9 - 0.3) / 2
    for i, (title, items) in enumerate(cols):
        x = 0.45 + i * (w + 0.3)
        text_box(s, x, 1.3, w, 0.6, [title], size=15, bold=True, color=WHITE, anchor=MSO_ANCHOR.MIDDLE,
                 shape=MSO_SHAPE.ROUNDED_RECTANGLE, fill=BLUE if i == 0 else MID, margin=0.12)
        text_box(s, x, 2.05, w, 4.9, [(t, {"bullet": True}) for t in items], size=15, space=12,
                 shape=MSO_SHAPE.ROUNDED_RECTANGLE, fill=PALE, margin=0.15)


# ---------------------------------------------------------------- сборка

def build(out: Path) -> list[str]:
    d = Data()
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(SW), Inches(SH)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        s01_title(prs, d)
        s02_setup(prs, d)
        s03_noise(prs, d)
        s04_method(prs, d)
        s05_curves(prs, d, tmp)
        s06_levels(prs, d)
        s07_rauc(prs, d)
        s08_groups(prs, d)
        s09_features(prs, d, tmp)
        s10_bias(prs, d, tmp)
        s11_conclusions(prs, d)
        assert len(prs.slides) == N_MAIN
        out.parent.mkdir(parents=True, exist_ok=True)
        prs.save(out)
    return check_overflow(prs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    warns = build(args.out)
    print(f"сохранено: {args.out}")
    for w in warns:
        print("ВНИМАНИЕ:", w)
    sys.exit(1 if warns else 0)


if __name__ == "__main__":
    main()
