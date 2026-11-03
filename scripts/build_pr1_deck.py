"""Сборка презентации ПР1 «Профиль качества данных» (.pptx) из таблиц и графиков reports/pr1.

Числа на слайдах берутся из CSV отчёта ПР1 и artifacts/baseline_metrics.json, поэтому после
пересчёта ПР1 достаточно запустить скрипт заново. Вручную заданы только итоги ручного разбора
20 худших меток (MANUAL_WORST20, из ноутбука, раздел 7) и формулировки выводов.
План и соответствие пунктам методички — docs/context/PLAN_PR1_PRESENTATION.md.

После сборки скрипт оценивает, помещается ли текст в рамки (по метрикам Arial), и печатает
предупреждения: PowerPoint не сообщает о переполнении сам.

Запуск из корня проекта:
    venv\\Scripts\\python scripts\\build_pr1_deck.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

import pandas as pd
from PIL import Image, ImageFont
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports" / "pr1"
METRICS = ROOT / "artifacts" / "baseline_metrics.json"
OUT = ROOT / "presentations" / "ПР1 Качество данных.pptx"

# Оформление — как во вводной презентации (синяя шапка, тёмно-синий текст, светлые плашки).
BLUE = RGBColor(0x1D, 0x4E, 0xD8)
NAVY = RGBColor(0x0F, 0x3D, 0x91)
MID = RGBColor(0x3B, 0x6B, 0xC4)
LIGHT = RGBColor(0xDB, 0xEA, 0xFE)
PALE = RGBColor(0xF3, 0xF7, 0xFD)
GREY = RGBColor(0x5B, 0x6B, 0x85)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
ORANGE = RGBColor(0xE8, 0x7A, 0x1E)
GREEN = RGBColor(0x2E, 0x9E, 0x5B)
FONT = "Arial"

SW, SH = 13.333, 7.5  # размер слайда, дюймы
N_MAIN = 12

# Ручной разбор 20 худших меток (train без R2): ноутбук, раздел 7, таблица «Ручной разбор».
MANUAL_WORST20 = [
    ("Стоимость, похоже, указана в тысячах $", 11,
     "новые дома и дуплексы на 1 200–11 946 кв. футов за $145–1 227; прогноз в 650–2 000 раз выше"),
    ("Стоимость всего проекта на доп. разрешении", 1,
     "«Supp. permit to capture correct construction cost» — $6,4 млн"),
    ("Неоднозначно", 3,
     "смена числа квартир или назначения за $2–10 тыс.; частичный снос с усилением — этап или проект"),
    ("Правдоподобная метка — ошибка модели", 5,
     "этапные разрешения крупных объектов («early start», «foundation only») без площади, $8–262 млн"),
]
EXAMPLE_PERMITS = {"13010-10000-00494": "стоимость в тыс. $?",  # примеры из 20 худших и их категория
                   "12010-10007-01270": "ошибка модели"}


# ---------------------------------------------------------------- данные и форматирование

def csv(name: str) -> pd.DataFrame:
    return pd.read_csv(REP / f"{name}.csv", encoding="utf-8-sig")


def pick(df: pd.DataFrame, **cond) -> pd.Series:
    """Единственная строка таблицы, где столбцы равны заданным значениям (ключи — имена столбцов,
    пробелы и прочие символы — через словарь cond={"имя": значение})."""
    extra = cond.pop("cond", {})
    mask = pd.Series(True, index=df.index)
    for k, v in {**cond, **extra}.items():
        mask &= df[k] == v
    rows = df[mask]
    if len(rows) != 1:
        raise KeyError(f"ожидалась одна строка для {cond | extra}, найдено {len(rows)}")
    return rows.iloc[0]


def num(x: float, d: int = 0) -> str:
    """Число по-русски: десятичная запятая, неразрывный пробел между разрядами."""
    s = f"{x:,.{d}f}"
    return s.replace(",", "\x00").replace(".", ",").replace("\x00", " ").replace("-", "−")


def pdig(p: float) -> int:
    """Знаков после запятой для процента p: два значащих разряда, но не больше трёх знаков."""
    p = abs(p)
    return 1 if p >= 10 else 2 if p >= 0.1 else 3


def pct(v: float, d: int | None = None) -> str:
    d = pdig(v * 100) if d is None else d
    return num(v * 100, d) + "%"


def ci(lo: float, hi: float, d: int, scale: float = 1.0) -> str:
    return f"[{num(lo * scale, d)}; {num(hi * scale, d)}]"


def pct_ci(row: pd.Series, d: int | None = None) -> tuple[str, str]:
    """(«3,56%», «[3,51; 3,62]») для строки с value / ci_low / ci_high (доли)."""
    d = pdig(row["value"] * 100) if d is None else d
    return pct(row["value"], d), ci(row["ci_low"], row["ci_high"], d, 100)


def money(x: float) -> str:
    if x >= 1e9:
        return f"${num(x / 1e9, 2)} млрд"
    if x >= 1e6:
        return f"${num(x / 1e6, 2 if x < 1e7 else 1)} млн"
    return f"${num(x)}"


def first_int(s: str) -> int:
    return int(re.search(r"[\d,]+", s).group(0).replace(",", ""))


class Data:
    """Все числа, которые идут на слайды; собираются один раз, до рисования."""

    def __init__(self) -> None:
        self.metrics = json.loads(METRICS.read_text(encoding="utf-8"))
        dd = csv("describe_dataset").set_index("Параметр")["Значение"]
        self.n_rows_all = first_int(dd["Строк в наборах"])
        self.n_bldg = first_int(dd["Три типа Bldg (рабочая совокупность)"])
        excl = [int(x.replace(",", "")) for x in re.findall(r"[\d,]+", dd["Исключено: VALUATION пусто / ≤ 0"])]
        self.n_excl_val = sum(excl)
        self.n_excl_dup = first_int(dd["Исключено: дубли без учёта ID"])
        self.n_train_rows = first_int(dd["Строк для обучения"])
        self.split = {
            part: (int(n.replace(",", "")), a, b)
            for part, n, a, b in re.findall(r"(train|val|test) ([\d,]+) \((\S+) … (\S+)\)", dd["Split"])
        }
        assert set(self.split) == {"train", "val", "test"}, dd["Split"]
        self.source_version = dd["Версия"]
        self.n_cols = int(dd["Исходных столбцов"])

        feats = csv("describe_features")
        self.n_num = int((feats["тип"] == "числовой").sum())
        self.n_cat = int((feats["тип"] == "категориальный").sum())
        m = re.search(r"всего токенов (\d+)", dd["Признаки модели"])
        self.n_tokens = int(m.group(1))
        m = re.search(r"словарь ([\d,]+)", pick(feats, признак="WORK_DESC (слова)")["преобразование"])
        self.vocab = int(m.group(1).replace(",", ""))
        self.excluded = csv("describe_excluded").groupby("группа")["столбец"].apply(list).to_dict()
        self.target = pick(csv("describe_target"), часть="all")

        susp = csv("labels_suspicious")
        self.susp = {p: susp[susp["часть"] == p].set_index("check") for p in ("train", "val", "test")}
        tr = self.susp["train"]
        self.r2 = {p: self.susp[p].loc["служебные оценки R2 (исключены из метрик модели)"] for p in self.susp}
        self.n_nor2 = {p: int(self.susp[p].loc["|robust z| > 3.5", "n"]) for p in self.susp}
        self.ler = tr.loc["|robust z| > 3.5"]
        self.big = tr.loc["|r| > ln 10 (ошибка > 10 раз)"]
        self.susp_service = tr.loc["|robust z| > 3.5 среди служебных (порог — по строкам без R2)"]
        self.mad = float(self.ler["MAD_r"])
        self.oof = pick(csv("labels_oof_quality"), метрика="RMSLE OOF (train без R2)")
        sv = csv("labels_service_values").set_index("check")
        self.share_501 = sv.loc["VALUATION = $501"]
        self.share_r2 = sv.loc["служебные оценки R2: {$501, $301} ∪ {≤ $10}"]
        dep = csv("labels_model_dependence")
        self.jaccard = float(dep["jaccard_susp"].dropna().iloc[0])
        self.sq_share = float(dep.iloc[0]["доля квадратичной ошибки у подозрительных"])
        w20 = csv("labels_worst20").set_index("PERMIT_NBR")
        self.examples = [(w20.loc[p], cat) for p, cat in EXAMPLE_PERMITS.items()]
        self.z_max = float(w20["z"].abs().max())

        co = csv("completeness_objects").set_index("метрика")
        self.miss_cells = co.loc["доля пропущенных ячеек в используемых признаках"]
        self.miss_objects = co.loc["доля объектов хотя бы с одним пропуском в используемых признаках"]
        self.miss_sqft_new = co.loc["доля пропусков SQUARE_FOOTAGE у Bldg-New"]
        cbt = csv("completeness_by_type").set_index("столбец")
        self.miss_sqft_alter = float(cbt.loc["SQUARE_FOOTAGE", "Bldg-Alter/Repair"])

        va = csv("validity").set_index("check")
        self.validity = va
        self.valid_any = pick(csv("validity_overall"), метрика="доля объектов с ≥ 1 нарушением")

        un = csv("uniqueness").set_index("check")
        self.dup_noid = un.loc["дубли без учёта ID (PERMIT_NBR, REFRESH_TIME)"]
        self.dup_exact = un.loc["точные дубли (все столбцы)"]
        self.near = un.loc["близкие дубли: тот же APN, Jaccard ≥ 0.9, стоимость ±10%"]
        self.near_full = un.loc["близкие дубли, только на полных (незамаскированных) APN"]
        self.masked = un.loc["APN замаскирован («…***», совпадение = тот же квартал)"]
        self.suppl = un.loc["дополнительные разрешения («Supplemental…» в описании)"]
        ex = csv("uniqueness_near_dup_examples")
        self.near_example = ex[~ex["APN замаскирован"] & (ex["VALUATION_a"] != ex["VALUATION_b"])].iloc[0]

        self.leak = csv("leakage")
        self.bins = csv("balance_target_bins").set_index("часть")
        bs = csv("balance_subgroups")
        self.sub = {(r["значение"], r["часть"]): r for _, r in bs.iterrows()}
        self.chi2 = csv("balance_subgroups_chi2").set_index("признак")

        mmd = csv("shift_mmd")
        self.mmd = pick(mmd, сравнение="train vs test", представление="признаки без времени")
        self.mmd_ctrl = pick(mmd, сравнение="train_A vs train_B", представление="признаки без времени")
        c2 = csv("shift_c2st")
        self.c2st_lr = pick(c2, сравнение="train vs test", время="нет", классификатор="логистическая регрессия")
        self.c2st_mlp = pick(c2, сравнение="train vs test", время="нет", классификатор="MLP (64)")
        self.c2st_ctrl = pick(c2, сравнение="train_A vs train_B", время="нет", классификатор="MLP (64)")
        self.ks = pick(csv("shift_ks_target"), сравнение="train vs test")
        self.c2st_groups = csv("shift_feature_c2st")
        df = csv("shift_drift_facts")
        self.drift = {(r["check"], r["часть"]): r for _, r in df.iterrows()}
        self.vocab_year = csv("shift_text_vocab_by_year").set_index("год")
        self.err_year = csv("shift_error_by_year")

        an = csv("anomalies")
        self.anom_test = pick(an, check="IsolationForest: score ≤ 1-й перцентиль train", часть="test")
        self.anom_target_test = pick(an, check="выброс log-цели (|robust z| > 3.5, медиана/MAD train)", часть="test")
        ae = csv("anomalies_error").set_index("группа test")
        self.err_anom = ae.loc["аномалии IsolationForest"]
        self.err_rest = ae.loc["остальные"]
        ap = csv("anomalies_profile")
        self.anom_new_add = sum(float(pick(ap, check=f"PERMIT_TYPE = {t}", часть="test", группа="аномалии")["value"])
                                for t in ("Bldg-New", "Bldg-Addition"))
        self.anom_adu = pick(ap, check="ADU в описании работ", часть="test", группа="аномалии")
        self.rest_adu = pick(ap, check="ADU в описании работ", часть="test", группа="остальные")

    # производные величины, которые встречаются на нескольких слайдах
    @property
    def n_after_r2(self) -> int:
        return self.n_train_rows - int(self.share_r2["k"])

    def drift_share(self, check: str, part: str) -> float:
        return float(self.drift[(check, part)]["value"])

    def unk(self, year: int) -> float:
        return float(self.vocab_year.loc[year, "доля UNK среди слов"])


# ---------------------------------------------------------------- примитивы оформления

def _run(p, text: str, size: float, color: RGBColor, bold: bool = False, italic: bool = False) -> None:
    r = p.add_run()
    r.text = text
    f = r.font
    f.name, f.size, f.bold, f.italic = FONT, Pt(size), bold, italic
    f.color.rgb = color


def rich(p, text: str, size: float, color: RGBColor, bold: bool = False) -> None:
    """Абзац с разметкой **жирного** фрагмента."""
    for i, part in enumerate(re.split(r"(?<!\*)\*\*(?!\*)", text)):
        if part:
            _run(p, part, size, color, bold=bold or i % 2 == 1)


def text_box(slide, x, y, w, h, paras, size=14, color=NAVY, bold=False, align=PP_ALIGN.LEFT,
             anchor=MSO_ANCHOR.TOP, space=4, bullet=False, shape=None, fill=None, margin=0.05):
    """Текстовая рамка (или фигура shape с текстом). paras — список строк; элемент может быть
    кортежем (строка, {size, color, bold, align, space}) для отдельного оформления абзаца."""
    if shape is None:
        sh = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    else:
        sh = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
        sh.shadow.inherit = False
        sh.line.fill.background()
        if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
            sh.adjustments[0] = 0.12
    if fill is not None:
        sh.fill.solid()
        sh.fill.fore_color.rgb = fill
    elif shape is not None:
        sh.fill.background()
    tf = sh.text_frame
    tf.word_wrap = True
    tf.auto_size = None
    m = Inches(margin)
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = m
    tf.vertical_anchor = anchor
    for i, item in enumerate(paras):
        txt, opt = (item, {}) if isinstance(item, str) else item
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = opt.get("align", align)
        p.space_after = Pt(opt.get("space", space))
        b = opt.get("bullet", bullet)
        rich(p, ("•  " if b else "") + txt, opt.get("size", size), opt.get("color", color), opt.get("bold", bold))
    return sh


def header(slide, title: str, tag: str | None, page: str) -> None:
    """Синяя шапка с заголовком-выводом, метка исследуемого свойства и номер слайда."""
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(SW), Inches(1.0))
    bar.fill.solid()
    bar.fill.fore_color.rgb = BLUE
    bar.line.fill.background()
    bar.shadow.inherit = False
    tag_w = 0.0
    if tag:
        tag_w = 0.25 + 0.115 * len(tag)
        text_box(slide, SW - 0.45 - tag_w, 0.3, tag_w, 0.4, [tag], size=12, color=NAVY, bold=True,
                 align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, shape=MSO_SHAPE.ROUNDED_RECTANGLE, fill=LIGHT)
    text_box(slide, 0.45, 0.08, SW - 0.9 - tag_w - 0.2, 0.84, [title], size=22, color=WHITE, bold=True,
             anchor=MSO_ANCHOR.MIDDLE)
    text_box(slide, SW - 2.2, SH - 0.37, 1.8, 0.3, [page], size=11, color=MID, align=PP_ALIGN.RIGHT, space=0)


def note(slide, x, y, w, h, text: str) -> None:
    """Сноска мелким серым шрифтом: совокупность, метод CI, оговорки."""
    text_box(slide, x, y, w, h, [text], size=10.5, color=GREY, space=0)


def stat(slide, x, y, w, h, value: str, label: str, sub: str = "", vsize: float = 24) -> None:
    """Плашка с крупным числом, подписью и CI."""
    paras = [(value, {"size": vsize, "bold": True, "color": BLUE, "space": 2}),
             (label, {"size": 11, "color": NAVY, "space": 2})]
    if sub:
        paras.append((sub, {"size": 10, "color": GREY}))
    text_box(slide, x, y, w, h, paras, shape=MSO_SHAPE.ROUNDED_RECTANGLE, fill=LIGHT, margin=0.1,
             anchor=MSO_ANCHOR.MIDDLE)


def flow(slide, x, y, w, h, steps: list[str], size: float = 12, gap: float = 0.32) -> None:
    """Горизонтальная схема «шаг → шаг → …»."""
    bw = (w - gap * (len(steps) - 1)) / len(steps)
    for i, s in enumerate(steps):
        bx = x + i * (bw + gap)
        text_box(slide, bx, y, bw, h, [s], size=size, color=NAVY, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE,
                 shape=MSO_SHAPE.ROUNDED_RECTANGLE, fill=LIGHT, margin=0.08)
        if i < len(steps) - 1:
            text_box(slide, bx + bw, y, gap, h, ["→"], size=18, color=MID, bold=True, align=PP_ALIGN.CENTER,
                     anchor=MSO_ANCHOR.MIDDLE, margin=0)


def table(slide, x, y, w, rows: list[list[str]], col_w: list[float], size: float = 11,
          row_h: float = 0.36, head_size: float | None = None, bold_first_col: bool = False):
    """Таблица: первая строка — шапка (тёмно-синяя), остальные — чередование белого и светлого."""
    nr, nc = len(rows), len(rows[0])
    gf = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(row_h * nr))
    t = gf.table
    total = sum(col_w)
    for j, cw in enumerate(col_w):
        t.columns[j].width = Inches(w * cw / total)
    for i in range(nr):
        t.rows[i].height = Inches(row_h)
        for j in range(nc):
            cell = t.cell(i, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = NAVY if i == 0 else (WHITE if i % 2 else PALE)
            cell.margin_left = cell.margin_right = Inches(0.07)
            cell.margin_top = cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            if i == 0:
                rich(p, rows[i][j], head_size or size, WHITE, bold=True)
            else:
                rich(p, rows[i][j], size, NAVY, bold=bold_first_col and j == 0)
    return t


def picture(slide, path: Path, x, y, w, h, crop: tuple[float, float, float, float] | None = None,
            tmp: Path | None = None):
    """Картинка, вписанная в рамку (x, y, w, h) с сохранением пропорций и по центру рамки.
    crop — доли (x0, y0, x1, y1): вырезать одну панель из многопанельного графика."""
    if crop is not None:
        img = Image.open(path)
        W, H = img.size
        x0, y0, x1, y1 = crop
        path_c = tmp / f"{path.stem}_{x0:.2f}_{x1:.2f}.png"
        img.crop((int(W * x0), int(H * y0), int(W * x1), int(H * y1))).save(path_c)
        path = path_c
    W, H = Image.open(path).size
    k = min(w / W, h / H)
    pw, ph = W * k, H * k
    return slide.shapes.add_picture(str(path), Inches(x + (w - pw) / 2), Inches(y + (h - ph) / 2),
                                    Inches(pw), Inches(ph))


def style_chart(chart, legend: bool = True, size: float = 10) -> None:
    chart.font.size = Pt(size)
    chart.font.name = FONT
    chart.font.color.rgb = NAVY
    chart.has_legend = legend
    if legend:
        chart.legend.position = XL_LEGEND_POSITION.TOP
        chart.legend.include_in_layout = False
        chart.legend.font.size = Pt(size)


def color_series(chart, colors: list[RGBColor], line: bool = False) -> None:
    for s, c in zip(chart.plots[0].series, colors):
        if line:
            s.format.line.color.rgb = c
            s.format.line.width = Pt(2.25)
            s.smooth = False
            s.marker.format.fill.solid()
            s.marker.format.fill.fore_color.rgb = c
            s.marker.format.line.color.rgb = c
        else:
            s.format.fill.solid()
            s.format.fill.fore_color.rgb = c


def new_slide(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


# ---------------------------------------------------------------- слайды

def s01_title(prs, d: Data) -> None:
    s = new_slide(prs)
    bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(SW), Inches(SH))
    bg.fill.solid()
    bg.fill.fore_color.rgb = BLUE
    bg.line.fill.background()
    text_box(s, 0.9, 1.3, 11.5, 0.5, ["ПРАКТИЧЕСКАЯ РАБОТА 1 · КАЧЕСТВО ДАННЫХ"], size=16, color=LIGHT, bold=True)
    text_box(s, 0.9, 1.9, 11.5, 1.9, ["Исследование качества набора данных для обучения глубокой модели"],
             size=38, color=WHITE, bold=True)
    text_box(s, 0.9, 3.85, 11.5, 0.9,
             ["Разрешения на строительство г. Лос-Анджелес (LADBS) → оценочная стоимость работ; "
              "модель FT-Transformer (таблица + текст)"], size=18, color=LIGHT)
    text_box(s, 0.9, 5.3, 11.5, 0.4, ["КОМАНДА"], size=14, color=LIGHT, bold=True)
    text_box(s, 0.9, 5.7, 11.5, 0.9, ["Андрусенко К.А. · Синёв В.Е. · Шестакова М.А.", "гр. 1379 · ноябрь 2026"],
             size=18, color=WHITE)


def s02_task(prs, d: Data) -> None:
    s = new_slide(prs)
    header(s, "Постановка задачи и набор данных", None, f"2 / {N_MAIN}")
    rm = d.metrics["test"]["rmsle"]
    text_box(s, 0.45, 1.3, 6.3, 3.9, [
        "**Задача:** регрессия log1p(VALUATION) — оценочной стоимости работ по разрешению; "
        "сценарий — проверка адекватности оценки, поиск заниженных и завышенных.",
        f"**Модальности:** вход — таблица ({d.n_num} числовых, {d.n_cat} категориальных признаков) "
        "и текст описания работ WORK_DESC; выход — одно число.",
        f"**Источник:** открытые данные г. Лос-Анджелес (data.lacity.org), два набора LADBS "
        f"2010–2019 и 2020 — н. в., {d.n_cols} столбцов; выгрузка 30.10.2026.",
        "**Модель:** FT-Transformer (таблица + до 96 слов описания), основная метрика RMSLE — меньше лучше.",
    ], size=15, bullet=True, space=12)
    stat(s, 0.45, 5.35, 3.0, 1.25, num(rm["estimate"], 3), "RMSLE baseline на test без служебных сумм",
         f"95% CI {ci(rm['low'], rm['high'], 3)}, {rm['method']}")
    stat(s, 3.65, 5.35, 3.1, 1.25, "B = 2000", "bootstrap для CI метрик; доли — Уилсон", "seed = 42 везде",
         vsize=22)

    # воронка строк
    steps = [
        (num(d.n_rows_all), "строк в двух наборах"),
        (num(d.n_bldg), "три типа Bldg: New, Addition, Alter/Repair"),
        (num(d.n_train_rows), f"VALUATION > 0 (−{num(d.n_excl_val)}), без дублей (−{num(d.n_excl_dup)})"),
        (num(d.n_after_r2), f"без служебных сумм $501, $301, ≤ $10 (−{num(int(d.share_r2['k']))})"),
    ]
    for i, (n, lab) in enumerate(steps):
        x = 7.1 + 0.22 * i
        w = 5.8 - 0.44 * i
        text_box(s, x, 1.3 + 0.82 * i, w, 0.72, [(n, {"size": 18, "bold": True, "color": BLUE, "space": 0}),
                                                   (lab, {"size": 11})],
                 shape=MSO_SHAPE.ROUNDED_RECTANGLE, fill=LIGHT if i < 3 else PALE, margin=0.08,
                 align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)

    # хронологический split
    text_box(s, 7.1, 4.7, 5.8, 0.35, ["**Разбиение — хронологическое по дате выдачи (70 / 15 / 15)**"], size=12)
    total = sum(v[0] for v in d.split.values())
    x = 7.1
    rows = [["Часть", "Строк", "Без служебных", "Период выдачи"]]
    for part, color in (("train", BLUE), ("val", MID), ("test", NAVY)):
        n, a, b = d.split[part]
        w = 5.8 * n / total
        text_box(s, x, 5.07, w, 0.4, [part], size=12, color=WHITE, bold=True, align=PP_ALIGN.CENTER,
                 anchor=MSO_ANCHOR.MIDDLE, shape=MSO_SHAPE.RECTANGLE, fill=color, space=0)
        rows.append([part, num(n), num(d.n_nor2[part]), f"{a} … {b}"])
        x += w
    table(s, 7.1, 5.55, 5.8, rows, [0.75, 1.0, 1.45, 2.6], size=10.5, row_h=0.29)


def s03_features(prs, d: Data, tmp: Path) -> None:
    s = new_slide(prs)
    header(s, "Признаки и целевая переменная", None, f"3 / {N_MAIN}")
    rows = [["Группа", "Число", "Что входит"],
            ["Числовые", str(d.n_num), "площадь, высота, изменение числа квартир и ADU, координаты, "
                                       "дата выдачи, дни до выдачи, длина описания"],
            ["Категориальные", str(d.n_cat), "тип и подтип, назначение, конструкция, зона, ZIP, census tract, "
                                             "районы (CD, APC, CPA, CNC)"],
            ["Текст", "1 + 96", f"WORK_DESC: TF-IDF → SVD(64) и до 96 слов-токенов (словарь {num(d.vocab)})"]]
    table(s, 0.45, 1.25, 6.5, rows, [1.4, 0.8, 4.3], size=11, row_h=0.62)
    ex = d.excluded
    text_box(s, 0.45, 3.95, 6.5, 2.0, [
        f"Всего **{d.n_tokens} токенов** на объект (с [CLS]).",
        ("**Исключены из признаков:**", {"space": 2}),
        (f"идентификаторы — {', '.join(ex['id'][:3])} и др.; адрес объекта", {"bullet": True, "size": 12, "space": 2}),
        (f"признаки из будущего — {', '.join(ex['future'])}", {"bullet": True, "size": 12, "space": 2}),
        (f"константа {', '.join(ex['constant'])}; почти пустой {', '.join(ex['missing>99%'])}",
         {"bullet": True, "size": 12, "space": 2}),
    ], size=13)
    picture(s, REP / "target_distribution.png", 7.2, 1.2, 5.7, 3.4, crop=(0, 0, 0.5, 1), tmp=tmp)
    t = d.target
    text_box(s, 7.2, 4.65, 5.7, 1.4, [
        f"Медиана **{money(t['медиана'])}**, 99-й перцентиль {money(t['p99'])}, максимум {money(t['max'])}; "
        f"в log-шкале распределение близко к симметричному (асимметрия {num(t['log1p: асимметрия'], 2)}).",
        f"Пик на **$501** — {pct(d.share_501['value'])} всех меток: служебная сумма.",
    ], size=12, space=6)


def s04_completeness_validity(prs, d: Data, tmp: Path) -> None:
    s = new_slide(prs)
    header(s, "Полнота и валидность", None, f"4 / {N_MAIN}")
    text_box(s, 0.45, 1.15, 6.2, 0.35, ["**Полнота**"], size=15, color=BLUE)
    v, c = pct_ci(d.miss_cells, 2)
    stat(s, 0.45, 1.55, 3.0, 1.05, v, "пропущенных ячеек (признаки модели)", f"95% CI {c}, percentile")
    v, c = pct_ci(d.miss_objects, 2)
    stat(s, 3.6, 1.55, 3.05, 1.05, v, "объектов хотя бы с одним пропуском", f"95% CI {c}, Уилсон")
    picture(s, REP / "completeness.png", 0.45, 2.7, 6.2, 3.95, crop=(0.5, 0, 1, 1), tmp=tmp)
    note(s, 0.45, 6.65, 6.2, 0.4, "Цвет — доля пропусков признака в группе: 0 — заполнен у всех, 1 — пуст у всех.")

    text_box(s, 6.95, 1.15, 6.0, 0.35, ["**Валидность**"], size=15, color=BLUE)
    v, c = pct_ci(d.valid_any, 2)
    stat(s, 6.95, 1.55, 6.0, 1.05, v, "строк нарушают формат, диапазон или правила предметной области",
         f"95% CI {c}, Уилсон")
    va = d.validity

    def vrow(kind: str, check: str, label: str) -> list[str]:
        v_, c_ = pct_ci(va.loc[check])
        return [kind, label, f"{v_} {c_}"]

    rows = [["Вид", "Проверка", "Доля, 95% CI"],
            vrow("Формат", "ZONE содержит несколько значений", "ZONE: несколько значений"),
            vrow("Формат", "HL вне {YES, NO} (из заполненных)", "HL вне {YES, NO} (из заполненных)"),
            vrow("Диапазон", "SQUARE_FOOTAGE ≤ 0 (из заполненных)", "площадь ≤ 0 (из заполненных)"),
            vrow("Диапазон", "HEIGHT > 1100 футов (выше самого высокого здания LA)", "высота > 1 100 футов"),
            vrow("Предм. обл.", "DU_CHANGED > 2 у «1 or 2 Family Dwelling»", "> 2 новых квартир у «1–2 Family»"),
            vrow("Предм. обл.", "STATUS_DATE < ISSUE_DATE", "статус раньше выдачи"),
            vrow("Метка", "VALUATION ≤ 0", "VALUATION ≤ 0 — исключены")]
    table(s, 6.95, 2.75, 6.0, rows, [1.15, 3.0, 1.85], size=10.5, row_h=0.45)


def s05_uniqueness(prs, d: Data) -> None:
    s = new_slide(prs)
    header(s, "Уникальность", None, f"5 / {N_MAIN}")
    v, c = pct_ci(d.dup_noid, 3)
    stat(s, 0.45, 1.25, 3.0, 1.3, v, "точные дубли (все столбцы, кроме ID)", f"95% CI {c}, Уилсон")
    v, c = pct_ci(d.near)
    stat(s, 3.6, 1.25, 3.0, 1.3, v, "близкие дубли", f"95% CI {c}, Уилсон; полные APN {pct(d.near_full['value'])}")
    v, c = pct_ci(d.masked, 1)
    stat(s, 6.75, 1.25, 3.0, 1.3, v, "строк с замаскированным APN («…***»)", f"95% CI {c}, Уилсон")
    v, c = pct_ci(d.suppl, 1)
    stat(s, 9.9, 1.25, 3.0, 1.3, v, "дополнительных разрешений (Supplemental)", f"95% CI {c}, Уилсон")
    text_box(s, 0.45, 2.85, 12.4, 0.35, ["**Близкий дубль** — более поздняя строка пары, у которой одновременно:"],
             size=13)
    flow(s, 0.45, 3.3, 9.0, 1.0, ["тот же участок (APN)",
                                    "текст: Jaccard ≥ 0,9 (MinHash, символьные 5-граммы)",
                                    "стоимость отличается ≤ 10%"], size=12)
    text_box(s, 9.6, 3.3, 3.3, 1.0, ["= однотипные работы на одном участке"], size=12, bold=True, color=WHITE,
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, shape=MSO_SHAPE.ROUNDED_RECTANGLE, fill=MID)
    e = d.near_example
    rows = [["Пример пары", "Выдано", "VALUATION", "WORK_DESC"],
            [e["PERMIT_NBR_a"], e["ISSUE_a"], money(e["VALUATION_a"]), e["WORK_DESC_a"]],
            [e["PERMIT_NBR_b"], e["ISSUE_b"], money(e["VALUATION_b"]), e["WORK_DESC_b"]]]
    table(s, 0.45, 4.6, 12.4, rows, [2.0, 1.4, 1.3, 7.7], size=11, row_h=0.5)
    note(s, 0.45, 6.25, 12.4, 0.7,
         "Точные дубли — по 760 227 строкам трёх типов Bldg, остальное — по строкам с VALUATION > 0. Доли посчитаны "
         "на фиксированном наборе и точны для него; CI приводятся, когда доля трактуется как оценка для генеральной "
         "совокупности разрешений.")


def s06_leakage_balance(prs, d: Data) -> None:
    s = new_slide(prs)
    header(s, "Утечка между выборками и баланс", None, f"6 / {N_MAIN}")
    text_box(s, 0.45, 1.15, 7.0, 0.35, ["**Train-test leakage** — совпадения с train"], size=15, color=BLUE)
    checks = [("точная копия входа модели в train (без абсолютного времени)", "копия входа модели"),
              ("точная копия входа и метки в train", "копия входа и метки"),
              ("близкий дубль в train (APN + Jaccard ≥ 0,9)", "близкий текст на том же участке"),
              ("близкий дубль в train + стоимость ±10%", "то же + стоимость ±10%"),
              ("участок (APN) встречается в train", "тот же участок"),
              ("участок встречается в train, только полные APN", "тот же участок, полные APN")]
    rows = [["Совпадение", "val: k", "val: доля [CI]", "test: k", "test: доля [CI]"]]
    for chk, lab in checks:
        cells = [lab]
        for part in ("val", "test"):
            r = pick(d.leak, check=chk, часть=part)
            v, c = pct_ci(r)
            cells += [num(r["k"]), f"{v} {c}"]
        rows.append(cells)
    table(s, 0.45, 1.6, 7.0, rows, [2.5, 0.85, 1.75, 0.85, 1.75], size=10, row_h=0.6)

    text_box(s, 7.75, 1.15, 5.2, 0.35, ["**Баланс** — классы = бины стоимости"], size=15,
             color=BLUE)
    cols = [c for c in d.bins.columns if c.startswith("доля")]
    cd = CategoryChartData()
    cd.categories = [c.replace("доля ", "") for c in cols]
    for part in ("train", "val", "test"):
        cd.add_series(part, [float(d.bins.loc[part, c]) for c in cols])
    ch = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(7.75), Inches(1.55), Inches(5.2), Inches(2.9),
                            cd).chart
    style_chart(ch)
    ch.has_title = False
    color_series(ch, [BLUE, ORANGE, GREEN])
    ch.plots[0].gap_width = 60
    va = ch.value_axis
    va.tick_labels.number_format = "0%"
    va.tick_labels.number_format_is_linked = False
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = LIGHT
    ir = d.bins["imbalance_ratio (max/min бин)"]
    en = d.bins["энтропия (норм.)"]
    stat(s, 7.75, 4.6, 2.5, 1.2, num(ir["train"], 0), "imbalance ratio, train", f"val {num(ir['val'], 0)} · "
         f"test {num(ir['test'], 0)}")
    stat(s, 10.4, 4.6, 2.55, 1.2, num(en["train"], 2), "нормир. энтропия, train", f"1 — равномерно; test "
         f"{num(en['test'], 2)}")
    note(s, 0.45, 6.2, 12.4, 0.7,
         f"Leakage и баланс посчитаны по всем строкам с VALUATION > 0, включая служебные суммы: бин «< $1k» в "
         f"основном состоит из $501 ({pct(d.share_501['value'], 1)} всех меток). Метки и ошибка модели — без "
         "служебных сумм.")


def s07_labels(prs, d: Data, tmp: Path) -> None:
    s = new_slide(prs)
    header(s, "Качество разметки", None, f"7 / {N_MAIN}")
    flow(s, 0.45, 1.2, 12.4, 0.8, [
        "OOF-прогноз: 5-fold FT-Transformer на train без служебных сумм",
        "остаток r = log1p(y) − ŷ",
        "robust z = 0,6745·(r − med r) / MAD",
        "|z| > 3,5 → метка подозрительна (Iglewicz–Hoaglin)",
    ], size=12)
    v, c = pct_ci(d.ler)
    stat(s, 0.45, 2.25, 3.0, 1.2, v, "label error rate, train без служебных сумм", f"95% CI {c}, Уилсон")
    v, c = pct_ci(d.big)
    stat(s, 3.6, 2.25, 3.0, 1.2, v, "метка и прогноз расходятся > 10 раз", f"95% CI {c}, Уилсон")
    stat(s, 0.45, 3.6, 3.0, 1.2, num(d.oof["value"], 3), "RMSLE OOF-прогноза — «точность судьи»",
         f"95% CI {ci(d.oof['ci_low'], d.oof['ci_high'], 3)}, percentile")
    v, c = pct_ci(d.susp_service, 1)
    stat(s, 3.6, 3.6, 3.0, 1.2, v, f"подозрительны среди служебных сумм (против {pct(d.ler['value'], 1)})",
         f"95% CI {c}, Уилсон")
    text_box(s, 0.45, 4.95, 6.2, 1.2, [
        f"Порог относительный: при MAD = {num(d.mad, 3)} |z| > 3,5 ≈ расхождение в "
        f"{num(float(pick(csv('labels_model_dependence'), cond={'строки': 'train без R2', 'модель': 'C96 без R2 (текущий baseline)'})['порог, раз']), 1)} раза.",
        f"Подозрительные {pct(d.ler['value'], 1)} строк дают {pct(d.sq_share, 0)} квадратичной ошибки OOF.",
    ], size=11, color=GREY, space=4)
    # подпись оси x в PNG отчёта содержит внутреннее название фильтра — обрезаем её и подписываем сами
    picture(s, REP / "labels.png", 6.85, 2.2, 6.05, 3.7, crop=(0, 0, 0.33, 0.935), tmp=tmp)
    text_box(s, 6.85, 5.92, 6.05, 0.35, ["robust z остатка (OOF, train без служебных сумм)"], size=11,
             color=NAVY, align=PP_ALIGN.CENTER, space=0)
    note(s, 6.85, 6.3, 6.05, 0.45, f"Ось обрезана на ±12: крайние значения (до |z| = {num(d.z_max, 1)}) не показаны.")


def s08_worst20(prs, d: Data) -> None:
    s = new_slide(prs)
    header(s, "Качество разметки: ручная проверка", None, f"8 / {N_MAIN}")
    text_box(s, 0.45, 1.15, 12.4, 0.4, [
        "20 объектов train (без служебных сумм) с наибольшим |z| (= наименьший label quality), просмотрены вручную"], size=13)
    rows = [["Категория", "Кол-во", "Что видно"]] + [[c, str(n), t] for c, n, t in MANUAL_WORST20]
    table(s, 0.45, 1.6, 12.4, rows, [3.6, 0.9, 7.9], size=11.5, row_h=0.6, bold_first_col=True)
    rows = [["Пример", "Описание работ", "Метка", "OOF-прогноз", "z"]]
    for e, cat in d.examples:
        desc = re.sub(r"\s+", " ", str(e["WORK_DESC"])).strip()
        desc = desc if len(desc) <= 150 else desc[:147].rstrip() + "…"
        rows.append([f"{e['PERMIT_TYPE']}, {e['ISSUE_DATE'][:4]}: {cat}", desc, money(e["VALUATION"]),
                     money(e["OOF прогноз $"]), num(e["z"], 1)])
    table(s, 0.45, 4.8, 12.4, rows, [2.3, 6.6, 1.2, 1.3, 0.6], size=11, row_h=0.65)


def s09_shift(prs, d: Data) -> None:
    s = new_slide(prs)
    header(s, "Репрезентативность", None, f"9 / {N_MAIN}")
    m = d.mmd
    stat(s, 0.45, 1.25, 3.0, 1.3, num(m["mmd2_u"], 3), "MMD² train vs test, без времени",
         f"95% CI {ci(m['ci_low'], m['ci_high'], 3)}; перестановки: p ≤ {num(m['p_value'], 3)}")
    stat(s, 3.6, 1.25, 3.0, 1.3, "≈ 0", "MMD² контроля: две половины train",
         f"p = {num(d.mmd_ctrl['p_value'], 2)} — различие не значимо", vsize=22)
    c = d.c2st_mlp
    stat(s, 0.45, 2.7, 3.0, 1.3, num(c["auc"], 3), "C2ST ROC-AUC, MLP",
         f"95% CI {ci(c['auc_ci_low'], c['auc_ci_high'], 3)}, BCa")
    c = d.c2st_lr
    stat(s, 3.6, 2.7, 3.0, 1.3, num(c["auc"], 3), "C2ST ROC-AUC, логрегрессия",
         f"95% CI {ci(c['auc_ci_low'], c['auc_ci_high'], 3)}; контроль {num(d.c2st_ctrl['auc'], 2)}")
    sub = lambda v, p: " ".join(pct_ci(d.sub[(v, p)], 1))  # noqa: E731
    text_box(s, 0.45, 4.15, 6.2, 1.85, [
        "**Важные подгруппы:** состав меняется во времени (χ², p < 0,001)",
        (f"Commercial: {sub('Commercial', 'train')} в train → {sub('Commercial', 'test')} в test", {"bullet": True}),
        (f"1–2 Family: {sub('1 or 2 Family Dwelling', 'train')} → {sub('1 or 2 Family Dwelling', 'test')}",
         {"bullet": True}),
        (f"цель: KS D = {num(d.ks['KS D'], 3)} (p < 0,001) — сдвиг цели мал по величине", {"bullet": True}),
    ], size=12, space=4)
    note(s, 0.45, 6.1, 6.2, 0.7, "MMD и C2ST — подвыборки по всем строкам train и test, включая служебные "
                                 "суммы; доли подгрупп — 95% CI Уилсона.")

    g = d.c2st_groups.head(8).iloc[::-1]
    cd = CategoryChartData()
    cd.categories = [x.replace("число: ", "").replace("категория: ", "").replace("текст: ", "текст ")
                     .replace("issue_year", "год выдачи (время)") for x in g["группа признаков"]]
    cd.add_series("AUC", [float(v) for v in g["value"]])
    ch = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(6.9), Inches(1.55), Inches(6.0), Inches(4.55),
                            cd).chart
    style_chart(ch, legend=False)
    ch.has_title = False  # иначе PowerPoint подставляет имя единственного ряда
    color_series(ch, [BLUE])
    pl = ch.plots[0]
    pl.gap_width = 45
    pl.has_data_labels = True
    pl.data_labels.number_format = "0.00"
    pl.data_labels.number_format_is_linked = False
    pl.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
    pl.data_labels.font.size = Pt(10)
    va = ch.value_axis
    va.minimum_scale, va.maximum_scale = 0.5, 1.0
    va.tick_labels.number_format = "0.0"
    va.tick_labels.number_format_is_linked = False
    va.major_gridlines.format.line.color.rgb = LIGHT
    text_box(s, 6.9, 1.15, 6.0, 0.4, ["**C2ST по одной группе признаков** (AUC; 0,5 — неразличимы)"], size=12)


def s10_time_anomalies(prs, d: Data, tmp: Path) -> None:
    s = new_slide(prs)
    header(s, "Сдвиг во времени и аномалии", None, f"10 / {N_MAIN}")
    picture(s, REP / "shift_by_year.png", 0.45, 1.15, 5.2, 3.9, crop=(0.66, 0, 1, 1), tmp=tmp)
    ey = d.err_year
    t_lo, t_hi = ey[ey["часть"] == "test"]["value"].agg(["min", "max"])
    v_lo, v_hi = ey[ey["часть"] == "val"]["value"].agg(["min", "max"])
    eplan = "текст: «Eplan» (электронная подача)"
    text_box(s, 5.8, 1.2, 2.75, 3.9, [
        "**Ошибка прогноза вперёд (RMSLE):**",
        (f"val {num(v_lo, 2)}–{num(v_hi, 2)}, test {num(t_lo, 2)}–{num(t_hi, 2)}", {"bullet": True}),
        "**Что сдвинулось (train → test):**",
        (f"«Eplan» в тексте {pct(d.drift_share(eplan, 'train'), 1)} → {pct(d.drift_share(eplan, 'test'), 1)}",
         {"bullet": True}),
        (f"census tract нет в train у {pct(d.drift_share('CT: значение не встречается в train', 'test'), 1)}",
         {"bullet": True}),
        (f"район CNC нет в train у {pct(d.drift_share('CNC: значение не встречается в train', 'test'), 1)}",
         {"bullet": True}),
        (f"незнакомых слов {pct(d.unk(2022), 2)} → {pct(d.unk(2026), 2)}\u00a0(2022\u00a0→\u00a02026)",
         {"bullet": True}),
    ], size=12, space=5)
    text_box(s, 0.45, 5.2, 8.1, 0.75, [
        "Train — OOF-интерполяция, val/test — прогноз вперёд: ряды не сравниваются как один; "
        "CI на графике — BCa по годам."], size=11, color=GREY)

    text_box(s, 8.75, 1.15, 4.15, 0.35, ["**Аномалии**"], size=15, color=BLUE)
    text_box(s, 8.75, 1.5, 4.15, 1.3, [
        "**Критерий:** IsolationForest (300 деревьев, обучен на train, признаки без времени); аномалия — score "
        "ниже 1-го перцентиля train, т. е. в train ровно 1%.",
        "Второй критерий — выброс log-цели, |robust z| > 3,5."], size=11, space=3)
    v, c = pct_ci(d.anom_test)
    stat(s, 8.75, 2.6, 2.0, 1.2, v, "аномалий в test", f"95% CI {c}, Уилсон", vsize=22)
    v, c = pct_ci(d.anom_target_test)
    stat(s, 10.9, 2.6, 2.0, 1.2, v, "выбросов цели в test", f"95% CI {c}, Уилсон", vsize=22)
    ea, er = d.err_anom, d.err_rest
    stat(s, 8.75, 3.95, 4.15, 1.2, f"{num(ea['rmsle'], 2)} против {num(er['rmsle'], 2)}",
         "RMSLE на аномалиях и остальных (test без служебных сумм)",
         f"95% CI {ci(ea['ci_low'], ea['ci_high'], 2)} ({ea['ci_method']}) и "
         f"{ci(er['ci_low'], er['ci_high'], 2)} ({er['ci_method']})", vsize=20)
    note(s, 8.75, 6.25, 4.15, 0.6, f"Доли — по всем {num(d.anom_test['n'])} строкам test; RMSLE — по "
                                   f"{num(ea['n'] + er['n'])} строкам без служебных сумм.")
    text_box(s, 8.75, 5.25, 4.15, 0.95, [
        f"Аномалии — разрешения New и Addition ({pct(d.anom_new_add, 0)} аномалий test) с заполненной площадью; "
        f"ADU в описании у {pct(d.anom_adu['value'], 0)} аномалий против {pct(d.rest_adu['value'], 0)} остальных. "
        "Ошибка ниже и внутри каждого типа."], size=10.5, color=GREY)


def s11_summary(prs, d: Data) -> None:
    s = new_slide(prs)
    header(s, "Профиль качества данных", None, f"11 / {N_MAIN}")
    mc, mcc = pct_ci(d.miss_cells, 2)
    dv, dc = pct_ci(d.dup_noid, 3)
    nv, nc = pct_ci(d.near)
    lv, lc = pct_ci(d.ler)
    ir = d.bins["imbalance_ratio (max/min бин)"]
    m, c2 = d.mmd, d.c2st_mlp
    rows = [
        ["Свойство", "Метрика", "Значение", "95% CI", "Интерпретация"],
        ["Полнота", "missing rate", mc, mcc,
         "доля пропущенных ячеек в признаках модели; пропуски структурные — площадь, высота, конструкция "
         "заполняются в основном для New и Addition"],
        ["Дубликаты", "duplicate rate", f"{dv} / {nv}", f"{dc} / {nc}",
         "точные (без ID) / близкие (участок + текст + стоимость ±10%); близкие — однотипные работы на одном участке"],
        ["Качество меток", "label error rate", lv, lc,
         f"|robust z| > 3,5 по OOF-прогнозу, train без служебных сумм; сами служебные суммы подозрительны в "
         f"{pct(d.susp_service['value'], 0)} случаев и исключены"],
        ["Баланс", "imbalance ratio", num(ir["train"], 0), "—",
         f"бины стоимости по порядку величины, train; в test {num(ir['test'], 0)}; редкий хвост ≥ $10M"],
        ["Shift", "MMD / C2ST", f"{num(m['mmd2_u'], 3)} / {num(c2['auc'], 3)}",
         f"{ci(m['ci_low'], m['ci_high'], 3)} / {ci(c2['auc_ci_low'], c2['auc_ci_high'], 3)}",
         f"MMD² (p ≤ {num(m['p_value'], 3)}) / C2ST AUC (MLP), без признаков времени: train и test "
         "сильно различимы"],
    ]
    table(s, 0.45, 1.3, 12.4, rows, [1.55, 1.55, 1.6, 2.4, 5.3], size=12, row_h=0.78, head_size=13,
          bold_first_col=True)
    text_box(s, 0.45, 6.1, 12.4, 0.95, [
        "Совокупности: полнота и точные дубли — 760 227 строк трёх типов Bldg; близкие дубли, баланс и shift — "
        "строки с VALUATION > 0, включая служебные суммы; качество меток — train без служебных сумм.",
        "CI: доли — Уилсон; missing rate — bootstrap percentile; MMD — стратифицированный bootstrap percentile, "
        "p — перестановочный тест; AUC — BCa. B = 2000."], size=10.5, color=GREY, space=2)


def s12_impact(prs, d: Data) -> None:
    s = new_slide(prs)
    header(s, "Выводы для ПР2–ПР4", None, f"12 / {N_MAIN}")
    cols = [
        ("ПР2 · устойчивость к шуму", [
            "метки уже зашумлены → сравнение парным bootstrap",
            "типы шума из самих данных: пропуски и обнуление площади, округление, искажения текста",
            "результаты — по подгруппам",
        ]),
        ("ПР3 · состязательная устойчивость", [
            "ограничения допустимости — из доменных правил валидности",
            "текст описания — основное пространство атаки",
        ]),
        ("ПР4 · приватность (MIA)", [
            "privacy unit — строка или участок: участки повторяются между выборками",
            "non-members не из val/test: они отличаются по времени",
            "шумные метки дают ложные срабатывания loss-атаки",
        ]),
    ]
    w = (SW - 0.9 - 2 * 0.25) / 3
    for i, (title, items) in enumerate(cols):
        x = 0.45 + i * (w + 0.25)
        text_box(s, x, 1.35, w, 0.65, [title], size=14, bold=True, color=WHITE, anchor=MSO_ANCHOR.MIDDLE,
                 shape=MSO_SHAPE.ROUNDED_RECTANGLE, fill=BLUE, margin=0.12)
        text_box(s, x, 2.15, w, 4.3, [(t, {"bullet": True}) for t in items], size=18, space=18,
                 shape=MSO_SHAPE.ROUNDED_RECTANGLE, fill=PALE, margin=0.15)


# ---------------------------------------------------------------- проверка переполнения текста

_FONTS: dict[tuple[bool, int], ImageFont.FreeTypeFont] = {}


def _font(bold: bool, size_pt: float) -> ImageFont.FreeTypeFont:
    key = (bold, int(size_pt * 10))
    if key not in _FONTS:
        name = "arialbd.ttf" if bold else "arial.ttf"
        _FONTS[key] = ImageFont.truetype(name, size=max(1, round(size_pt * 10)))  # 1 pt = 10 px
    return _FONTS[key]


def _lines(runs: list[tuple[str, bool, float]], width_pt: float) -> tuple[int, float]:
    """Число строк абзаца при переносе по словам и максимальный кегль."""
    words: list[tuple[str, bool, float]] = []
    for t, b, sz in runs:
        for w in re.split(r"(\s+)", t):
            if w:
                words.append((w, b, sz))
    n, cur = 1, 0.0
    for w, b, sz in words:
        wl = _font(b, sz).getlength(w) / 10
        if cur + wl > width_pt and not w.isspace() and cur > 0:
            n += 1
            cur = wl
        else:
            cur += wl
    return n, max((sz for _, _, sz in runs), default=12)


def _frame_height(tf, width_pt: float) -> float:
    h = 0.0
    paras = [p for p in tf.paragraphs if p.runs]
    for k, p in enumerate(paras):
        runs = [(r.text, bool(r.font.bold), r.font.size.pt if r.font.size else 18) for r in p.runs]
        n, sz = _lines(runs, width_pt)
        h += n * sz * 1.2
        if k < len(paras) - 1 and p.space_after is not None:  # отступ после последнего абзаца не занимает места
            h += p.space_after.pt
    return h


def check_overflow(prs) -> list[str]:
    warns = []
    for i, slide in enumerate(prs.slides, 1):
        for sh in slide.shapes:
            if sh.has_text_frame and sh.text_frame.text.strip():
                tf = sh.text_frame
                w = Emu(sh.width - tf.margin_left - tf.margin_right).pt
                need = _frame_height(tf, w)
                have = Emu(sh.height - tf.margin_top - tf.margin_bottom).pt
                if need > have * 1.02:
                    warns.append(f"слайд {i}: текст «{tf.text[:40]}…» требует {need:.0f} pt, рамка {have:.0f} pt")
            if sh.has_table:
                t = sh.table
                total = 0.0
                for r_i, row in enumerate(t.rows):
                    need_row = Emu(row.height).pt
                    for c_i, cell in enumerate(row.cells):
                        w = Emu(t.columns[c_i].width - cell.margin_left - cell.margin_right).pt
                        need = _frame_height(cell.text_frame, w) + Emu(cell.margin_top + cell.margin_bottom).pt
                        need_row = max(need_row, need)
                    total += need_row
                bottom = Emu(sh.top).pt + total
                if bottom > Inches(SH - 0.45).pt:
                    warns.append(f"слайд {i}: таблица выходит за низ слайда ({bottom / 72:.2f} in)")
                elif total > Emu(sh.height).pt * 1.02:
                    warns.append(f"слайд {i}: таблица растёт с {Emu(sh.height).pt / 72:.2f} до {total / 72:.2f} in")
    return warns


# ---------------------------------------------------------------- сборка

def build(out: Path) -> list[str]:
    d = Data()
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(SW), Inches(SH)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        s01_title(prs, d)
        s02_task(prs, d)
        s03_features(prs, d, tmp)
        s04_completeness_validity(prs, d, tmp)
        s05_uniqueness(prs, d)
        s06_leakage_balance(prs, d)
        s07_labels(prs, d, tmp)
        s08_worst20(prs, d)
        s09_shift(prs, d)
        s10_time_anomalies(prs, d, tmp)
        s11_summary(prs, d)
        s12_impact(prs, d)
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
