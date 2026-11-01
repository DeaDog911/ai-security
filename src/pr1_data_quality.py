"""ПР1. Количественный профиль качества данных → reports/pr1/ (CSV-таблицы и PNG-графики).

Запуск из корня проекта:
    venv\\Scripts\\python src\\pr1_data_quality.py                 # все разделы
    venv\\Scripts\\python src\\pr1_data_quality.py --only labels    # один или несколько разделов
    venv\\Scripts\\python src\\pr1_data_quality.py --force          # пересчитать кеши (OOF, MinHash, эмбеддинги)

Разделы: describe, completeness, validity, uniqueness, leakage, balance, labels, shift, shift_features,
drift_facts, anomalies, summary.
Тяжёлые шаги кешируются в artifacts/ (oof_predictions.parquet, minhash_signatures.npy,
near_duplicate_pairs.parquet, pr1_cls_embeddings.npz) и без --force не пересчитываются.

Доверительные интервалы: доли — Уилсон; остальное — bootstrap по объектам, B = 2000
(BCa при n ≤ 60 000, иначе percentile; метод указан в таблицах). Модель не переобучается,
кроме OOF-прогнозов для оценки качества меток (5-fold на train, гиперпараметры baseline).
"""
from __future__ import annotations

import argparse
import json
import logging

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy import stats as sps  # noqa: E402

import stats as st  # noqa: E402
from config import ARTIFACTS_DIR, BOOTSTRAP_B, DATASETS, REPORTS_DIR, SEED  # noqa: E402
from data import ID_COLS, PERMIT_TYPES, SPLIT_INFO_PATH, load_dataset, load_parsed, raw_columns  # noqa: E402
from features import CAT_COLS, EXCLUDED, normalize_text  # noqa: E402
from model import DEFAULT_HP  # noqa: E402
from train_baseline import MODEL_PATH, PRED_PATH, get_logger, load_features  # noqa: E402
from training import TRAIN_HP, build_model, embed, fit, get_device, predict, set_seed  # noqa: E402

OUT = REPORTS_DIR / "pr1"
OOF_PATH = ARTIFACTS_DIR / "oof_predictions.parquet"
OOF_HIST_PATH = ARTIFACTS_DIR / "oof_history.csv"
MINHASH_PATH = ARTIFACTS_DIR / "minhash_signatures.npy"
PAIRS_PATH = ARTIFACTS_DIR / "near_duplicate_pairs.parquet"
CLS_PATH = ARTIFACTS_DIR / "pr1_cls_embeddings.npz"

Z_THRESHOLD = 3.5            # Iglewicz–Hoaglin
CLEANLAB_K = 100             # соседей в OUTRE (cleanlab.regression)
BIG_RESIDUAL = np.log(10)    # ошибка больше чем в 10 раз
JACCARD_NEAR = 0.9
VALUE_CLOSE = 0.10           # «близкая стоимость»: относительная разница ≤ 10%
NUM_PERM = 64
SHINGLE = 5
N_FOLDS = 5
MMD_N = 5000
MMD_PERM = 1000
C2ST_N = 20000
LA_BBOX = {"lat": (33.70, 34.34), "lon": (-118.67, -118.15)}  # город Лос-Анджелес с запасом
TIME_FEATURES = {"issue_year", "issue_month", "issue_dow", "issue_doy"}
SPLITS = ("train", "val", "test")

log = logging.getLogger("ai-security")


# ====================================================================== общее
def save_table(df: pd.DataFrame, name: str) -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")
    return df


def save_fig(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=120)
    plt.close(fig)


def ci_cols(ci: st.CI, prefix: str = "") -> dict:
    return {f"{prefix}value": ci.estimate, f"{prefix}ci_low": ci.low, f"{prefix}ci_high": ci.high,
            f"{prefix}ci_method": ci.method, f"{prefix}n": ci.n}


def wilson_row(name: str, mask, denom=None, **extra) -> dict:
    mask = np.asarray(mask, dtype=bool)
    if denom is not None:
        mask = mask[np.asarray(denom, dtype=bool)]
    ci = st.proportion_ci(mask)
    return {"check": name, "k": int(mask.sum()), **ci_cols(ci), **extra}


def summary_row(prop: str, metric: str, ci: st.CI | None, value=None, interp: str = "", fmt: str = "{:.4g}",
                scale: float = 1.0) -> dict:
    if ci is None:
        return {"Свойство": prop, "Метрика": metric, "Значение": fmt.format(value * scale), "95% CI": "—",
                "Метод CI": "—", "Интерпретация": interp}
    return {"Свойство": prop, "Метрика": metric, "Значение": fmt.format(ci.estimate * scale),
            "95% CI": f"[{fmt.format(ci.low * scale)}; {fmt.format(ci.high * scale)}]",
            "Метод CI": ci.method, "Интерпретация": interp}


def write_summary(section: str, rows: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"_summary_{section}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                                                  encoding="utf-8")


def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


class Ctx:
    """Ленивая загрузка данных, общих для разделов."""

    def __init__(self, force: bool = False):
        self.force = force
        self._cache = {}

    def _get(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    @property
    def raw(self) -> pd.DataFrame:  # все строки обоих наборов
        return self._get("raw", load_parsed)

    @property
    def scope(self) -> pd.DataFrame:  # три типа Bldg — рабочая совокупность
        return self._get("scope", lambda: self.raw[self.raw["PERMIT_TYPE"].isin(PERMIT_TYPES)].reset_index(drop=True))

    @property
    def clean(self) -> pd.DataFrame:  # строки для обучения со split, порядок row_id (как в features.npz)
        return self._get("clean", lambda: load_dataset().sort_values("row_id").reset_index(drop=True))

    @property
    def feats(self):
        return self._get("feats", lambda: load_features(log=log))

    @property
    def split(self) -> np.ndarray:
        return self.clean["split"].to_numpy()

    @property
    def y_log(self) -> np.ndarray:
        return np.log1p(self.clean["valuation"].to_numpy())

    @property
    def preds(self) -> pd.DataFrame:  # прогнозы baseline (log1p), выровненные по clean
        def f():
            p = pd.read_parquet(PRED_PATH).set_index("row_id")
            return p.loc[self.clean["row_id"]].reset_index()
        return self._get("preds", f)


# ====================================================================== 1. описание
def section_describe(ctx: Ctx) -> None:
    info = json.loads(SPLIT_INFO_PATH.read_text(encoding="utf-8"))
    meta, X, prep = ctx.feats
    s = prep.summary()
    w = s.get("text_words")
    words_txt = f" + {w['len']} слов-токенов" if w else ""
    n_tokens = len(s["num_features"]) + len(s["cat_features"]) + 2 + (w["len"] if w else 0)
    rows = [
        ("Источник", "; ".join(f"{v['name']} ({k}), {v['url']}" for k, v in DATASETS.items())),
        ("Версия", "выгрузка 30.10.2026, данные обновлены 25.10.2026; файлы "
                   + ", ".join(v["file"] for v in DATASETS.values())),
        ("Тип задачи", "регрессия log1p(VALUATION); метрика RMSLE (меньше — лучше), доп. MdAPE, доля в ±25%"),
        ("Модальности", "таблица (числовые и категориальные признаки) + короткий текст WORK_DESC"),
        ("Строк в наборах", f"{info['n_raw']:,} ({', '.join(f'{k}: {v:,}' for k, v in info['n_raw_by_source'].items())})"),
        ("Три типа Bldg (рабочая совокупность)", f"{info['n_permit_types']:,}"),
        ("Исключено: VALUATION пусто / ≤ 0", f"{info['n_valuation_nan']:,} / {info['n_valuation_le0']:,}"),
        ("Исключено: дубли без учёта ID", f"{info['n_dup_without_id_removed']:,}"),
        ("Строк для обучения", f"{info['n_clean']:,}"),
        ("Признаки модели", f"{len(s['num_features'])} числовых, {len(s['cat_features'])} категориальных, "
                            f"текст → SVD(64){words_txt}; всего токенов {n_tokens}"),
        ("Исходных столбцов", f"{len(raw_columns(ctx.raw))}"),
        ("Split", f"хронологический 70/15/15 по ISSUE_DATE: train {info['n_train']:,} "
                  f"({info['train_dates'][0]} … {info['train_dates'][1]}), val {info['n_val']:,} "
                  f"({info['val_dates'][0]} … {info['val_dates'][1]}), test {info['n_test']:,} "
                  f"({info['test_dates'][0]} … {info['test_dates'][1]})"),
        ("seed", str(SEED)),
    ]
    save_table(pd.DataFrame(rows, columns=["Параметр", "Значение"]), "describe_dataset")

    feat = [{"признак": c, "тип": "числовой", "преобразование": s["num_transform"][c]} for c in s["num_features"]]
    feat += [{"признак": c, "тип": "категориальный", "преобразование": f"{k} категорий (вкл. MISSING, OTHER)"}
             for c, k in zip(s["cat_features"], s["cat_cardinalities"])]
    feat += [{"признак": "WORK_DESC", "тип": "текст", "преобразование": s["text_feature"]}]
    if w:
        feat += [{"признак": "WORK_DESC (слова)", "тип": "текст",
                  "преобразование": f"до {w['len']} слов-токенов, словарь {w['vocab_size']:,} (df ≥ {w['min_df']}), "
                                    f"эмбеддинг слова + позиционный, PAD маскируется"}]
    save_table(pd.DataFrame(feat), "describe_features")
    excl = [{"группа": k, "столбец": c} for k, v in s["dropped"].items() for c in v]
    save_table(pd.DataFrame(excl), "describe_excluded")

    v = ctx.clean["valuation"]
    rows = []
    for part in ("all",) + SPLITS:
        x = v if part == "all" else v[ctx.split == part]
        q = x.quantile([0, .01, .05, .25, .5, .75, .95, .99, 1])
        lx = np.log1p(x)
        rows.append({"часть": part, "n": len(x), "min": q.iloc[0], "p1": q.iloc[1], "p5": q.iloc[2],
                     "p25": q.iloc[3], "медиана": q.iloc[4], "p75": q.iloc[5], "p95": q.iloc[6], "p99": q.iloc[7],
                     "max": q.iloc[8], "среднее": x.mean(), "log1p: среднее": lx.mean(), "log1p: std": lx.std(),
                     "log1p: асимметрия": sps.skew(lx)})
    save_table(pd.DataFrame(rows), "describe_target")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    bins = np.linspace(0, 9.5, 96)
    for part in SPLITS:
        ax[0].hist(np.log10(v[ctx.split == part]), bins=bins, density=True, histtype="step", lw=1.4, label=part)
    ax[0].axvline(np.log10(501), color="red", ls=":", lw=1, label="$501")
    ax[0].set(xlabel="log10(VALUATION, $)", ylabel="плотность", title="Распределение цели по частям")
    ax[0].legend()
    x = np.sort(np.log10(v))
    ax[1].plot(x, np.linspace(0, 1, len(x)))
    ax[1].set(xlabel="log10(VALUATION, $)", ylabel="доля", title="Эмпирическая функция распределения")
    for a in ax:
        a.grid(alpha=0.3)
    save_fig(fig, "target_distribution")
    log.info("describe: готово")


# ====================================================================== 2. полнота
def used_raw_columns(prep) -> list[str]:
    m = {"square_footage": "SQUARE_FOOTAGE", "height": "HEIGHT", "du_changed": "DU_CHANGED",
         "adu_changed": "ADU_CHANGED", "lat": "LAT", "lon": "LON", "days_to_issue": "SUBMITTED_DATE",
         "desc_len": "WORK_DESC"}
    cols = [m[c] for c in prep.num_names if c in m]
    cols += ["ZONE" if c == "zone_base" else c for c in prep.cat_names]
    return list(dict.fromkeys(cols))


def section_completeness(ctx: Ctx) -> list[dict]:
    sc = ctx.scope
    cols = raw_columns(sc)
    rows = []
    for c in cols:
        miss = sc[c].isna().to_numpy()
        rows.append({"столбец": c, "k_missing": int(miss.sum()), **ci_cols(st.proportion_ci(miss))})
    tab = pd.DataFrame(rows).sort_values("value", ascending=False)
    _, _, prep = ctx.feats
    used = used_raw_columns(prep)
    tab["используется_в_модели"] = tab["столбец"].isin(used)
    save_table(tab, "completeness_columns")

    m = sc[used].isna().to_numpy()
    any_ci = st.proportion_ci(m.any(1))
    cell_ci = st.bootstrap_ci(st.mean_stat, m.mean(1), vectorized=True)  # доля пропущенных ячеек; бутстреп по объектам
    struct_cols = ["SQUARE_FOOTAGE", "HEIGHT", "CONSTRUCTION", "DU_CHANGED", "ADU_CHANGED", "SUBMITTED_DATE", "HL", "CNC"]
    by_type = sc.groupby("PERMIT_TYPE")[struct_cols].apply(lambda g: g.isna().mean()).T
    by_sub = sc.groupby("PERMIT_SUB_TYPE")[struct_cols].apply(lambda g: g.isna().mean()).T
    tab2 = by_type.join(by_sub).reset_index().rename(columns={"index": "столбец"})
    save_table(tab2, "completeness_by_type")
    new_sqft = st.proportion_ci(sc.loc[sc["PERMIT_TYPE"] == "Bldg-New", "SQUARE_FOOTAGE"].isna())
    obj = pd.DataFrame([
        {"метрика": "доля пропущенных ячеек в используемых признаках", **ci_cols(cell_ci)},
        {"метрика": "доля объектов хотя бы с одним пропуском в используемых признаках", **ci_cols(any_ci)},
        {"метрика": "доля пропусков SQUARE_FOOTAGE у Bldg-New", **ci_cols(new_sqft)},
    ])
    save_table(obj, "completeness_objects")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.1, 1]})
    t = tab[tab["value"] > 0].sort_values("value")
    colors = np.where(t["используется_в_модели"], "C0", "lightgray")
    ax[0].barh(t["столбец"], t["value"], color=colors,
               xerr=[t["value"] - t["ci_low"], t["ci_high"] - t["value"]])
    ax[0].set(xlabel="доля пропусков", title="Пропуски по столбцам (синие — признаки модели)")
    im = ax[1].imshow(tab2.set_index("столбец").to_numpy(), cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax[1].set_xticks(range(tab2.shape[1] - 1), tab2.columns[1:], rotation=40, ha="right")
    ax[1].set_yticks(range(len(tab2)), tab2["столбец"])
    for i in range(len(tab2)):
        for j in range(tab2.shape[1] - 1):
            val = tab2.iloc[i, j + 1]
            ax[1].text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color="w" if val < 0.6 else "k")
    ax[1].set_title("Структурные пропуски по типу и подтипу")
    fig.colorbar(im, ax=ax[1], fraction=0.04)
    save_fig(fig, "completeness")
    log.info("completeness: пропущенных ячеек %s, объектов с пропуском %s", cell_ci, any_ci)
    return [
        summary_row("Полнота", "доля пропущенных ячеек (используемые признаки)", cell_ci, scale=100, fmt="{:.2f}",
                    interp="в %; пропуски в основном структурные (площадь, высота, конструкция заполняются для New/Addition)"),
        summary_row("Полнота", "доля объектов с ≥ 1 пропуском (используемые признаки)", any_ci, scale=100, fmt="{:.2f}",
                    interp="в %; почти у каждого объекта есть пропуск → модель должна работать с пропусками (эмбеддинг «пропуск»)"),
        summary_row("Полнота", "пропуск площади у Bldg-New", new_sqft, scale=100, fmt="{:.2f}",
                    interp="в %; для нового строительства площадь почти всегда заполнена"),
    ]


# ====================================================================== 3. валидность
def section_validity(ctx: Ctx) -> list[dict]:
    sc = ctx.scope
    n = len(sc)
    t = sc["PERMIT_TYPE"]
    lat, lon = sc["lat"], sc["lon"]
    has_coord = lat.notna() & lon.notna()
    out_bbox = has_coord & ~(lat.between(*LA_BBOX["lat"]) & lon.between(*LA_BBOX["lon"]))
    desc = sc["work_desc"].str.strip()
    hl = sc["HL"].dropna()
    checks = [
        ("VALUATION пусто", sc["valuation"].isna(), None, "метка"),
        ("VALUATION ≤ 0", sc["valuation"] <= 0, None, "метка"),
        ("SQUARE_FOOTAGE ≤ 0 (из заполненных)", sc["square_footage"] <= 0, sc["square_footage"].notna(), "диапазон"),
        ("SQUARE_FOOTAGE ≤ 0 у Bldg-New (из заполненных)", sc["square_footage"] <= 0,
         (t == "Bldg-New") & sc["square_footage"].notna(), "диапазон"),
        ("HEIGHT ≤ 0 (из заполненных)", sc["height"] <= 0, sc["height"].notna(), "диапазон"),
        ("HEIGHT > 1100 футов (выше самого высокого здания LA)", sc["height"] > 1100, sc["height"].notna(), "диапазон"),
        ("DU_CHANGED > 2 у «1 or 2 Family Dwelling»", sc["du_changed"] > 2,
         (sc["PERMIT_SUB_TYPE"] == "1 or 2 Family Dwelling") & sc["du_changed"].notna(), "предметная область"),
        ("DU_CHANGED < 0 у Bldg-New", sc["du_changed"] < 0, (t == "Bldg-New") & sc["du_changed"].notna(),
         "предметная область"),
        ("ISSUE_DATE < SUBMITTED_DATE", sc["issue_date"] < sc["submitted_date"], sc["submitted_date"].notna(), "даты"),
        ("STATUS_DATE < ISSUE_DATE", sc["status_date"] < sc["issue_date"], None, "даты"),
        ("COFO_DATE < ISSUE_DATE", sc["cofo_date"] < sc["issue_date"], sc["cofo_date"].notna(), "даты"),
        ("координаты вне bbox Лос-Анджелеса (из заполненных)", out_bbox, has_coord, "диапазон"),
        ("координаты пропущены", ~has_coord, None, "полнота/формат"),
        ("WORK_DESC короче 10 символов", desc.str.len() < 10, None, "текст"),
        ("ZONE содержит несколько значений", sc["ZONE"].fillna("").str.contains(","), None, "формат"),
        ("CT содержит несколько значений", sc["CT"].fillna("").str.contains(","), None, "формат"),
        ("HL вне {YES, NO} (из заполненных)", ~sc["HL"].isin(["YES", "NO"]), sc["HL"].notna(), "формат"),
        ("ZIP_CODE не из 5 цифр", ~sc["ZIP_CODE"].fillna("").str.fullmatch(r"\d{5}"), None, "формат"),
    ]
    rows, any_viol = [], np.zeros(n, bool)
    for name, mask, denom, kind in checks:
        mask = mask.fillna(False).to_numpy(bool)
        d = None if denom is None else denom.fillna(False).to_numpy(bool)
        rows.append(wilson_row(name, mask, d, тип=kind, знаменатель=int(n if d is None else d.sum())))
        if kind != "полнота/формат":
            any_viol |= mask & (True if d is None else d)
    tab = pd.DataFrame(rows)
    save_table(tab, "validity")
    any_ci = st.proportion_ci(any_viol)
    no_label_ci = st.proportion_ci(any_viol[(sc["valuation"] > 0).to_numpy()])
    save_table(pd.DataFrame([{"метрика": "доля объектов с ≥ 1 нарушением", **ci_cols(any_ci)},
                             {"метрика": "то же среди VALUATION > 0", **ci_cols(no_label_ci)}]), "validity_overall")
    hl_vals = hl.value_counts().head(10).reset_index()
    save_table(hl_vals, "validity_hl_values")

    fig, ax = plt.subplots(figsize=(9, 6))
    tt = tab.sort_values("value")
    ax.barh(tt["check"], tt["value"] * 100, xerr=[(tt["value"] - tt["ci_low"]) * 100, (tt["ci_high"] - tt["value"]) * 100])
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set(xlabel="доля нарушений, % (симметричная лог-шкала)", title="Проверки валидности (95% CI Уилсона)")
    ax.grid(alpha=0.3, axis="x")
    save_fig(fig, "validity")
    log.info("validity: доля объектов с нарушением %s", any_ci)
    le0 = st.CI(**{**st.proportion_ci((sc["valuation"] <= 0).fillna(False)).as_dict()})
    sq = tab.set_index("check").loc["SQUARE_FOOTAGE ≤ 0 (из заполненных)"]
    sq_ci = st.CI(sq["value"], sq["ci_low"], sq["ci_high"], "wilson", int(sq["n"]))
    return [
        summary_row("Валидность", "доля объектов с ≥ 1 нарушением формата/диапазона/правил", any_ci, scale=100,
                    fmt="{:.2f}", interp="в %; основной вклад — VALUATION ≤ 0 и нулевая площадь"),
        summary_row("Валидность", "VALUATION ≤ 0 (невалидная метка)", le0, scale=100, fmt="{:.2f}",
                    interp="в %; строки исключены из обучения"),
        summary_row("Валидность", "SQUARE_FOOTAGE ≤ 0 среди заполненных", sq_ci, scale=100, fmt="{:.2f}",
                    interp="в %; ноль используется как «не указано» — шум в важном признаке"),
    ]


# ====================================================================== 4. уникальность (+ MinHash)
def _shingles(t: str) -> list[bytes]:
    if len(t) < SHINGLE:
        return [t.encode("utf-8")]
    return list({t[i:i + SHINGLE].encode("utf-8") for i in range(len(t) - SHINGLE + 1)})


def minhash_signatures(ctx: Ctx) -> np.ndarray:
    if MINHASH_PATH.exists() and not ctx.force:
        sig = np.load(MINHASH_PATH)
        if len(sig) == len(ctx.clean):
            return sig
    from datasketch import MinHash
    t0 = time.perf_counter()
    texts = normalize_text(ctx.clean["work_desc"]).tolist()
    sig = np.empty((len(texts), NUM_PERM), dtype=np.uint64)
    chunk = 50_000
    for i in range(0, len(texts), chunk):
        mh = MinHash.bulk([_shingles(t) for t in texts[i:i + chunk]], num_perm=NUM_PERM, seed=SEED)
        sig[i:i + chunk] = np.stack([m.hashvalues for m in mh])
        log.info("MinHash: %d / %d", min(i + chunk, len(texts)), len(texts))
    np.save(MINHASH_PATH, sig)
    log.info("MinHash: %.0f с", time.perf_counter() - t0)
    return sig


def masked_apn(ctx: Ctx) -> np.ndarray:
    """APN с замаскированными последними цифрами («4118013***»): совпадение означает тот же квартал, а не участок."""
    return ctx.clean["parcel"].astype("string").str.contains("*", regex=False).fillna(False).to_numpy(bool)


def near_duplicate_pairs(ctx: Ctx) -> pd.DataFrame:
    """Пары строк clean на одном участке (APN) с оценкой Jaccard текста ≥ 0,9."""
    if PAIRS_PATH.exists() and not ctx.force:
        return pd.read_parquet(PAIRS_PATH)
    sig = minhash_signatures(ctx)
    t0 = time.perf_counter()
    parcel = ctx.clean["parcel"]
    # позиции строк во всей таблице clean (строки без APN в группы не попадают; groupby по
    # отфильтрованной серии дал бы позиции внутри неё — со сдвигом после каждой строки без APN)
    groups = parcel.groupby(parcel, dropna=True).indices
    out_a, out_b, out_j = [], [], []
    for idx in groups.values():
        k = len(idx)
        if k < 2:
            continue
        s = sig[idx]
        for i0 in range(0, k, 256):
            eq = (s[i0:i0 + 256, None, :] == s[None, :, :]).mean(-1)  # (≤256, k)
            ii, jj = np.nonzero(eq >= JACCARD_NEAR)
            keep = ii + i0 < jj
            out_a.append(idx[ii[keep] + i0])
            out_b.append(idx[jj[keep]])
            out_j.append(eq[ii[keep], jj[keep]])
    pairs = pd.DataFrame({"a": np.concatenate(out_a), "b": np.concatenate(out_b),
                          "jaccard": np.concatenate(out_j).astype(np.float32)})
    v = ctx.clean["valuation"].to_numpy()
    va, vb = v[pairs["a"]], v[pairs["b"]]
    pairs["close_value"] = np.abs(va - vb) / np.maximum(va, vb) <= VALUE_CLOSE
    pa, pb = parcel.to_numpy()[pairs["a"]], parcel.to_numpy()[pairs["b"]]
    assert all(x == y for x, y in zip(pa, pb)), "в паре близких дублей разные APN"
    pairs.to_parquet(PAIRS_PATH, index=False)
    log.info("near-dup пары: %d за %.0f с", len(pairs), time.perf_counter() - t0)
    return pairs


def _later(ctx: Ctx, pairs: pd.DataFrame) -> np.ndarray:
    """Для каждой пары — индекс более поздней строки (по ISSUE_DATE, затем row_id)."""
    order = np.lexsort((ctx.clean["row_id"].to_numpy(), ctx.clean["issue_date"].to_numpy()))
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    a, b = pairs["a"].to_numpy(), pairs["b"].to_numpy()
    return np.where(rank[a] > rank[b], a, b)


def section_uniqueness(ctx: Ctx) -> list[dict]:
    sc = ctx.scope
    cols = raw_columns(sc)
    content = [c for c in cols if c not in ID_COLS]
    exact = sc.duplicated(subset=cols, keep="first").to_numpy()
    no_id = sc.duplicated(subset=content, keep="first").to_numpy()
    nbr = sc["PERMIT_NBR"].duplicated(keep="first").to_numpy()
    desc_only = sc.duplicated(subset=["parcel", "work_desc", "valuation", "PERMIT_TYPE"], keep="first").to_numpy()

    pairs = near_duplicate_pairs(ctx)
    n = len(ctx.clean)
    near = np.zeros(n, bool)
    near[_later(ctx, pairs[pairs["close_value"]])] = True
    near_text = np.zeros(n, bool)
    near_text[_later(ctx, pairs)] = True
    suppl = ctx.clean["work_desc"].str.contains("supplement", case=False).to_numpy()
    masked = masked_apn(ctx)
    full = pairs[pairs["close_value"] & ~masked[pairs["a"].to_numpy()]]
    near_full = np.zeros(n, bool)
    near_full[_later(ctx, full)] = True

    rows = [
        wilson_row("точные дубли (все столбцы)", exact, совокупность="три типа Bldg"),
        wilson_row("дубли без учёта ID (PERMIT_NBR, REFRESH_TIME)", no_id, совокупность="три типа Bldg"),
        wilson_row("повторы PERMIT_NBR", nbr, совокупность="три типа Bldg"),
        wilson_row("совпадают участок + текст + стоимость + тип", desc_only, совокупность="три типа Bldg"),
        wilson_row(f"близкие дубли: тот же APN, Jaccard ≥ {JACCARD_NEAR}, стоимость ±{int(VALUE_CLOSE * 100)}%",
                   near, совокупность="строки для обучения"),
        wilson_row(f"близкий текст на том же APN (Jaccard ≥ {JACCARD_NEAR}, без условия на стоимость)", near_text,
                   совокупность="строки для обучения"),
        wilson_row("дополнительные разрешения («Supplemental…» в описании)", suppl, совокупность="строки для обучения"),
        wilson_row("APN замаскирован («…***», совпадение = тот же квартал)", masked, совокупность="строки для обучения"),
        wilson_row("близкие дубли, только на полных (незамаскированных) APN", near_full,
                   совокупность="строки для обучения"),
    ]
    tab = pd.DataFrame(rows)
    save_table(tab, "uniqueness")

    # примеры близких дублей для ручной проверки
    ex = pairs[pairs["close_value"]].sample(min(15, int(pairs["close_value"].sum())), random_state=SEED)
    c = ctx.clean
    save_table(pd.DataFrame({
        "PERMIT_NBR_a": c["PERMIT_NBR"].to_numpy()[ex["a"]], "PERMIT_NBR_b": c["PERMIT_NBR"].to_numpy()[ex["b"]],
        "APN": c["parcel"].to_numpy()[ex["a"]], "APN замаскирован": masked[ex["a"]], "jaccard": ex["jaccard"].to_numpy(),
        "VALUATION_a": c["valuation"].to_numpy()[ex["a"]], "VALUATION_b": c["valuation"].to_numpy()[ex["b"]],
        "ISSUE_a": c["issue_date"].dt.date.to_numpy()[ex["a"]], "ISSUE_b": c["issue_date"].dt.date.to_numpy()[ex["b"]],
        "WORK_DESC_a": c["work_desc"].str[:150].to_numpy()[ex["a"]],
        "WORK_DESC_b": c["work_desc"].str[:150].to_numpy()[ex["b"]],
    }), "uniqueness_near_dup_examples")

    # $501 и дополнительные разрешения
    v501 = (c["valuation"] == 501).to_numpy()
    save_table(pd.DataFrame([
        wilson_row("$501 среди дополнительных разрешений", v501, suppl),
        wilson_row("$501 среди остальных", v501, ~suppl),
    ]), "uniqueness_supplemental_501")
    log.info("uniqueness: дубли без ID %d, близкие дубли %d", no_id.sum(), near.sum())
    r = tab.set_index("check")
    to_ci = lambda row: st.CI(row["value"], row["ci_low"], row["ci_high"], "wilson", int(row["n"]))  # noqa: E731
    return [
        summary_row("Уникальность", "duplicate rate (дубли без учёта ID)", st.proportion_ci(no_id), scale=100,
                    fmt="{:.3f}", interp="в %; точных дублей нет (PERMIT_NBR уникален); дубли удалены до split"),
        summary_row("Уникальность", "near-duplicate rate (APN + MinHash Jaccard ≥ 0,9 + стоимость ±10%)",
                    to_ci(r.iloc[4]), scale=100, fmt="{:.2f}",
                    interp="в %; повторные разрешения на те же работы — часть может попасть в разные части split"),
        summary_row("Уникальность", "доля дополнительных разрешений (Supplemental)", to_ci(r.iloc[6]), scale=100,
                    fmt="{:.2f}", interp="в %; связаны с основным разрешением, стоимость часто служебная ($501)"),
    ]


# ====================================================================== 5. leakage
def feature_key(ctx: Ctx) -> pd.Series:
    """Хеш входных данных модели без абсолютного времени (дата выдачи/подачи → срок рассмотрения)."""
    _, _, prep = ctx.feats
    c = ctx.clean
    cols = [x for x in used_raw_columns(prep) if x not in ("SUBMITTED_DATE", "WORK_DESC")]
    d = c[cols].astype("string").fillna("<NA>")
    d["days"] = (c["issue_date"] - c["submitted_date"]).dt.days.astype("string").fillna("<NA>")
    d["text"] = normalize_text(c["work_desc"])
    return pd.util.hash_pandas_object(d, index=False)


def section_leakage(ctx: Ctx) -> list[dict]:
    c, split = ctx.clean, ctx.split
    key = feature_key(ctx).to_numpy()
    key_y = pd.util.hash_pandas_object(pd.DataFrame({"k": key, "v": c["valuation"].to_numpy()}), index=False).to_numpy()
    tr = split == "train"
    pairs = near_duplicate_pairs(ctx)
    a, b = pairs["a"].to_numpy(), pairs["b"].to_numpy()
    masked = masked_apn(ctx)
    rows, summ = [], []
    for part in ("val", "test"):
        m = split == part
        in_tr = np.isin(key, key[tr])
        in_tr_y = np.isin(key_y, key_y[tr])
        # близкий дубль в train: пара, где одна строка в part, другая в train
        nd = np.zeros(len(c), bool)
        sel = (split[a] == "train") & (split[b] == part)
        nd[b[sel]] = True
        sel = (split[b] == "train") & (split[a] == part)
        nd[a[sel]] = True
        ndv = np.zeros(len(c), bool)
        pv = pairs["close_value"].to_numpy()
        for x, y_ in ((a, b), (b, a)):
            s = (split[x] == "train") & (split[y_] == part) & pv
            ndv[y_[s]] = True
        parcel_tr = set(c.loc[tr, "parcel"].dropna())
        shared = c["parcel"].isin(parcel_tr).to_numpy()
        rows += [
            wilson_row("точная копия входа модели в train (без абсолютного времени)", in_tr, m, часть=part),
            wilson_row("точная копия входа и метки в train", in_tr_y, m, часть=part),
            wilson_row("близкий дубль в train (APN + Jaccard ≥ 0,9)", nd, m, часть=part),
            wilson_row("близкий дубль в train + стоимость ±10%", ndv, m, часть=part),
            wilson_row("участок (APN) встречается в train", shared, m, часть=part),
            wilson_row("участок встречается в train, только полные APN", shared, m & ~masked, часть=part),
        ]
    tab = pd.DataFrame(rows)
    save_table(tab, "leakage")
    excl = [{"группа": k, "столбец": col} for k, v in EXCLUDED.items() for col in v]
    save_table(pd.DataFrame(excl), "leakage_excluded_features")
    t = tab[tab["часть"] == "test"].set_index("check")
    to_ci = lambda row: st.CI(row["value"], row["ci_low"], row["ci_high"], "wilson", int(row["n"]))  # noqa: E731
    log.info("leakage: test с точной копией в train %s", to_ci(t.iloc[0]))
    return [
        summary_row("Train-test leakage", "test: точная копия входа модели в train", to_ci(t.iloc[0]), scale=100,
                    fmt="{:.2f}", interp="в %; типовые разрешения с одинаковым описанием — не утечка объекта, но завышают качество"),
        summary_row("Train-test leakage", "test: близкий дубль в train (APN + текст + стоимость ±10%)",
                    to_ci(t.iloc[3]), scale=100, fmt="{:.3f}", interp="в %; повтор того же объекта/работ"),
        summary_row("Train-test leakage", "test: участок встречается в train", to_ci(t.iloc[4]), scale=100,
                    fmt="{:.1f}", interp="в %; важно для MIA (ПР4): при privacy unit = участок выборки не независимы"),
    ]


# ====================================================================== 6. баланс
VAL_BINS = [0, 1e3, 1e4, 1e5, 1e6, 1e7, np.inf]
VAL_LABELS = ["< $1k", "$1k–10k", "$10k–100k", "$100k–1M", "$1M–10M", "≥ $10M"]


def section_balance(ctx: Ctx) -> list[dict]:
    c, split = ctx.clean, ctx.split
    binned = pd.cut(c["valuation"], VAL_BINS, labels=VAL_LABELS, right=False)
    counts = pd.crosstab(binned, split)[list(SPLITS)]
    rows = []
    for part in SPLITS:
        cnt = counts[part].to_numpy()
        p = cnt / cnt.sum()
        h = -(p[p > 0] * np.log(p[p > 0])).sum()
        rows.append({"часть": part, "n": int(cnt.sum()), "imbalance_ratio (max/min бин)": cnt.max() / cnt.min(),
                     "энтропия (норм.)": h / np.log(len(cnt)),
                     **{f"доля {lab}": v for lab, v in zip(VAL_LABELS, p)}})
    tab = pd.DataFrame(rows)
    save_table(tab, "balance_target_bins")

    p99 = np.quantile(c.loc[split == "train", "valuation"], 0.99)
    tail = [wilson_row(f"VALUATION > p99(train) = ${p99:,.0f}", c["valuation"] > p99, split == part, часть=part)
            for part in SPLITS]
    save_table(pd.DataFrame(tail), "balance_tail")

    grp = []
    for col in ("PERMIT_TYPE", "PERMIT_SUB_TYPE"):
        for part in SPLITS:
            m = split == part
            for v, k in c.loc[m, col].value_counts().items():
                ci = st.wilson_ci(k, m.sum())
                grp.append({"признак": col, "значение": v, "часть": part, "k": k, **ci_cols(ci)})
    gt = pd.DataFrame(grp)
    save_table(gt, "balance_subgroups")
    chi = {col: sps.chi2_contingency(pd.crosstab(c[col], split)) for col in ("PERMIT_TYPE", "PERMIT_SUB_TYPE")}
    save_table(pd.DataFrame([{"признак": k, "chi2": v.statistic, "p_value": v.pvalue, "dof": v.dof}
                             for k, v in chi.items()]), "balance_subgroups_chi2")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    (counts / counts.sum()).plot.bar(ax=ax[0], rot=30)
    ax[0].set(ylabel="доля", title="Цель по бинам стоимости")
    for i, col in enumerate(("PERMIT_TYPE", "PERMIT_SUB_TYPE")):
        pv = gt[gt["признак"] == col].pivot(index="часть", columns="значение", values="value").loc[list(SPLITS)]
        pv.plot.bar(stacked=True, ax=ax[i + 1], rot=0)
        ax[i + 1].set(ylabel="доля", title=col)
        ax[i + 1].legend(fontsize=7, loc="lower right")
    save_fig(fig, "balance")
    tr = tab.iloc[0]
    log.info("balance: IR(train) %.1f, энтропия %.3f", tr.iloc[2], tr.iloc[3])
    com = gt[(gt["значение"] == "Commercial")].set_index("часть")
    return [
        summary_row("Баланс", "imbalance ratio бинов стоимости (train, max/min)", None, value=tr.iloc[2],
                    fmt="{:.1f}", interp=f"бины по порядку величины; ≥ $10M — редкий хвост; в test IR {tab.iloc[2, 2]:.1f}"),
        summary_row("Баланс", "нормированная энтропия бинов (train)", None, value=tr.iloc[3], fmt="{:.3f}",
                    interp="1 — равномерно; умеренный дисбаланс в сторону $1k–100k"),
        summary_row("Баланс", "доля Commercial (test)",
                    st.CI(com.loc["test", "value"], com.loc["test", "ci_low"], com.loc["test", "ci_high"], "wilson",
                          int(com.loc["test", "n"])), scale=100, fmt="{:.2f}",
                    interp=f"в %; train {100 * com.loc['train', 'value']:.2f}% — состав подгрупп меняется во времени "
                           f"(χ² p = {chi['PERMIT_SUB_TYPE'].pvalue:.1e})"),
    ]


# ====================================================================== 7. качество меток
def compute_oof(ctx: Ctx) -> pd.DataFrame:
    """OOF-прогнозы на train: 5-fold, гиперпараметры baseline, early stopping по val (как у baseline)."""
    if OOF_PATH.exists() and not ctx.force:
        return pd.read_parquet(OOF_PATH)
    from sklearn.model_selection import KFold
    meta, X, prep = ctx.feats
    split = meta["split"].to_numpy()
    y = meta["y_log"].to_numpy()
    tr, va = np.flatnonzero(split == "train"), np.flatnonzero(split == "val")
    sub = lambda ix: {k: v[ix] for k, v in X.items()}  # noqa: E731
    device = get_device()
    oof = np.full(len(tr), np.nan)
    fold = np.full(len(tr), -1)
    hist = []
    for f, (a, b) in enumerate(KFold(N_FOLDS, shuffle=True, random_state=SEED).split(tr)):
        t0 = time.perf_counter()
        log.info("OOF fold %d/%d: train %d, held-out %d", f + 1, N_FOLDS, len(a), len(b))
        set_seed(SEED)
        model = build_model(X, prep.cat_cardinalities, DEFAULT_HP, **prep.model_kw)
        model, h, best = fit(model, sub(tr[a]), (y[tr[a]] - prep.y_mean) / prep.y_std, sub(va), y[va],
                             prep.y_mean, prep.y_std, device, TRAIN_HP, log)
        oof[b] = predict(model, sub(tr[b]), device) * prep.y_std + prep.y_mean
        fold[b] = f
        hist += [{**r, "fold": f, "best_epoch": best} for r in h]
        log.info("OOF fold %d: RMSLE held-out %.4f, %.0f с", f + 1, st.rmsle(y[tr[b]], oof[b]),
                 time.perf_counter() - t0)
        del model
        torch.cuda.empty_cache()
    out = pd.DataFrame({"row_id": meta["row_id"].to_numpy()[tr], "fold": fold, "y_log": y[tr], "oof_log": oof})
    out.to_parquet(OOF_PATH, index=False)
    pd.DataFrame(hist).to_csv(OOF_HIST_PATH, index=False)
    return out


def robust_z(r: np.ndarray) -> tuple[np.ndarray, float, float]:
    med = np.median(r)
    mad = np.median(np.abs(r - med))
    return 0.6745 * (r - med) / mad, float(med), float(mad)


def section_labels(ctx: Ctx) -> list[dict]:
    oof = compute_oof(ctx)
    c, split = ctx.clean, ctx.split
    tr = split == "train"
    assert np.array_equal(oof["row_id"].to_numpy(), c.loc[tr, "row_id"].to_numpy())
    # остатки: train — OOF, val/test — baseline (модель обучена только на train)
    pred = ctx.preds["pred_log"].to_numpy().copy()
    pred[tr] = oof["oof_log"].to_numpy()
    y = ctx.y_log
    r = y - pred
    z = np.empty_like(r)
    zinfo = {}
    for part in SPLITS:
        m = split == part
        z[m], med, mad = robust_z(r[m])
        zinfo[part] = {"median_r": med, "MAD_r": mad}
    susp = np.abs(z) > Z_THRESHOLD
    big = np.abs(r) > BIG_RESIDUAL
    ptype = c["PERMIT_TYPE"].to_numpy()

    rows = []
    for part in SPLITS:
        m = split == part
        src = "OOF 5-fold" if part == "train" else "baseline"
        rows.append(wilson_row(f"|robust z| > {Z_THRESHOLD}", susp, m, часть=part, прогноз=src, **zinfo[part]))
        rows.append(wilson_row("|r| > ln 10 (ошибка > 10 раз)", big, m, часть=part, прогноз=src))
        rows.append(wilson_row("  из них прогноз выше метки (метка занижена?)", (r < 0) & susp, m & susp, часть=part,
                               прогноз=src))
        for t in PERMIT_TYPES:
            rows.append(wilson_row(f"|robust z| > {Z_THRESHOLD}, {t}", susp, m & (ptype == t), часть=part, прогноз=src))
    tab = pd.DataFrame(rows)
    save_table(tab, "labels_suspicious")

    # качество OOF-прогноза (стратифицированный по PERMIT_TYPE bootstrap)
    m = tr
    oof_rmsle = st.stratified_bootstrap_ci(st.rmsle, y[m], pred[m], strata=ptype[m], vectorized=True)
    oof_mdape = st.stratified_bootstrap_ci(st.mdape, y[m], pred[m], strata=ptype[m], vectorized=True)
    save_table(pd.DataFrame([{"метрика": "RMSLE OOF (train)", **ci_cols(oof_rmsle)},
                             {"метрика": "MdAPE OOF (train), %", **ci_cols(oof_mdape)}]), "labels_oof_quality")

    # cleanlab: label quality score (регрессия) на OOF
    # Метод OUTRE (по умолчанию в cleanlab.regression) строит kNN с k = 0,5·n: для n ≈ 5·10⁵
    # это ~900 ГБ памяти. Вызываем ту же функцию с k = CLEANLAB_K соседей (frac_neighbors = k / n).
    from cleanlab.regression.rank import _get_outre_score_for_each_label
    t0 = time.perf_counter()
    lq = _get_outre_score_for_each_label(y[m], pred[m], frac_neighbors=CLEANLAB_K / m.sum())
    k = int(susp[m].sum())
    low_cl = np.argsort(lq)[:k]
    low_z = np.argsort(-np.abs(z[m]))[:k]
    overlap = len(np.intersect1d(low_cl, low_z)) / max(k, 1)
    rho = sps.spearmanr(lq, -np.abs(r[m])).statistic
    lq_mean = st.bootstrap_ci(st.mean_stat, lq, vectorized=True)
    save_table(pd.DataFrame([
        {"метрика": "средний label quality score (cleanlab, OOF)", **ci_cols(lq_mean)},
        {"метрика": f"пересечение {k} худших по cleanlab и по |z|", "value": overlap},
        {"метрика": "Spearman(score, −|r|)", "value": rho},
        {"метрика": "доля score < 0,1", **ci_cols(st.proportion_ci(lq < 0.1))},
    ]), "labels_cleanlab")
    log.info("cleanlab: %.0f с, пересечение %.3f", time.perf_counter() - t0, overlap)

    # служебные и округлённые значения
    v = c["valuation"].to_numpy()
    top = c["valuation"].value_counts().head(15)
    save_table(pd.DataFrame({"VALUATION": top.index, "k": top.to_numpy(), "доля": top.to_numpy() / len(c)}),
               "labels_top_values")
    v501 = v == 501
    svc = [
        wilson_row("VALUATION = $501", v501),
        wilson_row("VALUATION ≤ $100", v <= 100),
        wilson_row("VALUATION кратна $1 000", np.isclose(v % 1000, 0)),
        wilson_row("VALUATION кратна $5 000", np.isclose(v % 5000, 0)),
        wilson_row("подозрительные (|z| > 3,5) среди $501, train", susp, tr & v501),
        wilson_row("подозрительные (|z| > 3,5) среди остальных, train", susp, tr & ~v501),
        wilson_row("доля $501 среди подозрительных, train", v501, tr & susp),
    ]
    save_table(pd.DataFrame(svc), "labels_service_values")

    # диагностика случайного KFold: близкий дубль (тот же APN, Jaccard ≥ 0,9, стоимость ±10%) в другом фолде
    # «подсказывает» метку, и OOF-остаток занижен; естественный контроль — дубль в том же фолде
    pairs = near_duplicate_pairs(ctx)
    pairs = pairs[pairs["close_value"]]
    a, b = pairs["a"].to_numpy(), pairs["b"].to_numpy()
    fold = np.full(len(c), -1)
    fold[tr] = oof["fold"].to_numpy()
    both = tr[a] & tr[b]
    a, b = a[both], b[both]
    other = np.zeros(len(c), bool)
    same = np.zeros(len(c), bool)
    diff = fold[a] != fold[b]
    for x in (a, b):
        other[x[diff]] = True
        same[x[~diff]] = True
    groups = {"близкий дубль в другом фолде": tr & other, "близкий дубль только в том же фолде": tr & same & ~other,
              "без близкого дубля в train": tr & ~other & ~same}
    diag = []
    for name, g in groups.items():
        abs_r = np.abs(r[g])
        diag.append({"группа": name, "n": int(g.sum()),
                     **ci_cols(st.bootstrap_ci(st.median_stat, abs_r, vectorized=True), "median_abs_r_"),
                     **ci_cols(st.proportion_ci(susp[g]), "susp_")})
    save_table(pd.DataFrame(diag), "labels_oof_neardup")

    # зависимость критерия от модели: сравнение с прежним baseline (SVD(64), архив artifacts/baseline_svd64/)
    old_path = ARTIFACTS_DIR / "baseline_svd64" / OOF_PATH.name
    if old_path.exists():
        old = pd.read_parquet(old_path)
        assert np.array_equal(old["row_id"].to_numpy(), oof["row_id"].to_numpy())
        r_new, r_old = r[tr], (old["y_log"] - old["oof_log"]).to_numpy()
        z_old, med_old, mad_old = robust_z(r_old)
        med_new, mad_new = zinfo["train"]["median_r"], zinfo["train"]["MAD_r"]
        thr_old = Z_THRESHOLD * mad_old / 0.6745  # порог |r − median| прежней модели в log-единицах
        s_new, s_old = np.abs(z[tr]) > Z_THRESHOLD, np.abs(z_old) > Z_THRESHOLD
        dep = []
        for name, mad, med, s, rr in (("C96 (текущий baseline)", mad_new, med_new, s_new, r_new),
                                      ("SVD(64) (прежний baseline)", mad_old, med_old, s_old, r_old)):
            thr = Z_THRESHOLD * mad / 0.6745
            dep.append({"модель": name, "MAD остатков": mad, "порог |r − median|, log": thr,
                        "порог, раз": float(np.exp(thr)), **ci_cols(st.proportion_ci(s), "susp_"),
                        **ci_cols(st.proportion_ci(np.abs(rr - med) > thr_old), "susp_old_abs_thr_"),
                        **ci_cols(st.proportion_ci(np.abs(rr) > BIG_RESIDUAL), "big_"),
                        "доля квадратичной ошибки у подозрительных": float((rr[s] ** 2).sum() / (rr ** 2).sum())})
        dep.append({"модель": "Jaccard наборов подозрительных (C96 vs SVD(64))",
                    "MAD остатков": (s_new & s_old).sum() / (s_new | s_old).sum()})
        save_table(pd.DataFrame(dep), "labels_model_dependence")
        # прежние 20 худших: насколько изменились прогноз и z
        top = np.argsort(-np.abs(z_old))[:20]
        save_table(pd.DataFrame({
            "PERMIT_NBR": c.loc[tr, "PERMIT_NBR"].to_numpy()[top], "VALUATION": c.loc[tr, "valuation"].to_numpy()[top],
            "прогноз SVD(64) $": np.expm1(old["oof_log"].to_numpy()[top]).round(),
            "прогноз C96 $": np.expm1(pred[tr][top]).round(),
            "C96 / SVD(64)": np.expm1(pred[tr][top]) / np.expm1(old["oof_log"].to_numpy()[top]),
            "z SVD(64)": z_old[top], "z C96": z[tr][top],
            "WORK_DESC": c.loc[tr, "work_desc"].str[:120].to_numpy()[top]}), "labels_prev_worst20")

    # 20 худших примеров train (для ручного разбора)
    idx = np.flatnonzero(tr)[np.argsort(-np.abs(z[tr]))[:20]]
    worst = pd.DataFrame({
        "PERMIT_NBR": c["PERMIT_NBR"].to_numpy()[idx], "ISSUE_DATE": c["issue_date"].dt.date.to_numpy()[idx],
        "PERMIT_TYPE": ptype[idx], "SUB_TYPE": c["PERMIT_SUB_TYPE"].to_numpy()[idx],
        "USE_DESC": c["USE_DESC"].to_numpy()[idx], "SQUARE_FOOTAGE": c["square_footage"].to_numpy()[idx],
        "VALUATION": v[idx], "OOF прогноз $": np.round(np.expm1(pred[idx])),
        "прогноз/метка": np.expm1(pred[idx]) / v[idx], "z": z[idx], "cleanlab score": lq[np.argsort(-np.abs(z[tr]))[:20]],
        "WORK_DESC": c["work_desc"].str.slice(0, 220).to_numpy()[idx],
    })
    save_table(worst, "labels_worst20")
    pd.DataFrame({"row_id": c["row_id"], "split": split, "residual": r, "robust_z": z,
                  "suspicious": susp}).to_parquet(ARTIFACTS_DIR / "pr1_label_residuals.parquet", index=False)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].hist(z[tr], bins=np.linspace(-12, 12, 241), color="C0")
    ax[0].axvline(-Z_THRESHOLD, color="r", ls="--")
    ax[0].axvline(Z_THRESHOLD, color="r", ls="--", label=f"|z| = {Z_THRESHOLD}")
    ax[0].set(yscale="log", xlabel="robust z остатка (train, OOF)", ylabel="число объектов",
              title="Остатки OOF в robust z")
    ax[0].legend()
    hb = ax[1].hexbin(pred[tr], y[tr], gridsize=80, bins="log", cmap="viridis", mincnt=1)
    lim = [0, 20]
    ax[1].plot(lim, lim, "w--", lw=1)
    ax[1].set(xlim=lim, ylim=lim, xlabel="OOF прогноз log1p", ylabel="метка log1p", title="Метка vs OOF прогноз (train)")
    fig.colorbar(hb, ax=ax[1], fraction=0.04)
    yrs = c["issue_date"].dt.year.to_numpy()
    yr = pd.DataFrame({"год": yrs, "susp": susp, "s501": v501}).groupby("год").mean()
    ax[2].plot(yr.index, yr["susp"] * 100, "o-", label="подозрительные метки")
    ax[2].plot(yr.index, yr["s501"] * 100, "s-", label="$501")
    ax[2].set(xlabel="год выдачи", ylabel="%", title="Доля подозрительных меток и $501 по годам")
    ax[2].legend()
    ax[2].grid(alpha=0.3)
    save_fig(fig, "labels")

    t = tab[(tab["часть"] == "train")].set_index("check")
    to_ci = lambda row: st.CI(row["value"], row["ci_low"], row["ci_high"], "wilson", int(row["n"]))  # noqa: E731
    lab = to_ci(t.iloc[0])
    log.info("labels: label error rate (train, OOF) %s", lab)
    return [
        summary_row("Качество меток", f"label error rate (|robust z| > {Z_THRESHOLD}, OOF, train)", lab, scale=100,
                    fmt="{:.2f}", interp="в %; доля меток, далеко расходящихся с out-of-sample прогнозом"),
        summary_row("Качество меток", "доля ошибок > 10 раз (|r| > ln 10, OOF, train)", to_ci(t.iloc[1]), scale=100,
                    fmt="{:.2f}", interp="в %; грубые расхождения; дают большую часть квадратичной ошибки"),
        summary_row("Качество меток", "доля служебного значения $501", st.proportion_ci(v501), scale=100, fmt="{:.2f}",
                    interp="в %; минимальная/служебная оценка, по признакам не предсказуема"),
        summary_row("Качество меток", "RMSLE OOF (train), стратиф. bootstrap по PERMIT_TYPE", oof_rmsle, fmt="{:.4f}",
                    interp="качество out-of-sample прогноза, на котором построена проверка меток"),
    ]


# ====================================================================== 8. сдвиг распределений
def representation(ctx: Ctx, with_time: bool) -> np.ndarray:
    """Вектор признаков для MMD/C2ST/IsolationForest: числа + индикаторы пропуска + текст + one-hot малых категорий."""
    _, X, prep = ctx.feats
    keep = [i for i, n in enumerate(prep.num_names) if with_time or n not in TIME_FEATURES]
    parts = [X["num"][:, keep], X["num_mask"][:, keep].astype(np.float32), X["txt"]]
    for j, (name, card) in enumerate(zip(prep.cat_names, prep.cat_cardinalities)):
        if card <= 30:
            parts.append(np.eye(card, dtype=np.float32)[X["cat"][:, j]])
    return np.concatenate(parts, 1).astype(np.float32)


def mmd_test(x: np.ndarray, y: np.ndarray, device, n_perm: int = MMD_PERM, B: int = BOOTSTRAP_B,
             seed: int = SEED) -> dict:
    """Несмещённая MMD² (U-статистика, RBF, медианная эвристика), permutation p-value и bootstrap CI.

    Статистика записана через веса-счётчики a (для x) и b (для y): пары «объект сам с собой»
    (в т. ч. повторы одного объекта в bootstrap-перевыборке, где K = 1) исключаются:
        MMD²_u = (aᵀKa − Σa²)/(m² − Σa²) + (bᵀKb − Σb²)/(n² − Σb²) − 2·aᵀKb/(m·n).
    При a = b = 1 это стандартная несмещённая оценка Gretton et al. (2012).
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    z = torch.as_tensor(np.concatenate([x, y]), device=device, dtype=torch.float32)
    m, n = len(x), len(y)
    N = m + n
    d2 = torch.cdist(z, z).pow(2)
    sigma2 = d2[torch.triu(torch.ones(N, N, dtype=torch.bool, device=device), 1)].median()
    K = torch.exp(-d2 / (2 * sigma2))
    del d2

    def stat(A, Bw):  # A, Bw: (N, P) счётчики для x и y
        KA, KB = K @ A, K @ Bw
        sxx, syy, sxy = (A * KA).sum(0), (Bw * KB).sum(0), (A * KB).sum(0)
        a2, b2 = (A * A).sum(0), (Bw * Bw).sum(0)
        return (sxx - a2) / (m * m - a2) + (syy - b2) / (n * n - b2) - 2 * sxy / (m * n)

    ind = torch.cat([torch.ones(m), torch.zeros(n)]).to(device)
    obs = float(stat(ind[:, None], (1 - ind)[:, None])[0])
    perm_vals = []
    for i in range(0, n_perm, 100):
        P = min(100, n_perm - i)
        idx = torch.stack([torch.randperm(N, generator=g) for _ in range(P)], 1).to(device)
        A = ind[idx]
        perm_vals.append(stat(A, 1 - A))
    perm_vals = torch.cat(perm_vals).cpu().numpy()
    p = (1 + (perm_vals >= obs).sum()) / (1 + n_perm)
    boot = []  # стратифицированный bootstrap: перевыборка внутри x и внутри y
    for i in range(0, B, 100):
        P = min(100, B - i)
        cx = torch.stack([torch.bincount(torch.randint(0, m, (m,), generator=g), minlength=m) for _ in range(P)], 1)
        cy = torch.stack([torch.bincount(torch.randint(0, n, (n,), generator=g), minlength=n) for _ in range(P)], 1)
        A = torch.cat([cx, torch.zeros(n, P, dtype=cx.dtype)]).to(device, torch.float32)
        Bw = torch.cat([torch.zeros(m, P, dtype=cy.dtype), cy]).to(device, torch.float32)
        boot.append(stat(A, Bw))
    boot = torch.cat(boot).cpu().numpy()
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {"mmd2_u": obs, "ci_low": float(lo), "ci_high": float(hi), "ci_method": "percentile (стратиф.)",
            "p_value": float(p), "n_perm": n_perm, "perm_q95": float(np.quantile(perm_vals, 0.95)),
            "sigma2": float(sigma2), "m": m, "n": n}


def auc_vec(score, label, axis=-1):
    """ROC-AUC через ранги (Mann–Whitney), векторизовано по последней оси."""
    score, label = np.asarray(score), np.asarray(label).astype(bool)
    rk = sps.rankdata(score, axis=axis)
    n1 = label.sum(axis=axis)
    n0 = label.shape[axis] - n1
    return ((rk * label).sum(axis=axis) - n1 * (n1 + 1) / 2) / (n1 * n0)


def c2st(x: np.ndarray, y: np.ndarray, seed: int = SEED) -> list[dict]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    data = np.concatenate([x, y])
    lab = np.r_[np.zeros(len(x), int), np.ones(len(y), int)]
    a_tr, a_te, l_tr, l_te = train_test_split(data, lab, test_size=0.5, stratify=lab, random_state=seed)
    sc = StandardScaler().fit(a_tr)
    a_tr, a_te = sc.transform(a_tr), sc.transform(a_te)
    out = []
    for name, clf in (("логистическая регрессия", LogisticRegression(max_iter=3000, C=1.0)),
                      ("MLP (64)", MLPClassifier((64,), early_stopping=True, max_iter=200, random_state=seed))):
        clf.fit(a_tr, l_tr)
        s = clf.predict_proba(a_te)[:, 1]
        ci = st.stratified_bootstrap_ci(auc_vec, s, l_te, strata=l_te, vectorized=True)
        acc = st.proportion_ci((s > 0.5) == l_te)
        out.append({"классификатор": name, "auc": ci.estimate, "auc_ci_low": ci.low, "auc_ci_high": ci.high,
                    "auc_ci_method": ci.method, "accuracy": acc.estimate, "acc_ci_low": acc.low,
                    "acc_ci_high": acc.high, "n_eval": len(l_te), "_ci": ci})
    return out


def section_shift(ctx: Ctx) -> list[dict]:
    c, split = ctx.clean, ctx.split
    device = get_device()
    rng = np.random.default_rng(SEED)
    idx = {p: np.flatnonzero(split == p) for p in SPLITS}
    tr_perm = rng.permutation(idx["train"])
    samp = {"train": rng.choice(idx["train"], MMD_N, replace=False), "val": rng.choice(idx["val"], MMD_N, replace=False),
            "test": rng.choice(idx["test"], MMD_N, replace=False),
            "train_A": tr_perm[:MMD_N], "train_B": tr_perm[MMD_N:2 * MMD_N]}
    R = {False: representation(ctx, False), True: representation(ctx, True)}

    # эмбеддинги [CLS] baseline-модели для подвыборок
    if CLS_PATH.exists() and not ctx.force:
        z = np.load(CLS_PATH)
        cls = {k: z[k] for k in z.files}
    else:
        _, X, prep = ctx.feats
        model = build_model(X, prep.cat_cardinalities, DEFAULT_HP, **prep.model_kw)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        cls = {k: embed(model, {kk: vv[v] for kk, vv in X.items()}, device) for k, v in samp.items()}
        np.savez_compressed(CLS_PATH, **cls)

    # train vs *: объекты train — обучающие для baseline, поэтому в пространстве [CLS] сдвиг данных смешан
    # с подгонкой модели под train; val vs test — обе части модель не видела (сдвиг между поздними периодами)
    comps = [("train", "test"), ("train", "val"), ("train_A", "train_B"), ("val", "test")]
    rows = []
    for a, b in comps:
        for rep_name, getter in (("признаки без времени", lambda k: R[False][samp[k]]),
                                 ("признаки с временем", lambda k: R[True][samp[k]]),
                                 ("эмбеддинги [CLS] baseline", lambda k: cls[k])):
            res = mmd_test(getter(a), getter(b), device)
            rows.append({"сравнение": f"{a} vs {b}", "представление": rep_name, **res})
            log.info("MMD %s vs %s (%s): %.5f, p = %.4f", a, b, rep_name, res["mmd2_u"], res["p_value"])
    mmd = pd.DataFrame(rows)
    save_table(mmd, "shift_mmd")

    # C2ST
    rows, c2_ci = [], {}
    for a, b in (("train", "test"), ("train", "val")):
        ia = rng.choice(idx[a], C2ST_N, replace=False)
        ib = rng.choice(idx[b], C2ST_N, replace=False)
        for with_time in (False, True):
            for res in c2st(R[with_time][ia], R[with_time][ib]):
                c2_ci[(a, b, with_time, res["классификатор"])] = res.pop("_ci")
                rows.append({"сравнение": f"{a} vs {b}", "время": "да" if with_time else "нет", **res})
    # контроль: две случайные половины train
    for res in c2st(R[False][tr_perm[:C2ST_N]], R[False][tr_perm[C2ST_N:2 * C2ST_N]]):
        res.pop("_ci")
        rows.append({"сравнение": "train_A vs train_B", "время": "нет", **res})
    c2 = pd.DataFrame(rows)
    save_table(c2, "shift_c2st")

    # KS для цели
    y = ctx.y_log
    ks = []
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        r = sps.ks_2samp(y[split == a], y[split == b])
        ks.append({"сравнение": f"{a} vs {b}", "KS D": r.statistic, "p_value": r.pvalue,
                   "медиана log1p a": np.median(y[split == a]), "медиана log1p b": np.median(y[split == b])})
    save_table(pd.DataFrame(ks), "shift_ks_target")

    # динамика по годам
    yr = c["issue_date"].dt.year
    pred = ctx.preds["pred_log"].to_numpy().copy()
    if OOF_PATH.exists():
        pred[split == "train"] = pd.read_parquet(OOF_PATH)["oof_log"].to_numpy()
    dyn = pd.DataFrame({"год": yr, "v": c["valuation"], "y": y, "r2": (y - pred) ** 2,
                        "s501": c["valuation"] == 501, "alter": c["PERMIT_TYPE"] == "Bldg-Alter/Repair",
                        "comm": c["PERMIT_SUB_TYPE"] == "Commercial", "supp": c["work_desc"].str.contains("supplement", case=False),
                        "sqft_miss": c["square_footage"].isna()})
    g = dyn.groupby("год")
    dt = pd.DataFrame({"n": g.size(), "медиана $": g["v"].median(), "p25 $": g["v"].quantile(.25),
                       "p75 $": g["v"].quantile(.75), "доля $501": g["s501"].mean(), "доля Alter/Repair": g["alter"].mean(),
                       "доля Commercial": g["comm"].mean(), "доля Supplemental": g["supp"].mean(),
                       "пропуск площади": g["sqft_miss"].mean()}).reset_index()
    save_table(dt, "shift_by_year")

    # ошибка по годам отдельно для частей: train — OOF (случайные фолды внутри периода, интерполяция),
    # val/test — baseline (прогноз вперёд по времени); 2022 г. содержит train и val
    err = []
    for (year, part), ix in pd.DataFrame({"год": yr, "часть": split}).groupby(["год", "часть"]).indices.items():
        ci = st.bootstrap_ci(st.rmsle, y[ix], pred[ix], vectorized=True)
        err.append({"год": year, "часть": part, "прогноз": "OOF 5-fold" if part == "train" else "baseline",
                    "n": len(ix), **ci_cols(ci)})
    err = pd.DataFrame(err).sort_values(["год", "часть"]).reset_index(drop=True)
    save_table(err, "shift_error_by_year")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
    ax[0].fill_between(dt["год"], dt["p25 $"], dt["p75 $"], alpha=0.3, label="IQR")
    ax[0].plot(dt["год"], dt["медиана $"], "o-", label="медиана")
    ax[0].set(yscale="log", xlabel="год выдачи", ylabel="VALUATION, $", title="Стоимость по годам")
    ax[0].legend()
    for col in ("доля $501", "доля Alter/Repair", "доля Commercial", "доля Supplemental", "пропуск площади"):
        ax[1].plot(dt["год"], dt[col], "o-", ms=3, label=col)
    ax[1].set(xlabel="год выдачи", ylabel="доля", title="Состав по годам")
    ax[1].legend(fontsize=7)
    for part, col in (("train", "C0"), ("val", "C1"), ("test", "C3")):
        e = err[err["часть"] == part]
        x = e["год"] + {"train": -0.1, "val": 0.0, "test": 0.1}[part]
        ax[2].errorbar(x, e["value"], yerr=[e["value"] - e["ci_low"], e["ci_high"] - e["value"]], fmt="o-",
                       ms=3, capsize=2, color=col, label=f"{part} ({'OOF' if part == 'train' else 'baseline'})")
    ax[2].legend(fontsize=7)
    b1 = c.loc[split == "val", "issue_date"].min().year + c.loc[split == "val", "issue_date"].min().dayofyear / 366
    b2 = c.loc[split == "test", "issue_date"].min().year + c.loc[split == "test", "issue_date"].min().dayofyear / 366
    for bx, lab in ((b1, "начало val"), (b2, "начало test")):
        ax[2].axvline(bx, color="gray", ls="--")
        ax[2].text(bx, ax[2].get_ylim()[1], lab, fontsize=8, va="top")
    ax[2].set(xlabel="год выдачи", ylabel="RMSLE", title="Ошибка модели по годам")
    for a in ax:
        a.grid(alpha=0.3)
    save_fig(fig, "shift_by_year")

    fig, ax = plt.subplots(figsize=(9, 4))
    sub = mmd[~mmd["сравнение"].isin(["train vs val", "val vs test"])]
    labels = [f"{r['сравнение']}\n{r['представление']}" for _, r in sub.iterrows()]
    ax.bar(range(len(sub)), sub["mmd2_u"], yerr=[(sub["mmd2_u"] - sub["ci_low"]).clip(lower=0),
                                                (sub["ci_high"] - sub["mmd2_u"]).clip(lower=0)],
           color=["C3" if p < 0.05 else "C0" for p in sub["p_value"]])
    ax.bar(range(len(sub)), sub["perm_q95"], fill=False, edgecolor="k", ls="--", label="95% квантиль H0 (перестановки)")
    ax.set_xticks(range(len(sub)), labels, fontsize=7)
    ax.set(ylabel="MMD² (несмещённая)", title="MMD train vs test (красный — p < 0,05) и контроль train_A vs train_B")
    ax.legend()
    save_fig(fig, "shift_mmd")

    m0 = mmd[(mmd["сравнение"] == "train vs test") & (mmd["представление"] == "признаки без времени")].iloc[0]
    mc = mmd[(mmd["сравнение"] == "train vs test") & (mmd["представление"] == "эмбеддинги [CLS] baseline")].iloc[0]
    ctl = mmd[(mmd["сравнение"] == "train_A vs train_B") & (mmd["представление"] == "признаки без времени")].iloc[0]
    mvt = mmd[(mmd["сравнение"] == "val vs test") & (mmd["представление"] == "эмбеддинги [CLS] baseline")].iloc[0]
    ci_lr = c2_ci[("train", "test", False, "логистическая регрессия")]
    ci_mlp = c2_ci[("train", "test", False, "MLP (64)")]
    ks_t = ks[1]
    return [
        summary_row("Shift (репрезентативность)", "MMD²_u train vs test (признаки без времени, n = 5000 + 5000)",
                    st.CI(m0["mmd2_u"], m0["ci_low"], m0["ci_high"], "percentile", int(m0["m"] + m0["n"])), fmt="{:.4f}",
                    interp=f"permutation p = {m0['p_value']:.3g} (контроль train/train: MMD² {ctl['mmd2_u']:.4f}, "
                           f"p = {ctl['p_value']:.2f})"),
        summary_row("Shift (репрезентативность)", "MMD²_u train vs test (эмбеддинги [CLS])",
                    st.CI(mc["mmd2_u"], mc["ci_low"], mc["ci_high"], "percentile", int(mc["m"] + mc["n"])), fmt="{:.4f}",
                    interp=f"permutation p = {mc['p_value']:.3g}; train-объекты — обучающие для модели, поэтому сдвиг "
                           f"смешан с подгонкой под train; val vs test (обе не видела): MMD² {mvt['mmd2_u']:.4f}, "
                           f"p = {mvt['p_value']:.3g}"),
        summary_row("Shift (репрезентативность)", "C2ST ROC-AUC train vs test (логрег, без времени)", ci_lr,
                    fmt="{:.3f}", interp="0,5 — неразличимы; стратиф. bootstrap по классам"),
        summary_row("Shift (репрезентативность)", "C2ST ROC-AUC train vs test (MLP, без времени)", ci_mlp,
                    fmt="{:.3f}", interp="нелинейный классификатор"),
        summary_row("Shift (репрезентативность)", "KS D для log1p(VALUATION), train vs test", None,
                    value=ks_t["KS D"], fmt="{:.4f}", interp=f"p = {ks_t['p_value']:.2g}; при n ≈ 10⁵ значим даже малый сдвиг"),
    ]


def section_shift_features(ctx: Ctx) -> None:
    """Какие признаки отличают train от test: C2ST (логрег) по каждой группе признаков отдельно."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    _, X, prep = ctx.feats
    split = ctx.split
    rng = np.random.default_rng(SEED)
    ia = rng.choice(np.flatnonzero(split == "train"), C2ST_N, replace=False)
    ib = rng.choice(np.flatnonzero(split == "test"), C2ST_N, replace=False)
    idx = np.r_[ia, ib]
    lab = np.r_[np.zeros(C2ST_N, int), np.ones(C2ST_N, int)]
    groups = {}
    for j, name in enumerate(prep.num_names):
        groups[f"число: {name}"] = np.c_[X["num"][idx, j], X["num_mask"][idx, j]]
    groups["текст: SVD(64)"] = X["txt"][idx]
    for j, name in enumerate(prep.cat_names):
        groups[f"категория: {name}"] = OneHotEncoder(handle_unknown="ignore").fit_transform(X["cat"][idx, j:j + 1])
    rows = []
    for name, feat in groups.items():
        f_tr, f_te, l_tr, l_te = train_test_split(feat, lab, test_size=0.5, stratify=lab, random_state=SEED)
        if not hasattr(feat, "tocsr"):
            sc = StandardScaler().fit(f_tr)
            f_tr, f_te = sc.transform(f_tr), sc.transform(f_te)
        clf = LogisticRegression(max_iter=2000).fit(f_tr, l_tr)
        ci = st.stratified_bootstrap_ci(auc_vec, clf.predict_proba(f_te)[:, 1], l_te, strata=l_te, vectorized=True)
        rows.append({"группа признаков": name, **ci_cols(ci)})
    tab = pd.DataFrame(rows).sort_values("value", ascending=False)
    save_table(tab, "shift_feature_c2st")
    fig, ax = plt.subplots(figsize=(8, 7))
    t = tab.sort_values("value")
    ax.barh(t["группа признаков"], t["value"] - 0.5, left=0.5,
            xerr=[(t["value"] - t["ci_low"]).clip(lower=0), (t["ci_high"] - t["value"]).clip(lower=0)])
    ax.axvline(0.5, color="k", lw=0.8)
    ax.set(xlabel="ROC-AUC train vs test по одной группе признаков", title="Вклад признаков в сдвиг (C2ST по группам)")
    ax.grid(alpha=0.3, axis="x")
    save_fig(fig, "shift_feature_c2st")
    log.info("shift_features: топ-3 %s", tab.head(3)[["группа признаков", "value"]].values.tolist())


DRIFT_PATTERNS = {
    "«Eplan» (электронная подача)": r"\beplan",
    "номер другого разрешения в тексте": r"\d{5}-\d{5}-\d{5}",
    "«supplement»": r"supplement",
    "шаблон re-roof «Class A or B»": r"class a or b",
    "ADU / accessory dwelling": r"\badu\b|accessory dwelling",
}


def section_drift_facts(ctx: Ctx) -> None:
    """Объяснение сдвига: доли текстовых шаблонов и новых значений CT/CNC по частям (CI Уилсона)."""
    c, split = ctx.clean, ctx.split
    t = c["work_desc"].str.lower()
    rows = []
    for name, pat in DRIFT_PATTERNS.items():
        hit = t.str.contains(pat, regex=True).to_numpy()
        rows += [wilson_row(f"текст: {name}", hit, split == part, часть=part) for part in SPLITS]
    tr = split == "train"
    for col in ("CT", "CNC"):
        known = set(c.loc[tr, col].dropna())
        new_val = (~c[col].isin(known) & c[col].notna()).to_numpy()
        rows += [wilson_row(f"{col}: значение не встречается в train", new_val, split == part, часть=part)
                 for part in ("val", "test")]
    save_table(pd.DataFrame(rows), "shift_drift_facts")

    # слова-токены baseline по годам: словарь собран по train, новые тексты его «перерастают»
    _, X, prep = ctx.feats
    if "words" in X:
        w = X["words"]
        nz = w != 0
        dfw = pd.DataFrame({"год": c["issue_date"].dt.year.to_numpy(), "unk": (w == 1).sum(1), "tok": nz.sum(1),
                            "trunc": nz.sum(1) == prep.word_len})
        g = dfw.groupby("год")
        save_table(pd.DataFrame({"доля UNK среди слов": g["unk"].sum() / g["tok"].sum(),
                                 "средняя длина, слов": g["tok"].mean(),
                                 f"обрезано на {prep.word_len} словах": g["trunc"].mean(),
                                 "n": g.size()}).reset_index(), "shift_text_vocab_by_year")
    log.info("drift_facts: готово")


# ====================================================================== 9. аномалии
def section_anomalies(ctx: Ctx) -> list[dict]:
    from sklearn.ensemble import IsolationForest
    c, split = ctx.clean, ctx.split
    R = representation(ctx, False)
    tr = np.flatnonzero(split == "train")
    rng = np.random.default_rng(SEED)
    fit_idx = rng.choice(tr, min(200_000, len(tr)), replace=False)
    iso = IsolationForest(n_estimators=300, contamination="auto", random_state=SEED, n_jobs=-1).fit(R[fit_idx])
    score = iso.score_samples(R)
    thr = np.quantile(score[tr], 0.01)  # критерий: 1% самых изолированных объектов train
    anom_auto = iso.predict(R) == -1
    anom = score <= thr
    y = ctx.y_log
    zt = 0.6745 * (y - np.median(y[tr])) / np.median(np.abs(y[tr] - np.median(y[tr])))
    out_y = np.abs(zt) > Z_THRESHOLD

    rows = []
    for part in SPLITS:
        m = split == part
        rows.append(wilson_row("IsolationForest: score ≤ 1-й перцентиль train", anom, m, часть=part))
        rows.append(wilson_row("IsolationForest: contamination='auto'", anom_auto, m, часть=part))
        rows.append(wilson_row(f"выброс log-цели (|robust z| > {Z_THRESHOLD}, медиана/MAD train)", out_y, m, часть=part))
    tab = pd.DataFrame(rows)
    save_table(tab, "anomalies")

    # влияют ли аномалии на ошибку baseline (test)
    te = split == "test"
    p = ctx.preds["pred_log"].to_numpy()
    eff = []
    for name, mm in (("аномалии IsolationForest", anom), ("остальные", ~anom)):
        s = te & mm
        ci = st.bootstrap_ci(st.rmsle, y[s], p[s], vectorized=True)
        eff.append({"группа test": name, "rmsle": ci.estimate, "ci_low": ci.low, "ci_high": ci.high,
                    "ci_method": ci.method, "n": int(s.sum())})
    save_table(pd.DataFrame(eff), "anomalies_error")

    idx = np.flatnonzero(tr)[np.argsort(score[tr])[:15]]
    save_table(pd.DataFrame({
        "PERMIT_NBR": c["PERMIT_NBR"].to_numpy()[idx], "PERMIT_TYPE": c["PERMIT_TYPE"].to_numpy()[idx],
        "USE_DESC": c["USE_DESC"].to_numpy()[idx], "SQUARE_FOOTAGE": c["square_footage"].to_numpy()[idx],
        "HEIGHT": c["height"].to_numpy()[idx], "DU_CHANGED": c["du_changed"].to_numpy()[idx],
        "VALUATION": c["valuation"].to_numpy()[idx], "score": score[idx],
        "WORK_DESC": c["work_desc"].str.slice(0, 160).to_numpy()[idx]}), "anomalies_top15")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for part in SPLITS:
        ax[0].hist(score[split == part], bins=100, density=True, histtype="step", label=part)
    ax[0].axvline(thr, color="r", ls="--", label="1-й перцентиль train")
    ax[0].set(xlabel="score IsolationForest (меньше — аномальнее)", ylabel="плотность", title="Аномальность объектов")
    ax[0].legend()
    e = pd.DataFrame(eff)
    ax[1].bar(e["группа test"], e["rmsle"], yerr=[e["rmsle"] - e["ci_low"], e["ci_high"] - e["rmsle"]])
    ax[1].set(ylabel="RMSLE baseline (test)", title="Ошибка на аномалиях и остальных")
    save_fig(fig, "anomalies")
    t = tab[tab["часть"] == "test"].set_index("check")
    to_ci = lambda row: st.CI(row["value"], row["ci_low"], row["ci_high"], "wilson", int(row["n"]))  # noqa: E731
    log.info("anomalies: test %s", to_ci(t.iloc[0]))
    return [
        summary_row("Аномалии", "доля аномалий в test (IsolationForest, порог — 1% train)", to_ci(t.iloc[0]),
                    scale=100, fmt="{:.2f}", interp=f"в %; ожидалось 1%, если test похож на train; RMSLE на них "
                                                    f"{eff[0]['rmsle']:.2f} против {eff[1]['rmsle']:.2f}"),
        summary_row("Аномалии", "доля выбросов log-цели в test (|robust z| > 3,5)", to_ci(t.iloc[2]), scale=100,
                    fmt="{:.2f}", interp="в %; очень дешёвые/дорогие объекты (хвосты стоимости)"),
    ]


# ====================================================================== итог
SECTIONS = {
    "describe": section_describe, "completeness": section_completeness, "validity": section_validity,
    "uniqueness": section_uniqueness, "leakage": section_leakage, "balance": section_balance,
    "labels": section_labels, "shift": section_shift, "shift_features": section_shift_features,
    "drift_facts": section_drift_facts,
    "anomalies": section_anomalies,
}
SUMMARY_ORDER = ["completeness", "validity", "uniqueness", "leakage", "balance", "labels", "shift", "anomalies"]


def section_summary(_: Ctx | None = None) -> None:
    rows = []
    for s in SUMMARY_ORDER:
        p = OUT / f"_summary_{s}.json"
        if p.exists():
            rows += json.loads(p.read_text(encoding="utf-8"))
    tab = pd.DataFrame(rows)
    save_table(tab, "summary")
    md = ["| " + " | ".join(tab.columns) + " |", "|" + "---|" * len(tab.columns)]
    md += ["| " + " | ".join(str(v) for v in r) + " |" for r in tab.itertuples(index=False)]
    (OUT / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log.info("summary: %d строк", len(tab))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=list(SECTIONS) + ["summary"], help="выполнить только эти разделы")
    ap.add_argument("--force", action="store_true", help="пересчитать кеши (OOF, MinHash, эмбеддинги)")
    args = ap.parse_args()
    get_logger(ARTIFACTS_DIR / "logs" / "pr1.log")
    set_seed(SEED)
    ctx = Ctx(force=args.force)
    todo = args.only or list(SECTIONS) + ["summary"]
    for name in todo:
        if name == "summary":
            continue
        t0 = time.perf_counter()
        log.info("=== раздел %s", name)
        rows = SECTIONS[name](ctx)
        if rows:
            write_summary(name, rows)
        log.info("=== %s: %.0f с", name, time.perf_counter() - t0)
    section_summary()


if __name__ == "__main__":
    main()
