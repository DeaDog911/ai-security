"""Эксперимент label_cleaning: даёт ли удаление служебных оценок из обучения прирост сверх фильтрации оценки.

План — docs/context/PLAN_LABEL_CLEANING.md (утверждён 2026-11-02). Модель — быстрый вариант A
(SVD(64) → 1 токен) как прокси C96; протокол — как в text_tokens_holdout: val делится по дате,
val_sel — ранняя остановка и выбор эпохи, val_hold — только оценка. Test не используется.

Правила очистки (только по метке и тексту, без модели):
    R2 — служебные значения VALUATION: {$501, $301} ∪ {≤ $10};
    R3 — административные разрешения по тексту — отброшен на шаге 0 (ни одна фраза не набрала
         точность ≥ 70% при покрытии ≥ 0,5% train; разметка — reports/experiments/label_cleaning/).

Варианты (set_seed перед build_model):
    A_full    — полный train и val_sel (эталон; совпадает с text_tokens_holdout A), seed 42, 43, 44;
    A_noR2    — train и val_sel без R2, seed 42, 43, 44;
    C96_full  — эталон C96 (берётся из text_tokens_holdout C96), seed 42, 43;
    C96_noR2  — подтверждающий запуск C96 без R2, seed 42, 43 (решение пользователя 2026-11-02).

Запуск из корня проекта:
    venv\\Scripts\\python src\\experiments\\label_cleaning.py --step0          # шаг 0: выборки и точность правил
    venv\\Scripts\\python src\\experiments\\label_cleaning.py --smoke --variants C96_noR2   # 5000 строк, 2 эпохи
    venv\\Scripts\\python src\\experiments\\label_cleaning.py --variants A_full,A_noR2 --seeds 42,43,44
    venv\\Scripts\\python src\\experiments\\label_cleaning.py --variants C96_full,C96_noR2 --seeds 42,43
    venv\\Scripts\\python src\\experiments\\label_cleaning.py --report
Готовые запуски (artifacts/experiments/label_cleaning/<вариант>_s<seed>/) без --force не пересчитываются.
Отчёт: reports/experiments/label_cleaning/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import stats as st  # noqa: E402
from config import ARTIFACTS_DIR, BOOTSTRAP_B, REPORTS_DIR, SEED  # noqa: E402
from model import DEFAULT_HP  # noqa: E402
from text_tokens import DECISION_DELTA, _md  # noqa: E402
from text_tokens_holdout import HOLD_DIR, inputs, split_val  # noqa: E402
from train_baseline import get_logger, load_features  # noqa: E402
from training import TRAIN_HP, build_model, device_name, fit, get_device, predict, set_seed  # noqa: E402

EXP_DIR = ARTIFACTS_DIR / "experiments" / "label_cleaning"
REP_DIR = REPORTS_DIR / "experiments" / "label_cleaning"
LOG_PATH = ARTIFACTS_DIR / "logs" / "label_cleaning.log"

SERVICE_VALUES = (501.0, 301.0)  # служебные оценки (ПР1: $501 — 7,7% меток)
SERVICE_MAX = 10.0               # оценки ≤ $10 ($1 и т. п.) — тоже служебные

# Исходная регулярка R3 из плана: слова для разметки на шаге 0 (сработавшее слово → выборка)
R3_WORDS = {"supplement": r"supplement", "fee": r"\bfees?\b", "capture": r"capture",
            "correct permit": r"correct permit", "extend permit": r"extend permit",
            "no actual work": r"no actual work"}
R3_SAMPLE = {"supplement": 30, "fee": 25, "capture": 24, "correct permit": 10, "extend permit": 10,
             "no actual work": 1}
R3_MIN_PRECISION = 0.70
R3_MIN_COVERAGE = 0.005  # доля train; меньше — R3 отбрасывается

# вариант → (описание, правило очистки, вариант модели из text_tokens_holdout)
VARIANTS = {
    "A_full": ("A: полный train и val_sel (эталон)", None, "A"),
    "A_noR2": ("A: train и val_sel без R2", "R2", "A"),
    "C96_full": ("C96: полный train и val_sel (эталон)", None, "C96"),
    "C96_noR2": ("C96: train и val_sel без R2", "R2", "C96"),
}
REFERENCE_OF = {"A_noR2": "A_full", "C96_noR2": "C96_full"}
# собственный очищенный срез варианта (решение пользователя 2026-11-02) и минимум seed для правила решения
CRITERION_SLICE = {"A_noR2": "noR2", "C96_noR2": "noR2"}
MIN_SEEDS = {"A_noR2": 3, "C96_noR2": 2}
REFERENCE = "A_full"  # эталон, воспроизводимость которого проверяется бит-в-бит
SMOKE_ROWS = 5000


def service_mask(valuation: np.ndarray) -> np.ndarray:
    """R2: служебные значения VALUATION. Общая функция для эксперимента и (при принятии) для data.py."""
    v = np.asarray(valuation, dtype=float)
    return np.isin(np.round(v, 6), SERVICE_VALUES) | (v <= SERVICE_MAX + 1e-9)


def rule_mask(rule: str | None, meta) -> np.ndarray:
    """Строки, которые исключаются правилом (False для всех при rule=None)."""
    if rule is None:
        return np.zeros(len(meta), dtype=bool)
    if rule == "R2":
        return service_mask(np.expm1(meta["y_log"].to_numpy()))
    raise ValueError(f"неизвестное правило {rule}")


# ---------------------------------------------------------------- шаг 0: проверка правил
def step0(log) -> None:
    """Таблица малых значений, выборки для ручной разметки (seed 42) и точность по словам R3.

    Разметка (sample, row_id, label ∈ {admin, normal}, comment) — reports/experiments/label_cleaning/step0_labels.csv;
    admin — «работ нет / стоимость не своих работ», normal — нормальная оценка.
    """
    from data import load_dataset
    from features import normalize_text

    REP_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    df["text"] = normalize_text(df["work_desc"])
    df["r2"] = service_mask(df["valuation"].to_numpy())
    tr = df[df["split"] == "train"]
    cols = ["row_id", "PERMIT_TYPE", "valuation", "square_footage", "text"]

    small = tr.loc[tr["valuation"] <= 600, "valuation"].value_counts().rename_axis("valuation").reset_index(
        name="n_train")
    small["share_of_le600"] = small["n_train"] / small["n_train"].sum()
    small["in_r2"] = service_mask(small["valuation"].to_numpy())
    small.to_csv(REP_DIR / "r2_small_values.csv", index=False)

    labels = pd.read_csv(REP_DIR / "step0_labels.csv")
    other = tr[(tr["valuation"] <= 600) & ~tr["r2"]].sample(30, random_state=SEED)
    other = other[cols].merge(labels[labels["sample"] == "r2_other"].drop(columns="sample"), on="row_id", how="left")
    other.to_csv(REP_DIR / "r2_other_review.csv", index=False)

    taken, parts = set(), []
    for w, pat in R3_WORDS.items():
        m = tr["text"].str.contains(pat, regex=True) & ~tr["r2"] & ~tr["row_id"].isin(taken)
        s = tr[m].sample(min(R3_SAMPLE[w], int(m.sum())), random_state=SEED).assign(word=w)
        taken |= set(s["row_id"])
        parts.append(s)
    r3 = pd.concat(parts)[["word"] + cols].merge(labels[labels["sample"] == "r3"].drop(columns="sample"),
                                                  on="row_id", how="left")
    if r3["label"].isna().any() or other["label"].isna().any():
        raise ValueError("в step0_labels.csv размечены не все строки выборок")
    r3.to_csv(REP_DIR / "r3_review.csv", index=False)

    rows = []
    for w, pat in R3_WORDS.items():
        d = r3[r3["word"] == w]
        k, n = int((d["label"] == "admin").sum()), len(d)
        ci = st.wilson_ci(k, n)
        cov = tr["text"].str.contains(pat, regex=True) & ~tr["r2"]
        rows.append({"word": w, "n_reviewed": n, "n_admin": k, "precision": ci.estimate, "precision_low": ci.low,
                     "precision_high": ci.high, "n_train_outside_r2": int(cov.sum()),
                     "share_train": float(cov.mean()), "median_valuation": float(tr.loc[cov, "valuation"].median()),
                     "kept": bool(ci.estimate >= R3_MIN_PRECISION)})
    prec = pd.DataFrame(rows)
    kept = prec[prec["kept"]]
    keep_r3 = bool(kept["share_train"].sum() >= R3_MIN_COVERAGE)  # верхняя оценка покрытия (пересечения не вычтены)
    prec.to_csv(REP_DIR / "r3_precision.csv", index=False)

    k_other = int((other["label"] == "admin").sum())
    ci_other = st.wilson_ci(k_other, len(other))
    summary = {
        "r2_rule": f"VALUATION ∈ {{{', '.join(f'${v:.0f}' for v in SERVICE_VALUES)}}} ∪ {{≤ ${SERVICE_MAX:.0f}}}",
        "r2_share": {p: float(df.loc[df["split"] == p, "r2"].mean()) for p in ("train", "val")},
        "r2_n": {p: int(df.loc[df["split"] == p, "r2"].sum()) for p in ("train", "val")},
        "le600_n_train": int((tr["valuation"] <= 600).sum()),
        "r2_other_reviewed": len(other), "r2_other_admin": k_other,
        "r2_other_admin_share": ci_other.estimate, "r2_other_admin_low": ci_other.low,
        "r2_other_admin_high": ci_other.high,
        "r3_kept_words": kept["word"].tolist(), "r3_kept_share_train": float(kept["share_train"].sum()),
        "r3_min_precision": R3_MIN_PRECISION, "r3_min_coverage": R3_MIN_COVERAGE, "r3_kept": keep_r3,
    }
    (REP_DIR / "step0_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info("шаг 0: R2 train %.2f%%, val %.2f%%; прочее ≤ $600: admin %d из %d", 100 * summary["r2_share"]["train"],
             100 * summary["r2_share"]["val"], k_other, len(other))
    for _, r in prec.iterrows():
        log.info("  R3 «%s»: точность %.2f [%.2f; %.2f] (n = %d), покрытие train %.2f%%", r["word"], r["precision"],
                 r["precision_low"], r["precision_high"], r["n_reviewed"], 100 * r["share_train"])
    log.info("R3 %s", "сохраняется" if keep_r3 else "отбрасывается")


# ---------------------------------------------------------------- обучение
def run_dir(v: str, seed: int, smoke: bool) -> Path:
    return (EXP_DIR / "smoke" if smoke else EXP_DIR) / f"{v}_s{seed}"


def run(v, seed, meta, X, prep, device, log, smoke: bool, force: bool) -> None:
    out = run_dir(v, seed, smoke)
    if (out / "metrics.json").exists() and not force:
        log.info("%s seed %d: уже готов, пропускаю", v, seed)
        return
    Xv, model_kw = inputs(VARIANTS[v][2], meta, X, log)  # те же входы, что у эталона в text_tokens_holdout
    y_log = meta["y_log"].to_numpy()
    drop = rule_mask(VARIANTS[v][1], meta)
    tr = np.flatnonzero((meta["split"].to_numpy() == "train") & ~drop)
    sel, hold, _ = split_val(meta)
    sel = sel[~drop[sel]]
    thp = dict(TRAIN_HP)
    if smoke:
        rng = np.random.default_rng(SEED)
        tr = np.sort(rng.choice(tr, min(SMOKE_ROWS, len(tr)), replace=False))
        sel = np.sort(rng.choice(sel, min(SMOKE_ROWS // 5, len(sel)), replace=False))
        thp.update(max_epochs=2, warmup_steps=20)
    sub = lambda ix: {k: a[ix] for k, a in Xv.items()}  # noqa: E731
    X_hold = sub(hold)
    hold_preds = []

    def on_epoch(epoch, model):
        hold_preds.append((predict(model, X_hold, device) * prep.y_std + prep.y_mean).astype(np.float32))

    set_seed(seed)
    model = build_model(Xv, prep.cat_cardinalities, DEFAULT_HP, **model_kw)
    log.info("=== %s (%s), seed %d: train %d, val_sel %d, val_hold %d", v, VARIANTS[v][0], seed, len(tr), len(sel),
             len(hold))
    t0 = time.perf_counter()
    # стандартизация цели — по полному train (prep.y_mean, prep.y_std), как в baseline
    model, history, best_epoch = fit(model, sub(tr), (y_log[tr] - prep.y_mean) / prep.y_std, sub(sel), y_log[sel],
                                     prep.y_mean, prep.y_std, device, thp, log, seed=seed, epoch_callback=on_epoch)
    train_sec = time.perf_counter() - t0
    P = np.stack(hold_preds)
    y_hold = y_log[hold]
    p = P[best_epoch - 1].astype(np.float64)

    out.mkdir(parents=True, exist_ok=True)
    hist = pd.DataFrame(history).rename(columns={"val_rmsle": "sel_rmsle"})
    hist["hold_rmsle"] = np.sqrt(((P - y_hold) ** 2).mean(1))
    hist.to_csv(out / "history.csv", index=False)
    pd.DataFrame({"row_id": meta["row_id"].to_numpy()[hold], "y_log": y_hold, "pred_log": p}).to_parquet(
        out / "hold_predictions.parquet", index=False)
    m = {"variant": v, "seed": seed, "smoke": smoke, "n_train": int(len(tr)), "n_sel": int(len(sel)),
         "n_hold": int(len(hold)), "train_y_mean": float(y_log[tr].mean()), "train_y_std": float(y_log[tr].std()),
         "best_epoch": int(best_epoch), "epochs_run": len(history),
         "sel_rmsle": float(hist["sel_rmsle"].iloc[best_epoch - 1]),
         "sec_per_epoch": float(hist["sec"].mean()), "train_seconds": round(train_sec, 1),
         "device": device_name(device)}
    (out / "metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    torch.save(model.state_dict(), out / "model.pt")
    log.info("%s seed %d: эпоха %d из %d (val_sel %.5f), val_hold RMSLE %.5f, %.1f с/эпоху", v, seed, best_epoch,
             len(history), m["sel_rmsle"], st.rmsle(y_hold, p), m["sec_per_epoch"])


def check_reference(log) -> dict:
    """A_full seed 42 против text_tokens_holdout A_s42: прогнозы на val_hold должны совпасть бит-в-бит."""
    a, b = EXP_DIR / f"{REFERENCE}_s{SEED}" / "hold_predictions.parquet", HOLD_DIR / f"A_s{SEED}" / "hold_predictions.parquet"
    if not (a.exists() and b.exists()):
        return {}
    pa, pb = pd.read_parquet(a), pd.read_parquet(b)
    same_rows = bool(np.array_equal(pa["row_id"].to_numpy(), pb["row_id"].to_numpy()))
    diff = float(np.abs(pa["pred_log"].to_numpy() - pb["pred_log"].to_numpy()).max()) if same_rows else float("nan")
    res = {"same_rows": same_rows, "max_abs_diff": diff, "bitwise": same_rows and diff == 0.0}
    log.info("A_full seed %d против text_tokens_holdout A: %s", SEED, res)
    return res


def adopt_reference(v: str, seed: int, meta, log) -> bool:
    """Эталон (*_full) берётся из text_tokens_holdout (тот же код, входы и данные), если A_full seed 42 совпал
    с ним бит-в-бит. Для C96 повторный запуск seed 42 (≈ 30 мин) не делается: совпадение A подтверждает,
    что путь обучения не изменился."""
    out, src = run_dir(v, seed, False), HOLD_DIR / f"{VARIANTS[v][2]}_s{seed}"
    if (out / "metrics.json").exists():
        return True
    if not (src / "metrics.json").exists() or not check_reference(log).get("bitwise"):
        return False
    m0 = json.loads((src / "metrics.json").read_text(encoding="utf-8"))
    y_log = meta["y_log"].to_numpy()
    tr = meta["split"].to_numpy() == "train"
    sel, _, _ = split_val(meta)
    out.mkdir(parents=True, exist_ok=True)
    for f in ("history.csv", "hold_predictions.parquet"):
        (out / f).write_bytes((src / f).read_bytes())
    m = {"variant": v, "seed": seed, "smoke": False, "n_train": int(tr.sum()), "n_sel": int(len(sel)),
         "n_hold": m0["n_hold"], "train_y_mean": float(y_log[tr].mean()), "train_y_std": float(y_log[tr].std()),
         **{k: m0[k] for k in ("best_epoch", "epochs_run", "sel_rmsle", "sec_per_epoch", "train_seconds", "device")},
         "adopted_from": str(src.relative_to(ARTIFACTS_DIR.parent))}
    (out / "metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info("%s seed %d: взят из %s (A_full seed 42 совпал бит-в-бит)", v, seed, src)
    return True


# ---------------------------------------------------------------- отчёт
def _ci_cols(ci: st.CI, name: str) -> dict:
    return {name: ci.estimate, f"{name}_low": ci.low, f"{name}_high": ci.high, f"{name}_method": ci.method}


def _slices(meta, hold_row_id: np.ndarray) -> dict[str, np.ndarray]:
    r2 = rule_mask("R2", meta.set_index("row_id").loc[hold_row_id].reset_index())
    return {"full": np.ones(len(hold_row_id), dtype=bool), "noR2": ~r2, "onlyR2": r2}


def report(meta, log, smoke: bool = False) -> None:
    base = EXP_DIR / "smoke" if smoke else EXP_DIR
    rep = REP_DIR / "smoke" if smoke else REP_DIR
    runs, preds = [], {}
    for f in sorted(base.glob("*_s*/metrics.json")):
        m = json.loads(f.read_text(encoding="utf-8"))
        runs.append(m)
        preds[(m["variant"], m["seed"])] = pd.read_parquet(f.parent / "hold_predictions.parquet")
    if not runs:
        log.info("нет готовых запусков")
        return
    rep.mkdir(parents=True, exist_ok=True)
    ref = check_reference(log) if not smoke else {}

    rows = []
    for m in runs:
        d = preds[(m["variant"], m["seed"])]
        y, p = d["y_log"].to_numpy(), d["pred_log"].to_numpy()
        for sl, mask in _slices(meta, d["row_id"].to_numpy()).items():
            rows.append({**{k: m[k] for k in ("variant", "seed", "n_train", "n_sel", "train_y_mean", "train_y_std",
                                             "best_epoch", "epochs_run", "sel_rmsle", "sec_per_epoch")},
                         "slice": sl, "n": int(mask.sum()), "rmsle": float(st.rmsle(y[mask], p[mask])),
                         "mdape_pct": float(st.mdape(y[mask], p[mask])),
                         "within25": float(st.within25(y[mask], p[mask]))})
    res = pd.DataFrame(rows).sort_values(["variant", "seed", "slice"]).reset_index(drop=True)
    res.to_csv(rep / "results.csv", index=False)

    paired = []
    for (v, s), d in sorted(preds.items()):
        r = REFERENCE_OF.get(v)
        if r is None or (r, s) not in preds:
            continue
        a = preds[(r, s)]
        assert np.array_equal(a["row_id"].to_numpy(), d["row_id"].to_numpy())
        for sl, mask in _slices(meta, d["row_id"].to_numpy()).items():
            y, pv, pa = d["y_log"].to_numpy()[mask], d["pred_log"].to_numpy()[mask], a["pred_log"].to_numpy()[mask]
            crit = CRITERION_SLICE.get(v) == sl
            kw = dict(vectorized=True, B=BOOTSTRAP_B, seed=SEED, method="BCa" if crit else "percentile")
            row = {"variant": v, "reference": r, "seed": s, "slice": sl, "criterion": crit, "n": len(y)}
            row |= _ci_cols(st.paired_bootstrap_ci(st.rmsle, y, pv, pa, **kw), "d_rmsle")
            kw["method"] = "percentile"
            row |= _ci_cols(st.paired_bootstrap_ci(st.mdape, y, pv, pa, **kw), "d_mdape_pp")
            row |= _ci_cols(st.paired_bootstrap_ci(st.within25, y, pv, pa, **kw), "d_within25")
            row["ci_excludes_0"] = bool(row["d_rmsle_high"] < 0 or row["d_rmsle_low"] > 0)
            paired.append(row)
            log.info("%s vs %s, seed %d, срез %s: ΔRMSLE %+.4f [%+.4f; %+.4f] (%s)", v, r, s, sl,
                     row["d_rmsle"], row["d_rmsle_low"], row["d_rmsle_high"], row["d_rmsle_method"])
    paired = pd.DataFrame(paired)
    paired.to_csv(rep / "paired_vs_full.csv", index=False)

    decision = pd.DataFrame()
    if not paired.empty:
        c = paired[paired["criterion"]]
        decision = c.groupby(["variant", "slice"]).agg(
            n_seeds=("seed", "nunique"), d_rmsle_mean=("d_rmsle", "mean"), d_rmsle_min=("d_rmsle", "min"),
            d_rmsle_max=("d_rmsle", "max"), d_rmsle_high_max=("d_rmsle_high", "max"),
            d_rmsle_low_min=("d_rmsle_low", "min"), d_mdape_mean=("d_mdape_pp", "mean"),
            d_within25_mean=("d_within25", "mean")).reset_index()
        decision["min_seeds"] = decision["variant"].map(MIN_SEEDS)
        decision["passes_rule"] = (decision["d_rmsle_mean"] <= DECISION_DELTA) & \
                                  (decision["d_rmsle_high_max"] < 0) & (decision["n_seeds"] >= decision["min_seeds"])
        decision.to_csv(rep / "decision.csv", index=False)

    # разбивка по типу разрешения при seed 42 на срезе критерия
    by_type = []
    for v, sl in CRITERION_SLICE.items():
        r = REFERENCE_OF[v]
        if (v, SEED) not in preds or (r, SEED) not in preds:
            continue
        d, a = preds[(v, SEED)], preds[(r, SEED)]
        mask = _slices(meta, d["row_id"].to_numpy())[sl]
        g = meta.set_index("row_id").loc[d["row_id"].to_numpy(), "PERMIT_TYPE"].to_numpy()
        for t in np.unique(g):
            ix = mask & (g == t)
            y, pv, pa = d["y_log"].to_numpy()[ix], d["pred_log"].to_numpy()[ix], a["pred_log"].to_numpy()[ix]
            row = {"variant": v, "reference": r, "slice": sl, "PERMIT_TYPE": t, "n": int(ix.sum()),
                   "rmsle_reference": float(st.rmsle(y, pa)), "rmsle_variant": float(st.rmsle(y, pv))}
            row |= _ci_cols(st.paired_bootstrap_ci(st.rmsle, y, pv, pa, vectorized=True, B=BOOTSTRAP_B, seed=SEED,
                                                   method="percentile"), "d_rmsle")
            by_type.append(row)
    by_type = pd.DataFrame(by_type)
    by_type.to_csv(rep / "by_type.csv", index=False)
    if ref:
        pd.DataFrame([ref]).to_csv(rep / "reference_check.csv", index=False)

    write_summary(rep, res, paired, decision, by_type, ref, meta)
    log.info("отчёт: %s", rep)


def write_summary(rep, res, paired, decision, by_type, ref, meta) -> None:
    _, _, info = split_val(meta)
    txt = ["# Эксперимент label_cleaning: очистка служебных оценок (варианты A и C96, val_hold)", "",
           "План — `docs/context/PLAN_LABEL_CLEANING.md`. Правило R2: VALUATION ∈ {$501, $301} ∪ {≤ $10}; "
           "R3 отброшен на шаге 0 (`step0_summary.json`, `r3_precision.csv`).",
           f"val_sel {info['sel_dates'][0]} … {info['sel_dates'][1]} — ранняя остановка и выбор эпохи "
           f"(у *_noR2 без R2); val_hold {info['hold_dates'][0]} … {info['hold_dates'][1]} — только оценка. "
           "Test не используется.",
           "Срезы val_hold: full — все строки, noR2 — без служебных (критерий для *_noR2), onlyR2 — только служебные.",
           "Парный bootstrap против эталона того же семейства (A_noR2 → A_full, C96_noR2 → C96_full) с тем же seed, "
           "B = 2000: ΔRMSLE на срезе критерия — BCa, остальное — percentile (метод — в столбцах `*_method`).",
           "Эталоны A_full (seed 43, 44) и C96_full взяты из text_tokens_holdout (`adopted_from` в metrics.json).",
           ""]
    if ref:
        txt += [f"Проверка эталона: A_full seed 42 против text_tokens_holdout A — совпадение бит-в-бит: "
                f"{ref['bitwise']} (max |Δ| = {ref['max_abs_diff']:.3g}).", ""]
    txt += ["## Запуски и метрики по срезам", "",
            _md(res[["variant", "seed", "slice", "n", "rmsle", "mdape_pct", "within25", "best_epoch", "epochs_run",
                     "sel_rmsle", "n_train", "train_y_mean", "train_y_std"]])]
    if not paired.empty:
        txt += ["", "## Парные сравнения с эталоном", "",
                _md(paired[["variant", "reference", "seed", "slice", "criterion", "n", "d_rmsle", "d_rmsle_low", "d_rmsle_high",
                            "d_rmsle_method", "d_mdape_pp", "d_within25", "ci_excludes_0"]])]
    if not decision.empty:
        txt += ["", "## Правило решения", "",
                f"Средний ΔRMSLE по seed ≤ {DECISION_DELTA} и CI не содержит 0 при каждом seed — "
                "на собственном очищенном срезе варианта; минимум seed: A — 3, C96 — 2 (`min_seeds`).", "",
                _md(decision), ""]
        for _, r in decision.iterrows():
            txt.append(f"- {r['variant']} (срез {r['slice']}): средний ΔRMSLE {r['d_rmsle_mean']:+.4f} "
                       f"(min–max {r['d_rmsle_min']:+.4f}…{r['d_rmsle_max']:+.4f}) — "
                       f"{'**проходит**' if r['passes_rule'] else 'не проходит'}.")
    if not by_type.empty:
        txt += ["", "## По типу разрешения (seed 42, срез критерия)", "", _md(by_type)]
    notes = rep / "notes.md"
    if notes.exists():
        txt += ["", notes.read_text(encoding="utf-8").strip()]
    (rep / "summary.md").write_text("\n".join(txt) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step0", action="store_true", help="только шаг 0 (проверка правил)")
    ap.add_argument("--variants", default="A_full,A_noR2")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--smoke", action="store_true", help=f"{SMOKE_ROWS} строк train, 2 эпохи, seed 42")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", action="store_true", help="только отчёт")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    log = get_logger(LOG_PATH)
    if args.step0:
        step0(log)
        return
    device = get_device(args.device)
    log.info("устройство: %s", device_name(device))
    meta, X, prep = load_features(log=log)
    log.info("разбиение val: %s", split_val(meta)[2])
    variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    seeds = [SEED] if args.smoke else [int(x) for x in args.seeds.split(",") if x.strip()]
    if not args.report:
        for v in variants:
            if v not in VARIANTS:
                raise ValueError(f"неизвестный вариант {v}; есть {list(VARIANTS)}")
            # seed 42 первым: от совпадения A_full seed 42 с text_tokens_holdout зависит, брать ли оттуда эталоны
            for s in sorted(seeds, key=lambda x: x != SEED):
                adoptable = v.endswith("_full") and not (v == REFERENCE and s == SEED)
                if adoptable and not args.smoke and not args.force and adopt_reference(v, s, meta, log):
                    continue
                run(v, s, meta, X, prep, device, log, args.smoke, args.force)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    report(meta, log, smoke=args.smoke)


if __name__ == "__main__":
    main()
