"""Эксперимент time_clip: время без обрезки и доменные правила вместо квантильной обрезки числовых признаков.

План — docs/context/PLAN_TIME_CLIP.md (утверждён 2026-11-02). Модель — C96; меняются только числовые признаки
(категории, SVD-токен и слова — из features.npz и кеша text_tokens без пересчёта). Протокол — как в
text_tokens_holdout: val делится по дате, val_sel — ранняя остановка и выбор эпохи, val_hold — только оценка.
Test не используется.

Варианты (set_seed перед build_model):
    C96    — числа как в C96: year_clip + quantile (эталон; берётся из готовых запусков, если они есть);
    A      — time_mode="continuous": t = (год − 2010) + (день года − 1)/365,25 без обрезки, месяц и день года —
             sin/cos; остальные числа — с квантильной обрезкой, как в C96;
    Aplus  — A + clip_mode="domain": доменные правила (ошибка → пропуск) вместо обрезки, предохранитель ±50σ.

Обучающая выборка (--filter) — как у принятого baseline (решение по R2 — за пользователем):
    full   — полный train и val_sel; эталон C96 — text_tokens_holdout/C96_s<seed>;
    noR2   — train и val_sel без служебных оценок R2; эталон — label_cleaning/C96_noR2_s<seed>
             (seed 44 нет — обучается здесь как вариант C96).
Решающий срез val_hold совпадает с фильтром (full или noR2); метрики считаются на обоих.

Запуск из корня проекта:
    venv\\Scripts\\python src\\experiments\\time_clip.py --step1           # шаг 1: правила и |z| без обрезки
    venv\\Scripts\\python src\\experiments\\time_clip.py --check           # поведение по умолчанию = C96 бит-в-бит
    venv\\Scripts\\python src\\experiments\\time_clip.py --smoke --filter full --variants C96,A,Aplus
    venv\\Scripts\\python src\\experiments\\time_clip.py --filter noR2 --variants A,Aplus --seeds 42
    venv\\Scripts\\python src\\experiments\\time_clip.py --report
Готовые запуски (artifacts/experiments/time_clip/<фильтр>/<вариант>_s<seed>/) без --force не пересчитываются.
Отчёт: reports/experiments/time_clip/.
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
from config import ARTIFACTS_DIR, BASELINE_CLIP_MODE, BASELINE_TIME_MODE, BOOTSTRAP_B, REPORTS_DIR, SEED  # noqa: E402
from data import load_dataset, service_mask  # noqa: E402
from features import FUSE_SIGMA, Preprocessor, _transform_num, domain_rule_masks, numeric_frame  # noqa: E402
from label_cleaning import EXP_DIR as LC_DIR  # noqa: E402
from model import DEFAULT_HP  # noqa: E402
from text_tokens import DECISION_DELTA, _md  # noqa: E402
from text_tokens_holdout import HOLD_DIR, inputs, split_val  # noqa: E402
from train_baseline import MODEL_PATH, PRED_PATH, get_logger, load_features  # noqa: E402
from training import TRAIN_HP, build_model, device_name, fit, get_device, predict, set_seed  # noqa: E402

EXP_DIR = ARTIFACTS_DIR / "experiments" / "time_clip"
REP_DIR = REPORTS_DIR / "experiments" / "time_clip"
LOG_PATH = ARTIFACTS_DIR / "logs" / "time_clip.log"

# вариант → (описание, time_mode, clip_mode)
VARIANTS = {
    "C96": ("числа как в C96 (эталон)", "year_clip", "quantile"),
    "A": ("время без обрезки (t, sin/cos)", "continuous", "quantile"),
    "Aplus": ("время + доменные правила вместо обрезки", "continuous", "domain"),
}
COMPARISONS = [("A", "C96"), ("Aplus", "C96"), ("Aplus", "A")]
FILTERS = ("full", "noR2")
DECISION_SLICE = {"full": "full", "noR2": "noR2"}
PROMISING_DELTA = -0.01  # промежуточное решение после этапа 1: на этап 2 — ΔRMSLE ≤ −0,01
MIN_SEEDS = 3
SMOKE_ROWS = 5000
HOLD_RMSLE_SANE = 10.0  # выше — сбой (у константного прогноза ≈ 1,7)
TAIL_LOG = float(np.log1p(1e6))  # хвост VALUATION ≥ $1M
PARTS = ("train", "val_sel", "val_hold", "test")


def external_reference(filt: str, seed: int, smoke: bool) -> Path | None:
    """Готовый запуск C96 с тем же протоколом и обучающей выборкой."""
    if smoke:
        return LC_DIR / "smoke" / f"C96_{filt}_s{seed}"
    return HOLD_DIR / f"C96_s{seed}" if filt == "full" else LC_DIR / f"C96_noR2_s{seed}"


def run_dir(filt: str, v: str, seed: int, smoke: bool) -> Path:
    return (EXP_DIR / "smoke" if smoke else EXP_DIR) / filt / f"{v}_s{seed}"


def load_frame(meta) -> pd.DataFrame:
    df = load_dataset().sort_values("row_id").reset_index(drop=True)
    assert np.array_equal(df["row_id"].to_numpy(), meta["row_id"].to_numpy())
    return df


def parts_of(meta) -> np.ndarray:
    part = meta["split"].to_numpy().astype(object)
    sel, hold, _ = split_val(meta)
    part[sel], part[hold] = "val_sel", "val_hold"
    return part


# ---------------------------------------------------------------- числовые признаки вариантов
def variant_prep(v: str, df, meta) -> Preprocessor:
    """Препроцессор чисел варианта: fit по полному train (метка не используется), как у baseline."""
    _, time_mode, clip_mode = VARIANTS[v]
    tr = meta["split"].to_numpy() == "train"
    return Preprocessor(time_mode=time_mode, clip_mode=clip_mode).fit_num(df[tr])


def num_features(v: str, df, meta, log, force: bool = False) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(num, num_mask, имена) варианта; кеш num_<вариант>.npz."""
    path = EXP_DIR / f"num_{v}.npz"
    if path.exists() and not force:
        z = np.load(path)
        if np.array_equal(z["row_id"], meta["row_id"].to_numpy()):
            return z["num"], z["num_mask"], json.loads(str(z["names"]))
    t0 = time.perf_counter()
    p = variant_prep(v, df, meta)
    num, mask = p.transform_num(df)
    info = {"variant": v, "time_mode": p.time_mode, "clip_mode": p.clip_mode, "names": p.num_names,
            "kind": p.num_kind, "clip": p.num_clip, "mean": p.num_mean, "std": p.num_std,
            "fuse_sigma": FUSE_SIGMA if p.clip_mode == "domain" else None,
            "n_fused": int((np.abs(num) >= FUSE_SIGMA).sum()) if p.clip_mode == "domain" else None}
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(path, row_id=meta["row_id"].to_numpy(), num=num, num_mask=mask, names=json.dumps(p.num_names))
    (EXP_DIR / f"num_{v}.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info("числа %s: %d признаков (%s), %.0f с", v, len(p.num_names), ", ".join(p.num_names),
             time.perf_counter() - t0)
    return num, mask, p.num_names


# ---------------------------------------------------------------- шаг 1 и проверка поведения по умолчанию
def step1(meta, log) -> None:
    """Доли строк, затронутых правилами, и |z| без обрезки по частям (до и после правил)."""
    REP_DIR.mkdir(parents=True, exist_ok=True)
    df = load_frame(meta)
    part = parts_of(meta)
    masks = domain_rule_masks(df)
    masks["any"] = np.logical_or.reduce(list(masks.values()))
    rows = []
    for k, m in masks.items():
        r = {"rule": k}
        for p in PARTS:
            r[f"n_{p}"] = int(m[part == p].sum())
            r[f"share_{p}"] = float(m[part == p].mean())
        rows.append(r)
    pd.DataFrame(rows).to_csv(REP_DIR / "step1_rules.csv", index=False)

    c96 = Preprocessor.load()
    raw_c96 = numeric_frame(df)
    tr = part == "train"
    rows = []
    for v in ("A", "Aplus"):
        p = variant_prep(v, df, meta)
        num = numeric_frame(df, p.time_mode, p.clip_mode)
        for c in p.num_names:
            x = _transform_num(num[c].to_numpy(dtype=float), p.num_kind[c])
            z = (np.clip(x, *p.num_clip[c]) - p.num_mean[c]) / p.num_std[c]  # без предохранителя
            r = {"variant": v, "feature": c, "train_std": p.num_std[c],
                 "train_mad_sigma": float(1.4826 * np.nanmedian(np.abs(x[tr] - np.nanmedian(x[tr]))))}
            for q in PARTS:
                r[f"max_abs_z_{q}"] = float(np.nanmax(np.abs(z[part == q])))
            for f in (10, 20, FUSE_SIGMA):
                r[f"n_gt_{f:g}sigma"] = int(np.nansum(np.abs(z) > f))
            if c in c96.num_clip:
                xc = _transform_num(raw_c96[c].to_numpy(dtype=float), c96.num_kind[c])
                lo, hi = c96.num_clip[c]
                for q in PARTS:
                    r[f"c96_clipped_{q}"] = int(((xc < lo) | (xc > hi))[part == q].sum())
            rows.append(r)
    z = pd.DataFrame(rows)
    z.to_csv(REP_DIR / "step1_z.csv", index=False)
    zp = z[z["variant"] == "Aplus"].set_index("feature")
    zmax = zp[[f"max_abs_z_{q}" for q in PARTS]].max(axis=1)
    log.info("шаг 1: правила затрагивают (любое) train %.3f%%, val_hold %.3f%%; Aplus max |z| %.1f (%s)",
             100 * masks["any"][tr].mean(), 100 * masks["any"][part == "val_hold"].mean(), zmax.max(), zmax.idxmax())


def check_default(meta, X, prep, device, log) -> dict:
    """Поведение по умолчанию не изменилось: числа baseline пересчитываются бит-в-бит, веса baseline грузятся
    strict, прогноз на 1000 строк val совпадает с baseline_predictions.parquet; режимы чисел — как в config."""
    df = load_frame(meta)
    tr = meta["split"].to_numpy() == "train"
    res = {"prep_time_mode": prep.time_mode, "prep_clip_mode": prep.clip_mode}
    num, mask = prep.transform_num(df)
    res["num_bitwise"] = bool(np.array_equal(num, X["num"]))
    res["num_mask_bitwise"] = bool(np.array_equal(mask, X["num_mask"]))
    p2 = Preprocessor(time_mode=prep.time_mode, clip_mode=prep.clip_mode).fit_num(df[tr])
    res["refit_same"] = bool(p2.num_names == prep.num_names and p2.num_clip == prep.num_clip
                             and p2.num_mean == prep.num_mean and p2.num_std == prep.num_std)
    model = build_model(X, prep.cat_cardinalities, DEFAULT_HP, **prep.model_kw)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"), strict=True)
    res["strict_load"] = True
    model.to(device)
    val = np.flatnonzero(meta["split"].to_numpy() == "val")[:1000]
    p = predict(model, {k: a[val] for k, a in X.items()}, device) * prep.y_std + prep.y_mean
    ref = pd.read_parquet(PRED_PATH).set_index("row_id").loc[meta["row_id"].to_numpy()[val], "pred_log"].to_numpy()
    res["pred_max_abs_diff"] = float(np.abs(p - ref).max())
    res["ok"] = bool(res["num_bitwise"] and res["num_mask_bitwise"] and res["refit_same"]
                     and res["pred_max_abs_diff"] == 0.0 and prep.time_mode == BASELINE_TIME_MODE
                     and prep.clip_mode == BASELINE_CLIP_MODE)
    REP_DIR.mkdir(parents=True, exist_ok=True)
    (REP_DIR / "check_default.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info("проверка поведения по умолчанию: %s", res)
    return res


# ---------------------------------------------------------------- обучение
def run(filt, v, seed, meta, X, nums, prep, device, log, smoke: bool, force: bool) -> None:
    out = run_dir(filt, v, seed, smoke)
    if (out / "metrics.json").exists() and not force:
        log.info("%s/%s seed %d: уже готов, пропускаю", filt, v, seed)
        return
    Xv, model_kw = inputs("C96", meta, X, log)  # те же входы, что у эталона в text_tokens_holdout
    # числа варианта (в т. ч. C96 — свои, а не из features.npz: baseline после time_clip может сменить режимы)
    Xv = {**Xv, "num": nums[v][0], "num_mask": nums[v][1]}
    y_log = meta["y_log"].to_numpy()
    drop = service_mask(np.expm1(y_log)) if filt == "noR2" else np.zeros(len(meta), dtype=bool)
    tr = np.flatnonzero((meta["split"].to_numpy() == "train") & ~drop)
    sel, hold, _ = split_val(meta)
    sel = sel[~drop[sel]]
    thp = dict(TRAIN_HP)
    if smoke:  # та же подвыборка, что в label_cleaning --smoke
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
    n_params = int(sum(q.numel() for q in model.parameters()))
    log.info("=== %s/%s (%s), seed %d: %d параметров, %d токенов; train %d, val_sel %d, val_hold %d", filt, v,
             VARIANTS[v][0], seed, n_params, model.tokenizer.n_tokens, len(tr), len(sel), len(hold))
    t0 = time.perf_counter()
    # стандартизация цели — по полному train (prep.y_mean, prep.y_std), как в baseline и label_cleaning
    model, history, best_epoch = fit(model, sub(tr), (y_log[tr] - prep.y_mean) / prep.y_std, sub(sel), y_log[sel],
                                     prep.y_mean, prep.y_std, device, thp, log, seed=seed, epoch_callback=on_epoch)
    train_sec = time.perf_counter() - t0
    P = np.stack(hold_preds)
    y_hold = y_log[hold]
    p = P[best_epoch - 1].astype(np.float64)
    hold_rmsle = np.sqrt(((P - y_hold) ** 2).mean(1))
    if not np.all(np.isfinite(hold_rmsle)) or hold_rmsle.max() > HOLD_RMSLE_SANE:
        # 2026-11-02, smoke: однажды GPU вернул мусор (val_hold RMSLE ≈ 4·10³⁴), повтор дал верный результат
        raise RuntimeError(f"{filt}/{v} seed {seed}: прогноз на val_hold вырожден (RMSLE по эпохам "
                           f"{np.round(hold_rmsle, 3).tolist()}) — запуск не сохранён, повторите")

    out.mkdir(parents=True, exist_ok=True)
    hist = pd.DataFrame(history).rename(columns={"val_rmsle": "sel_rmsle"})
    hist["hold_rmsle"] = hold_rmsle
    hist.to_csv(out / "history.csv", index=False)
    np.save(out / "hold_preds_by_epoch.npy", P)
    pd.DataFrame({"row_id": meta["row_id"].to_numpy()[hold], "y_log": y_hold, "pred_log": p}).to_parquet(
        out / "hold_predictions.parquet", index=False)
    m = {"variant": v, "filter": filt, "seed": seed, "smoke": smoke, "n_train": int(len(tr)), "n_sel": int(len(sel)),
         "n_hold": int(len(hold)), "best_epoch": int(best_epoch), "epochs_run": len(history),
         "sel_rmsle": float(hist["sel_rmsle"].iloc[best_epoch - 1]), "hold_rmsle": float(st.rmsle(y_hold, p)),
         "sec_per_epoch": float(hist["sec"].mean()), "train_seconds": round(train_sec, 1), "n_params": n_params,
         "n_tokens": int(model.tokenizer.n_tokens), "device": device_name(device)}
    (out / "metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    torch.save(model.state_dict(), out / "model.pt")
    log.info("%s/%s seed %d: эпоха %d из %d (val_sel %.5f), val_hold RMSLE %.5f, %.1f с/эпоху", filt, v, seed,
             best_epoch, len(history), m["sel_rmsle"], m["hold_rmsle"], m["sec_per_epoch"])


# ---------------------------------------------------------------- отчёт
def _ci_cols(ci: st.CI, name: str) -> dict:
    return {name: ci.estimate, f"{name}_low": ci.low, f"{name}_high": ci.high, f"{name}_method": ci.method}


def _paired(y, pv, pr, metrics=(("d_rmsle", st.rmsle), ("d_mdape_pp", st.mdape), ("d_within25", st.within25))):
    # n ≈ 53 тыс.: percentile (по плану; BCa с jackknife по медиане слишком дорог)
    kw = dict(vectorized=True, B=BOOTSTRAP_B, seed=SEED, method="percentile")
    row = {}
    for name, fn in metrics:
        row |= _ci_cols(st.paired_bootstrap_ci(fn, y, pv, pr, **kw), name)
    return row


def load_runs(filt: str, smoke: bool, log) -> tuple[list[dict], dict]:
    """Запуски фильтра; эталон C96 — свой запуск или готовый внешний (text_tokens_holdout / label_cleaning)."""
    base = (EXP_DIR / "smoke" if smoke else EXP_DIR) / filt
    runs, preds = [], {}
    for f in sorted(base.glob("*_s*/metrics.json")):
        m = json.loads(f.read_text(encoding="utf-8"))
        runs.append({**m, "source": "time_clip"})
        preds[(m["variant"], m["seed"])] = pd.read_parquet(f.parent / "hold_predictions.parquet")
    for s in sorted({m["seed"] for m in runs}):
        src = external_reference(filt, s, smoke)
        if ("C96", s) in preds or not (src / "hold_predictions.parquet").exists():
            continue
        m = json.loads((src / "metrics.json").read_text(encoding="utf-8"))
        runs.append({**{k: m.get(k) for k in ("best_epoch", "epochs_run", "sel_rmsle", "sec_per_epoch")},
                     "variant": "C96", "filter": filt, "seed": s, "source": str(src.relative_to(ARTIFACTS_DIR.parent))})
        preds[("C96", s)] = pd.read_parquet(src / "hold_predictions.parquet")
        log.info("%s: эталон C96 seed %d — %s", filt, s, src)
    return runs, preds


def check_path(preds, filt: str, smoke: bool, log) -> dict:
    """Свой запуск C96 против готового внешнего с тем же seed: совпадение бит-в-бит подтверждает, что путь
    обучения не изменился (в smoke — против label_cleaning --smoke)."""
    out = {}
    for (v, s), d in preds.items():
        src = external_reference(filt, s, smoke)
        if v != "C96" or not (src / "hold_predictions.parquet").exists() or \
                not (run_dir(filt, v, s, smoke) / "metrics.json").exists():
            continue
        e = pd.read_parquet(src / "hold_predictions.parquet")
        same = bool(np.array_equal(e["row_id"].to_numpy(), d["row_id"].to_numpy()))
        diff = float(np.abs(e["pred_log"].to_numpy() - d["pred_log"].to_numpy()).max()) if same else float("nan")
        out[s] = {"reference": str(src.relative_to(ARTIFACTS_DIR.parent)), "same_rows": same, "max_abs_diff": diff,
                  "bitwise": same and diff == 0.0}
        log.info("%s: свой C96 seed %d против %s: %s", filt, s, src, out[s])
    return out


def slices(meta_h) -> dict[str, np.ndarray]:
    r2 = service_mask(np.expm1(meta_h["y_log"].to_numpy()))
    return {"full": np.ones(len(meta_h), dtype=bool), "noR2": ~r2}


def c96_clip_mask(df) -> np.ndarray:
    """Строки, где C96 обрезал бы хотя бы одно нечисловое по времени значение (квантили train)."""
    c96 = Preprocessor.load()
    raw = numeric_frame(df)
    m = np.zeros(len(df), dtype=bool)
    for c in c96.num_names:
        if c.startswith("issue_"):
            continue
        x = _transform_num(raw[c].to_numpy(dtype=float), c96.num_kind[c])
        lo, hi = c96.num_clip[c]
        m |= (x < lo) | (x > hi)
    return m


def report(meta, log, smoke: bool = False) -> None:
    rep = REP_DIR / "smoke" if smoke else REP_DIR
    meta_i = meta.set_index("row_id")
    df = None
    res, paired, decision, by_q, by_g, rules, checks = [], [], [], [], [], [], {}
    for filt in FILTERS:
        runs, preds = load_runs(filt, smoke, log)
        if not any(m["source"] == "time_clip" and m["variant"] != "C96" for m in runs) and not smoke:
            continue
        if not runs:
            continue
        checks[filt] = check_path(preds, filt, smoke, log)
        dec_sl = DECISION_SLICE[filt]
        hold_ids = next(iter(preds.values()))["row_id"].to_numpy()
        mh = meta_i.loc[hold_ids].reset_index()
        sl = slices(mh)
        for m in runs:
            d = preds[(m["variant"], m["seed"])]
            assert np.array_equal(d["row_id"].to_numpy(), hold_ids)
            y, p = d["y_log"].to_numpy(), d["pred_log"].to_numpy()
            for name, mask in sl.items():
                res.append({"filter": filt, "variant": m["variant"], "seed": m["seed"], "slice": name,
                            "decision_slice": name == dec_sl, "n": int(mask.sum()),
                            "rmsle": float(st.rmsle(y[mask], p[mask])), "mdape_pct": float(st.mdape(y[mask], p[mask])),
                            "within25": float(st.within25(y[mask], p[mask])), "bias": float(np.mean(p[mask] - y[mask])),
                            **{k: m.get(k) for k in ("best_epoch", "epochs_run", "sel_rmsle", "sec_per_epoch",
                                                     "n_tokens", "source")}})
        seeds = sorted({s for _, s in preds})
        for v, r in COMPARISONS:
            for s in seeds:
                if (v, s) not in preds or (r, s) not in preds:
                    continue
                y = preds[(v, s)]["y_log"].to_numpy()
                pv, pr = preds[(v, s)]["pred_log"].to_numpy(), preds[(r, s)]["pred_log"].to_numpy()
                for name, mask in sl.items():
                    row = {"filter": filt, "variant": v, "reference": r, "seed": s, "slice": name,
                           "decision_slice": name == dec_sl, "n": int(mask.sum()),
                           **_paired(y[mask], pv[mask], pr[mask])}
                    row["ci_excludes_0"] = bool(row["d_rmsle_high"] < 0 or row["d_rmsle_low"] > 0)
                    paired.append(row)
                    log.info("%s: %s vs %s, seed %d, срез %s: ΔRMSLE %+.4f [%+.4f; %+.4f]", filt, v, r, s, name,
                             row["d_rmsle"], row["d_rmsle_low"], row["d_rmsle_high"])

        # по кварталам val_hold: RMSLE и смещение (средний pred − y) — нет ли завышения от экстраполяции тренда
        q = mh["issue_date"].dt.to_period("Q").astype(str).to_numpy()
        for (v, s), d in sorted(preds.items()):
            y, p = d["y_log"].to_numpy(), d["pred_log"].to_numpy()
            for qq in np.unique(q):
                ix = sl[dec_sl] & (q == qq)
                by_q.append({"filter": filt, "variant": v, "seed": s, "slice": dec_sl, "quarter": qq,
                             "n": int(ix.sum()), "rmsle": float(st.rmsle(y[ix], p[ix])),
                             "bias": float(np.mean(p[ix] - y[ix]))})

        # группы (seed 42, решающий срез): хвост ≥ $1M, тип и подтип разрешения
        if ("C96", SEED) in preds:
            y = preds[("C96", SEED)]["y_log"].to_numpy()
            groups = {"all": np.ones(len(mh), dtype=bool), "valuation>=1M": y >= TAIL_LOG}
            for col in ("PERMIT_TYPE", "PERMIT_SUB_TYPE"):
                for g in mh[col].dropna().unique():
                    groups[f"{col}={g}"] = (mh[col] == g).to_numpy()
            for v, r in COMPARISONS:
                if (v, SEED) not in preds or (r, SEED) not in preds:
                    continue
                pv, pr = preds[(v, SEED)]["pred_log"].to_numpy(), preds[(r, SEED)]["pred_log"].to_numpy()
                for g, gm in groups.items():
                    ix = sl[dec_sl] & gm
                    if ix.sum() < 30:
                        continue
                    by_g.append({"filter": filt, "variant": v, "reference": r, "slice": dec_sl, "group": g,
                                 "n": int(ix.sum()), "rmsle_reference": float(st.rmsle(y[ix], pr[ix])),
                                 "rmsle_variant": float(st.rmsle(y[ix], pv[ix])),
                                 **_paired(y[ix], pv[ix], pr[ix], (("d_rmsle", st.rmsle),))})

        # строки, которые меняют правила A+ или обрезал бы C96 (seed 42, решающий срез)
        if ("Aplus", SEED) in preds and ("C96", SEED) in preds:
            if df is None:
                df = load_frame(meta)
                pos = pd.Series(np.arange(len(meta)), index=meta["row_id"].to_numpy())
                rule_m = domain_rule_masks(df)
                rule_m = {"any_rule": np.logical_or.reduce(list(rule_m.values())), **rule_m,
                          "c96_clipped": c96_clip_mask(df)}
            ix_h = pos.loc[hold_ids].to_numpy()
            y = preds[("C96", SEED)]["y_log"].to_numpy()
            pa, pc = preds[("Aplus", SEED)]["pred_log"].to_numpy(), preds[("C96", SEED)]["pred_log"].to_numpy()
            for k, rm in rule_m.items():
                ix = sl[dec_sl] & rm[ix_h]
                row = {"filter": filt, "rows": k, "slice": dec_sl, "n": int(ix.sum())}
                if ix.sum() > 0:
                    row |= {"rmsle_C96": float(st.rmsle(y[ix], pc[ix])), "rmsle_Aplus": float(st.rmsle(y[ix], pa[ix]))}
                if ix.sum() >= 30:
                    row |= _paired(y[ix], pa[ix], pc[ix], (("d_rmsle", st.rmsle),))
                rules.append(row)

    if not res:
        log.info("нет готовых запусков")
        return
    rep.mkdir(parents=True, exist_ok=True)
    res, paired = pd.DataFrame(res), pd.DataFrame(paired)
    res.to_csv(rep / "results.csv", index=False)
    paired.to_csv(rep / "paired_vs_C96.csv", index=False)
    if not paired.empty:
        c = paired[paired["decision_slice"]]
        decision = c.groupby(["filter", "variant", "reference", "slice"]).agg(
            n_seeds=("seed", "nunique"), d_rmsle_mean=("d_rmsle", "mean"), d_rmsle_min=("d_rmsle", "min"),
            d_rmsle_max=("d_rmsle", "max"), d_rmsle_high_max=("d_rmsle_high", "max"),
            d_mdape_mean=("d_mdape_pp", "mean"), d_within25_mean=("d_within25", "mean")).reset_index()
        decision["promising"] = decision["d_rmsle_mean"] <= PROMISING_DELTA
        decision["passes_rule"] = (decision["d_rmsle_mean"] <= DECISION_DELTA) & \
                                  (decision["d_rmsle_high_max"] < 0) & (decision["n_seeds"] >= MIN_SEEDS)
        decision.loc[decision["reference"] != "C96", ["promising", "passes_rule"]] = False  # A+ vs A — для понимания
        decision.to_csv(rep / "decision.csv", index=False)
    by_q, by_g, rules = pd.DataFrame(by_q), pd.DataFrame(by_g), pd.DataFrame(rules)
    by_q.to_csv(rep / "by_quarter.csv", index=False)
    by_g.to_csv(rep / "by_group.csv", index=False)
    rules.to_csv(rep / "rules_rows.csv", index=False)
    (rep / "path_check.json").write_text(json.dumps(checks, ensure_ascii=False, indent=1), encoding="utf-8")
    write_summary(rep, meta, res, paired, decision, by_q, by_g, rules, checks, smoke)
    log.info("отчёт: %s", rep)


def write_summary(rep, meta, res, paired, decision, by_q, by_g, rules, checks, smoke) -> None:
    _, _, info = split_val(meta)
    txt = [f"# Эксперимент time_clip: время без обрезки и доменные правила{' (smoke)' if smoke else ''}", "",
           "План — `docs/context/PLAN_TIME_CLIP.md`. Модель C96, меняются только числовые признаки: "
           "A — `t` без обрезки + sin/cos месяца и дня года; Aplus — A + доменные правила вместо квантильной обрезки, "
           f"предохранитель ±{FUSE_SIGMA:g}σ.",
           f"val_sel {info['sel_dates'][0]} … {info['sel_dates'][1]} — ранняя остановка и выбор эпохи; "
           f"val_hold {info['hold_dates'][0]} … {info['hold_dates'][1]} — только оценка. Test не используется.",
           "Фильтр — обучающая выборка (full или noR2, как у принятого baseline); решающий срез val_hold совпадает "
           "с фильтром. Парный bootstrap против эталона с тем же seed, B = 2000, percentile. "
           "bias — средний (прогноз − факт) в log-шкале.", ""]
    for filt, ch in checks.items():
        for s, c in ch.items():
            txt.append(f"Проверка пути обучения ({filt}, seed {s}): свой C96 против `{c['reference']}` — "
                       f"бит-в-бит: {c['bitwise']} (max |Δ| = {c['max_abs_diff']:.3g}).")
    txt += ["", "## Запуски и метрики по срезам", "",
            _md(res[["filter", "variant", "seed", "slice", "n", "rmsle", "mdape_pct", "within25", "bias", "best_epoch",
                     "epochs_run", "sel_rmsle", "n_tokens", "source"]])]
    if not paired.empty:
        txt += ["", "## Парные сравнения", "",
                _md(paired[["filter", "variant", "reference", "seed", "slice", "decision_slice", "n", "d_rmsle",
                            "d_rmsle_low", "d_rmsle_high", "d_mdape_pp", "d_within25", "ci_excludes_0"]])]
    if len(decision):
        txt += ["", "## Решение (на решающем срезе)", "",
                f"Правило принятия: средний ΔRMSLE против C96 по seed ≤ {DECISION_DELTA} и CI без 0 при каждом seed "
                f"(не меньше {MIN_SEEDS} seed). Промежуточное решение после этапа 1: на этап 2 — средний ΔRMSLE ≤ "
                f"{PROMISING_DELTA}. Aplus против A — только для понимания.", "", _md(decision), ""]
        for _, r in decision[decision["reference"] == "C96"].iterrows():
            txt.append(f"- {r['filter']}/{r['variant']}: средний ΔRMSLE {r['d_rmsle_mean']:+.4f} "
                       f"({r['n_seeds']} seed) — {'перспективный' if r['promising'] else 'не перспективный'}, "
                       f"правило {'**проходит**' if r['passes_rule'] else 'не проходит'}.")
    if not by_q.empty:
        piv = by_q.groupby(["filter", "variant", "quarter"])[["rmsle", "bias"]].mean().reset_index()
        txt += ["", "## По кварталам val_hold (среднее по seed, решающий срез)", "", _md(piv)]
    if not by_g.empty:
        txt += ["", "## По группам (seed 42, решающий срез)", "",
                _md(by_g[["filter", "variant", "reference", "group", "n", "rmsle_reference", "rmsle_variant", "d_rmsle",
                          "d_rmsle_low", "d_rmsle_high"]])]
    if not rules.empty:
        txt += ["", "## Строки, затронутые правилами Aplus или обрезкой C96 (seed 42, решающий срез)", "",
                _md(rules)]
    notes = rep / "notes.md"
    if notes.exists():
        txt += ["", notes.read_text(encoding="utf-8").strip()]
    (rep / "summary.md").write_text("\n".join(txt) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step1", action="store_true", help="только шаг 1 (правила и |z| без обрезки)")
    ap.add_argument("--check", action="store_true", help="только проверка поведения по умолчанию")
    ap.add_argument("--filter", choices=FILTERS, default=None, help="обучающая выборка (как у принятого baseline)")
    ap.add_argument("--variants", default="A,Aplus")
    ap.add_argument("--seeds", default=str(SEED))
    ap.add_argument("--smoke", action="store_true", help=f"{SMOKE_ROWS} строк train, 2 эпохи, seed 42")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", action="store_true", help="только отчёт")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    log = get_logger(LOG_PATH)
    device = get_device(args.device)
    log.info("устройство: %s", device_name(device))
    meta, X, prep = load_features(log=log)
    if args.step1:
        step1(meta, log)
        return
    if args.check:
        check_default(meta, X, prep, device, log)
        return
    log.info("разбиение val: %s", split_val(meta)[2])
    if not args.report:
        if args.filter is None:
            raise SystemExit("укажите --filter full|noR2 — обучающую выборку принятого baseline (решение по R2)")
        if not check_default(meta, X, prep, device, log)["ok"]:
            raise SystemExit("поведение по умолчанию изменилось — запуски остановлены (см. check_default.json)")
        variants = [x.strip() for x in args.variants.split(",") if x.strip()]
        seeds = [SEED] if args.smoke else [int(x) for x in args.seeds.split(",") if x.strip()]
        df = None
        nums = {}
        for v in variants:
            if v not in VARIANTS:
                raise ValueError(f"неизвестный вариант {v}; есть {list(VARIANTS)}")
            if df is None and not (EXP_DIR / f"num_{v}.npz").exists():
                df = load_frame(meta)
            nums[v] = num_features(v, df, meta, log)[:2]
        for v in variants:
            for s in seeds:
                src = external_reference(args.filter, s, args.smoke)
                if v == "C96" and not args.smoke and not args.force and (src / "metrics.json").exists():
                    log.info("%s/C96 seed %d: эталон есть — %s", args.filter, s, src)
                    continue
                run(args.filter, v, s, meta, X, nums, prep, device, log, args.smoke, args.force)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    report(meta, log, smoke=args.smoke)


if __name__ == "__main__":
    main()
