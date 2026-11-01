"""Проверка эксперимента text_tokens без смещения от выбора эпохи (только val; test не используется).

val делится по дате на две половины (все разрешения одного дня — в одной половине):
    val_sel   — ранние даты: ранняя остановка, ReduceLROnPlateau и выбор лучшей эпохи;
    val_hold  — поздние даты: не участвует ни в каких решениях при обучении, на ней считаются метрики.
Прогнозы на val_hold сохраняются по всем эпохам, чтобы оценить оптимизм выбора эпохи
(«оракул» — минимум RMSLE на val_hold по эпохам).

Варианты: A (baseline-представление), C (слова-токены, L = 72), C96 (то же, L = 96).

Запуск из корня проекта:
    venv\\Scripts\\python src\\experiments\\text_tokens_holdout.py --variants A,C,C96 --seeds 42,43,44
    venv\\Scripts\\python src\\experiments\\text_tokens_holdout.py --report
Готовые запуски (artifacts/experiments/text_tokens/holdout/<вариант>_s<seed>/) без --force не пересчитываются.
Отчёт: reports/experiments/text_tokens/holdout/.
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
from config import ARTIFACTS_DIR, BOOTSTRAP_B, SEED  # noqa: E402
from model import DEFAULT_HP  # noqa: E402
from text_tokens import DECISION_DELTA, EXP_DIR, REP_DIR, _md, base_inputs, word_features  # noqa: E402
from train_baseline import get_logger, load_features  # noqa: E402
from training import TRAIN_HP, build_model, device_name, fit, get_device, predict, set_seed  # noqa: E402

HOLD_DIR = EXP_DIR / "holdout"
HOLD_REP = REP_DIR / "holdout"
LOG_PATH = ARTIFACTS_DIR / "logs" / "text_tokens_holdout.log"

VARIANTS = {
    "A": ("SVD(64) → 1 токен (baseline)", None),
    "C": ("SVD(64) + слова-токены, L = 72", 72),
    "C96": ("SVD(64) + слова-токены, L = 96", 96),
}


def split_val(meta) -> tuple[np.ndarray, np.ndarray, dict]:
    """Индексы val_sel (ранние даты) и val_hold (поздние); граница — по дате, ≈ 50/50."""
    val = np.flatnonzero(meta["split"].to_numpy() == "val")
    dates = meta["issue_date"].to_numpy()[val]
    order = np.argsort(dates, kind="stable")
    mid = dates[order[len(order) // 2]]
    sel, hold = val[dates < mid], val[dates >= mid]
    info = {"boundary": str(pd.Timestamp(mid).date()), "n_sel": int(len(sel)), "n_hold": int(len(hold)),
            "sel_dates": [str(pd.Timestamp(dates[dates < mid].min()).date()),
                          str(pd.Timestamp(dates[dates < mid].max()).date())],
            "hold_dates": [str(pd.Timestamp(dates[dates >= mid].min()).date()),
                           str(pd.Timestamp(dates[dates >= mid].max()).date())]}
    return sel, hold, info


def inputs(v: str, meta, X, log) -> tuple[dict, dict]:
    X = base_inputs(X)
    L = VARIANTS[v][1]
    if L is None:
        return X, {}
    words, info = word_features(meta, log, L=L)
    return {**X, "words": words}, {"word_vocab": info["vocab_size"], "word_len": L}


def run(v, seed, meta, X, prep, device, log, force: bool) -> None:
    out = HOLD_DIR / f"{v}_s{seed}"
    if (out / "metrics.json").exists() and not force:
        log.info("%s seed %d: уже готов, пропускаю", v, seed)
        return
    Xv, model_kw = inputs(v, meta, X, log)
    y_log = meta["y_log"].to_numpy()
    tr = np.flatnonzero(meta["split"].to_numpy() == "train")
    sel, hold, _ = split_val(meta)
    sub = lambda ix: {k: a[ix] for k, a in Xv.items()}  # noqa: E731
    X_hold = sub(hold)
    hold_preds = []

    def on_epoch(epoch, model):
        hold_preds.append((predict(model, X_hold, device) * prep.y_std + prep.y_mean).astype(np.float32))

    set_seed(seed)
    model = build_model(Xv, prep.cat_cardinalities, DEFAULT_HP, **model_kw)
    n_params = int(sum(q.numel() for q in model.parameters()))
    log.info("=== %s (%s), seed %d: %d параметров, %d токенов; val_sel %d, val_hold %d",
             v, VARIANTS[v][0], seed, n_params, model.tokenizer.n_tokens, len(sel), len(hold))
    t0 = time.perf_counter()
    model, history, best_epoch = fit(model, sub(tr), (y_log[tr] - prep.y_mean) / prep.y_std, sub(sel), y_log[sel],
                                     prep.y_mean, prep.y_std, device, dict(TRAIN_HP), log, seed=seed,
                                     epoch_callback=on_epoch)
    train_sec = time.perf_counter() - t0
    P = np.stack(hold_preds)  # (эпохи, n_hold)
    y_hold = y_log[hold]
    hold_rmsle = np.sqrt(((P - y_hold) ** 2).mean(1))
    p = P[best_epoch - 1].astype(np.float64)

    out.mkdir(parents=True, exist_ok=True)
    hist = pd.DataFrame(history).rename(columns={"val_rmsle": "sel_rmsle"})
    hist["hold_rmsle"] = hold_rmsle
    hist.to_csv(out / "history.csv", index=False)
    np.save(out / "hold_preds_by_epoch.npy", P)
    pd.DataFrame({"row_id": meta["row_id"].to_numpy()[hold], "y_log": y_hold, "pred_log": p}).to_parquet(
        out / "hold_predictions.parquet", index=False)
    m = {"variant": v, "seed": seed, "n_hold": int(len(hold)), "best_epoch": int(best_epoch),
         "epochs_run": len(history), "sel_rmsle": float(hist["sel_rmsle"].iloc[best_epoch - 1]),
         "hold_rmsle": float(st.rmsle(y_hold, p)), "hold_mdape_pct": float(st.mdape(y_hold, p)),
         "hold_within25": float(st.within25(y_hold, p)),
         "hold_oracle_rmsle": float(hold_rmsle.min()), "hold_oracle_epoch": int(hold_rmsle.argmin() + 1),
         "sec_per_epoch": float(hist["sec"].mean()), "train_seconds": round(train_sec, 1),
         "n_params": n_params, "n_tokens": int(model.tokenizer.n_tokens), "device": device_name(device)}
    (out / "metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    torch.save(model.state_dict(), out / "model.pt")
    log.info("%s seed %d: эпоха %d (по val_sel %.5f) → val_hold RMSLE %.5f (оракул %.5f, эпоха %d) | "
             "MdAPE %.2f%% | ±25%% %.4f", v, seed, best_epoch, m["sel_rmsle"], m["hold_rmsle"],
             m["hold_oracle_rmsle"], m["hold_oracle_epoch"], m["hold_mdape_pct"], m["hold_within25"])


def report(meta, log) -> None:
    rows, preds = [], {}
    for f in sorted(HOLD_DIR.glob("*_s*/metrics.json")):
        m = json.loads(f.read_text(encoding="utf-8"))
        rows.append(m)
        preds[(m["variant"], m["seed"])] = pd.read_parquet(f.parent / "hold_predictions.parquet")
    if not rows:
        return
    HOLD_REP.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame(rows).drop(columns=["device"]).sort_values(["variant", "seed"])
    res["optimism"] = res["hold_rmsle"] - res["hold_oracle_rmsle"]
    res.to_csv(HOLD_REP / "results.csv", index=False)

    # n_hold ≈ 53 тыс.: percentile, как в основном эксперименте (BCa с jackknife по медиане слишком дорог)
    kw = dict(vectorized=True, B=BOOTSTRAP_B, seed=SEED, method="percentile")
    paired = []
    for (v, s), d in sorted(preds.items()):
        if v == "A" or ("A", s) not in preds:
            continue
        a = preds[("A", s)]
        assert np.array_equal(a["row_id"].to_numpy(), d["row_id"].to_numpy())
        y, pv, pa = d["y_log"].to_numpy(), d["pred_log"].to_numpy(), a["pred_log"].to_numpy()
        row = {"variant": v, "seed": s, "n": len(y)}
        for name, fn in (("d_rmsle", st.rmsle), ("d_mdape_pp", st.mdape), ("d_within25", st.within25)):
            ci = st.paired_bootstrap_ci(fn, y, pv, pa, **kw)
            row |= {name: ci.estimate, f"{name}_low": ci.low, f"{name}_high": ci.high}
        row["ci_excludes_0"] = bool(row["d_rmsle_high"] < 0 or row["d_rmsle_low"] > 0)
        paired.append(row)
        log.info("holdout %s vs A, seed %d: ΔRMSLE %+.4f [%+.4f; %+.4f]", v, s, row["d_rmsle"],
                 row["d_rmsle_low"], row["d_rmsle_high"])
    paired = pd.DataFrame(paired)
    paired.to_csv(HOLD_REP / "paired_vs_A.csv", index=False)

    agg = res.groupby("variant").agg(n_seeds=("seed", "nunique"), hold_rmsle_mean=("hold_rmsle", "mean"),
                                     hold_rmsle_min=("hold_rmsle", "min"), hold_rmsle_max=("hold_rmsle", "max"),
                                     optimism_mean=("optimism", "mean"), mdape_mean=("hold_mdape_pct", "mean"),
                                     within25_mean=("hold_within25", "mean"),
                                     sec_per_epoch=("sec_per_epoch", "mean")).reset_index()
    if not paired.empty:
        pg = paired.groupby("variant").agg(d_rmsle_mean=("d_rmsle", "mean"), d_rmsle_min=("d_rmsle", "min"),
                                           d_rmsle_max=("d_rmsle", "max"), d_rmsle_high_max=("d_rmsle_high", "max"),
                                           d_mdape_mean=("d_mdape_pp", "mean"),
                                           d_within25_mean=("d_within25", "mean")).reset_index()
        agg = agg.merge(pg, on="variant", how="left")
        agg["passes_rule"] = (agg["d_rmsle_mean"] <= DECISION_DELTA) & (agg["d_rmsle_high_max"] < 0) & \
                             (agg["n_seeds"] >= 3)
    agg.to_csv(HOLD_REP / "by_variant.csv", index=False)

    _, _, info = split_val(meta)
    txt = ["# Проверка text_tokens на отложенной половине val", "",
           f"val разделён по дате: val_sel {info['sel_dates'][0]} … {info['sel_dates'][1]} (n = {info['n_sel']}) — "
           f"ранняя остановка, LR plateau, выбор эпохи; val_hold {info['hold_dates'][0]} … {info['hold_dates'][1]} "
           f"(n = {info['n_hold']}) — только оценка. Test не используется.",
           "«Оракул» — минимум RMSLE на val_hold по эпохам; optimism = выбранная по val_sel − оракул.",
           "Парный bootstrap против A с тем же seed, B = 2000, percentile.", "",
           "## Запуски", "", _md(res[["variant", "seed", "best_epoch", "epochs_run", "sel_rmsle", "hold_rmsle",
                                      "hold_oracle_rmsle", "hold_oracle_epoch", "optimism", "hold_mdape_pct",
                                      "hold_within25", "sec_per_epoch"]])]
    if not paired.empty:
        txt += ["", "## Парные сравнения с A на val_hold", "",
                _md(paired[["variant", "seed", "d_rmsle", "d_rmsle_low", "d_rmsle_high", "d_mdape_pp",
                            "d_mdape_pp_low", "d_mdape_pp_high", "d_within25", "d_within25_low",
                            "d_within25_high", "ci_excludes_0"]])]
    txt += ["", "## По вариантам", "", _md(agg), "", "## Правило решения", "",
            f"Средний ΔRMSLE по seed ≤ {DECISION_DELTA} и CI не содержит 0 при каждом seed.", ""]
    if "passes_rule" in agg:
        for _, r in agg[agg.variant != "A"].iterrows():
            txt.append(f"- {r['variant']}: средний ΔRMSLE {r['d_rmsle_mean']:+.4f} (min–max {r['d_rmsle_min']:+.4f}"
                       f"…{r['d_rmsle_max']:+.4f}) — {'**проходит**' if r['passes_rule'] else 'не проходит'}.")
    notes = HOLD_REP / "notes.md"
    if notes.exists():
        txt += ["", notes.read_text(encoding="utf-8").strip()]
    (HOLD_REP / "summary.md").write_text("\n".join(txt) + "\n", encoding="utf-8")
    log.info("отчёт: %s", HOLD_REP)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", default="A,C,C96")
    ap.add_argument("--seeds", default="42,43,44")
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
    log.info("разбиение val: %s", split_val(meta)[2])
    if not args.report:
        for v in [x.strip().upper() for x in args.variants.split(",") if x.strip()]:
            for s in [int(x) for x in args.seeds.split(",") if x.strip()]:
                run(v, s, meta, X, prep, device, log, args.force)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    report(meta, log)


if __name__ == "__main__":
    main()
