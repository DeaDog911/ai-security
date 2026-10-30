"""Обучение baseline FT-Transformer и оценка на test с 95% CI.

Запуск из корня проекта:
    venv\\Scripts\\python src\\train_baseline.py            # полное обучение (идемпотентно)
    venv\\Scripts\\python src\\train_baseline.py --smoke    # быстрый прогон: 5000 строк, 2 эпохи → artifacts/smoke/
    venv\\Scripts\\python src\\train_baseline.py --force    # переобучить (baseline после фиксации не меняем!)

Артефакты (artifacts/):
    preprocessor.joblib, features.npz        препроцессор и признаки всех строк для обучения
    baseline_model.pt                        веса лучшей эпохи
    config.json                              seed, гиперпараметры, split, версии, устройство, датасет
    baseline_history.csv                     кривые обучения
    baseline_predictions.parquet             прогнозы на train/val/test (log1p-шкала)
    baseline_metrics.json                    метрики test/val с 95% CI
reports/baseline/learning_curves.png
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import sys
import time

import numpy as np
import pandas as pd
import torch

import stats as st
from config import ARTIFACTS_DIR, BOOTSTRAP_B, DATASET_PATHS, DATASETS, REPORTS_DIR, SEED
from data import SPLIT_INFO_PATH, TARGET, load_dataset
from features import Preprocessor
from model import DEFAULT_HP
from training import TRAIN_HP, build_model, device_name, fit, get_device, predict, set_seed

FEATURES_PATH = ARTIFACTS_DIR / "features.npz"
MODEL_PATH = ARTIFACTS_DIR / "baseline_model.pt"
CONFIG_PATH = ARTIFACTS_DIR / "config.json"
HISTORY_PATH = ARTIFACTS_DIR / "baseline_history.csv"
PRED_PATH = ARTIFACTS_DIR / "baseline_predictions.parquet"
METRICS_PATH = ARTIFACTS_DIR / "baseline_metrics.json"
CURVES_PATH = REPORTS_DIR / "baseline" / "learning_curves.png"


def get_logger(path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("ai-security")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(path, encoding="utf-8")):
        h.setFormatter(fmt)
        log.addHandler(h)
    return log


def load_features(force: bool = False, log: logging.Logger | None = None):
    """Строки для обучения (meta), признаки X для всех строк и препроцессор (fit на train).

    Кеш: artifacts/preprocessor.joblib + artifacts/features.npz.
    """
    log = log or logging.getLogger("ai-security")
    df = load_dataset().sort_values("row_id").reset_index(drop=True)
    meta = df[["row_id", "split", "issue_date", "PERMIT_TYPE", "PERMIT_SUB_TYPE", "parcel"]].copy()
    meta["y_log"] = np.log1p(df[TARGET].to_numpy())
    prep_path = ARTIFACTS_DIR / "preprocessor.joblib"
    if prep_path.exists() and FEATURES_PATH.exists() and not force:
        prep = Preprocessor.load(prep_path)
        z = np.load(FEATURES_PATH)
        if np.array_equal(z["row_id"], meta["row_id"].to_numpy()):
            return meta, {k: z[k] for k in ("num", "num_mask", "cat", "txt")}, prep
        log.info("features.npz не соответствует split — пересчитываю")
    t0 = time.perf_counter()
    tr = meta["split"].to_numpy() == "train"
    prep = Preprocessor().fit(df[tr], meta["y_log"].to_numpy()[tr])
    X = prep.transform(df)
    prep.save(prep_path)
    np.savez_compressed(FEATURES_PATH, row_id=meta["row_id"].to_numpy(), **X)
    log.info("препроцессор обучен на train и применён ко всем строкам за %.0f с", time.perf_counter() - t0)
    return meta, X, prep


def library_versions() -> dict:
    import scipy
    import sklearn
    return {"python": platform.python_version(), "torch": torch.__version__,
            "hip": getattr(torch.version, "hip", None), "cuda": torch.version.cuda, "numpy": np.__version__,
            "pandas": pd.__version__, "scikit-learn": sklearn.__version__, "scipy": scipy.__version__}


def file_md5(path, chunk=1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def evaluate(y_log: np.ndarray, p_log: np.ndarray, B: int = BOOTSTRAP_B, seed: int = SEED,
             bca_max_n: int = st.BCA_MAX_N) -> dict:
    """RMSLE и MdAPE — bootstrap (BCa при n ≤ bca_max_n); доля в ±25% — Уилсон (основной) и bootstrap."""
    rel = np.abs(np.expm1(p_log) - np.expm1(y_log)) / np.expm1(y_log)
    kw = dict(vectorized=True, B=B, seed=seed, bca_max_n=bca_max_n)
    out = {
        "n": int(len(y_log)),
        "rmsle": st.bootstrap_ci(st.rmsle, y_log, p_log, **kw).as_dict(),
        "mdape_pct": st.bootstrap_ci(st.mdape, y_log, p_log, **kw).as_dict(),
        "within25_wilson": st.proportion_ci(rel <= 0.25).as_dict(),
        "within25_bootstrap": st.bootstrap_ci(st.within25, y_log, p_log, **kw).as_dict(),
    }
    return out


def plot_curves(history: pd.DataFrame, best_epoch: int, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(history["epoch"], history["train_rmsle"], label="train (по батчам, с dropout)")
    ax[0].plot(history["epoch"], history["val_rmsle"], label="val")
    ax[0].axvline(best_epoch, color="gray", ls="--", label=f"лучшая эпоха {best_epoch}")
    ax[0].set(xlabel="эпоха", ylabel="RMSLE", title="Кривые обучения FT-Transformer")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    ax[1].plot(history["epoch"], history["lr"])
    ax[1].set(xlabel="эпоха", ylabel="learning rate", yscale="log", title="Learning rate")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="переобучить модель, даже если веса уже есть")
    ap.add_argument("--force-features", action="store_true", help="пересчитать препроцессор и признаки")
    ap.add_argument("--smoke", action="store_true", help="быстрый прогон на подвыборке в artifacts/smoke/")
    ap.add_argument("--smoke-rows", type=int, default=5000)
    ap.add_argument("--epochs", type=int, default=None, help="переопределить max_epochs")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    out_dir = ARTIFACTS_DIR / "smoke" if args.smoke else ARTIFACTS_DIR
    log = get_logger(ARTIFACTS_DIR / "logs" / ("smoke.log" if args.smoke else "train_baseline.log"))
    set_seed(SEED)
    device = get_device(args.device)
    log.info("устройство: %s", device_name(device))

    model_path = out_dir / MODEL_PATH.name
    metrics_path = out_dir / METRICS_PATH.name
    if not args.smoke and model_path.exists() and metrics_path.exists() and not args.force:
        log.info("baseline уже обучен (%s); для переобучения нужен --force", model_path)
        print(json.dumps(json.loads(metrics_path.read_text(encoding="utf-8"))["test"], ensure_ascii=False, indent=1))
        return

    meta, X, prep = load_features(force=args.force_features, log=log)
    split = meta["split"].to_numpy()
    y_log = meta["y_log"].to_numpy()
    idx = {p: np.flatnonzero(split == p) for p in ("train", "val", "test")}
    if args.smoke:
        rng = np.random.default_rng(SEED)
        idx = {p: np.sort(rng.choice(v, min(args.smoke_rows if p == "train" else args.smoke_rows // 5, len(v)),
                                     replace=False)) for p, v in idx.items()}
    log.info("размеры: %s", {p: len(v) for p, v in idx.items()})

    hp = dict(DEFAULT_HP)
    thp = dict(TRAIN_HP)
    if args.smoke:
        thp.update(max_epochs=args.epochs or 2, warmup_steps=20)
    elif args.epochs:
        thp["max_epochs"] = args.epochs

    sub = lambda p: {k: v[idx[p]] for k, v in X.items()}  # noqa: E731
    y_tr_std = (y_log[idx["train"]] - prep.y_mean) / prep.y_std
    model = build_model(X, prep.cat_cardinalities, hp)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("FT-Transformer: %d параметров, %d токенов", n_params, model.tokenizer.n_tokens)

    t0 = time.perf_counter()
    model, history, best_epoch = fit(model, sub("train"), y_tr_std, sub("val"), y_log[idx["val"]],
                                     prep.y_mean, prep.y_std, device, thp, log)
    train_sec = time.perf_counter() - t0
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    hist = pd.DataFrame(history)
    hist.to_csv(out_dir / HISTORY_PATH.name, index=False)
    plot_curves(hist, best_epoch, (out_dir / "learning_curves.png") if args.smoke else CURVES_PATH)

    # прогнозы для всех частей (train — in-sample, нужен в ПР4)
    preds = []
    for p, ix in idx.items():
        pl = predict(model, sub(p), device) * prep.y_std + prep.y_mean
        preds.append(pd.DataFrame({"row_id": meta["row_id"].to_numpy()[ix], "split": p,
                                   "y_log": y_log[ix], "pred_log": pl.astype(np.float64)}))
    preds = pd.concat(preds, ignore_index=True)
    preds.to_parquet(out_dir / PRED_PATH.name, index=False)

    log.info("bootstrap CI (B=%d)...", BOOTSTRAP_B)
    metrics = {}
    for p in ("val", "test"):
        d = preds[preds["split"] == p]
        # основная оценка (test) — BCa даже при n ≈ 100 тыс. (jackknife считается один раз);
        # для val достаточно percentile
        metrics[p] = evaluate(d["y_log"].to_numpy(), d["pred_log"].to_numpy(),
                              bca_max_n=250_000 if p == "test" else st.BCA_MAX_N)
    # наивный ориентир: медиана log-цели train
    d = preds[preds["split"] == "test"]
    naive = np.full(len(d), np.median(y_log[idx["train"]]))
    metrics["test_naive_median"] = {"rmsle": float(st.rmsle(d["y_log"], naive)),
                                    "mdape_pct": float(st.mdape(d["y_log"].to_numpy(), naive))}
    metrics["train_in_sample_rmsle"] = float(st.rmsle(*preds.loc[preds.split == "train", ["y_log", "pred_log"]].T.to_numpy()))
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")

    config = {
        "task": "регрессия log1p(Valuation); метрика RMSLE (меньше — лучше), доп. MdAPE и доля в ±25%",
        "seed": SEED,
        "smoke": args.smoke,
        "dataset": {k: {"file": DATASETS[k]["file"], "name": DATASETS[k]["name"], "url": DATASETS[k]["url"],
                        "md5": file_md5(p), "export": "30.10.2026 (данные обновлены 25.10.2026)"}
                    for k, p in DATASET_PATHS.items()},
        "split": json.loads(SPLIT_INFO_PATH.read_text(encoding="utf-8")),
        "features": prep.summary(),
        "model": {"arch": "FT-Transformer (pre-LN, ReGLU, [CLS]-голова, текст — отдельный токен)",
                  "hp": hp, "n_params": int(n_params), "n_tokens": int(model.tokenizer.n_tokens),
                  "init": f"torch default + uniform(±1/sqrt(d)) для токенизатора, seed {SEED}"},
        "training": {**thp, "loss": "MSE на стандартизованном log1p(y)", "optimizer": "AdamW",
                     "early_stopping": "по val RMSLE", "best_epoch": best_epoch,
                     "epochs_run": len(history), "train_seconds": round(train_sec, 1)},
        "device": device_name(device),
        "versions": library_versions(),
        "bootstrap": {"B": BOOTSTRAP_B, "method": "BCa (scipy.stats.bootstrap), fallback percentile"},
    }
    (out_dir / CONFIG_PATH.name).write_text(json.dumps(config, ensure_ascii=False, indent=1, default=str),
                                            encoding="utf-8")
    t = metrics["test"]
    log.info("TEST: RMSLE %s | MdAPE %s %% | ±25%%: %s",
             st.CI(**t["rmsle"]), st.CI(**t["mdape_pct"]), st.CI(**t["within25_wilson"]))
    log.info("наивный прогноз (медиана train): RMSLE %.4f", metrics["test_naive_median"]["rmsle"])


if __name__ == "__main__":
    main()
