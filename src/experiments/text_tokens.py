"""Эксперимент: более богатое представление текста WORK_DESC (только val; baseline не меняется).

Варианты (гиперпараметры модели и обучения — как у baseline, DEFAULT_HP / TRAIN_HP):
    A  эталон: TF-IDF → SVD(64) → 1 токен (признаки baseline, artifacts/features.npz)
    B  TF-IDF → SVD(256) → 8 токенов по 32 компоненты, у каждого своя проекция
    C  A + слова как токены: Embedding(слово) + позиционный эмбеддинг, L = 72, PAD маскируется
    D  TF-IDF → SVD(128) → 1 токен (необязательный)
TF-IDF — векторизаторы baseline (artifacts/preprocessor.joblib); SVD и словарь обучаются на train.

Запуск из корня проекта:
    venv\\Scripts\\python src\\experiments\\text_tokens.py --check-baseline
    venv\\Scripts\\python src\\experiments\\text_tokens.py --smoke --variants B,C
    venv\\Scripts\\python src\\experiments\\text_tokens.py --variants A,B,C --seeds 42,43,44
    venv\\Scripts\\python src\\experiments\\text_tokens.py --report          # только отчёт по готовым запускам

Готовые запуски (metrics.json в artifacts/experiments/text_tokens/<вариант>_s<seed>/) без --force
не пересчитываются. Отчёт: reports/experiments/text_tokens/ (results.csv, paired_vs_A.csv,
by_group.csv, summary.md, PNG).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import torch  # noqa: E402

import stats as st  # noqa: E402
from config import ARTIFACTS_DIR, BOOTSTRAP_B, REPORTS_DIR, SEED  # noqa: E402
from data import load_dataset  # noqa: E402
from features import TEXT_COL, WordVocab, normalize_text, word_tokenize  # noqa: E402
from model import DEFAULT_HP, FTTransformer  # noqa: E402
from train_baseline import MODEL_PATH, PRED_PATH, get_logger, load_features  # noqa: E402
from training import TRAIN_HP, build_model, device_name, fit, get_device, predict, set_seed  # noqa: E402

EXP_DIR = ARTIFACTS_DIR / "experiments" / "text_tokens"
REP_DIR = REPORTS_DIR / "experiments" / "text_tokens"
LOG_PATH = ARTIFACTS_DIR / "logs" / "text_tokens.log"

WORD_LEN = 72          # ≈ 95-й перцентиль длины на train (67 слов), кратно 8, ≤ 96
WORD_MIN_DF = 5
WORD_MAX_VOCAB = 20_000
B_COMPONENTS, B_GROUPS = 256, 8
D_COMPONENTS = 128
BASELINE_VAL_RMSLE = 0.98162

VARIANTS = {
    "A": "SVD(64) → 1 токен (baseline)",
    "B": f"SVD({B_COMPONENTS}) → {B_GROUPS} токенов × {B_COMPONENTS // B_GROUPS}",
    "C": f"SVD(64) + слова-токены (L = {WORD_LEN})",
    "D": f"SVD({D_COMPONENTS}) → 1 токен",
}
DECISION_DELTA = -0.02  # правило решения: средний ΔRMSLE по seed ≤ −0,02 и CI без 0 при каждом seed


# ---------------------------------------------------------------- кеши представлений текста
def _tfidf_matrix(prep, texts: pd.Series, chunk: int = 100_000) -> sp.csr_matrix:
    parts = []
    for i in range(0, len(texts), chunk):
        t = texts.iloc[i:i + chunk]
        parts.append(sp.hstack([prep.tfidf_word.transform(t), prep.tfidf_char.transform(t)]).tocsr())
    return sp.vstack(parts).tocsr()


def svd_features(n_comp: int, meta, prep, log, force: bool = False) -> tuple[np.ndarray, dict]:
    """TruncatedSVD(n_comp) по TF-IDF baseline: fit на train, стандартизация по train. Кеш svd<n>.npz."""
    from sklearn.decomposition import TruncatedSVD
    path = EXP_DIR / f"svd{n_comp}.npz"
    if path.exists() and not force:
        z = np.load(path)
        if np.array_equal(z["row_id"], meta["row_id"].to_numpy()):
            return z["txt"], json.loads(str(z["info"]))
    t0 = time.perf_counter()
    df = load_dataset().sort_values("row_id").reset_index(drop=True)
    assert np.array_equal(df["row_id"].to_numpy(), meta["row_id"].to_numpy())
    m = _tfidf_matrix(prep, normalize_text(df[TEXT_COL]))
    tr = meta["split"].to_numpy() == "train"
    svd = TruncatedSVD(n_components=n_comp, algorithm="randomized", n_iter=5, random_state=SEED)
    svd.fit(m[tr])
    z = svd.transform(m)
    mu, sd = z[tr].mean(0), z[tr].std(0) + 1e-8
    z = ((z - mu) / sd).astype(np.float32)
    info = {"n_components": n_comp, "explained_variance": float(svd.explained_variance_ratio_.sum()),
            "seconds": round(time.perf_counter() - t0, 1)}
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(path, row_id=meta["row_id"].to_numpy(), txt=z, info=json.dumps(info))
    log.info("SVD(%d): объяснённая дисперсия %.3f, %.0f с", n_comp, info["explained_variance"], info["seconds"])
    return z, info


def word_features(meta, log, force: bool = False) -> tuple[np.ndarray, dict]:
    """Коды слов (N, WORD_LEN) int32; словарь по train. Кеш words_L<L>.npz."""
    path = EXP_DIR / f"words_L{WORD_LEN}.npz"
    if path.exists() and not force:
        z = np.load(path)
        if np.array_equal(z["row_id"], meta["row_id"].to_numpy()):
            return z["words"], json.loads(str(z["info"]))
    t0 = time.perf_counter()
    df = load_dataset().sort_values("row_id").reset_index(drop=True)
    assert np.array_equal(df["row_id"].to_numpy(), meta["row_id"].to_numpy())
    tokens = word_tokenize(df[TEXT_COL])
    split = meta["split"].to_numpy()
    tr = split == "train"
    vocab = WordVocab(WORD_MIN_DF, WORD_MAX_VOCAB).fit(tokens[tr])
    words = vocab.transform(tokens, WORD_LEN)
    lens = tokens.str.len().to_numpy()
    info = {"L": WORD_LEN, "vocab_size": vocab.size, "min_df": WORD_MIN_DF, "max_vocab": WORD_MAX_VOCAB,
            "len_quantiles_train": {str(q): float(np.quantile(lens[tr], q)) for q in (0.5, 0.9, 0.95, 0.99)},
            "seconds": round(time.perf_counter() - t0, 1)}
    for p in ("train", "val"):
        m = split == p
        nz = words[m] != 0
        info[f"truncated_{p}"] = {"n": int((lens[m] > WORD_LEN).sum()), "share": float((lens[m] > WORD_LEN).mean())}
        info[f"unk_share_{p}"] = float((words[m][nz] == WordVocab.UNK).mean())
        info[f"empty_share_{p}"] = float((lens[m] == 0).mean())
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(path, row_id=meta["row_id"].to_numpy(), words=words, info=json.dumps(info))
    (EXP_DIR / "vocab.json").write_text(json.dumps(vocab.index, ensure_ascii=False), encoding="utf-8")
    log.info("словарь: %d слов; обрезано на train %d (%.2f%%); UNK на val %.2f%%",
             vocab.size, info["truncated_train"]["n"], 100 * info["truncated_train"]["share"],
             100 * info["unk_share_val"])
    return words, info


def variant_inputs(v: str, meta, X, prep, log) -> tuple[dict, dict, dict]:
    """(признаки X, опции модели, сведения о представлении текста)."""
    if v == "A":
        return X, {}, {"explained_variance": float(prep.svd.explained_variance_ratio_.sum())}
    if v == "B":
        txt, info = svd_features(B_COMPONENTS, meta, prep, log)
        return {**X, "txt": txt}, {"txt_groups": B_GROUPS}, info
    if v == "D":
        txt, info = svd_features(D_COMPONENTS, meta, prep, log)
        return {**X, "txt": txt}, {}, info
    if v == "C":
        words, info = word_features(meta, log)
        return {**X, "words": words}, {"word_vocab": info["vocab_size"], "word_len": WORD_LEN}, info
    raise ValueError(v)


# ---------------------------------------------------------------- проверка baseline
def check_baseline(meta, X, prep, device, log) -> dict:
    """Веса baseline грузятся в FTTransformer(DEFAULT_HP) без изменений; прогноз на 1000 строк val совпадает."""
    model = build_model(X, prep.cat_cardinalities, DEFAULT_HP)
    assert isinstance(model, FTTransformer)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"), strict=True)
    model.to(device)
    val = np.flatnonzero(meta["split"].to_numpy() == "val")[:1000]
    p = predict(model, {k: X[k][val] for k in ("num", "num_mask", "cat", "txt")}, device) * prep.y_std + prep.y_mean
    ref = pd.read_parquet(PRED_PATH).set_index("row_id").loc[meta["row_id"].to_numpy()[val], "pred_log"].to_numpy()
    diff = np.abs(p - ref)
    out = {"strict_load": True, "n_tokens": model.tokenizer.n_tokens,
           "n_params": int(sum(q.numel() for q in model.parameters())),
           "max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean())}
    log.info("проверка baseline: %s", out)
    return out


# ---------------------------------------------------------------- один запуск
def run_dir(v: str, seed: int, smoke: bool) -> Path:
    return (EXP_DIR / "smoke" if smoke else EXP_DIR) / f"{v}_s{seed}"


def run(v: str, seed: int, meta, X, prep, device, log, smoke: bool, smoke_rows: int, force: bool) -> None:
    out = run_dir(v, seed, smoke)
    if (out / "metrics.json").exists() and not force:
        log.info("%s seed %d: уже готов (%s), пропускаю", v, seed, out)
        return
    Xv, model_kw, txt_info = variant_inputs(v, meta, X, prep, log)
    split = meta["split"].to_numpy()
    y_log = meta["y_log"].to_numpy()
    idx = {p: np.flatnonzero(split == p) for p in ("train", "val")}  # test не используется
    thp = dict(TRAIN_HP)
    if smoke:
        rng = np.random.default_rng(SEED)
        idx = {p: np.sort(rng.choice(ix, min(smoke_rows if p == "train" else smoke_rows // 5, len(ix)),
                                     replace=False)) for p, ix in idx.items()}
        thp.update(max_epochs=2, warmup_steps=20)
    sub = lambda p: {k: a[idx[p]] for k, a in Xv.items()}  # noqa: E731
    y_tr_std = (y_log[idx["train"]] - prep.y_mean) / prep.y_std

    set_seed(seed)  # до build_model: одинаковая инициализация при одинаковом seed
    model = build_model(Xv, prep.cat_cardinalities, DEFAULT_HP, **model_kw)
    n_params = int(sum(q.numel() for q in model.parameters()))
    log.info("=== %s (%s), seed %d: %d параметров, %d токенов, train %d, val %d",
             v, VARIANTS[v], seed, n_params, model.tokenizer.n_tokens, len(idx["train"]), len(idx["val"]))
    t0 = time.perf_counter()
    model, history, best_epoch = fit(model, sub("train"), y_tr_std, sub("val"), y_log[idx["val"]],
                                     prep.y_mean, prep.y_std, device, thp, log, seed=seed)
    train_sec = time.perf_counter() - t0
    p_val = predict(model, sub("val"), device) * prep.y_std + prep.y_mean
    y_val = y_log[idx["val"]]

    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    hist = pd.DataFrame(history)
    hist.to_csv(out / "history.csv", index=False)
    pd.DataFrame({"row_id": meta["row_id"].to_numpy()[idx["val"]], "y_log": y_val,
                  "pred_log": p_val.astype(np.float64)}).to_parquet(out / "val_predictions.parquet", index=False)
    metrics = {
        "variant": v, "description": VARIANTS[v], "seed": seed, "smoke": smoke,
        "n_val": int(len(y_val)),
        "val_rmsle": float(st.rmsle(y_val, p_val)),
        "val_mdape_pct": float(st.mdape(y_val, p_val)),
        "val_within25": float(st.within25(y_val, p_val)),
        "best_epoch": int(best_epoch), "epochs_run": len(history),
        "sec_per_epoch": float(hist["sec"].mean()), "train_seconds": round(train_sec, 1),
        "n_params": n_params, "n_tokens": int(model.tokenizer.n_tokens),
        "model_kw": model_kw, "text": txt_info, "device": device_name(device),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info("%s seed %d: val RMSLE %.5f | MdAPE %.2f%% | ±25%% %.4f | лучшая эпоха %d из %d | %.1f с/эпоху",
             v, seed, metrics["val_rmsle"], metrics["val_mdape_pct"], metrics["val_within25"],
             best_epoch, len(history), metrics["sec_per_epoch"])
    if v == "A" and seed == SEED and not smoke:
        log.info("sanity: A seed %d val RMSLE %.5f (baseline %.5f)", seed, metrics["val_rmsle"], BASELINE_VAL_RMSLE)


# ---------------------------------------------------------------- отчёт
def _load_runs(smoke: bool) -> tuple[pd.DataFrame, dict]:
    base = EXP_DIR / "smoke" if smoke else EXP_DIR
    rows, preds = [], {}
    for f in sorted(base.glob("*_s*/metrics.json")):
        m = json.loads(f.read_text(encoding="utf-8"))
        rows.append({k: m[k] for k in ("variant", "description", "seed", "n_val", "val_rmsle", "val_mdape_pct",
                                       "val_within25", "best_epoch", "epochs_run", "sec_per_epoch",
                                       "train_seconds", "n_params", "n_tokens")})
        preds[(m["variant"], m["seed"])] = pd.read_parquet(f.parent / "val_predictions.parquet")
    return pd.DataFrame(rows), preds


def _ci_cols(ci: st.CI, name: str) -> dict:
    return {name: ci.estimate, f"{name}_low": ci.low, f"{name}_high": ci.high, f"{name}_method": ci.method}


def report(meta, log, smoke: bool = False) -> None:
    res, preds = _load_runs(smoke)
    if res.empty:
        log.info("нет готовых запусков")
        return
    rep = REP_DIR / "smoke" if smoke else REP_DIR
    rep.mkdir(parents=True, exist_ok=True)
    res = res.sort_values(["variant", "seed"]).reset_index(drop=True)
    res.to_csv(rep / "results.csv", index=False)

    kw = dict(vectorized=True, B=BOOTSTRAP_B, seed=SEED)
    paired = []
    for (v, s), d in sorted(preds.items()):
        if v == "A" or ("A", s) not in preds:
            continue
        a = preds[("A", s)]
        assert np.array_equal(a["row_id"].to_numpy(), d["row_id"].to_numpy())
        y, pv, pa = d["y_log"].to_numpy(), d["pred_log"].to_numpy(), a["pred_log"].to_numpy()
        row = {"variant": v, "seed": s, "n": len(y)}
        row |= _ci_cols(st.paired_bootstrap_ci(st.rmsle, y, pv, pa, **kw), "d_rmsle")
        row |= _ci_cols(st.paired_bootstrap_ci(st.mdape, y, pv, pa, **kw), "d_mdape_pp")
        row |= _ci_cols(st.paired_bootstrap_ci(st.within25, y, pv, pa, **kw), "d_within25")
        row["ci_excludes_0"] = bool(row["d_rmsle_high"] < 0 or row["d_rmsle_low"] > 0)
        paired.append(row)
        log.info("%s vs A, seed %d: ΔRMSLE %+.4f [%+.4f; %+.4f]", v, s, row["d_rmsle"], row["d_rmsle_low"],
                 row["d_rmsle_high"])
    paired = pd.DataFrame(paired)
    paired.to_csv(rep / "paired_vs_A.csv", index=False)

    # среднее и разброс по seed
    agg = res.groupby("variant").agg(
        n_seeds=("seed", "nunique"), rmsle_mean=("val_rmsle", "mean"), rmsle_min=("val_rmsle", "min"),
        rmsle_max=("val_rmsle", "max"), mdape_mean=("val_mdape_pct", "mean"), within25_mean=("val_within25", "mean"),
        best_epoch_mean=("best_epoch", "mean"), sec_per_epoch=("sec_per_epoch", "mean"), n_params=("n_params", "first"),
        n_tokens=("n_tokens", "first")).reset_index()
    if not paired.empty:
        pg = paired.groupby("variant").agg(d_rmsle_mean=("d_rmsle", "mean"), d_rmsle_min=("d_rmsle", "min"),
                                           d_rmsle_max=("d_rmsle", "max"), d_mdape_mean=("d_mdape_pp", "mean"),
                                           d_within25_mean=("d_within25", "mean"),
                                           ci_excl0_all=("ci_excludes_0", "all"),
                                           d_rmsle_high_max=("d_rmsle_high", "max")).reset_index()
        agg = agg.merge(pg, on="variant", how="left")
        agg["passes_rule"] = (agg["d_rmsle_mean"] <= DECISION_DELTA) & (agg["d_rmsle_high_max"] < 0) & \
                             (agg["n_seeds"] >= 3)
    agg.to_csv(rep / "by_variant.csv", index=False)

    # разбивка по типу разрешения: A и лучший вариант при seed 42
    others = res[(res.variant != "A") & (res.seed == SEED)]
    by_group = pd.DataFrame()
    best = None
    if ("A", SEED) in preds and not others.empty:
        best = others.sort_values("val_rmsle").iloc[0]["variant"]
        a, b = preds[("A", SEED)], preds[(best, SEED)]
        g = meta.set_index("row_id").loc[a["row_id"].to_numpy(), ["PERMIT_TYPE", "PERMIT_SUB_TYPE"]]
        rows = []
        for col in ("PERMIT_TYPE", "PERMIT_SUB_TYPE"):
            for val, ix in g.reset_index(drop=True).groupby(col).indices.items():
                if len(ix) < 30:
                    continue
                y, pa, pb = a["y_log"].to_numpy()[ix], a["pred_log"].to_numpy()[ix], b["pred_log"].to_numpy()[ix]
                row = {"group_col": col, "group": val, "n": len(ix), "rmsle_A": float(st.rmsle(y, pa)),
                       f"rmsle_{best}": float(st.rmsle(y, pb))}
                row |= _ci_cols(st.paired_bootstrap_ci(st.rmsle, y, pb, pa, **kw), f"d_rmsle_{best}_vs_A")
                rows.append(row)
        by_group = pd.DataFrame(rows)
        by_group.to_csv(rep / "by_group.csv", index=False)

    plot_curves(rep / "learning_curves.png", smoke)
    write_summary(rep, res, paired, agg, by_group, best)
    log.info("отчёт: %s", rep)


def plot_curves(path: Path, smoke: bool) -> None:
    base = EXP_DIR / "smoke" if smoke else EXP_DIR
    colors = {"A": "tab:gray", "B": "tab:blue", "C": "tab:red", "D": "tab:green"}
    styles = {42: "-", 43: "--", 44: ":"}
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for f in sorted(base.glob("*_s*/history.csv")):
        v, s = f.parent.name.split("_s")
        h = pd.read_csv(f)
        kw = dict(color=colors.get(v, "k"), ls=styles.get(int(s), "-."), label=f"{v}, seed {s}")
        ax[0].plot(h["epoch"], h["val_rmsle"], **kw)
        ax[1].plot(h["epoch"], h["train_rmsle"], **kw)
    ax[0].axhline(BASELINE_VAL_RMSLE, color="k", lw=0.8, alpha=0.5)
    ax[0].set(xlabel="эпоха", ylabel="val RMSLE", title="val RMSLE по эпохам")
    ax[1].set(xlabel="эпоха", ylabel="train RMSLE", title="train RMSLE (по батчам, с dropout)")
    for a in ax:
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _md(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(format(x, floatfmt) if isinstance(x, float) else str(x) for x in r) + " |")
    return "\n".join(lines)


def write_summary(rep: Path, res, paired, agg, by_group, best) -> None:
    txt = ["# Эксперимент: представление текста WORK_DESC (val)", "",
           "Все сравнения — на val (n = 106 421); test не используется. Baseline не меняется.",
           "Гиперпараметры модели и обучения — как у baseline. Парный bootstrap B = 2000 против A с тем же seed.", "",
           "## Варианты", ""]
    txt += [f"- **{k}** — {d}" for k, d in VARIANTS.items() if k in set(res.variant)]
    info = {}
    for v in ("B", "C", "D"):
        f = EXP_DIR / f"{v}_s{SEED}" / "metrics.json"
        if f.exists():
            info[v] = json.loads(f.read_text(encoding="utf-8"))["text"]
    if "B" in info:
        txt.append(f"\nSVD({B_COMPONENTS}): объяснённая дисперсия {info['B']['explained_variance']:.3f} "
                   f"(SVD(64) — 0,437).")
    if "D" in info:
        txt.append(f"SVD({D_COMPONENTS}): объяснённая дисперсия {info['D']['explained_variance']:.3f}.")
    if "C" in info:
        c = info["C"]
        txt.append(f"\nСловарь C: {c['vocab_size']} кодов (с PAD и UNK), df ≥ {c['min_df']}; длина на train — "
                   f"медиана {c['len_quantiles_train']['0.5']:.0f}, 95-й перцентиль "
                   f"{c['len_quantiles_train']['0.95']:.0f}; L = {c['L']}: обрезано "
                   f"{c['truncated_train']['n']} текстов train ({100 * c['truncated_train']['share']:.2f}%) и "
                   f"{c['truncated_val']['n']} val ({100 * c['truncated_val']['share']:.2f}%); "
                   f"доля UNK среди слов val {100 * c['unk_share_val']:.2f}%.")
    txt += ["", "## Все запуски", "", _md(res.drop(columns=["description"]))]
    if not paired.empty:
        p = paired[["variant", "seed", "d_rmsle", "d_rmsle_low", "d_rmsle_high", "d_mdape_pp", "d_mdape_pp_low",
                    "d_mdape_pp_high", "d_within25", "d_within25_low", "d_within25_high", "ci_excludes_0"]]
        txt += ["", "## Парные сравнения с A (тот же seed), percentile bootstrap", "", _md(p)]
    txt += ["", "## По вариантам (среднее и min–max по seed)", "", _md(agg)]
    if not by_group.empty:
        txt += ["", f"## Разбивка RMSLE по типу разрешения: A и {best}, seed {SEED}", "", _md(by_group)]
    txt += ["", "![кривые обучения](learning_curves.png)", "", "## Вывод", "",
            f"Правило решения: средний ΔRMSLE по seed ≤ {DECISION_DELTA} и CI парного bootstrap не содержит 0 "
            "при каждом seed.", ""]
    if "passes_rule" in agg:
        for _, r in agg[agg.variant != "A"].iterrows():
            verdict = "**проходит**" if r["passes_rule"] else "не проходит"
            txt.append(f"- {r['variant']}: средний ΔRMSLE {r['d_rmsle_mean']:+.4f} "
                       f"(min–max {r['d_rmsle_min']:+.4f}…{r['d_rmsle_max']:+.4f}); CI без 0 при всех seed: "
                       f"{'да' if r['ci_excl0_all'] else 'нет'} — {verdict}.")
    (rep / "summary.md").write_text("\n".join(txt) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", default="A,B,C")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--force", action="store_true", help="пересчитать готовые запуски")
    ap.add_argument("--smoke", action="store_true", help="5000 строк train, 2 эпохи → .../smoke/")
    ap.add_argument("--smoke-rows", type=int, default=5000)
    ap.add_argument("--check-baseline", action="store_true", help="только проверить загрузку весов baseline")
    ap.add_argument("--report", action="store_true", help="только собрать отчёт по готовым запускам")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):  # при перенаправлении в файл консоль Windows — cp1251
        sys.stdout.reconfigure(encoding="utf-8")
    log = get_logger(LOG_PATH.with_name("text_tokens_smoke.log") if args.smoke else LOG_PATH)
    device = get_device(args.device)
    log.info("устройство: %s", device_name(device))
    meta, X, prep = load_features(log=log)

    if args.check_baseline:
        out = check_baseline(meta, X, prep, device, log)
        EXP_DIR.mkdir(parents=True, exist_ok=True)
        (EXP_DIR / "check_baseline.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        return
    if not args.report:
        variants = [v.strip().upper() for v in args.variants.split(",") if v.strip()]
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        for v in variants:
            for s in seeds:
                run(v, s, meta, X, prep, device, log, args.smoke, args.smoke_rows, args.force)
                torch.cuda.empty_cache() if device.type == "cuda" else None
    report(meta, log, smoke=args.smoke)


if __name__ == "__main__":
    main()
