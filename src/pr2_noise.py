"""ПР2. Устойчивость baseline (C96 без R2) к шуму и естественным искажениям входа → reports/pr2/.

План — docs/context/PLAN_PR2.md. Модель и препроцессор не меняются; шум накладывается на исходные столбцы
test до препроцессора, дальше — обычный transform и прогноз.

Запуск из корня проекта:
    venv\\Scripts\\python src\\pr2_noise.py              # прогоны (идемпотентно, кеш artifacts/pr2/) + отчёт
    venv\\Scripts\\python src\\pr2_noise.py --smoke      # 5 000 строк test, s ∈ {0,2; 0,4}, r = 2, B = 200
    venv\\Scripts\\python src\\pr2_noise.py --report     # только отчёт по готовому кешу
    venv\\Scripts\\python src\\pr2_noise.py --only T1,X2 # часть типов шума
    venv\\Scripts\\python src\\pr2_noise.py --force      # удалить кеш прогонов и посчитать заново

Типы искажений (s — доля затронутых значений, сетка s ∈ {0; 0,1; …; 0,5}, r = 5 реализаций):
    T1  таблица  пропуски (MCAR): значение → пусто (числа: площадь, высота, DU, ADU, координаты — пара lat/lon
                 стирается вместе; категории — все, кроме PERMIT_TYPE; даты не трогаем)
    T2  таблица  порча категорий: непустая категория → случайное значение из частот train
    X1  текст    опечатки: слово (≥ 3 букв, не число) получает одну ошибку — перестановка соседних, удаление,
                 вставка буквы или соседняя клавиша QWERTY (равновероятно)
    X2  текст    удаление слов: каждое слово (токен модели: буквы или число) удаляется
Seed реализации j уровня k типа t: 42 + 1000·t + 100·k + j (k = 10·s). Профиль по признакам (T1, T2 при s = 0,3,
портится одна группа признаков): 42 + 100 000 + 1000·t + 10·g + j.

Статистика: M = RMSLE (меньше — лучше); ΔM(s) = M(0) − M(s) (отрицательное — ухудшение), R_rel = M(s)/M(0)
(> 1 — ухудшение), R-AUC = 1/(s_max − s_min)·∫R(s)ds по трапециям с s₀ = 0 (меньше — лучше).
Ошибка объекта на уровне s — средний квадрат лог-ошибки по r реализациям; парный bootstrap по объектам
(одни и те же перевыборки для clean и всех точек), B = 2000, percentile; M(0) — BCa, как у baseline.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import time
from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import stats as st  # noqa: E402
from config import ARTIFACTS_DIR, BOOTSTRAP_B, EXCLUDE_SERVICE, REPORTS_DIR, SEED  # noqa: E402
from data import load_dataset, service_mask  # noqa: E402
from features import TEXT_COL, categorical_frame, numeric_frame  # noqa: E402
from model import DEFAULT_HP  # noqa: E402
from train_baseline import METRICS_PATH, MODEL_PATH, PRED_PATH, evaluate, file_md5, get_logger, load_features  # noqa: E402
from training import build_model, device_name, get_device, predict  # noqa: E402

LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5)
R_REAL = 5
SUBGROUP_LEVELS = (0.2, 0.4)
FEATURE_LEVEL = 0.3
PRACTICAL = 0.02      # ΔRMSLE, которую в экспериментах считали значимой
SEED_SPREAD = 0.01    # разброс RMSLE между seed обучения

TYPES = {
    "T1": {"t": 1, "modality": "таблица", "name": "Пропуски (MCAR)", "kind": "tab", "natural": 0.237,
           "natural_label": "ПР1: 23,7% пропусков"},
    "T2": {"t": 2, "modality": "таблица", "name": "Порча категорий", "kind": "tab", "natural": 0.234,
           "natural_label": "ПР1: 23,4% переименованных CNC"},
    "X1": {"t": 3, "modality": "текст", "name": "Опечатки", "kind": "text"},
    "X2": {"t": 4, "modality": "текст", "name": "Удаление слов", "kind": "text"},
}
ELIGIBLE = {"T1": "все ячейки затрагиваемых признаков", "T2": "непустые категории",
            "X1": "слова из ≥ 3 букв", "X2": "все слова (токены модели)"}
COLORS = {"T1": "#2a78d6", "T2": "#eb6834", "X1": "#1baf7a", "X2": "#eda100"}  # dataviz: слоты 1–4
MARKERS = {"T1": "o", "T2": "s", "X1": "^", "X2": "D"}

# исходные числовые столбцы, которые портит T1 (time-признаки, days_to_issue и desc_len — производные дат и текста)
RAW_NUM = ("square_footage", "height", "du_changed", "adu_changed", "junior_adu", "lat", "lon")
PAIRS = {"lat": ("lat", "lon"), "lon": ("lat", "lon")}  # координаты пропадают вместе (как в данных)
KEEP_CAT = {"PERMIT_TYPE"}
DERIVED_CAT = {"zone_base": "ZONE"}  # производный признак → исходный столбец

# группы признаков для профиля «какой признак виноват» (берутся только существующие в препроцессоре)
FEATURE_GROUPS = [
    ("Площадь", ["square_footage"]),
    ("Высота", ["height"]),
    ("Квартиры (DU, ADU)", ["du_changed", "adu_changed", "junior_adu"]),
    ("Координаты", ["lat", "lon"]),
    ("Подтип и назначение", ["PERMIT_SUB_TYPE", "USE_DESC"]),
    ("Конструкция", ["CONSTRUCTION"]),
    ("Business Unit", ["Business Unit"]),
    ("Зонирование", ["ZONE"]),
    ("ZIP и census tract", ["ZIP_CODE", "CT"]),
    ("Районы (CD, CNC, CPA, APC)", ["CD", "CNC", "CPA", "APC"]),
    ("Флаги (HL, TYPE_LAT_LON, EV)", ["HL", "TYPE_LAT_LON", "EV"]),
]

LETTER_RE = re.compile(r"[A-Za-z]+")
TOKEN_RE = re.compile(r"[a-z]+|\d+(?:\.\d+)?", re.IGNORECASE)  # как features.WORD_RE после lower()
MIN_TYPO_LEN = 3
ALPHA = "abcdefghijklmnopqrstuvwxyz"
QWERTY = {
    "q": "wa", "w": "qeas", "e": "wrsd", "r": "etdf", "t": "ryfg", "y": "tugh", "u": "yihj", "i": "uojk",
    "o": "ipkl", "p": "ol", "a": "qwsz", "s": "weadzx", "d": "erfsxc", "f": "rtdgcv", "g": "tyfhvb",
    "h": "yugjbn", "j": "uihknm", "k": "iojlm", "l": "opk", "z": "asx", "x": "zsdc", "c": "xdfv", "v": "cfgb",
    "b": "vghn", "n": "bhjm", "m": "njk",
}

log = logging.getLogger("ai-security")


# ====================================================================== шумы
def _case(c: str, ref: str) -> str:
    return c.upper() if ref.isupper() else c


def typo(w: str, u_op: float, u_pos: float, u_chr: float) -> str:
    """Одна символьная ошибка в слове w (len ≥ 2) по трём равномерным числам."""
    n = len(w)
    op = min(int(u_op * 4), 3)
    if op == 0:  # перестановка соседних
        i = min(int(u_pos * (n - 1)), n - 2)
        return w[:i] + w[i + 1] + w[i] + w[i + 2:]
    if op == 1:  # удаление
        i = min(int(u_pos * n), n - 1)
        return w[:i] + w[i + 1:]
    if op == 2:  # вставка случайной буквы
        i = min(int(u_pos * (n + 1)), n)
        return w[:i] + _case(ALPHA[min(int(u_chr * 26), 25)], w[min(i, n - 1)]) + w[i:]
    i = min(int(u_pos * n), n - 1)  # соседняя клавиша
    nb = QWERTY[w[i].lower()]
    return w[:i] + _case(nb[min(int(u_chr * len(nb)), len(nb) - 1)], w[i]) + w[i + 1:]


def typo_texts(texts: list[str], p: float, rng: np.random.Generator):
    """X1. Возвращает (тексты, eligible, selected, changed) — последние три по строкам."""
    n = len(texts)
    elig, sel, ch = np.zeros(n, np.int32), np.zeros(n, np.int32), np.zeros(n, np.int32)
    out = []
    for i, t in enumerate(texts):
        runs = [m.span() for m in LETTER_RE.finditer(t) if m.end() - m.start() >= MIN_TYPO_LEN]
        elig[i] = len(runs)
        if not runs:
            out.append(t)
            continue
        u = rng.random((len(runs), 4))  # расход случайных чисел не зависит от p
        hit = u[:, 0] < p
        if not hit.any():
            out.append(t)
            continue
        pieces, pos = [], 0
        for (a, b), uu, h in zip(runs, u, hit):
            if not h:
                continue
            w = t[a:b]
            nw = typo(w, uu[1], uu[2], uu[3])
            sel[i] += 1
            ch[i] += nw != w
            pieces += [t[pos:a], nw]
            pos = b
        pieces.append(t[pos:])
        out.append("".join(pieces))
    return out, elig, sel, ch


def delete_words(texts: list[str], p: float, rng: np.random.Generator):
    """X2. Удаляется токен модели вместе с одним соседним пробелом; соседние токены не склеиваются."""
    n = len(texts)
    elig, sel = np.zeros(n, np.int32), np.zeros(n, np.int32)
    out = []
    for i, t in enumerate(texts):
        spans = [m.span() for m in TOKEN_RE.finditer(t)]
        elig[i] = len(spans)
        if not spans:
            out.append(t)
            continue
        hit = rng.random(len(spans)) < p
        if not hit.any():
            out.append(t)
            continue
        pieces, pos, L, last = [], 0, len(t), ""
        for (a, b), h in zip(spans, hit):
            if not h:
                continue
            if b < L and t[b].isspace():
                b += 1
            elif a > pos and t[a - 1].isspace():
                a -= 1
            if a > pos:
                pieces.append(t[pos:a])
                last = t[a - 1]
            if last.isalnum() and b < L and t[b].isalnum():  # «abc1def» без «1» не склеивается в «abcdef»
                pieces.append(" ")
                last = " "
            pos = b
            sel[i] += 1
        pieces.append(t[pos:])
        out.append("".join(pieces))
    return out, elig, sel, sel.copy()


@dataclass
class Run:
    key: str
    ntype: str
    level: float
    j: int
    seed: int
    group: str | None = None
    units: list[tuple[str, ...]] = field(default_factory=list)  # для T1/T2


class Ctx:
    """Test, чистые признаки, модель и всё, что нужно шумам."""

    def __init__(self, smoke: bool, smoke_rows: int, device_arg: str | None):
        df = load_dataset().sort_values("row_id").reset_index(drop=True)
        meta, X, prep = load_features(log=log)
        if not np.array_equal(meta["row_id"].to_numpy(), df["row_id"].to_numpy()):
            raise RuntimeError("порядок строк features.npz и load_dataset не совпадает")
        idx = np.flatnonzero(df["split"].to_numpy() == "test")
        if smoke:
            idx = np.sort(np.random.default_rng(SEED).choice(idx, smoke_rows, replace=False))
        self.idx, self.prep = idx, prep
        self.df = df.iloc[idx].reset_index(drop=True)
        self.X = {k: v[idx] for k, v in X.items()}
        self.y = meta["y_log"].to_numpy()[idx]
        self.task = ~service_mask(np.expm1(self.y)) if EXCLUDE_SERVICE else np.ones(len(idx), bool)
        self.texts = self.df[TEXT_COL].tolist()
        self.n_tokens = np.array([len(TOKEN_RE.findall(t)) for t in self.texts], np.int32)
        # какие исходные столбцы портим
        self.num_cols = [c for c in prep.num_names if c in RAW_NUM]
        cats = []
        for c in prep.cat_names:
            c = DERIVED_CAT.get(c, c)
            if c not in KEEP_CAT and c not in cats:
                cats.append(c)
        self.cat_cols = cats
        num_view = numeric_frame(self.df, prep.time_mode, prep.clip_mode)  # как видит модель
        catf = categorical_frame(self.df)
        self.cat_norm = {c: catf[c].fillna("").to_numpy(object) for c in cats}
        self.present = {c: num_view[c].notna().to_numpy() for c in self.num_cols}
        self.present.update({c: catf[c].notna().to_numpy() for c in cats})
        tr = categorical_frame(df[df["split"] == "train"])
        self.train_dist = {}
        for c in cats:
            vc = tr[c].value_counts().sort_index()
            self.train_dist[c] = (vc.index.to_numpy(object), (vc / vc.sum()).to_numpy(float))
        self.has_area = (self.df["square_footage"].fillna(0).to_numpy() > 0)
        self.device = get_device(device_arg)
        self._model = None

    # ------------------------------------------------------------ единицы шума
    def units(self, cols=None, ntype="T1") -> list[tuple[str, ...]]:
        allowed = (self.num_cols + self.cat_cols) if ntype == "T1" else self.cat_cols
        out = []
        for c in (cols or allowed):
            if c not in allowed:
                continue
            u = PAIRS.get(c, (c,)) if ntype == "T1" else (c,)
            u = tuple(x for x in u if x in allowed)
            if u not in out:
                out.append(u)
        return out

    def unit_present(self, u: tuple[str, ...]) -> np.ndarray:
        return np.logical_or.reduce([self.present[c] for c in u])

    # ------------------------------------------------------------ применение шума
    def apply(self, run: Run) -> tuple[pd.DataFrame, dict]:
        rng = np.random.default_rng(run.seed)
        p, task, n = run.level, self.task, len(self.df)
        df = self.df.copy()
        if run.ntype == "T1":
            cells = sel_n = ch_n = 0
            for u in run.units:
                sel = rng.random(n) < p
                for c in u:
                    df[c] = df[c].mask(sel)
                cells += int(task.sum())
                sel_n += int((sel & task).sum())
                ch_n += int((sel & self.unit_present(u) & task).sum())
            return df, {"eligible": cells, "selected": sel_n, "changed": ch_n}
        if run.ntype == "T2":
            cells = sel_n = ch_n = 0
            for u in run.units:
                c = u[0]
                vals, probs = self.train_dist[c]
                pres = self.present[c]
                sel = (rng.random(n) < p) & pres
                draw = vals[rng.choice(len(vals), size=n, p=probs)]
                new = df[c].astype(object).to_numpy(copy=True)
                new[sel] = draw[sel]
                df[c] = new
                cells += int((pres & task).sum())
                sel_n += int((sel & task).sum())
                ch_n += int((sel & (draw != self.cat_norm[c]) & task).sum())
            return df, {"eligible": cells, "selected": sel_n, "changed": ch_n}
        fn = typo_texts if run.ntype == "X1" else delete_words
        texts, elig, sel, ch = fn(self.texts, p, rng)
        df[TEXT_COL] = texts
        return df, {"eligible": int(elig[task].sum()), "selected": int(sel[task].sum()),
                    "changed": int(ch[task].sum()), "words": int(self.n_tokens[task].sum())}

    def featurize(self, df: pd.DataFrame, kind: str) -> dict[str, np.ndarray]:
        """Признаки искажённого test; не затронутые шумом части берутся из чистых признаков."""
        X = dict(self.X)
        X["num"], X["num_mask"] = self.prep.transform_num(df)  # у текста — desc_len
        if kind == "tab":
            X["cat"] = self.prep.transform_cat(df)
        else:
            X["txt"] = self.prep.transform_txt(df)
            X["words"] = self.prep.transform_words(df)
        return X

    @property
    def model(self):
        if self._model is None:
            m = build_model(self.X, self.prep.cat_cardinalities, DEFAULT_HP, **self.prep.model_kw)
            m.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
            self._model = m.to(self.device)
        return self._model

    def predict(self, X) -> np.ndarray:
        """log1p-прогноз (float32) — тем же путём, что в train_baseline."""
        return predict(self.model, X, self.device) * self.prep.y_std + self.prep.y_mean


# ====================================================================== прогоны и кеш
def plan_runs(ctx: Ctx, types, levels, r, features: bool) -> list[Run]:
    runs = []
    for nt in types:
        t = TYPES[nt]["t"]
        units = ctx.units(ntype=nt) if TYPES[nt]["kind"] == "tab" else []
        for s in levels:
            k = int(round(s * 10))
            for j in range(r):
                runs.append(Run(f"{nt}_s{s:.2f}_j{j}", nt, s, j, SEED + 1000 * t + 100 * k + j, None, units))
    if features:
        for nt in [x for x in types if TYPES[x]["kind"] == "tab"]:
            t = TYPES[nt]["t"]
            for g, (gname, cols) in enumerate(FEATURE_GROUPS, start=1):
                units = ctx.units(cols, nt)
                if not units:
                    continue
                for j in range(r):
                    runs.append(Run(f"{nt}_g{g:02d}_s{FEATURE_LEVEL:.2f}_j{j}", nt, FEATURE_LEVEL, j,
                                    SEED + 100_000 + 1000 * t + 10 * g + j, gname, units))
    return runs


def cache_meta(ctx: Ctx, smoke: bool) -> dict:
    return {"model_md5": file_md5(MODEL_PATH), "preprocessor_md5": file_md5(ARTIFACTS_DIR / "preprocessor.joblib"),
            "n_rows": int(len(ctx.idx)), "row_ids_sum": int(ctx.df["row_id"].sum()), "smoke": smoke}


def check_cache(run_dir, meta: dict, force: bool) -> None:
    path = run_dir / "meta.json"
    if force and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved != meta:
            raise RuntimeError(f"кеш {run_dir} посчитан для другой модели/строк: {saved} ≠ {meta}; "
                               "пересчёт — флаг --force")
    else:
        path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


def save_run(run_dir, run: Run, pred: np.ndarray, stats: dict, sec: float) -> None:
    info = {"key": run.key, "type": run.ntype, "level": run.level, "j": run.j, "seed": run.seed,
            "group": run.group, "units": ["+".join(u) for u in run.units], "sec": round(sec, 1), **stats}
    np.save(run_dir / f"{run.key}.npy", pred.astype(np.float32))
    (run_dir / f"{run.key}.json").write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")


def self_checks(ctx: Ctx, smoke: bool) -> np.ndarray:
    """s = 0 через весь путь даёт чистые признаки и прогноз baseline; одинаковый seed → одинаковый шум."""
    for nt, spec in TYPES.items():
        run = Run("check", nt, 0.0, 0, SEED, None, ctx.units(ntype=nt) if spec["kind"] == "tab" else [])
        df, stats = ctx.apply(run)
        if stats["changed"] != 0:
            raise AssertionError(f"{nt}: при s = 0 изменено {stats['changed']} значений")
        X = ctx.featurize(df, spec["kind"])
        bad = [k for k in X if not np.array_equal(X[k], ctx.X[k])]
        if bad:
            raise AssertionError(f"{nt}: при s = 0 признаки {bad} не совпали с features.npz")
    log.info("проверка: s = 0 по всем типам — признаки совпали с features.npz бит-в-бит")
    for nt, spec in TYPES.items():  # воспроизводимость реализации шума
        run = Run("check", nt, 0.3, 0, SEED + 7, None, ctx.units(ntype=nt) if spec["kind"] == "tab" else [])
        a, _ = ctx.apply(run)
        b, _ = ctx.apply(run)
        c, _ = ctx.apply(Run("check", nt, 0.3, 1, SEED + 8, None, run.units))
        cols = [TEXT_COL] if spec["kind"] == "text" else sorted({x for u in run.units for x in u})
        same = a[cols].astype(str).equals(b[cols].astype(str))
        diff = not a[cols].astype(str).equals(c[cols].astype(str))
        if not (same and diff):
            raise AssertionError(f"{nt}: одинаковый seed → одинаковый шум: {same}; разные seed → разный: {diff}")
    log.info("проверка: одинаковый seed даёт одинаковую реализацию, разные seed — разные")

    pred = ctx.predict(ctx.X)
    base = pd.read_parquet(PRED_PATH).set_index("row_id").loc[ctx.df["row_id"], "pred_log"].to_numpy()
    diff = float(np.max(np.abs(pred.astype(np.float64) - base)))
    if smoke:
        log.info("проверка (smoke): прогноз на чистом test против baseline — max |Δ| = %.2e "
                 "(другие батчи и, возможно, устройство — бит-в-бит не требуется)", diff)
    elif diff != 0:
        raise AssertionError(f"прогноз на чистом test не совпал с baseline_predictions: max |Δ| = {diff:.3e}")
    else:
        log.info("проверка: прогноз на чистом test совпал с baseline бит-в-бит (Δ = 0)")
    return pred


def run_all(ctx: Ctx, runs: list[Run], run_dir) -> None:
    todo = [r for r in runs if not (run_dir / f"{r.key}.npy").exists()]
    log.info("прогонов: %d, в кеше: %d, считать: %d", len(runs), len(runs) - len(todo), len(todo))
    t_all = time.perf_counter()
    for i, run in enumerate(todo, start=1):
        t0 = time.perf_counter()
        df, stats = ctx.apply(run)
        X = ctx.featurize(df, TYPES[run.ntype]["kind"])
        pred = ctx.predict(X)
        sec = time.perf_counter() - t0
        save_run(run_dir, run, pred, stats, sec)
        el = time.perf_counter() - t_all
        log.info("[%d/%d] %s: изменено %.3f (%s), RMSLE %.4f, %.0f с; осталось ≈ %.0f мин", i, len(todo), run.key,
                 stats["changed"] / max(stats["eligible"], 1), ELIGIBLE[run.ntype],
                 float(st.rmsle(ctx.y[ctx.task], pred[ctx.task])), sec, el / i * (len(todo) - i) / 60)


# ====================================================================== bootstrap
def boot_weights(n: int, B: int, seed: int, batch: int = 50):
    """Пачки весов-счётчиков (bb, n) одной и той же последовательности перевыборок для данного seed."""
    rng = np.random.default_rng(seed)
    done = 0
    while done < B:
        bb = min(batch, B - done)
        idx = rng.integers(0, n, size=(bb, n))
        W = np.empty((bb, n), np.float32)
        for i in range(bb):
            W[i] = np.bincount(idx[i], minlength=n)
        yield W
        done += bb


def wmedian(vals_sorted: np.ndarray, obj_sorted: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Взвешенная медиана (нижняя) для пачки весов объектов W (bb, n)."""
    cs = np.cumsum(W[:, obj_sorted], axis=1)
    pos = (cs < cs[:, -1:] / 2).sum(axis=1)
    return vals_sorted[np.minimum(pos, len(vals_sorted) - 1)]


def boot(E: np.ndarray, B: int, seed: int, W25: np.ndarray | None = None, ape: list | None = None) -> dict:
    """Bootstrap-распределения (B, K): RMSLE по средним квадратам E (n, K); при W25/ape — доля в ±25% и MdAPE."""
    n = len(E)
    out = {"rmsle": [], "w25": [], "mdape": []}
    sorted_ape = None
    if ape is not None:
        sorted_ape = []
        for a in ape:  # a: (n, r) относительные ошибки
            r = a.shape[1]
            flat = a.ravel()
            order = np.argsort(flat, kind="stable")
            sorted_ape.append((flat[order], (order // r).astype(np.int64)))
    for W in boot_weights(n, B, seed):
        out["rmsle"].append(np.sqrt(W.astype(np.float64) @ E / n))
        if W25 is not None:
            out["w25"].append(W.astype(np.float64) @ W25 / n)
        if sorted_ape is not None:
            out["mdape"].append(np.stack([100 * wmedian(v, o, W) for v, o in sorted_ape], 1))
    return {k: np.concatenate(v) for k, v in out.items() if v}


def pci(x: np.ndarray) -> tuple[float, float]:
    lo, hi = np.quantile(x, [0.025, 0.975])
    return float(lo), float(hi)


# ====================================================================== отчёт
class Report:
    def __init__(self, ctx: Ctx, run_dir, out_dir, types, levels, r, B, features: bool, smoke: bool):
        self.ctx, self.run_dir, self.out, self.types, self.levels, self.r, self.B = \
            ctx, run_dir, out_dir, types, list(levels), r, B
        self.features, self.smoke = features, smoke
        self.task = ctx.task
        self.y = ctx.y[self.task]
        out_dir.mkdir(parents=True, exist_ok=True)

    def load(self, key: str) -> np.ndarray:
        path = self.run_dir / f"{key}.npy"
        if not path.exists():
            raise FileNotFoundError(f"нет прогона {key} в {self.run_dir}: запустите без --report")
        return np.load(path)[self.task].astype(np.float64)

    def info(self, key: str) -> dict:
        return json.loads((self.run_dir / f"{key}.json").read_text(encoding="utf-8"))

    def save(self, df: pd.DataFrame, name: str) -> pd.DataFrame:
        df.to_csv(self.out / f"{name}.csv", index=False, encoding="utf-8-sig")
        return df

    def savefig(self, fig, name: str) -> None:
        fig.tight_layout()
        fig.savefig(self.out / f"{name}.png", dpi=130)
        plt.close(fig)

    # ------------------------------------------------------------ данные
    def collect(self) -> None:
        y = self.y
        self.P0 = self.load("clean")[:, None]
        self.cols = [("clean", 0.0)]
        P = [self.P0]
        for nt in self.types:
            for s in self.levels:
                P.append(np.stack([self.load(f"{nt}_s{s:.2f}_j{j}") for j in range(self.r)], 1))
                self.cols.append((nt, s))
        self.P = P
        self.E = np.stack([((p - y[:, None]) ** 2).mean(1) for p in P], 1)
        yv = np.expm1(y)[:, None]
        self.APE = [np.abs(np.expm1(p) - yv) / yv for p in P]
        self.W25 = np.stack([(a <= 0.25).mean(1) for a in self.APE], 1)
        self.col_idx = {c: i for i, c in enumerate(self.cols)}

    # ------------------------------------------------------------ baseline
    def baseline(self) -> dict:
        p0 = self.P0[:, 0]
        ev = evaluate(self.y, p0, B=BOOTSTRAP_B, bca_max_n=250_000) if not self.smoke else \
            evaluate(self.y, p0, B=self.B)
        saved = json.loads(METRICS_PATH.read_text(encoding="utf-8"))["test"]
        match = (not self.smoke) and ev == saved
        rows = []
        for key, name in (("rmsle", "RMSLE"), ("mdape_pct", "MdAPE, %"), ("within25_wilson", "доля в ±25%"),
                          ("within25_bootstrap", "доля в ±25% (bootstrap)")):
            c = ev[key]
            rows.append({"metric": name, "value": c["estimate"], "ci_low": c["low"], "ci_high": c["high"],
                         "ci_method": c["method"], "n": c["n"], "B": c["B"],
                         "baseline_metrics_json": saved[key]["estimate"], "matches_baseline": match})
        self.save(pd.DataFrame(rows), "baseline")
        if not self.smoke and not match:
            log.warning("M(0) не совпал с baseline_metrics.json бит-в-бит — проверьте модель и строки")
        log.info("M(0) = RMSLE %.4f [%.4f; %.4f] (%s); совпадает с baseline_metrics.json: %s",
                 ev["rmsle"]["estimate"], ev["rmsle"]["low"], ev["rmsle"]["high"], ev["rmsle"]["method"], match)
        return ev

    # ------------------------------------------------------------ таблица шума
    def noise_grid(self) -> pd.DataFrame:
        rows = []
        for nt in self.types:
            for s in self.levels:
                for j in range(self.r):
                    inf = self.info(f"{nt}_s{s:.2f}_j{j}")
                    e = max(inf["eligible"], 1)
                    p = self.load(inf["key"])
                    rows.append({"type": nt, "name": TYPES[nt]["name"], "modality": TYPES[nt]["modality"],
                                 "level": s, "j": j, "seed": inf["seed"], "eligible_unit": ELIGIBLE[nt],
                                 "eligible": inf["eligible"], "selected": inf["selected"], "changed": inf["changed"],
                                 "frac_selected": inf["selected"] / e, "frac_changed": inf["changed"] / e,
                                 "frac_changed_of_words": inf["changed"] / inf["words"] if "words" in inf else np.nan,
                                 "rmsle": float(st.rmsle(self.y, p)), "sec": inf["sec"]})
        runs = self.save(pd.DataFrame(rows), "noise_runs")
        g = runs.groupby(["type", "name", "modality", "level", "eligible_unit"], sort=False).agg(
            r=("j", "size"), eligible=("eligible", "mean"), frac_selected=("frac_selected", "mean"),
            frac_changed=("frac_changed", "mean"), frac_changed_of_words=("frac_changed_of_words", "mean"),
            rmsle_real_min=("rmsle", "min"), rmsle_real_max=("rmsle", "max"),
            rmsle_real_std=("rmsle", "std")).reset_index()
        g.insert(4, "seeds", [", ".join(str(SEED + 1000 * TYPES[t]["t"] + 100 * int(round(s * 10)) + j)
                                        for j in range(self.r)) for t, s in zip(g["type"], g["level"])])
        self.grid = self.save(g, "noise_grid")
        return self.grid

    # ------------------------------------------------------------ кривые и R-AUC
    def curves(self) -> None:
        log.info("парный bootstrap кривых (B = %d)...", self.B)
        t0 = time.perf_counter()
        bs = boot(self.E, self.B, SEED, self.W25, self.APE)
        log.info("  готово за %.0f с", time.perf_counter() - t0)
        rm, w25, md = bs["rmsle"], bs["w25"], bs["mdape"]
        M = np.sqrt(self.E.mean(0))
        W = self.W25.mean(0)
        MD = np.array([100 * np.median(a) for a in self.APE])
        rows = []
        for nt in self.types:
            for s in [0.0] + self.levels:
                c = 0 if s == 0 else self.col_idx[(nt, s)]
                g = self.grid[(self.grid["type"] == nt) & (self.grid["level"] == s)]
                d = rm[:, 0] - rm[:, c]
                dm = md[:, 0] - md[:, c]
                dw = w25[:, c] - w25[:, 0]
                dlo, dhi = pci(d)
                rows.append({
                    "type": nt, "name": TYPES[nt]["name"], "modality": TYPES[nt]["modality"], "level": s,
                    "frac_changed": float(g["frac_changed"].iloc[0]) if len(g) else 0.0,
                    "rmsle": M[c], **dict(zip(("rmsle_low", "rmsle_high"), pci(rm[:, c]))),
                    "delta": M[0] - M[c], "delta_low": dlo, "delta_high": dhi,
                    "significant": bool(s > 0 and (dhi < 0 or dlo > 0)),
                    "practical": bool(M[0] - M[c] <= -PRACTICAL),
                    "r_rel": M[c] / M[0], **dict(zip(("r_rel_low", "r_rel_high"), pci(rm[:, c] / rm[:, 0]))),
                    "mdape": MD[c], **dict(zip(("mdape_low", "mdape_high"), pci(md[:, c]))),
                    "delta_mdape": MD[0] - MD[c], **dict(zip(("delta_mdape_low", "delta_mdape_high"), pci(dm))),
                    "within25": W[c], **dict(zip(("within25_low", "within25_high"), pci(w25[:, c]))),
                    "delta_within25": W[c] - W[0], **dict(zip(("delta_within25_low", "delta_within25_high"),
                                                              pci(dw))),
                    "rmsle_real_min": float(g["rmsle_real_min"].iloc[0]) if len(g) else M[0],
                    "rmsle_real_max": float(g["rmsle_real_max"].iloc[0]) if len(g) else M[0],
                    "rmsle_real_std": float(g["rmsle_real_std"].iloc[0]) if len(g) else 0.0,
                    "r": self.r if s > 0 else 1, "n": len(self.y), "B": self.B, "ci_method": "percentile (парный)"})
        self.curve = self.save(pd.DataFrame(rows), "curves")

        grid = np.array([0.0] + self.levels)
        span = grid[-1] - grid[0]
        rows = []
        for nt in self.types:
            cols = [0] + [self.col_idx[(nt, s)] for s in self.levels]
            ra_b = np.trapezoid(rm[:, cols], grid, axis=1) / span
            ra = float(np.trapezoid(M[cols], grid) / span)
            rel_b = ra_b / rm[:, 0]
            deg_b = ra_b - rm[:, 0]
            rows.append({"type": nt, "name": TYPES[nt]["name"], "modality": TYPES[nt]["modality"],
                         "s_min": grid[0], "s_max": grid[-1], "n_points": len(grid),
                         "rauc": ra, **dict(zip(("rauc_low", "rauc_high"), pci(ra_b))),
                         "rauc_rel": ra / M[0], **dict(zip(("rauc_rel_low", "rauc_rel_high"), pci(rel_b))),
                         "rauc_minus_m0": ra - M[0], **dict(zip(("rauc_minus_m0_low", "rauc_minus_m0_high"),
                                                                pci(deg_b))),
                         "m0": M[0], "B": self.B, "ci_method": "percentile (кривая восстанавливается на каждой "
                                                               "перевыборке)"})
        self.rauc = self.save(pd.DataFrame(rows), "rauc")

    # ------------------------------------------------------------ подгруппы
    def groupings(self) -> list[tuple[str, np.ndarray, list[str] | None]]:
        ctx, task = self.ctx, self.task
        out = [("PERMIT_TYPE", ctx.df["PERMIT_TYPE"].to_numpy(object)[task], None),
               ("Площадь", np.where(ctx.has_area[task], "есть площадь (> 0)", "нет площади"), None)]
        ntok = ctx.n_tokens[task]
        q = np.quantile(ntok, [0.25, 0.5, 0.75])
        b = np.digitize(ntok, q, right=True)
        edges = [0] + [int(x) for x in q]
        labels = [f"Q1: ≤ {edges[1]} слов", f"Q2: {edges[1] + 1}–{edges[2]}", f"Q3: {edges[2] + 1}–{edges[3]}",
                  f"Q4: > {edges[3]}"]
        out.append(("Длина описания (квартили)", np.array([labels[i] for i in b], object), ["X1", "X2"]))
        wl = self.ctx.prep.word_len
        out.append(("Обрезка текста", np.where(ntok > wl, f"обрезан (> {wl} слов)", f"≤ {wl} слов"), ["X1", "X2"]))
        return out

    def by_group(self) -> pd.DataFrame:
        lv = [s for s in SUBGROUP_LEVELS if s in self.levels]
        cols = [0] + [self.col_idx[(nt, s)] for nt in self.types for s in lv]
        rows = []
        for gname, lab, only in self.groupings():
            for g in sorted(pd.unique(lab)):
                m = lab == g
                E = self.E[m][:, cols]
                rm = boot(E, self.B, SEED)["rmsle"]
                M = np.sqrt(E.mean(0))
                for i, c in enumerate(cols[1:], start=1):
                    nt, s = self.cols[c]
                    if only and nt not in only:
                        continue
                    d = rm[:, 0] - rm[:, i]
                    lo, hi = pci(d)
                    rows.append({"type": nt, "name": TYPES[nt]["name"], "level": s, "grouping": gname, "group": g,
                                 "n": int(m.sum()), "rmsle0": M[0],
                                 **dict(zip(("rmsle0_low", "rmsle0_high"), pci(rm[:, 0]))),
                                 "rmsle": M[i], **dict(zip(("rmsle_low", "rmsle_high"), pci(rm[:, i]))),
                                 "delta": M[0] - M[i], "delta_low": lo, "delta_high": hi,
                                 "significant": bool(hi < 0 or lo > 0), "r_rel": M[i] / M[0],
                                 "ci_method": "percentile (парный, внутри группы)", "B": self.B})
        self.group = self.save(pd.DataFrame(rows), "by_group")
        return self.group

    # ------------------------------------------------------------ профиль по признакам
    def by_feature(self) -> pd.DataFrame | None:
        tab = [nt for nt in self.types if TYPES[nt]["kind"] == "tab"]
        if not self.features or not tab:
            return None
        keys, cols = [], [self.E[:, 0]]
        for nt in tab:
            for g, (gname, _) in enumerate(FEATURE_GROUPS, start=1):
                base = f"{nt}_g{g:02d}_s{FEATURE_LEVEL:.2f}"
                if not (self.run_dir / f"{base}_j0.npy").exists():
                    continue
                P = np.stack([self.load(f"{base}_j{j}") for j in range(self.r)], 1)
                cols.append(((P - self.y[:, None]) ** 2).mean(1))
                infos = [self.info(f"{base}_j{j}") for j in range(self.r)]
                keys.append((nt, gname, infos))
            if (nt, FEATURE_LEVEL) in self.col_idx:  # все признаки сразу — для сравнения
                cols.append(self.E[:, self.col_idx[(nt, FEATURE_LEVEL)]])
                infos = [self.info(f"{nt}_s{FEATURE_LEVEL:.2f}_j{j}") for j in range(self.r)]
                keys.append((nt, "Все признаки типа", infos))
        E = np.stack(cols, 1)
        log.info("bootstrap профиля по признакам (%d групп)...", len(keys))
        rm = boot(E, self.B, SEED)["rmsle"]
        M = np.sqrt(E.mean(0))
        rows = []
        for i, (nt, gname, infos) in enumerate(keys, start=1):
            d = rm[:, 0] - rm[:, i]
            lo, hi = pci(d)
            el = sum(x["eligible"] for x in infos)
            rows.append({"type": nt, "name": TYPES[nt]["name"], "level": FEATURE_LEVEL, "group": gname,
                         "columns": infos[0]["units"] if gname != "Все признаки типа" else "все",
                         "frac_changed": sum(x["changed"] for x in infos) / max(el, 1),
                         "rmsle": M[i], **dict(zip(("rmsle_low", "rmsle_high"), pci(rm[:, i]))),
                         "delta": M[0] - M[i], "delta_low": lo, "delta_high": hi,
                         "significant": bool(hi < 0 or lo > 0), "r_rel": M[i] / M[0], "r": self.r, "B": self.B,
                         "ci_method": "percentile (парный)"})
        df = pd.DataFrame(rows)
        df["rank"] = df[df["group"] != "Все признаки типа"].groupby("type")["delta"].rank(method="min")
        self.feat = self.save(df.sort_values(["type", "delta"]).reset_index(drop=True), "by_feature")
        return self.feat

    # ------------------------------------------------------------ графики
    def plot_curves(self) -> None:
        c, m0 = self.curve, float(self.curve["rmsle"].iloc[0])
        for name, ycol, ylab, ref in (("curves_rmsle", "rmsle", "RMSLE (меньше — лучше)", m0),
                                      ("curves_rel", "r_rel", "R_rel = M(s) / M(0) (> 1 — хуже)", 1.0)):
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
            for ax, mod in zip(axes, ("таблица", "текст")):
                for nt in [t for t in self.types if TYPES[t]["modality"] == mod]:
                    d = c[c["type"] == nt]
                    ax.fill_between(d["level"], d[f"{ycol}_low"], d[f"{ycol}_high"], color=COLORS[nt], alpha=0.2,
                                    lw=0)
                    ax.plot(d["level"], d[ycol], color=COLORS[nt], marker=MARKERS[nt], ms=6, lw=2,
                            label=f"{nt}: {TYPES[nt]['name']}")
                    last = d.iloc[-1]
                    ax.annotate(nt, (last["level"], last[ycol]), xytext=(6, 0), textcoords="offset points",
                                va="center", fontsize=9, color="#333333")
                    if "natural" in TYPES[nt]:
                        ax.axvline(TYPES[nt]["natural"], color=COLORS[nt], ls=":", lw=1.2)
                        ax.text(TYPES[nt]["natural"], 0.02 if nt == "T1" else 0.10, " " + TYPES[nt]["natural_label"],
                                transform=ax.get_xaxis_transform(), fontsize=7.5, color="#555555")
                ax.axhline(ref, color="#888888", lw=1, ls="--", label="чистый test" if ycol == "rmsle" else "R_rel = 1")
                ax.set(title=f"Модальность: {mod}", xlabel="интенсивность s (доля затронутых значений)",
                       xlim=(-0.01, max(self.levels) + 0.06))
                ax.grid(alpha=0.25, lw=0.6)
                ax.legend(fontsize=8, loc="upper left")
            axes[0].set_ylabel(ylab)
            n_txt = f"{len(self.y):,}".replace(",", " ")
            fig.suptitle(f"Robustness curves baseline на test без R2 (n = {n_txt}); "
                         f"коридор — 95% CI (парный bootstrap, B = {self.B}, r = {self.r})", fontsize=10)
            self.savefig(fig, name)

    def plot_groups(self) -> None:
        if self.group.empty:
            return
        s = max(self.group["level"])
        d0 = self.group[self.group["level"] == s]
        ncol = min(2, len(self.types))
        nrow = int(np.ceil(len(self.types) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(7 * ncol, 4.6 * nrow), squeeze=False)
        axes = axes.ravel()
        for ax in axes[len(self.types):]:
            ax.set_visible(False)
        for ax, nt in zip(axes, self.types):
            d = d0[d0["type"] == nt].reset_index(drop=True)
            ylab = [f"{g} | {v}" for g, v in zip(d["grouping"], d["group"])]
            yy = np.arange(len(d))[::-1]
            ax.barh(yy, d["delta"], color=COLORS[nt], height=0.6)
            ax.errorbar(d["delta"], yy, xerr=[d["delta"] - d["delta_low"], d["delta_high"] - d["delta"]],
                        fmt="none", ecolor="#333333", lw=1, capsize=2)
            ax.axvline(0, color="#888888", lw=1)
            ax.set_yticks(yy, ylab, fontsize=7)
            ax.set(title=f"{nt}: {TYPES[nt]['name']}, s = {s:g}", xlabel="ΔM = M(0) − M(s) (< 0 — хуже)")
            ax.grid(axis="x", alpha=0.25, lw=0.6)
        self.savefig(fig, "delta_by_group")

    def plot_features(self) -> None:
        if getattr(self, "feat", None) is None:
            return
        tab = list(pd.unique(self.feat["type"]))
        fig, axes = plt.subplots(1, len(tab), figsize=(6.5 * len(tab), 4.8))
        axes = np.atleast_1d(axes)
        for ax, nt in zip(axes, tab):
            d = self.feat[self.feat["type"] == nt].sort_values("delta", ascending=False).reset_index(drop=True)
            yy = np.arange(len(d))
            colors = ["#bbbbbb" if g == "Все признаки типа" else COLORS[nt] for g in d["group"]]
            ax.barh(yy, d["delta"], color=colors, height=0.6)
            ax.errorbar(d["delta"], yy, xerr=[d["delta"] - d["delta_low"], d["delta_high"] - d["delta"]],
                        fmt="none", ecolor="#333333", lw=1, capsize=2)
            ax.axvline(0, color="#888888", lw=1)
            ax.set_yticks(yy, d["group"], fontsize=8)
            ax.set(title=f"{nt}: {TYPES[nt]['name']}, s = {FEATURE_LEVEL:g}, портится одна группа",
                   xlabel="ΔM = M(0) − M(s) (< 0 — хуже)")
            ax.grid(axis="x", alpha=0.25, lw=0.6)
        self.savefig(fig, "by_feature")

    # ------------------------------------------------------------ итог
    def summary(self, ev: dict) -> None:
        c, ra = self.curve, self.rauc.set_index("type")
        rows = [{"Тип": "baseline", "Модальность": "—", "Искажение": "нет (чистый test)",
                 "R-AUC (RMSLE)": f"{ev['rmsle']['estimate']:.4f}",
                 "95% CI": f"[{ev['rmsle']['low']:.4f}; {ev['rmsle']['high']:.4f}] ({ev['rmsle']['method']})",
                 "R-AUC_rel": "1", "95% CI rel": "—", "M(s_max)": "—", "ΔM(s_max) [95% CI]": "—",
                 "Значимо с s": "—", "ΔM ≤ −0,02 с s": "—", "Вывод": "M(0) — RMSLE на чистом test без R2"}]
        order = ra.sort_values("rauc_rel", ascending=False).index.tolist()
        for nt in self.types:
            d = c[(c["type"] == nt) & (c["level"] > 0)]
            last = d.iloc[-1]
            sig = d.loc[d["significant"] & (d["delta"] < 0), "level"]
            pr = d.loc[d["practical"], "level"]
            first_sig = f"{sig.min():g}" if len(sig) else "нет"
            first_pr = f"{pr.min():g}" if len(pr) else "нет"
            r = ra.loc[nt]
            rank = order.index(nt) + 1
            if len(pr):
                v = f"практически значимое ухудшение (ΔM ≤ −{PRACTICAL:g}) с s = {first_pr}"
            elif len(sig):
                v = (f"ухудшение статистически значимо с s = {first_sig}, но |ΔM| < {PRACTICAL:g} на всём диапазоне"
                     + (f" (порядка разброса между seed ≈ {SEED_SPREAD:g})" if abs(last["delta"]) <= 2 * SEED_SPREAD
                        else ""))
            else:
                v = "значимого ухудшения нет"
            v = f"место {rank} из {len(self.types)} по R-AUC_rel; {v}"
            rows.append({"Тип": nt, "Модальность": TYPES[nt]["modality"], "Искажение": TYPES[nt]["name"],
                         "R-AUC (RMSLE)": f"{r['rauc']:.4f}", "95% CI": f"[{r['rauc_low']:.4f}; {r['rauc_high']:.4f}]",
                         "R-AUC_rel": f"{r['rauc_rel']:.4f}",
                         "95% CI rel": f"[{r['rauc_rel_low']:.4f}; {r['rauc_rel_high']:.4f}]",
                         "M(s_max)": f"{last['rmsle']:.4f} (s = {last['level']:g})",
                         "ΔM(s_max) [95% CI]": f"{last['delta']:+.4f} [{last['delta_low']:+.4f}; "
                                               f"{last['delta_high']:+.4f}]",
                         "Значимо с s": first_sig, "ΔM ≤ −0,02 с s": first_pr, "Вывод": v})
        df = self.save(pd.DataFrame(rows), "summary")
        lines = [f"# ПР2: итог{' (SMOKE — для выводов не годится)' if self.smoke else ''}", "",
                 f"Test без R2, n = {len(self.y)}; M = RMSLE (меньше — лучше); ΔM = M(0) − M(s), отрицательное — "
                 f"ухудшение; R-AUC — средняя RMSLE на s ∈ [0; {max(self.levels):g}] (трапеции, s₀ = 0), "
                 f"меньше — лучше; r = {self.r} реализаций; CI — парный bootstrap по объектам, B = {self.B}, "
                 "percentile (M(0) — как у baseline).", "",
                 "| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
        lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
        (self.out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ------------------------------------------------------------ сдвиг прогноза
    def bias(self) -> pd.DataFrame:
        """Средний остаток (прогноз − факт, log1p; > 0 — завышение) по уровням шума: в целом, по квартилям цели
        и для обрезанных описаний. Объясняет расхождение RMSLE и MdAPE: шум, сдвигающий прогноз вниз, частично
        компенсирует завышение baseline на мелких разрешениях."""
        y = self.y
        res = np.stack([(p - y[:, None]).mean(1) for p in self.P], 1)  # (n, K): средний остаток по реализациям
        q = np.quantile(y, [0.25, 0.5, 0.75])
        yq = np.digitize(y, q, right=True)
        money = np.round(np.expm1(q)).astype(int)
        qlab = [f"Q1: ≤ ${money[0]}", f"Q2: ${money[0]}–{money[1]}", f"Q3: ${money[1]}–{money[2]}",
                f"Q4: > ${money[2]}"]
        ntok = self.ctx.n_tokens[self.task]
        wl = self.ctx.prep.word_len
        groups = [("Все", "все", np.ones(len(y), bool), None)]
        groups += [("Квартиль VALUATION", qlab[k], yq == k, None) for k in range(4)]
        groups += [("Обрезка текста", f"обрезан (> {wl} слов)", ntok > wl, ["X1", "X2"])]
        rows = []
        for gname, glab, m, only in groups:
            Rg, Eg = res[m], self.E[m]
            ng = len(Rg)
            bm = np.concatenate([W.astype(np.float64) @ Rg / ng for W in boot_weights(ng, self.B, SEED)])
            M = Rg.mean(0)
            rm = np.sqrt(Eg.mean(0))
            for c, (nt, s) in enumerate(self.cols):
                for t in ([nt] if nt != "clean" else self.types):
                    if only and t not in only:
                        continue
                    lo, hi = pci(bm[:, c])
                    dlo, dhi = pci(bm[:, c] - bm[:, 0])
                    rows.append({"type": t, "level": s, "grouping": gname, "group": glab, "n": ng,
                                 "bias": M[c], "bias_low": lo, "bias_high": hi,
                                 "delta_bias": M[c] - M[0], "delta_bias_low": dlo, "delta_bias_high": dhi,
                                 "rmsle": rm[c], "rmsle0": rm[0], "delta": rm[0] - rm[c], "B": self.B,
                                 "ci_method": "percentile (парный, внутри группы)"})
        df = pd.DataFrame(rows).sort_values(["type", "grouping", "group", "level"], kind="stable")
        self.bias_df = self.save(df.reset_index(drop=True), "bias")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
        d = self.bias_df[self.bias_df["grouping"] == "Все"]
        for nt in self.types:
            x = d[d["type"] == nt]
            axes[0].fill_between(x["level"], x["bias_low"], x["bias_high"], color=COLORS[nt], alpha=0.2, lw=0)
            axes[0].plot(x["level"], x["bias"], color=COLORS[nt], marker=MARKERS[nt], ms=6, lw=2,
                         label=f"{nt}: {TYPES[nt]['name']}")
        axes[0].axhline(0, color="#888888", lw=1)
        axes[0].set(title="Средний остаток прогноза (> 0 — завышение)", xlabel="интенсивность s",
                    ylabel="mean(прогноз − факт), log1p")
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.25, lw=0.6)
        s_max = max(self.levels)
        x = self.bias_df[(self.bias_df["grouping"] == "Квартиль VALUATION") & (self.bias_df["level"] == s_max)]
        w = 0.8 / len(self.types)
        for i, nt in enumerate(self.types):
            xx = x[x["type"] == nt]
            pos = np.arange(len(xx)) + (i - (len(self.types) - 1) / 2) * w
            axes[1].bar(pos, xx["delta"], width=w * 0.9, color=COLORS[nt], label=nt)
        axes[1].set_xticks(np.arange(4), qlab, fontsize=8)
        axes[1].axhline(0, color="#888888", lw=1)
        axes[1].set(title=f"ΔM по квартилям VALUATION, s = {s_max:g}", ylabel="ΔM = M(0) − M(s) (< 0 — хуже)")
        axes[1].legend(fontsize=8)
        axes[1].grid(axis="y", alpha=0.25, lw=0.6)
        self.savefig(fig, "bias_curves")
        return self.bias_df

    def run(self) -> None:
        self.collect()
        ev = self.baseline()
        self.noise_grid()
        self.curves()
        self.by_group()
        self.by_feature()
        self.bias()
        self.plot_curves()
        self.plot_groups()
        self.plot_features()
        self.summary(ev)
        log.info("отчёт: %s", self.out)
        log.info("\n%s", (self.out / "summary.md").read_text(encoding="utf-8"))


# ====================================================================== main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="5 000 строк test, s ∈ {0,2; 0,4}, r = 2, B = 200")
    ap.add_argument("--smoke-rows", type=int, default=5000)
    ap.add_argument("--only", default=None, help="типы шума через запятую (T1,T2,X1,X2)")
    ap.add_argument("--report", action="store_true", help="только отчёт по готовому кешу")
    ap.add_argument("--force", action="store_true", help="удалить кеш прогонов и посчитать заново")
    ap.add_argument("--no-features", action="store_true", help="без профиля по признакам")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    types = [t.strip().upper() for t in args.only.split(",")] if args.only else list(TYPES)
    bad = [t for t in types if t not in TYPES]
    if bad:
        raise SystemExit(f"неизвестные типы: {bad}; есть {list(TYPES)}")
    levels, r, B = ((0.2, 0.4), 2, 200) if args.smoke else (LEVELS, R_REAL, BOOTSTRAP_B)
    run_dir = ARTIFACTS_DIR / "pr2" / ("smoke" if args.smoke else "runs")
    out_dir = REPORTS_DIR / "pr2" / "smoke" if args.smoke else REPORTS_DIR / "pr2"
    get_logger(ARTIFACTS_DIR / "logs" / ("pr2_noise_smoke.log" if args.smoke else "pr2_noise.log"))
    t0 = time.perf_counter()
    ctx = Ctx(args.smoke, args.smoke_rows, args.device)
    log.info("ПР2: test %d строк (без R2: %d), устройство %s; типы %s, уровни %s, r = %d, B = %d",
             len(ctx.idx), int(ctx.task.sum()), device_name(ctx.device), types, levels, r, B)
    log.info("T1 портит: %s; T2 — категории: %s", [("+".join(u)) for u in ctx.units(ntype="T1")], ctx.cat_cols)
    check_cache(run_dir, cache_meta(ctx, args.smoke), args.force and not args.report)
    runs = plan_runs(ctx, types, levels, r, not args.no_features)
    if not args.report:
        pred0 = self_checks(ctx, args.smoke)
        if not (run_dir / "clean.npy").exists():
            save_run(run_dir, Run("clean", "clean", 0.0, 0, SEED), pred0,
                     {"eligible": 0, "selected": 0, "changed": 0}, 0.0)
        run_all(ctx, runs, run_dir)
    Report(ctx, run_dir, out_dir, types, levels, r, B, not args.no_features, args.smoke).run()
    log.info("всего %.1f мин", (time.perf_counter() - t0) / 60)


if __name__ == "__main__":
    main()
