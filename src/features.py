"""Препроцессор признаков для FT-Transformer. Обучается только на train.

Исключено из признаков:
* ID и технические поля: PERMIT_NBR, PIN_NBR, APN (ключ участка — используется в ПР1/ПР4,
  но не как признак), USE_CODE (дублирует USE_DESC), PERMIT_GROUP (константа), REFRESH_TIME,
  GEOLOCATION (дублирует LAT/LON);
* адрес объекта: PRIMARY_ADDRESS (квазиидентификатор);
* признаки из будущего относительно момента выдачи: STATUS_DESC, STATUS_DATE, COFO_DATE;
* столбцы с долей пропусков в train > 99% и категориальные столбцы, у которых в train
  остаётся меньше двух значений с частотой ≥ MIN_FREQ (EV, Solar — фактически константы).

Выход `transform`: словарь массивов
    num       (N, n_num) float32 — стандартизованные числа, пропуск → 0
    num_mask  (N, n_num) bool    — индикатор пропуска (в модели — отдельный эмбеддинг)
    cat       (N, n_cat) int64   — коды категорий: 0 = MISSING, 1 = OTHER (редкая/новая)
    txt       (N, 64)    float32 — WORK_DESC: TF-IDF (слова + симв. n-граммы) → SVD(64)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from config import ARTIFACTS_DIR, SEED

PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.joblib"

EXCLUDED = {
    "id": ["PERMIT_NBR", "PIN_NBR", "APN", "USE_CODE", "PERMIT_GROUP", "REFRESH_TIME", "GEOLOCATION"],
    "address": ["PRIMARY_ADDRESS"],
    "future": ["STATUS_DESC", "STATUS_DATE", "COFO_DATE"],
    "target": ["VALUATION"],
}

# Числовые признаки: имя → преобразование ("log" = log1p, "slog" = sign·log1p|x|, "none")
NUM_SPEC = {
    "square_footage": "log",
    "height": "log",
    "du_changed": "slog",   # изменение числа квартир может быть отрицательным (снос/объединение)
    "adu_changed": "slog",
    "junior_adu": "slog",
    "lat": "none",
    "lon": "none",
    "issue_year": "none",
    "issue_month": "none",
    "issue_dow": "none",
    "issue_doy": "none",
    "days_to_issue": "log",  # ISSUE_DATE − SUBMITTED_DATE
    "desc_len": "log",
}
# Категориальные признаки (исходные столбцы + производный zone_base).
CAT_COLS = ["PERMIT_TYPE", "PERMIT_SUB_TYPE", "USE_DESC", "CONSTRUCTION", "Business Unit", "ZONE", "zone_base",
            "ZIP_CODE", "CT", "CD", "APC", "CPA", "CNC", "HL", "TYPE_LAT_LON", "EV", "Solar"]
TEXT_COL = "work_desc"

MISSING_MAX = 0.99  # столбцы с большей долей пропусков в train исключаются
MIN_FREQ = 10        # категории реже этого (в train) → OTHER
CLIP_Q = (0.001, 0.999)
TXT_DIM = 64


def numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Сырые числовые признаки (до log/клиппинга/стандартизации)."""
    issue = df["issue_date"]
    out = pd.DataFrame(index=df.index)
    for c in ("square_footage", "height", "du_changed", "adu_changed", "junior_adu", "lat", "lon"):
        out[c] = df[c].astype(float)
    # Отрицательные площадь/высота и нулевые координаты невалидны → пропуск (учтено в ПР1).
    for c in ("square_footage", "height"):
        out.loc[out[c] < 0, c] = np.nan
    for c in ("lat", "lon"):
        out.loc[out[c] == 0, c] = np.nan
    out["issue_year"] = issue.dt.year.astype(float)
    out["issue_month"] = issue.dt.month.astype(float)
    out["issue_dow"] = issue.dt.dayofweek.astype(float)
    out["issue_doy"] = issue.dt.dayofyear.astype(float)
    days = (issue - df["submitted_date"]).dt.days.astype(float)
    out["days_to_issue"] = days.where(days >= 0)
    out["desc_len"] = df[TEXT_COL].str.len().astype(float)
    return out


def categorical_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in CAT_COLS:
        if c == "zone_base":
            s = df["ZONE"].astype("string").str.split(",").str[0].str.split("-").str[0]
        elif c in ("CT", "ZONE"):
            s = df[c].astype("string").str.split(",").str[0]  # несколько значений через запятую → первое
        else:
            s = df[c].astype("string")
        out[c] = s.str.strip().str.upper()
    return out


def _transform_num(x: np.ndarray, kind: str) -> np.ndarray:
    if kind == "log":
        return np.log1p(x)
    if kind == "slog":
        return np.sign(x) * np.log1p(np.abs(x))
    return x


def normalize_text(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()


@dataclass
class Preprocessor:
    num_names: list[str] = field(default_factory=list)
    num_kind: dict[str, str] = field(default_factory=dict)
    num_clip: dict[str, tuple[float, float]] = field(default_factory=dict)
    num_mean: dict[str, float] = field(default_factory=dict)
    num_std: dict[str, float] = field(default_factory=dict)
    cat_names: list[str] = field(default_factory=list)
    cat_vocab: dict[str, dict[str, int]] = field(default_factory=dict)
    dropped: dict[str, list[str]] = field(default_factory=dict)
    tfidf_word: TfidfVectorizer | None = None
    tfidf_char: TfidfVectorizer | None = None
    svd: TruncatedSVD | None = None
    txt_mean: np.ndarray | None = None
    txt_std: np.ndarray | None = None
    y_mean: float = 0.0
    y_std: float = 1.0

    # ---------------------------------------------------------------- fit
    def fit(self, train: pd.DataFrame, y_log: np.ndarray | None = None) -> "Preprocessor":
        self.dropped = {k: list(v) for k, v in EXCLUDED.items()}
        self.dropped["missing>99%"] = []
        self.dropped["constant"] = []

        num = numeric_frame(train)
        for c in num.columns:
            if num[c].isna().mean() > MISSING_MAX:
                self.dropped["missing>99%"].append(c)
                continue
            kind = NUM_SPEC[c]
            x = _transform_num(num[c].to_numpy(), kind)
            lo, hi = np.nanquantile(x, CLIP_Q)
            x = np.clip(x, lo, hi)
            self.num_names.append(c)
            self.num_kind[c] = kind
            self.num_clip[c] = (float(lo), float(hi))
            self.num_mean[c] = float(np.nanmean(x))
            self.num_std[c] = float(np.nanstd(x)) or 1.0

        cat = categorical_frame(train)
        for c in cat.columns:
            if cat[c].isna().mean() > MISSING_MAX:
                self.dropped["missing>99%"].append(c)
                continue
            vc = cat[c].value_counts()
            keep = vc[vc >= MIN_FREQ].index
            if len(keep) < 2 and cat[c].isna().mean() < 0.01:
                self.dropped["constant"].append(c)
                continue
            self.cat_names.append(c)
            self.cat_vocab[c] = {v: i + 2 for i, v in enumerate(sorted(keep))}

        texts = normalize_text(train[TEXT_COL])
        self.tfidf_word = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=30_000,
                                          sublinear_tf=True, dtype=np.float32)
        self.tfidf_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5,
                                          max_features=30_000, sublinear_tf=True, dtype=np.float32)
        m = sp.hstack([self.tfidf_word.fit_transform(texts), self.tfidf_char.fit_transform(texts)]).tocsr()
        self.svd = TruncatedSVD(n_components=TXT_DIM, algorithm="randomized", n_iter=5, random_state=SEED)
        z = self.svd.fit_transform(m)
        self.txt_mean, self.txt_std = z.mean(0), z.std(0) + 1e-8

        if y_log is not None:
            self.y_mean, self.y_std = float(np.mean(y_log)), float(np.std(y_log))
        return self

    # ---------------------------------------------------------------- transform
    def transform(self, df: pd.DataFrame, chunk: int = 100_000) -> dict[str, np.ndarray]:
        num = numeric_frame(df)
        xs, masks = [], []
        for c in self.num_names:
            x = _transform_num(num[c].to_numpy(dtype=float), self.num_kind[c])
            mask = np.isnan(x)
            x = (np.clip(x, *self.num_clip[c]) - self.num_mean[c]) / self.num_std[c]
            xs.append(np.where(mask, 0.0, x))
            masks.append(mask)
        cat = categorical_frame(df)
        codes = []
        for c in self.cat_names:
            v = cat[c]
            code = v.map(self.cat_vocab[c]).fillna(1).astype(np.int64).to_numpy()
            code[v.isna().to_numpy()] = 0
            codes.append(code)
        texts = normalize_text(df[TEXT_COL])
        z = []
        for i in range(0, len(texts), chunk):  # по частям, чтобы не держать всю TF-IDF-матрицу
            t = texts.iloc[i:i + chunk]
            m = sp.hstack([self.tfidf_word.transform(t), self.tfidf_char.transform(t)]).tocsr()
            z.append(self.svd.transform(m))
        z = (np.concatenate(z) - self.txt_mean) / self.txt_std
        return {
            "num": np.stack(xs, 1).astype(np.float32),
            "num_mask": np.stack(masks, 1),
            "cat": np.stack(codes, 1),
            "txt": z.astype(np.float32),
        }

    # ---------------------------------------------------------------- прочее
    @property
    def cat_cardinalities(self) -> list[int]:
        return [len(self.cat_vocab[c]) + 2 for c in self.cat_names]

    def summary(self) -> dict:
        return {
            "num_features": self.num_names,
            "num_transform": self.num_kind,
            "cat_features": self.cat_names,
            "cat_cardinalities": self.cat_cardinalities,
            "text_feature": f"WORK_DESC → TF-IDF(word 1-2, char_wb 3-5) → SVD({TXT_DIM})",
            "text_svd_explained_variance": float(self.svd.explained_variance_ratio_.sum()),
            "min_freq_other": MIN_FREQ,
            "clip_quantiles": list(CLIP_Q),
            "dropped": self.dropped,
            "y_mean": self.y_mean,
            "y_std": self.y_std,
        }

    def save(self, path=PREPROCESSOR_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path=PREPROCESSOR_PATH) -> "Preprocessor":
        return joblib.load(path)


# ---------------------------------------------------------------- слова как токены (эксперимент text_tokens)
WORD_RE = r"[a-z]+|\d+(?:\.\d+)?"  # слова и числа нормализованного текста


def word_tokenize(s: pd.Series) -> pd.Series:
    """Списки слов и чисел WORK_DESC (после normalize_text)."""
    return normalize_text(s).str.findall(WORD_RE)


@dataclass
class WordVocab:
    """Словарь слов по train: документная частота ≥ min_df, не больше max_size; PAD = 0, UNK = 1."""
    min_df: int = 5
    max_size: int = 20_000
    index: dict[str, int] = field(default_factory=dict)

    PAD, UNK = 0, 1

    def fit(self, tokens: pd.Series) -> "WordVocab":
        from collections import Counter
        df = Counter()
        for ws in tokens:
            df.update(set(ws))
        words = sorted((w for w, c in df.items() if c >= self.min_df), key=lambda w: (-df[w], w))[:self.max_size]
        self.index = {w: i + 2 for i, w in enumerate(words)}
        return self

    @property
    def size(self) -> int:
        return len(self.index) + 2

    def transform(self, tokens: pd.Series, max_len: int) -> np.ndarray:
        """(N, max_len) int32: коды первых max_len слов, хвост — PAD."""
        out = np.zeros((len(tokens), max_len), dtype=np.int32)
        get = self.index.get
        for i, ws in enumerate(tokens):
            ids = [get(w, self.UNK) for w in ws[:max_len]]
            out[i, :len(ids)] = ids
        return out
