"""Препроцессор признаков для FT-Transformer. Обучается только на train.

Исключено из признаков:
* ID: PCIS Permit #, Reference # (Old Permit #), License #; ключ участка
  (Assessor Book/Page/Parcel, Tract/Block/Lot) — идентификаторы объекта;
* PII: имена руководителей и заявителей, адреса заявителей и подрядчика,
  адрес объекта (номер дома, улица, unit);
* признаки из будущего относительно момента выдачи: Status, Status Date (CofO-статусы);
* столбцы с долей пропусков в train > 99% (Event Code, Occupancy, Project Number и т. п.).

Выход `transform`: словарь массивов
    num       (N, n_num) float32 — стандартизованные числа, пропуск → 0
    num_mask  (N, n_num) bool    — индикатор пропуска (в модели — отдельный эмбеддинг)
    cat       (N, n_cat) int64   — коды категорий: 0 = MISSING, 1 = OTHER (редкая/новая)
    txt       (N, 64)    float32 — Work Description: TF-IDF (слова + симв. n-граммы) → SVD(64)
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
    "id": ["PCIS Permit #", "Reference # (Old Permit #)", "License #", "Assessor Book", "Assessor Page",
           "Assessor Parcel", "Tract", "Block", "Lot"],
    "pii": ["Principal First Name", "Principal Middle Name", "Principal Last Name", "Applicant First Name",
            "Applicant Last Name", "Applicant Business Name", "Applicant Address 1", "Applicant Address 2",
            "Applicant Address 3", "Contractor Address", "Address Start", "Address Fraction Start",
            "Address End", "Address Fraction End", "Street Direction", "Street Name", "Street Suffix",
            "Suffix Direction", "Unit Range Start", "Unit Range End"],
    "future": ["Status", "Status Date"],
    "target": ["Valuation"],
}

# Числовые признаки: имя → (источник, log1p?)
NUM_SPEC = {
    "floor_area_zoning": True,
    "floor_area_building": True,
    "units": True,
    "stories": True,
    "lat": False,
    "lon": False,
    "issue_year": False,
    "issue_month": False,
    "issue_dow": False,
    "issue_doy": False,
    "license_days_left": False,
    "desc_len": False,
}
# Категориальные признаки (исходные столбцы + производный zone_base).
CAT_COLS = ["Permit Type", "Permit Sub-Type", "Permit Category", "Initiating Office", "Zone", "zone_base",
            "Zip Code", "Census Tract", "License Type", "Contractor State", "Contractor City",
            "Contractor's Business Name", "Occupancy", "Project Number", "Event Code"]
TEXT_COL = "work_desc"

MISSING_MAX = 0.99  # столбцы с большей долей пропусков в train исключаются
MIN_FREQ = 10        # категории реже этого (в train) → OTHER
CLIP_Q = (0.001, 0.999)
TXT_DIM = 64


def numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Сырые числовые признаки (до log/клиппинга/стандартизации)."""
    issue = df["issue_date"]
    out = pd.DataFrame(index=df.index)
    for c in ("floor_area_zoning", "floor_area_building", "units", "stories", "lat", "lon"):
        out[c] = df[c].astype(float)
    # Отрицательные площади/этажи/квартиры невалидны → пропуск (учтено в ПР1 как нарушение валидности).
    for c in ("floor_area_zoning", "floor_area_building", "units", "stories"):
        out.loc[out[c] < 0, c] = np.nan
    out["issue_year"] = issue.dt.year.astype(float)
    out["issue_month"] = issue.dt.month.astype(float)
    out["issue_dow"] = issue.dt.dayofweek.astype(float)
    out["issue_doy"] = issue.dt.dayofyear.astype(float)
    out["license_days_left"] = (df["license_exp_date"] - issue).dt.days.astype(float)
    out["desc_len"] = df[TEXT_COL].str.len().astype(float)
    return out


def categorical_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in CAT_COLS:
        if c == "zone_base":
            out[c] = df["Zone"].astype("string").str.split("-").str[0].str.strip()
        else:
            out[c] = df[c].astype("string").str.strip().str.upper()
    return out


def normalize_text(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()


@dataclass
class Preprocessor:
    num_names: list[str] = field(default_factory=list)
    num_log: dict[str, bool] = field(default_factory=dict)
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

        num = numeric_frame(train)
        for c in num.columns:
            if num[c].isna().mean() > MISSING_MAX:
                self.dropped["missing>99%"].append(c)
                continue
            x = num[c].to_numpy()
            log = NUM_SPEC[c]
            if log:
                x = np.log1p(x)
            lo, hi = np.nanquantile(x, CLIP_Q)
            x = np.clip(x, lo, hi)
            self.num_names.append(c)
            self.num_log[c] = log
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
    def transform(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        num = numeric_frame(df)
        xs, masks = [], []
        for c in self.num_names:
            x = num[c].to_numpy(dtype=float)
            if self.num_log[c]:
                x = np.log1p(x)
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
        m = sp.hstack([self.tfidf_word.transform(texts), self.tfidf_char.transform(texts)]).tocsr()
        z = (self.svd.transform(m) - self.txt_mean) / self.txt_std
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
            "num_log1p": [c for c in self.num_names if self.num_log[c]],
            "cat_features": self.cat_names,
            "cat_cardinalities": self.cat_cardinalities,
            "text_feature": f"Work Description → TF-IDF(word 1-2, char_wb 3-5) → SVD({TXT_DIM})",
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
