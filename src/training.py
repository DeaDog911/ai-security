"""Обучение и инференс FT-Transformer: общие функции для baseline и OOF (ПР1)."""
from __future__ import annotations

import copy
import logging
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from config import SEED
from model import FTTransformer

TRAIN_HP = {
    "batch_size": 512,
    "lr": 3e-4,
    "weight_decay": 1e-5,
    "warmup_steps": 500,
    "max_epochs": 60,
    "patience": 6,            # ранняя остановка по val RMSLE
    "lr_plateau_patience": 2,  # ReduceLROnPlateau(factor=0.5)
    "lr_plateau_factor": 0.5,
    "grad_clip": 1.0,
    "amp": True,               # float16 autocast на GPU
    "loss": "mse",             # mse | huber
    "huber_delta": 1.0,        # порог Huber в единицах log1p(y) (переводится в стандартизованную шкалу)
}

KEYS = ("num", "num_mask", "cat", "txt")
OPTIONAL_KEYS = ("words",)  # слова WORK_DESC как токены (эксперимент text_tokens)


def _keys(X: dict) -> list[str]:
    return list(KEYS) + [k for k in OPTIONAL_KEYS if k in X]


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return f"cuda:{torch.cuda.get_device_name(0)}"
    return "cpu"


def subset(X: dict[str, np.ndarray], idx) -> dict[str, np.ndarray]:
    return {k: X[k][idx] for k in _keys(X)}


def to_device(X: dict[str, np.ndarray], device) -> dict[str, torch.Tensor]:
    return {k: torch.as_tensor(X[k]).to(device) for k in _keys(X)}


def build_model(X: dict[str, np.ndarray], cat_cardinalities: list[int], hp: dict, **model_kw) -> FTTransformer:
    """model_kw — опции FTTransformer для экспериментов (txt_groups, word_vocab, word_len)."""
    return FTTransformer(X["num"].shape[1], cat_cardinalities, X["txt"].shape[1], hp, **model_kw)


def _param_groups(model, wd):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        # как в FT-Transformer: без weight decay для токенизатора, LayerNorm и bias
        if name.startswith("tokenizer") or ".ln" in name or "head_ln" in name or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


@torch.no_grad()
def predict(model, X: dict[str, np.ndarray], device, batch_size: int = 4096) -> np.ndarray:
    """Прогноз в стандартизованной шкале (для перевода в log1p: · y_std + y_mean)."""
    model.eval()
    out = []
    n = len(X["num"])
    for i in range(0, n, batch_size):
        b = to_device(subset(X, slice(i, i + batch_size)), device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            out.append(model(**b).float().cpu().numpy())
    return np.concatenate(out) if out else np.empty(0, np.float32)


@torch.no_grad()
def embed(model, X: dict[str, np.ndarray], device, batch_size: int = 4096) -> np.ndarray:
    """Эмбеддинги [CLS] (после финального LayerNorm)."""
    model.eval()
    out = []
    for i in range(0, len(X["num"]), batch_size):
        b = to_device(subset(X, slice(i, i + batch_size)), device)
        out.append(model.embed(**b).float().cpu().numpy())
    return np.concatenate(out)


def fit(model, X_tr, y_tr, X_va, y_va_log, y_mean, y_std, device, thp: dict | None = None,
        log: logging.Logger | None = None, seed: int = SEED):
    """Обучение с ранней остановкой по val RMSLE.

    y_tr — стандартизованная цель ((log1p(y) − y_mean) / y_std), y_va_log — log1p(y) для val.
    Возвращает (модель с лучшими весами, история по эпохам, номер лучшей эпохи).
    """
    thp = {**TRAIN_HP, **(thp or {})}
    log = log or logging.getLogger(__name__)
    set_seed(seed)
    model.to(device)
    use_amp = thp["amp"] and device.type == "cuda"
    opt = torch.optim.AdamW(_param_groups(model, thp["weight_decay"]), lr=thp["lr"])
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=thp["lr_plateau_factor"],
                                                         patience=thp["lr_plateau_patience"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    Xt = to_device(X_tr, device)
    yt = torch.as_tensor(y_tr, dtype=torch.float32, device=device)
    n, bs = len(yt), thp["batch_size"]
    steps_per_epoch = math.ceil(n / bs)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    if thp["loss"] == "huber":
        delta_std = thp["huber_delta"] / y_std  # цель стандартизована: δ в log-шкале → δ / y_std
        loss_fn = lambda p, t: F.huber_loss(p, t, delta=delta_std)  # noqa: E731
    else:
        loss_fn = F.mse_loss

    best = (float("inf"), None, -1)
    history, step, bad = [], 0, 0
    for epoch in range(1, thp["max_epochs"] + 1):
        model.train()
        t0 = time.perf_counter()
        perm = torch.randperm(n, generator=gen).to(device)
        tot = 0.0
        for i in range(steps_per_epoch):
            idx = perm[i * bs:(i + 1) * bs]
            if step < thp["warmup_steps"]:
                for g in opt.param_groups:
                    g["lr"] = thp["lr"] * (step + 1) / thp["warmup_steps"]
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                pred = model(**{k: v[idx] for k, v in Xt.items()})
            loss = loss_fn(pred.float(), yt[idx])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), thp["grad_clip"])
            scaler.step(opt)
            scaler.update()
            tot += F.mse_loss(pred.float().detach(), yt[idx]).item() * len(idx)
            step += 1
        train_mse_std = tot / n
        p_va = predict(model, X_va, device) * y_std + y_mean
        val_rmsle = float(np.sqrt(np.mean((p_va - y_va_log) ** 2)))
        lr_now = opt.param_groups[0]["lr"]
        if step >= thp["warmup_steps"]:
            plateau.step(val_rmsle)
        rec = {"epoch": epoch, "train_rmsle": float(np.sqrt(train_mse_std) * y_std), "val_rmsle": val_rmsle,
               "lr": lr_now, "sec": time.perf_counter() - t0}
        history.append(rec)
        improved = val_rmsle < best[0] - 1e-4
        if improved:
            best, bad = (val_rmsle, copy.deepcopy(model.state_dict()), epoch), 0
        else:
            bad += 1
        log.info("эпоха %3d | train RMSLE %.4f | val RMSLE %.4f%s | lr %.1e | %.1f с",
                 epoch, rec["train_rmsle"], val_rmsle, " *" if improved else "", lr_now, rec["sec"])
        if not np.isfinite(val_rmsle):
            log.error("val RMSLE не конечен — остановка")
            break
        if bad >= thp["patience"]:
            log.info("ранняя остановка: %d эпох без улучшения", bad)
            break
    model.load_state_dict(best[1])
    return model, history, best[2]
