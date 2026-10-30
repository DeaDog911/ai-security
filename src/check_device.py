"""Проверка окружения: PyTorch, GPU (ROCm/CUDA) и вспомогательные библиотеки.

Запуск из корня проекта:  venv\\Scripts\\python src\\check_device.py
Код возврата 0 — всё в порядке (GPU или CPU), 1 — найдены ошибки.
"""
from __future__ import annotations

import importlib
import platform
import sys
import time

OK, WARN, FAIL = "[OK]  ", "[WARN]", "[FAIL]"
problems: list[str] = []


def report(status: str, msg: str) -> None:
    print(f"{status} {msg}")
    if status == FAIL:
        problems.append(msg)


def check_packages() -> None:
    print("\n--- Библиотеки ---")
    for name in ["numpy", "pandas", "pyarrow", "scipy", "sklearn", "matplotlib", "cleanlab", "datasketch"]:
        try:
            mod = importlib.import_module(name)
            report(OK, f"{name} {getattr(mod, '__version__', '?')}")
        except Exception as e:  # noqa: BLE001
            report(FAIL, f"{name} не импортируется: {e}")


def sync(torch, device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def bench_train(torch, device, steps: int = 20) -> float:
    """Несколько шагов обучения трансформера размера, близкого к FT-Transformer проекта."""
    nn = torch.nn
    torch.manual_seed(42)
    batch, tokens, d = 512, 32, 128
    layer = nn.TransformerEncoderLayer(d, nhead=8, dim_feedforward=256, dropout=0.1,
                                       batch_first=True, norm_first=True)
    model = nn.Sequential(nn.TransformerEncoder(layer, num_layers=3, enable_nested_tensor=False)).to(device)
    head = nn.Linear(d, 1).to(device)
    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=1e-3)
    x = torch.randn(batch, tokens, d, device=device)
    y = x[:, 0, :4].sum(dim=1, keepdim=True)

    def step():
        opt.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(head(model(x)[:, 0]), y)
        loss.backward()
        opt.step()
        return loss

    first = step().item()  # прогрев
    sync(torch, device)
    t0 = time.perf_counter()
    for _ in range(steps):
        last = step()
    sync(torch, device)
    dt = (time.perf_counter() - t0) / steps
    last = last.item()
    if not (last == last) or last > first * 1.5:  # NaN или расходимость
        report(FAIL, f"обучение на {device}: loss {first:.3f} -> {last:.3f}")
    else:
        report(OK, f"обучение на {device}: loss {first:.3f} -> {last:.3f}, {dt * 1000:.0f} мс/шаг")
    return dt


def main() -> int:
    print("--- Система ---")
    print(f"Python   {sys.version.split()[0]} ({platform.system()} {platform.release()}, {platform.machine()})")
    if sys.version_info[:2] != (3, 12):
        report(WARN, "колёса PyTorch для ROCm собраны под Python 3.12")

    try:
        import torch
    except Exception as e:  # noqa: BLE001
        report(FAIL, f"PyTorch не импортируется: {e}")
        check_packages()
        return 1

    print(f"PyTorch  {torch.__version__}")
    hip = getattr(torch.version, "hip", None)
    print(f"ROCm/HIP {hip or '—'}   CUDA {torch.version.cuda or '—'}")

    print("\n--- Устройство ---")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(0)
        arch = getattr(props, "gcnArchName", "")
        report(OK, f"GPU: {torch.cuda.get_device_name(0)} {arch} | память {props.total_memory / 2**30:.1f} ГБ")
    else:
        device = torch.device("cpu")
        if hip is None and torch.version.cuda is None:
            report(WARN, "установлена сборка PyTorch без поддержки GPU")
        report(WARN, "GPU недоступен — обучение пойдёт на CPU "
                     f"({torch.get_num_threads()} потоков). Проверьте драйвер AMD и сборку torch+rocm.")

    print("\n--- Тесты вычислений ---")
    torch.manual_seed(42)
    a, b = torch.randn(1024, 1024), torch.randn(1024, 1024)
    ref = a @ b
    try:
        got = (a.to(device) @ b.to(device)).cpu()
        err = (got - ref).abs().max().item()
        report(OK if err < 1e-2 else FAIL, f"matmul {device} vs CPU: макс. расхождение {err:.2e}")

        q = torch.randn(64, 8, 32, 16, device=device, requires_grad=True)
        out = torch.nn.functional.scaled_dot_product_attention(q, q, q)
        out.sum().backward()
        finite = torch.isfinite(q.grad).all().item()
        report(OK if finite else FAIL, "scaled_dot_product_attention + backward")

        if device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                h = (a.to(device) @ b.to(device))
            report(OK if torch.isfinite(h).all().item() else WARN, "autocast float16")

        print("\n--- Скорость (модель размера проекта, batch 512) ---")
        t_dev = bench_train(torch, device)
        if device.type == "cuda":
            t_cpu = bench_train(torch, torch.device("cpu"), steps=5)
            print(f"Ускорение GPU относительно CPU: x{t_cpu / t_dev:.1f}")
    except Exception as e:  # noqa: BLE001
        report(FAIL, f"ошибка вычислений на {device}: {type(e).__name__}: {e}")

    check_packages()

    print("\n--- Итог ---")
    if problems:
        print("Есть ошибки:\n  - " + "\n  - ".join(problems))
        return 1
    print(f"Окружение готово, устройство для обучения: {device}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
