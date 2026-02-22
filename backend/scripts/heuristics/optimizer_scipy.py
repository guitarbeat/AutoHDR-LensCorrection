"""Optional SciPy optimizers for heuristic coefficient search."""

from __future__ import annotations

import time
from typing import Callable, Sequence

import numpy as np


def differential_evolution_search(
    *,
    objective: Callable[[np.ndarray], float],
    bounds: Sequence[tuple[float, float]],
    seed: int = 42,
    maxiter: int = 20,
    popsize: int = 10,
    timeout_s: float = 45.0,
) -> dict[str, object]:
    """Run bounded differential-evolution search with timeout guard."""
    try:
        from scipy.optimize import differential_evolution
    except Exception as exc:
        raise RuntimeError("SciPy is required for --optimizer de") from exc

    start = time.monotonic()
    timed_out = False
    eval_count = 0
    best_x: np.ndarray | None = None
    best_loss = float("inf")

    def wrapped(params: np.ndarray) -> float:
        nonlocal eval_count, best_x, best_loss
        eval_count += 1
        loss = float(objective(params))
        if loss < best_loss:
            best_loss = loss
            best_x = np.array(params, dtype=np.float64)
        return loss

    def callback(_xk: np.ndarray, _convergence: float) -> bool:
        nonlocal timed_out
        if time.monotonic() - start > timeout_s:
            timed_out = True
            return True
        return False

    result = differential_evolution(
        wrapped,
        bounds=bounds,
        seed=seed,
        maxiter=maxiter,
        popsize=popsize,
        polish=False,
        updating="deferred",
        workers=1,
        callback=callback,
    )

    final_x = np.array(result.x, dtype=np.float64)
    final_loss = float(result.fun)
    if best_x is not None and best_loss < final_loss:
        final_x = best_x
        final_loss = best_loss

    return {
        "x": final_x,
        "loss": final_loss,
        "success": bool(result.success) and not timed_out,
        "message": str(result.message),
        "timed_out": timed_out,
        "evaluations": int(eval_count),
        "elapsed_s": float(time.monotonic() - start),
    }
