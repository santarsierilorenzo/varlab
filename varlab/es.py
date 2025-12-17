from typing import Iterable, Optional
import numpy as np
from .base import (
    estimate_sigma,
    exponential_weights,
    left_tail_quantile,
    time_scaling
)

ArrayLike = Iterable[float]


def es(
    returns: ArrayLike,
    n_days: int = 1,
    confidence: float = 0.99,
    method: str = "empirical",
    weights: Optional[ArrayLike] = None,
    distribution: str = "normal",
    df: Optional[int] = None,
    lamb: Optional[float] = None,
) -> float:
    """
    """
    returns_arr = np.asarray(returns, dtype=float)

    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1).")

    gamma = 1.0 - confidence

    if method == "empirical":
        return _empirical_es(
            returns_arr=returns_arr,
            gamma=gamma,
            n_days=n_days,
            lamb=lamb,
        )

    if method == "parametric":
        return _parametric_es(
            returns_arr=returns_arr,
            gamma=gamma,
            n_days=n_days,
            weights=weights,
            distribution=distribution,
            df=df,
        )

    raise ValueError(f"Unsupported VaR method: {method}.")


def _empirical_es():
    pass


def _parametric_es():
    pass

