from typing import Iterable, Optional
from scipy.stats import norm, t
import numpy as np
from .base import (
    estimate_sigma,
    weighted_sorted_dist,
    left_tail_quantile,
    time_scaling,
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
    Compute Expected Shortfall (ES).
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

    raise ValueError(f"Unsupported ES method: {method}.")


def _empirical_es(
    returns_arr: np.ndarray,
    gamma: float,
    n_days: int,
    lamb: Optional[float],
) -> float:
    """
    Empirical (historical) Expected Shortfall.
    """
    if returns_arr.ndim != 1:
        raise ValueError("Empirical ES requires 1D returns.")

    if lamb is None:
        q = np.quantile(returns_arr, gamma, method="higher")
        tail = returns_arr[returns_arr <= q]
        es_value = -float(tail.mean())

    else:
        sorted_pnl, sorted_w, cum_w = weighted_sorted_dist(
            returns_arr,
            lamb,
        )

        tail_mask = cum_w <= gamma

        if not np.any(tail_mask):
            raise RuntimeError("Empty ES tail.")

        es_value = -float(
            np.sum(sorted_pnl[tail_mask] * sorted_w[tail_mask])
            / np.sum(sorted_w[tail_mask])
        )

    return time_scaling(es_value, n_days)


def _parametric_es(
    returns_arr: np.ndarray,
    gamma: float,
    n_days: int,
    weights: Optional[ArrayLike],
    distribution: str,
    df: Optional[int],
) -> float:
    """
    Parametric ES assuming zero-mean i.i.d. returns.
    """
    sigma = estimate_sigma(
        returns=returns_arr,
        weights=weights,
    )

    q = left_tail_quantile(
        gamma=gamma,
        distribution=distribution,
        df=df,
    )

    if distribution == "normal":
        es_value = sigma * norm.pdf(q) / gamma

    elif distribution == "t":
        if df is None:
            raise ValueError("df must be provided for t distribution.")
        num = t.pdf(q, df=df) * (df + q ** 2)
        den = (df - 1) * gamma
        es_value = sigma * num / den

    else:
        raise ValueError(f"Unsupported distribution: {distribution}.")

    return time_scaling(es_value, n_days)
