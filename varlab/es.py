from typing import Iterable, Optional, Literal
from scipy.stats import norm, t
import numpy as np
from .base import (
    estimate_sigma,
    weighted_sorted_dist,
    time_scaling,
)

Method = Literal["empirical", "parametric"]
ArrayLike = Iterable[float]


def es(
    returns: ArrayLike,
    n_days: int = 1,
    confidence: float = 0.99,
    method: Method = "empirical",
    weights: Optional[ArrayLike] = None,
    distribution: str = "normal",
    df: Optional[int] = None,
    lamb: Optional[float] = None,
) -> float:
    """
    Compute Expected Shortfall (ES).
    """
    losses = -np.asarray(returns, dtype=float)

    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1).")

    if method == "empirical":
        return _empirical_es(
            losses=losses,
            gamma=confidence,
            n_days=n_days,
            lamb=lamb,
        )

    if method == "parametric":
        return _parametric_es(
            losses=losses,
            gamma=confidence,
            n_days=n_days,
            weights=weights,
            distribution=distribution,
            df=df,
        )

    raise ValueError(f"Unsupported ES method: {method}.")


def _empirical_es(
    losses: np.ndarray,
    gamma: float,
    n_days: int,
    lamb: Optional[float],
) -> float:
    """
    Empirical (historical) Expected Shortfall.
    """
    if losses.ndim != 1:
        raise ValueError("Empirical ES requires 1D returns.")

    if lamb is None:
        q = np.quantile(losses, gamma, method="higher")
        tail = losses[losses >= q]
        es_value = float(tail.mean())

    else:
        sorted_pnl, sorted_w, cum_w = weighted_sorted_dist(
            losses,
            lamb,
        )

        idx = int(np.searchsorted(cum_w, gamma, side="right"))
        idx = min(idx, len(sorted_pnl) - 1)

        tail_pnl = sorted_pnl[idx:]
        tail_w = sorted_w[idx:]

        if tail_w.size == 0:
            raise RuntimeError("Empty ES tail.")

        es_value = float(
            np.sum(tail_pnl * tail_w) / np.sum(tail_w)
        )

    return time_scaling(es_value, n_days)


def _parametric_es(
    losses: np.ndarray,
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
        returns=losses,
        weights=weights,
    )

    if distribution == "normal":
        q = norm.ppf(gamma)
        num = np.e**(-q**2 / 2)
        den = (np.sqrt(2 * np.pi) * (1-gamma))

    elif distribution == "t":
        if df is None:
            raise ValueError("df must be provided for t distribution.")
        q = t.ppf(gamma, df=df)

        num = t.pdf(q, df=df) * (df + q ** 2)
        den = (df - 1) * (1.0 - gamma)
        
    else:
        raise ValueError(f"Unsupported distribution: {distribution}.")

    es_value = sigma * num / den

    return time_scaling(es_value, n_days)
