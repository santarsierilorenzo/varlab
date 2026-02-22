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

        tail_mask = cum_w >= gamma

        if not np.any(tail_mask):
            raise RuntimeError("Empty ES tail.")

        es_value = float(
            np.sum(sorted_pnl[tail_mask] * sorted_w[tail_mask])
            / np.sum(sorted_w[tail_mask])
        )

    return time_scaling(es_value, n_days)


def _parametric_es(
    losses: np.ndarray,
    gamma: float,
    n_days: int,
    weights: Optional[ArrayLike],
    distribution: str,
    df: Optional[int],
    mean: Literal["zero", "sample"] = "zero",
) -> float:
    """
    Parametric Expected Shortfall under i.i.d. assumption.

    Assumes:

        L_t = mu + sigma * Z_t

    where Z_t follows a standardized distribution (normal or Student-t).

    Multi-day ES is computed as:

        ES_N = N * mu + sqrt(N) * ES_1_centered
    """
    sigma = estimate_sigma(
        returns=losses,
        weights=weights,
    )

    mu = np.mean(losses) if mean == "sample" else 0.0

    if distribution == "normal":
        z = norm.ppf(gamma)
        es_std = norm.pdf(z) / (1.0 - gamma)

    elif distribution == "t":
        if df is None:
            raise ValueError(
                "df must be provided for Student-t distribution."
            )

        q = t.ppf(gamma, df=df)

        # ES for standard Student-t
        es_std = (
            t.pdf(q, df=df)
            * (df + q**2)
            / ((df - 1) * (1.0 - gamma))
        )

        # Standardize Student-t to unit variance
        es_std *= np.sqrt((df - 2) / df)

    else:
        raise ValueError(
            f"Unsupported distribution: {distribution}."
        )

    es_value = mu * n_days + time_scaling(
        sigma * es_std,
        n_days,
    )

    return float(es_value)