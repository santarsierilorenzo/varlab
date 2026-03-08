from typing import Iterable, Optional, Literal
from scipy.stats import norm, t
import numpy as np
from .base import (
    estimate_sigma,
    weighted_sorted_dist,
    time_scaling,
    estimate_student_df,
)

Method = Literal["empirical", "parametric"]
Dist = Literal["normal", "t"]
ArrayLike = Iterable[float]


def es(
    returns: ArrayLike,
    n_days: int = 1,
    confidence: float = 0.99,
    method: Method = "empirical",
    weights: Optional[ArrayLike] = None,
    distribution: Dist = "normal",
    df: Optional[float] = None,
    lamb: Optional[float] = None,
    mean: Literal["zero", "sample"] = "zero",
) -> float:
    """
    Compute Expected Shortfall (ES).

    Supports empirical and parametric methods with normal or t-Student
    distributions. Returns a positive ES.

    Notes
    -----
    The function is scale-agnostic with respect to the input returns:

    - If `returns` represent PnL values, the ES is expressed in monetary units
    - If `returns` represent simple returns, the ES is expressed in percentage
      terms (assuming unit notional).

    Implementation details:
    - For the unweighted empirical method, the tail threshold is computed with
      `method="higher"` for consistency with the VaR quantile convention used
      in this package. ES is then estimated as the mean of losses in the tail.

    Parameters
    ----------
    returns : ArrayLike
        Return/PnL observations.
        For `method="empirical"` this must be a 1D time series of
        already-aggregated (already-weighted) portfolio returns/losses.
        For `method="parametric"`, 2D input (T, N) is interpreted as
        asset returns.
    n_days : int, default=1
        Forecast horizon in days. Must be positive.
    confidence : float, default=0.99
        Confidence level in (0, 1).
    method : {"empirical", "parametric"}, default="empirical"
        Estimation method. Empirical method supports only 1D inputs.
    weights : Optional[ArrayLike], default=None
        Portfolio weights. Used for 2D inputs in the parametric method.
    distribution : {"normal", "t"}, default="normal"
        Parametric distribution: "normal" or "t".
    df : Optional[float], default=None
        Degrees of freedom for Student-t. If None and
        `distribution="t"`, estimated from sample via MLE.
    lamb : Optional[float], default=None
        Exponential decay parameter for weighted empirical ES. If provided,
        must be in (0, 1). If None, equal weights are used.

    Returns
    -------
    float
        ES as a positive number (loss magnitude).
    """
    if distribution not in {"normal", "t"}:
        raise ValueError("distribution must be 'normal' or 't'")

    if mean not in {"zero", "sample"}:
        raise ValueError("mean must be 'zero' or 'sample'")
    
    losses = -np.asarray(returns, dtype=float)

    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1).")

    if method == "parametric":
        return _parametric_es(
            losses=losses,
            gamma=confidence,
            n_days=n_days,
            weights=weights,
            distribution=distribution,
            df=df,
            mean=mean,
        )
    
    if method == "empirical":
        return _empirical_es(
            losses=losses,
            gamma=confidence,
            n_days=n_days,
            lamb=lamb,
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
    distribution: Dist,
    df: Optional[float],
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
        values=losses,
        weights=weights,
    )

    portfolio_losses: Optional[np.ndarray] = None

    if mean == "sample":
        if weights is None:
            mu = np.mean(losses)
        else:
            weights_arr = np.asarray(weights, dtype=float)
            if losses.ndim == 1:
                mu = np.mean(losses)
            else:
                portfolio_losses = np.sum(losses * weights_arr, axis=1)
                mu = np.mean(portfolio_losses)
    else:
        mu = 0

    if distribution == "normal":
        z = norm.ppf(gamma)
        es_std = norm.pdf(z) / (1.0 - gamma)

    elif distribution == "t":
        if df is None:
            if losses.ndim == 1:
                sample_for_df = losses
            elif weights is not None:
                if portfolio_losses is None:
                    weights_arr = np.asarray(weights, dtype=float)
                    portfolio_losses = np.sum(losses * weights_arr, axis=1)
                sample_for_df = portfolio_losses
            else:
                sample_for_df = losses.ravel()

            df = estimate_student_df(
                sample_for_df,
                fit_mean=(mean == "sample"),
            )

        if df <= 2:
            raise ValueError("df must be > 2 for finite variance.")

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
