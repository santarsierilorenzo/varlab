from typing import Iterable, Optional
import numpy as np
from .base import (
    estimate_sigma,
    exponential_weights,
    left_tail_quantile,
    time_scaling
)

ArrayLike = Iterable[float]


def var(
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
    Compute Value at Risk (VaR) for a return series or portfolio.

    Supports empirical and parametric methods with normal or t-Student
    distributions. Returns a positive VaR.

    Notes
    -----
    The function is scale-agnostic with respect to the input returns:

    - If `returns` represent PnL values, the VaR is expressed in monetary units
    - If `returns` represent simple returns, the VaR is expressed in percentage
    terms (assuming unit notional).
    """
    returns_arr = np.asarray(returns, dtype=float)

    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1).")

    gamma = 1.0 - confidence

    if method == "empirical":
        return _empirical_var(
            returns_arr=returns_arr,
            gamma=gamma,
            n_days=n_days,
            lamb=lamb,
        )

    if method == "parametric":
        return _parametric_var(
            returns_arr=returns_arr,
            gamma=gamma,
            n_days=n_days,
            weights=weights,
            distribution=distribution,
            df=df,
        )

    raise ValueError(f"Unsupported VaR method: {method}.")


def _empirical_var(
    returns_arr: np.ndarray,
    gamma: float,
    n_days: int,
    lamb: Optional[float],
) -> float:
    """
    Historical VaR using empirical quantiles.
    """
    if returns_arr.ndim != 1:
        raise ValueError("Empirical VaR requires 1D returns.")

    if np.min(returns_arr) >= 0.0:
        raise ValueError("Historical VaR undefined: no negative returns.")

    if lamb is None:
        # NOTE: The formal definition of VaR requires finding the infimum of
        # the set of losses where the cumulative distribution exceeds gamma.
        # In practical terms: always choose the greater value when estimating
        # the quantile =^.^=
        q = np.quantile(returns_arr, gamma, method="higher")
    else:
        weights = exponential_weights(
            n_obs=returns_arr.size,
            lamb=lamb,
        )
        q = _weighted_quantile(
            values=returns_arr,
            weights=weights,
            gamma=gamma,
        )

    var_value = -q
    return time_scaling(var_value, n_days)


def _parametric_var(
    returns_arr: np.ndarray,
    gamma: float,
    n_days: int,
    weights: Optional[ArrayLike],
    distribution: str,
    df: Optional[int],
) -> float:
    """
    Parametric VaR assuming zero-mean i.i.d. returns.
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

    var_value = -q * sigma
    return time_scaling(var_value, n_days)


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    gamma: float,
) -> float:
    """
    Compute weighted empirical quantile.
    """
    order = np.argsort(values)
    cum_weights = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cum_weights, gamma)])
