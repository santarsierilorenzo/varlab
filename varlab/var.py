from typing import Iterable, Optional
from scipy.stats import norm
import numpy as np


def var(
    daily_returns: Iterable[float],
    n_days: int = 1,
    confidence: float = 0.99,
    method: str = "empirical",
    rolling_window: Optional[int] = None,
    n_sims: int = 100_000,
) -> float:
    """
    Compute Value at Risk (VaR) for a single return series.

    Parameters
    ----------
    daily_returns : Iterable[float]
        Daily returns time series.
    n_days : int, default=1
        Horizon in days.
    confidence : float, default=0.99
        Confidence level.
    method : str, default="empirical"
        One of {"empirical", "parametric", "montecarlo"}.
    n_sims : int, default=100_000
        Number of Monte Carlo simulations (only for montecarlo).

    Returns
    -------
    float
        Value at Risk (positive number).
    """
    daily_returns = np.asarray(daily_returns, dtype=float)

    if daily_returns.ndim != 1:
        raise ValueError("daily_returns must be a 1D array-like.")

    if method == "empirical":
        return _empirical_method(daily_returns, n_days, confidence)

    if method == "parametric":
        return _parametric_method(daily_returns, n_days, confidence)

    if method == "montecarlo":
        raise NotImplementedError(
            "Montecarlo method has not been implemented yet"
        )

    raise ValueError(f"{method} method is not supported.")


def _empirical_method(
    returns: np.ndarray,
    n_days: int,
    confidence: float,
) -> float:
    """
    Compute historical (empirical) Value at Risk (VaR) using quantiles and
    square-root-of-time scaling.

    The VaR is estimated as the left-tail quantile of the empirical
    distribution of single-period returns at level (1 - confidence), scaled by
    sqrt(n_days).

    This method requires the presence of negative returns in the sample.
    If no downside observations are available, the historical VaR is
    statistically undefined and an error is raised.

    Assumptions:
    - Time aggregation follows the square-root-of-time rule.

    Parameters
    ----------
    returns : np.ndarray
        Array of single-period returns.
    n_days : int
        Forecast horizon in days.
    confidence : float
        Confidence level (e.g. 0.99 for 99% VaR).

    Returns
    -------
    float
        Positive Value at Risk.

    Raises
    ------
    ValueError
        If no negative returns are present in the input sample.
    """
    if np.min(returns) >= 0.0:
        raise ValueError(
            "Historical VaR undefined: no negative returns in sample."
        )

    gamma: float = 1.0 - confidence

    var: float = np.quantile(returns, gamma) * np.sqrt(n_days)

    return -var


def _parametric_method(
    returns: np.ndarray,
    n_days: int,
    confidence: float,
) -> float:
    """
    Compute parametric Gaussian Value at Risk (VaR).

    The VaR is estimated assuming normally distributed single-period returns.
    The left-tail quantile at level (1 - confidence) is scaled by the
    square-root-of-time rule.

    Assumptions:
    - Returns are i.i.d.
    - Returns follow a Gaussian distribution.
    - Time aggregation follows the square-root-of-time rule.

    Parameters
    ----------
    returns : np.ndarray
        Array of single-period returns.
    n_days : int
        Forecast horizon in days.
    confidence : float
        Confidence level (e.g. 0.99 for 99% VaR).

    Returns
    -------
    float
        Positive Value at Risk.

    Raises
    ------
    ValueError
        If the sample volatility is zero.
    """
    if returns.size < 2:
        raise ValueError(
            "Parametric VaR undefined: insufficient sample size."
        )

    sigma: float = returns.std(ddof=1)

    if sigma == 0.0:
        raise ValueError(
            "Parametric VaR undefined: zero volatility."
        )

    gamma: float = 1.0 - confidence
    z: float = norm.ppf(gamma)

    var: float = z * sigma * np.sqrt(n_days)

    return -var
