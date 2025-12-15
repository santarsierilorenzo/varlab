from typing import Iterable, Optional
import numpy as np
from scipy.stats import norm


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
    rolling_window : Optional[int], default=None
        Window size for rolling estimation.
    n_sims : int, default=100_000
        Number of Monte Carlo simulations (only for montecarlo).

    Returns
    -------
    float
        Value at Risk (positive number).
    """
    returns = np.asarray(daily_returns, dtype=float)

    if returns.ndim != 1:
        raise ValueError("daily_returns must be a 1D array-like.")

    if rolling_window is not None:
        if rolling_window <= 1:
            raise ValueError("rolling_window must be > 1.")
        returns = returns[-rolling_window:]

    if method == "empirical":
        return _empirical_method(returns, n_days, confidence)

    if method == "parametric":
        return _parametric_method(returns, n_days, confidence)

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
    Empirical (historical) VaR with square-root-of-time scaling.
    """
    gamma: float = 1.0 - confidence

    var: float = -np.quantile(returns, gamma) * np.sqrt(n_days)

    return var


def _parametric_method(
    returns: np.ndarray,
    n_days: int,
    confidence: float,
) -> float:
    """
    Parametric Gaussian VaR.
    """
    gamma: float = confidence
    z: float = norm.ppf(gamma)
    sigma: float = returns.std(ddof=1)

    if sigma == 0.0:
        raise ValueError("Zero volatility: VaR undefined.")

    var: float = z * sigma * np.sqrt(n_days)

    return var
