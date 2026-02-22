from typing import Iterable, Optional, Tuple, Literal
from numpy.typing import NDArray
from scipy.stats import norm, t
import numpy as np

Dist = Literal["normal", "t"]
ArrayLike = Iterable[float]


def estimate_sigma(
    values: np.ndarray,
    weights: Optional[ArrayLike] = None,
) -> float:
    """
    Estimate volatility for single asset or portfolio returns.
    """
    if values.ndim == 1:
        if values.size < 2:
            raise ValueError("Insufficient sample size.")
        sigma = values.std(ddof=1)

    elif values.ndim == 2:
        if weights is None:
            raise ValueError("Weights required for portfolio volatility.")

        weights_arr = np.asarray(weights, dtype=float)
        sigma = _portfolio_volatility(values, weights_arr)

    else:
        raise ValueError("Values must be 1D or 2D.")

    if sigma <= 0.0:
        raise ValueError("Zero or negative volatility.")

    return float(sigma)


def tail_quantile(
    gamma: float,
    distribution: Dist,
    df: Optional[int] = None,
) -> float:
    """
    Return left-tail quantile for the selected distribution.
    """
    if not 0.0 < gamma < 1.0:
        raise ValueError("gamma must be in (0, 1).")

    if distribution == "normal":
        return float(norm.ppf(gamma))

    if distribution == "t":
        if df is None:
            raise ValueError("df must be provided for t distribution.")
        if df <= 2:
            raise ValueError("df must be > 2.")
        return float(t.ppf(gamma, df=df))

    raise ValueError(f"Unsupported distribution: {distribution}.")


def time_scaling(
    value: float,
    n_days: int,
) -> float:
    """
    Apply square-root-of-time scaling.
    """
    if n_days <= 0:
        raise ValueError("n_days must be positive.")
    return value * np.sqrt(n_days)


def weighted_sorted_dist(
    returns: NDArray[np.floating],
    lamb: float,
) -> Tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
]:
    """
    Build a weighted, sorted returns distribution using
    Hull-style exponential decay.
    """
    if returns.ndim != 1:
        raise ValueError("returns must be a 1D array.")

    if returns.size < 2:
        raise ValueError("returns must contain at least two observations.")

    if not 0.0 < lamb < 1.0:
        raise ValueError("lamb must be in (0, 1).")

    n: int = returns.shape[0]

    weights: NDArray[np.floating] = (
        lamb ** (n - np.arange(1, n + 1))
        * (1.0 - lamb)
        / (1.0 - lamb ** n)
    )

    order = np.argsort(returns)

    sorted_returns = returns[order]
    sorted_weights = weights[order]
    cum_weights = np.cumsum(sorted_weights)

    return sorted_returns, sorted_weights, cum_weights


def _portfolio_volatility(
    returns: np.ndarray,
    weights: np.ndarray,
) -> float:
    """
    Compute portfolio volatility using covariance matrix.
    """
    if returns.shape[1] != weights.shape[0]:
        raise ValueError("Weights dimension mismatch.")

    cov = np.cov(returns, rowvar=False)
    variance = weights.T @ cov @ weights

    return float(np.sqrt(variance))
