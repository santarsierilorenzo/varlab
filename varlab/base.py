from typing import Iterable, Optional
from scipy.stats import norm, t
import numpy as np

ArrayLike = Iterable[float]


def estimate_sigma(
    returns: np.ndarray,
    weights: Optional[ArrayLike] = None,
) -> float:
    """
    Estimate volatility for single asset or portfolio returns.
    """
    if returns.ndim == 1:
        if returns.size < 2:
            raise ValueError("Insufficient sample size.")
        sigma = returns.std(ddof=1)

    elif returns.ndim == 2:
        if weights is None:
            raise ValueError("Weights required for portfolio volatility.")
        
        weights_arr = np.asarray(weights, dtype=float)
        sigma = _portfolio_volatility(returns, weights_arr)
        
    else:
        raise ValueError("Returns must be 1D or 2D.")

    if sigma <= 0.0:
        raise ValueError("Zero or negative volatility.")

    return float(sigma)


def left_tail_quantile(
    gamma: float,
    distribution: str,
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


def exponential_weights(
    n_obs: int,
    lamb: float,
) -> np.ndarray:
    """
    This function implements the weighting formula proposed by John C. Hull in
    the book "Options, Futures, and Other Derivatives". 

    The function applies exponential weighting to the returns. The weights
    assigned to past observations decline at a rate controlled by lambda: a
    higher lambda (closer to 1) implies a slower decline, resulting in a
    "longer" memory of past volatility
    """
    if not 0.0 < lamb < 1.0:
        raise ValueError("lamb must be in (0, 1).")

    powers = np.arange(n_obs - 1, -1, -1)
    weights = (1.0 - lamb) * lamb ** powers
    return weights / weights.sum()


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
