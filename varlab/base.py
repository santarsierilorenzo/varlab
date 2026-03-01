from typing import Iterable, Optional, Tuple, Literal, Union
from numpy.typing import NDArray
from scipy.stats import norm, t
import pandas as pd
import numpy as np

Rounding = Literal["floor", "ceil", "round", "stochastic"]
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
    Return tail quantile for the selected distribution.
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
    return float(value * np.sqrt(n_days))


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


def factor_rounding(
    scaling_factor: Union[float, Iterable[float], pd.Series],
    mode: Rounding,
    seed: int | None = None,
) -> Union[float, np.ndarray, pd.Series]:
    """
    Apply a rounding rule to a scaling factor.

    Parameters
    ----------
    scaling_factor : float, Iterable[float], or pd.Series
        Scalar or array-like of scaling factors to be rounded.
        If a pandas Series is provided, the index is preserved.
    mode : {"floor", "ceil", "round", "stochastic"}
        Rounding method.
    seed : int, optional
        Seed used for stochastic rounding. If None, randomness
        is not deterministic.

    Returns
    -------
    float, np.ndarray, or pd.Series
        Rounded value(s). Returns a float if input is scalar.
        Returns a NumPy array for generic iterables.
        Returns a pandas Series if input is a Series.
    """
    is_scalar = np.isscalar(scaling_factor)
    is_series = isinstance(scaling_factor, pd.Series)

    if is_scalar:
        sf = np.array([scaling_factor], dtype=float)

    elif is_series:
        sf = scaling_factor.to_numpy(dtype=float)

    else:
        sf = np.asarray(scaling_factor, dtype=float)

    sf = np.nan_to_num(sf, nan=0.0, posinf=0.0, neginf=0.0)

    if mode == "floor":
        result = np.floor(sf)

    elif mode == "ceil":
        result = np.ceil(sf)

    elif mode == "round":
        result = np.round(sf)

    elif mode == "stochastic":
        rng = np.random.default_rng(seed)

        n = np.floor(sf).astype(int)
        delta = sf - n
        u = rng.uniform(0.0, 1.0, size=sf.shape)
        result = np.where(u < delta, n + 1, n)

    else:
        raise ValueError(f"{mode} is not an available rounding mode")

    if is_scalar:
        return float(result[0])

    if is_series:
        return pd.Series(result, index=scaling_factor.index)

    return result
