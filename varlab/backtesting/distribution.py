from __future__ import annotations

from typing import Iterable, Optional, Literal
from scipy.stats import norm, t
import numpy as np



Dist = Literal["normal", "t"]


def _pit(
    values: Iterable[float],
    mu: float,
    sigma: float,
    dist: Dist = "normal",
    df: Optional[float] = None,
) -> np.ndarray:
    """
    Compute the Probability Integral Transform (PIT).

    The PIT is defined as:

        U = F_X(x)

    where F_X is the cumulative distribution function of the assumed
    parametric distribution.

    Parameters
    ----------
    values : Iterable[float]
        Observations at which the CDF is evaluated.
    mu : float
        Location parameter of the distribution.
    sigma : float
        Scale parameter of the distribution. Must be strictly positive.
    dist : {"normal", "t"}, default="normal"
        Parametric distribution used to compute the CDF.
    df : float, optional
        Degrees of freedom of the Student's t distribution.
        Required if dist="t".

    Returns
    -------
    np.ndarray
        Array of PIT values in the interval [0, 1].

    Raises
    ------
    ValueError
        If sigma <= 0 or if df is missing/invalid when dist="t".
    NotImplementedError
        If the specified distribution is not supported.

    Notes
    -----
    For correctly specified predictive distributions, PIT values
    should be i.i.d. Uniform(0, 1).
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    x: np.ndarray = np.asarray(values, dtype=float)
    d: str = dist.strip().lower()

    if d == "normal":
        return norm.cdf(x, loc=mu, scale=sigma)

    if d == "t":
        if df is None or df <= 0:
            raise ValueError(
                "df must be provided and > 0 for t distribution"
            )
        return t.cdf(x, df=df, loc=mu, scale=sigma)

    raise NotImplementedError(f"{dist} not implemented")


def _randomized_pit(
    history: Iterable[float],
    x: float,
) -> float:
    """
    Compute the randomized PIT for an empirical discrete distribution.

    The randomized PIT is defined as:

        U = F(x^-) + V * P(X = x)

    where:
        - F(x^-) is the left-limit of the empirical CDF,
        - P(X = x) is the empirical probability mass at x,
        - V ~ Uniform(0, 1).

    Parameters
    ----------
    history : Iterable[float]
        Historical observations defining the empirical distribution.
        NaN values are ignored.
    x : float
        Observation at which the randomized PIT is evaluated.

    Returns
    -------
    float
        Randomized PIT value in the interval [0, 1].

    Raises
    ------
    ValueError
        If history is empty after removing NaN values.
    """
    hist: np.ndarray = np.asarray(history, dtype=float)
    hist = hist[~np.isnan(hist)]

    n: int = hist.size
    if n == 0:
        raise ValueError("history must not be empty")

    less: int = int(np.sum(hist < x))
    equal: int = int(np.sum(hist == x))

    v: float = float(np.random.uniform(0.0, 1.0))

    return (less + v * equal) / n
