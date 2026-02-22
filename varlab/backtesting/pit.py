from typing import Iterable, Optional, Tuple, Literal
from scipy.stats import norm, t
import pandas as pd
import numpy as np

Dist = Literal["normal", "t"]


def pit_checks(
    case: Literal["continuous", "discrete"],
    distribution: Dist,
    df: Optional[int],
) -> None:
    if case not in {"continuous", "discrete"}:
        raise ValueError("case must be either 'continuous' or 'discrete'")

    if distribution not in {"normal", "t"}:
        raise ValueError("distribution must be 'normal' or 't'")

    if distribution == "t":
        if df is None or df <= 2:
            raise ValueError("df must be provided and > 2 for t distribution")
        

def randomized_pit(
    history: Iterable[float],
    x: float,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Randomized PIT for empirical discrete distribution.

    U = F(x^-) + V * P(X = x),  V ~ Uniform(0,1)

    Parameters
    ----------
    history : Iterable[float]
        Historical observations.
    x : float
        Current observation.
    rng : Optional[np.random.Generator]
        Random number generator for reproducibility.
        If None, a new default generator is created.
    """
    hist = np.asarray(history, dtype=float)
    hist = hist[~np.isnan(hist)]

    n = hist.size
    if n == 0:
        raise ValueError("history must not be empty")

    less = np.sum(hist < x)
    equal = np.sum(hist == x)

    if rng is None:
        rng = np.random.default_rng()

    v = rng.uniform(0.0, 1.0)

    return float((less + v * equal) / n)


def rolling_pit(
    values: pd.Series,
    case: Literal["continuous", "discrete"] = "continuous",
    distribution: Dist = "normal",
    df: Optional[int] = None,
    window: int = 60,
    ddof: int = 1,
) -> pd.Series:
    """
    Rolling PIT consistent with mean-zero parametric VaR.

    Assumes:
        X_t = sigma_t * Z_t
        Z_t ~ D(0,1)
    """
    pit_checks(case, distribution, df)

    if window < 1:
        raise ValueError("window must be >= 1")

    values = values.astype(float)

    if case == "continuous":
        sigma = values.rolling(window).std(ddof=ddof).shift(1)
        pit = pd.Series(np.nan, index=values.index)
        valid = (~sigma.isna()) & (sigma > 0)

        if not valid.any():
            return pit

        z = values[valid] / sigma[valid]

        if distribution == "normal":
            pit[valid] = norm.cdf(z)
        else:
            pit[valid] = t.cdf(z, df=df)

        return pit

    # Discrete case
    u = [np.nan] * len(values)

    for t_idx in range(window, len(values)):
        hist = values.iloc[t_idx - window:t_idx].values
        x = values.iloc[t_idx]
        u[t_idx] = randomized_pit(hist, x)

    return pd.Series(u, index=values.index)


def expanding_pit(
    values: pd.Series,
    case: Literal["continuous", "discrete"] = "continuous",
    distribution: Dist = "normal",
    df: Optional[int] = None,
    min_periods: int = 30,
    ddof: int = 1,
) -> pd.Series:
    """
    Expanding-window PIT consistent with mean-zero parametric VaR.
    """
    pit_checks(case, distribution, df)

    if min_periods < 1:
        raise ValueError("min_periods must be >= 1")

    values = values.astype(float)
    u = [np.nan] * len(values)

    for t_idx in range(min_periods, len(values)):
        hist = values.iloc[:t_idx]

        if case == "continuous":
            sigma = hist.std(ddof=ddof)

            if sigma <= 0 or np.isnan(sigma):
                continue

            z = values.iloc[t_idx] / sigma

            if distribution == "normal":
                u[t_idx] = norm.cdf(z)
            else:
                u[t_idx] = t.cdf(z, df=df)

        else:
            u[t_idx] = randomized_pit(hist.values, values.iloc[t_idx])

    return pd.Series(u, index=values.index)
