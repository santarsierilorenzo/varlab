from typing import Iterable, Optional, Tuple, Literal
from ..window import RollingRisk, ExpandingRisk
from ..base import estimate_sigma
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
    returns: pd.Series | pd.DataFrame,
    case: Literal["continuous", "discrete"] = "continuous",
    weights: Optional[pd.Series | pd.DataFrame] = None,
    distribution: Dist = "normal",
    df: Optional[int] = None,
    window: int = 60,
    ddof: int = 1,
    mean: Literal["zero", "sample"] = "zero",
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

    if case == "continuous":

        returns = returns.astype(float)

        # Case 1: single asset
        if isinstance(returns, pd.Series):
            x = returns

            mu = (
                x.rolling(window).mean().shift(1)
                if mean == "sample"
                else 0.0
            )

            sigma = RollingRisk(
                estimate_sigma,
                window=window,
                ddof=ddof,
            )(x).shift(1)

        # Case 2: portfolio with time-varying weights
        else:

            if weights is None:
                raise ValueError(
                    "weights must be provided when returns is a DataFrame"
                )

            if not isinstance(weights, pd.DataFrame):
                raise ValueError(
                    "weights must be a DataFrame when returns is a DataFrame"
                )

            if not returns.index.equals(weights.index):
                raise ValueError("returns and weights index mismatch")

            if not returns.columns.equals(weights.columns):
                raise ValueError("returns and weights columns mismatch")

            # Portfolio observation
            x = (returns * weights).sum(axis=1)

            mu = (
                x.rolling(window).mean().shift(1)
                if mean == "sample"
                else 0.0
            )

            sigma = RollingRisk(
                estimate_sigma,
                window=window,
                ddof=ddof,
            )(returns, weights=weights).shift(1)

        pit = pd.Series(np.nan, index=x.index)

        valid = (~sigma.isna()) & (sigma > 0)

        if not valid.any():
            return pit

        if mean == "sample":
            z = (x[valid] - mu[valid]) / sigma[valid]
        else:
            z = x[valid] / sigma[valid]

        if distribution == "normal":
            pit[valid] = norm.cdf(z)
        else:
            # Map unit-variance standardized residuals to canonical Student-t
            # scale before applying the t CDF.
            z *= np.sqrt(df / (df - 2))

            pit[valid] = t.cdf(z, df=df)

        return pit

    # Discrete case
    returns = returns.astype(float)

    if isinstance(returns, pd.DataFrame):
        raise ValueError(
            "Discrete PIT currently supports only Series inputs"
        )

    u = [np.nan] * len(returns)

    for t_idx in range(window, len(returns)):
        hist = returns.iloc[t_idx - window:t_idx].to_numpy()
        x = returns.iloc[t_idx]
        u[t_idx] = randomized_pit(hist, x)

    return pd.Series(u, index=returns.index)


def expanding_pit(
    returns: pd.Series | pd.DataFrame,
    case: Literal["continuous", "discrete"] = "continuous",
    weights: Optional[pd.Series | pd.DataFrame] = None,
    distribution: Dist = "normal",
    df: Optional[int] = None,
    min_periods: int = 30,
    ddof: int = 1,
    mean: Literal["zero", "sample"] = "zero",
) -> pd.Series:
    """
    Expanding-window PIT consistent with mean-zero parametric VaR.
    """
    pit_checks(case, distribution, df)

    if min_periods < 1:
        raise ValueError("min_periods must be >= 1")

    if case == "continuous":

        returns = returns.astype(float)

        # Case 1: single asset
        if isinstance(returns, pd.Series):

            x = returns

            mu = (
                x.expanding(min_periods).mean().shift(1)
                if mean == "sample"
                else 0.0
            )

            sigma = ExpandingRisk(
                estimate_sigma,
                min_periods=min_periods,
                ddof=ddof,
                
            )(x).shift(1)

        # Case 2: portfolio
        else:

            if weights is None:
                raise ValueError(
                    "weights must be provided when returns is a DataFrame"
                )

            if not isinstance(weights, pd.DataFrame):
                raise ValueError(
                    "weights must be a DataFrame when returns is a DataFrame"
                )

            if not returns.index.equals(weights.index):
                raise ValueError("returns and weights index mismatch")

            if not returns.columns.equals(weights.columns):
                raise ValueError("returns and weights columns mismatch")

            # Portfolio return
            x = (returns * weights).sum(axis=1)

            mu = (
                x.expanding(min_periods).mean().shift(1)
                if mean == "sample"
                else 0.0
            )

            sigma = ExpandingRisk(
                estimate_sigma,
                min_periods=min_periods,
                ddof=ddof,
            )(returns, weights=weights).shift(1)

        pit = pd.Series(np.nan, index=x.index)

        valid = (~sigma.isna()) & (sigma > 0)

        if not valid.any():
            return pit

        if mean == "sample":
            z = (x[valid] - mu[valid]) / sigma[valid]
        else:
            z = x[valid] / sigma[valid]

        if distribution == "normal":
            pit[valid] = norm.cdf(z)
        else:
            # Map unit-variance standardized residuals to canonical Student-t
            # scale before applying the t CDF.
            z *= np.sqrt(df / (df - 2))
            
            pit[valid] = t.cdf(z, df=df)

        return pit

    # Discrete case
    returns = returns.astype(float)

    if isinstance(returns, pd.DataFrame):
        raise ValueError(
            "Discrete PIT currently supports only Series inputs"
        )

    u = [np.nan] * len(returns)

    for t_idx in range(min_periods, len(returns)):
        hist = returns.iloc[:t_idx]

        u[t_idx] = randomized_pit(
            hist.to_numpy(),
            returns.iloc[t_idx],
        )

    return pd.Series(u, index=returns.index)
