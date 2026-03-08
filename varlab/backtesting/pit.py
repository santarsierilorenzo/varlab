"""
Probability Integral Transform (PIT) utilities.

This module provides functions to compute PIT values for financial return
series using either parametric continuous distributions or empirical discrete
distributions.

The implementation supports both rolling and expanding estimation windows and
can handle single assets or portfolios with time-varying weights.

Continuous PIT assumes that returns follow

    X_t = μ_t + sigma_t Z_t

where the standardized residuals Z_t follow either a Normal or Student-t
distribution. The conditional volatility sigma_t is estimated using rolling or
expanding windows through the RollingRisk and ExpandingRisk utilities.

Discrete PIT is implemented via the randomized PIT to correctly handle
empirical distributions with point masses.

The main entry point is `pit`, with `rolling_pit` and `expanding_pit` provided
as convenience wrappers.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple, Literal, Callable
from ..window import ExpandingRisk, RollingRisk
from ..base import estimate_sigma
from scipy.stats import norm, t
import pandas as pd
import numpy as np


Dist = Literal["normal", "t"]
Case = Literal["continuous", "discrete"]
MeanSpec = Literal["zero", "sample"]
WindowType = Literal["rolling", "expanding"]


def pit_checks(
    case: Case,
    distribution: Dist,
    df: Optional[int],
    window_type: WindowType,
) -> None:
    """
    Validate PIT configuration.

    Parameters
    ----------
    case : {"continuous", "discrete"}
        Type of PIT transformation.
    distribution : {"normal", "t"}
        Parametric distribution assumed for standardized residuals.
    df : Optional[int]
        Degrees of freedom for the Student-t distribution.
    window_type : {"rolling", "expanding"}
        Window scheme for PIT estimation.
        
    Raises
    ------
    ValueError
        If configuration parameters are invalid.
    """
    if case not in {"continuous", "discrete"}:
        raise ValueError("case must be either 'continuous' or 'discrete'")

    if distribution not in {"normal", "t"}:
        raise ValueError("distribution must be 'normal' or 't'")

    if distribution == "t" and (df is None or df <= 2):
        raise ValueError(
            "df must be provided and greater than 2 for t distribution"
        )

    if window_type not in {"rolling", "expanding"}:
        raise ValueError("window_type must be either 'rolling' or 'expanding'")
   

def randomized_pit(
    history: Iterable[float],
    x: float,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Randomized PIT for empirical discrete distributions.

    U = F(x^-) + V * P(X = x) where V ~ Uniform(0,1).

    Parameters
    ----------
    history : Iterable[float]
        Historical observations defining the empirical distribution.
    x : float
        Current observation.
    rng : Optional[np.random.Generator]
        Random number generator for reproducibility.

    Returns
    -------
    float
        Randomized PIT value.
    """
    hist = np.asarray(history, dtype=float)
    hist = hist[~np.isnan(hist)]

    n = hist.size
    if n == 0:
        raise ValueError("history must not be empty")

    less = np.sum(hist < x)
    equal = np.sum(np.isclose(hist, x))

    if rng is None:
        rng = np.random.default_rng()

    v = rng.uniform(0.0, 1.0)

    return float((less + v * equal) / n)


def _prepare_observation(
    returns: pd.Series | pd.DataFrame,
    weights: Optional[pd.DataFrame],
) -> Tuple[pd.Series, pd.Series | pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Prepare observation series and volatility estimation input.

    Parameters
    ----------
    returns : Series or DataFrame
        Asset returns.
    weights : Optional[DataFrame]
        Portfolio weights aligned with returns.

    Returns
    -------
    x : Series
        Observed return series (asset or portfolio).
    data : Series or DataFrame
        Data used for volatility estimation.
    weights : Optional[DataFrame]
        Weights passed through after validation.
    """
    returns = returns.astype(float)

    if isinstance(returns, pd.Series):
        if weights is not None and not isinstance(weights, pd.DataFrame):
            raise ValueError(
                "weights must be a DataFrame when provided"
            )
        return returns, returns, None

    if weights is None:
        raise ValueError(
            "weights must be provided when returns is a DataFrame"
        )

    if not isinstance(weights, pd.DataFrame):
        raise ValueError(
            "weights must be a DataFrame when returns is a DataFrame"
        )

    if not returns.index.equals(weights.index):
        raise ValueError(
            "returns and weights must share identical index"
        )

    if not returns.columns.equals(weights.columns):
        raise ValueError(
            "returns and weights must share identical columns"
        )

    x = (returns * weights).sum(axis=1)

    return x, returns, weights


def _compute_mu(
    x: pd.Series,
    mean: MeanSpec,
    window_type: WindowType,
    window_size: int,
) -> pd.Series | float:
    """
    Estimate conditional mean of returns.
    """
    if mean == "zero":
        return 0.0

    if window_type == "rolling":
        return x.rolling(window_size).mean().shift(1)

    return x.expanding(min_periods=window_size).mean().shift(1)


def _build_risk_estimator(
    window_type: WindowType,
    window_size: int,
    ddof: int,
) -> Callable[..., pd.Series]:
    """
    Build rolling or expanding volatility estimator.
    """
    if window_type == "rolling":
        return RollingRisk(
            estimate_sigma,
            window=window_size,
            ddof=ddof,
        )

    return ExpandingRisk(
        estimate_sigma,
        min_periods=window_size,
        ddof=ddof,
    )


def _compute_sigma(
    data: pd.Series | pd.DataFrame,
    weights: Optional[pd.DataFrame],
    window_type: WindowType,
    window_size: int,
    ddof: int,
) -> pd.Series:
    """
    Estimate conditional volatility using rolling or expanding window.
    """
    risk = _build_risk_estimator(window_type, window_size, ddof)

    if weights is None:
        sigma = risk(data)
    else:
        sigma = risk(data, weights=weights)

    return sigma.shift(1)


def _pit_from_residuals(
    z: pd.Series,
    distribution: Dist,
    df: Optional[int],
) -> pd.Series:
    """
    Map standardized residuals to PIT values.
    """
    if distribution == "normal":
        return pd.Series(norm.cdf(z), index=z.index, dtype=float)

    if df is None:
        raise RuntimeError("df must not be None for t distribution")

    # Map unit-variance standardized residuals to canonical Student-t
    z = z * np.sqrt(df / (df - 2))

    return pd.Series(t.cdf(z, df=df), index=z.index, dtype=float)


def _continuous_pit(
    returns: pd.Series | pd.DataFrame,
    weights: Optional[pd.DataFrame],
    distribution: Dist,
    df: Optional[int],
    window_type: WindowType,
    window_size: int,
    ddof: int,
    mean: MeanSpec,
) -> pd.Series:
    """
    Core PIT computation shared by rolling and expanding implementations.
    """
    x, data, weights = _prepare_observation(returns, weights)

    mu = _compute_mu(x, mean, window_type, window_size)
    sigma = _compute_sigma(
        data,
        weights,
        window_type,
        window_size,
        ddof,
    )

    pit = pd.Series(np.nan, index=x.index, dtype=float)

    valid = (~sigma.isna()) & (sigma > 0)

    if mean == "sample":
        valid &= ~mu.isna()

    if not valid.any():
        return pit

    if mean == "sample":
        z = (x[valid] - mu[valid]) / sigma[valid]
    else:
        z = x[valid] / sigma[valid]

    pit[valid] = _pit_from_residuals(z, distribution, df)

    return pit


def _discrete_pit(
    returns: pd.Series | pd.DataFrame,
    window_type: WindowType,
    window_size: int,
    rng: Optional[np.random.Generator] = None,
) -> pd.Series:
    """
    Compute randomized PIT values for empirical discrete distributions.
    """
    returns = returns.astype(float)

    if isinstance(returns, pd.DataFrame):
        raise ValueError(
            "Discrete PIT currently supports only Series inputs"
        )

    values = returns.to_numpy(dtype=float)
    u = np.full(len(values), np.nan, dtype=float)

    for t_idx in range(window_size, len(values)):
        if window_type == "rolling":
            hist = values[t_idx - window_size:t_idx]
        else:
            hist = values[:t_idx]

        u[t_idx] = randomized_pit(hist, values[t_idx], rng=rng)

    return pd.Series(u, index=returns.index, dtype=float)


def pit(
    returns: pd.Series | pd.DataFrame,
    case: Case = "continuous",
    weights: Optional[pd.DataFrame] = None,
    distribution: Dist = "normal",
    df: Optional[int] = None,
    window_type: WindowType = "rolling",
    window_size: Optional[int] = None,
    window: Optional[int] = None,
    min_periods: Optional[int] = None,
    ddof: int = 1,
    mean: MeanSpec = "zero",
    rng: Optional[np.random.Generator] = None,
) -> pd.Series:
    """
    Compute PIT values using rolling or expanding windows.

    Continuous PIT assumes a parametric distribution for standardized
    residuals. Discrete PIT uses the randomized PIT for empirical
    distributions.

    Parameters
    ----------
    returns : Series or DataFrame
        Asset or panel of asset returns.
    case : {"continuous", "discrete"}
        PIT type.
    weights : Optional[DataFrame]
        Portfolio weights when returns is a DataFrame.
    distribution : {"normal", "t"}
        Parametric distribution for residuals.
    df : Optional[int]
        Degrees of freedom for Student-t.
    window_type : {"rolling", "expanding"}
        Window scheme used for estimation.
    window_size : Optional[int]
        Unified window parameter. For rolling it is the window length,
        for expanding it is the minimum number of observations.
    window : Optional[int]
        Alias for rolling window length. Kept for convenience and
        backward compatibility.
    min_periods : Optional[int]
        Alias for expanding minimum number of observations. Kept for
        convenience and backward compatibility.
    ddof : int
        Degrees of freedom used in volatility estimation.
    mean : {"zero", "sample"}
        Conditional mean specification.
    rng : Optional[np.random.Generator]
        Random generator for discrete PIT.

    Returns
    -------
    Series
        PIT values aligned with input index.

    Raises
    ------
    ValueError
        If PIT parameters are invalid or window parameters are
        inconsistent.
    """
    pit_checks(case, distribution, df, window_type)

    params = [window_size, window, min_periods]
    n_specified = sum(param is not None for param in params)

    if n_specified == 0:
        window_size = 60 if window_type == "rolling" else 30
    elif n_specified == 1:
        window_size = next(
            param for param in params if param is not None
        )
    else:
        raise ValueError(
            "Specify only one among window_size, window, min_periods"
        )

    if window_size is None or window_size < 1:
        raise ValueError("window size must be >= 1")

    if case == "continuous":
        return _continuous_pit(
            returns=returns,
            weights=weights,
            distribution=distribution,
            df=df,
            window_type=window_type,
            window_size=window_size,
            ddof=ddof,
            mean=mean,
        )

    return _discrete_pit(
        returns=returns,
        window_type=window_type,
        window_size=window_size,
        rng=rng,
    )


def rolling_pit(
    returns: pd.Series | pd.DataFrame,
    case: Case = "continuous",
    weights: Optional[pd.DataFrame] = None,
    distribution: Dist = "normal",
    df: Optional[int] = None,
    window: int = 60,
    ddof: int = 1,
    mean: MeanSpec = "zero",
    rng: Optional[np.random.Generator] = None,
) -> pd.Series:
    """
    Compute rolling-window PIT values.

    Continuous PIT assumes a parametric distribution for standardized
    residuals. Discrete PIT uses the randomized PIT for empirical
    distributions.

    Parameters
    ----------
    returns : Series or DataFrame
        Asset or panel of asset returns.
    case : {"continuous", "discrete"}
        PIT type.
    weights : Optional[DataFrame]
        Portfolio weights when returns is a DataFrame.
    distribution : {"normal", "t"}
        Parametric distribution for residuals.
    df : Optional[int]
        Degrees of freedom for Student-t.
    window : int
        Rolling window length.
    ddof : int
        Degrees of freedom used in volatility estimation.
    mean : {"zero", "sample"}
        Conditional mean specification.
    rng : Optional[np.random.Generator]
        Random generator for discrete PIT.

    Returns
    -------
    Series
        PIT values aligned with input index.
    """
    return pit(
        returns=returns,
        case=case,
        weights=weights,
        distribution=distribution,
        df=df,
        window_type="rolling",
        window_size=window,
        ddof=ddof,
        mean=mean,
        rng=rng,
    )


def expanding_pit(
    returns: pd.Series | pd.DataFrame,
    case: Case = "continuous",
    weights: Optional[pd.DataFrame] = None,
    distribution: Dist = "normal",
    df: Optional[int] = None,
    min_periods: int = 30,
    ddof: int = 1,
    mean: MeanSpec = "zero",
    rng: Optional[np.random.Generator] = None,
) -> pd.Series:
    """
    Compute expanding-window PIT values.

    Parameters
    ----------
    returns : Series or DataFrame
        Asset or panel of asset returns.
    case : {"continuous", "discrete"}
        PIT type.
    weights : Optional[DataFrame]
        Portfolio weights when returns is a DataFrame.
    distribution : {"normal", "t"}
        Parametric distribution for residuals.
    df : Optional[int]
        Degrees of freedom for Student-t.
    min_periods : int
        Minimum observations required for estimation.
    ddof : int
        Degrees of freedom used in volatility estimation.
    mean : {"zero", "sample"}
        Conditional mean specification.
    rng : Optional[np.random.Generator]
        Random generator for discrete PIT.

    Returns
    -------
    Series
        PIT values aligned with input index.
    """
    return pit(
        returns=returns,
        case=case,
        weights=weights,
        distribution=distribution,
        df=df,
        window_type="expanding",
        window_size=min_periods,
        ddof=ddof,
        mean=mean,
        rng=rng,
    )
