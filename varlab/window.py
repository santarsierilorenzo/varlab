from __future__ import annotations

from typing import Callable, Iterable, Any, Optional
from .var import var
import pandas as pd
import numpy as np
from .es import es


RiskFunc = Callable[..., float]


class _BaseRisk:
    """Base class for window-based risk metrics."""

    def __init__(
        self,
        func: RiskFunc,
        *,
        min_periods: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self._func: RiskFunc = func
        self._min_periods: Optional[int] = min_periods
        self._kwargs: dict[str, Any] = kwargs

    def _apply(
        self,
        x: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> float:
        """
        Apply the risk function to a window.

        Parameters
        ----------
        x : np.ndarray
            Window of returns.
        weights : np.ndarray, optional
            Portfolio weights corresponding to the last observation
            in the window (time-varying weights).

        Returns
        -------
        float
            Risk metric value.
        """
        if weights is None:
            return self._func(x, **self._kwargs)

        return self._func(x, weights=weights, **self._kwargs)

    @staticmethod
    def _to_pandas(
        returns: Iterable[float] | pd.Series | pd.DataFrame,
    ) -> pd.Series | pd.DataFrame:
        """Ensure input is a pandas object."""
        if isinstance(returns, (pd.Series, pd.DataFrame)):
            return returns
        return pd.Series(returns)


class RollingRisk(_BaseRisk):
    """Rolling risk metric engine."""

    def __init__(
        self,
        func: RiskFunc,
        *,
        window: int,
        min_periods: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        if window <= 0:
            raise ValueError("window must be positive.")

        super().__init__(
            func,
            min_periods=min_periods or window,
            **kwargs,
        )

        self._window: int = window

    def __call__(
        self,
        returns: Iterable[float] | pd.Series | pd.DataFrame,
        *,
        weights: Optional[pd.Series | pd.DataFrame] = None,
    ) -> pd.Series:
        """
        Compute rolling risk metric.

        This method intentionally replicates the behavior of
        ``pandas.Series.rolling`` / ``pandas.DataFrame.rolling`` but
        implements the window iteration explicitly.

        The explicit loop is required to support:

        - multi-asset return matrices
        - time-varying portfolio weights
        - arbitrary risk functions requiring both returns and weights

        For each timestamp ``t`` the risk metric is computed using the
        window:

            returns[t-window+1 : t]

        and the portfolio weights observed at time ``t``.

        Parameters
        ----------
        returns : Series or DataFrame
            Asset returns.
        weights : Series or DataFrame, optional
            Time-varying portfolio weights.

        Returns
        -------
        pd.Series
            Rolling risk metric.
        """
        data = self._to_pandas(returns)

        index = data.index
        n_obs = len(data)

        result = np.full(n_obs, np.nan)

        for t in range(n_obs):
            if self._min_periods is not None and t < self._min_periods - 1:
                continue

            start = max(0, t - self._window + 1)
            window = data.iloc[start:t + 1]

            window_values = window.to_numpy()

            w = None
            if weights is not None:
                if isinstance(weights, (pd.Series, pd.DataFrame)):
                    w = weights.iloc[t].to_numpy()
                else:
                    w = np.asarray(weights)

            result[t] = self._apply(window_values, w)

        return pd.Series(result, index=index)


class ExpandingRisk(_BaseRisk):
    """Expanding risk metric engine."""

    def __call__(
        self,
        returns: Iterable[float] | pd.Series | pd.DataFrame,
        *,
        weights: Optional[pd.Series | pd.DataFrame] = None,
    ) -> pd.Series:
        """
        Compute expanding risk metric.

        This method replicates the behavior of ``pandas.expanding`` but
        performs the window expansion manually.

        The expanding window grows over time:

            returns[0 : t]

        As with the rolling engine, the implementation uses an explicit
        loop to support multi-asset returns and time-varying weights.

        Parameters
        ----------
        returns : Series or DataFrame
            Asset returns.
        weights : Series or DataFrame, optional
            Time-varying portfolio weights.

        Returns
        -------
        pd.Series
            Expanding risk metric.
        """
        data = self._to_pandas(returns)

        index = data.index
        n_obs = len(data)

        result = np.full(n_obs, np.nan)

        for t in range(n_obs):
            if self._min_periods is not None and t < self._min_periods - 1:
                continue

            window = data.iloc[: t + 1]
            window_values = window.to_numpy()

            w = None
            if weights is not None:
                if isinstance(weights, pd.DataFrame):
                    w = weights.iloc[t].to_numpy()
                elif isinstance(weights, pd.Series):
                    w = np.asarray([weights.iloc[t]])
                else:
                    w = np.asarray(weights)

            result[t] = self._apply(window_values, w)

        return pd.Series(result, index=index)


def rolling_var(
    returns: Iterable[float] | pd.Series | pd.DataFrame,
    *,
    window: int,
    min_periods: Optional[int] = None,
    weights: Optional[pd.Series | pd.DataFrame] = None,
    **kwargs: Any,
) -> pd.Series:
    """Rolling Value at Risk."""
    return RollingRisk(
        var,
        window=window,
        min_periods=min_periods,
        **kwargs,
    )(returns, weights=weights)


def rolling_es(
    returns: Iterable[float] | pd.Series | pd.DataFrame,
    *,
    window: int,
    min_periods: Optional[int] = None,
    weights: Optional[pd.Series | pd.DataFrame] = None,
    **kwargs: Any,
) -> pd.Series:
    """Rolling Expected Shortfall."""
    return RollingRisk(
        es,
        window=window,
        min_periods=min_periods,
        **kwargs,
    )(returns, weights=weights)


def expanding_var(
    returns: Iterable[float] | pd.Series | pd.DataFrame,
    *,
    min_periods: Optional[int] = None,
    weights: Optional[pd.Series | pd.DataFrame] = None,
    **kwargs: Any,
) -> pd.Series:
    """Expanding Value at Risk."""
    return ExpandingRisk(
        var,
        min_periods=min_periods,
        **kwargs,
    )(returns, weights=weights)


def expanding_es(
    returns: Iterable[float] | pd.Series | pd.DataFrame,
    *,
    min_periods: Optional[int] = None,
    weights: Optional[pd.Series | pd.DataFrame] = None,
    **kwargs: Any,
) -> pd.Series:
    """Expanding Expected Shortfall."""
    return ExpandingRisk(
        es,
        min_periods=min_periods,
        **kwargs,
    )(returns, weights=weights)
