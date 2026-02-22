from __future__ import annotations

from typing import Callable, Iterable, Any, Optional
from .var import var
import pandas as pd
from .es import es
import numpy as np


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

    def _apply(self, x: np.ndarray) -> float:
        """Apply risk function to window."""
        return self._func(x, **self._kwargs)

    @staticmethod
    def _to_series(
        returns: Iterable[float],
    ) -> pd.Series:
        """Ensure input is a pandas Series."""
        if isinstance(returns, pd.Series):
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
        returns: Iterable[float],
    ) -> pd.Series:
        """Compute rolling risk metric."""
        series = self._to_series(returns)

        return series.rolling(
            window=self._window,
            min_periods=self._min_periods,
        ).apply(self._apply, raw=True)


class ExpandingRisk(_BaseRisk):
    """Expanding risk metric engine."""

    def __call__(
        self,
        returns: Iterable[float],
    ) -> pd.Series:
        """Compute expanding risk metric."""
        series = self._to_series(returns)

        return series.expanding(
            min_periods=self._min_periods,
        ).apply(self._apply, raw=True)


def rolling_var(
    returns: Iterable[float],
    *,
    window: int,
    min_periods: Optional[int] = None,
    **kwargs: Any,
) -> pd.Series:
    """Rolling Value at Risk."""
    return RollingRisk(
        var,
        window=window,
        min_periods=min_periods,
        **kwargs,
    )(returns)


def rolling_es(
    returns: Iterable[float],
    *,
    window: int,
    min_periods: Optional[int] = None,
    **kwargs: Any,
) -> pd.Series:
    """Rolling Expected Shortfall."""
    return RollingRisk(
        es,
        window=window,
        min_periods=min_periods,
        **kwargs,
    )(returns)


def expanding_var(
    returns: Iterable[float],
    *,
    min_periods: Optional[int] = None,
    **kwargs: Any,
) -> pd.Series:
    """Expanding Value at Risk."""
    return ExpandingRisk(
        var,
        min_periods=min_periods,
        **kwargs,
    )(returns)


def expanding_es(
    returns: Iterable[float],
    *,
    min_periods: Optional[int] = None,
    **kwargs: Any,
) -> pd.Series:
    """Expanding Expected Shortfall."""
    return ExpandingRisk(
        es,
        min_periods=min_periods,
        **kwargs,
    )(returns)
