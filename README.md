# varlab

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white" alt="SciPy">
  <img src="https://img.shields.io/badge/Value%20at%20Risk-4B5563?style=for-the-badge" alt="Value at Risk">
  <img src="https://img.shields.io/badge/Expected%20Shortfall-111827?style=for-the-badge" alt="Expected Shortfall">
</p>

<p align="center">
  <i>
    A minimal Python package for Value at Risk, Expected
    Shortfall, and risk-based position scaling.
  </i>
</p>

`varlab` is a focused toolkit for market risk practitioners who want clean, transparent VaR/ES estimation and practical backtesting diagnostics without heavyweight framework overhead.


## 📦 Installation
Install from PyPI:

```bash
pip install varlab
```

## 🧰 What varlab Provides
`varlab` focuses on practical market risk workflows:

- Value at Risk (`var`)
- Expected Shortfall (`es`)
- Rolling and expanding estimators (`rolling_var`, `rolling_es`, `expanding_var`, `expanding_es`)
- VaR backtesting diagnostics with a concise ASCII report and structured outputs
- PIT-based distribution diagnostics (including randomized PIT for empirical/discrete cases)

Both VaR and ES support:

- `method="empirical"` (historical)
- `method="parametric"` with `distribution="normal"` or `distribution="t"`
- Optional exponential weighting in empirical mode via `lamb`
- 1-day and multi-day horizons (`n_days`)
- Optional sample mean component (`mean="sample"`)

## ⚡ Quick Start

```python
from varlab import rolling_var

var_estimates = rolling_var(
    returns,
    window=60,
    method="empirical",
    confidence=0.99,
    n_days=1,
    distribution="normal",
    mean="sample",
).shift(1).fillna(0)
```

## Parametric Example (Student-t)
If `df` is not provided, degrees of freedom are estimated by MLE.

```python
from varlab import var, es

var_t = var(
    returns,
    method="parametric",
    distribution="t",
    confidence=0.99,
    mean="sample",
)

es_t = es(
    returns,
    method="parametric",
    distribution="t",
    confidence=0.99,
    mean="sample",
)
```

## Exponentially Weighted Historical Risk
For empirical VaR/ES you can apply exponential weighting by setting `lamb`:

```python
from varlab import var, es

var_ew = var(returns, method="empirical", confidence=0.99, lamb=0.97)
es_ew = es(returns, method="empirical", confidence=0.99, lamb=0.97)
```

## Backtesting Suite
Run a full backtesting diagnostic:

```python
from varlab import backtesting

back_res = backtesting.diagnostic.run(
    returns=returns,
    exceedances=exceedances,
    confidence=0.99,
    window_type="rolling",
    pit_case="discrete",
    alpha=0.05,
    window=60,
)
```

Example report:

```text
==================================================================
                  VALUE AT RISK BACKTEST REPORT
==================================================================
Confidence level: 99.00% | Sample size: 4445 | Date: 2026-03-08
Overall result: FAIL

COVERAGE (4/4)
------------------------------------------------------------------
Exact Binomial                                          PASS
Kupiec Pof                                              PASS
Christoffersen Conditional                              PASS
Basel Traffic Light                                     PASS

DISTRIBUTION (0/3)
------------------------------------------------------------------
Uniformity                                              FAIL
Independence                                            FAIL
Berkowitz                                               FAIL

INDEPENDENCE (2/2)
------------------------------------------------------------------
Christoffersen                                          PASS
Loss Quantile                                           PASS
```

The diagnostic object is typed and machine-friendly, so you can inspect details programmatically:

```python
back_res.results['independence']["christoffersen"]
```

## Test Families Included
- Coverage:
  exact binomial, Kupiec POF, Christoffersen conditional coverage, Basel traffic light (when confidence is 99%)
- Distribution:
  PIT-based diagnostics (uniformity, independence, Berkowitz)
- Independence:
  Christoffersen exceedance independence, loss-quantile independence

## Probability Integral Transform (PIT)
Distribution diagnostics rely on PIT values:

- `pit_case="continuous"` for parametric settings
- `pit_case="discrete"` for empirical settings (randomized PIT is used)

Window logic can be:

- `window_type="rolling"`
- `window_type="expanding"`

## Public API

```python
from varlab import (
    var,
    es,
    rolling_var,
    rolling_es,
    expanding_var,
    expanding_es,
    backtesting,
)
```

## Notes
- Inputs can be plain arrays/lists or pandas objects.
- Risk values are returned as positive loss magnitudes.
- For multi-asset parametric workflows, pass a return matrix and portfolio weights.
- Shift rolling/expanding estimates by one step when using them for out-of-sample backtests.

## License
MIT © 2025 — Developed with ❤️ by Lorenzo Santarsieri
