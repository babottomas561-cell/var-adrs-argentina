"""The event-driven engine must agree with an independent vectorized implementation, and
must not be able to see the future."""

import numpy as np
import pandas as pd
import pytest

from quantkit.backtest import FixedBps, run_backtest, vectorized_reference


def synthetic_bars(n: int = 800, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, n)))
    open_ = close * np.exp(rng.normal(0, 0.005, n))  # gap vs previous close
    return pd.DataFrame({"X": open_}, idx), pd.DataFrame({"X": close}, idx)


def test_engine_matches_vectorized_reference_without_costs():
    opens, closes = synthetic_bars()
    res = run_backtest(opens, closes, fast=10, slow=40, costs=FixedBps(0))
    ref = vectorized_reference(opens["X"], closes["X"], 10, 40)
    engine = res.equity / res.equity.iloc[0]
    assert res.n_trades > 5
    np.testing.assert_allclose(engine.to_numpy(), ref.to_numpy(), rtol=1e-10)


def test_costs_only_reduce_equity_and_scale_with_turnover():
    opens, closes = synthetic_bars(seed=1)
    free = run_backtest(opens, closes, 10, 40, FixedBps(0)).equity.iloc[-1]
    cheap = run_backtest(opens, closes, 10, 40, FixedBps(10)).equity.iloc[-1]
    dear = run_backtest(opens, closes, 10, 40, FixedBps(50)).equity.iloc[-1]
    assert free > cheap > dear


def test_no_lookahead_future_shock_does_not_change_past_equity():
    """Changing prices after day k must leave the equity curve up to day k untouched."""
    opens, closes = synthetic_bars(seed=2)
    k = 500
    base = run_backtest(opens, closes, 10, 40, FixedBps(5)).equity
    o2, c2 = opens.copy(), closes.copy()
    o2.iloc[k + 1:] *= 3.0
    c2.iloc[k + 1:] *= 3.0
    shocked = run_backtest(o2, c2, 10, 40, FixedBps(5)).equity
    pd.testing.assert_series_equal(base.iloc[: k + 1], shocked.iloc[: k + 1])


@pytest.mark.parametrize("fast,slow", [(5, 20), (20, 100), (50, 200)])
def test_reference_agreement_across_parameters(fast, slow):
    opens, closes = synthetic_bars(n=1200, seed=fast)
    res = run_backtest(opens, closes, fast, slow, FixedBps(0))
    ref = vectorized_reference(opens["X"], closes["X"], fast, slow)
    np.testing.assert_allclose((res.equity / res.equity.iloc[0]).to_numpy(), ref.to_numpy(), rtol=1e-10)
