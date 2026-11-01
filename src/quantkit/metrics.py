"""Performance metrics shared by all three projects, so numbers are comparable across them."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def cagr(returns: pd.Series) -> float:
    growth = (1 + returns).prod()
    years = len(returns) / TRADING_DAYS
    return growth ** (1 / years) - 1 if years > 0 and growth > 0 else np.nan


def ann_vol(returns: pd.Series) -> float:
    return returns.std(ddof=1) * np.sqrt(TRADING_DAYS)


def sharpe(returns: pd.Series, rf: float = 0.0) -> float:
    excess = returns - rf / TRADING_DAYS
    sd = excess.std(ddof=1)
    return excess.mean() / sd * np.sqrt(TRADING_DAYS) if sd > 0 else np.nan


def sortino(returns: pd.Series) -> float:
    downside = np.sqrt((returns.clip(upper=0) ** 2).mean())
    return returns.mean() / downside * np.sqrt(TRADING_DAYS) if downside > 0 else np.nan


def drawdown(returns: pd.Series) -> pd.Series:
    equity = (1 + returns).cumprod()
    return equity / equity.cummax() - 1


def max_drawdown(returns: pd.Series) -> float:
    return drawdown(returns).min()


def calmar(returns: pd.Series) -> float:
    mdd = max_drawdown(returns)
    return cagr(returns) / abs(mdd) if mdd < 0 else np.nan


def sharpe_tstat(returns: pd.Series) -> float:
    """t-statistic of the mean daily return (H0: true Sharpe = 0)."""
    return returns.mean() / returns.std(ddof=1) * np.sqrt(len(returns))


def summary(returns: pd.Series) -> dict[str, float]:
    returns = returns.dropna()
    return {
        "CAGR": cagr(returns),
        "Vol": ann_vol(returns),
        "Sharpe": sharpe(returns),
        "Sortino": sortino(returns),
        "MaxDD": max_drawdown(returns),
        "Calmar": calmar(returns),
        "t-stat": sharpe_tstat(returns),
    }


def summary_table(series: dict[str, pd.Series]) -> pd.DataFrame:
    return pd.DataFrame({name: summary(r) for name, r in series.items()}).T


def yearly_returns(returns: pd.Series) -> pd.Series:
    return returns.groupby(returns.index.year).apply(lambda r: (1 + r).prod() - 1)


def format_table(df: pd.DataFrame) -> str:
    """Markdown table with percentages where they belong, for pasting into READMEs."""
    pct = {"CAGR", "Vol", "MaxDD"}
    out = df.copy().astype(object)
    for col in df.columns:
        out[col] = [f"{v:.1%}" if col in pct else f"{v:.2f}" for v in df[col]]
    return out.to_markdown()
