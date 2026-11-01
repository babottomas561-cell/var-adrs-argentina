"""Price loading with a local CSV cache so every result in the repo is reproducible offline.

The first run downloads from Yahoo Finance and freezes the data in ``data/``; later runs
read the frozen file. Delete the file to refresh. Freezing matters: Yahoo revises adjusted
prices after dividends and splits, which would silently change published numbers.
"""

from __future__ import annotations

import hashlib
import warnings
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _cache_path(tickers: list[str], start: str, end: str | None, field: str, adjusted: bool) -> Path:
    key = f"{','.join(sorted(tickers))}|{start}|{end}|{field}|{adjusted}"
    digest = hashlib.sha1(key.encode()).hexdigest()[:10]
    return DATA_DIR / f"{field.lower()}_{'adj' if adjusted else 'raw'}_{digest}.csv"


def load_prices(
    tickers: list[str],
    start: str = "2005-01-01",
    end: str | None = "2026-10-31",
    field: str = "Close",
    adjusted: bool = True,
    how: str = "any",
) -> pd.DataFrame:
    """Daily prices from Yahoo Finance, one column per ticker.

    ``adjusted=True`` returns dividend- and split-adjusted prices (for returns);
    ``adjusted=False`` returns the prices actually printed on the exchange (needed when
    two listings must be compared at the same moment, e.g. an ADR vs its local share).
    ``how="any"`` keeps only days where every ticker traded (a rectangular panel);
    ``how="all"`` keeps every day with at least one price.
    """
    path = _cache_path(tickers, start, end, field, adjusted)
    if path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True)[tickers]

    import yfinance as yf

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(tickers, start=start, end=end, auto_adjust=adjusted, progress=False)
    prices = raw[field].reindex(columns=tickers).dropna(how=how)
    missing = [t for t in tickers if prices[t].notna().sum() < 250]
    if missing:  # never freeze a partial download (Yahoo times out intermittently)
        raise RuntimeError(f"Download incomplete for {missing}; re-run to retry.")
    DATA_DIR.mkdir(exist_ok=True)
    prices.to_csv(path)
    return prices


def load_ohlcv(ticker: str, start: str = "2005-01-01", end: str | None = "2026-10-31") -> pd.DataFrame:
    """Adjusted OHLCV bars for a single ticker (columns: Open, High, Low, Close, Volume)."""
    fields = ["Open", "High", "Low", "Close", "Volume"]
    cols = {f: load_prices([ticker], start, end, field=f)[ticker] for f in fields}
    return pd.DataFrame(cols).dropna()


def load_panel(tickers: list[str], start: str = "2000-01-01", end: str | None = "2026-10-31",
               adjusted: bool = True) -> dict[str, pd.DataFrame]:
    """Open/High/Low/Close panels for many tickers, keeping each ticker's own history."""
    return {f: load_prices(tickers, start, end, field=f, adjusted=adjusted, how="all")
            for f in ["Open", "High", "Low", "Close"]}
