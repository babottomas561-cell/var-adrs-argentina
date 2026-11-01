"""A small event-driven backtester.

The loop mirrors how a live system works, which is what makes look-ahead hard to commit:

    DataHandler --MarketEvent--> Strategy --SignalEvent--> Portfolio --OrderEvent-->
    ExecutionHandler --FillEvent--> Portfolio

Signals are computed from bars up to today's close; orders generated tonight are filled at
tomorrow's open, through a pluggable cost model.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------- events

@dataclass(frozen=True)
class MarketEvent:
    date: pd.Timestamp


@dataclass(frozen=True)
class SignalEvent:
    date: pd.Timestamp
    ticker: str
    direction: str  # "LONG" or "EXIT"


@dataclass(frozen=True)
class OrderEvent:
    date: pd.Timestamp
    ticker: str
    side: str       # "BUY" or "SELL"
    quantity: float | None  # None = size at fill time (target value / fill price)
    target_value: float = 0.0


@dataclass(frozen=True)
class FillEvent:
    date: pd.Timestamp
    ticker: str
    side: str
    quantity: float
    price: float
    commission: float


# ----------------------------------------------------------------------------- costs

class CostModel:
    """Turns a market order at a reference price into (fill price, commission)."""

    def fill(self, side: str, quantity: float, price: float) -> tuple[float, float]:
        raise NotImplementedError


class FixedBps(CostModel):
    """All-in cost as a fixed fraction of traded notional, each way."""

    def __init__(self, bps: float):
        self.bps = bps

    def fill(self, side: str, quantity: float, price: float) -> tuple[float, float]:
        return price, abs(quantity) * price * self.bps / 1e4


class ArgentineBrokerCost(CostModel):
    """Cost of trading the same shares through an Argentine broker on BYMA.

    A typical retail ticket pays, on traded notional:
      - broker commission (e.g. 0.50%), with a minimum per ticket in some brokers,
      - market fees ("derechos de mercado", ~0.08%),
      - VAT (21%) on the commission and the market fees,
      - plus the bid-ask spread, which on mid caps can be wider than the fees.

    Prices here are in USD (ADR prices as a proxy for local shares at the CCL), so a
    minimum ticket must be expressed in USD too.
    """

    def __init__(self, commission_pct: float = 0.50, market_fee_pct: float = 0.08,
                 vat: float = 0.21, min_ticket_usd: float = 0.0, half_spread_bps: float = 10.0):
        self.commission_pct = commission_pct
        self.market_fee_pct = market_fee_pct
        self.vat = vat
        self.min_ticket_usd = min_ticket_usd
        self.half_spread_bps = half_spread_bps

    def fill(self, side: str, quantity: float, price: float) -> tuple[float, float]:
        # TODO(tomas): return (fill_price, commission) for one market order.
        #   - fill_price: the reference price moved against us by half the spread
        #     (buys pay more, sells receive less).
        #   - commission: broker commission (respecting min_ticket_usd) + market fees,
        #     both grossed up by VAT, on the traded notional at the fill price.
        raise NotImplementedError


# ----------------------------------------------------------------------------- components

class DataHandler:
    """Feeds bars one date at a time and never exposes data after the current date."""

    def __init__(self, opens: pd.DataFrame, closes: pd.DataFrame):
        self.opens, self.closes = opens, closes
        self.dates = closes.index
        self.tickers = list(closes.columns)
        self._i = -1

    def update(self) -> MarketEvent | None:
        self._i += 1
        if self._i >= len(self.dates):
            return None
        return MarketEvent(self.dates[self._i])

    @property
    def today(self) -> pd.Timestamp:
        return self.dates[self._i]

    def history(self, ticker: str) -> np.ndarray:
        """Closes up to and including today (NaNs, e.g. before listing, removed)."""
        h = self.closes[ticker].to_numpy()[: self._i + 1]
        return h[~np.isnan(h)]

    def open(self, ticker: str) -> float:
        return self.opens.at[self.today, ticker]

    def close(self, ticker: str) -> float:
        return self.closes.at[self.today, ticker]


class MovingAverageCross:
    """Long when the fast SMA is above the slow SMA, flat otherwise."""

    def __init__(self, data: DataHandler, fast: int, slow: int):
        self.data, self.fast, self.slow = data, fast, slow
        self.state = {t: "FLAT" for t in data.tickers}

    def on_market(self, event: MarketEvent) -> list[SignalEvent]:
        signals = []
        for t in self.data.tickers:
            h = self.data.history(t)
            if len(h) < self.slow or np.isnan(self.data.close(t)):
                continue
            up = h[-self.fast:].mean() > h[-self.slow:].mean()
            if up and self.state[t] == "FLAT":
                signals.append(SignalEvent(event.date, t, "LONG"))
                self.state[t] = "LONG"
            elif not up and self.state[t] == "LONG":
                signals.append(SignalEvent(event.date, t, "EXIT"))
                self.state[t] = "FLAT"
        return signals


class Portfolio:
    """Equal slots: each ticker may hold at most 1/N of equity (N = universe size)."""

    def __init__(self, data: DataHandler, initial_capital: float = 100_000.0):
        self.data = data
        self.cash = initial_capital
        self.holdings = {t: 0.0 for t in data.tickers}
        self.last_price = {t: np.nan for t in data.tickers}
        self.slot = 1 / len(data.tickers)
        self.equity_curve: list[tuple[pd.Timestamp, float]] = []
        self.fills: list[FillEvent] = []

    def equity(self) -> float:
        pos = sum(q * self.last_price[t] for t, q in self.holdings.items() if q)
        return self.cash + pos

    def on_signal(self, s: SignalEvent) -> OrderEvent | None:
        if s.direction == "LONG" and self.holdings[s.ticker] == 0:
            return OrderEvent(s.date, s.ticker, "BUY", None, target_value=self.equity() * self.slot)
        if s.direction == "EXIT" and self.holdings[s.ticker] > 0:
            return OrderEvent(s.date, s.ticker, "SELL", self.holdings[s.ticker])
        return None

    def on_fill(self, f: FillEvent) -> None:
        sign = 1 if f.side == "BUY" else -1
        self.holdings[f.ticker] += sign * f.quantity
        self.cash -= sign * f.quantity * f.price + f.commission
        self.last_price[f.ticker] = f.price
        self.fills.append(f)

    def mark_to_market(self, date: pd.Timestamp) -> None:
        for t in self.data.tickers:
            c = self.data.close(t)
            if not np.isnan(c):
                self.last_price[t] = c
        self.equity_curve.append((date, self.equity()))


class SimulatedExecution:
    """Fills yesterday's orders at today's open through the cost model."""

    def __init__(self, data: DataHandler, costs: CostModel):
        self.data, self.costs = data, costs
        self.pending: list[OrderEvent] = []

    def queue(self, order: OrderEvent) -> None:
        self.pending.append(order)

    def execute_pending(self) -> list[FillEvent]:
        fills, keep = [], []
        for o in self.pending:
            ref = self.data.open(o.ticker)
            if np.isnan(ref):  # no trade today in this name: try again tomorrow
                keep.append(o)
                continue
            qty = o.quantity if o.quantity is not None else o.target_value / ref
            price, comm = self.costs.fill(o.side, qty, ref)
            if o.quantity is None:  # buy sized on the reference; re-size so cost fits the slot
                qty = o.target_value / (price + comm / qty)
                price, comm = self.costs.fill(o.side, qty, ref)
            fills.append(FillEvent(self.data.today, o.ticker, o.side, qty, price, comm))
        self.pending = keep
        return fills


# ----------------------------------------------------------------------------- engine

@dataclass
class BacktestResult:
    equity: pd.Series
    fills: list[FillEvent] = field(repr=False)

    @property
    def returns(self) -> pd.Series:
        return self.equity.pct_change().fillna(0.0)

    @property
    def n_trades(self) -> int:
        return sum(1 for f in self.fills if f.side == "SELL")

    @property
    def total_costs(self) -> float:
        return sum(f.commission for f in self.fills)


def run_backtest(opens: pd.DataFrame, closes: pd.DataFrame, fast: int, slow: int,
                 costs: CostModel, initial_capital: float = 100_000.0) -> BacktestResult:
    data = DataHandler(opens, closes)
    strategy = MovingAverageCross(data, fast, slow)
    portfolio = Portfolio(data, initial_capital)
    broker = SimulatedExecution(data, costs)
    events: deque = deque()

    while (market := data.update()) is not None:
        events.append(market)
        while events:
            e = events.popleft()
            if isinstance(e, MarketEvent):
                events.extend(broker.execute_pending())  # last night's orders, at today's open
                portfolio.mark_to_market(e.date)          # value the book at today's close
                events.extend(strategy.on_market(e))      # signals from today's close
            elif isinstance(e, SignalEvent):
                if (order := portfolio.on_signal(e)) is not None:
                    events.append(order)
            elif isinstance(e, OrderEvent):
                broker.queue(e)
            elif isinstance(e, FillEvent):
                portfolio.on_fill(e)

    eq = pd.Series(dict(portfolio.equity_curve))
    return BacktestResult(eq, portfolio.fills)


def vectorized_reference(opens: pd.Series, closes: pd.Series, fast: int, slow: int) -> pd.Series:
    """Independent, cost-free implementation for a single asset with the full equity in
    the slot. Used only to verify the event-driven engine produces identical equity."""
    target = (closes.rolling(fast).mean() > closes.rolling(slow).mean()).astype(float)
    target[closes.rolling(slow).count() < slow] = 0.0
    held_prev, held = target.shift(2).fillna(0), target.shift(1).fillna(0)
    r = pd.Series(0.0, index=closes.index)
    stay = (held_prev == 1) & (held == 1)
    enter = (held_prev == 0) & (held == 1)
    leave = (held_prev == 1) & (held == 0)
    r[stay] = (closes / closes.shift(1) - 1)[stay]
    r[enter] = (closes / opens - 1)[enter]
    r[leave] = (opens / closes.shift(1) - 1)[leave]
    return (1 + r).cumprod()
