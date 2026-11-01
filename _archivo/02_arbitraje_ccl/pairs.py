"""Mean-reversion band rule shared by spread strategies. Strictly causal."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_zscore(x: pd.Series, lookback: int) -> pd.Series:
    """z-score of x against its own trailing mean and std (window includes today)."""
    mu = x.rolling(lookback, min_periods=lookback).mean()
    sd = x.rolling(lookback, min_periods=lookback).std()
    return (x - mu) / sd


def positions(z: pd.Series, entry: float, exit: float, stop: float) -> pd.Series:
    """Spread position decided at each close: +1 long spread, -1 short spread, 0 flat.

    Enter when |z| crosses ``entry``, take profit when |z| falls below ``exit`` (or z
    crosses through zero), cut the trade when |z| exceeds ``stop``. After a stop, stay out
    until z returns inside the entry band, so we do not re-enter a broken relationship.
    Missing z (no quote that day) keeps the current state.
    """
    pos = np.zeros(len(z))
    state, locked = 0, False
    for i, v in enumerate(z.to_numpy()):
        if np.isnan(v):
            pos[i] = state
            continue
        if locked and abs(v) < entry:
            locked = False
        if state == 0 and not locked:
            if entry < v < stop:
                state = -1
            elif -stop < v < -entry:
                state = 1
        elif state != 0:
            if abs(v) < exit or np.sign(v) == state:
                state = 0
            elif abs(v) > stop:
                state, locked = 0, True
        pos[i] = state
    return pd.Series(pos, index=z.index)
