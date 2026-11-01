"""Argentine market helpers: ADR <-> local share mapping and the implied CCL exchange rate.

An ADR represents ``ratio`` local shares. Buying the local share in pesos and selling the
ADR in dollars (or vice versa) is how Argentines move money across the capital control
("contado con liquidacion", CCL). Each stock therefore implies its own exchange rate:

    CCL_i = local_price_ARS * ratio / ADR_price_USD

If markets were frictionless every CCL_i would be identical (law of one price).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantkit.data import load_prices

# ADR ticker -> (local ticker on Yahoo, local shares per ADR).
# YPFD.BA is quoted by Yahoo 10x smaller than on BYMA for its whole history, so its effective
# ratio is 10 instead of 1 (verified: constant -2.30 log deviation vs the market CCL, 2000-2026).
ADRS: dict[str, tuple[str, float]] = {
    "GGAL": ("GGAL.BA", 10),
    "BMA": ("BMA.BA", 10),
    "BBAR": ("BBAR.BA", 3),
    "SUPV": ("SUPV.BA", 5),
    "YPF": ("YPFD.BA", 10),
    "PAM": ("PAMP.BA", 25),
    "TGS": ("TGSU2.BA", 5),
    "CEPU": ("CEPU.BA", 10),
    "EDN": ("EDN.BA", 20),
    "LOMA": ("LOMA.BA", 5),
    "TEO": ("TECO2.BA", 5),
    "CRESY": ("CRES.BA", 10),
    "IRS": ("IRSA.BA", 10),
}

SECTOR = {
    "GGAL": "Bancos", "BMA": "Bancos", "BBAR": "Bancos", "SUPV": "Bancos",
    "YPF": "Energia", "PAM": "Energia", "TGS": "Energia", "CEPU": "Energia", "EDN": "Energia",
    "LOMA": "Materiales", "TEO": "Telecom", "CRESY": "Agro/Real estate", "IRS": "Real estate",
}


def load_listings(start: str = "2000-01-01", end: str = "2026-10-31") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Unadjusted closes for every ADR (USD) and its local share (ARS), each with its own history."""
    adrs = list(ADRS)
    locs = [ADRS[a][0] for a in adrs]
    usd = load_prices(adrs, start, end, adjusted=False, how="all")
    ars = load_prices(locs, start, end, adjusted=False, how="all")
    ars.columns = adrs  # index local prices by their ADR ticker
    return usd, ars


def implied_ccl(usd: pd.DataFrame, ars: pd.DataFrame) -> pd.DataFrame:
    """Implied CCL per stock; NaN where either listing has no price that day."""
    ratios = pd.Series({a: r for a, (_, r) in ADRS.items()})
    ccl = ars * ratios / usd
    return ccl.where(ccl > 0)


def market_ccl(ccl: pd.DataFrame, min_names: int = 4) -> pd.Series:
    """Cross-sectional median CCL: robust to one stale or mis-scaled listing."""
    med = ccl.median(axis=1)
    return med.where(ccl.notna().sum(axis=1) >= min_names)


def ccl_deviation(ccl: pd.DataFrame) -> pd.DataFrame:
    """log(CCL_i / median CCL): how expensive stock i's local share is vs its ADR, relative
    to the rest of the market. Positive = local share rich (or ADR cheap)."""
    return np.log(ccl.div(market_ccl(ccl), axis=0))
