"""Hypothetical stress scenarios: the risk a statistical model cannot see.

VaR and ES only extrapolate from the past window. Regulators (Basel) also require stress
tests: plausible but severe shocks chosen by someone who knows the market. Each scenario
is an instantaneous return per ADR, applied to today's portfolio and compared with the
models' current VaR and ES.
"""

from __future__ import annotations

TICKERS = ["GGAL", "BMA", "BBAR", "YPF", "TGS", "EDN", "TEO", "CRESY"]


def escenarios_hipoteticos() -> dict[str, dict[str, float]]:
    """Return {scenario name: {ticker: one-day return}} for every ticker in TICKERS.

    TODO(tomas): define 2-4 scenarios from your knowledge of the Argentine market, e.g.
      - an electoral shock (how hard are banks hit vs energy vs utilities?),
      - a sovereign default / debt restructuring headline,
      - an oil price collapse (who is exposed: YPF, TGS, ... and who is not?),
      - a regulatory shock to utilities (tariff freeze: EDN, TGS).
    Returns are simple returns in USD, e.g. -0.35 for a 35% fall.
    """
    raise NotImplementedError
