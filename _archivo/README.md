# Archivo

Estudios exploratorios que quedaron fuera del foco del repo (el estudio de VaR).
No se mantienen ni se ejecutan en CI; los imports apuntan a módulos que se movieron acá.

- `02_arbitraje_ccl/`: arbitraje de CCL implícito ADR vs acción local (2003-2026).
  Hallazgo: Sharpe neto 1.29 en 2003-2015 con costo institucional, 0.33 (no significativo)
  desde 2016; negativo con costos minoristas. Incluye `argentina.py` (ratios ADR, CCL implícito)
  y `pairs.py` (regla de bandas).
- `03_backtest_event_driven/`: motor event-driven (MarketEvent → Signal → Order → Fill).
  Estado: 4 de 6 tests fallan por una diferencia de ~0.9% contra la referencia vectorizada;
  sin depurar. `ArgentineBrokerCost.fill` está sin implementar.
