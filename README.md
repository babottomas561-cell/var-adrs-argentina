# ¿Qué modelo de VaR sobrevive al mercado argentino?

**Backtest fuera de muestra de 7 modelos de Value-at-Risk sobre una cartera de ADRs argentinos, 2009-2026.**

![Abanico de 10.000 trayectorias simuladas a un año](estudio/figures/01_abanico.png)

## Resumen

- **Pregunta.** Un banco o un fondo que mide el riesgo de una cartera argentina con el Monte Carlo de manual (retornos normales, covarianza muestral) ¿está midiendo bien el riesgo?
- **Respuesta: no.** Sobre 4.410 días fuera de muestra, el VaR 99% del Monte Carlo normal falló 63 veces contra 44 esperadas (Kupiec p = 0,007), con las fallas agrupadas en crisis (cobertura condicional p < 0,001). Un banco con este modelo habría pasado el 8% de los días en **zona roja** del semáforo de Basilea.
- **Qué funciona.** Cuatro modelos aprueban los cuatro tests (frecuencia, independencia, cobertura condicional y el test Z2 del Expected Shortfall): **Monte Carlo t-Student con covarianza EWMA, simulación histórica filtrada, GARCH(1,1)-t y EVT filtrado**. Lo que tienen en común: **colas gruesas + volatilidad que reacciona**.
- **Lo que ningún modelo ve.** Las PASO de 2019 (−47% la cartera en un día, 18 desvíos estándar). El mejor pronóstico de la noche anterior fue 14%. Para eso existen el ES y los escenarios de estrés, no el VaR.
- **El ruido de Monte Carlo es lo de menos.** Con 5.000 simulaciones el error de simulación es ~4,5% del VaR; la diferencia entre modelos el mismo día es ~43%. Elegir bien el modelo importa 10 veces más que simular más.
- **Antes de los modelos, los datos.** Los días extremos de la cartera son elecciones y decisiones de política, no crisis globales. Además, los 8 ADRs caen juntos 6 veces más seguido de lo que permite una normal. Ver el [análisis descriptivo](#análisis-descriptivo-de-los-datos).

## 1. Hipótesis

> **H0:** un VaR 99% a 1 día calculado por Monte Carlo con retornos normales multivariados y covarianza muestral de 500 ruedas tiene una tasa de excepciones del 1%, sin agrupamiento, sobre una cartera equiponderada de ADRs argentinos.

Se rechaza si el test de cobertura condicional de Christoffersen da p < 5%. Antes de mirar resultados se fijaron también las alternativas a comparar, la ventana, el λ de EWMA, la cantidad de simulaciones y la semilla (ver [METODOLOGIA.md](METODOLOGIA.md)).

## 2. Datos

| | |
|---|---|
| Activos | GGAL, BMA, BBAR, YPF, TGS, EDN, TEO, CRESY (ADRs en NYSE/Nasdaq) |
| Cartera | Partes iguales, rebalanceo diario, en USD |
| Fuente | Yahoo Finance, precios ajustados, congelados a CSV en `data/` |
| Muestra | 2007-04-27 a 2026-10-30 (4.910 ruedas) |
| Fuera de muestra | 2009-04-22 a 2026-10-30 (4.410 ruedas, 44 excepciones esperadas al 99%) |

**Por qué ADRs y no acciones locales:** cotizan en dólares, en un mercado líquido y sin los saltos de precio que introducen el cepo y los cambios de régimen cambiario en el peso. El riesgo que medimos es el que ve un inversor en dólares.

**Sesgos que hay que nombrar:**
- *Supervivencia:* son las empresas que hoy siguen listadas. Un universo con las que dejaron de cotizar sería más riesgoso, así que el VaR de este estudio es, si algo, optimista.
- *Selección:* la cartera la elegí yo, por liquidez y representatividad sectorial (bancos, energía, utilities, telecom, real estate). No se probaron otras carteras para elegir la "mejor".

![Tres hechos estilizados](estudio/figures/02_hechos_estilizados.png)

Tres hechos de los datos que anticipan el resultado:
1. **Colas gruesas.** Curtosis en exceso de la cartera: 30 (la normal tiene 0). Cada ADR individual, entre 14 y 18.
2. **Volatilidad en racimos.** Los retornos casi no tienen autocorrelación, pero su valor absoluto sí, y durante más de 60 ruedas.
3. **La diversificación desaparece en crisis.** La correlación promedio entre ADRs es 0,70 en el 20% de ruedas más volátiles vs 0,44 en el resto. Un modelo con covarianza constante sobreestima la diversificación justo cuando importa.

![Colas](estudio/figures/03_colas.png)

### Análisis descriptivo de los datos

> Resumen del capítulo [**Análisis descriptivo de las variables**](estudio/descriptivo/ANALISIS_DESCRIPTIVO.md). El capítulo tiene unas 1.500 líneas:
> - marco teórico;
> - contexto macro de 2007 a 2026;
> - una ficha por variable;
> - una sección por cada ADR;
> - 28 figuras y 49 tablas.
>
> Se regenera con `uv run python estudio/descriptivo/run_descriptivo.py`.

#### Las variables

| Variable | Qué es | Por qué |
|---|---|---|
| Precio ajustado de cada ADR | Cierre diario en USD, ajustado por dividendos y splits | Variable primaria; en dólares y sin las distorsiones del cepo |
| Retorno simple diario $r_t = P_t/P_{t-1} - 1$ | Variación diaria del precio | Solo el retorno simple se agrega entre activos. En logaritmos, el −47% de las PASO sería −64% |
| Retorno de la cartera | Promedio de los 8 retornos, con rebalanceo diario | Es la variable objetivo: el VaR que se backtestea |
| S&P 500 (SPY) | Mismo cálculo | Control: separa lo argentino de lo global |
| Volatilidad EWMA (λ = 0,94) | Promedio exponencial de retornos al cuadrado | Es la pieza que hace reaccionar a los modelos |
| VaR 99% y excepción | Pronóstico de cada modelo; 1 si la pérdida lo supera | Es la variable que se evalúa |

#### Cómo evolucionaron los precios

![Precios en base 100](estudio/descriptivo/figuras/d01_precios_base100.png)

Ningún ADR tuvo una tendencia estable. Alternan subas de varias veces su valor con caídas de 80% a 95%. Hay tres resultados en 19 años, en dólares:

| | Valor final (base 100) | Crecimiento anual | Volatilidad anual | Caída máxima |
|---|---:|---:|---:|---:|
| Cartera | 692 | 10,4% | 42,4% | −86,3% |
| S&P 500 | 722 | 10,7% | 19,6% | −55,2% |
| EDN | 89 | −0,6% | 63,7% | −95,5% |
| CRESY | 89 | −0,6% | 48,6% | −89,4% |

**El riesgo no se pagó.** La cartera terminó en el mismo lugar que el S&P 500, con 2,2 veces su volatilidad. Pasó 1.678 ruedas, unos 6,7 años, bajo su máximo de enero de 2018.

![Retornos anuales](estudio/descriptivo/figuras/d14_retornos_anuales.png)

Los años argentinos se mueven en bloque y no siguen el ciclo global:

| Año | Cartera | S&P 500 |
|---|---:|---:|
| 2019 | −38% | +31% |
| 2022 | +62% | −18% |
| 2024 | +164% | +25% |

En 2024 los bancos subieron entre 278% y 316%.

![Drawdowns](estudio/descriptivo/figuras/d15_drawdowns.png)

#### La forma de la distribución

![Histogramas con normal y t](estudio/descriptivo/figuras/d03_histogramas.png)

| | GGAL | BMA | BBAR | YPF | TGS | EDN | TEO | CRESY | Cartera | SPY |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Desvío diario | 3,63% | 3,58% | 3,87% | 3,19% | 3,35% | 4,02% | 2,99% | 3,06% | 2,67% | 1,24% |
| Peor día | −56,1% | −52,7% | −55,9% | −34,1% | −48,0% | −59,0% | −33,4% | −38,1% | −47,1% | −10,9% |
| Mejor día | +38,7% | +37,6% | +40,8% | +39,9% | +36,8% | +36,9% | +37,5% | +39,8% | +34,9% | +14,5% |
| Asimetría | −0,25 | −0,13 | 0,02 | 0,20 | −0,06 | 0,09 | 0,10 | 0,43 | −0,50 | 0,01 |
| Curtosis en exceso | 18,3 | 15,7 | 16,0 | 18,5 | 14,8 | 15,3 | 14,2 | 17,7 | 30,4 | 14,6 |
| ν de la t-Student | 3,9 | 3,8 | 3,7 | 3,1 | 4,0 | 3,0 | 3,8 | 3,6 | 3,5 | 2,5 |
| Días a más de 5 desvíos | 14 | 11 | 11 | 9 | 9 | 14 | 9 | 9 | 12 | 19 |

*Bajo una normal se esperarían 0,003 días a más de 5 desvíos.*

- **Colas gruesas en las dos direcciones.** La asimetría de cada ADR es ≈ 0: hay días de −50% y también de +35%. La asimetría de la cartera (−0,50) viene de un solo día; sin las PASO es +0,68.
- **La diversificación aumenta la curtosis.** La de la cartera (30) es mayor que la de cualquier ADR, porque en los días extremos todos se mueven juntos.
- **La normal falla en el centro y en las colas.** El 41% de los días la cartera se mueve menos de ±1%; la normal predice 29%. Fuera de ±10% hubo 28 días, contra 0,9 esperados.

#### Los días extremos son políticos

| Peores días | Cartera | S&P 500 | | Mejores días | Cartera | S&P 500 |
|---|---:|---:|---|---|---:|---:|
| 12/08/2019 PASO | −47,1% | −1,2% | | 27/10/2025 Legislativas | +34,9% | +1,2% |
| 08/09/2025 Elección bonaerense | −18,9% | +0,2% | | 20/11/2023 Balotaje | +23,5% | +0,8% |
| 09/03/2020 COVID y petróleo | −17,2% | −7,8% | | 06/07/2020 Oferta de deuda | +18,9% | +1,5% |
| 21/10/2008 Estatización AFJP | −17,1% | −3,0% | | 31/08/2018 Rebote post corrida | +16,5% | 0,0% |
| 18/03/2020 COVID | −15,6% | −5,1% | | 22/09/2025 Apoyo del Tesoro de EE. UU. | +15,3% | +0,5% |

Cuando el S&P 500 está casi quieto, el shock es argentino:
- De los 10 peores días, 4 son locales puros y 4 son del COVID.
- De los 10 mejores, 9 son locales.
- Los 8 ADRs tuvieron su peor día el 12/08/2019.

[Análisis por activo, con la empresa, sus días extremos y sus caídas →](estudio/descriptivo/ANALISIS_DESCRIPTIVO.md#210-análisis-por-activo)

#### Los activos caen juntos

![Correlación móvil](estudio/descriptivo/figuras/d16_correlacion_movil.png)

La correlación promedio entre ADRs es 0,53. En ventanas de seis meses va de 0,18, entre 2009 y 2013, a 0,88, después de las PASO y de la elección bonaerense.

La correlación con el S&P 500 se comporta al revés:
- sube en las crisis globales;
- se derrumba en las locales.

![Dependencia en las colas](estudio/descriptivo/figuras/d18_dependencia_colas.png)

Una normal con la misma correlación subestima las caídas conjuntas:

| | Observado | Normal con igual correlación |
|---|---:|---:|
| P(un ADR en su peor 5% \| otro lo está) | 37% | 27% |
| Días con los 8 ADRs en su peor 5% | 17 | 2,9 |
| Días con 5 o más ADRs en su peor 5% | 111 | 69 |

Por eso el MC t-Student, que comparte un único shock de colas entre todos los activos, funciona mejor que el MC normal.

Los bancos son 37,5% del capital y aportan **44,7% de la varianza**: en la práctica, la cartera equiponderada es una apuesta al sistema financiero.

#### La volatilidad cambia y persiste

![Volatilidad móvil](estudio/descriptivo/figuras/d10_volatilidad_movil.png)

La volatilidad EWMA diaria de la cartera va de 0,9% a 11,8%. Su percentil 95 es casi 4 veces el percentil 5.

Por período, la volatilidad anual creció en cada gobierno:

| Período | Volatilidad anual |
|---|---:|
| 2007-2015 | 34,5% |
| 2015-2019 | 42,6% |
| 2019-2023 | 49,9% |
| 2023-2026 | 51,2% |

Desde 2018 no bajó de 37% en ningún año calendario.

No hay estacionalidad en la media (Kruskal-Wallis: p = 0,41 por mes y 0,68 por día de la semana). Sí la hay en la volatilidad: los lunes y los meses de agosto y octubre son más volátiles, porque las elecciones son los domingos.

#### Tests antes del modelo econométrico

| Test | Resultado | Implicancia |
|---|---|---|
| ADF y KPSS | Log-precios I(1); retornos I(0) en todas las series | Modelar retornos, no precios |
| Ljung-Box sobre $r$ | ACF(1) entre −0,05 y +0,03 en los ADRs | Media ≈ constante |
| Ljung-Box sobre $r^2$ y ARCH-LM(10) | p < 0,001 en todas las series | Hace falta varianza condicional (GARCH) |
| ACF de $\lvert r \rvert$ | Positiva hasta 40 rezagos | Volatilidad muy persistente |
| t-Student ajustada | ν entre 3 y 4 | Innovaciones t, no normales |
| Asimetría ≈ 0 | Saltos positivos tan grandes como los negativos | El efecto apalancamiento (GJR o EGARCH) probablemente sea débil |

**Propuesta para el capítulo econométrico:** estimar un AR(0)-GARCH(1,1)-t y compararlo con GJR, EGARCH e IGARCH, con y sin el día de las PASO.

**Ir al capítulo:**
- [Marco teórico](estudio/descriptivo/ANALISIS_DESCRIPTIVO.md#22-marco-teórico-hechos-estilizados-y-herramientas)
- [Contexto macro 2007-2026](estudio/descriptivo/ANALISIS_DESCRIPTIVO.md#23-contexto-macroeconómico-e-institucional-2007-2026)
- [Ficha de cada variable](estudio/descriptivo/ANALISIS_DESCRIPTIVO.md#25-ficha-de-cada-variable)
- [Análisis por activo](estudio/descriptivo/ANALISIS_DESCRIPTIVO.md#210-análisis-por-activo)
- [Preparación para el modelo](estudio/descriptivo/ANALISIS_DESCRIPTIVO.md#215-preparación-para-el-modelo-econométrico)
- [Conclusiones](estudio/descriptivo/ANALISIS_DESCRIPTIVO.md#216-conclusiones-del-capítulo)

## 3. Modelos

Todos se re-estiman **cada día** usando solo las 500 ruedas anteriores. Nada del día pronosticado entra en el pronóstico.

| Modelo | Distribución | Volatilidad | Idea |
|---|---|---|---|
| Simulación histórica | Empírica | Constante en la ventana | El pasado reciente se repite |
| **MC normal, cov. muestral** | Normal multivariada | Constante | El de manual (H0) |
| MC normal, cov. EWMA | Normal multivariada | EWMA λ = 0,94 (RiskMetrics) | Vol adaptativa, colas finas |
| **MC t-Student, cov. EWMA** | t multivariada (ν estimado mensual) | EWMA | Colas gruesas + **dependencia en las colas** |
| Simulación histórica filtrada | Residuos empíricos | EWMA | Forma empírica, escala de hoy |
| GARCH(1,1)-t | t univariada | GARCH re-ajustado cada 21 días | Estándar académico |
| EVT filtrado | GPD sobre la cola de residuos GARCH | GARCH | McNeil & Frey (2000) |

El detalle que hace funcionar al Monte Carlo t: dividir **todo** el vector normal por un único `sqrt(chi2/ν)` hace que, en un mal escenario, todos los activos caigan juntos. Es dependencia en las colas que la normal multivariada no puede producir, sin importar la correlación que le pongas ([`src/quantkit/risk.py`](src/quantkit/risk.py)).

## 4. Validación

Para cada modelo y cada día se compara la pérdida real con el VaR pronosticado la noche anterior.

| Test | Pregunta | Rechaza si |
|---|---|---|
| Kupiec (POF) | ¿La frecuencia de excepciones es 1%? | p < 0,05 |
| Christoffersen | ¿Las excepciones son independientes o vienen en racimos? | p < 0,05 |
| Cobertura condicional | Las dos cosas a la vez | p < 0,05 |
| Acerbi-Székely Z2 | ¿El ES 97,5% (la métrica de Basilea III) subestima las pérdidas? | Z2 < −0,70 |
| Semáforo de Basilea | Excepciones en las últimas 250 ruedas | 10+ = zona roja |
| Pérdida cuantílica | Ranking entre modelos (scoring rule consistente para cuantiles) | — |

Cada test está verificado contra una respuesta teórica conocida en [`tests/test_risk.py`](tests/test_risk.py) (16 tests).

## 5. Resultados

![Backtest del VaR 99%](estudio/figures/04_backtest_var.png)

![Boletín de calificaciones](estudio/figures/05_boletin.png)

| Modelo | Excepciones (esp. 44) | Kupiec p | Christoffersen p | Cob. cond. p | Días en zona roja | Z2 | Ranking |
|---|---:|---:|---:|---:|---:|---:|---:|
| Simulación histórica | 56 | 0,084 | **0,039** | **0,027** | 5,1% | −0,18 | 6 |
| MC normal, cov. muestral | 63 | **0,007** | **0,002** | **0,000** | 8,2% | −0,29 | 7 |
| MC normal, cov. EWMA | 62 | **0,011** | **0,012** | **0,002** | 4,5% | −0,29 | 3 |
| MC t-Student, cov. EWMA | 45 | 0,892 | 0,088 | 0,231 | 0,0% | −0,05 | 4 |
| Simulación histórica filtrada | 51 | 0,308 | 0,275 | 0,328 | 0,0% | −0,02 | 5 |
| GARCH(1,1)-t | 45 | 0,892 | 0,485 | 0,776 | 0,8% | 0,02 | 2 |
| EVT filtrado | 52 | 0,245 | 0,647 | 0,458 | 0,8% | −0,09 | 1 |

En negrita, rechazos al 5%. Ranking por pérdida cuantílica (menor es mejor).

**Lectura:**
- Pasar de covarianza muestral a EWMA (vol adaptativa) **no alcanza**: sigue fallando. Lo que corrige la frecuencia es agregar **colas gruesas**. Y lo que corrige el agrupamiento es que la volatilidad **reaccione** (EWMA o GARCH). Hacen falta las dos cosas.
- El MC normal con EWMA sale 3º en el ranking aunque falla los tests: la pérdida cuantílica premia reaccionar rápido, y eso lo hace bien. Un buen ranking no reemplaza a pasar los tests de cobertura.
- Ningún modelo es rechazado por el test del ES (Z2 > −0,70 en todos), aunque los tres modelos con colas finas o volatilidad fija tienen los Z2 más negativos.

![Semáforo de Basilea](estudio/figures/06_semaforo_basilea.png)

### Los peores días

![Los 6 peores días](estudio/figures/07_peores_dias.png)

En los días de COVID (marzo 2020) los modelos con volatilidad adaptativa ya habían subido el VaR por encima de la pérdida real. En las PASO 2019 y el 8 de septiembre de 2025 (−18,9%, tras la elección bonaerense) **ninguno** lo anticipó: fueron saltos políticos desde una volatilidad baja. Un modelo estadístico solo extrapola la ventana pasada.

### Monte Carlo: cuántas simulaciones

![Error de Monte Carlo](estudio/figures/08_error_montecarlo.png)

El error de simulación cae como 1/√N. Con N = 5.000 es ~4,5% del VaR (0,29 pp sobre un VaR de 6,4%). La diferencia entre el modelo más alto y el más bajo el mismo día es 43% del VaR mediano. **Más simulaciones no arreglan un mal modelo.**

### Horizonte de 10 días

![VaR a 10 días](estudio/figures/10_horizonte_10_dias.png)

Basilea pide VaR a 10 días y la práctica habitual es multiplicar el de 1 día por √10. Sobre 440 ventanas no superpuestas, la normal × √10 se rechaza (p = 0,02); la FHS × √10 y las trayectorias FHS simuladas día a día (donde un mal primer día sube la volatilidad de los siguientes) no se rechazan. Con 440 ventanas el test tiene poca potencia: es evidencia débil.

### Horizonte de un año

El abanico del principio usa bootstrap por bloques de 20 días (preserva los racimos de volatilidad). A un año, la cartera tiene **VaR 95% de 51%**, ES 95% de 61% y **36% de probabilidad de caer más de 40% en algún momento del año**. Con retornos normales esas cifras serían 44%, 53% y 32%: la normal subestima el riesgo también a horizontes largos.

## 6. Robustez

![Robustez](estudio/figures/09_robustez.png)

Las conclusiones no dependen de la ventana (250 o 1.000 ruedas) ni de λ (0,97). El **control con el S&P 500** es revelador: allí también falla la normal (2,0% y 2,3% de excepciones), pero el MC t-Student falla también (1,86%) mientras EVT, FHS y simulación histórica pasan. Una explicación posible: en los ADRs argentinos las colas son gruesas en las dos direcciones (asimetría cercana a 0 en cada activo), mientras que en el S&P 500 la cola izquierda pesa más que la derecha y una t simétrica no la captura. No lo testeé formalmente.

## 7. Limitaciones y lo que no funcionó

- **Un piloto previo con acciones de EEUU** rechazó los cuatro modelos que tenía entonces. La simulación histórica filtrada se agregó **después** de ver ese resultado: es una decisión post-hoc y está declarada como tal. Los otros seis modelos estaban fijados antes de correr el estudio argentino.
- **GARCH degenerado.** En 2 de los re-ajustes mensuales (ventanas 2024-2025) la máxima verosimilitud colapsó a varianza constante (α + β ≈ 0). La regla adoptada es conservar el último ajuste válido si α + β < 0,5. Es una decisión de implementación que afecta a GARCH-t y EVT.
- **Trayectorias FHS a 10 días.** Un residuo de −18σ (PASO) multiplicado por una volatilidad de crisis puede generar retornos simulados menores a −100%; se recortan en −99%.
- **Cartera fija, sin posiciones cortas ni derivados.** Una cartera con opciones necesitaría revaluación completa en cada escenario.
- **Supervivencia** (ver Datos) y una sola cartera.
- **Escenarios de estrés hipotéticos:** pendientes (`estudio/escenarios.py`). El informe incluye solo la repetición histórica de las PASO 2019.

## 8. Cómo reproducir

```bash
uv sync
uv run pytest
uv run python estudio/run.py
uv run python estudio/descriptivo/run_descriptivo.py
```

`run.py` regenera las 10 figuras de `estudio/figures/` y las tablas de [`estudio/results.md`](estudio/results.md) en ~80 segundos. Los precios se descargan una vez y quedan congelados en `data/`, así que los números del informe no cambian si Yahoo revisa la historia.

```
src/quantkit/risk.py     modelos de VaR/ES y tests de backtest
src/quantkit/data.py     descarga y caché de precios
estudio/models.py        pronósticos rodantes fuera de muestra
estudio/run.py           análisis, figuras y tablas
estudio/escenarios.py    escenarios de estrés hipotéticos
estudio/descriptivo/     capítulo descriptivo: script, tablas y figuras
tests/test_risk.py       cada modelo contra una respuesta teórica
_archivo/                estudios exploratorios fuera de foco
```

## Referencias

- Kupiec (1995), *Techniques for verifying the accuracy of risk measurement models*.
- Christoffersen (1998), *Evaluating interval forecasts*.
- Barone-Adesi, Giannopoulos & Vosper (1999), *VaR without correlations for portfolios of derivative securities*.
- McNeil & Frey (2000), *Estimation of tail-related risk measures for heteroscedastic financial time series: an extreme value approach*.
- Acerbi & Székely (2014), *Backtesting Expected Shortfall*.
- Comité de Basilea (1996), *Supervisory framework for the use of backtesting*.
- J.P. Morgan/Reuters (1996), *RiskMetrics Technical Document*.
