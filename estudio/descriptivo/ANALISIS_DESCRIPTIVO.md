# Capítulo 2. Análisis descriptivo de las variables

*Estudio: ¿Qué modelo de VaR sobrevive al mercado argentino? Backtest de 7 modelos sobre una cartera de ADRs argentinos, 2009-2026.*

Este capítulo describe, una por una, las variables que usa el estudio antes de modelarlas. Tiene tres objetivos: (i) dejar claro qué mide cada variable, cómo se construye y por qué se eligió; (ii) documentar los hechos estilizados que condicionan la elección de un modelo de riesgo; y (iii) preparar el terreno para el capítulo econométrico (un modelo de volatilidad condicional), que **no se estima acá**.

Todo se reproduce con un comando, que tarda unos 35 segundos y regenera las 13 figuras, las tablas en CSV y [`tablas.md`](tablas.md), de donde salen todos los números de este texto:

```bash
uv run python estudio/descriptivo/run_descriptivo.py
```

**Convenciones.** Los retornos son simples, diarios y en dólares. Los porcentajes diarios y los anualizados se presentan en tablas separadas. Se anualiza con 252 ruedas: la media por 252 y el desvío por √252. Los números usan coma decimal.

---

## 2.1 Datos y muestra

| | |
|---|---|
| Fuente | Yahoo Finance, precios de cierre **ajustados** por dividendos y splits |
| Congelamiento | Descargados una vez y guardados en `data/` (CSV): los números no cambian si Yahoo revisa la historia |
| Mercado | NYSE / Nasdaq, en USD |
| Frecuencia | Diaria (ruedas hábiles de EE. UU.) |
| Período de precios | 26/04/2007 a 30/10/2026 |
| Período de retornos | 27/04/2007 a 30/10/2026: **4.910 ruedas** (el SPY tiene 4.909 porque su caché empieza un día después) |
| Período del VaR fuera de muestra | 22/04/2009 a 30/10/2026: 4.410 ruedas (las primeras 500 se usan solo para estimar) |
| Activos | GGAL, BMA, BBAR, YPF, TGS, EDN, TEO, CRESY + SPY como control |

**Alineación de calendarios.** Los ocho ADRs y el SPY cotizan en el mismo calendario (NYSE), pero el panel se arma igual con la regla general del repo: se conservan los días en que operó el SPY y se completa hacia adelante (`p[p["SPY"].notna()].ffill()`) antes de calcular retornos. Así, un feriado de un solo listado no borra el retorno del día siguiente en todo el panel. En esta muestra la regla no cambia ningún dato: los retornos de los ADRs son idénticos a los de `estudio/run.py`, y el backtest reproduce exactamente las excepciones del informe (63, 62, 45, 51, 45 y 52; ver T11).

---

## 2.2 Ficha de cada variable

### V1. Precio de cierre ajustado de cada ADR, $P_{i,t}$

- **Qué es.** El precio en dólares de un *American Depositary Receipt*: un certificado que cotiza en Nueva York y representa acciones de una empresa argentina.
- **Construcción.** Cierre diario ajustado por dividendos y splits (Yahoo, `auto_adjust=True`). Unidades: USD por ADR. En las figuras se expresa en base 100 al 26/04/2007.
- **Por qué se eligió.** Es la variable primaria de la que sale todo lo demás. Se usan **ADRs y no acciones locales** porque cotizan en dólares, en un mercado líquido y sin los saltos que introducen el cepo y los cambios de régimen cambiario en el precio en pesos. El riesgo que se mide es el que ve un inversor en dólares.
- **Los 8 activos** se eligieron por liquidez y cobertura sectorial, sin optimizar: bancos (GGAL Grupo Financiero Galicia, BMA Banco Macro, BBAR BBVA Argentina), petróleo y gas (YPF, TGS Transportadora de Gas del Sur), electricidad (EDN Edenor), telecomunicaciones (TEO Telecom Argentina) y agro e inmobiliario (CRESY Cresud).

### V2. Retorno diario simple de cada ADR, $r_{i,t} = P_{i,t}/P_{i,t-1} - 1$

- **Qué es.** La variación porcentual diaria del precio en dólares, dividendos incluidos.
- **Construcción.** A partir de V1. Frecuencia diaria. Es una proporción (−0,05 = −5%).
- **Por qué retorno simple y no log-retorno.** El VaR se calcula sobre una **cartera**, y solo el retorno simple se agrega linealmente entre activos: $r_{p,t} = \sum_i w_i r_{i,t}$ vale exactamente para retornos simples y no para log-retornos. El log-retorno, $\ln(1+r)$, es cómodo para agregar en el **tiempo**, pero no entre activos. Además, deforma justo la cola que interesa: el −47,1% de la cartera en las PASO se convierte en −63,8% en logaritmos, y la curtosis de la cartera pasa de 30 a 68. Para no mezclar escalas, todo el capítulo usa retornos simples, igual que el resto del estudio. La media de los log-retornos solo se reporta en T2, para medir el crecimiento compuesto.
- **Por qué se eligió.** Es la variable de la que el VaR es un cuantil. Todo lo que el modelo tiene que capturar, como colas gruesas, volatilidad en racimos y correlaciones que cambian, se ve primero acá.

### V3. Retorno de la cartera equiponderada, $r_{p,t} = \sum_{i=1}^{8} \tfrac{1}{8}\, r_{i,t}$

- **Qué es.** El retorno diario de invertir 12,5% en cada ADR, rebalanceando todos los días a pesos iguales.
- **Por qué se eligió.** Es la **variable objetivo del estudio**: el VaR que se backtestea es el de esta cartera. Los pesos iguales evitan elegir la cartera "a dedo" mirando los resultados (sesgo de selección), aunque, como se ve en 2.6, iguales por activo no significa iguales por sector ni por riesgo.

### V4. Retorno del S&P 500 (ETF SPY), $r_{m,t}$

- **Qué es.** El retorno diario del ETF que replica el índice S&P 500, construido igual que V2.
- **Por qué se eligió.** No es un factor de un modelo de precios. Es el **grupo de control**: en el capítulo de robustez del estudio, los mismos modelos de VaR se corren sobre el SPY para separar lo que es propio de Argentina de lo que tiene cualquier mercado accionario. Por eso aparece en todas las tablas como referencia.

### V5. Volatilidad condicional EWMA de la cartera, $\sigma_t$

- **Qué es.** Un promedio exponencialmente ponderado de los retornos al cuadrado pasados (RiskMetrics, 1996): $\sigma_t^2 = \lambda\,\sigma_{t-1}^2 + (1-\lambda)\, r_{p,t-1}^2$, con $\lambda = 0{,}94$.
- **Construcción.** Ex ante: la volatilidad del día $t$ solo usa información hasta $t-1$. Se inicializa con la varianza de las primeras 20 ruedas, que se descartan en las tablas. Unidades: desvío diario. Se anualiza por √252 en los gráficos.
- **Por qué se eligió.** Es la pieza que hace "reaccionar" a tres de los siete modelos (MC normal y MC t con covarianza EWMA, y simulación histórica filtrada). GARCH-t y EVT usan su generalización, el GARCH(1,1). Como referencia más simple se agrega el **desvío móvil de 63 ruedas** (un trimestre bursátil), que pondera igual los últimos 63 días.

### V6. VaR 99% a un día estimado por cada modelo, $\widehat{VaR}_{t}^{(m)}$, e indicador de excepción $I_t^{(m)} = \mathbb{1}\{r_{p,t} < -\widehat{VaR}_t^{(m)}\}$

- **Qué es.** La pérdida que, según el modelo $m$, solo debería superarse el 1% de los días. Se expresa como pérdida positiva en proporción del valor de la cartera. El indicador vale 1 los días en que la pérdida real superó al VaR.
- **Construcción.** Pronósticos rodantes estrictamente fuera de muestra (`estudio/models.py`): para cada día $t$, cada modelo usa solo las 500 ruedas previas. Son siete modelos (ver el README).
- **Por qué se eligió.** Es la **variable que se evalúa**. Los tests de Kupiec y Christoffersen son tests sobre la serie binaria $I_t$: su frecuencia tiene que ser 1% y no debe tener racimos. Describirla antes de testearla muestra *cómo* falla cada modelo, no solo *si* falla.

### V7. Sector económico (variable cualitativa)

- **Qué es.** La clasificación de cada ADR en cinco sectores.
- **Por qué se eligió.** Es la única composición real del estudio: permite ver cuánto pesa cada sector en la cartera y cuánto aporta al riesgo (2.6).

### Eventos de referencia

En las series de tiempo se marcan seis fechas. En rojo van las dos que pide el estudio, ambas saltos políticos desde volatilidad baja; en gris, el contexto.

| # | Fecha | Evento | Cartera ese día |
|---|---|---|---:|
| 1 | 21/10/2008 | Anuncio de estatización de las AFJP, en plena crisis global | −17,1% |
| **2** | **12/08/2019** | **PASO 2019** | **−47,1%** |
| 3 | 09/03/2020 | COVID-19 (desplome global) | −17,2% |
| 4 | 20/11/2023 | Balotaje 2023 | +23,5% |
| **5** | **08/09/2025** | **Elección legislativa bonaerense** | **−18,9%** |
| 6 | 27/10/2025 | Elecciones legislativas nacionales | +34,9% |

De los seis días con mayor movimiento absoluto de la cartera, cinco son **locales**: las cuatro elecciones de la tabla y el acuerdo de reestructuración de la deuda del 06/07/2020 (+18,9%). El sexto es el COVID. Tres de esos cinco son **subas**: es el primer indicio de que la cola argentina es gruesa **en las dos direcciones**.

---

## 2.3 Evolución temporal

![Precios en base 100](figuras/d01_precios_base100.png)

**Figura 1. Precios en base 100, escala logarítmica.** Se usa escala logarítmica porque así distancias verticales iguales son variaciones porcentuales iguales. Hay tres lecturas:

1. **Ningún ADR tuvo una tendencia estable.** El S&P 500 (gris) sube de forma casi monótona. Los ADRs alternan ciclos de +300% y −80%. Varios terminan la muestra por debajo de su valor inicial en dólares: EDN (89), CRESY (89) y TEO (105, prácticamente en el punto de partida) después de 19 años.
2. **Los quiebres son comunes a todos los activos.** La caída de 2008, el desplome del 12/08/2019 (línea roja 2) y la recuperación desde fines de 2023 aparecen en los ocho paneles a la vez. Es riesgo **país**, no riesgo de empresa, y anticipa correlaciones altas (2.7).
3. **La cartera (panel inferior derecho) multiplicó por 6,9 su valor, pero con una caída máxima de −86%** (T2). Un inversor que la tuvo desde 2007 ganó algo parecido al S&P 500, pero con un camino mucho más violento.

![Retornos diarios](figuras/d02_retornos.png)

**Figura 2. Retornos diarios.** La serie oscila alrededor de cero con una amplitud que **cambia en el tiempo**: hay tramos tranquilos (2016-2017) y tramos agitados (fines de 2008, 2018, 2019-2020, 2025). Es el primer indicio visual de heterocedasticidad condicional, que la sección 2.9 testea formalmente. Los valores mínimos de cada ADR, entre −33% y −59%, ocurren **todos el 12/08/2019**: las PASO fueron el peor día de la historia de cada uno de estos activos.

---

## 2.4 Estadística descriptiva

### Retornos diarios

**Tabla 1. Estadística descriptiva de los retornos diarios** (27/04/2007 a 30/10/2026; momentos en %, salvo asimetría, curtosis y JB).

|               | N     | Media   | Mediana   | Desvío   | Mín.    | Máx.   | P1     | P5     | P95   | P99    | Asimetría   | Curtosis exceso   | Jarque-Bera   | JB p    |
|:--------------|:------|:--------|:----------|:---------|:--------|:-------|:-------|:-------|:------|:-------|:------------|:------------------|:--------------|:--------|
| GGAL          | 4.910 | 0,10%   | 0,00%     | 3,63%    | -56,12% | 38,70% | -8,21% | -5,14% | 5,71% | 10,53% | -0,25       | 18,3              | 68.423        | < 0,001 |
| BMA           | 4.910 | 0,09%   | -0,06%    | 3,58%    | -52,67% | 37,63% | -8,29% | -5,23% | 5,61% | 10,07% | -0,13       | 15,7              | 50.416        | < 0,001 |
| BBAR          | 4.910 | 0,09%   | 0,00%     | 3,87%    | -55,85% | 40,75% | -9,22% | -5,58% | 6,20% | 10,86% | 0,02        | 16,0              | 52.067        | < 0,001 |
| YPF           | 4.910 | 0,06%   | 0,00%     | 3,19%    | -34,05% | 39,89% | -7,79% | -4,37% | 5,01% | 8,39%  | 0,20        | 18,5              | 70.042        | < 0,001 |
| TGS           | 4.910 | 0,10%   | 0,00%     | 3,35%    | -48,04% | 36,79% | -7,98% | -4,75% | 5,30% | 9,47%  | -0,06       | 14,8              | 44.582        | < 0,001 |
| EDN           | 4.910 | 0,08%   | -0,08%    | 4,02%    | -58,98% | 36,95% | -9,40% | -5,54% | 6,35% | 12,73% | 0,09        | 15,3              | 47.828        | < 0,001 |
| TEO           | 4.910 | 0,05%   | 0,00%     | 2,99%    | -33,38% | 37,50% | -7,11% | -4,40% | 4,66% | 8,22%  | 0,10        | 14,2              | 41.054        | < 0,001 |
| CRESY         | 4.910 | 0,04%   | 0,00%     | 3,06%    | -38,09% | 39,80% | -7,07% | -4,39% | 4,67% | 8,80%  | 0,43        | 17,7              | 64.382        | < 0,001 |
| Cartera       | 4.910 | 0,08%   | 0,07%     | 2,67%    | -47,15% | 34,86% | -6,37% | -3,81% | 4,19% | 6,94%  | -0,50       | 30,4              | 189.370       | < 0,001 |
| S&P 500 (SPY) | 4.909 | 0,05%   | 0,07%     | 1,24%    | -10,94% | 14,52% | -3,58% | -1,81% | 1,69% | 3,29%  | 0,01        | 14,6              | 43.305        | < 0,001 |

*Curtosis en exceso: la normal tiene 0. JB: estadístico de Jarque-Bera, $H_0$: normalidad.*

**Lectura.**

- **Centro.** La media diaria es positiva pero minúscula frente al desvío: entre 0,04% y 0,10% por día contra un desvío de 3% a 4%. La mediana es casi exactamente cero en varios ADRs, porque entre el 0,6% (BMA) y el 4,4% (TGS) de los días el precio **no cambió** (cierres sin operaciones o con muy poco volumen). Eso explica el pico en cero que se ve en los histogramas.
- **Dispersión.** Los ADRs tienen un desvío diario de 3,0% a 4,0%, entre 2,4 y 3,2 veces el del S&P 500 (1,24%). La cartera baja a 2,67% gracias a la diversificación, pero sigue más del doble que el SPY.
- **Colas.** El percentil 1 de la cartera es −6,37%. Es el **VaR 99% histórico incondicional**: en uno de cada cien días se pierde más que eso. Bajo una normal con el mismo desvío, ese percentil estaría en −6,1%: el número se parece, pero lo que pasa *más allá* de ese percentil es muy distinto (ver T5).
- **Asimetría.** Es cercana a cero en cada ADR (entre −0,25 y +0,43). La cola izquierda no pesa sistemáticamente más que la derecha: Argentina tiene días de −50% y también de +35%. La cartera es la excepción (−0,50), y eso se debe casi entero a un solo día: sin las PASO, su asimetría es **+0,68**.
- **Curtosis.** Está entre 14 y 18 en cada ADR y llega a **30 en la cartera**. Normalmente diversificar reduce la curtosis. Acá la aumenta, porque en los días extremos todos los activos se mueven juntos: la diversificación reduce la varianza de los días normales, pero no la de los días extremos (ver 2.7). Sin las PASO, la curtosis de la cartera baja a 12, lo que muestra lo influyente que es una sola observación.
- **Normalidad.** Jarque-Bera rechaza la normalidad en todas las series con p prácticamente nulo. Los estadísticos (de 41.000 a 189.000) son tan grandes que la pregunta deja de ser *si* los retornos son normales y pasa a ser *cuánto* se alejan de la normal.

### Retornos y riesgo anualizados

**Tabla 2. Retornos y riesgo anualizados.**

|               | Media aritmética anual   | Media log anual   | Crecimiento anual compuesto   | Volatilidad anual   | Máxima caída acumulada   |
|:--------------|:-------------------------|:------------------|:------------------------------|:--------------------|:-------------------------|
| GGAL          | 24,3%                    | 7,3%              | 7,6%                          | 57,6%               | -91,7%                   |
| BMA           | 22,2%                    | 5,8%              | 5,9%                          | 56,8%               | -91,7%                   |
| BBAR          | 23,3%                    | 4,1%              | 4,2%                          | 61,4%               | -91,5%                   |
| YPF           | 15,8%                    | 3,0%              | 3,0%                          | 50,6%               | -94,5%                   |
| TGS           | 24,4%                    | 10,1%             | 10,7%                         | 53,1%               | -83,1%                   |
| EDN           | 20,1%                    | -0,6%             | -0,6%                         | 63,7%               | -95,5%                   |
| TEO           | 11,6%                    | 0,3%              | 0,3%                          | 47,4%               | -85,9%                   |
| CRESY         | 11,2%                    | -0,6%             | -0,6%                         | 48,6%               | -89,4%                   |
| Cartera       | 19,1%                    | 9,9%              | 10,4%                         | 42,4%               | -86,3%                   |
| S&P 500 (SPY) | 12,1%                    | 10,1%             | 10,7%                         | 19,6%               | -55,2%                   |

*Media aritmética anual = media diaria × 252. Media log anual = media de ln(1 + r) × 252. Crecimiento anual compuesto: (valor final / valor inicial)^(1/años) − 1.*

**Lectura.** La diferencia entre la primera y la segunda columna es el **costo de la volatilidad** (*volatility drag*): aproximadamente $\sigma^2/2$. Para EDN, una media aritmética de +20,1% anual convive con un crecimiento compuesto de −0,6%: con 64% de volatilidad, el promedio de los retornos diarios dice que se gana, pero el inversor que mantuvo el ADR perdió plata. Por eso, en activos tan volátiles, la media aritmética anualizada **no** sirve como medida de rendimiento. La cartera creció 10,4% anual compuesto, casi lo mismo que el S&P 500 (10,7%), con 2,2 veces su volatilidad y una caída máxima de −86% contra −55%. Para el estudio importa que todos los ADRs perdieron entre 83% y 95% de su valor en algún momento de la muestra. En un mercado así, el riesgo de cola no es un caso teórico.

---

## 2.5 Forma de la distribución

![Histogramas](figuras/d03_histogramas.png)

**Figura 3. Histogramas con la normal y la t-Student superpuestas.** Las dos distribuciones se ajustan por máxima verosimilitud a toda la muestra. En todos los paneles se repite el mismo patrón: el histograma tiene **más altura en el centro, menos en los "hombros"** (entre 1 y 2 desvíos, aproximadamente) **y más en las colas** que la normal naranja. Es la firma de una distribución leptocúrtica. La t-Student (línea punteada) sigue el centro y los hombros casi a la perfección. Los grados de libertad estimados, ν, se reportan en el título de cada panel y en la Tabla 6.

**Tabla 6. Parámetros de la t-Student ajustada por máxima verosimilitud.**

|               | ν (grados de libertad)   | Escala   |
|:--------------|:-------------------------|:---------|
| GGAL          | 3,88                     | 2,50%    |
| BMA           | 3,76                     | 2,45%    |
| BBAR          | 3,73                     | 2,64%    |
| YPF           | 3,11                     | 1,98%    |
| TGS           | 3,99                     | 2,35%    |
| EDN           | 3,02                     | 2,50%    |
| TEO           | 3,78                     | 2,06%    |
| CRESY         | 3,62                     | 2,04%    |
| Cartera       | 3,50                     | 1,74%    |
| S&P 500 (SPY) | 2,47                     | 0,67%    |

**Lectura.** Los ADRs tienen ν entre 3,0 y 4,0. Esto importa por dos razones. Primero, con ν ≤ 4 la t no tiene cuarto momento finito: la curtosis "poblacional" es infinita, y por eso las curtosis muestrales de la Tabla 1 son tan grandes e inestables, ya que dependen de unos pocos días. Segundo, con ν > 2 la varianza sí existe, así que tiene sentido hablar de volatilidad. Que el SPY dé ν = 2,5 no indica que sea más riesgoso que los ADRs. Indica que, **relativo a su propio desvío**, sus días extremos (octubre de 2008, marzo de 2020) son más raros y más lejanos: su riesgo está más concentrado en episodios puntuales.

![QQ-plots](figuras/d04_qq.png)

**Figura 4. Gráficos QQ contra la normal.** Cada punto compara un cuantil observado, en desvíos estándar, con el que tendría una normal. Los puntos siguen la diagonal entre −2 y +2 desvíos y se despegan en **las dos** colas, hasta −15 desvíos en los bancos y −18 en la cartera (las PASO). La curva de la t ajustada sigue esa forma en S mucho mejor que la recta. Es la justificación gráfica de los modelos con colas gruesas del estudio (MC t-Student, GARCH-t, EVT).

### Distribución de frecuencias

**Tabla 3. Distribución de frecuencias del retorno diario de la cartera.** Intervalos cerrados a izquierda; la última columna es la probabilidad del intervalo bajo una normal con la misma media y desvío.

| Intervalo    | Frec. absoluta   | Frec. relativa   | Frec. acumulada   | Frec. rel. acumulada   | Esperada si fuera normal   |
|:-------------|:-----------------|:-----------------|:------------------|:-----------------------|:---------------------------|
| menor a -10% | 10               | 0,20%            | 10                | 0,20%                  | 0,01%                      |
| [-10%, -5%)  | 105              | 2,14%            | 115               | 2,34%                  | 2,85%                      |
| [-5%, -4%)   | 92               | 1,87%            | 207               | 4,22%                  | 3,48%                      |
| [-4%, -3%)   | 205              | 4,18%            | 412               | 8,39%                  | 6,12%                      |
| [-3%, -2%)   | 354              | 7,21%            | 766               | 15,60%                 | 9,38%                      |
| [-2%, -1%)   | 649              | 13,22%           | 1.415             | 28,82%                 | 12,51%                     |
| [-1%, 0%)    | 968              | 19,71%           | 2.383             | 48,53%                 | 14,52%                     |
| [0%, 1%)     | 1.051            | 21,41%           | 3.434             | 69,94%                 | 14,68%                     |
| [1%, 2%)     | 655              | 13,34%           | 4.089             | 83,28%                 | 12,91%                     |
| [2%, 3%)     | 368              | 7,49%            | 4.457             | 90,77%                 | 9,89%                      |
| [3%, 4%)     | 181              | 3,69%            | 4.638             | 94,46%                 | 6,59%                      |
| [4%, 5%)     | 114              | 2,32%            | 4.752             | 96,78%                 | 3,82%                      |
| [5%, 10%)    | 140              | 2,85%            | 4.892             | 99,63%                 | 3,24%                      |
| 10% o más    | 18               | 0,37%            | 4.910             | 100,00%                | 0,01%                      |

![Frecuencias](figuras/d05_frecuencias_cartera.png)

**Figura 5. Frecuencias relativas y ojiva.** La tabla cuantifica lo que muestran los histogramas:

- El **41%** de los días la cartera se mueve menos de ±1%. Una normal predeciría 29%.
- Entre 2% y 5% en cualquiera de las dos direcciones hay **menos** días que en la normal: por ejemplo, 3,7% de los días caen en [3%, 4%) contra 6,6% esperado.
- Fuera de ±10% hay **28 días** observados contra **0,9** esperados: 30 veces más.

Por eso la ojiva observada (azul) es más empinada en el centro que la normal. En la cola izquierda casi coincide con ella: la frecuencia acumulada hasta −5% es 2,3% contra 2,9% de la normal. Lo que cambia es *qué tan lejos* llegan las pérdidas cuando se cruza ese umbral. Para el VaR la consecuencia es concreta: una normal puede acertar *aproximadamente* el cuantil del 1%, pero subestima por órdenes de magnitud **cuánto** se pierde más allá de él. Eso es lo que mide el Expected Shortfall.

**Tabla 4. Frecuencia relativa por intervalo, por activo.**

|               | menor a -10%   | [-10%, -5%)   | [-5%, -2%)   | [-2%, 0%)   | [0%, 2%)   | [2%, 5%)   | [5%, 10%)   | 10% o más   | Días a más de 3 desvíos   |
|:--------------|:---------------|:--------------|:-------------|:------------|:-----------|:-----------|:------------|:------------|:--------------------------|
| GGAL          | 0,6%           | 4,9%          | 17,3%        | 26,3%       | 27,0%      | 17,3%      | 5,5%        | 1,2%        | 1,3%                      |
| BMA           | 0,5%           | 4,9%          | 17,5%        | 27,7%       | 26,1%      | 16,8%      | 5,5%        | 1,0%        | 1,3%                      |
| BBAR          | 0,8%           | 5,5%          | 17,9%        | 25,2%       | 26,1%      | 16,9%      | 6,3%        | 1,3%        | 1,3%                      |
| YPF           | 0,5%           | 3,3%          | 15,1%        | 30,5%       | 31,1%      | 14,5%      | 4,5%        | 0,5%        | 1,2%                      |
| TGS           | 0,5%           | 3,9%          | 17,2%        | 26,2%       | 29,9%      | 16,5%      | 5,0%        | 0,8%        | 1,2%                      |
| EDN           | 0,9%           | 5,5%          | 18,0%        | 25,9%       | 26,1%      | 15,3%      | 6,5%        | 1,8%        | 1,5%                      |
| TEO           | 0,2%           | 3,6%          | 15,7%        | 30,3%       | 30,3%      | 15,6%      | 3,7%        | 0,5%        | 1,2%                      |
| CRESY         | 0,2%           | 3,2%          | 16,1%        | 30,2%       | 30,8%      | 15,1%      | 3,8%        | 0,6%        | 1,2%                      |
| Cartera       | 0,2%           | 2,1%          | 13,3%        | 32,9%       | 34,7%      | 13,5%      | 2,9%        | 0,4%        | 1,0%                      |
| S&P 500 (SPY) | 0,0%           | 0,3%          | 3,9%         | 40,6%       | 51,9%      | 2,9%       | 0,3%        | 0,1%        | 1,7%                      |

**Tabla 5. Días extremos: cantidad observada a más de k desvíos de la media vs la esperada bajo normalidad.**

|                          | Más de 3 desvíos   | Más de 4 desvíos   | Más de 5 desvíos   |
|:-------------------------|:-------------------|:-------------------|:-------------------|
| GGAL                     | 62                 | 23                 | 14                 |
| BMA                      | 63                 | 25                 | 11                 |
| BBAR                     | 65                 | 19                 | 11                 |
| YPF                      | 57                 | 24                 | 9                  |
| TGS                      | 60                 | 22                 | 9                  |
| EDN                      | 75                 | 29                 | 14                 |
| TEO                      | 57                 | 19                 | 9                  |
| CRESY                    | 58                 | 20                 | 9                  |
| Cartera                  | 50                 | 20                 | 12                 |
| S&P 500 (SPY)            | 82                 | 37                 | 19                 |
| Esperado si fuera normal | 13                 | 0,311              | 0,003              |

**Lectura.** Con 4.910 días, una normal predice 13 días a más de 3 desvíos, 0,3 a más de 4 y 0,003 (un solo día en unas 350 muestras como esta) a más de 5. Cada ADR tiene entre 9 y 14 días a más de 5 desvíos. Con una normal, eventos así no deberían ocurrir nunca en la vida de un inversor. Acá ocurren cada uno o dos años.

### Diagramas de caja

![Boxplots](figuras/d06_boxplots.png)

**Figura 6. Diagramas de caja.** A la izquierda, en escala completa, dominan los atípicos: los puntos del 12/08/2019 por debajo de −30% y las subas de octubre de 2025 por encima de +30%. A la derecha, el zoom en ±8% muestra la caja: el 50% central de los días cae aproximadamente entre −1,9% y +1,9% en los bancos y EDN, y entre −1,5% y +1,8% en el resto. En los ADRs, entre el 4,1% y el 5,8% de los días queda **fuera de los bigotes** (1,5 rangos intercuartílicos), cuando con una normal sería 0,7%. El S&P 500 tiene la caja más angosta (de −0,4% a +0,6%) y, en proporción, todavía más atípicos (7,5%): colas gruesas tiene cualquier mercado. Lo propio de Argentina es su **escala**: un atípico del SPY es un −5%; uno de un ADR, un −30%.

---

## 2.6 Comparación entre activos y composición

![Barras de riesgo](figuras/d07_barras_riesgo.png)

**Figura 7. Volatilidad anual, peor día y curtosis por activo.** El orden de los tres rankings **no coincide**. EDN es el más volátil y el que tuvo el peor día (−59%), pero no el de mayor curtosis. YPF tiene la curtosis más alta entre los ADRs (18,5) con una volatilidad intermedia: sus retornos son "tranquilos con saltos". Por eso una sola medida de riesgo, como la volatilidad, no alcanza para ordenar estos activos. El dato más importante es la **cartera**: es la de menor volatilidad entre los activos argentinos (42%), pero tiene la **mayor curtosis** (30). La diversificación reduce el desvío y concentra el riesgo en la cola.

![Sectores](figuras/d08_sectores.png)

**Tabla 7. Composición y contribución al riesgo por sector.**

| Sector              | Peso   | Contribución al riesgo   |
|:--------------------|:-------|:-------------------------|
| Bancos              | 37,5%  | 44,7%                    |
| Petróleo y gas      | 25,0%  | 21,8%                    |
| Electricidad        | 12,5%  | 13,6%                    |
| Telecomunicaciones  | 12,5%  | 10,1%                    |
| Agro e inmobiliario | 12,5%  | 9,8%                     |

**Figura 8. ¿De dónde viene el riesgo?** Acá el gráfico de torta es honesto porque representa una **composición real**: los pesos de la cartera suman 100%. Los pesos iguales por activo resultan en **37,5% en bancos** (tres de los ocho ADRs), 25% en petróleo y gas, y 12,5% en cada uno de los otros tres sectores. Para la contribución al riesgo se usan **barras y no otra torta**, porque lo que interesa es *comparar* dos cantidades por sector (peso vs riesgo), y eso se lee mejor en barras alineadas. La contribución se calcula con la descomposición de Euler de la varianza, $RC_i = w_i (\Sigma w)_i / w'\Sigma w$, que suma exactamente 100%. Los **bancos aportan 44,7% de la varianza con 37,5% del capital**: son los más volátiles y los más correlacionados entre sí (0,76 a 0,80). Telecomunicaciones y agro aportan menos que su peso. En la práctica, la cartera "equiponderada" es sobre todo una apuesta al sistema financiero argentino.

---

## 2.7 Dependencia entre activos

![Correlaciones](figuras/d09_correlaciones.png)

**Tabla 8. Matriz de correlaciones de Pearson entre retornos diarios.**

|               | GGAL   | BMA   | BBAR   | YPF   | TGS   | EDN   | TEO   | CRESY   | Cartera   | S&P 500 (SPY)   |
|:--------------|:-------|:------|:-------|:------|:------|:------|:------|:--------|:----------|:----------------|
| GGAL          | 1,00   | 0,80  | 0,78   | 0,54  | 0,55  | 0,55  | 0,56  | 0,50    | 0,87      | 0,40            |
| BMA           | 0,80   | 1,00  | 0,76   | 0,55  | 0,55  | 0,54  | 0,59  | 0,52    | 0,87      | 0,38            |
| BBAR          | 0,78   | 0,76  | 1,00   | 0,52  | 0,53  | 0,53  | 0,55  | 0,49    | 0,85      | 0,36            |
| YPF           | 0,54   | 0,55  | 0,52   | 1,00  | 0,48  | 0,41  | 0,44  | 0,42    | 0,70      | 0,30            |
| TGS           | 0,55   | 0,55  | 0,53   | 0,48  | 1,00  | 0,47  | 0,46  | 0,45    | 0,73      | 0,27            |
| EDN           | 0,55   | 0,54  | 0,53   | 0,41  | 0,47  | 1,00  | 0,44  | 0,42    | 0,73      | 0,27            |
| TEO           | 0,56   | 0,59  | 0,55   | 0,44  | 0,46  | 0,44  | 1,00  | 0,46    | 0,72      | 0,35            |
| CRESY         | 0,50   | 0,52  | 0,49   | 0,42  | 0,45  | 0,42  | 0,46  | 1,00    | 0,68      | 0,39            |
| Cartera       | 0,87   | 0,87  | 0,85   | 0,70  | 0,73  | 0,73  | 0,72  | 0,68    | 1,00      | 0,44            |
| S&P 500 (SPY) | 0,40   | 0,38  | 0,36   | 0,30  | 0,27  | 0,27  | 0,35  | 0,39    | 0,44      | 1,00            |

**Lectura.** Las correlaciones entre ADRs van de 0,41 a 0,80, con un promedio de 0,53. El bloque de los tres bancos (0,76 a 0,80) es el más correlacionado: son prácticamente el mismo factor de riesgo. Con el S&P 500 las correlaciones son bajas (0,27 a 0,40). Lo que mueve a estos activos es local. Hay dos advertencias:

1. **La correlación de toda la muestra es un promedio de regímenes.** El estudio principal muestra que la correlación promedio entre ADRs sube de 0,44 en calma a 0,70 en el 20% de días más volátiles (figura `02_hechos_estilizados.png`). Un modelo con correlación constante sobreestima la diversificación justo cuando importa.
2. **Pearson mide dependencia lineal promedio y no dependencia en las colas.** Que todos caigan juntos en los días extremos es justamente lo que la cartera no puede diversificar y lo que explica su curtosis de 30.

---

## 2.8 Volatilidad y VaR estimado en el tiempo

![Volatilidad móvil](figuras/d10_volatilidad_movil.png)

**Tabla 10. Volatilidad EWMA diaria de la cartera (λ = 0,94).**

|                                | N     | Media   | Mediana   | Desvío   | Mín.   | Máx.   | P5    | P95   |
|:-------------------------------|:------|:--------|:----------|:---------|:-------|:-------|:------|:------|
| Vol. EWMA diaria de la cartera | 4.890 | 2,42%   | 2,29%     | 1,14%    | 0,87%  | 11,84% | 1,21% | 4,69% |

**Figura 10. Volatilidad móvil.** Hay dos regularidades que justifican un modelo de volatilidad condicional:

- **Cambia de nivel.** La volatilidad diaria EWMA va de 0,87% en el día más tranquilo a 11,8% después de las PASO. Su P5 es 1,2% y su P95, 4,7%: un día "normal-malo" es casi cuatro veces más riesgoso que uno "normal-bueno". Cualquier VaR con volatilidad fija está desalineado en los dos extremos a la vez.
- **Persiste.** Después de cada shock la volatilidad tarda meses en volver a su nivel. La EWMA (azul) reacciona al día siguiente y decae de forma gradual. El desvío móvil de 63 ruedas (negro) reacciona de golpe y "olvida" de golpe 63 ruedas después: es el escalón que se ve en noviembre de 2019. Esa diferencia es la que separa a los modelos con EWMA de la simulación histórica en el backtest.

El panel inferior muestra que la volatilidad de la cartera (negro) está casi siempre por debajo de la mediana de los ADRs (azul), salvo en los picos de crisis, donde se acerca: **la diversificación funciona en calma y se pierde en crisis**. El S&P 500 tiene sus propios picos (2008, 2020) pero no los argentinos (2019, 2025): son shocks idiosincráticos del país.

**Tabla 9. La cartera por período de gobierno.**

|                                   | N     | Media anual   | Volatilidad anual   | Peor día   | Mejor día   | Curtosis exceso   |
|:----------------------------------|:------|:--------------|:--------------------|:-----------|:------------|:------------------|
| Fernández de Kirchner (2007-2015) | 2.172 | 16,5%         | 34,5%               | -17,1%     | 12,3%       | 5,4               |
| Macri (2015-2019)                 | 1.006 | -1,0%         | 42,6%               | -47,1%     | 16,5%       | 95,9              |
| Fernández (2019-2023)             | 1.007 | 31,8%         | 49,9%               | -17,2%     | 23,5%       | 6,0               |
| Milei (2023-2026)                 | 725   | 37,2%         | 51,2%               | -18,9%     | 34,9%       | 20,2              |

![Por período](figuras/d13_por_periodo.png)

**Figura 13. Riesgo por período.** Los mandatos presidenciales se usan solo como partición del calendario, no como hipótesis causal. La volatilidad anual de la cartera **creció en cada período**: 34%, 43%, 50% y 51%. La curtosis de 96 en el período 2015-2019 es casi entera de un día: sin las PASO, baja a 6,0, en línea con los demás períodos. Esto anticipa dos cuestiones para el modelo: (i) un único juego de parámetros para 19 años puede no ser estable (conviene un test de quiebre estructural o un modelo con cambio de régimen); (ii) las estimaciones de momentos altos son extremadamente sensibles a unos pocos días.

![VaR estimado](figuras/d12_var_estimado.png)

**Tabla 11. VaR 99% a un día pronosticado por cada modelo** (22/04/2009 a 30/10/2026, 4.410 ruedas; esperadas: 44 excepciones).

|                               | Media   | Mediana   | Desvío   | Mín.   | Máx.   | Coef. de variación   | Excepciones   |
|:------------------------------|:--------|:----------|:---------|:-------|:-------|:---------------------|:--------------|
| Simulación histórica          | 6,25%   | 6,23%     | 1,82%    | 3,10%  | 11,36% | 0,29                 | 56            |
| MC normal, cov. muestral      | 5,96%   | 5,70%     | 1,65%    | 3,14%  | 10,32% | 0,28                 | 63            |
| MC normal, cov. EWMA          | 5,68%   | 5,42%     | 2,63%    | 1,97%  | 27,68% | 0,46                 | 62            |
| MC t-Student, cov. EWMA       | 6,32%   | 5,98%     | 2,96%    | 2,17%  | 32,71% | 0,47                 | 45            |
| Simulación histórica filtrada | 6,39%   | 5,73%     | 3,51%    | 2,18%  | 40,16% | 0,55                 | 51            |
| GARCH(1,1)-t                  | 6,31%   | 6,08%     | 2,74%    | 3,02%  | 79,67% | 0,43                 | 45            |
| EVT filtrado (GARCH + GPD)    | 6,10%   | 5,91%     | 2,72%    | 2,68%  | 78,17% | 0,45                 | 52            |

**Figura 12. La variable que se evalúa.** Los siete modelos tienen un VaR **medio parecido** (5,7% a 6,4%), cercano al percentil 1 incondicional de la cartera (6,4%). La diferencia entre ellos no está en el nivel promedio sino en **cuánto y cuándo se mueven**. El coeficiente de variación va de 0,28 en los modelos de ventana fija (normal con covarianza muestral, simulación histórica) a 0,43-0,55 en los que tienen volatilidad adaptativa. Los dos modelos de ventana fija nunca superan 11,4% de VaR. Los modelos GARCH llegan a 78-80% el 13/08/2019, el día después de las PASO, porque su volatilidad condicional explota con un retorno de 18 desvíos. Que el promedio sea similar y la dispersión muy distinta es la razón por la que el estudio necesita tests de **independencia** (Christoffersen) además de frecuencia (Kupiec).

**Tabla 12. Excepciones del VaR 99% por año** (tres modelos representativos).

| Año   | Ruedas   | Esperadas (1%)   | MC normal, cov. muestral   | MC t-Student, cov. EWMA   | EVT filtrado (GARCH + GPD)   |
|:------|:---------|:-----------------|:---------------------------|:--------------------------|:-----------------------------|
| 2009  | 177      | 1,8              | 0                          | 0                         | 0                            |
| 2010  | 252      | 2,5              | 0                          | 3                         | 3                            |
| 2011  | 252      | 2,5              | 7                          | 5                         | 5                            |
| 2012  | 250      | 2,5              | 4                          | 2                         | 2                            |
| 2013  | 252      | 2,5              | 4                          | 1                         | 1                            |
| 2014  | 252      | 2,5              | 4                          | 2                         | 2                            |
| 2015  | 252      | 2,5              | 3                          | 2                         | 4                            |
| 2016  | 252      | 2,5              | 0                          | 3                         | 1                            |
| 2017  | 251      | 2,5              | 2                          | 3                         | 2                            |
| 2018  | 251      | 2,5              | 20                         | 8                         | 9                            |
| 2019  | 252      | 2,5              | 5                          | 4                         | 6                            |
| 2020  | 253      | 2,5              | 4                          | 2                         | 1                            |
| 2021  | 252      | 2,5              | 0                          | 1                         | 2                            |
| 2022  | 251      | 2,5              | 3                          | 3                         | 4                            |
| 2023  | 250      | 2,5              | 3                          | 1                         | 4                            |
| 2024  | 252      | 2,5              | 1                          | 1                         | 1                            |
| 2025  | 250      | 2,5              | 3                          | 3                         | 5                            |
| 2026  | 209      | 2,1              | 0                          | 1                         | 0                            |

**Lectura.** Esta es la tabla de frecuencias de la variable binaria $I_t$. Con un modelo correcto, cada año debería tener unas 2,5 excepciones, con variaciones al azar. El Monte Carlo normal con covarianza muestral concentra **20 excepciones en 2018** (la crisis cambiaria y el acuerdo con el FMI), 8 veces lo esperado, y 0 en 2009, 2010, 2016, 2021 y 2026: las excepciones vienen en **racimos**. Los modelos con volatilidad adaptativa también fallan más en 2018 (7 y 9), pero reparten mejor las excepciones entre años. Es la versión descriptiva del rechazo del test de Christoffersen que reporta el estudio.

---

## 2.9 Preparación para el modelo econométrico

El capítulo siguiente modela la volatilidad de la cartera con un modelo de la familia GARCH. Antes hay que responder tres preguntas: ¿qué serie se modela, los precios o los retornos? ¿Hace falta una ecuación para la media? ¿Hay evidencia de heterocedasticidad condicional?

### Estacionariedad

**Tabla 13. Tests de raíz unitaria (ADF) y de estacionariedad (KPSS).**

|               | ADF log-precio   | ADF log-precio p   | KPSS log-precio   | KPSS log-precio p   | ADF retorno   | ADF retorno p   | KPSS retorno   | KPSS retorno p   |
|:--------------|:-----------------|:-------------------|:------------------|:--------------------|:--------------|:----------------|:---------------|:-----------------|
| GGAL          | -2,39            | 0,384              | 0,71              | 0,010               | -13,59        | < 0,001         | 0,09           | 0,100            |
| BMA           | -2,14            | 0,523              | 0,68              | 0,010               | -23,91        | < 0,001         | 0,11           | 0,100            |
| BBAR          | -1,89            | 0,657              | 0,85              | 0,010               | -72,72        | < 0,001         | 0,10           | 0,100            |
| YPF           | -1,45            | 0,847              | 1,07              | 0,010               | -32,29        | < 0,001         | 0,20           | 0,100            |
| TGS           | -2,46            | 0,346              | 0,54              | 0,010               | -30,87        | < 0,001         | 0,15           | 0,100            |
| EDN           | -2,12            | 0,534              | 0,53              | 0,010               | -18,66        | < 0,001         | 0,13           | 0,100            |
| TEO           | -2,46            | 0,349              | 0,82              | 0,010               | -29,38        | < 0,001         | 0,06           | 0,100            |
| CRESY         | -2,37            | 0,395              | 0,59              | 0,010               | -69,24        | < 0,001         | 0,12           | 0,100            |
| Cartera       | -1,93            | 0,639              | 0,66              | 0,010               | -28,59        | < 0,001         | 0,15           | 0,100            |
| S&P 500 (SPY) | -3,92            | 0,011              | 0,57              | 0,010               | -17,51        | < 0,001         | 0,20           | 0,100            |

**Lectura.** ADF y KPSS tienen hipótesis nulas opuestas, así que se leen juntos:

- **Log-precios:** ADF **no rechaza** la raíz unitaria en ningún ADR ni en la cartera (p entre 0,34 y 0,85), y KPSS **rechaza** la estacionariedad en todos (p ≤ 0,01). Los dos tests coinciden: los log-precios son **I(1)**, es decir, no estacionarios. El SPY es un caso ambiguo: ADF rechaza al 5% (p = 0,011, con tendencia) y KPSS también rechaza. Es habitual en series con una tendencia muy fuerte y no cambia nada para este estudio, porque no se modelan precios.
- **Retornos:** ADF rechaza la raíz unitaria con p < 0,001 en todos los casos, y KPSS no rechaza la estacionariedad (p ≥ 0,10, el máximo que reporta su tabla). Los retornos son **I(0)**.

**Implicancia:** el modelo se especifica sobre **retornos**. Una regresión con precios en niveles sería espuria. Esto también confirma que lo que hace el estudio (estimar sobre ventanas de retornos) es coherente.

### Autocorrelación de los retornos y de su magnitud

![ACF](figuras/d11_acf.png)

**Tabla 14. Autocorrelación (Ljung-Box) y efectos ARCH (test LM de Engle).**

|               | ACF(1) retorno   | Q(10) retorno   | Q(10) p   | ACF(1) retorno²   | Q(10) retorno²   | Q²(10) p   | ACF(1) valor abs.   | Q(10) valor abs.   | ARCH-LM(10)   | ARCH-LM p   |
|:--------------|:-----------------|:----------------|:----------|:------------------|:-----------------|:-----------|:--------------------|:-------------------|:--------------|:------------|
| GGAL          | 0,002            | 32,9            | < 0,001   | 0,049             | 81,0             | < 0,001    | 0,151               | 751,3              | 61,8          | < 0,001     |
| BMA           | -0,010           | 16,6            | 0,083     | 0,079             | 113,9            | < 0,001    | 0,170               | 802,6              | 83,7          | < 0,001     |
| BBAR          | -0,037           | 17,3            | 0,069     | 0,101             | 105,8            | < 0,001    | 0,198               | 775,9              | 82,5          | < 0,001     |
| YPF           | 0,030            | 27,8            | 0,002     | 0,108             | 278,5            | < 0,001    | 0,203               | 1.146,9            | 183,3         | < 0,001     |
| TGS           | -0,049           | 28,3            | 0,002     | 0,067             | 65,5             | < 0,001    | 0,178               | 632,0              | 52,5          | < 0,001     |
| EDN           | 0,007            | 18,2            | 0,052     | 0,085             | 133,4            | < 0,001    | 0,214               | 863,3              | 101,3         | < 0,001     |
| TEO           | 0,006            | 11,8            | 0,298     | 0,160             | 224,2            | < 0,001    | 0,190               | 771,7              | 171,0         | < 0,001     |
| CRESY         | 0,012            | 11,3            | 0,337     | 0,072             | 140,7            | < 0,001    | 0,203               | 1.011,4            | 97,4          | < 0,001     |
| Cartera       | 0,035            | 28,0            | 0,002     | 0,050             | 56,1             | < 0,001    | 0,173               | 967,6              | 45,0          | < 0,001     |
| S&P 500 (SPY) | -0,105           | 93,2            | < 0,001   | 0,254             | 3.738,8          | < 0,001    | 0,313               | 5.581,9            | 1.251,7       | < 0,001     |

*Q(10): estadístico de Ljung-Box con 10 rezagos, $H_0$: sin autocorrelación hasta el rezago 10. ARCH-LM(10): regresión de $r_t^2$ sobre 10 rezagos propios, $H_0$: sin efectos ARCH.*

**Lectura.**

1. **Los retornos casi no tienen memoria.** La autocorrelación de primer orden de cada ADR está entre −0,05 y +0,03: no sirve para predecir la dirección del día siguiente. Ljung-Box rechaza al 5% en GGAL, YPF, TGS, la cartera y el SPY, pero hay que tomarlo con cautela. Con heterocedasticidad, el test de Ljung-Box estándar **rechaza de más**, porque los períodos de alta volatilidad inflan las autocorrelaciones muestrales. Antes de agregar términos ARMA conviene repetirlo sobre los residuos estandarizados del GARCH. La excepción es el SPY, con una autocorrelación de −0,105 en el primer rezago, un efecto conocido de reversión de corto plazo en índices.
2. **La magnitud de los retornos sí tiene memoria, y mucha.** Ljung-Box sobre los retornos al cuadrado rechaza con p < 0,001 en todas las series, y el ARCH-LM también (estadísticos de 45 a 183 en los activos argentinos). La autocorrelación del **valor absoluto** está entre 0,15 y 0,21 en el primer rezago y sigue siendo positiva 40 ruedas después (Figura 11): la volatilidad es **muy persistente**.
3. **Un detalle técnico sobre la cartera.** Su ACF de retornos al cuadrado es baja (0,05) porque un solo día, el −47% de las PASO, domina la serie de cuadrados: ese día al cuadrado pesa más que cientos de días normales y "aplana" las autocorrelaciones. Sin ese día, el ARCH-LM de la cartera sube de 45 a 144 y la ACF(1) de los cuadrados, de 0,05 a 0,08. El valor absoluto, menos sensible a un solo día, muestra la persistencia con claridad (0,17). Es otra muestra de cómo un evento político domina las estimaciones.

### Qué sugieren estos resultados para el modelo

| Hallazgo descriptivo | Implicancia para la especificación |
|---|---|
| Precios I(1), retornos I(0) | Modelar retornos, no precios |
| Media diaria ≈ 0 y autocorrelación pequeña | Ecuación de la media con constante o AR(1) a lo sumo; confirmar con Ljung-Box sobre residuos estandarizados |
| ARCH-LM y Ljung-Box sobre $r^2$ rechazan en todas las series | Hace falta varianza condicional: **GARCH(1,1)** como punto de partida |
| ACF de $\lvert r \rvert$ positiva hasta 40 rezagos | Se espera alta persistencia (α + β cercano a 1). Contrastar con IGARCH, que es el modelo implícito en la EWMA con λ = 0,94 que usa el estudio, y considerar componentes de memoria larga (FIGARCH) si la persistencia es extrema |
| Curtosis 14-30, ν de la t entre 3 y 4, días a más de 5 desvíos | Innovaciones **t-Student** (o t asimétrica) en lugar de normales; verificar si los residuos estandarizados siguen teniendo colas gruesas (lo que justificaría EVT sobre los residuos, como el modelo EVT filtrado del estudio) |
| Asimetría ≈ 0 en los ADRs; la negativa de la cartera viene de un solo día | Testear si existe efecto apalancamiento (**GJR-GARCH** o **EGARCH**). Por lo descriptivo, se espera que sea débil |
| Saltos políticos desde volatilidad baja (PASO, bonaerense, legislativas) | Ningún GARCH los anticipa. Considerar variables dummy de días electorales en la media y/o la varianza, o un modelo con saltos, y declarar el resto como límite del modelo |
| Volatilidad media creciente entre períodos y curtosis inestable | Testear estabilidad de parámetros (quiebre estructural) o un GARCH con cambio de régimen (Markov-switching) |
| Correlaciones que suben en crisis | Para modelar la cartera desde sus componentes, un GARCH multivariado con correlación dinámica (**DCC-GARCH**) en lugar de correlación constante |

**Propuesta para el capítulo siguiente.** Estimar sobre la muestra completa un **AR(0)-GARCH(1,1) con innovaciones t-Student** para la cartera. Compararlo con GJR-GARCH-t, EGARCH-t e IGARCH por AIC/BIC y por log-verosimilitud fuera de muestra. Diagnosticar los residuos estandarizados con Ljung-Box sobre $z$ y $z^2$, ARCH-LM y un QQ-plot contra la t. Repetirlo con y sin las PASO (o con una dummy) para medir cuánto depende la estimación de ese día. El estudio de VaR ya usa un GARCH(1,1)-t re-estimado cada 21 días: el capítulo econométrico le daría la base formal que hoy no tiene.

---

## 2.10 Limitaciones de este análisis descriptivo

- **Un solo día domina varios estadísticos.** La asimetría, la curtosis y la ACF de los cuadrados de la cartera cambian mucho con o sin el 12/08/2019. Se reportan con ese día incluido, porque es un dato real y el riesgo que el estudio quiere medir, y se aclara el efecto donde importa.
- **Precios con días sin operaciones.** Entre el 0,6% y el 4,4% de las ruedas tienen retorno exactamente cero, sobre todo en TGS y EDN. Puede tratarse de iliquidez o de precios sin actualizar: subestima la volatilidad en esos días y agrega un pico artificial en cero.
- **Supervivencia y selección.** Son empresas que hoy siguen cotizando, elegidas por quien hace el estudio. Un universo con las que dejaron de cotizar mostraría colas todavía más gruesas.
- **Correlación de Pearson.** Resume la dependencia lineal promedio. No describe la dependencia en las colas, que es la relevante para el VaR. Una medida de dependencia de cola (cópulas, coeficiente de cola) queda para el capítulo econométrico.
- **KPSS acotado.** Los p-valores de KPSS están truncados entre 0,01 y 0,10 por la tabla de valores críticos de statsmodels: "0,100" significa "≥ 0,10" y "0,010" significa "≤ 0,01".

---

### Archivos

```
estudio/descriptivo/
├── run_descriptivo.py        script reproducible (uv run python estudio/descriptivo/run_descriptivo.py)
├── ANALISIS_DESCRIPTIVO.md   este capítulo
├── tablas.md                 todas las tablas, generadas por el script
├── tablas/*.csv              las mismas tablas sin formato, para reutilizar
└── figuras/d01 … d13         13 figuras, mismo estilo que el resto del repo (src/quantkit/style.py)
```

### Referencias

- Cont, R. (2001), *Empirical properties of asset returns: stylized facts and statistical issues*, Quantitative Finance.
- Engle, R. (1982), *Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation*, Econometrica.
- Bollerslev, T. (1986), *Generalized autoregressive conditional heteroskedasticity*, Journal of Econometrics.
- Dickey, D. y Fuller, W. (1979), *Distribution of the estimators for autoregressive time series with a unit root*, JASA.
- Kwiatkowski, D., Phillips, P., Schmidt, P. y Shin, Y. (1992), *Testing the null hypothesis of stationarity against the alternative of a unit root*, Journal of Econometrics.
- Ljung, G. y Box, G. (1978), *On a measure of lack of fit in time series models*, Biometrika.
- Jarque, C. y Bera, A. (1980), *Efficient tests for normality, homoscedasticity and serial independence of regression residuals*, Economics Letters.
- J.P. Morgan/Reuters (1996), *RiskMetrics Technical Document*.
