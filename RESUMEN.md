# En cristiano: qué es esto

> Una página. Sin jerga. Para entender de qué va el proyecto sin saber programar.

## El problema

Un banco tiene **1,5 millones de clientes**. Su equipo comercial puede llamar, como mucho, a
unos pocos miles al mes.

¿A quién llama?

La respuesta típica es "a los de siempre" o "a los de la lista que pasó el jefe". Y la lista
casi nunca está ordenada por nada en particular. Llamar al azar a 50.000 personas para vender
un plan de pensiones significa molestar a 49.000 que no lo quieren, quemar la paciencia del
cliente y gastar el presupuesto del mes.

## Qué hace este sistema

Tres cosas, en este orden:

1. **Predice qué producto va a contratar cada cliente el mes que viene.** Mira su historial —
   qué tiene contratado, desde cuándo, qué ha ido moviendo últimamente — y estima qué es lo
   siguiente que le encaja.

2. **Predice a quién está a punto de perder.** El mismo historial, leído al revés: qué señales
   aparecen en los meses previos a que alguien cancele un producto.

3. **Ordena la lista de llamadas.** Junta las dos cosas anteriores y devuelve la cartera
   ordenada, de forma que el equipo comercial empiece por arriba y pare cuando se le acabe el
   tiempo.

Esa tercera parte es la importante. Un modelo que acierta mucho pero devuelve un número entre
0 y 1 no le sirve a nadie. Lo que sirve es una lista con nombres.

## La parte contraintuitiva: vender y retener no son lo mismo

Cruzar las dos predicciones parte la cartera en cuatro grupos, y **cada uno necesita una
llamada distinta**:

|  | Poco riesgo de irse | Mucho riesgo de irse |
|---|---|---|
| **Probable que compre** | Véndele | Reténle **primero**, ya le venderás |
| **Poco probable que compre** | No gastes la llamada | Campaña de retención |

La casilla de arriba a la derecha es la que justifica todo el cruce. Un cliente con muchas
papeletas de comprar *y* muchas de marcharse: si le llamas para venderle algo, es probable que
sea la última conversación que tengas con él. Con un solo modelo, ese cliente aparece arriba
del todo en la lista de ventas y nadie se entera del riesgo.

## Por qué es difícil (y por qué no es un Excel)

**Por el tamaño.** Son 13,6 millones de filas: cada cliente, cada mes, durante año y medio.
Eso no cabe en una hoja de cálculo ni en la memoria de un ordenador normal. Hace falta una
herramienta que reparta el trabajo en trozos — aquí, Spark.

**Y por la trampa.** Los datos traen, para cada cliente y cada mes, qué productos tenía ese
mes. Es tentador usar esa información para predecir qué contrata ese mismo mes… pero esa
columna *ya contiene la respuesta*. Un modelo entrenado así acierta casi el 100% y no vale
absolutamente para nada, porque en la vida real, cuando decides a quién llamar, todavía no
sabes qué va a pasar.

Todo el proyecto está construido alrededor de esa separación: **las variables describen el mes
anterior, la respuesta es el mes actual, y nunca se mezclan.** Hay incluso un experimento que
hace trampas a propósito, solo para demostrar lo grande que es el salto y que aquí no lo hay.

## Qué herramientas se usan y por qué

| Herramienta | Para qué | Por qué esa |
|---|---|---|
| **PySpark** | Procesar los 13,6 millones de filas | Reparte el trabajo entre varios procesadores. Sin esto, no cabe en memoria |
| **SQL** | Agrupar, cruzar y resumir los datos | Es el lenguaje natural para preguntarle cosas a una tabla |
| **Machine learning** | Las dos predicciones | Un árbol de decisión encuentra patrones que nadie programó a mano |
| **Docker** | Que todo funcione igual en cualquier máquina | Fija las versiones exactas y evita el "en mi ordenador funcionaba" |

## El resultado

<!-- Se rellena al ejecutar sobre el dataset completo (src/business.py lo calcula). -->
*Pendiente de ejecutar sobre los datos reales.*

La frase será de esta forma: **"contactando al X% de la cartera se alcanza el Y% de las
contrataciones del mes, frente al X% que se conseguiría llamando al azar"**.

Ese multiplicador es todo el valor del sistema, y es una cifra que un director comercial
entiende sin que nadie le explique qué es un árbol de decisión.

## Lo que este proyecto **no** hace

- No decide nada solo: **ordena una lista**, y la llamada la hace una persona.
- No usa datos personales sensibles. Trabaja con producto contratado, antigüedad, segmento y
  provincia.
- No promete un número de euros. La cifra de ahorro depende de supuestos de coste que aquí
  están escritos de forma explícita para poder discutirlos, no escondidos en el código.
