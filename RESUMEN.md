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

**Contactando al 5% de la cartera se alcanza el 38,3% de las contrataciones del mes: 7,6
veces lo que se conseguiría llamando al azar.**

En número de personas: llamando a 46.993 clientes de los 926.663 que hay, se llega a más de
un tercio de todos los que iban a contratar algo ese mes. Si esas mismas 46.993 llamadas se
reparten al azar, se llega al 5%.

Y si el presupuesto da para menos, la ventaja es todavía mayor: llamando solo al 1% se
alcanza el 17% de las contrataciones, casi **16 veces** el azar. Cuanto más escaso es el
tiempo del equipo comercial, más vale ordenar bien la lista.

Por el lado de las bajas, el modelo detecta 14 veces mejor que el azar quién va a cancelar.
Y el punto de corte — a partir de qué nivel de riesgo merece la pena llamar — no se deja en
el 0,5 que viene por defecto, sino que se calcula con lo que cuesta un contacto y lo que
vale retener a alguien. Esa diferencia sola vale **187.877 €** en el mes medido.

Ese multiplicador es todo el valor del sistema, y es una cifra que un director comercial
entiende sin que nadie le explique qué es un árbol de decisión.

## Una cosa que salió mal, y que cuenta más que las que salieron bien

Al final del proyecto se intentó exprimir el mejor modelo: probar 12 combinaciones de
ajustes, cambiar cómo aprende, añadir técnicas que suelen funcionar. El resultado mejoró un
**0,02%**.

Antes de cantar victoria se hizo una comprobación que mucha gente se salta: entrenar el
**mismo** modelo tres veces cambiando solo el número aleatorio con el que arranca. Esas tres
versiones idénticas se diferenciaron entre sí **cinco veces más** que lo que había mejorado
el ajuste.

Traducido: la mejora no existía. Era ruido.

Lo que sí funcionó fue mucho más simple — añadir una variable en la que nadie había pensado:
cuántos meses lleva el cliente con *ese* producto concreto. Eso solo mejoró la detección de
bajas un 1,8%, noventa veces más que toda la optimización.

Saberlo distinguir es la diferencia entre un informe honesto y uno que promete algo que no
va a pasar cuando el sistema entre en producción.

## Lo que este proyecto **no** hace

- No decide nada solo: **ordena una lista**, y la llamada la hace una persona.
- No usa datos personales sensibles. Trabaja con producto contratado, antigüedad, segmento y
  provincia.
- No promete un número de euros. La cifra de ahorro depende de supuestos de coste que aquí
  están escritos de forma explícita para poder discutirlos, no escondidos en el código.
