# Ablación de la U-Net sin atención

Corrida del 2026-11-20 en la laptop Windows con la RTX 4070, con el código de new_exp. Todo número sale de resultados.csv, por_semilla.csv y los csv de train y eval de esta carpeta. La base cf_filtro sale de los csv de prueba de exp_cf2 en code_v3, con 5 semillas en gaps y bugtraps y 3 en mazes.

## Qué pregunta responde

cf_filtro tiene 2 bloques de atención en el cuello de 13x13, que dejan relacionar cualquier par de zonas del mapa en una sola capa. Esta ablación mide si esa atención aporta o si la U-Net convolucional sola alcanza.

## Qué se comparó

- cf_filtro es la U-Net con atención de la tesis, con ancho 16 y 3 042 369 parámetros.
- cf_sin_atencion es la misma U-Net con el mismo ancho, sin los 2 bloques de atención y sin la posición aprendida del cuello, que sin atención solo sería un sesgo fijo por celda. Tiene 1 944 897 parámetros. El cuello convolucional de 13x13 sigue viendo casi todo el mapa por las 4 bajadas, así que la diferencia es la atención y no el campo receptivo.
- Las dos se entrenan igual. Misma imagen de 5 canales del mapa descubierto, misma etiqueta cf* = h_fs / h*, Adam 1e-3, lote 8, 10 épocas con fotos de h_fs y 2 rondas de DAgger de 10 épocas, validación cada 5 épocas, corte tras dos derrumbes seguidos, el gauss fijo de desvío 16 en gaps, 40 en bugtraps y 12 en mazes, y la última capa en cero para que la época 0 sea la voraz con h_fs.
- 3 semillas por conjunto, cada una evaluada en los 100 mapas de prueba (91 resolubles en gaps).

## Las dos métricas

- Expansiones. La relación es expansiones sobre las de la referencia, menos de 1 es mejor.
- Camino. Es el largo del camino sobre el óptimo, 1 es el camino más corto posible. La comparación con cf_filtro se hace en los mapas que resuelven las dos redes con todas sus semillas.
- Las dos llevan Wilcoxon pareado por mapa sobre la media de las semillas. Todas las corridas resuelven todos los mapas resolubles.

## Resultados

- En gaps_and_forest la U-Net sin atención aprende casi lo mismo. Expande 0.513 veces lo de la voraz con h_fs y cf_filtro 0.488. La relación con cf_filtro es 1.050, con 1.066, 1.068 y 1.016 por semilla y p 0.0522. El camino es 1.061 contra 1.053, una relación de 1.008 con p 0.0000. Las 3 semillas bajaron en validación de 1579.2 a entre 779.1 y 874.7 expansiones y ninguna se cortó.
- En multiple_bugtraps se vuelve inestable. La semilla 0 aprende y llega a 0.848 veces h_fs, mejor que cf_filtro. Las semillas 1 y 2 se quedan con la época 0, que es h_fs, y la semilla 2 se cortó por dos derrumbes en la época 10. En conjunto expande 0.949 veces lo de h_fs y la relación con cf_filtro es 1.085, con 0.969, 1.143 y 1.143 por semilla y p 0.0000. El camino es 1.193 contra 1.194, una relación de 0.999 con p 0.0003, es decir igual.
- En mazes dos semillas se quedan con la época 0 y la semilla 2 baja a 0.977 veces h_fs. En conjunto expande 0.992 veces lo de h_fs, y la relación con cf_filtro es 0.981 con p 0.0054. El camino es 1.077 contra 1.070, una relación de 1.006 con p 0.0233.
- El costo por nodo es 3.837 ms en gaps, 4.659 en bugtraps y 3.946 en mazes, contra 1.399, 1.712 y 1.477 de cf_filtro. Esta red cuesta menos que cf_filtro por paso de entrenamiento, 31 contra 35 ms con lote 8 en la 4070. La diferencia en ms por nodo viene de que cf_filtro se midió en la Mac y esta red aquí con 4 corridas a la vez, así que esta columna no sirve para comparar costos.

## Lectura

El plan no fijó una regla para esta variante. Leída con la misma regla que TransPath, más de 10 % arriba o abajo de cf_filtro con p menor a 0.05, empata en los 3 conjuntos. Queda 5.0 % por encima en gaps y 8.5 % en bugtraps.

Lo que cambia sin atención es la estabilidad más que el promedio. Con atención las 5 semillas de cf_filtro bajan de h_fs en bugtraps, y sin atención 2 de 3 no salen de la época 0. Cuando aprende, como la semilla 0 de bugtraps, llega al mismo nivel que cf_filtro. La atención no es lo que hace funcionar al corrector, eso lo hace la U-Net con el suavizado, pero lo vuelve más confiable entre semillas por un costo bajo, 2 capas sobre 169 posiciones. En camino las dos son iguales en todos los conjuntos, con diferencias de menos de 1 %.

## Archivos

- resultados.csv tiene una fila por conjunto para cf_filtro y la U-Net sin atención, con el éxito, las expansiones, sus relaciones con la voraz con h_fs y con cf_filtro, el camino sobre el óptimo, el porcentaje de caminos óptimos, el camino contra h_fs, el camino en los mapas comunes con cf_filtro, sus p y los ms por nodo.
- por_semilla.csv tiene una fila por corrida con las épocas corridas, la mejor época, su validación, si se cortó por derrumbe y las relaciones de prueba.
- eval tiene los csv de prueba mapa por mapa y train los train.csv de cada corrida.
- La red es UNetSinAtencion en new_exp/redes.py y la exportación de esta carpeta sale de new_exp/exportar.py.
