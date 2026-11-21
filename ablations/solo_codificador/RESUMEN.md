# Ablación del codificador solo

Corrida del 2026-11-20 en la laptop Windows con la RTX 4070, con el código de new_exp. Todo número sale de resultados.csv, por_semilla.csv y los csv de train y eval de esta carpeta. La base cf_filtro sale de los csv de prueba de exp_cf2 en code_v3, con 5 semillas en gaps y bugtraps y 3 en mazes.

## Qué pregunta responde

La pregunta es por qué una U-Net y no un codificador solo. Un codificador entrega una grilla gruesa y para dar un factor por celda hay que interpolarla. Esta ablación mide si el decodificador con sus conexiones de salto, que devuelven el detalle fino, hace falta.

## Qué se comparó

- cf_filtro es la U-Net con atención de la tesis, con ancho 16 y 3 042 369 parámetros.
- cf_codificador es el codificador de cf_filtro con su atención en el cuello, sin decodificador ni conexiones de salto. Una capa 1x1 que arranca en cero da el factor en el cuello de 13x13, se interpola bilineal a 208x208 y pasa por el mismo gauss fijo. Tiene 2 278 769 parámetros.
- Las dos se entrenan igual. Misma imagen de 5 canales del mapa descubierto, misma etiqueta cf* = h_fs / h*, Adam 1e-3, lote 8, 10 épocas con fotos de h_fs y 2 rondas de DAgger de 10 épocas, validación cada 5 épocas, corte tras dos derrumbes seguidos, el gauss fijo de desvío 16 en gaps, 40 en bugtraps y 12 en mazes, y la época 0 igual a la voraz con h_fs.
- 3 semillas por conjunto, cada una evaluada en los 100 mapas de prueba (91 resolubles en gaps).

## Las dos métricas

- Expansiones. La relación es expansiones sobre las de la referencia, menos de 1 es mejor.
- Camino. Es el largo del camino sobre el óptimo, 1 es el camino más corto posible. La comparación con cf_filtro se hace en los mapas que resuelven las dos redes con todas sus semillas.
- Las dos llevan Wilcoxon pareado por mapa sobre la media de las semillas. Todas las corridas resuelven todos los mapas resolubles.

## Resultados

- En gaps_and_forest el codificador solo aprende casi lo mismo. Expande 0.523 veces lo de la voraz con h_fs y cf_filtro 0.488. La relación con cf_filtro es 1.070, con 1.022, 1.095 y 1.092 por semilla y p 0.0058. El camino es 1.071 contra 1.053, una relación de 1.018 con p 0.0000. Las 3 semillas bajaron en validación de 1579.2 a entre 787.6 y 843.6 expansiones y ninguna se cortó.
- En multiple_bugtraps es inestable. La semilla 2 llega a 0.864 veces h_fs, al nivel de cf_filtro, la semilla 0 a 0.924 y la semilla 1 se queda con la época 0. En conjunto expande 0.930 veces lo de h_fs y la relación con cf_filtro es 1.062, con 1.056, 1.143 y 0.988 por semilla y p 0.0000. El camino es 1.203 contra 1.194, una relación de 1.007 con p 0.0027.
- En mazes empata con h_fs y con cf_filtro. Expande 1.006 veces lo de h_fs, la relación con cf_filtro es 0.994 con p 0.0120, y el camino es 1.074 contra 1.070, una relación de 1.004 con p 0.3131.
- El costo por nodo es 1.932 ms en gaps, 2.343 en bugtraps y 2.111 en mazes, contra 1.399, 1.712 y 1.477 de cf_filtro. Por paso de entrenamiento cuesta menos de la mitad que cf_filtro, 13 contra 35 ms con lote 8 en la 4070. cf_filtro se midió en la Mac y esta red aquí con 4 corridas a la vez, así que la columna de ms por nodo no sirve para comparar costos.

## Lectura

El plan no fijó una regla para esta variante. Leída con la misma regla que TransPath, más de 10 % arriba o abajo de cf_filtro con p menor a 0.05, empata en los 3 conjuntos. Queda 7.0 % por encima en gaps y 6.2 % en bugtraps, y en los dos la diferencia es significativa.

El codificador solo es la variante que más pierde en camino, 1.8 % en gaps y 0.7 % en bugtraps, las dos significativas. Es lo esperable sin conexiones de salto. El factor sale de una grilla de 13x13, cada valor cubre unas 16x16 celdas, y el detalle fino junto a las paredes vistas se pierde. Coincide con cf_grueso de exp_cf2, que da la salida en 26x26 desde el decodificador y no bajó nada en bugtraps. El decodificador no es lo que hace funcionar al corrector, pero mejora las dos métricas y la estabilidad entre semillas, y por eso la tesis usa la U-Net completa.

## Archivos

- resultados.csv tiene una fila por conjunto para cf_filtro y el codificador solo, con el éxito, las expansiones, sus relaciones con la voraz con h_fs y con cf_filtro, el camino sobre el óptimo, el porcentaje de caminos óptimos, el camino contra h_fs, el camino en los mapas comunes con cf_filtro, sus p y los ms por nodo.
- por_semilla.csv tiene una fila por corrida con las épocas corridas, la mejor época, su validación, si se cortó por derrumbe y las relaciones de prueba.
- eval tiene los csv de prueba mapa por mapa y train los train.csv de cada corrida.
- La red es SoloCodificador en new_exp/redes.py y la exportación de esta carpeta sale de new_exp/exportar.py.
