# Ablación de la CNN local 11x11

Corrida del 2026-11-19 al 2026-11-20 en la laptop Windows con la RTX 4070, con el código de new_exp. gaps y bugtraps corrieron en la primera tanda y mazes en la segunda. Todo número sale de resultados.csv, por_semilla.csv y los csv de train y eval de esta carpeta. La base cf_filtro sale de los csv de prueba de exp_cf2 en code_v3, con 5 semillas en gaps y bugtraps y 3 en mazes.

## Qué pregunta responde

El título y la hipótesis hablan de contexto espacial global. Esta ablación mide si hace falta que el corrector vea el mapa descubierto entero o si basta con ver cerca de cada celda.

## Qué se comparó

- cf_filtro es la U-Net con atención en el cuello de 13x13, que puede relacionar cualquier par de celdas del mapa.
- cf_local_filtro son 5 convoluciones 3x3 sin submuestreo, así cada celda solo ve un vecindario de 11x11. Tiene 32 canales para gastar el mismo cómputo por imagen que la U-Net, unos 2 GMAC, y así la diferencia queda en cuánto ve y no en el tamaño.
- Las dos se entrenan igual. Misma imagen de 5 canales del mapa descubierto, misma etiqueta cf* = h_fs / h*, Adam 1e-3, lote 8, 10 épocas con fotos de h_fs y 2 rondas de DAgger de 10 épocas, validación cada 5 épocas, corte tras dos derrumbes seguidos, el mismo gauss fijo de desvío 16 en gaps, 40 en bugtraps y 12 en mazes, y la última capa en cero para que la época 0 sea la voraz con h_fs.
- 3 semillas por conjunto, cada una evaluada en los 100 mapas de prueba (91 resolubles en gaps).

## Las dos métricas

- Expansiones. La relación es expansiones sobre las de la referencia, menos de 1 es mejor.
- Camino. Es el largo del camino sobre el óptimo, 1 es el camino más corto posible. La comparación con cf_filtro se hace en los mapas que resuelven las dos redes con todas sus semillas.
- Las dos llevan Wilcoxon pareado por mapa sobre la media de las semillas. Todas las corridas resuelven todos los mapas resolubles.

## Resultados

- En gaps_and_forest la red local no aprende nada útil. Las 3 semillas se quedaron con la época 0, que es la voraz con h_fs, y en prueba expanden 1.000 veces lo de h_fs contra 0.488 de cf_filtro. La relación con cf_filtro es 2.047 con p 0.0000. El camino también empeora, 1.271 contra 1.053 de cf_filtro, una relación de 1.208 con p 0.0000. Las semillas 0 y 2 se cortaron por dos derrumbes seguidos en las épocas 20 y 10, con validaciones de hasta 5428.7 y 7002.0 expansiones contra 1579.2 de h_fs. La semilla 1 llegó a la época 30 sin bajar nunca de 1579.2, y su mejor validación aprendida fue 2070.8.
- En multiple_bugtraps la red local empata con la U-Net. Expande 0.858 veces lo de h_fs y cf_filtro 0.875. La relación con cf_filtro es 0.980, con 0.994, 0.975 y 0.971 por semilla y p 0.2926. El camino es 1.192 contra 1.194, una relación de 0.998 que es significativa con p 0.0014 pero de 0.2 %. Las 3 semillas mejoraron en validación de 1163.4 a entre 977.3 y 995.9 expansiones y eligieron la época 30.
- En mazes la red local no aprende, igual que en gaps. Las 3 semillas se quedaron con la época 0 y en prueba son la voraz con h_fs. cf_filtro en mazes queda 1.012 veces h_fs, así que la relación con cf_filtro es 0.988 con p 0.0018, y en camino 1.084 contra 1.070, una relación de 1.013 con p 0.0018. Las dos diferencias son de poco más de 1 %.
- El costo por nodo de la red local es 2.458 ms en gaps, 2.824 en bugtraps y 3.873 en mazes, contra 1.399, 1.712 y 1.477 de cf_filtro. cf_filtro se midió en la Mac y la red local aquí con 4 corridas a la vez, así que esta comparación de tiempo es solo orientativa.

## Lectura con la regla del plan

La regla fijada antes de correr dice que lo global aporta si la red local queda más de 10 % por encima de cf_filtro con Wilcoxon p menor a 0.05 en gaps y en bugtraps, y que aporta en parte si eso pasa en uno solo. Pasa en gaps y no en bugtraps, así que lo global aporta en parte. mazes no entra en la regla porque ahí nada baja de h_fs, y la red local tampoco.

La lectura es coherente con los mapas. En bugtraps lo que decide la búsqueda es la trampa que tiene delante, que entra en un vecindario de 11x11. En gaps hay que saber hacia dónde se abre el paso entre obstáculos lejanos, y una red que solo ve 5 celdas a cada lado no tiene con qué estimar cuánto se desvía el camino real de la línea recta. Ahí pierde en las dos métricas, expande el doble y su camino es 21 % más largo. La palabra global del título queda sostenida por gaps, que es justo el conjunto donde cf_filtro gana más.

## Archivos

- resultados.csv tiene una fila por conjunto para cf_filtro y la red local, con el éxito, las expansiones, sus relaciones con la voraz con h_fs y con cf_filtro, el camino sobre el óptimo, el porcentaje de caminos óptimos, el camino contra h_fs, el camino en los mapas comunes con cf_filtro, sus p y los ms por nodo.
- por_semilla.csv tiene una fila por corrida con las épocas corridas, la mejor época, su validación, si se cortó por derrumbe y las relaciones de prueba.
- eval tiene los csv de prueba mapa por mapa y train los train.csv de cada corrida.
- pipeline.png muestra los pasos de la red local, el entrenamiento fuera de la búsqueda y el uso dentro de la voraz, con miniaturas reales del mapa 900 de prueba de multiple_bugtraps y la semilla 0. Sale de new_exp/figura_pipeline.py.
- El código está en new_exp, el plan en new_exp/PLAN.md y la exportación de esta carpeta en new_exp/exportar.py.
