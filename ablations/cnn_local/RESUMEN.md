# Ablación de la CNN local 11x11

Corrida del 2026-11-19 al 2026-11-20 en la laptop Windows con la RTX 4070, con el código de new_exp. Todo número sale de resultados.csv, por_semilla.csv y los csv de train y eval de esta carpeta. La base cf_filtro sale de los csv de prueba de exp_cf2 en code_v3, con 5 semillas.

## Qué pregunta responde

El título y la hipótesis hablan de contexto espacial global. Esta ablación mide si hace falta que el corrector vea el mapa descubierto entero o si basta con ver cerca de cada celda.

## Qué se comparó

- cf_filtro es la U-Net con atención en el cuello de 13x13, que puede relacionar cualquier par de celdas del mapa.
- cf_local_filtro son 5 convoluciones 3x3 sin submuestreo, así cada celda solo ve un vecindario de 11x11. Tiene 32 canales para gastar el mismo cómputo por imagen que la U-Net, unos 2 GMAC, y así la diferencia queda en cuánto ve y no en el tamaño.
- Las dos se entrenan igual. Misma imagen de 5 canales del mapa descubierto, misma etiqueta cf* = h_fs / h*, Adam 1e-3, lote 8, 10 épocas con fotos de h_fs y 2 rondas de DAgger de 10 épocas, validación cada 5 épocas, corte tras dos derrumbes seguidos, el mismo gauss fijo de desvío 16 en gaps y 40 en bugtraps, y la última capa en cero para que la época 0 sea la voraz con h_fs.
- 3 semillas en gaps_and_forest y 3 en multiple_bugtraps, cada una evaluada en los 100 mapas de prueba (91 resolubles en gaps).

## Resultados

- En multiple_bugtraps la red local empata con la U-Net. Expande 0.858 veces lo de la voraz con h_fs y cf_filtro 0.875. La relación con cf_filtro es 0.980, con 0.994, 0.975 y 0.971 por semilla y Wilcoxon p 0.2926. Las 3 semillas mejoraron en validación de 1163.4 a entre 977.3 y 995.9 expansiones y eligieron la época 30.
- En gaps_and_forest la red local no aprende nada útil. Las 3 semillas se quedaron con la época 0, que es la voraz con h_fs, y en prueba expanden 1.000 veces lo de h_fs contra 0.488 de cf_filtro. La relación con cf_filtro es 2.047 con p 0.0000. Las semillas 0 y 2 se cortaron por dos derrumbes seguidos en las épocas 20 y 10, con validaciones de hasta 5428.7 y 7002.0 expansiones contra 1579.2 de h_fs. La semilla 1 llegó a la época 30 sin bajar nunca de 1579.2, su mejor validación aprendida fue 2070.8.
- El costo por nodo de la red local es 2.458 ms en gaps y 2.824 ms en bugtraps, contra 1.399 y 1.712 de cf_filtro. Va aparte de las expansiones.

## Lectura con la regla del plan

La regla fijada antes de correr dice que lo global aporta si la red local queda más de 10 % por encima de cf_filtro con Wilcoxon p menor a 0.05 en gaps y en bugtraps, y que aporta en parte si eso pasa en uno solo. Pasa en gaps y no en bugtraps, así que lo global aporta en parte.

La lectura es coherente con los mapas. En bugtraps lo que decide la búsqueda es la trampa que tiene delante, que entra en un vecindario de 11x11. En gaps hay que saber hacia dónde se abre el paso entre obstáculos lejanos, y una red que solo ve 5 celdas a cada lado no tiene con qué estimar cuánto se desvía el camino real de la línea recta. La palabra global del título queda sostenida por gaps, que es justo el conjunto donde cf_filtro gana más.

## Archivos

- resultados.csv tiene una fila por conjunto para cf_filtro y la red local, con expansiones medias, relaciones con la voraz con h_fs y con cf_filtro, sus p, el largo del camino sobre el óptimo y los ms por nodo.
- por_semilla.csv tiene una fila por corrida con las épocas corridas, la mejor época, su validación, si se cortó por derrumbe y las relaciones de prueba.
- eval tiene los csv de prueba mapa por mapa y train los train.csv de cada corrida.
- El código está en new_exp, el plan en new_exp/PLAN.md y la exportación de esta carpeta en new_exp/exportar.py.
