# Ablaciones del corrector cf_filtro

Cuatro variantes de la red que estima el factor de corrección, cada una entrenada con la receta de cf_filtro, 3 semillas y evaluada en los 100 mapas de prueba de gaps_and_forest, multiple_bugtraps y mazes. Cada carpeta tiene su resultados.csv, por_semilla.csv, los csv de prueba y entrenamiento y un RESUMEN.md con la explicación. Se miden dos métricas, las expansiones y el largo del camino sobre el óptimo, las dos contra cf_filtro con Wilcoxon pareado por mapa.

| carpeta | qué cambia frente a cf_filtro | pregunta |
|---|---|---|
| cnn_local | 5 convoluciones 3x3 sin submuestreo, campo de 11x11 | si hace falta ver el mapa entero |
| transpath | el autoencoder de TransPath | si una arquitectura publicada rinde igual |
| unet_sin_atencion | sin la atención del cuello | si la atención aporta |
| solo_codificador | sin decodificador ni conexiones de salto | si hace falta el detalle fino |

## Resumen

Ninguna variante supera a cf_filtro en ningún conjunto en ninguna de las dos métricas por más de 2 %.

- cnn_local es la única que pierde por la regla del plan. En gaps no aprende, expande 2.047 veces lo de cf_filtro y su camino es 1.208 veces el de cf_filtro. En bugtraps empata. Lo global hace falta donde los obstáculos que deciden el camino están lejos.
- transpath empata en los 3 conjuntos, con 1.057 y 1.100 en expansiones en gaps y bugtraps, y no pierde en camino. Cuesta unas 6 a 8 veces más por nodo.
- unet_sin_atencion y solo_codificador empatan por la misma regla, entre 5 y 9 % por encima en expansiones en gaps y bugtraps. Las dos pierden estabilidad, porque en bugtraps 2 de 3 y 1 de 3 semillas se quedan en la voraz con h_fs. solo_codificador además pierde algo de camino, 1.8 % en gaps.
- En mazes nada baja de h_fs, ni cf_filtro ni las variantes, y todas quedan a poco más de 1 % de cf_filtro en las dos métricas.

La U-Net con atención es la opción que mejor rinde y la más estable entre las probadas, y la ganancia del corrector no depende de una arquitectura en particular. Cada pieza, el campo global, la atención y el decodificador, suma una parte y ninguna sola explica el resultado.
