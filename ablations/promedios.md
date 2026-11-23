# Arquitectura de la CNN, promedios por mapa

Prueba en todos los mapas resolubles, 91 en gaps_and_forest, 100 en multiple_bugtraps, 100 en mazes. Los 6 modelos resuelven el 100 % de los mapas. Media de las semillas 0 a 2, con un tope de 10 000 expansiones. El óptimo sale de una búsqueda en anchura desde la meta sobre el mapa real. Generado por experiments/tabla_arquitectura.py.

## Nodos expandidos, promedio por mapa

| modelo                         | gaps_and_forest | multiple_bugtraps | mazes   |
|--------------------------------|-----------------|-------------------|---------|
| Voraz con h_fs (referencia)    | 1603.967        | 1127.270          | 442.400 |
| U-Net con atención (cf_filtro) | 810.538         | 992.270           | 447.560 |
| U-Net sin atención             | 822.597         | 1070.103          | 438.943 |
| Solo codificador               | 838.275         | 1047.830          | 445.040 |
| TransPath                      | 828.106         | 1084.703          | 442.400 |
| CNN local 11x11                | 1603.967        | 966.757           | 442.400 |

## Largo medio del camino, en pasos

| modelo                         | gaps_and_forest | multiple_bugtraps | mazes   |
|--------------------------------|-----------------|-------------------|---------|
| Óptimo (BFS)                   | 472.978         | 267.660           | 266.710 |
| Voraz con h_fs (referencia)    | 600.527         | 324.610           | 291.840 |
| U-Net con atención (cf_filtro) | 496.516         | 323.733           | 288.473 |
| U-Net sin atención             | 501.560         | 323.100           | 290.033 |
| Solo codificador               | 506.443         | 325.763           | 289.103 |
| TransPath                      | 493.440         | 321.730           | 291.840 |
| CNN local 11x11                | 600.527         | 322.983           | 291.840 |

## Caminos óptimos, en porcentaje

| modelo                         | gaps_and_forest | multiple_bugtraps | mazes |
|--------------------------------|-----------------|-------------------|-------|
| Voraz con h_fs (referencia)    | 0.00            | 13.00             | 35.00 |
| U-Net con atención (cf_filtro) | 2.93            | 14.00             | 47.33 |
| U-Net sin atención             | 0.00            | 14.00             | 43.00 |
| Solo codificador               | 0.37            | 13.00             | 38.00 |
| TransPath                      | 1.10            | 15.33             | 35.00 |
| CNN local 11x11                | 0.00            | 14.00             | 35.00 |
