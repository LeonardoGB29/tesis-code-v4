# Plan de new_exp en code_v4

Escrito el 2026-10-30 en la Mac dentro de code_v3 y rearmado el 2026-11-19 en la laptop Windows dentro de code_v4. El código salió de la conversación exportada de la Mac, archivo por archivo con sus dos ediciones. Quedan los 2 experimentos esenciales. Nada se lanza sin la luz verde del usuario.

## Qué se hace

| exp | pregunta                                                | modelo nuevo                                                                     | corridas                                  |
|-----|---------------------------------------------------------|----------------------------------------------------------------------------------|-------------------------------------------|
| 1   | importa que el corrector vea el mapa descubierto entero | cf_local_filtro, 5 convs 3x3 sin bajadas, campo de 11x11, con el mismo suavizado | 3 semillas en gaps y en bugtraps, 6 corridas |
| 2   | la U-Net con atención es la arquitectura adecuada       | cf_transpath, el autoencoder de TransPath con el mismo suavizado                 | 3 semillas en gaps y en bugtraps, 6 corridas |

El exp 3 del plan original, PHIL sobre el corrector en prueba, quedó cubierto por exp_escalera (R2 a R4) y no se corre. Mazes queda fuera porque ahí cf_filtro casi no se separa de la voraz con h_fs.

Todo usa el mismo protocolo que cf_filtro. Son las mismas imágenes de 5 canales, la etiqueta cf* = h_fs / h*, Adam con tasa 1e-3, lote de 8, 10 épocas con imágenes de h_fs y 2 rondas de 10 con imágenes propias, validación cada 5 épocas con la época 0 como candidata, el gauss fijo de desvío 16 y 40 dentro del modelo y la evaluación en los 100 mapas de prueba. Solo cambia la red.

## Por qué

- Exp 1. El título dice contexto espacial global y la única comparación entre local y global que hay no se puede leer, porque se hizo con el factor crudo y ahí todas las redes quedaron en 1.000. La red local usa 32 canales para tener el mismo cómputo por imagen que la U-Net, unos 2 GMAC, así la diferencia queda en el campo de visión y no en el tamaño.
- Exp 2. TransPath es de donde sale el factor de corrección, así que es la comparación natural con la literatura. Se copia su arquitectura del repositorio AIRI-Institute/TransPath con sus valores por defecto. Son 64 canales, 3 bajadas, 4 bloques transformer de 4 cabezas, dropout de 0.15 y 949 569 parámetros. El único cambio es que la última convolución arranca en cero, para que la época 0 sea la voraz con h_fs como en cf_filtro. Su entrenamiento original con mapa completo, Adam 4e-4, OneCycle, lote de 256 y 160 épocas no se usa, porque la comparación es de arquitectura en las mismas condiciones.

## Reglas de decisión, fijadas antes de correr

| exp | se lee como                 | condición                                                                                                                         |
|-----|-----------------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| 1   | lo global aporta            | el corrector local queda más de 10 % por encima de cf_filtro en expansiones, con Wilcoxon pareado p < 0.05, en gaps y en bugtraps |
| 1   | lo global aporta en parte   | lo mismo en uno de los dos conjuntos                                                                                              |
| 2   | TransPath gana o pierde     | más de 10 % por debajo o por encima de cf_filtro con Wilcoxon p < 0.05, si no empata                                              |

Además se aplica a los dos modelos nuevos la regla general de la tesis contra la voraz con h_fs. Las expansiones y el tiempo por nodo se reportan por separado.

## Cambios respecto de la versión de la Mac

- Rutas. Las fotos, las referencias y cf_filtro se leen de ../code_v3/results, igual que exp_geo y exp_escalera.
- Equipo. CUDA en vez de mps y experiments/despierto.py en vez de caffeinate.
- Suavizado. En la RTX 4070 cuDNN elige para el gradiente de la conv de 1x129 (desvío 16, gaps) un algoritmo de 620 ms por paso de lote 8, 18 veces todo lo demás. new_exp/redes.py usa SuavizadoMat, la misma operación escrita como producto de matrices. Da lo mismo en salida y en gradiente (prueba test_suavizado_igual, diferencia menor a 1e-10 en doble precisión) y guarda los mismos buffers. src no se toca y cf_filtro, entrenado en la Mac, no cambia.
- Procesos. 4 a la vez para la red local y 2 para TransPath, porque la 4070 tiene 8 GB.

## Tiempo estimado

Medido el 2026-11-19 en esta laptop. El paso de lote 8 tarda 27 ms la red local y 103 ms TransPath a solas (116 y 389 ms en la Mac). Una época completa de gaps, 5 982 imágenes, tardó 34 s la red local y 85 s TransPath con la carga de datos incluida. Cada corrida pasa por unas 300 000 imágenes entre las 30 épocas.

| ola | contenido                  | entrenamiento por corrida a solas | reloj estimado |
|-----|----------------------------|-----------------------------------|----------------|
| 1   | red local, 6 corridas      | unos 30 min más validación y DAgger | 2 a 3 h      |
| 2   | TransPath, 6 corridas      | unos 70 min más validación y DAgger | 6 a 8 h      |
| total |                          |                                   | 8 a 11 h       |

Los resultados del exp 1 quedan en new_exp/results/tablas.md al terminar la ola 1.

## Cómo se lanza

Desde code_v4 en Git Bash.

```bash
mkdir -p new_exp/results
DEVICE=cuda nohup bash new_exp/run.sh >> new_exp/results/maestro.log 2>&1 &
nohup .venv/bin/python experiments/despierto.py new_exp/results > new_exp/results/despierto.log 2>&1 &
```

El maestro corre las pruebas, verifica que estén las fotos, las referencias y los checkpoints de cf_filtro, y lanza las dos olas. Reescribe new_exp/results/estado.txt cada minuto. Es relanzable y salta lo que ya tiene listo.

## Qué no se toca

- ../code_v3/results solo se lee.
- experiments y src no se modifican. new_exp/train.py y new_exp/eval.py agregan los modelos nuevos desde afuera.
- La prueba de humo quedó en new_exp/results/humo y no cuenta como resultado.

## Archivos

| archivo     | qué hace                                                                                     |
|-------------|----------------------------------------------------------------------------------------------|
| redes.py    | la red local con suavizado, la arquitectura de TransPath y SuavizadoMat                      |
| train.py    | entrena con experiments.train_cf más los dos modelos nuevos                                  |
| eval.py     | evalúa en prueba con experiments.eval_cf                                                     |
| test_new.py | 6 pruebas, forma, época 0 constante, guardado y carga, gradiente, suavizado igual y época 0 igual a la voraz con h_fs |
| _run.sh     | un trabajo, entrenar y evaluar                                                               |
| run.sh      | el maestro de las dos olas                                                                   |
| resumen.py  | tablas contra la voraz con h_fs y contra cf_filtro, con la lectura de las reglas             |
