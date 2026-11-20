# Ablación de la arquitectura de TransPath

Corrida del 2026-11-20 en la laptop Windows con la RTX 4070, con el código de new_exp. Todo número sale de resultados.csv, por_semilla.csv y los csv de train y eval de esta carpeta. La base cf_filtro sale de los csv de prueba de exp_cf2 en code_v3, con 5 semillas.

## Qué pregunta responde

La tesis usa una U-Net con atención en el cuello para estimar el factor de corrección. El factor de corrección viene de TransPath, así que la comparación natural con la literatura es poner su arquitectura en el lugar de la U-Net y ver si la elección de red cambia el resultado.

## Qué se comparó

- cf_filtro es la U-Net con atención en el cuello de 13x13 de la tesis.
- cf_transpath es el autoencoder del repositorio AIRI-Institute/TransPath con sus valores por defecto. Son 64 canales, 3 bajadas, 4 bloques transformer de 4 cabezas, dropout 0.15 y 949 569 parámetros. Lleva el mismo gauss fijo que cf_filtro y la última convolución arranca en cero para que la época 0 sea la voraz con h_fs.
- Las dos se entrenan igual. Misma imagen de 5 canales del mapa descubierto, misma etiqueta cf* = h_fs / h*, Adam 1e-3, lote 8, 10 épocas con fotos de h_fs y 2 rondas de DAgger de 10 épocas, validación cada 5 épocas, corte tras dos derrumbes seguidos y el gauss de desvío 16 en gaps y 40 en bugtraps. El entrenamiento original de TransPath con mapa completo, Adam 4e-4, OneCycle, lote 256 y 160 épocas no se usa, porque la pregunta es de arquitectura en las mismas condiciones.
- 3 semillas en gaps_and_forest y 3 en multiple_bugtraps, cada una evaluada en los 100 mapas de prueba (91 resolubles en gaps).

## Resultados

- En gaps_and_forest TransPath aprende casi lo mismo que la U-Net. Expande 0.516 veces lo de la voraz con h_fs y cf_filtro 0.488. La relación con cf_filtro es 1.057, con 1.043, 1.099 y 1.029 por semilla y Wilcoxon p 0.0000. Las 3 semillas bajaron en validación de 1579.2 a entre 820.3 y 898.8 expansiones y ninguna se cortó. La semilla 2 tuvo un derrumbe en la época 10 con 3775.2 expansiones y un mapa perdido, y se recuperó con DAgger.
- En multiple_bugtraps TransPath queda por detrás. Expande 0.962 veces lo de h_fs con p 0.0529 contra h_fs, mientras cf_filtro llega a 0.875. La relación con cf_filtro es 1.100, con 1.195, 1.056 y 1.048 por semilla y p 0.0000. En validación oscila cerca de los 1163.4 de h_fs sin una bajada sostenida, y las mejores épocas son la 25, la 5 y la 15. La semilla 0 queda peor que h_fs en prueba con 1.045.
- El camino sobre el óptimo es 1.044 en gaps y 1.188 en bugtraps, contra 1.053 y 1.194 de cf_filtro.
- El costo por nodo es 11.195 ms en gaps y 10.956 ms en bugtraps, contra 1.399 y 1.712 de cf_filtro, unas 6 a 8 veces más. Va aparte de las expansiones. Las 6 corridas tardaron 671 minutos con 2 a la vez en la GPU.

## Lectura con la regla del plan

La regla fijada antes de correr dice que TransPath gana si queda más de 10 % por debajo de cf_filtro con Wilcoxon p menor a 0.05, pierde si queda más de 10 % por encima con p menor a 0.05 y empata en otro caso. En gaps queda 5.7 % por encima, así que empata. En bugtraps la relación sin redondear es 1.0997, justo por debajo del umbral, así que la regla también da empate, aunque la diferencia es significativa y las 3 semillas quedan por encima de cf_filtro.

La lectura es que la arquitectura de la tesis no pierde contra la de TransPath en ningún conjunto y la supera en bugtraps casi por el margen fijado, con 6 a 8 veces menos costo por nodo. El aporte de la tesis no depende de una arquitectura rara. Con un autoencoder con transformer de la literatura el corrector también funciona en gaps, lo que apoya que la ganancia viene de la formulación del corrector sobre el mapa descubierto y del suavizado, y la U-Net con atención es la opción más barata y algo mejor.

## Archivos

- resultados.csv tiene una fila por conjunto para cf_filtro y TransPath, con expansiones medias, relaciones con la voraz con h_fs y con cf_filtro, sus p, el largo del camino sobre el óptimo y los ms por nodo.
- por_semilla.csv tiene una fila por corrida con las épocas corridas, la mejor época, su validación, si se cortó por derrumbe y las relaciones de prueba.
- eval tiene los csv de prueba mapa por mapa y train los train.csv de cada corrida.
- El código está en new_exp, la red en new_exp/redes.py, el plan en new_exp/PLAN.md y la exportación de esta carpeta en new_exp/exportar.py.
