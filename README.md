# code_v4. exp_geo, geometría para PHIL en mapa oculto

Código de la tesis sobre heurísticas aprendidas para búsqueda voraz best-first en grillas de 201 por 201 con mapa oculto. Esta carpeta corre el experimento exp_geo, una CNN que adivina los obstáculos no vistos y una heurística h_geo que la usa.txt.

## Qué hace falta para correrlo en otra máquina

1. Python 3.14 con los paquetes de requirements.txt, torch con MPS si es una Mac con Apple Silicon.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

2. Los mapas de motion_planning_datasets de Bhardwaj 2017 en data/motion_planning_datasets/<conjunto>/{train,validation,test}/*.png, con los conjuntos gaps_and_forest, multiple_bugtraps y mazes.

3. Los resultados de code_v3 que exp_geo lee sin copiar. Van en una carpeta code_v3 al lado de code_v4, o en cualquier lado si se apunta con la variable TESIS_CODE_V3.

| ruta dentro de code_v3 | qué es | peso |
|---|---|---|
| results/exp_cf/fotos/<conjunto>/hfs | fotos base de la voraz con h_fs en los 200 mapas de entrenamiento | 1.2 GB |
| results/exp_cf/refs | validación de h_fs y prueba de h_fs y h* | 100 KB |
| results/exp_cf2/cf_filtro/runs/<conjunto>/cf_filtro_s<k>/best.pt, config.json y train.csv | el modelo cf_filtro de cada semilla | 160 MB |
| results/exp_cf2/cf_filtro/eval | prueba de cf_filtro por mapa | 200 KB |

El archivo code_v3_refs_para_exp_geo.tar que se generó junto con este repo tiene todo eso más los mapas, y se desempaca al lado de code_v4.

```bash
cd <carpeta de la tesis>
tar xf code_v3_refs_para_exp_geo.tar
cd code_v4
ln -s ../code_v3/data data
```

Si code_v3 queda en otro lugar

```bash
export TESIS_CODE_V3=/ruta/a/code_v3
```

## Cómo se corre

Pruebas y humo primero, el maestro después. El maestro corre las pruebas y las verificaciones previas por su cuenta y no arranca nada si fallan.

```bash
.venv/bin/python -m tests.test_geo
```

```bash
bash experiments/humo_geo.sh
```

```bash
nohup caffeinate -dimsu bash experiments/run_geo.sh >> results/exp_geo/maestro.log 2>&1 &
```

En Linux no hay caffeinate, se quita de la orden, y con GPU de Nvidia se lanza con DEVICE=cuda. El maestro es relanzable con la misma orden, salta lo que ya tiene marcador listo o csv, cada entrenamiento sigue desde su last.pt y los logs solo se anexan. Variables que se pueden cambiar al lanzar, EPOCHS, DEVICE, SEEDS, LAMBDAS, DATASETS y PAR_BARRIDO.

## Qué hizo falta en Windows con Git Bash

Se corrió el 2026-11-05 en una laptop con Windows 11, 16 GB de RAM y una RTX 4070 de 8 GB. Todo lo demás es igual, con DEVICE=cuda y sin caffeinate.

- Si git tiene core.autocrlf en true el clon trae CRLF y bash no corre los .sh. Se pone en false en el repo y se normalizan los archivos con `git ls-files -z | xargs -0 sed -i 's/\r$//'`.
- Sin modo desarrollador no hay enlaces simbólicos. data y .venv/bin se crean como junctions, el segundo apunta a .venv/Scripts para que .venv/bin/python exista como en Mac.

```bash
cmd //c "mklink /J data ..\code_v3\data"
py -3.14 -m venv .venv
cmd //c "mklink /J .venv\bin .venv\Scripts"
.venv/bin/pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu130
.venv/bin/pip install -r requirements.txt
```

- Si el toolkit de CUDA del sistema está en el PATH, su cuDNN choca con la que trae torch y toda convolución en la GPU falla con CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH. Se quita del PATH del proceso con un sitecustomize.py dentro del venv que filtra las entradas con NVIDIA GPU Computing Toolkit. Es un ajuste de la máquina y no del repo.
- El maestro usa ps, vm_stat y sysctl para el estado y el vigía. En Git Bash no existen y run_geo.sh pregunta a WMI con powershell. En Linux usa /proc/vmstat y free.
- No hay caffeinate. experiments/despierto.py pide al sistema que no se suspenda y se va solo cuando estado.txt lleva 5 minutos sin reescribirse.

```bash
mkdir -p results/exp_geo/logs
DEVICE=cuda nohup bash experiments/run_geo.sh >> results/exp_geo/maestro.log 2>&1 &
nohup .venv/bin/python experiments/despierto.py results/exp_geo > results/exp_geo/logs/despierto.log 2>&1 &
```

## Dónde queda cada cosa

Todo en results/exp_geo. Fotos nuevas en fotos, entrenamientos en runs, barrido de lambda en barrido, prueba en eval, métricas de la CNN en metricas, verificaciones en checks, tablas en tablas.md y el borrador del informe en RESUMEN_auto.md. El RESUMEN.md se escribe a mano al final.
