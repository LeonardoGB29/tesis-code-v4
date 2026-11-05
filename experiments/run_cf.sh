#!/bin/bash
# maestro del experimento cf; relanzable, no repite nada que ya este listo
#   nohup caffeinate -dimsu bash experiments/run_cf.sh > results/exp_cf/maestro.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
ROOT=${ROOT:-results/exp_cf}
export ROOT DEVICE=${DEVICE:-mps} EPOCHS=${EPOCHS:-30} RUIDO=${RUIDO:-0.01} EXTRA=${EXTRA:-}
PARALLEL=${PARALLEL:-10}
DATASETS=(gaps_and_forest multiple_bugtraps mazes)
SEEDS5="0 1 2 3 4"; SEEDS3="0 1 2"
mkdir -p $ROOT/checks $ROOT/logs
log() { echo "[maestro $(date '+%Y-%m-%d %H:%M:%S')] $*"; }
T0=$(date +%s)
log "inicio"

log "pruebas"
$PY -m tests.test_cf > $ROOT/checks/tests.log 2>&1 && log "pruebas ok" || { log "FALLO pruebas, ver $ROOT/checks/tests.log"; exit 1; }

log "fotos y referencias de validacion"
printf "fotos %s\n" "${DATASETS[@]}" | xargs -P 3 -L 1 bash experiments/_run_cf.sh >> $ROOT/trabajos.log 2>&1
for ds in "${DATASETS[@]}"; do [ -f $ROOT/fotos/$ds/hfs/listo ] || { log "FALLO fotos $ds"; exit 1; }; done

if [ ! -f $ROOT/checks/estado.txt ]; then
  log "verificaciones previas"
  $PY -m experiments.check_cf --root $ROOT > $ROOT/checks/check_cf.log 2>&1
fi
grep -q " 0$" $ROOT/checks/estado.txt && { log "FALLO verificaciones, ver $ROOT/checks"; cat $ROOT/checks/estado.txt; exit 1; }
log "verificaciones ok"

ola1=$ROOT/trabajos_ola1.txt; : > $ola1
for s in $SEEDS5; do for ds in "${DATASETS[@]}"; do echo "train $ds $s cf_global" >> $ola1; done; done
for s in $SEEDS3; do for ds in "${DATASETS[@]}"; do echo "train $ds $s cf_local" >> $ola1; done; done
for ds in "${DATASETS[@]}"; do
  echo "eval $ds hfs" >> $ola1; echo "eval $ds hstar" >> $ola1
  for s in $SEEDS5; do echo "eval $ds phil $s" >> $ola1; echo "eval $ds ruido_hfs $s" >> $ola1; echo "eval $ds ruido_phil $s" >> $ola1; done
  for s in $SEEDS3; do echo "eval $ds res $s" >> $ola1; done
done
log "ola 1: $(wc -l < $ola1) trabajos, $PARALLEL a la vez"
xargs -P $PARALLEL -L 1 bash experiments/_run_cf.sh < $ola1 >> $ROOT/trabajos.log 2>&1
log "ola 1 lista, fallos: $(grep -c FALLO $ROOT/trabajos.log || true)"

ola2=$ROOT/trabajos_ola2.txt; : > $ola2
for s in $SEEDS3; do for ds in "${DATASETS[@]}"; do echo "train $ds $s cf_global_gnn" >> $ola2; echo "train $ds $s cf_global_dagger" >> $ola2; done; done
log "ola 2: $(wc -l < $ola2) trabajos"
xargs -P $PARALLEL -L 1 bash experiments/_run_cf.sh < $ola2 >> $ROOT/trabajos.log 2>&1
log "ola 2 lista, fallos: $(grep -c FALLO $ROOT/trabajos.log || true)"

log "resumen y figuras"
$PY -m experiments.resumen_cf --root $ROOT > $ROOT/resumen.log 2>&1 || log "FALLO resumen"
$PY -m experiments.figuras_cf --root $ROOT > $ROOT/figuras.log 2>&1 || log "FALLO figuras"
log "fin, $(( ($(date +%s) - T0) / 60 )) minutos"
