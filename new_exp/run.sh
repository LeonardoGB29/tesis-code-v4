#!/bin/bash
# maestro de new_exp en code_v4, los 2 experimentos esenciales, relanzable, no repite lo listo
#   ola 1, corrector local 11x11 con el suavizado de cf_filtro (exp 1). ola 2, arquitectura de transpath (exp 2)
#   en windows con git bash, desde code_v4
#     DEVICE=cuda nohup bash new_exp/run.sh >> new_exp/results/maestro.log 2>&1 &
#     nohup .venv/bin/python experiments/despierto.py new_exp/results > new_exp/results/despierto.log 2>&1 &
#   fotos y referencias de ../code_v3/results/exp_cf y cf_filtro de ../code_v3/results/exp_cf2, solo lectura
#   el exp 3 del plan original (phil sobre el corrector) quedo cubierto por exp_escalera y no se corre
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
ROOT=${ROOT:-new_exp/results}
P1=${P1:-4}                    # procesos de la ola 1
P2=${P2:-2}                    # procesos de la ola 2, transpath usa unos 3 GB de memoria de la gpu por proceso y la 4070 tiene 8
SEEDS=${SEEDS:-"0 1 2"}
DATASETS=${DATASETS:-"multiple_bugtraps gaps_and_forest"}
MODELOS=${MODELOS:-"cf_local_filtro cf_transpath"}
export ROOT DEVICE=${DEVICE:-cuda} EPOCHS=${EPOCHS:-30} EXTRA=${EXTRA:-} EVALX=${EVALX:-}
export PYTHONUTF8=1
mkdir -p $ROOT/checks
T0=$(date +%s)
log() { echo "[maestro $(date '+%Y-%m-%d %H:%M:%S')] $*"; }
fallos() { grep -c FALLO $ROOT/trabajos.log 2>/dev/null || true; }

estado() {
  # estado.txt se reescribe cada minuto, despierto.py lo mira para saber que el maestro sigue vivo
  local tmp=$ROOT/estado.txt.tmp.$$.$RANDOM
  {
    echo "Estado de new_exp al $(date '+%Y-%m-%d %H:%M:%S'), maestro pid $$, $(( ($(date +%s) - T0) / 60 )) minutos desde el lanzamiento"
    for m in $MODELOS; do for d in $DATASETS; do for s in $SEEDS; do
      local r=$ROOT/$m/runs/$d/${m}_s$s e=$ROOT/$m/eval/$d/${m}_s$s.csv
      if [ -f $e ]; then echo "  listo      $m $d s$s"
      elif [ -f $r/listo ]; then echo "  evaluando  $m $d s$s"
      elif [ -f $r/corriendo ]; then echo "  entrenando $m $d s$s, $(ls $r 2>/dev/null | grep -c '^epoca\|\.pt$') archivos"
      else echo "  pendiente  $m $d s$s"; fi
    done; done; done
    echo "fallos $(fallos)"
  } > $tmp && mv -f $tmp $ROOT/estado.txt
}
vigia() { while true; do sleep 60; estado; done; }
estado
vigia &
VIGIA=$!
trap 'kill $VIGIA 2>/dev/null' EXIT
log "inicio, DEVICE=$DEVICE semillas $SEEDS conjuntos $DATASETS modelos $MODELOS, ola 1 con $P1 procesos y ola 2 con $P2"

if [ ! -f $ROOT/checks/tests_ok ]; then
  log "pruebas"
  $PY -m tests.test_cf > $ROOT/checks/tests_cf.log 2>&1 && log "pruebas cf ok" || { log "FALLO pruebas cf, ver $ROOT/checks/tests_cf.log"; exit 1; }
  $PY -m new_exp.test_new > $ROOT/checks/tests_new.log 2>&1 && log "pruebas new_exp ok" || { log "FALLO pruebas new_exp, ver $ROOT/checks/tests_new.log"; exit 1; }
  touch $ROOT/checks/tests_ok
fi
BASE="$(dirname "$PWD")/code_v3/results"
for ds in $DATASETS; do
  [ -f "$BASE/exp_cf/fotos/$ds/hfs/listo" ] && [ -f "$BASE/exp_cf/refs/val_hfs_$ds.csv" ] && [ -f "$BASE/exp_cf/refs/eval/$ds/voraz_hfs.csv" ] \
    || { log "FALLO faltan fotos o referencias de $ds en $BASE/exp_cf"; exit 1; }
  for s in 0 1 2; do [ -f "$BASE/exp_cf2/cf_filtro/runs/$ds/cf_filtro_s$s/best.pt" ] || { log "FALLO falta cf_filtro s$s de $ds"; exit 1; }; done
done
log "fotos, referencias y cf_filtro ok"

ola() {
  # una ola por modelo, lo mas largo primero (bugtraps)
  local m=$1 p=$2 jobs=$ROOT/trabajos_$1.txt
  : > $jobs
  for ds in $DATASETS; do for s in $SEEDS; do echo "train $ds $s $m" >> $jobs; done; done
  log "ola $m, $(wc -l < $jobs) trabajos, $p a la vez"
  xargs -P $p -L 1 bash new_exp/_run.sh < $jobs >> $ROOT/trabajos.log 2>&1
  log "ola $m lista, fallos acumulados $(fallos)"
  $PY -m new_exp.resumen --root $ROOT > $ROOT/resumen.log 2>&1 && log "numeros en $ROOT/resultados.csv y lectura en $ROOT/RESUMEN_auto.md" || log "FALLO resumen, ver $ROOT/resumen.log"
}
for m in $MODELOS; do
  case $m in cf_transpath) ola $m $P2 ;; *) ola $m $P1 ;; esac
done
estado
log "fin, $(( ($(date +%s) - T0) / 60 )) minutos"
