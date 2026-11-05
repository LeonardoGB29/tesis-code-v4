#!/bin/bash
# maestro de exp_cf2: cf con la escala gruesa dentro del modelo (cf_filtro, cf_grueso) entrenado con dagger; relanzable, no repite lo listo
#   nohup caffeinate -dimsu bash experiments/run_cf2.sh > results/exp_cf2/maestro.log 2>&1 &
# las fotos, las referencias de validacion y las evaluaciones de h_fs, h*, PHIL, phil_res_s y ruido se leen de results/exp_cf
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
ROOT=${ROOT:-results/exp_cf2}
REFS=${REFS:-results/exp_cf}
export ROOT DEVICE=${DEVICE:-mps} EPOCHS=${EPOCHS:-30} EXTRA=${EXTRA:-}
PARALLEL=${PARALLEL:-8}
mkdir -p $ROOT/checks
log() { echo "[maestro $(date '+%Y-%m-%d %H:%M:%S')] $*"; }
T0=$(date +%s)
log "inicio"

log "pruebas"
$PY -m tests.test_cf > $ROOT/checks/tests.log 2>&1 && log "pruebas ok" || { log "FALLO pruebas, ver $ROOT/checks/tests.log"; exit 1; }
for ds in gaps_and_forest multiple_bugtraps mazes; do
  [ -f $REFS/fotos/$ds/hfs/listo ] && [ -f $REFS/refs/val_hfs_$ds.csv ] && [ -f $REFS/refs/eval/$ds/voraz_hfs.csv ] || { log "FALLO faltan fotos o referencias de $ds en $REFS"; exit 1; }
done

if [ ! -f $ROOT/checks/estado.txt ]; then
  log "verificaciones previas, los tres conjuntos a la vez"
  for ds in gaps_and_forest multiple_bugtraps mazes; do $PY -m experiments.check_cf2 --root $ROOT --refs $REFS --dataset $ds > $ROOT/checks/check_cf2_$ds.log 2>&1 & done
  wait
  cat $ROOT/checks/estado_*.txt > $ROOT/checks/estado.txt
fi
grep -q " 0$" $ROOT/checks/estado.txt && { log "FALLO verificaciones, ver $ROOT/checks"; cat $ROOT/checks/estado.txt; exit 1; }
log "verificaciones ok"

# los mas largos primero (bugtraps), despues gaps, mazes al final
jobs=$ROOT/trabajos_ola1.txt; : > $jobs
for s in 0 1 2 3 4; do echo "train multiple_bugtraps $s cf_filtro" >> $jobs; done
for s in 0 1 2; do echo "train multiple_bugtraps $s cf_grueso" >> $jobs; done
for s in 0 1 2 3 4; do echo "train gaps_and_forest $s cf_filtro" >> $jobs; done
for s in 0 1 2; do echo "train gaps_and_forest $s cf_grueso" >> $jobs; done
for s in 0 1 2; do echo "train mazes $s cf_filtro" >> $jobs; done
log "ola 1: $(wc -l < $jobs) trabajos, $PARALLEL a la vez"
xargs -P $PARALLEL -L 1 bash experiments/_run_cf.sh < $jobs >> $ROOT/trabajos.log 2>&1
log "ola 1 lista, fallos: $(grep -c FALLO $ROOT/trabajos.log || true)"

log "resumen y figuras"
$PY -m experiments.resumen_cf --root $ROOT --refs $REFS > $ROOT/resumen.log 2>&1 || log "FALLO resumen"
$PY -m experiments.figuras_cf --root $ROOT --refs $REFS --modelo cf_filtro > $ROOT/figuras.log 2>&1 || log "FALLO figuras"
log "fin, $(( ($(date +%s) - T0) / 60 )) minutos"
