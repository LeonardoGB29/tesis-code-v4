#!/bin/bash
# maestro de exp_mem: PHIL, GNN y memoria temporal sobre cf_filtro congelado; relanzable, no repite lo listo
#   nohup caffeinate -dimsu bash experiments/run_mem.sh > results/exp_mem/maestro.log 2>&1 &
# fase 1: datos por semilla y conjunto, phil_res_cf primero, despues mem_no, mem_gru y mem_mamba; solo validacion
# fase 2 (FASE=2): evaluacion en prueba de lo que paso, semillas extra con SEEDS
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
ROOT=${ROOT:-results/exp_mem}
REFS=${REFS:-results/exp_cf}
REFS2=${REFS2:-results/exp_cf2}
export ROOT REFS2 DEVICE=${DEVICE:-mps} DEVTRAIN=${DEVTRAIN:-cpu} EPOCHS=${EPOCHS:-20} NITER=${NITER:-108} VALEVERY=${VALEVERY:-4} EXTRA=${EXTRA:-}
PARALLEL=${PARALLEL:-8}
SEEDS=${SEEDS:-"0 1"}
FASE=${FASE:-1}
mkdir -p $ROOT/checks
log() { echo "[maestro $(date '+%Y-%m-%d %H:%M:%S')] $*"; }
T0=$(date +%s)
log "inicio fase $FASE, semillas $SEEDS, $PARALLEL a la vez"

log "pruebas"
$PY -m tests.test_cf > $ROOT/checks/tests_cf.log 2>&1 && log "pruebas cf ok" || { log "FALLO pruebas cf, ver $ROOT/checks/tests_cf.log"; exit 1; }
$PY -m tests.test_mem > $ROOT/checks/tests_mem.log 2>&1 && log "pruebas mem ok" || { log "FALLO pruebas mem, ver $ROOT/checks/tests_mem.log"; exit 1; }
for ds in gaps_and_forest multiple_bugtraps mazes; do
  [ -f $REFS/refs/val_hfs_$ds.csv ] && [ -f $REFS/refs/eval/$ds/voraz_hfs.csv ] || { log "FALLO faltan referencias de $ds en $REFS"; exit 1; }
  for s in $SEEDS; do [ -f $REFS2/cf_filtro/runs/$ds/cf_filtro_s$s/best.pt ] || { log "FALLO falta cf_filtro s$s de $ds en $REFS2"; exit 1; }; done
done

if [ $FASE = 1 ]; then
  faltan=""
  for ds in gaps_and_forest multiple_bugtraps mazes; do for s in $SEEDS; do [ -f $ROOT/checks/estado_${ds}_s$s.txt ] || faltan="$faltan $ds:$s"; done; done
  if [ -n "$faltan" ]; then
    log "verificaciones previas de$faltan, en paralelo"
    for par in $faltan; do ds=${par%:*}; s=${par#*:}; $PY -m experiments.check_mem --root $ROOT --refs2 $REFS2 --dataset $ds --seed $s --device $DEVICE > $ROOT/checks/check_mem_${ds}_s$s.log 2>&1 & done
    wait
  fi
  cat $ROOT/checks/estado_*_s*.txt > $ROOT/checks/estado.txt
  grep -q " 0$" $ROOT/checks/estado.txt && { log "FALLO verificaciones, ver $ROOT/checks"; cat $ROOT/checks/estado.txt; exit 1; }
  log "verificaciones ok"

  # datos primero (6 procesos), phil_res_cf decide el modelo final y va antes que los brazos mem; gaps y bugtraps antes que mazes
  jobs=$ROOT/trabajos_fase1.txt; : > $jobs
  for ds in gaps_and_forest multiple_bugtraps mazes; do for s in $SEEDS; do echo "datos $ds $s" >> $jobs; done; done
  for ds in gaps_and_forest multiple_bugtraps mazes; do for s in $SEEDS; do echo "train $ds $s phil_res_cf" >> $jobs; done; done
  for ds in gaps_and_forest multiple_bugtraps; do for m in mem_no mem_gru mem_mamba; do for s in $SEEDS; do echo "train $ds $s $m" >> $jobs; done; done; done
  for m in mem_no mem_gru mem_mamba; do for s in $SEEDS; do echo "train mazes $s $m" >> $jobs; done; done
  log "fase 1: $(wc -l < $jobs) trabajos"
  xargs -P $PARALLEL -L 1 bash experiments/_run_mem.sh < $jobs >> $ROOT/trabajos.log 2>&1
  log "fase 1 lista, fallos: $(grep -c FALLO $ROOT/trabajos.log || true)"
  log "resumen de validacion"
  $PY -m experiments.resumen_mem --root $ROOT --refs $REFS --refs2 $REFS2 > $ROOT/resumen_mem.log 2>&1 || log "FALLO resumen_mem"
  $PY -m experiments.figuras_mem --root $ROOT --refs2 $REFS2 > $ROOT/figuras_mem.log 2>&1 || log "FALLO figuras_mem"
else
  # fase 2: entrena las semillas que falten de los brazos en BRAZOS y evalua en prueba
  BRAZOS=${BRAZOS:?"FASE=2 necesita BRAZOS, por ejemplo 'mem_no mem_gru'"}
  jobs=$ROOT/trabajos_fase2.txt; : > $jobs
  for ds in gaps_and_forest multiple_bugtraps mazes; do
    for s in $SEEDS; do
      [ $ds = mazes ] && [ $s -gt 2 ] && continue                 # cf_filtro tiene 3 semillas en mazes
      [ -f $REFS2/cf_filtro/runs/$ds/cf_filtro_s$s/best.pt ] || continue
      echo "datos $ds $s" >> $jobs
    done
  done
  for m in $BRAZOS; do for ds in gaps_and_forest multiple_bugtraps mazes; do for s in $SEEDS; do
    [ $ds = mazes ] && [ $s -gt 2 ] && continue
    [ -f $REFS2/cf_filtro/runs/$ds/cf_filtro_s$s/best.pt ] || continue
    echo "train $ds $s $m" >> $jobs
  done; done; done
  log "fase 2: $(wc -l < $jobs) trabajos de entrenamiento"
  xargs -P $PARALLEL -L 1 bash experiments/_run_mem.sh < $jobs >> $ROOT/trabajos.log 2>&1
  : > $jobs
  for m in $BRAZOS; do for ds in gaps_and_forest multiple_bugtraps mazes; do for s in $SEEDS; do
    [ -f $ROOT/$m/runs/$ds/${m}_s$s/best.pt ] && echo "eval $ds $s $m" >> $jobs
  done; done; done
  log "fase 2: $(wc -l < $jobs) evaluaciones"
  xargs -P $PARALLEL -L 1 bash experiments/_run_mem.sh < $jobs >> $ROOT/trabajos.log 2>&1
  log "fase 2 lista, fallos: $(grep -c FALLO $ROOT/trabajos.log || true)"
  $PY -m experiments.resumen_cf --root $ROOT --refs $REFS --refs2 $REFS2 > $ROOT/resumen.log 2>&1 || log "FALLO resumen"
  $PY -m experiments.resumen_mem --root $ROOT --refs $REFS --refs2 $REFS2 > $ROOT/resumen_mem.log 2>&1 || log "FALLO resumen_mem"
  $PY -m experiments.figuras_cf --root $ROOT --refs $REFS --refs2 $REFS2 --modelo cf_filtro > $ROOT/figuras.log 2>&1 || log "FALLO figuras"
fi
log "fin, $(( ($(date +%s) - T0) / 60 )) minutos"
