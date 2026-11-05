#!/bin/bash
# un trabajo de exp_mem; relanzable, salta lo que ya tiene listo o csv
#   bash experiments/_run_mem.sh datos <ds> <seed>
#   bash experiments/_run_mem.sh train <ds> <seed> <modelo>      modelo: phil_res_cf | mem_no | mem_gru | mem_mamba
#   bash experiments/_run_mem.sh eval <ds> <seed> <modelo>       fase 2: evaluacion en prueba
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
ROOT=${ROOT:-results/exp_mem}
REFS2=${REFS2:-results/exp_cf2}
DEVICE=${DEVICE:-mps}
DEVTRAIN=${DEVTRAIN:-cpu}
EPOCHS=${EPOCHS:-20}
NITER=${NITER:-108}
VALEVERY=${VALEVERY:-4}
ESPERA=${ESPERA:-120}
EVALDEV=${EVALDEV:-cpu}
EXTRA=${EXTRA:-}
paso=$1; ds=$2; s=$3
log() { echo "[$(date '+%H:%M:%S')] $*"; }
case $paso in
  datos)
    out=$ROOT/datos/$ds/s$s
    if [ -f $out/pre/listo ]; then log "datos $ds s$s ya estan"; exit 0; fi
    mkdir -p $out
    $PY -m experiments.fotos_mem --dataset $ds --seed $s --root $ROOT --refs2 $REFS2 --device $DEVICE || { log "FALLO datos $ds s$s"; touch $out/fallo; exit 1; } ;;
  train)
    modelo=$4
    out=$ROOT/$modelo/runs/$ds/${modelo}_s$s
    if [ -f $out/listo ]; then log "train $ds $modelo s$s ya esta"; exit 0; fi
    mkdir -p $out
    if [ -f $out/corriendo ] && kill -0 "$(cat $out/corriendo)" 2>/dev/null; then log "$ds $modelo s$s ya corre en el proceso $(cat $out/corriendo), se salta"; exit 0; fi
    echo $$ > $out/corriendo
    if [ $modelo = phil_res_cf ]; then
      $PY -m experiments.train_phil_res --dataset $ds --seed $s --root $ROOT --refs2 $REFS2 --device $DEVICE --n-iter $NITER --val-every $VALEVERY $EXTRA \
        || { rm -f $out/corriendo; log "FALLO train $ds $modelo s$s"; exit 1; }
    else
      # espera los datos de la semilla hasta ESPERA minutos; si fallaron, falla
      datos=$ROOT/datos/$ds/s$s
      for ((i = 0; i < ESPERA; i++)); do
        [ -f $datos/pre/listo ] && break
        [ -f $datos/fallo ] && { rm -f $out/corriendo; log "FALLO train $ds $modelo s$s, los datos fallaron"; exit 1; }
        sleep 60
      done
      [ -f $datos/pre/listo ] || { rm -f $out/corriendo; log "FALLO train $ds $modelo s$s, sin datos tras $ESPERA minutos"; exit 1; }
      $PY -m experiments.train_mem --dataset $ds --seed $s --modelo $modelo --root $ROOT --refs2 $REFS2 --device $DEVTRAIN --dev-unet $DEVICE --epochs $EPOCHS $EXTRA \
        || { rm -f $out/corriendo; log "FALLO train $ds $modelo s$s"; exit 1; }
    fi
    rm -f $out/corriendo ;;
  eval)
    modelo=$4
    out=$ROOT/$modelo/runs/$ds/${modelo}_s$s
    ev=$ROOT/$modelo/eval/$ds/${modelo}_s$s.csv
    [ -f $out/best.pt ] || { log "FALLO eval $ds $modelo s$s, falta $out/best.pt"; exit 1; }
    if [ ! -f $ev ]; then
      if [ $modelo = phil_res_cf ]; then
        $PY -m experiments.eval_cf --dataset $ds --phil $out/best.pt --device $EVALDEV --out $ev || { log "FALLO eval $ds $modelo s$s"; exit 1; }
      else
        $PY -m experiments.eval_cf --dataset $ds --mem $out/best.pt --device $EVALDEV --out $ev || { log "FALLO eval $ds $modelo s$s"; exit 1; }
      fi
    fi ;;
  *) log "FALLO paso desconocido $paso"; exit 1 ;;
esac
log "fin $*"
