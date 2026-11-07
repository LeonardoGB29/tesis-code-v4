#!/bin/bash
# un trabajo de exp_escalera, relanzable, salta lo que ya tiene listo o csv, con un solo reintento
#   bash experiments/_run_escalera.sh train <ds> <seed> <peldano>     entrena y despues prueba en los 100 mapas con la u-net en cpu
#   bash experiments/_run_escalera.sh eval <ds> <seed> <peldano>      solo la prueba
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
ROOT=${ROOT:-results/exp_escalera}
DEVICE=${DEVICE:-cuda}
NITER=${NITER:-108}
VALEVERY=${VALEVERY:-4}
VALDENSO=${VALDENSO:-24}
VALTARDE=${VALTARDE:-12}
EXTRA_TRAIN=${EXTRA_TRAIN:-}
EXTRA_EVAL=${EXTRA_EVAL:-}
export PYTHONUTF8=1
paso=$1; ds=$2; s=$3; r=${4:-}
N_CHECK=${N_CHECK:-70}
log() { echo "[$(date '+%H:%M:%S')] $*"; }

intentar() {
  # corre la orden, si falla reintenta una vez, si vuelve a fallar marca FALLO
  "$@" && return 0
  log "reintento $*"
  "$@" && return 0
  log "FALLO $*"
  return 1
}

out=$ROOT/runs/$ds/${r}_s$s
ev=$ROOT/eval/$ds/${r}_s$s.csv
case $paso in
  check)
    [ -f $ROOT/checks/estado_${ds}_s$s.txt ] || intentar $PY -m experiments.check_escalera --root $ROOT --dataset $ds --seed $s --n $N_CHECK --device $DEVICE > $ROOT/checks/check_escalera_${ds}_s$s.log 2>&1 || exit 1 ;;
  train|eval)
    if [ $paso = train ] && [ ! -f $out/listo ]; then
      mkdir -p $out
      if [ -f $out/corriendo ] && kill -0 "$(cat $out/corriendo)" 2>/dev/null; then log "$ds $r s$s ya corre en el proceso $(cat $out/corriendo), se salta"; exit 0; fi
      echo $$ > $out/corriendo
      intentar $PY -m experiments.train_escalera --dataset $ds --seed $s --peldano $r --root $ROOT --device $DEVICE --n-iter $NITER --val-every $VALEVERY --val-denso $VALDENSO --val-every-tarde $VALTARDE $EXTRA_TRAIN \
        || { rm -f $out/corriendo; exit 1; }
      rm -f $out/corriendo
    fi
    if [ ! -f $ev ]; then
      [ -f $out/best.pt ] || { log "FALLO eval $ds $r s$s, falta $out/best.pt"; exit 1; }
      mkdir -p $ROOT/eval/$ds
      intentar $PY -m experiments.eval_cf --dataset $ds --phil $out/best.pt --device cpu --name ${r}_s$s --out $ev $EXTRA_EVAL || exit 1
    fi ;;
  *) log "FALLO paso desconocido $paso"; exit 1 ;;
esac
log "fin $*"
