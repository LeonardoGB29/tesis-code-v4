#!/bin/bash
# un trabajo de new_exp; relanzable, salta lo que ya tiene listo o csv
#   bash new_exp/_run.sh train <ds> <seed> <modelo>      modelo: cf_local_filtro | cf_transpath; entrena y evalua en prueba
#   bash new_exp/_run.sh phil <ds> <seed>               phil_res_cf sobre el cf_filtro de exp_cf2, con experiments/_run_mem.sh
#   bash new_exp/_run.sh eval_phil <ds> <seed>          evaluacion de prueba de phil_res_cf
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
ROOT=${ROOT:-new_exp/results}
DEVICE=${DEVICE:-cuda}
export PYTHONUTF8=1
EPOCHS=${EPOCHS:-30}
EXTRA=${EXTRA:-}
EVALX=${EVALX:-}
paso=$1; ds=$2; s=$3
log() { echo "[$(date '+%H:%M:%S')] $*"; }
case $paso in
  train)
    modelo=$4
    case $modelo in
      cf_local_filtro) ancho="--ancho 32" ;;              # mismo computo por imagen que la u-net de cf_filtro
      cf_transpath) ancho="" ;;                           # transpath fija sus anchos en new_exp/redes.py
      *) log "FALLO modelo desconocido $modelo"; exit 1 ;;
    esac
    out=$ROOT/$modelo/runs/$ds/${modelo}_s$s
    if [ ! -f $out/listo ]; then
      # candado: si otro proceso vivo ya entrena esta corrida, se salta
      mkdir -p $out
      if [ -f $out/corriendo ] && kill -0 "$(cat $out/corriendo)" 2>/dev/null; then log "$ds $modelo s$s ya corre en el proceso $(cat $out/corriendo), se salta"; exit 0; fi
      echo $$ > $out/corriendo
      $PY -m new_exp.train --dataset $ds --seed $s --modelo $modelo --epochs $EPOCHS --device $DEVICE --out-root $ROOT/$modelo/runs $ancho $EXTRA \
        || { rm -f $out/corriendo; log "FALLO train $ds $modelo s$s"; exit 1; }
      rm -f $out/corriendo
    fi
    ev=$ROOT/$modelo/eval/$ds/${modelo}_s$s.csv
    [ -f $ev ] || $PY -m new_exp.eval --dataset $ds --ckpt $out/best.pt --out $ev $EVALX || { log "FALLO eval $ds $modelo s$s"; exit 1; } ;;
  phil)
    ROOT=$ROOT bash experiments/_run_mem.sh train $ds $s phil_res_cf || { log "FALLO phil $ds s$s"; exit 1; } ;;
  eval_phil)
    ROOT=$ROOT bash experiments/_run_mem.sh eval $ds $s phil_res_cf || { log "FALLO eval_phil $ds s$s"; exit 1; } ;;
  *) log "FALLO paso desconocido $paso"; exit 1 ;;
esac
log "fin $*"
