#!/bin/bash
# un trabajo del experimento cf; relanzable, salta lo que ya tiene listo o csv
#   bash experiments/_run_cf.sh fotos <ds>
#   bash experiments/_run_cf.sh train <ds> <seed> <modelo>      (ROOT=results/exp_cf2 para exp_cf2, las fotos y referencias siempre salen de exp_cf)
#   bash experiments/_run_cf.sh eval <ds> <que> [arg]      que: phil <seed> | res <seed> | hfs | hstar | ruido_hfs <k> | ruido_phil <k> | cf <modelo> <seed>
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
ROOT=${ROOT:-results/exp_cf}
DEVICE=${DEVICE:-mps}
EPOCHS=${EPOCHS:-30}
RUIDO=${RUIDO:-0.01}
EXTRA=${EXTRA:-}
paso=$1; ds=$2
log() { echo "[$(date '+%H:%M:%S')] $*"; }
case $paso in
  fotos)
    if [ -f $ROOT/fotos/$ds/hfs/listo ] && [ -f $ROOT/refs/val_hfs_$ds.csv ]; then log "fotos $ds ya estan"; exit 0; fi
    $PY -m experiments.fotos --dataset $ds ;;
  train)
    s=$3; modelo=$4
    out=$ROOT/$modelo/runs/$ds/${modelo}_s$s
    init=""
    case $modelo in
      cf_global_gnn|cf_global_dagger)
        # parte de cf_global entrenado; si la seleccion se quedo en la epoca 0 (cf constante), parte de la epoca 10
        run=$ROOT/cf_global/runs/$ds/cf_global_s$s
        base=$run/${INIT_CKPT:-best.pt}
        if [ -z "${INIT_CKPT:-}" ] && [ "$(awk -F, 'NR>1 && $13!="" {b=$13} END{print b}' $run/train.csv)" = "0" ] && [ -f $run/epoca_10.pt ]; then
          base=$run/epoca_10.pt; log "$ds s$s: best.pt es la epoca 0, $modelo parte de epoca_10.pt"
        fi
        [ -f $base ] || { log "FALLO falta $base"; exit 1; }
        init="--init $base" ;;
    esac
    if [ ! -f $out/listo ]; then
      # candado: si otro proceso vivo ya entrena esta corrida (lanzada a mano), se salta
      mkdir -p $out
      if [ -f $out/corriendo ] && kill -0 "$(cat $out/corriendo)" 2>/dev/null; then log "$ds $modelo s$s ya corre en el proceso $(cat $out/corriendo), se salta"; exit 0; fi
      echo $$ > $out/corriendo
      $PY -m experiments.train_cf --dataset $ds --seed $s --modelo $modelo --epochs $EPOCHS --device $DEVICE --out-root $ROOT/$modelo/runs $init $EXTRA || { rm -f $out/corriendo; log "FALLO train $ds $modelo s$s"; exit 1; }
      rm -f $out/corriendo
    fi
    ev=$ROOT/$modelo/eval/$ds/${modelo}_s$s.csv
    [ -f $ev ] || $PY -m experiments.eval_cf --dataset $ds --ckpt $out/best.pt --out $ev || { log "FALLO eval $ds $modelo s$s"; exit 1; } ;;
  eval)
    que=$3
    case $que in
      phil)  s=$4; ev=$ROOT/phil/eval/$ds/phil_s_s$s.csv
             [ -f $ev ] || $PY -m experiments.eval_cf --dataset $ds --phil $ROOT/base/runs/$ds/phil_s_s$s/best.pt --out $ev ;;
      res)   s=$4; ev=$ROOT/phil_res_s/eval/$ds/phil_res_s_s$s.csv
             [ -f $ev ] || $PY -m experiments.eval_cf --dataset $ds --phil $ROOT/phil_res_s/runs/$ds/phil_res_s_n108_s$s/best.pt --name phil_res_s_s$s --out $ev ;;
      hfs)   ev=$ROOT/refs/eval/$ds/voraz_hfs.csv
             [ -f $ev ] || $PY -m experiments.eval_cf --dataset $ds --ref hfs --out $ev ;;
      hstar) ev=$ROOT/refs/eval/$ds/voraz_hstar.csv
             [ -f $ev ] || $PY -m experiments.eval_cf --dataset $ds --ref hstar --out $ev ;;
      ruido_hfs)  k=$4; ev=$ROOT/ruido/eval/$ds/voraz_hfs_ruido$k.csv
             [ -f $ev ] || $PY -m experiments.eval_cf --dataset $ds --ref hfs --ruido $RUIDO --ruido-seed $k --out $ev ;;
      ruido_phil) k=$4; ev=$ROOT/ruido/eval/$ds/phil_s_s0_ruido$k.csv
             [ -f $ev ] || $PY -m experiments.eval_cf --dataset $ds --phil $ROOT/base/runs/$ds/phil_s_s0/best.pt --ruido $RUIDO --ruido-seed $k --out $ev ;;
      *) log "FALLO eval desconocido $que"; exit 1 ;;
    esac || { log "FALLO eval $ds $que ${4:-}"; exit 1; } ;;
  *) log "FALLO paso desconocido $paso"; exit 1 ;;
esac
log "fin $*"
