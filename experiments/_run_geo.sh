#!/bin/bash
# un trabajo de exp_geo, relanzable, salta lo que ya tiene listo o csv, con un solo reintento
#   bash experiments/_run_geo.sh fotos_hfs_val <ds>
#   bash experiments/_run_geo.sh fotos_cf <ds> <seed> <train|validation>
#   bash experiments/_run_geo.sh train <ds> <seed>
#   bash experiments/_run_geo.sh barrido <ds> <geo|real> <lam>
#   bash experiments/_run_geo.sh elegir <ds>
#   bash experiments/_run_geo.sh val <ds> <seed>             espera lambda.json
#   bash experiments/_run_geo.sh eval <ds> <seed|real>       espera lambda.json
#   bash experiments/_run_geo.sh metricas <ds> <seed>
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
ROOT=${ROOT:-results/exp_geo}
DEVICE=${DEVICE:-mps}
EPOCHS=${EPOCHS:-10}
EXTRA_TRAIN=${EXTRA_TRAIN:-}
EXTRA_EVAL=${EXTRA_EVAL:-}
EXTRA_MET=${EXTRA_MET:-}
ESPERA_MAX=${ESPERA_MAX:-14400}
paso=$1; ds=$2
log() { echo "[$(date '+%H:%M:%S')] $*"; }

intentar() {
  # corre la orden, si falla reintenta una vez, si vuelve a fallar marca FALLO
  "$@" && return 0
  log "reintento $*"
  "$@" && return 0
  log "FALLO $*"
  return 1
}

esperar() {
  # espera a que exista un archivo, como maximo ESPERA_MAX segundos
  local f=$1 t=0
  while [ ! -f "$f" ]; do
    [ $t -ge $ESPERA_MAX ] && { log "FALLO esperando $f"; return 1; }
    sleep 30; t=$((t + 30))
  done
}

lam_de() { $PY -c "import json,sys; print(json.load(open(sys.argv[1]))['lambda'])" "$1"; }

case $paso in
  fotos_hfs_val)
    out=$ROOT/fotos/$ds/hfs_val
    [ -f $out/listo ] || intentar $PY -m experiments.fotos_geo --dataset $ds --que hfs_val --out-root $ROOT/fotos || exit 1 ;;
  fotos_cf)
    s=$3; split=$4; suf=train; [ "$split" = validation ] && suf=val
    out=$ROOT/fotos/$ds/cf_s${s}_$suf
    [ -f $out/listo ] || intentar $PY -m experiments.fotos_geo --dataset $ds --que cf --seed $s --split $split --out-root $ROOT/fotos || exit 1 ;;
  train)
    s=$3; out=$ROOT/runs/$ds/geo_s$s
    if [ ! -f $out/listo ]; then
      mkdir -p $out
      if [ -f $out/corriendo ] && kill -0 "$(cat $out/corriendo)" 2>/dev/null; then log "$ds geo_s$s ya corre en el proceso $(cat $out/corriendo), se salta"; exit 0; fi
      echo $$ > $out/corriendo
      intentar $PY -m experiments.train_geo --dataset $ds --seed $s --epochs $EPOCHS --device $DEVICE --fotos-root $ROOT/fotos --out-root $ROOT/runs $EXTRA_TRAIN || { rm -f $out/corriendo; exit 1; }
      rm -f $out/corriendo
    fi ;;
  barrido)
    quien=$3; lam=$4; mkdir -p $ROOT/barrido/$ds
    if [ "$quien" = real ]; then
      ev=$ROOT/barrido/$ds/real_lam$lam.csv
      [ -f $ev ] || intentar $PY -m experiments.eval_geo --dataset $ds --real --lam $lam --split validation --out $ev --name geo_real_lam$lam $EXTRA_EVAL || exit 1
    else
      ev=$ROOT/barrido/$ds/geo_s0_lam$lam.csv
      [ -f $ev ] || intentar $PY -m experiments.eval_geo --dataset $ds --ckpt $ROOT/runs/$ds/geo_s0/best.pt --lam $lam --split validation --out $ev --name geo_s0_lam$lam $EXTRA_EVAL || exit 1
    fi ;;
  elegir)
    [ -f $ROOT/barrido/$ds/lambda.json ] || intentar $PY -m experiments.lambda_geo --dataset $ds --root $ROOT || exit 1 ;;
  val)
    s=$3; ev=$ROOT/val/$ds/geo_s$s.csv
    if [ ! -f $ev ]; then
      esperar $ROOT/barrido/$ds/lambda.json || exit 1
      lam=$(lam_de $ROOT/barrido/$ds/lambda.json)
      intentar $PY -m experiments.eval_geo --dataset $ds --ckpt $ROOT/runs/$ds/geo_s$s/best.pt --lam $lam --split validation --out $ev --name geo_s$s $EXTRA_EVAL || exit 1
    fi ;;
  eval)
    quien=$3; mkdir -p $ROOT/eval/$ds
    esperar $ROOT/barrido/$ds/lambda.json || exit 1
    lam=$(lam_de $ROOT/barrido/$ds/lambda.json)
    if [ "$quien" = real ]; then
      ev=$ROOT/eval/$ds/geo_real.csv
      [ -f $ev ] || intentar $PY -m experiments.eval_geo --dataset $ds --real --lam $lam --out $ev --name geo_real $EXTRA_EVAL || exit 1
    else
      ev=$ROOT/eval/$ds/geo_s$quien.csv
      [ -f $ev ] || intentar $PY -m experiments.eval_geo --dataset $ds --ckpt $ROOT/runs/$ds/geo_s$quien/best.pt --lam $lam --out $ev --name geo_s$quien $EXTRA_EVAL || exit 1
    fi ;;
  metricas)
    s=$3; ev=$ROOT/metricas/$ds/geo_s$s.csv
    [ -f $ev ] || intentar $PY -m experiments.metricas_geo --dataset $ds --seed $s --root $ROOT $EXTRA_MET || exit 1 ;;
  *) log "FALLO paso desconocido $paso"; exit 1 ;;
esac
log "fin $*"
