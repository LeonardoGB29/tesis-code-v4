#!/bin/bash
# humo de exp_geo, todo el circuito con pocas fotos y pocos mapas, entrenamiento en cpu y en el acelerador DEVICE, mps por defecto, lo demas sobre la corrida del acelerador
#   bash experiments/humo_geo.sh            -> results/exp_geo/humo
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
H=${H:-results/exp_geo/humo}
ds=${DS:-mazes}
DEVICE=${DEVICE:-mps}
export PYTHONUTF8=1
mkdir -p $H
log() { echo "[humo $(date '+%H:%M:%S')] $*"; }
T0=$(date +%s)
log "fotos con 4 mapas"
$PY -m experiments.fotos_geo --dataset $ds --que hfs_val --out-root $H/fotos --n-grids 4 > $H/fotos.log 2>&1 || { log "FALLO fotos hfs_val"; exit 1; }
$PY -m experiments.fotos_geo --dataset $ds --que cf --seed 0 --split train --out-root $H/fotos --n-grids 4 >> $H/fotos.log 2>&1 || { log "FALLO fotos cf train"; exit 1; }
$PY -m experiments.fotos_geo --dataset $ds --que cf --seed 0 --split validation --out-root $H/fotos --n-grids 4 >> $H/fotos.log 2>&1 || { log "FALLO fotos cf val"; exit 1; }
$PY -m experiments.fotos_geo --dataset $ds --que cf --seed 2 --split train --out-root $H/fotos --n-grids 4 >> $H/fotos.log 2>&1 || { log "FALLO fotos cf s2"; exit 1; }
for dev in cpu $DEVICE; do
  out=runs; [ $dev = cpu ] && out=runs_cpu
  log "entrenamiento en $dev, 2 epocas, 48 fotos"
  $PY -m experiments.train_geo --dataset $ds --seed 0 --epochs 2 --device $dev --fotos-root $H/fotos --out-root $H/$out --max-fotos 48 --max-val 16 > $H/train_$dev.log 2>&1 || { log "FALLO train $dev, ver $H/train_$dev.log"; exit 1; }
  log "reanudacion en $dev desde last.pt con una epoca mas"
  rm -f $H/$out/$ds/geo_s0/listo
  $PY -m experiments.train_geo --dataset $ds --seed 0 --epochs 3 --device $dev --fotos-root $H/fotos --out-root $H/$out --max-fotos 48 --max-val 16 >> $H/train_$dev.log 2>&1 || { log "FALLO reanudacion $dev"; exit 1; }
  grep -q "sigue desde la epoca 2" $H/train_$dev.log || { log "FALLO la reanudacion no siguio desde la epoca 2"; exit 1; }
done
log "barrido con 2 lambdas y 3 mapas, eleccion, validacion, prueba y metricas"
export ROOT=$H EXTRA_EVAL="--n-grids 3" EXTRA_MET="--max-val 16" ESPERA_MAX=60
R="bash experiments/_run_geo.sh"
for l in 0 4; do $R barrido $ds geo $l; $R barrido $ds real $l; done > $H/trabajos.log 2>&1 || { log "FALLO barrido"; exit 1; }
$R elegir $ds >> $H/trabajos.log 2>&1 || { log "FALLO elegir"; exit 1; }
$R eval $ds 0 >> $H/trabajos.log 2>&1 && $R eval $ds real >> $H/trabajos.log 2>&1 || { log "FALLO eval"; exit 1; }
$R metricas $ds 0 >> $H/trabajos.log 2>&1 || { log "FALLO metricas"; exit 1; }
log "resumen y figuras"
$PY -m experiments.resumen_geo --root $H > $H/resumen.log 2>&1 || { log "FALLO resumen, ver $H/resumen.log"; exit 1; }
$PY -m experiments.figuras_geo --root $H --n 2 > $H/figuras.log 2>&1 || { log "FALLO figuras, ver $H/figuras.log"; exit 1; }
grep -q FALLO $H/figuras.log && { log "FALLO alguna figura, ver $H/figuras.log"; exit 1; }
grep -q FALLO $H/trabajos.log && { log "FALLO en trabajos, ver $H/trabajos.log"; exit 1; }
log "humo ok en $(( ($(date +%s) - T0) )) segundos"
echo "humo ok $(date '+%Y-%m-%d %H:%M:%S')" > $H/listo
