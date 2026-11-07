#!/bin/bash
# humo de exp_escalera. cada peldano con 2 iteraciones, 2 mapas de entrenamiento y 2 de validacion en mazes, reanudacion desde last.pt,
# prueba en 2 mapas con la u-net en cpu, y el resumen y las figuras si ya existen sus scripts
#   DEVICE=cuda bash experiments/humo_escalera.sh            -> results/exp_escalera/humo
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
H=${H:-results/exp_escalera/humo}
ds=${DS:-mazes}
DEVICE=${DEVICE:-cuda}
export PYTHONUTF8=1
mkdir -p $H
log() { echo "[humo $(date '+%H:%M:%S')] $*"; }
T0=$(date +%s)
for r in R2 R4 R1 R0 R3 C; do
  out=$H/runs/$ds/${r}_s0
  log "$r, 2 iteraciones, 2 mapas"
  t1=$(date +%s)
  $PY -m experiments.train_escalera --dataset $ds --seed 0 --peldano $r --root $H --device $DEVICE --n-iter 2 --val-every 1 --n-val 2 --n-train 2 --n-stats 8 > $H/train_$r.log 2>&1 \
    || { log "FALLO train $r, ver $H/train_$r.log"; exit 1; }
  [ -f $out/listo ] || { log "FALLO $r sin listo"; exit 1; }
  log "$r reanudacion desde last.pt con una iteracion mas"
  rm -f $out/listo
  $PY -m experiments.train_escalera --dataset $ds --seed 0 --peldano $r --root $H --device $DEVICE --n-iter 3 --val-every 1 --n-val 2 --n-train 2 --n-stats 8 >> $H/train_$r.log 2>&1 \
    || { log "FALLO reanudacion $r"; exit 1; }
  grep -q "sigue desde la iteracion 2" $H/train_$r.log || { log "FALLO la reanudacion de $r no siguio desde la iteracion 2"; exit 1; }
  mkdir -p $H/eval/$ds
  $PY -m experiments.eval_cf --dataset $ds --phil $out/best.pt --device cpu --name ${r}_s0 --out $H/eval/$ds/${r}_s0.csv --n-grids 2 > $H/eval_$r.log 2>&1 \
    || { log "FALLO eval $r, ver $H/eval_$r.log"; exit 1; }
  log "$r ok en $(( $(date +%s) - t1 )) segundos"
done
if [ -f experiments/resumen_escalera.py ]; then
  log "resumen y figuras"
  $PY -m experiments.resumen_escalera --root $H > $H/resumen.log 2>&1 || { log "FALLO resumen, ver $H/resumen.log"; exit 1; }
  [ -f experiments/figuras_escalera.py ] && { $PY -m experiments.figuras_escalera --root $H --n 2 > $H/figuras.log 2>&1 || { log "FALLO figuras, ver $H/figuras.log"; exit 1; }; }
fi
log "humo ok en $(( $(date +%s) - T0 )) segundos"
echo "humo ok $(date '+%Y-%m-%d %H:%M:%S')" > $H/listo
