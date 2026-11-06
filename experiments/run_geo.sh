#!/bin/bash
# maestro de exp_geo. pruebas, verificaciones, fotos, entrenamiento de la cnn de geometria, barrido de lambda, prueba, metricas, resumen y figuras
# relanzable, salta lo que ya esta listo, cada entrenamiento sigue desde su last.pt, los logs solo se anexan
#   nohup caffeinate -dimsu bash experiments/run_geo.sh >> results/exp_geo/maestro.log 2>&1 &
#   en linux o en windows con git bash no hay caffeinate, se quita, y con gpu de nvidia se lanza con DEVICE=cuda
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
ROOT=${ROOT:-results/exp_geo}
export ROOT DEVICE=${DEVICE:-mps} EPOCHS=${EPOCHS:-10} EXTRA_TRAIN=${EXTRA_TRAIN:-} EXTRA_EVAL=${EXTRA_EVAL:-} EXTRA_MET=${EXTRA_MET:-} EXTRA_FOTOS=${EXTRA_FOTOS:-}
export PYTHONUTF8=1   # misma codificacion de texto que en mac
SEEDS=${SEEDS:-"0 1 2"}
LAMBDAS=${LAMBDAS:-"0 0.5 1 2 4 8 16 32"}
DATASETS=${DATASETS:-"gaps_and_forest multiple_bugtraps mazes"}
PAR_BARRIDO=${PAR_BARRIDO:-2}
N_CHECK=${N_CHECK:-100}
mkdir -p $ROOT/checks $ROOT/logs
T0=$(date +%s)
log() { echo "[maestro $(date '+%Y-%m-%d %H:%M:%S')] $*"; }
# herramientas portables. mac, linux y git bash en windows
hay() { command -v "$1" >/dev/null 2>&1; }
CAFE=""; hay caffeinate && CAFE="caffeinate -dimsu "
RELANZAR="cd \"$PWD\" && DEVICE=$DEVICE nohup ${CAFE}bash experiments/run_geo.sh >> results/exp_geo/maestro.log 2>&1 &"
PS_OK=0; ps -eo pid,etime,args >/dev/null 2>&1 && PS_OK=1
procesos() {
  # pid, tiempo y orden de cada proceso. el ps de git bash no muestra los argumentos y se pregunta a wmi
  # el python del venv de windows es un lanzador que arranca otro python con la misma orden, se deja uno solo por trabajo
  if [ $PS_OK = 1 ]; then ps -eo pid,etime,args
  else powershell -NoProfile -Command "\$p = @(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\"); \$ids = @(\$p | ForEach-Object { \$_.ProcessId }); \$p | Where-Object { \$ids -notcontains \$_.ParentProcessId } | ForEach-Object { '{0} {1}min {2}' -f \$_.ProcessId, [int]((Get-Date) - \$_.CreationDate).TotalMinutes, \$_.CommandLine }" 2>/dev/null | tr -d '\r'
  fi
}
paginado() {
  # contador acumulado de paginas escritas al intercambio
  if hay vm_stat; then vm_stat | awk '/Pageouts/ {gsub(/\./,"",$2); print $2}'
  elif [ -r /proc/vmstat ]; then awk '/^pswpout/ {print $2}' /proc/vmstat
  else powershell -NoProfile -Command "(Get-CimInstance Win32_PerfRawData_PerfOS_Memory).PagesOutputPersec" 2>/dev/null | tr -d '\r'
  fi
}
intercambio() {
  # intercambio usado
  if sysctl -n vm.swapusage >/dev/null 2>&1; then sysctl -n vm.swapusage | awk '{print $6}'
  elif hay free; then free -m | awk '/Swap/ {print $3"M"}'
  else powershell -NoProfile -Command "'{0}M' -f (Get-CimInstance Win32_PageFileUsage | Measure-Object CurrentUsage -Sum).Sum" 2>/dev/null | tr -d '\r'
  fi
}
PAGINA=$(sysctl -n hw.pagesize 2>/dev/null || { [ -r /proc/vmstat ] && getconf PAGESIZE 2>/dev/null; } || echo 4096)   # en windows la pagina es de 4096
hora_de() { if stat --version >/dev/null 2>&1; then date -r "$1" +%H:%M; else stat -f %Sm -t %H:%M "$1"; fi; }   # stat de gnu o de bsd

estado() {
  # se escribe en un temporal con nombre propio y se mueve entero, sin glob, asi el vigia y el maestro no se pisan
  local tmp=$ROOT/estado.txt.tmp.$$.$RANDOM
  {
    echo "Estado de exp_geo al $(date '+%Y-%m-%d %H:%M:%S'), maestro pid $$, $(( ($(date +%s) - T0) / 60 )) minutos desde el lanzamiento"
    echo
    echo "Terminado"
    [ -f $ROOT/checks/tests_ok ] && echo "  pruebas"
    [ -f $ROOT/checks/estado.txt ] && echo "  verificaciones $(tr '\n' ' ' < $ROOT/checks/estado.txt)"
    for d in $DATASETS; do
      for f in $ROOT/fotos/$d/*/listo; do [ -f "$f" ] && echo "  fotos $d/$(basename $(dirname $f)) $(head -c 60 $f | tr '\n' ' ')"; done
      for f in $ROOT/runs/$d/*/listo; do [ -f "$f" ] && echo "  entrenamiento $d/$(basename $(dirname $f)) $(tr '\n' ' ' < $f)"; done
      [ -f $ROOT/barrido/$d/lambda.json ] && echo "  lambda $d $($PY -c "import json;d=json.load(open('$ROOT/barrido/$d/lambda.json'));print(d['lambda'],'val exp',round(d['exp'],1),'rel',round(d['rel_hfs'],3))" 2>/dev/null)"
      for f in $ROOT/val/$d/*.csv $ROOT/eval/$d/*.csv $ROOT/metricas/$d/geo_s*.csv; do [ -f "$f" ] && echo "  $(basename $(dirname $(dirname $f))) $d/$(basename $f)"; done
    done
    [ -f $ROOT/RESUMEN_auto.md ] && echo "  resumen RESUMEN_auto.md de las $(hora_de $ROOT/RESUMEN_auto.md)"
    echo
    echo "Corriendo"
    procesos | grep -E "experiments\.(train_geo|eval_geo|fotos_geo|metricas_geo|check_geo|lambda_geo|resumen_geo|figuras_geo)|tests\.test_" | grep -v grep | sed 's/^/  /' | cut -c1-200
    echo
    echo "Fallos"
    grep -h FALLO $ROOT/trabajos.log $ROOT/maestro.log 2>/dev/null | sed 's/^/  /' || true
    grep -qh FALLO $ROOT/trabajos.log $ROOT/maestro.log 2>/dev/null || echo "  ninguno"
    echo
    echo "Memoria"
    tail -1 $ROOT/logs/memoria.log 2>/dev/null | sed 's/^/  /'
    grep -h AVISO $ROOT/logs/memoria.log 2>/dev/null | tail -3 | sed 's/^/  /'
    echo
    echo "Para relanzar o completar, la misma orden, no repite nada listo"
    echo "  $RELANZAR"
  } > "$tmp"; mv -f "$tmp" $ROOT/estado.txt 2>/dev/null
}

vigia() {
  # cada minuto reescribe estado.txt y anota el intercambio. dos minutos seguidos con mas de 40 MB/s de paginado dejan un AVISO
  local prev=$(paginado) seguidos=0
  while true; do
    sleep 60
    local cur=$(paginado)
    local tasa=$(( (${cur:-0} - ${prev:-0}) * PAGINA / 1048576 / 60 ))
    prev=$cur
    local swap=$(intercambio)
    local nproc=$(procesos | grep -cE "experiments\.(train_geo|eval_geo|fotos_geo|metricas_geo)")
    echo "$(date '+%Y-%m-%d %H:%M:%S') intercambio usado $swap paginado ${tasa} MB/s procesos $nproc" >> $ROOT/logs/memoria.log
    if [ $tasa -gt 40 ]; then seguidos=$((seguidos + 1)); else seguidos=0; fi
    [ $seguidos -ge 2 ] && echo "$(date '+%Y-%m-%d %H:%M:%S') AVISO paginado ${tasa} MB/s dos minutos seguidos, bajar PAR_BARRIDO o esperar" >> $ROOT/logs/memoria.log
    estado
  done
}

log "inicio, EPOCHS=$EPOCHS DEVICE=$DEVICE SEEDS=$SEEDS LAMBDAS=$LAMBDAS DATASETS=$DATASETS PAR_BARRIDO=$PAR_BARRIDO, maquina $(uname -sm)"
estado
vigia &
VIGIA=$!
trap 'kill $VIGIA 2>/dev/null' EXIT

# 1 pruebas
if [ ! -f $ROOT/checks/tests_ok ]; then
  log "pruebas"
  $PY -m tests.test_geo > $ROOT/checks/tests_geo.log 2>&1 && $PY -m tests.test_cf > $ROOT/checks/tests_cf.log 2>&1 && touch $ROOT/checks/tests_ok && log "pruebas ok" \
    || { log "FALLO pruebas, ver $ROOT/checks/tests_geo.log y tests_cf.log"; estado; exit 1; }
fi
BASE="$(dirname "$PWD")/code_v3/results"
for ds in $DATASETS; do
  [ -f "$BASE/exp_cf/fotos/$ds/hfs/listo" ] && [ -f "$BASE/exp_cf/refs/val_hfs_$ds.csv" ] && [ -f "$BASE/exp_cf2/cf_filtro/runs/$ds/cf_filtro_s0/best.pt" ] || { log "FALLO faltan fotos base, referencias o cf_filtro de $ds en code_v3"; exit 1; }
done

# 2 verificaciones previas, los tres conjuntos a la vez
if [ ! -f $ROOT/checks/estado.txt ]; then
  log "verificaciones previas"
  cpids=""
  for ds in $DATASETS; do $PY -m experiments.check_geo --root $ROOT --dataset $ds --n $N_CHECK > $ROOT/checks/check_geo_$ds.log 2>&1 & cpids="$cpids $!"; done
  wait $cpids
  cat $ROOT/checks/estado_*.txt > $ROOT/checks/estado.txt 2>/dev/null
fi
if grep -q " 0$" $ROOT/checks/estado.txt; then log "FALLO verificaciones, ver $ROOT/checks"; cat $ROOT/checks/estado.txt; estado; exit 1; fi
log "verificaciones ok"
estado

# 3 una cadena por conjunto. fotos, entrenamiento por semilla, barrido con la primera semilla, prueba y metricas
cadena() {
  local ds=$1
  local R="bash experiments/_run_geo.sh"
  set -- $SEEDS; local s0=$1; shift; local resto="$*"
  log "cadena $ds arranca"
  $R fotos_hfs_val $ds & local p1=$!
  $R fotos_cf $ds $s0 train & local p2=$!
  $R fotos_cf $ds $s0 validation & local p3=$!
  wait $p1 $p2 $p3
  ( for s in $resto; do $R fotos_cf $ds $s train; $R fotos_cf $ds $s validation; done ) & local pfotos=$!
  local pids=""
  $R train $ds $s0 || log "la cadena $ds sigue sin la semilla $s0"
  if [ -f $ROOT/runs/$ds/geo_s$s0/listo ]; then
    ( for l in $LAMBDAS; do echo "$ds geo $l"; echo "$ds real $l"; done | xargs -P $PAR_BARRIDO -L 1 $R barrido
      $R elegir $ds && { $R eval $ds $s0; $R eval $ds real; }
      $R metricas $ds $s0 ) & pids="$pids $!"
  fi
  wait $pfotos
  local s
  for s in $resto; do
    $R train $ds $s || log "la cadena $ds sigue sin la semilla $s"
    if [ -f $ROOT/runs/$ds/geo_s$s/listo ]; then
      ( $R val $ds $s; $R eval $ds $s; $R metricas $ds $s ) & pids="$pids $!"
    fi
  done
  wait $pids
  log "cadena $ds termina"
}
cpids=""
for ds in $DATASETS; do cadena $ds >> $ROOT/trabajos.log 2>&1 & cpids="$cpids $!"; done
wait $cpids
estado

# 4 resumen y figuras
log "resumen y figuras"
$PY -m experiments.resumen_geo --root $ROOT > $ROOT/resumen.log 2>&1 || log "FALLO resumen, ver $ROOT/resumen.log"
$PY -m experiments.figuras_geo --root $ROOT > $ROOT/figuras.log 2>&1 || log "FALLO figuras, ver $ROOT/figuras.log"
log "fin, $(( ($(date +%s) - T0) / 60 )) minutos, fallos en trabajos.log $(grep -c FALLO $ROOT/trabajos.log 2>/dev/null || echo 0)"
estado
