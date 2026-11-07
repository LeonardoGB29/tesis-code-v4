#!/bin/bash
# maestro de exp_escalera. pruebas, verificaciones, cola de entrenamiento con prueba de cada peldano, resumen y figuras
# relanzable, salta lo que ya esta listo, cada entrenamiento sigue desde su last.pt, los logs solo se anexan
#   DEVICE=cuda nohup bash experiments/run_escalera.sh >> results/exp_escalera/maestro.log 2>&1 &
#   en mac, nohup caffeinate -dimsu bash experiments/run_escalera.sh >> results/exp_escalera/maestro.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
ROOT=${ROOT:-results/exp_escalera}
export ROOT DEVICE=${DEVICE:-cuda} NITER=${NITER:-108} VALEVERY=${VALEVERY:-4} VALDENSO=${VALDENSO:-24} VALTARDE=${VALTARDE:-12} EXTRA_TRAIN=${EXTRA_TRAIN:-} EXTRA_EVAL=${EXTRA_EVAL:-} N_CHECK=${N_CHECK:-70}
export PYTHONUTF8=1   # misma codificacion de texto que en mac
PARALLEL=${PARALLEL:-5}
PAR_CHECK=${PAR_CHECK:-3}
SEEDS=${SEEDS:-"0 1 2"}
PELDANOS=${PELDANOS:-"R2 R4 R1 R0 R3 C"}
DATASETS=${DATASETS:-"gaps_and_forest multiple_bugtraps mazes"}
mkdir -p $ROOT/checks $ROOT/logs
T0=$(date +%s)
log() { echo "[maestro $(date '+%Y-%m-%d %H:%M:%S')] $*"; }
# herramientas portables. mac, linux y git bash en windows
hay() { command -v "$1" >/dev/null 2>&1; }
CAFE=""; hay caffeinate && CAFE="caffeinate -dimsu "
RELANZAR="cd \"$PWD\" && DEVICE=$DEVICE VALEVERY=$VALEVERY VALDENSO=$VALDENSO VALTARDE=$VALTARDE PARALLEL=$PARALLEL nohup ${CAFE}bash experiments/run_escalera.sh >> results/exp_escalera/maestro.log 2>&1 &"
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
    echo "Estado de exp_escalera al $(date '+%Y-%m-%d %H:%M:%S'), maestro pid $$, $(( ($(date +%s) - T0) / 60 )) minutos desde el lanzamiento"
    echo
    echo "Terminado"
    [ -f $ROOT/checks/tests_ok ] && echo "  pruebas"
    [ -f $ROOT/checks/estado.txt ] && echo "  verificaciones $(tr '\n' ' ' < $ROOT/checks/estado.txt)"
    for r in $PELDANOS; do for d in $DATASETS; do for s in $SEEDS; do
      f=$ROOT/runs/$d/${r}_s$s/listo
      [ -f $f ] && echo "  entrenamiento $r $d s$s $(tr '\n' ' ' < $f)$([ -f $ROOT/eval/$d/${r}_s$s.csv ] && echo 'prueba lista')"
    done; done; done
    [ -f $ROOT/RESUMEN_auto.md ] && echo "  resumen RESUMEN_auto.md de las $(hora_de $ROOT/RESUMEN_auto.md)"
    echo
    echo "Corriendo"
    procesos | grep -E "experiments\.(train_escalera|eval_cf|check_escalera|resumen_escalera|figuras_escalera)|tests\.test_" | grep -v grep | sed 's/^/  /' | cut -c1-200
    echo
    echo "Pendientes de la cola"
    if [ -f $ROOT/trabajos.txt ]; then
      while read -r paso d s r; do [ -f $ROOT/eval/$d/${r}_s$s.csv ] || echo "  $r $d s$s"; done < $ROOT/trabajos.txt | head -60
    fi
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
    local nproc=$(procesos | grep -cE "experiments\.(train_escalera|eval_cf|check_escalera)")
    echo "$(date '+%Y-%m-%d %H:%M:%S') intercambio usado $swap paginado ${tasa} MB/s procesos $nproc" >> $ROOT/logs/memoria.log
    if [ $tasa -gt 40 ]; then seguidos=$((seguidos + 1)); else seguidos=0; fi
    [ $seguidos -ge 2 ] && echo "$(date '+%Y-%m-%d %H:%M:%S') AVISO paginado ${tasa} MB/s dos minutos seguidos, bajar PARALLEL o esperar" >> $ROOT/logs/memoria.log
    estado
  done
}

log "inicio, NITER=$NITER VALEVERY=$VALEVERY VALDENSO=$VALDENSO VALTARDE=$VALTARDE DEVICE=$DEVICE PARALLEL=$PARALLEL SEEDS=$SEEDS PELDANOS=$PELDANOS DATASETS=$DATASETS, maquina $(uname -sm)"
estado
vigia &
VIGIA=$!
trap 'kill $VIGIA 2>/dev/null' EXIT

# 1 pruebas
if [ ! -f $ROOT/checks/tests_ok ]; then
  log "pruebas"
  $PY -m tests.test_cf > $ROOT/checks/tests_cf.log 2>&1 && $PY -m tests.test_geo > $ROOT/checks/tests_geo.log 2>&1 && $PY -m tests.test_escalera > $ROOT/checks/tests_escalera.log 2>&1 \
    && touch $ROOT/checks/tests_ok && log "pruebas ok" || { log "FALLO pruebas, ver $ROOT/checks/tests_*.log"; estado; exit 1; }
fi
BASE="$(dirname "$PWD")/code_v3/results"
for ds in $DATASETS; do
  [ -f "$BASE/exp_cf/refs/val_hfs_$ds.csv" ] && [ -f "$BASE/exp_cf/refs/eval/$ds/voraz_hfs.csv" ] && [ -f "$BASE/exp_cf/fotos/$ds/hfs/listo" ] || { log "FALLO faltan referencias o fotos base de $ds en code_v3"; exit 1; }
  for s in $SEEDS; do
    [ -f "$BASE/exp_cf2/cf_filtro/runs/$ds/cf_filtro_s$s/best.pt" ] && [ -f "$BASE/exp_cf/phil_res_s/runs/$ds/phil_res_s_n108_s$s/best.pt" ] || { log "FALLO falta cf_filtro o phil_res_s de $ds s$s en code_v3"; exit 1; }
  done
done

# 2 verificaciones previas, por conjunto y semilla, PAR_CHECK a la vez
if [ ! -f $ROOT/checks/estado.txt ]; then
  log "verificaciones previas, $PAR_CHECK a la vez"
  for ds in $DATASETS; do for s in $SEEDS; do echo "check $ds $s"; done; done | xargs -P $PAR_CHECK -L 1 bash experiments/_run_escalera.sh >> $ROOT/trabajos.log 2>&1
  cat $ROOT/checks/estado_*_s*.txt > $ROOT/checks/estado.txt 2>/dev/null
fi
if grep -q " 0$" $ROOT/checks/estado.txt; then log "hay verificaciones que no pasan, se saltan esos pares"; cat $ROOT/checks/estado.txt; fi
log "verificaciones hechas"
estado

# 3 cola de entrenamiento con prueba, en el orden de los peldanos, semilla por semilla, los tres conjuntos
jobs=$ROOT/trabajos.txt; : > $jobs
for r in $PELDANOS; do for s in $SEEDS; do for ds in $DATASETS; do
  if grep -qE "^(epoca_cero|fuga)_${ds}_s$s 0$" $ROOT/checks/estado.txt; then log "se salta $r $ds s$s, su verificacion no paso"; continue; fi
  echo "train $ds $s $r" >> $jobs
done; done; done
log "cola de $(wc -l < $jobs) trabajos, $PARALLEL a la vez, validacion cada $VALEVERY hasta la $VALDENSO y cada $VALTARDE despues"
xargs -P $PARALLEL -L 1 bash experiments/_run_escalera.sh < $jobs >> $ROOT/trabajos.log 2>&1
log "cola lista, fallos en trabajos.log $(grep -c FALLO $ROOT/trabajos.log 2>/dev/null || echo 0)"
estado

# 4 resumen y figuras
log "resumen y figuras"
if [ -f experiments/resumen_escalera.py ]; then
  $PY -m experiments.resumen_escalera --root $ROOT > $ROOT/resumen.log 2>&1 || log "FALLO resumen, ver $ROOT/resumen.log"
else
  log "FALLO resumen, falta experiments/resumen_escalera.py, relanzar cuando exista"
fi
if [ -f experiments/figuras_escalera.py ]; then
  $PY -m experiments.figuras_escalera --root $ROOT > $ROOT/figuras.log 2>&1 || log "FALLO figuras, ver $ROOT/figuras.log"
fi
log "fin, $(( ($(date +%s) - T0) / 60 )) minutos, fallos en trabajos.log $(grep -c FALLO $ROOT/trabajos.log 2>/dev/null || echo 0)"
estado
