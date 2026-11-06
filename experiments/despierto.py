# caffeinate para windows. mantiene la maquina despierta mientras el maestro corre
#   nohup .venv/bin/python experiments/despierto.py results/exp_geo > results/exp_geo/logs/despierto.log 2>&1 &
# pide al sistema que no entre en suspension por inactividad y se va solo cuando estado.txt lleva 5 minutos sin reescribirse,
# que es la senal de que el vigia del maestro ya no esta. en mac o linux no hace nada util y termina enseguida
import ctypes
import sys
import time
from pathlib import Path

ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results/exp_geo")
    estado = root / "estado.txt"
    if sys.platform != "win32":
        print("no es windows, no hace falta", flush=True)
        return
    k32 = ctypes.windll.kernel32
    k32.SetThreadExecutionState.restype = ctypes.c_uint32
    previo = k32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} despierto, estado previo {previo:#x}, vigilo {estado}", flush=True)
    t0 = time.time()
    try:
        while True:
            time.sleep(60)
            edad = time.time() - estado.stat().st_mtime if estado.exists() else time.time() - t0
            if edad > 300:
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} estado.txt lleva {edad / 60:.0f} minutos quieto, el maestro termino, me voy", flush=True)
                break
    finally:
        k32.SetThreadExecutionState(ES_CONTINUOUS)


if __name__ == "__main__":
    main()
