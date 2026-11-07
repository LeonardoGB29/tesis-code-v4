# tesis. tiempos limpios de la busqueda en una sola maquina y un solo proceso, semilla 0, los 100 mapas de prueba de cada conjunto
#   python -m experiments.tiempos_limpios --out results/tesis/tiempos --nucleo 4 --probar-nucleos 0 2 4 10 12 14 16 19
# el i7-12700H tiene nucleos de rendimiento, logicos 0 a 11, y de eficiencia, 12 a 19, y windows mueve un proceso de un hilo entre los dos.
# por eso el proceso se fija a un nucleo logico de rendimiento con --nucleo, y --probar-nucleos deja en nucleos.csv el costo de la U-Net en cada uno
# corre el mismo codigo de la prueba, eval_cf.evaluar con un hilo de torch, un metodo detras del otro dentro de este proceso.
# cada metodo se calienta antes con 2 mapas que no se guardan. la U-Net de cf_filtro va en CPU como en la prueba y otra vez en CUDA.
# PHIL, R0 y phil_res_s corren en CPU porque el codigo de prueba no los mueve de dispositivo.
# PHIL viejo de exp_cf se evaluo en la Mac desde exp_cf/base/runs/<conjunto>/phil_s_s<k>/best.pt, que no esta en esta maquina. queda anotado en
# faltantes.csv y en su lugar se mide R0 semilla 0 de exp_escalera, que es la misma clase PHIL reentrenada aca. cuando lleguen esos best.pt,
#   python -m experiments.tiempos_limpios --out results/tesis/tiempos --nucleo 4 --metodos phil_viejo_s0
# mide solo el PHIL viejo y deja los demas tiempos como estan
import argparse
import ctypes
import os
import platform
import sys
import tempfile
import threading
import time
from argparse import Namespace
from pathlib import Path

import torch

from experiments.common import DATASETS, EXP_CF2, MAX_EXP, RESULTS, leer_csv, set_threads, write_csv
from experiments.eval_cf import evaluar
from experiments.train_escalera import ROOT_ESC


def metodos(ds):
    cf = EXP_CF2 / "cf_filtro" / "runs" / ds / "cf_filtro_s0" / "best.pt"
    return [("voraz_hstar", "cpu", {"ref": "hstar"}),
            ("voraz_hfs", "cpu", {"ref": "hfs"}),
            ("phil_viejo_s0", "cpu", {"phil": RESULTS / "base" / "runs" / ds / "phil_s_s0" / "best.pt"}),
            ("R0_s0", "cpu", {"phil": ROOT_ESC / "runs" / ds / "R0_s0" / "best.pt"}),
            ("phil_res_s_s0", "cpu", {"phil": RESULTS / "phil_res_s" / "runs" / ds / "phil_res_s_n108_s0" / "best.pt"}),
            ("cf_filtro_s0", "cpu", {"ckpt": cf}),
            ("cf_filtro_s0", "cuda", {"ckpt": cf})]


def args_de(ds, nombre, dev, fuente, out, n_grids=0):
    a = Namespace(dataset=ds, out=str(out), ckpt=None, phil=None, mem=None, ref=None, name=nombre, ruido=0.0, ruido_seed=0, cf_fijo=None,
                  every_fs=16, n_grids=n_grids, max_exp=MAX_EXP, device=dev, verbose=False)
    for k, v in fuente.items():
        setattr(a, k, str(v))
    return a


def fijar_mascara(mascara):
    # afinidad del proceso a los nucleos logicos de la mascara, devuelve si windows la acepto
    try:
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        k32.SetProcessAffinityMask.restype = ctypes.c_int
        return bool(k32.SetProcessAffinityMask(k32.GetCurrentProcess(), ctypes.c_size_t(mascara)))
    except Exception:
        return False


def fijar_nucleo(cpu):
    return fijar_mascara(1 << cpu)


def procesador():
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
    except Exception:
        return platform.processor()


def enchufada():
    # 1 enchufada, 0 a bateria, vacio si no se sabe
    try:
        class SPS(ctypes.Structure):
            _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte), ("BatteryLifePercent", ctypes.c_ubyte),
                        ("SystemStatusFlag", ctypes.c_ubyte), ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]
        s = SPS()
        ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s))
        return {0: 0, 1: 1}.get(s.ACLineStatus, "")
    except Exception:
        return ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/tesis/tiempos")
    p.add_argument("--datasets", nargs="*", default=DATASETS)
    p.add_argument("--n-grids", type=int, default=0, help="0 son los 100 mapas, otro numero solo para el humo")
    p.add_argument("--calentar", type=int, default=2)
    p.add_argument("--nucleo", type=int, default=-1, help="nucleo logico fijo para todo el proceso, -1 deja que decida windows")
    p.add_argument("--probar-nucleos", type=int, nargs="*", default=[], help="costo de la U-Net de cf_filtro en 2 mapas de gaps_and_forest en cada nucleo")
    p.add_argument("--metodos", nargs="*", default=[], help="solo estos metodos, sin tocar lo ya medido de los demas ni entorno.csv")
    a = p.parse_args()
    set_threads(1)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    estado = out / "estado.txt"
    paso = {"txt": "arrancando"}
    listo = threading.Event()

    def escribir_estado():
        estado.write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {paso['txt']}\n")

    def latido():
        # despierto.py vigila este archivo y se va cuando lleva 5 minutos quieto
        while not listo.wait(60):
            escribir_estado()

    threading.Thread(target=latido, daemon=True).start()
    escribir_estado()
    inicio = time.strftime("%Y-%m-%d %H:%M:%S")
    corridas, faltantes = [], []
    cola = [(dev, ds, nombre, fuente) for dev in ("cpu", "cuda") for ds in a.datasets for (nombre, d, fuente) in metodos(ds)
            if d == dev and (not a.metodos or nombre in a.metodos)]
    with tempfile.TemporaryDirectory() as tmp:
        filas_n = []
        for cpu in a.probar_nucleos:
            paso["txt"] = f"probando el nucleo {cpu}"
            escribir_estado()
            ok = fijar_nucleo(cpu)
            cf = {"ckpt": EXP_CF2 / "cf_filtro" / "runs" / "gaps_and_forest" / "cf_filtro_s0" / "best.pt"}
            evaluar(args_de("gaps_and_forest", "cf_filtro_s0", "cpu", cf, Path(tmp) / "calentar.csv", 1))
            dest = Path(tmp) / f"nucleo_{cpu}.csv"
            evaluar(args_de("gaps_and_forest", "cf_filtro_s0", "cpu", cf, dest, 2))
            r = leer_csv(dest)
            sr, nr = sum(float(x["secs_red"]) for x in r), sum(int(x["n_red"]) for x in r)
            secs, exp = sum(float(x["secs"]) for x in r), sum(float(x["exp"]) for x in r)
            filas_n.append([cpu, int(ok), len(r), nr, round(1000 * sr / max(nr, 1), 1), round(1000 * secs / max(exp, 1), 3)])
        if filas_n:
            write_csv(out / "nucleos.csv", ["nucleo logico", "afinidad aceptada", "mapas", "llamadas a la U-Net", "ms por llamada a la U-Net", "ms por nodo"], filas_n)
        if a.nucleo >= 0:
            fijo = fijar_nucleo(a.nucleo)
        else:
            fijo = False
            if filas_n:
                fijar_mascara((1 << (os.cpu_count() or 1)) - 1)       # la prueba por nucleo no deja el proceso fijo en el ultimo probado
        print(f"nucleo fijo {a.nucleo}, afinidad aceptada {fijo}", flush=True)
        for dev, ds, nombre, fuente in cola:
            ck = fuente.get("phil") or fuente.get("ckpt")
            if ck is not None and not Path(ck).exists():
                faltantes.append([ds, nombre, dev, "no hay checkpoint en esta maquina", str(ck)])
                print(f"[{ds}/{nombre}] falta {ck}, no se mide", flush=True)
                continue
            if dev == "cuda" and not torch.cuda.is_available():
                faltantes.append([ds, nombre, dev, "no hay CUDA", ""])
                continue
            paso["txt"] = f"{dev} {ds} {nombre}"
            escribir_estado()
            if a.calentar:
                evaluar(args_de(ds, nombre, dev, fuente, Path(tmp) / "calentar.csv", a.calentar))
            t0, ini = time.time(), time.strftime("%Y-%m-%d %H:%M:%S")
            evaluar(args_de(ds, nombre, dev, fuente, out / dev / ds / f"{nombre}.csv", a.n_grids))
            corridas.append([dev, ds, nombre, ini, time.strftime("%Y-%m-%d %H:%M:%S"), round(time.time() - t0, 1)])
    if a.metodos:
        # corrida parcial, se conservan las filas de los otros metodos y entorno.csv queda como estaba
        for nombre_csv, filas, k_ds, k_m, k_dev in (("corridas.csv", corridas, 1, 2, 0), ("faltantes.csv", faltantes, 0, 1, 2)):
            p_csv = out / nombre_csv
            if p_csv.exists():
                previas = [list(x.values()) for x in leer_csv(p_csv)]
                nuevas = {(r[k_ds], r[k_m], r[k_dev]) for r in filas}
                hechas = {(ds, nombre, dev) for dev, ds, nombre, _ in cola}
                filas[:0] = [r for r in previas if (r[k_ds], r[k_m], r[k_dev]) not in nuevas | hechas]
    write_csv(out / "corridas.csv", ["dispositivo", "dataset", "metodo", "inicio", "fin", "segundos"], corridas)
    write_csv(out / "faltantes.csv", ["dataset", "metodo", "dispositivo", "motivo", "ruta"], faltantes)
    if a.metodos:
        paso["txt"] = "terminado"
        escribir_estado()
        listo.set()
        print(f"-> {out}", flush=True)
        return
    entorno = [["inicio", inicio], ["fin", time.strftime("%Y-%m-%d %H:%M:%S")], ["procesador", procesador()],
               ["gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""], ["python", sys.version.split()[0]], ["torch", torch.__version__],
               ["hilos de torch", torch.get_num_threads()], ["procesos", 1], ["nucleo logico fijo", a.nucleo if fijo else "no"], ["enchufada", enchufada()], ["mapas por conjunto", a.n_grids or "todos los de prueba"],
               ["calentamiento", f"{a.calentar} mapas por metodo, no se guardan"]]
    write_csv(out / "entorno.csv", ["clave", "valor"], entorno)
    paso["txt"] = "terminado"
    escribir_estado()
    listo.set()
    print(f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
