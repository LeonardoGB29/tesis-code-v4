# fotos del mapa descubierto cada foto_cada expansiones en busquedas sobre los mapas de entrenamiento
#   python -m experiments.fotos --dataset mazes            -> results/exp_cf/fotos/mazes/hfs/ y refs de validacion
import argparse
import time
from pathlib import Path

import numpy as np

from experiments.common import ESCALA, MAX_EXP, RESULTS, grids_de, hstar_mapa, load_problem, set_threads, write_csv
from src.cf.busqueda import Evaluador, voraz
from src.cf.fotos import Fotos, hfs_entero

FOTO_CADA = 50


def recolectar(dataset, hacer_evaluador, foto_cada=FOTO_CADA, max_exp=MAX_EXP, split="train", verbose=False):
    # hacer_evaluador(data, adj, v_g) -> evaluador; devuelve Fotos y las filas de la busqueda
    nombres, reales, hstars, goals, starts = [], [], [], [], []
    idx, vistos, hfss, filas = [], [], [], []
    for m, gp in enumerate(grids_de(dataset, split)):
        data, adj, v_s, v_g, h_star = load_problem(gp)
        opt = int(h_star[v_s])
        if opt < 0:
            continue
        ev = hacer_evaluador(data, adj, v_g)
        t0 = time.time()
        res = voraz(ev, data, v_s, v_g, adj, max_exp=max_exp, foto_cada=foto_cada)
        nombres.append(gp.name); reales.append(ev.desc.real); hstars.append(hstar_mapa(data, h_star))
        goals.append(ev.goal); starts.append((int(ev.rc[v_s, 0]), int(ev.rc[v_s, 1])))
        k = len(nombres) - 1
        for v, h in res.fotos:
            idx.append(k); vistos.append(v); hfss.append(h)
        filas.append([dataset, gp.name, opt, res.n_expanded if res.found else max_exp, int(res.found),
                      (len(res.path) - 1) if res.path else "", len(res.fotos), round(time.time() - t0, 3)])
        if verbose:
            print(f"[{k + 1:3d}] {gp.name} exp={filas[-1][3]} fotos={len(res.fotos)} ({filas[-1][-1]}s)", flush=True)
    fotos = Fotos(nombres, np.stack(reales), np.stack(hstars), np.array(goals, dtype=np.int16), np.array(starts, dtype=np.int16),
                  np.array(idx, dtype=np.int32), np.stack(vistos), np.stack(hfss), ESCALA)
    return fotos, filas


def referencia(dataset, split, out, every_fs=16, max_exp=MAX_EXP):
    # voraz con h_fs en un split, una fila por mapa
    filas = []
    for gp in grids_de(dataset, split):
        data, adj, v_s, v_g, h_star = load_problem(gp)
        opt = int(h_star[v_s])
        if opt < 0:
            continue
        t0 = time.time()
        res = voraz(Evaluador(data, v_g, every_fs, ESCALA), data, v_s, v_g, adj, max_exp=max_exp)
        filas.append([dataset, "voraz_hfs", gp.name, opt, res.n_expanded if res.found else max_exp, int(res.found),
                      (len(res.path) - 1) if res.path else "", round(time.time() - t0, 4)])
    write_csv(out, ["dataset", "run", "grid", "optimo", "exp", "encontrado", "camino", "secs"], filas)
    return filas


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--out-dir", default=str(RESULTS / "fotos"))
    p.add_argument("--foto-cada", type=int, default=FOTO_CADA)
    p.add_argument("--every-fs", type=int, default=16)
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args()
    set_threads(1)
    out = Path(a.out_dir) / a.dataset
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    fotos, filas = recolectar(a.dataset, lambda data, adj, v_g: Evaluador(data, v_g, a.every_fs, ESCALA), a.foto_cada, verbose=a.verbose)
    fotos.guardar(out / "hfs")
    write_csv(out / "busquedas_hfs.csv", ["dataset", "grid", "optimo", "exp", "encontrado", "camino", "fotos", "secs"], filas)
    print(f"[{a.dataset}] {len(fotos)} fotos de {len(fotos.nombres)} mapas en {time.time() - t0:.0f}s -> {out / 'hfs'}", flush=True)
    referencia(a.dataset, "validation", RESULTS / "refs" / f"val_hfs_{a.dataset}.csv", a.every_fs)
    print(f"[{a.dataset}] referencia de validacion lista", flush=True)


if __name__ == "__main__":
    main()
