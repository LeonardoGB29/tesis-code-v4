# busca con h_geo en un split, una fila por mapa con el mismo formato que eval_cf
#   python -m experiments.eval_geo --dataset mazes --ckpt <best.pt> --lam 4 --out <csv> [--split validation]
#   python -m experiments.eval_geo --dataset mazes --real --lam 4 --out <csv>        (ocupacion real en vez de la cnn)
import argparse
import random
import time
from pathlib import Path

import numpy as np

from experiments.common import ESCALA, MAX_EXP, grids_de, load_problem, set_threads, write_csv
from experiments.eval_cf import HEADER
from src.cf.busqueda import voraz
from src.geo.hgeo import EvaluadorGeo, EvaluadorGeoReal
from src.geo.red import cargar_geo


def correr(dataset, split, hacer, run, out, n_grids=0, max_exp=MAX_EXP, epoca="", verbose=False):
    filas = []
    grids = grids_de(dataset, split)
    for k, gp in enumerate(grids[:n_grids] if n_grids else grids):
        data, adj, v_s, v_g, h_star = load_problem(gp)
        opt = int(h_star[v_s])
        if opt < 0:
            continue
        random.seed(0)
        t0 = time.time()
        ev = hacer(data, adj, v_g)
        res = voraz(ev, data, v_s, v_g, adj, max_exp=max_exp)
        secs = time.time() - t0
        filas.append([dataset, run, gp.name, opt, res.n_expanded if res.found else max_exp, int(res.found), (len(res.path) - 1) if res.path else "",
                      round(secs, 4), round(res.secs_red, 4), res.n_red, epoca])
        if verbose:
            print(f"[{k + 1:3d}] {gp.name} exp={filas[-1][4]} opt={opt} ({secs:.2f}s)", flush=True)
    write_csv(out, HEADER, filas)
    e = np.array([r[4] for r in filas], dtype=float)
    secs = sum(r[7] for r in filas)
    print(f"[{dataset}/{run}] {split} n={len(filas)} exp={e.mean():.1f} ms/nodo={1000 * secs / max(e.sum(), 1):.3f} -> {out}", flush=True)
    return filas


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--lam", type=float, required=True)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--real", action="store_true")
    p.add_argument("--split", default="test", choices=["test", "validation"])
    p.add_argument("--name", default=None)
    p.add_argument("--every-cf", type=int, default=20)
    p.add_argument("--every-fs", type=int, default=16)
    p.add_argument("--n-grids", type=int, default=0)
    p.add_argument("--max-exp", type=int, default=MAX_EXP)
    p.add_argument("--device", default="cpu")
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args()
    assert (a.ckpt is None) != (not a.real), "uno de --ckpt o --real"
    set_threads(1)
    if a.real:
        run = a.name or f"geo_real_lam{a.lam:g}"
        hacer = lambda data, adj, v_g: EvaluadorGeoReal(data, v_g, a.lam, a.every_fs, ESCALA)
        epoca = ""
    else:
        model, cfg, raw = cargar_geo(a.ckpt, a.device)
        run = a.name or f"{Path(a.ckpt).parent.name}_lam{a.lam:g}"
        hacer = lambda data, adj, v_g: EvaluadorGeo(data, v_g, model, a.lam, a.every_cf, a.every_fs, ESCALA, a.device)
        epoca = raw.get("epoca", "")
    correr(a.dataset, a.split, hacer, run, a.out, a.n_grids, a.max_exp, epoca, a.verbose)


if __name__ == "__main__":
    main()
