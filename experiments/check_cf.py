# verificaciones previas del experimento cf
#   1 cf* en [0, 1] en todas las celdas etiquetadas de las fotos
#   2 con cf* del oraculo la busqueda reproduce la voraz con h*
#   3 con cf = 1 la busqueda reproduce la voraz con h_fs, tambien la epoca 0 de cf_global y cf_local
#   4 la red solo ve el mapa descubierto
#   python -m experiments.check_cf --root results/exp_cf
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from experiments.common import DATASETS, ESCALA, MAX_EXP, grids_de, hstar_mapa, load_problem, set_threads, write_csv
from src.cf.busqueda import Evaluador, EvaluadorCF, EvaluadorOraculo, voraz
from src.cf.fotos import Fotos, canales, etiqueta, hfs_entero
from src.cf.red import CfConfig, construir
from src.mapa_oculto.search import greedy_fs
from src.search.baselines import _greedy_best_first


def check_cf_estrella(root):
    filas, ok_total = [], True
    for ds in DATASETS:
        fp = root / "fotos" / ds / "hfs"
        if not (fp / "listo").exists():
            filas.append([ds, "", "", "", "", "", 0]); ok_total = False
            continue
        fotos = Fotos.cargar(fp)
        n, mn, mx, fuera = 0, 1.0, 0.0, 0
        for i in range(len(fotos)):
            m = int(fotos.mapa_idx[i])
            y, msk = etiqueta(fotos.hfs[i], fotos.hstar[m], fotos.real[m], fotos.goal[m])
            v = y[msk]
            n += len(v); mn = min(mn, float(v.min())); mx = max(mx, float(v.max()))
            fuera += int(((v < 0) | (v > 1 + 1e-6)).sum())
        ok = fuera == 0
        ok_total &= ok
        filas.append([ds, len(fotos), n, round(mn, 4), round(mx, 4), fuera, int(ok)])
    write_csv(root / "checks" / "cf_estrella.csv", ["dataset", "fotos", "celdas", "min", "max", "fuera_de_rango", "pasa"], filas)
    return ok_total


def check_oraculo(root, n):
    filas, ok_total = [], True
    for ds in DATASETS:
        iguales, tot = 0, 0
        for gp in grids_de(ds, "test")[:n]:
            data, adj, v_s, v_g, h_star = load_problem(gp)
            if int(h_star[v_s]) < 0:
                continue
            hs = h_star.numpy().astype(np.float64); hs[hs < 0] = np.inf
            c1, f1, p1 = _greedy_best_first(hs, v_s, v_g, adj, MAX_EXP)
            res = voraz(EvaluadorOraculo(data, v_g, hstar_mapa(data, h_star), 16, ESCALA), data, v_s, v_g, adj, max_exp=MAX_EXP)
            tot += 1; iguales += int(res.found == f1 and res.closed == c1 and res.path == p1)
        ok = iguales == tot
        ok_total &= ok
        filas.append([ds, tot, iguales, int(ok)])
    write_csv(root / "checks" / "oraculo_igual_hstar.csv", ["dataset", "mapas", "iguales", "pasa"], filas)
    return ok_total


def check_cf_uno(root, n):
    filas, ok_total = [], True
    modelos = {"cf_global": construir(CfConfig(tipo="global", ancho=16)).eval(), "cf_local": construir(CfConfig(tipo="local", ancho=16)).eval()}
    for ds in DATASETS:
        cuenta = {"cf_1": 0, "cf_global_epoca0": 0, "cf_local_epoca0": 0}
        tot, tot0 = 0, 0
        for k, gp in enumerate(grids_de(ds, "test")[:n]):
            data, adj, v_s, v_g, h_star = load_problem(gp)
            if int(h_star[v_s]) < 0:
                continue
            c1, f1, p1, _ = greedy_fs(data, v_s, v_g, adj, 16, MAX_EXP)
            tot += 1
            res = voraz(Evaluador(data, v_g, 16, ESCALA), data, v_s, v_g, adj, max_exp=MAX_EXP)
            cuenta["cf_1"] += int(res.found == f1 and res.closed == c1 and res.path == p1)
            if k < 15:                                                           # la epoca 0 en los primeros 15 mapas, es mas lenta
                tot0 += 1
                for nombre, m in modelos.items():
                    r2 = voraz(EvaluadorCF(data, v_g, m, 20, 16, ESCALA), data, v_s, v_g, adj, max_exp=MAX_EXP)
                    cuenta[f"{nombre}_epoca0"] += int(r2.found == f1 and r2.closed == c1 and r2.path == p1)
        ok = cuenta["cf_1"] == tot and cuenta["cf_global_epoca0"] == tot0 and cuenta["cf_local_epoca0"] == tot0
        ok_total &= ok
        filas.append([ds, tot, cuenta["cf_1"], tot0, cuenta["cf_global_epoca0"], cuenta["cf_local_epoca0"], int(ok)])
    write_csv(root / "checks" / "cf_uno_igual_hfs.csv",
              ["dataset", "mapas", "iguales_cf_1", "mapas_epoca0", "iguales_cf_global_epoca0", "iguales_cf_local_epoca0", "pasa"], filas)
    return ok_total


def check_solo_descubierto(root):
    filas, ok_total = [], True
    torch.manual_seed(0)
    modelos = {"cf_global": construir(CfConfig(tipo="global", ancho=16)).eval(), "cf_local": construir(CfConfig(tipo="local", ancho=16)).eval()}
    for m in modelos.values():
        with torch.no_grad():
            for p in m.cabeza.conv.parameters():
                p.normal_()
    rng = np.random.default_rng(0)
    for ds in DATASETS:
        for gp in grids_de(ds, "test")[:3]:
            data, adj, v_s, v_g, h_star = load_problem(gp)
            ev = Evaluador(data, v_g, 16, ESCALA)
            voraz(ev, data, v_s, v_g, adj, max_exp=200)
            x0 = canales(ev.desc.visto, ev.desc.real, ev.goal, hfs_entero(ev.fs.h), ESCALA)
            real2 = ev.desc.real.copy()
            nv = np.argwhere(~ev.desc.visto)
            for r, c in nv[rng.choice(len(nv), 300, replace=False)]:
                real2[r, c] = ~real2[r, c]
            x1 = canales(ev.desc.visto, real2, ev.goal, hfs_entero(ev.fs.h), ESCALA)
            misma_entrada = np.array_equal(x0, x1)
            for nombre, m in modelos.items():
                with torch.no_grad():
                    o0, o1 = m(torch.from_numpy(x0)[None]), m(torch.from_numpy(x1)[None])
                ok = misma_entrada and torch.equal(o0, o1)
                ok_total &= ok
                filas.append([ds, gp.name, nombre, int(ev.desc.visto.sum()), 300, int(misma_entrada), int(torch.equal(o0, o1)), int(ok)])
    write_csv(root / "checks" / "solo_descubierto.csv",
              ["dataset", "grid", "modelo", "celdas_vistas", "celdas_cambiadas", "misma_entrada", "misma_salida", "pasa"], filas)
    return ok_total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results/exp_cf")
    p.add_argument("--n", type=int, default=100)
    a = p.parse_args()
    set_threads(1)
    root = Path(a.root)
    (root / "checks").mkdir(parents=True, exist_ok=True)
    res = {}
    for nombre, fn in [("cf_estrella", lambda: check_cf_estrella(root)), ("oraculo_igual_hstar", lambda: check_oraculo(root, a.n)),
                       ("cf_uno_igual_hfs", lambda: check_cf_uno(root, a.n)), ("solo_descubierto", lambda: check_solo_descubierto(root))]:
        t0 = time.time()
        res[nombre] = fn()
        print(f"[check] {nombre}: {'pasa' if res[nombre] else 'FALLA'} ({time.time() - t0:.0f}s)", flush=True)
    write_csv(root / "checks" / "resumen_checks.csv", ["check", "pasa"], [[k, int(v)] for k, v in res.items()])
    (root / "checks" / "estado.txt").write_text("".join(f"{k} {int(v)}\n" for k, v in res.items()))
    print("todas pasan" if all(res.values()) else "HAY FALLOS")


if __name__ == "__main__":
    main()
