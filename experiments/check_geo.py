# verificaciones previas de exp_geo
#   1 constante. la cnn recien construida da logit(p0) exacto en todas las celdas y lo mismo en cpu y en mps
#   2 epoca cero. con lambda 0, h_geo con la cnn y con la ocupacion real reproduce la voraz con h_fs en los mapas de prueba, expansion por expansion
#   3 fuga. cambiar celdas no vistas del mapa real no cambia la cnn de geometria, ni cf_filtro, ni los h_geo de la frontera
#   4 tiempos. ms por nodo de h_fs, cf_filtro y h_geo en 10 mapas de prueba, a solas
#   python -m experiments.check_geo --root results/exp_geo --dataset mazes [--n 100]
import argparse
import copy
import math
import time
from pathlib import Path

import numpy as np
import torch

from experiments.common import DATASETS, ESCALA, EXP_CF2, MAX_EXP, grids_de, load_problem, set_threads, write_csv
from src.cf.busqueda import Evaluador, EvaluadorCF, voraz
from src.cf.fotos import canales, hfs_entero
from src.cf.red import cargar
from src.geo.fotos import canales_geo
from src.geo.hgeo import EvaluadorGeo, EvaluadorGeoReal, distancia_geo
from src.geo.red import GeoConfig, construir_geo
from src.mapa_oculto.search import greedy_fs


def modelo_prueba(seed=0):
    # cabeza con pesos al azar para que la cnn no sea constante
    torch.manual_seed(seed)
    m = construir_geo(GeoConfig(p0=0.3))
    with torch.no_grad():
        m.logit.weight.normal_(0, 0.5)
    return m.eval()


def check_constante(root, ds):
    filas, ok = [], True
    m = construir_geo(GeoConfig(p0=0.3)).eval()
    x = torch.rand(2, 3, 208, 208)
    esperado = math.log(0.3 / 0.7)
    with torch.no_grad():
        cpu = m(x)
    exacto = bool(torch.all(cpu == cpu.flatten()[0])) and abs(float(cpu.flatten()[0]) - esperado) < 1e-6
    ok &= exacto
    filas.append([ds, "cpu", float(cpu.min()), float(cpu.max()), esperado, int(exacto), ""])
    if torch.backends.mps.is_available():
        with torch.no_grad():
            mps = m.to("mps")(x.to("mps")).cpu()
        m.to("cpu")
        dif = float((mps - cpu).abs().max())
        igual = dif < 1e-5
        ok &= igual
        filas.append([ds, "mps", float(mps.min()), float(mps.max()), esperado, int(igual), f"{dif:.2e}"])
    write_csv(root / "checks" / f"constante_{ds}.csv", ["dataset", "device", "min", "max", "esperado", "pasa", "dif_con_cpu"], filas)
    print(f"[{ds}] constante: {'pasa' if ok else 'NO pasa'}", flush=True)
    return ok


def check_epoca_cero(root, ds, n):
    filas, ok = [], True
    m = modelo_prueba()
    t0 = time.time()
    for gp in grids_de(ds, "test")[:n]:
        data, adj, v_s, v_g, h_star = load_problem(gp)
        if int(h_star[v_s]) < 0:
            continue
        c1, f1, p1, _ = greedy_fs(data, v_s, v_g, adj, 16, MAX_EXP)
        r = voraz(EvaluadorGeo(data, v_g, m, 0.0, 20, 16, ESCALA), data, v_s, v_g, adj, max_exp=MAX_EXP)
        igual = r.found == f1 and r.closed == c1 and r.path == p1 and r.n_red >= r.n_expanded // 20
        rr = voraz(EvaluadorGeoReal(data, v_g, 0.0, 16, ESCALA), data, v_s, v_g, adj, max_exp=MAX_EXP)
        igual_r = rr.found == f1 and rr.closed == c1 and rr.path == p1
        ok &= igual and igual_r
        filas.append([ds, gp.name, len(c1), r.n_expanded, int(igual), rr.n_expanded, int(igual_r), r.n_red])
    write_csv(root / "checks" / f"epoca_cero_{ds}.csv", ["dataset", "grid", "exp_hfs", "exp_geo_lam0", "igual_cnn", "exp_real_lam0", "igual_real", "llamadas_cnn"], filas)
    print(f"[{ds}] epoca cero: cnn {sum(f[4] for f in filas)} de {len(filas)} iguales a h_fs, real {sum(f[6] for f in filas)} de {len(filas)} ({time.time() - t0:.0f}s)", flush=True)
    return ok


def check_fuga(root, ds, n=3, n_exp=150, n_cambios=200):
    filas, ok = [], True
    m = modelo_prueba()
    cf, cfg_cf, _ = cargar(EXP_CF2 / "cf_filtro" / "runs" / ds / "cf_filtro_s0" / "best.pt")
    rng = np.random.default_rng(0)
    for gp in grids_de(ds, "test")[:n]:
        data, adj, v_s, v_g, h_star = load_problem(gp)
        if int(h_star[v_s]) < 0:
            continue
        ev = EvaluadorGeo(data, v_g, m, 4.0, 20, 16, ESCALA)
        voraz(ev, data, v_s, v_g, adj, max_exp=n_exp)
        visto, real, goal = ev.desc.visto, ev.desc.real, ev.goal
        no_vistas = np.argwhere(~visto)
        real2 = real.copy()
        for r, c in no_vistas[rng.choice(len(no_vistas), min(n_cambios, len(no_vistas)), replace=False)]:
            real2[r, c] = ~real2[r, c]
        with torch.no_grad():
            g0, g1 = m(torch.from_numpy(canales_geo(visto, real))[None]), m(torch.from_numpy(canales_geo(visto, real2))[None])
            hfs = hfs_entero(ev.fs.h)
            c0, c1 = cf(torch.from_numpy(canales(visto, real, goal, hfs, ESCALA))[None]), cf(torch.from_numpy(canales(visto, real2, goal, hfs, ESCALA))[None])
        igual_geo, igual_cf = torch.equal(g0, g1), torch.equal(c0, c1)
        # los h_geo de un estado de busqueda con el mapa cambiado en lo no visto
        ev2 = copy.copy(ev)
        ev2.desc = ev.desc.copia()
        ev2.desc.real = real2
        p1, p2 = ev.calcular(), ev2.calcular()
        h1 = distancia_geo(ev.desc.known_inmap(), ev.desc.visto, p1, goal, 4.0)
        h2 = distancia_geo(ev2.desc.known_inmap(), ev2.desc.visto, p2, goal, 4.0)
        igual_h = np.array_equal(p1, p2) and np.array_equal(h1, h2)
        # control, cambiar celdas vistas si cambia la cnn
        vistas = np.argwhere(visto & ~real)
        real3 = real.copy()
        for r, c in vistas[rng.choice(len(vistas), min(20, len(vistas)), replace=False)]:
            real3[r, c] = True
        with torch.no_grad():
            g3 = m(torch.from_numpy(canales_geo(visto, real3))[None])
        cambia = not torch.equal(g0, g3)
        ok &= igual_geo and igual_cf and igual_h and cambia
        filas.append([ds, gp.name, int(visto.sum()), n_cambios, int(igual_geo), int(igual_cf), int(igual_h), int(cambia)])
    write_csv(root / "checks" / f"fuga_{ds}.csv", ["dataset", "grid", "celdas_vistas", "celdas_cambiadas", "cnn_geo_igual", "cf_filtro_igual", "h_geo_igual", "cambia_si_cambia_lo_visto"], filas)
    print(f"[{ds}] fuga: {'pasa' if ok else 'NO pasa'} en {len(filas)} mapas", flush=True)
    return ok


def check_tiempos(root, ds, n=10, lam=4.0):
    m = modelo_prueba()
    cf, cfg_cf, _ = cargar(EXP_CF2 / "cf_filtro" / "runs" / ds / "cf_filtro_s0" / "best.pt")
    busquedas = {"voraz_hfs": lambda data, adj, v_g: Evaluador(data, v_g, 16, ESCALA),
                 "cf_filtro_s0": lambda data, adj, v_g: EvaluadorCF(data, v_g, cf, cfg_cf.every_cf, cfg_cf.every_fs, ESCALA),
                 f"h_geo_lam{lam:g}": lambda data, adj, v_g: EvaluadorGeo(data, v_g, m, lam, 20, 16, ESCALA),
                 f"h_geo_real_lam{lam:g}": lambda data, adj, v_g: EvaluadorGeoReal(data, v_g, lam, 16, ESCALA)}
    filas = []
    for nombre, hacer in busquedas.items():
        exp, secs, red, n_red, mapas = 0, 0.0, 0.0, 0, 0
        for gp in grids_de(ds, "test")[:n]:
            data, adj, v_s, v_g, h_star = load_problem(gp)
            if int(h_star[v_s]) < 0:
                continue
            t0 = time.time()
            r = voraz(hacer(data, adj, v_g), data, v_s, v_g, adj, max_exp=MAX_EXP)
            secs += time.time() - t0
            exp += r.n_expanded; red += r.secs_red; n_red += r.n_red; mapas += 1
        filas.append([ds, nombre, mapas, exp, round(secs, 2), round(1000 * secs / max(exp, 1), 3), round(1000 * red / max(exp, 1), 3), round(secs / max(mapas, 1), 3), n_red])
        print(f"[{ds}] tiempos {nombre}: {filas[-1][5]} ms por nodo, {filas[-1][7]} s por mapa, red {filas[-1][6]} ms por nodo", flush=True)
    write_csv(root / "checks" / f"tiempos_{ds}.csv", ["dataset", "busqueda", "mapas", "exp_total", "secs", "ms_nodo", "ms_nodo_red", "s_mapa", "llamadas_red"], filas)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results/exp_geo")
    p.add_argument("--dataset", default=None)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--n-tiempos", type=int, default=10)
    a = p.parse_args()
    set_threads(1)
    root = Path(a.root)
    (root / "checks").mkdir(parents=True, exist_ok=True)
    lineas = []
    for ds in ([a.dataset] if a.dataset else DATASETS):
        lineas.append(f"constante_{ds} {int(check_constante(root, ds))}")
        lineas.append(f"fuga_{ds} {int(check_fuga(root, ds))}")
        lineas.append(f"epoca_cero_{ds} {int(check_epoca_cero(root, ds, a.n))}")
        lineas.append(f"tiempos_{ds} {int(check_tiempos(root, ds, a.n_tiempos))}")
    out = root / "checks" / (f"estado_{a.dataset}.txt" if a.dataset else "estado.txt")
    out.write_text("\n".join(lineas) + "\n")
    print("\n".join(lineas))


if __name__ == "__main__":
    main()
