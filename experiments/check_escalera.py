# verificaciones previas de exp_escalera, por conjunto y semilla
#   1 iteracion 0. R2, R3, R4 y C reproducen la busqueda de cf_filtro best.pt de la semilla en los mapas de validacion, expansion por expansion,
#     con el tiempo por mapa y por nodo de cada brazo para la estimacion
#   2 fuga. cambiar 200 celdas no vistas del mapa real no cambia las 43 columnas ni la salida de R2 ni la de C
#   python -m experiments.check_escalera --root results/exp_escalera --dataset mazes --seed 0 [--n 70 --device cuda]
import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch

from experiments.common import ESCALA, EXP_CF2, MAX_EXP, RESULTS, grids_de, load_phil, load_problem, set_threads, write_csv
from experiments.train_escalera import ROOT_ESC, cf_ckpt_de, phil_ckpt_de
from src.cf.busqueda import EvaluadorCF, voraz
from src.cf.escalera import D_CNN, D_RASGOS, Compuerta, Escalera, EscaleraConfig, columnas
from src.cf.red import cargar
from src.search.greedy import greedy_search


def brazos_de(ds, seed, refs, refs2, device):
    ck = cf_ckpt_de(refs2, ds, seed)
    unet, ucfg, _ = cargar(ck, device)
    out = {}
    for r in ("R2", "R3", "R4", "C"):
        torch.manual_seed(seed)
        cfg = EscaleraConfig(peldano=r, d_enc=D_CNN, fs_every=ucfg.every_fs, every_cf=ucfg.every_cf, coord_scale=ESCALA, cf_ckpt=str(ck),
                             phil_ckpt=str(phil_ckpt_de(refs, ds, seed)), rasgos_media=[0.0] * D_RASGOS, rasgos_std=[1.0] * D_RASGOS)
        out[r] = (Compuerta(cfg, load_phil(cfg.phil_ckpt)[0], unet, device) if r == "C" else Escalera(cfg, unet, device)).eval()
    return unet, ucfg, out


def check_epoca_cero(root, ds, seed, n, unet, ucfg, brazos, device):
    nombres = ["cf_filtro"] + list(brazos)
    filas, exps, secs, ok = [], {b: [] for b in nombres}, {b: 0.0 for b in nombres}, True
    t_ini = time.time()
    for gp in grids_de(ds, "validation")[:n]:
        data, adj, v_s, v_g, h_star = load_problem(gp)
        if int(h_star[v_s]) < 0:
            continue
        t0 = time.time()
        ref = voraz(EvaluadorCF(data, v_g, unet, ucfg.every_cf, ucfg.every_fs, ESCALA, device), data, v_s, v_g, adj, max_exp=MAX_EXP)
        secs["cf_filtro"] += time.time() - t0
        exps["cf_filtro"].append(ref.n_expanded if ref.found else MAX_EXP)
        fila = [ds, seed, gp.name, ref.n_expanded]
        for b, m in brazos.items():
            t0 = time.time()
            r = greedy_search(m, data, v_s, v_g, adj=adj, max_exp=MAX_EXP, return_path=True)
            secs[b] += time.time() - t0
            exps[b].append(r.n_expanded if r.found else MAX_EXP)
            igual = r.found == ref.found and r.closed == ref.closed and r.path == ref.path
            ok &= igual
            fila += [r.n_expanded, int(igual)]
        filas.append(fila)
    head = ["dataset", "seed", "grid", "cf_filtro_exp"] + [f"{b}_{k}" for b in brazos for k in ("exp", "igual")]
    write_csv(root / "checks" / f"epoca_cero_{ds}_s{seed}.csv", head, filas)
    n_mapas = len(filas)
    write_csv(root / "checks" / f"tiempos_epoca0_{ds}_s{seed}.csv", ["dataset", "seed", "brazo", "mapas", "exp_media", "secs_por_mapa", "ms_por_nodo"],
              [[ds, seed, b, n_mapas, round(float(np.mean(exps[b])), 1), round(secs[b] / max(n_mapas, 1), 3), round(1000 * secs[b] / max(sum(exps[b]), 1), 3)] for b in nombres])
    iguales = sum(all(f_[5 + 2 * j] for j in range(len(brazos))) for f_ in filas)
    print(f"[{ds} s{seed}] iteracion 0: {iguales} de {n_mapas} mapas identicos a cf_filtro en los 4 brazos, media cf_filtro {np.mean(exps['cf_filtro']):.1f}, "
          f"segundos por mapa " + " ".join(f"{b} {secs[b] / max(n_mapas, 1):.2f}" for b in nombres) + f" ({time.time() - t_ini:.0f}s)", flush=True)
    return ok


def check_fuga(root, ds, seed, brazos, n=3, n_cambios=200):
    filas, ok = [], True
    rng = np.random.default_rng(seed)
    for gp in grids_de(ds, "test")[:n]:
        data, adj, v_s, v_g, h_star = load_problem(gp)
        if int(h_star[v_s]) < 0:
            continue
        fila = [ds, seed, gp.name]
        for r in ("R2", "C"):
            m = brazos[r]
            greedy_search(m, data, v_s, v_g, adj=adj, max_exp=150)
            ev = m.ev
            visto, real = ev.desc.visto, ev.desc.real
            nv = np.argwhere(~visto)
            real2 = real.copy()
            for rr, cc in nv[rng.choice(len(nv), min(n_cambios, len(nv)), replace=False)]:
                real2[rr, cc] = ~real2[rr, cc]
            ev2 = copy.copy(ev); ev2.desc = ev.desc.copia(); ev2.desc.real = real2
            for e in (ev, ev2):                  # los dos recalculan la cnn en el mismo estado de la busqueda
                e.mapa_cf = None; e.mapa_rasgos = None
            libres = np.argwhere(~real & ~real2)
            rc = libres[rng.choice(len(libres), 40, replace=False)]
            b1, e1 = columnas(ev, rc, m.media, m.std, ESCALA)
            b2, e2 = columnas(ev2, rc, m.media, m.std, ESCALA)
            x_f, x_n = torch.tensor(rc[:8], dtype=torch.float), torch.tensor(rc[8:], dtype=torch.float)
            ei = torch.tensor([list(range(8, 40)), [i % 8 for i in range(32)]], dtype=torch.long)
            with torch.no_grad():
                m.ev = ev; h1, _ = m(x_f, x_n, ei, data.x[v_g], torch.zeros(m.cfg.d_hidden))
                m.ev = ev2; h2, _ = m(x_f, x_n, ei, data.x[v_g], torch.zeros(m.cfg.d_hidden))
            vistas = np.argwhere(visto & ~real)
            real3 = real.copy()
            for rr, cc in vistas[rng.choice(len(vistas), min(20, len(vistas)), replace=False)]:
                real3[rr, cc] = True
            ev3 = copy.copy(ev); ev3.desc = ev.desc.copia(); ev3.desc.real = real3; ev3.mapa_cf = None; ev3.mapa_rasgos = None
            _, e3 = columnas(ev3, rc, m.media, m.std, ESCALA)
            igual_col, igual_sal, cambia = np.array_equal(b1, b2) and np.array_equal(e1, e2), torch.equal(h1, h2), not np.array_equal(e1, e3)
            ok &= igual_col and igual_sal and cambia
            fila += [int(igual_col), int(igual_sal), int(cambia)]
        filas.append(fila)
    write_csv(root / "checks" / f"fuga_{ds}_s{seed}.csv", ["dataset", "seed", "grid"] + [f"{r}_{k}" for r in ("R2", "C") for k in ("columnas_iguales", "salida_igual", "cambia_si_cambia_lo_visto")], filas)
    print(f"[{ds} s{seed}] fuga: {'pasa' if ok else 'NO pasa'} en {len(filas)} mapas", flush=True)
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(ROOT_ESC))
    p.add_argument("--refs2", default=str(EXP_CF2))
    p.add_argument("--refs", default=str(RESULTS))
    p.add_argument("--dataset", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n", type=int, default=70)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    set_threads(1)
    root = Path(a.root)
    (root / "checks").mkdir(parents=True, exist_ok=True)
    unet, ucfg, brazos = brazos_de(a.dataset, a.seed, a.refs, a.refs2, a.device)
    lineas = [f"fuga_{a.dataset}_s{a.seed} {int(check_fuga(root, a.dataset, a.seed, brazos))}",
              f"epoca_cero_{a.dataset}_s{a.seed} {int(check_epoca_cero(root, a.dataset, a.seed, a.n, unet, ucfg, brazos, a.device))}"]
    (root / "checks" / f"estado_{a.dataset}_s{a.seed}.txt").write_text("\n".join(lineas) + "\n")
    print("\n".join(lineas))


if __name__ == "__main__":
    main()
