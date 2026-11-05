# verificaciones previas de exp_mem, por conjunto y semilla
#   1 la epoca 0 de los cuatro brazos reproduce la busqueda de cf_filtro best.pt de la semilla en los 70 mapas de validacion, expansion por expansion,
#     y su media coincide con best_val de la corrida de cf_filtro en exp_cf2
#   2 bloques en cero dan campo R en cero exacto, en cpu y en mps
#   3 tiempo por mapa de cada brazo en la epoca 0, para la estimacion
#   python -m experiments.check_mem --root results/exp_mem --refs2 results/exp_cf2 --dataset mazes --seed 0 [--n 70]
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.common import ESCALA, MAX_EXP, grids_de, load_problem, set_threads, write_csv
from experiments.fotos_mem import REFS2, ROOT_MEM, cf_ckpt_de
from src.cf.busqueda import EvaluadorCF, EvaluadorMem, voraz
from src.cf.memoria import MODELOS_MEM, CampoR, MemConfig, construir_mem
from src.cf.phil_res import PhilResCF, ResCfConfig
from src.cf.red import cargar
from src.search.greedy import greedy_search


def check_epoca_cero(root, refs2, ds, seed, n, device):
    ck = cf_ckpt_de(refs2, ds, seed)
    unet, ucfg, raw = cargar(ck, device)
    best_val = (raw.get("val") or {}).get("exp")
    mems = {m: construir_mem(MemConfig(modelo=m, cf_ckpt=str(ck), sigma=ucfg.sigma, every_cf=ucfg.every_cf, every_fs=ucfg.every_fs), unet).eval() for m in MODELOS_MEM}
    for m in mems.values():
        m.unet.to(device); m.resto_a("cpu")
    phil = PhilResCF(ResCfConfig(fs_every=ucfg.every_fs, coord_scale=ESCALA, d_enc=2, cf_ckpt=str(ck), every_cf=ucfg.every_cf), unet=unet, device=device).eval()
    brazos = ["cf_filtro"] + list(MODELOS_MEM) + ["phil_res_cf"]
    filas, exps, secs, ok = [], {b: [] for b in brazos}, {b: 0.0 for b in brazos}, True
    t_ini = time.time()
    for gp in grids_de(ds, "validation")[:n]:
        data, adj, v_s, v_g, h_star = load_problem(gp)
        if int(h_star[v_s]) < 0:
            continue
        res, fila = {}, [ds, seed, gp.name]
        for b in brazos:
            t0 = time.time()
            if b == "cf_filtro":
                r = voraz(EvaluadorCF(data, v_g, unet, ucfg.every_cf, ucfg.every_fs, ESCALA, device), data, v_s, v_g, adj, max_exp=MAX_EXP)
            elif b == "phil_res_cf":
                r = greedy_search(phil, data, v_s, v_g, adj=adj, max_exp=MAX_EXP, return_path=True)
            else:
                r = voraz(EvaluadorMem(data, v_g, mems[b], gp.name, ucfg.every_cf, ucfg.every_fs, ESCALA, device, "cpu"), data, v_s, v_g, adj, max_exp=MAX_EXP)
            secs[b] += time.time() - t0
            res[b] = r
            exps[b].append(r.n_expanded if r.found else MAX_EXP)
        ref = res["cf_filtro"]
        for b in brazos[1:]:
            r = res[b]
            igual = r.found == ref.found and r.closed == ref.closed and r.path == ref.path
            ok &= igual
            fila += [r.n_expanded, int(igual)]
        filas.append(fila + [ref.n_expanded])
    head = ["dataset", "seed", "grid"] + [f"{b}_{k}" for b in brazos[1:] for k in ("exp", "igual")] + ["cf_filtro_exp"]
    write_csv(root / "checks" / f"epoca_cero_{ds}_s{seed}.csv", head, filas)
    media = {b: float(np.mean(exps[b])) for b in brazos}
    n_mapas = len(filas)
    coincide = best_val is not None and abs(media["cf_filtro"] - best_val) < 0.05
    write_csv(root / "checks" / f"tiempos_epoca0_{ds}_s{seed}.csv", ["dataset", "seed", "brazo", "mapas", "exp_media", "secs_por_mapa", "ms_por_nodo"],
              [[ds, seed, b, n_mapas, round(media[b], 1), round(secs[b] / max(n_mapas, 1), 3), round(1000 * secs[b] / max(sum(exps[b]), 1), 3)] for b in brazos])
    iguales = sum(all(f[4 + 2 * j] for j in range(len(brazos) - 1)) for f in filas)
    print(f"[{ds} s{seed}] epoca 0: {iguales} de {n_mapas} mapas identicos en los 4 brazos, media cf_filtro {media['cf_filtro']:.1f} contra best_val {best_val}, "
          f"segundos por mapa " + " ".join(f"{b} {secs[b] / max(n_mapas, 1):.2f}" for b in brazos) + f" ({time.time() - t_ini:.0f}s)", flush=True)
    return ok and (coincide or n < 70)


def check_constante(root, ds, seed, sigma):
    ok = True
    devs = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    for dev in devs:
        c = CampoR(sigma, 8).to(dev)
        with torch.no_grad():
            out = c(torch.zeros(3, 26, 26, device=dev)).cpu()
        ok &= torch.equal(out, torch.zeros_like(out))
    print(f"[{ds} s{seed}] campo R de bloques en cero: {'exacto' if ok else 'NO exacto'} en {devs}", flush=True)
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(ROOT_MEM))
    p.add_argument("--refs2", default=str(REFS2))
    p.add_argument("--dataset", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n", type=int, default=70)
    p.add_argument("--device", default="mps")
    a = p.parse_args()
    set_threads(1)
    torch.manual_seed(0)
    root, refs2 = Path(a.root), Path(a.refs2)
    (root / "checks").mkdir(parents=True, exist_ok=True)
    sigma = cargar(cf_ckpt_de(refs2, a.dataset, a.seed), "cpu")[1].sigma
    lineas = [f"constante_{a.dataset}_s{a.seed} {int(check_constante(root, a.dataset, a.seed, sigma))}",
              f"epoca_cero_{a.dataset}_s{a.seed} {int(check_epoca_cero(root, refs2, a.dataset, a.seed, a.n, a.device))}"]
    (root / "checks" / f"estado_{a.dataset}_s{a.seed}.txt").write_text("\n".join(lineas) + "\n")
    print("\n".join(lineas))


if __name__ == "__main__":
    main()
