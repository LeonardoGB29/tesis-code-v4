# evalua en los 100 mapas de prueba, una fila por mapa
#   --ckpt <cf best.pt> | --phil <phil best.pt> | --mem <mem best.pt> | --ref hfs|hstar ; con --ruido eps --ruido-seed k para el piso de ruido
#   exp_mem: --mem evalua mem_no, mem_gru o mem_mamba con EvaluadorMem; --phil con un phil_res_cf carga la u-net en --device
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from experiments.common import ESCALA, MAX_EXP, grids_de, load_phil, load_problem, set_threads, write_csv
from src.cf.busqueda import Evaluador, EvaluadorCF, EvaluadorGNN, EvaluadorMem, PhilConRuido, voraz
from src.cf.memoria import cargar_mem
from src.cf.red import cargar
from src.search.baselines import _greedy_best_first
from src.search.greedy import greedy_search

HEADER = ["dataset", "run", "grid", "optimo", "exp", "encontrado", "camino", "secs", "secs_red", "n_red", "epoca"]


def evaluar(a):
    run, model, cfg, epoca, sin_elegibles = a.name, None, None, "", False
    ckpt = a.ckpt or a.mem
    if ckpt:
        model, cfg, raw = cargar(a.ckpt, a.device) if a.ckpt else cargar_mem(a.mem, a.device, "cpu")
        epoca = raw.get("epoca", "")
        run = run or Path(ckpt).parent.name
        sin_elegibles = bool(raw.get("extra", {}).get("sin_elegibles"))     # camino5 sin epoca elegible: queda la voraz con h_fs
        if sin_elegibles:
            print(f"[{a.dataset}/{run}] sin epoca elegible, se evalua la voraz con h_fs", flush=True)
        cfgj = Path(ckpt).parent / "config.json"
        ref = (json.loads(cfgj.read_text()).get("ref_val") or {}) if cfgj.exists() else {}
        val = raw.get("val") or {}
        if not sin_elegibles and ref.get("exp") is not None and val.get("exp") is not None and val["exp"] > ref["exp"] + 1e-9:
            sin_elegibles = True                                               # camino5 incluye a h_fs: una epoca peor que h_fs en validacion no se elige
            print(f"[{a.dataset}/{run}] la epoca elegida expande mas que h_fs en validacion ({val['exp']:.1f} > {ref['exp']:.1f}), se evalua la voraz con h_fs", flush=True)
    elif a.phil:
        model, cfg, raw = load_phil(a.phil, a.device)
        epoca = raw.get("iteration", "")
        run = run or Path(a.phil).parent.name
        if "cf_ckpt" in raw["config"]:                                           # exp_mem: camino5 incluye a cf_filtro y a h_fs
            cfgj = Path(a.phil).parent / "config.json"
            ref = (json.loads(cfgj.read_text()).get("ref_val") or {}) if cfgj.exists() else {}
            val = raw.get("val") or {}
            if raw.get("extra", {}).get("sin_elegibles") or (ref.get("exp") is not None and val.get("exp") is not None and val["exp"] > ref["exp"] + 1e-9):
                sin_elegibles = True
                print(f"[{a.dataset}/{run}] sin iteracion elegible o peor que h_fs en validacion, se evalua la voraz con h_fs", flush=True)
    else:
        run = run or f"voraz_{a.ref}"
    if a.ruido:
        run = f"{run}_ruido{a.ruido_seed}"
    rng = np.random.default_rng(a.ruido_seed)
    filas = []
    for k, gp in enumerate(grids_de(a.dataset, "test")[:a.n_grids] if a.n_grids else grids_de(a.dataset, "test")):
        data, adj, v_s, v_g, h_star = load_problem(gp)
        opt = int(h_star[v_s])
        if opt < 0:
            continue
        random.seed(0)
        t0 = time.time()
        secs_red, n_red = 0.0, 0
        if a.phil and sin_elegibles:
            res = voraz(Evaluador(data, v_g, a.every_fs, ESCALA), data, v_s, v_g, adj, max_exp=a.max_exp)
            n_exp, found, path = res.n_expanded, res.found, res.path
        elif a.phil:
            m = PhilConRuido(model, a.ruido, rng) if a.ruido else model
            res = greedy_search(m, data, v_s, v_g, adj=adj, max_exp=a.max_exp, return_path=True)
            n_exp, found, path = res.n_expanded, res.found, res.path
            secs_red, n_red = float(getattr(model, "secs_red", 0.0)), int(getattr(model, "n_red", 0))
        elif a.ref == "hstar":
            hs = h_star.numpy().astype(np.float64)
            hs[hs < 0] = np.inf
            if a.ruido:
                hs = hs + rng.uniform(-a.ruido, a.ruido, size=hs.shape)
            closed, found, path = _greedy_best_first(hs, v_s, v_g, adj, a.max_exp)
            n_exp = len(closed)
        else:
            if ckpt and sin_elegibles:
                ev = Evaluador(data, v_g, cfg.every_fs, ESCALA)
            elif a.mem:
                ev = EvaluadorMem(data, v_g, model, gp.name, cfg.every_cf, cfg.every_fs, ESCALA, a.device, "cpu")
            elif a.ckpt and cfg.tipo == "gnn":
                ev = EvaluadorGNN(data, v_g, model, adj, cfg.every_cf, cfg.every_fs, ESCALA, a.device)
            elif a.ckpt:
                ev = EvaluadorCF(data, v_g, model, cfg.every_cf, cfg.every_fs, ESCALA, a.device, cf_fijo=a.cf_fijo)
            else:
                ev = Evaluador(data, v_g, a.every_fs, ESCALA)
            res = voraz(ev, data, v_s, v_g, adj, max_exp=a.max_exp, ruido=a.ruido, rng=rng)
            n_exp, found, path, secs_red, n_red = res.n_expanded, res.found, res.path, res.secs_red, res.n_red
        secs = time.time() - t0
        filas.append([a.dataset, run, gp.name, opt, n_exp if found else a.max_exp, int(found), (len(path) - 1) if path else "",
                      round(secs, 4), round(secs_red, 4), n_red, epoca])
        if a.verbose:
            print(f"[{k + 1:3d}] {gp.name} exp={filas[-1][4]} opt={opt} ({secs:.2f}s)", flush=True)
    write_csv(a.out, HEADER, filas)
    e = np.array([r[4] for r in filas], dtype=float)
    secs = sum(r[7] for r in filas)
    print(f"[{a.dataset}/{run}] n={len(filas)} exp={e.mean():.1f} ms/nodo={1000 * secs / max(e.sum(), 1):.3f} -> {a.out}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--phil", default=None)
    p.add_argument("--mem", default=None, help="exp_mem: best.pt de mem_no, mem_gru o mem_mamba")
    p.add_argument("--ref", default=None, choices=["hfs", "hstar"])
    p.add_argument("--name", default=None)
    p.add_argument("--ruido", type=float, default=0.0, help="ruido uniforme en saltos")
    p.add_argument("--ruido-seed", type=int, default=0)
    p.add_argument("--cf-fijo", type=float, default=None)
    p.add_argument("--every-fs", type=int, default=16)
    p.add_argument("--n-grids", type=int, default=0)
    p.add_argument("--max-exp", type=int, default=MAX_EXP)
    p.add_argument("--device", default="cpu")
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args()
    assert sum(x is not None for x in (a.ckpt, a.phil, a.mem, a.ref)) == 1, "uno de --ckpt, --phil, --mem o --ref"
    set_threads(1)
    evaluar(a)


if __name__ == "__main__":
    main()
