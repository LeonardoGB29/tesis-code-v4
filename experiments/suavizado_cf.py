# fuera del protocolo: cf_global (o cf_global_dagger) con el mapa de cf suavizado (gauss) antes de dividir, sigma elegido por conjunto en validacion
# reelige la epoca con camino5 entre los checkpoints guardados de la corrida de origen (la epoca 0 es h_fs y cuenta como candidata) y evalua en prueba
#   python -m experiments.suavizado_cf --dataset gaps_and_forest --seed 0 [--origen-modelo cf_global_dagger]
import argparse
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from experiments.common import ESCALA, MAX_EXP, RESULTS, grids_de, leer_csv, load_problem, set_threads, write_csv
from experiments.eval_cf import HEADER
from experiments.train_cf import COLS, SIGMA, elegible, referencia_val
from src.cf.busqueda import Evaluador, EvaluadorCF, voraz
from src.cf.red import cargar
from src.phil.training import CSVLogger


class EvaluadorSuave(EvaluadorCF):
    def __init__(self, *a, sigma=0.0, **k):
        super().__init__(*a, **k)
        self.sigma = sigma

    def calcular(self):
        cf = super().calcular()
        if self.sigma:
            cf = gaussian_filter(cf, self.sigma, mode="nearest")
        return np.clip(cf, 0.001, 1.0)


def buscar(model, cfg, sigma, data, v_s, v_g, adj, device, max_exp):
    ev = EvaluadorSuave(data, v_g, model, sigma=sigma, every_cf=cfg.every_cf, every_fs=cfg.every_fs, escala=ESCALA, device=device) \
        if model is not None else Evaluador(data, v_g, 16, ESCALA)
    return voraz(ev, data, v_s, v_g, adj, max_exp=max_exp)


def validar(model, cfg, sigma, dataset, device, n_val=0):
    exps, caminos, secs_red = [], [], 0.0
    grids = grids_de(dataset, "validation")
    for gp in grids[:n_val] if n_val else grids:
        data, adj, v_s, v_g, h_star = load_problem(gp)
        opt = int(h_star[v_s])
        if opt < 0:
            continue
        res = buscar(model, cfg, sigma, data, v_s, v_g, adj, device, MAX_EXP)
        exps.append(res.n_expanded if res.found else MAX_EXP)
        secs_red += res.secs_red
        if res.found:
            caminos.append((len(res.path) - 1) / max(opt, 1))
    return {"exp": float(np.mean(exps)), "camino": float(np.mean(caminos)) if caminos else float("nan"), "encontrados": len(caminos), "secs_red": secs_red}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sigma", type=float, default=None)
    p.add_argument("--origen-modelo", default="cf_global", help="cf_global o cf_global_dagger; la salida es <origen>_suave")
    p.add_argument("--origen", default=None, help="carpeta de la corrida de origen")
    p.add_argument("--out-root", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-val", type=int, default=0, help="humo")
    p.add_argument("--n-test", type=int, default=0, help="humo")
    a = p.parse_args()
    set_threads(1)
    sigma = a.sigma if a.sigma is not None else SIGMA[a.dataset]
    modelo = f"{a.origen_modelo}_suave"
    origen = Path(a.origen or RESULTS / a.origen_modelo / "runs" / a.dataset / f"{a.origen_modelo}_s{a.seed}")
    root = Path(a.out_root or RESULTS / modelo)
    tag = f"{modelo}_s{a.seed}"
    out = root / "runs" / a.dataset / tag
    out.mkdir(parents=True, exist_ok=True)
    ev_csv = root / "eval" / a.dataset / f"{tag}.csv"
    if ev_csv.exists():
        print(f"[{a.dataset}/{tag}] ya esta", flush=True)
        return
    ref = referencia_val(a.dataset)
    if a.n_val:
        import experiments.common as cm
        cm.N_VAL = a.n_val
        ref = referencia_val(a.dataset)
    cks = sorted([c for c in origen.glob("epoca_*.pt") if c.stem != "epoca_0"], key=lambda c: int(c.stem.split("_")[1]))
    hechas = {int(r["epoca"]): r for r in leer_csv(out / "train.csv")} if (out / "train.csv").exists() else {}
    log = CSVLogger(out / "train.csv", COLS) if not hechas else None
    filas = []
    if 0 not in hechas:
        t0 = time.time()
        v = validar(None, None, sigma, a.dataset, a.device, a.n_val)
        filas.append(dict(epoca=0, ronda=0, loss="", n_fotos=0, secs_train=0, val_exp=round(v["exp"], 1), val_camino=round(v["camino"], 4),
                          val_encontrados=v["encontrados"], elegible=int(elegible(v, ref)), derrumbe=0, secs_val=round(time.time() - t0, 1), secs_red_val=0))
    for ck in cks:
        ep = int(ck.stem.split("_")[1])
        if ep in hechas:
            continue
        model, cfg, raw = cargar(ck, a.device)
        t0 = time.time()
        v = validar(model, cfg, sigma, a.dataset, a.device, a.n_val)
        ok = elegible(v, ref)
        der = v["encontrados"] < ref["encontrados"] or v["exp"] > 1.5 * ref["exp"]
        filas.append(dict(epoca=ep, ronda=1, loss=raw.get("extra", {}).get("loss", ""), n_fotos="", secs_train="", val_exp=round(v["exp"], 1),
                          val_camino=round(v["camino"], 4), val_encontrados=v["encontrados"], elegible=int(ok), derrumbe=int(der),
                          secs_val=round(time.time() - t0, 1), secs_red_val=round(v["secs_red"], 1)))
        print(f"[{a.dataset}/{tag}] epoca {ep} sigma {sigma} val_exp={v['exp']:.1f} camino={v['camino']:.4f} enc={v['encontrados']} elegible={int(ok)} ({time.time() - t0:.0f}s)", flush=True)
    todas = list(hechas.values()) + filas
    cand = [r for r in todas if int(r["elegible"])]                   # la epoca 0 (h_fs) compite, como en train_cf
    best = min(cand, key=lambda r: (float(r["val_exp"]), int(r["epoca"]))) if cand else None
    best_ep = int(best["epoca"]) if best else 0
    best_val = float(best["val_exp"]) if best else float(ref["exp"])
    for r in todas:
        r["best_epoca"], r["best_val"] = best_ep, best_val
    if log is not None:
        for r in filas:
            log.log(**{k: r.get(k, "") for k in COLS})
        log.close()
    else:
        write_csv(out / "train.csv", COLS, [[r.get(k, "") for k in COLS] for r in todas])
    if best_ep:
        shutil.copy(origen / f"epoca_{best_ep}.pt", out / "best.pt")
    meta = {"dataset": a.dataset, "seed": a.seed, "modelo": modelo, "tag": tag, "sigma": sigma, "origen": str(origen), "ref_val": ref,
            "best_epoca": best_ep, "best_val": best_val, "epoca_final": int(cks[-1].stem.split("_")[1]) if cks else 0, "corte": 0,
            "nota": f"fuera del protocolo: cf de {a.origen_modelo} suavizado con gauss antes de dividir, epoca reelegida con camino5 (la epoca 0 es h_fs)"}
    (out / "config.json").write_text(json.dumps(meta, indent=1))
    (out / "listo").write_text(f"best {best_ep} sigma {sigma}\n")
    print(f"[{a.dataset}/{tag}] epoca elegida {best_ep} val {best_val:.1f} (h_fs {ref['exp']:.1f})", flush=True)

    model, cfg = (cargar(out / "best.pt", a.device)[:2]) if best_ep else (None, None)
    filas = []
    grids = grids_de(a.dataset, "test")
    for gp in grids[:a.n_test] if a.n_test else grids:
        data, adj, v_s, v_g, h_star = load_problem(gp)
        opt = int(h_star[v_s])
        if opt < 0:
            continue
        random.seed(0)
        t0 = time.time()
        res = buscar(model, cfg, sigma, data, v_s, v_g, adj, a.device, MAX_EXP)
        secs = time.time() - t0
        filas.append([a.dataset, tag, gp.name, opt, res.n_expanded if res.found else MAX_EXP, int(res.found), (len(res.path) - 1) if res.path else "",
                      round(secs, 4), round(res.secs_red, 4), res.n_red, best_ep])
    write_csv(ev_csv, HEADER, filas)
    e = np.array([r[4] for r in filas], dtype=float)
    print(f"[{a.dataset}/{tag}] n={len(filas)} exp={e.mean():.1f} ms/nodo={1000 * sum(r[7] for r in filas) / max(e.sum(), 1):.3f} -> {ev_csv}", flush=True)


if __name__ == "__main__":
    main()
