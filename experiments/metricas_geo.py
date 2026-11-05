# metricas de la cnn de geometria en las fotos de validacion, sobre celdas no vistas
#   python -m experiments.metricas_geo --dataset mazes --seed 0 --root results/exp_geo
# previo constante, epoca 0 y mejor epoca, en todo lo no visto, cerca del camino optimo y por distancia a la frontera, y por fuente de fotos
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.common import RESULTS_GEO, set_threads, write_csv
from src.geo.fotos import cargar_opcional, canales_geo
from src.geo.metricas import NOMBRES_FRONTERA, Acumulador, estratos
from src.geo.red import cargar_geo

ESTRATOS = ["todo", "corredor"] + NOMBRES_FRONTERA + ["todo_hfs", "todo_cf"]


@torch.no_grad()
def probs(model, visto, real, device):
    x = torch.from_numpy(canales_geo(visto, real))[None].to(device)
    return torch.sigmoid(model(x))[0].cpu().numpy()[:real.shape[0], :real.shape[1]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--root", default=str(RESULTS_GEO))
    p.add_argument("--fotos-root", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-val", type=int, default=0, help="humo")
    a = p.parse_args()
    set_threads(1)
    root = Path(a.root)
    froot = Path(a.fotos_root or root / "fotos") / a.dataset
    run = root / "runs" / a.dataset / f"geo_s{a.seed}"
    meta = json.loads((run / "config.json").read_text())
    p0 = float(meta["p0"])
    modelos = {"epoca_0": cargar_geo(run / "epoca_0.pt", a.device)[0], "mejor": cargar_geo(run / "best.pt", a.device)[0]}
    partes = [("hfs", cargar_opcional(froot / "hfs_val")), ("cf", cargar_opcional(froot / f"cf_s{a.seed}_val"))]
    acc = {(m, e): Acumulador() for m in ["previo"] + list(modelos) for e in ESTRATOS}
    t0, n_fotos = time.time(), 0
    for fuente, fotos in partes:
        if fotos is None:
            continue
        n = min(len(fotos), a.max_val) if a.max_val else len(fotos)
        for i in range(n):
            mi = int(fotos.mapa_idx[i])
            real, visto = fotos.real[mi], fotos.visto_de(i)
            masks = estratos(visto, real, fotos.hstar[mi], fotos.start[mi])
            masks[f"todo_{fuente}"] = masks["todo"]
            pr = {"previo": np.full(real.shape, p0, dtype=np.float32)}
            for nombre, m in modelos.items():
                pr[nombre] = probs(m, visto, real, a.device)
            for e, mk in masks.items():
                for nombre, pp in pr.items():
                    acc[(nombre, e)].agregar(pp[mk], real[mk])
            n_fotos += 1
    filas, cal = [], []
    res = {k: v.resultados() for k, v in acc.items()}
    for e in ESTRATOS:
        bce_prev = res[("previo", e)][0]["bce"]
        for m in ["previo"] + list(modelos):
            r, tabla = res[(m, e)]
            filas.append([a.dataset, a.seed, m, e, r["n"], round(r["frac_obst"], 5), round(r["bce"], 5), round(r["auc"], 4) if r["auc"] == r["auc"] else "",
                          round(r["ece"], 4), round(bce_prev, 5), int(r["bce"] < bce_prev - 1e-9) if m != "previo" else ""])
            if e in ("todo", "corredor"):
                for t in tabla:
                    cal.append([a.dataset, a.seed, m, e, t["tramo"], t["p_min"], t["p_max"], t["n"], round(t["p_media"], 4), round(t["frac_obst"], 4)])
    out = root / "metricas" / a.dataset
    write_csv(out / f"geo_s{a.seed}.csv", ["dataset", "seed", "modelo", "estrato", "n_celdas", "frac_obst", "bce", "auc", "ece", "bce_previo", "gana_al_previo"], filas)
    write_csv(out / f"calibracion_s{a.seed}.csv", ["dataset", "seed", "modelo", "estrato", "tramo", "p_min", "p_max", "n", "p_media", "frac_obst"], cal)
    print(f"[{a.dataset}/geo_s{a.seed}] metricas sobre {n_fotos} fotos en {time.time() - t0:.0f}s -> {out}", flush=True)
    for fila in filas:
        if fila[3] in ("todo", "corredor"):
            print(f"  {fila[2]:8s} {fila[3]:9s} n {fila[4]:>9d} obst {fila[5]:.3f} bce {fila[6]:.4f} auc {fila[7]} ece {fila[8]:.4f}")


if __name__ == "__main__":
    main()
