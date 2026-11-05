# entrena la cnn de geometria con bce sobre las celdas no vistas, validacion por bce en cada epoca
#   python -m experiments.train_geo --dataset mazes --seed 0 --epochs 10 --device mps
# fotos de entrenamiento, las base de h_fs de exp_cf mas las de cf_filtro de la misma semilla. validacion, las mismas dos busquedas en los 70 mapas
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from experiments.common import RESULTS, RESULTS_GEO, set_threads
from src.geo.fotos import cargar_opcional, lote_geo, previo, unir_opcional
from src.geo.metricas import Acumulador
from src.geo.red import GeoConfig, construir_geo, guardar_geo
from src.phil.training import CSVLogger

COLS = ["epoca", "loss", "n_fotos", "secs_train", "val_bce", "val_auc", "val_ece", "val_n", "derrumbe", "secs_val", "best_epoca", "best_val"]


def fmt(x, nd=5):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


def recortar(fotos, n):
    from src.cf.fotos import Fotos
    if not n or fotos is None or len(fotos) <= n:
        return fotos
    return Fotos(fotos.nombres, fotos.real, fotos.hstar, fotos.goal, fotos.start, fotos.mapa_idx[:n], fotos.visto[:n], fotos.hfs[:n], fotos.escala)


@torch.no_grad()
def validar(model, fotos, device, batch=16):
    model.eval()
    if device == "mps":
        torch.mps.empty_cache()
    acc = Acumulador()
    for k in range(0, len(fotos), batch):
        x, y, m = lote_geo(fotos, np.arange(k, min(k + batch, len(fotos))), None, device)
        p = torch.sigmoid(model(x)).cpu().numpy()
        y, m = y.cpu().numpy().astype(bool), m.cpu().numpy()
        acc.agregar(p[m], y[m])
    model.train()
    r, _ = acc.resultados()
    return r


def epoca_entrenamiento(model, fotos, opt, batch, rng, device):
    model.train()
    orden = rng.permutation(len(fotos))
    tot, n = 0.0, 0
    for k in range(0, len(orden), batch):
        idx = orden[k:k + batch]
        x, y, m = lote_geo(fotos, idx, rng, device)
        logits = model(x)
        mf = m.float()
        loss = (F.binary_cross_entropy_with_logits(logits, y, reduction="none") * mf).sum() / mf.sum().clamp(min=1)
        opt.zero_grad()
        loss.backward()
        opt.step()
        tot += float(loss.item()) * len(idx)
        n += len(idx)
    return tot / max(n, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--ancho", type=int, default=16)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--fotos-root", default=str(RESULTS_GEO / "fotos"))
    p.add_argument("--fotos-base", default=None, help="fotos base de la voraz con h_fs, por defecto las de exp_cf")
    p.add_argument("--out-root", default=str(RESULTS_GEO / "runs"))
    p.add_argument("--tag", default=None)
    p.add_argument("--max-fotos", type=int, default=0, help="humo, recorta las fotos de entrenamiento")
    p.add_argument("--max-val", type=int, default=0, help="humo, recorta las fotos de validacion")
    a = p.parse_args()
    set_threads(a.threads)
    tag = a.tag or f"geo_s{a.seed}"
    out = Path(a.out_root) / a.dataset / tag
    out.mkdir(parents=True, exist_ok=True)
    if (out / "listo").exists():
        print(f"[{a.dataset}/{tag}] ya esta listo", flush=True)
        return
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    froot = Path(a.fotos_root) / a.dataset
    base = cargar_opcional(a.fotos_base or RESULTS / "fotos" / a.dataset / "hfs")
    cf_train = cargar_opcional(froot / f"cf_s{a.seed}_train")
    fotos = unir_opcional([recortar(base, a.max_fotos), recortar(cf_train, a.max_fotos)])
    fotos_val = unir_opcional([recortar(cargar_opcional(froot / "hfs_val"), a.max_val), recortar(cargar_opcional(froot / f"cf_s{a.seed}_val"), a.max_val)])
    cfgj = out / "config.json"
    if cfgj.exists() and (out / "last.pt").exists():
        meta = json.loads(cfgj.read_text())
        p0 = meta["p0"]
    else:
        t0 = time.time()
        p0 = previo(fotos)
        print(f"[{a.dataset}/{tag}] previo p0 {p0:.4f} sobre {len(fotos)} fotos ({time.time() - t0:.0f}s)", flush=True)
    cfg = GeoConfig(ancho=a.ancho, p0=p0)
    model = construir_geo(cfg).to(a.device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    meta = {"dataset": a.dataset, "seed": a.seed, "modelo": "geo", "tag": tag, "args": vars(a), "config": cfg.to_dict(), "p0": p0,
            "n_params": sum(q.numel() for q in model.parameters()), "n_fotos": len(fotos), "n_fotos_base": len(base),
            "n_fotos_cf": len(cf_train) if cf_train is not None else 0, "n_fotos_val": len(fotos_val), "epocas": a.epochs}
    cfgj.write_text(json.dumps(meta, indent=1))
    log = CSVLogger(out / "train.csv", COLS)
    estado = {"epoca": 0, "best_epoca": None, "best_val": None, "val_previo": None, "derrumbes": 0, "corte": False}
    ultimo = out / "last.pt"
    if ultimo.exists():
        raw = torch.load(ultimo, map_location="cpu", weights_only=False)
        model.load_state_dict(raw["model_state"]); model.to(a.device)
        opt.load_state_dict(raw["extra"]["opt"]); estado = raw["extra"]["estado"]
        rng = np.random.default_rng([a.seed, estado["epoca"]])
        print(f"[{a.dataset}/{tag}] sigue desde la epoca {estado['epoca']}", flush=True)

    def validar_y_registrar(epoca, loss, secs_train):
        t0 = time.time()
        v = validar(model, fotos_val, a.device)
        if epoca == 0:
            estado["val_previo"] = v["bce"]
        derrumbe = estado["val_previo"] is not None and v["bce"] > estado["val_previo"] + 1e-9
        estado["derrumbes"] = estado["derrumbes"] + 1 if derrumbe else 0
        if estado["best_val"] is None or v["bce"] < estado["best_val"]:
            estado["best_epoca"], estado["best_val"] = epoca, v["bce"]
            guardar_geo(out / "best.pt", model, epoca, v)
        guardar_geo(out / f"epoca_{epoca}.pt", model, epoca, v)
        log.log(epoca=epoca, loss=fmt(loss), n_fotos=len(fotos), secs_train=round(secs_train, 1), val_bce=fmt(v["bce"]), val_auc=fmt(v["auc"], 4),
                val_ece=fmt(v["ece"], 4), val_n=v["n"], derrumbe=int(derrumbe), secs_val=round(time.time() - t0, 1),
                best_epoca=estado["best_epoca"], best_val=fmt(estado["best_val"]))
        print(f"[{a.dataset}/{tag}] epoca {epoca} loss={fmt(loss)} val_bce={v['bce']:.5f} auc={v['auc']:.4f} ece={v['ece']:.4f} "
              f"derrumbe={int(derrumbe)} best={estado['best_epoca']} ({time.time() - t0:.0f}s val)", flush=True)
        if estado["derrumbes"] >= 2:
            estado["corte"] = True
            print(f"[{a.dataset}/{tag}] corte por segundo derrumbe seguido en la epoca {epoca}", flush=True)

    if estado["epoca"] == 0 and not ultimo.exists():
        validar_y_registrar(0, float("nan"), 0.0)
        guardar_geo(ultimo, model, 0, None, {"opt": opt.state_dict(), "estado": estado})
    while estado["epoca"] < a.epochs and not estado["corte"]:
        epoca = estado["epoca"] + 1
        t0 = time.time()
        loss = epoca_entrenamiento(model, fotos, opt, a.batch, rng, a.device)
        secs_train = time.time() - t0
        if a.device == "mps":
            torch.mps.empty_cache()
        estado["epoca"] = epoca
        validar_y_registrar(epoca, loss, secs_train)
        guardar_geo(ultimo, model, epoca, None, {"opt": opt.state_dict(), "estado": estado})
    meta.update({"best_epoca": estado["best_epoca"], "best_val": estado["best_val"], "val_previo": estado["val_previo"], "corte": estado["corte"],
                 "epoca_final": estado["epoca"], "gana_al_previo": bool(estado["best_epoca"] not in (None, 0))})
    cfgj.write_text(json.dumps(meta, indent=1))
    log.close()
    (out / "listo").write_text(f"epoca {estado['epoca']} best {estado['best_epoca']}\n")
    print(f"[{a.dataset}/{tag}] listo, best epoca {estado['best_epoca']} val_bce {fmt(estado['best_val'])} previo {fmt(estado['val_previo'])}", flush=True)


if __name__ == "__main__":
    main()
