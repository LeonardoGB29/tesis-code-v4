# exp_mem: entrena mem_no, mem_gru o mem_mamba sobre cf_filtro congelado con las fotos precalculadas, por secuencias de mapas,
# validacion con busqueda en los 70 mapas cada val_every epocas, camino5 respecto de la voraz con h_fs, la epoca 0 es cf_filtro exacto
#   python -m experiments.train_mem --dataset mazes --seed 0 --modelo mem_gru [--root results/exp_mem --refs2 results/exp_cf2 --device mps]
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from experiments.common import ESCALA, MAX_EXP, grids_de, load_problem, set_threads
from experiments.fotos_mem import REFS2, ROOT_MEM, cf_ckpt_de
from experiments.train_cf import COLS, elegible, f, referencia_val
from src.cf.busqueda import EvaluadorMem, voraz
from src.cf.datos_mem import FotosMem, Precalculo, armar_lote, lotes_por_fotos
from src.cf.memoria import MODELOS_MEM, MemConfig, construir_mem, guardar_mem
from src.cf.red import cargar
from src.phil.training import CSVLogger


def validar_mem(model, cfg, dataset, dev_unet, dev_mem, dev_train, max_exp=MAX_EXP):
    # busqueda en los mapas de validacion con la u-net en dev_unet y el resto en dev_mem; despues el resto vuelve a dev_train
    model.eval()
    model.resto_a(dev_mem)
    if "mps" in (dev_unet, dev_train):
        torch.mps.empty_cache()
    exps, caminos, secs_red, secs = [], [], 0.0, 0.0
    for gp in grids_de(dataset, "validation"):
        data, adj, v_s, v_g, h_star = load_problem(gp)
        opt = int(h_star[v_s])
        if opt < 0:
            continue
        t0 = time.time()
        ev = EvaluadorMem(data, v_g, model, gp.name, cfg.every_cf, cfg.every_fs, ESCALA, dev_unet, dev_mem)
        res = voraz(ev, data, v_s, v_g, adj, max_exp=max_exp)
        secs += time.time() - t0
        exps.append(res.n_expanded if res.found else max_exp)
        secs_red += res.secs_red
        if res.found:
            caminos.append((len(res.path) - 1) / max(opt, 1))
    model.resto_a(dev_train)
    model.train()
    return {"exp": float(np.mean(exps)), "camino": float(np.mean(caminos)) if caminos else float("nan"), "encontrados": len(caminos),
            "secs_red": secs_red, "secs": secs, "n": len(exps)}


def epoca_entrenamiento(model, fotos, pre, cfg, opt, rng, device, tope):
    model.train()
    lotes = lotes_por_fotos(fotos.secuencias(), rng, tope)
    tot, n = 0.0, 0
    for seqs in lotes:
        lote = armar_lote(fotos, pre, seqs, cfg, device)
        R = model.campo_lote(lote)
        cf = model.cf_de(lote["cf0"], R)
        loss = (((cf - lote["y"]) ** 2) * lote["m"]).sum() / lote["m"].sum().clamp(min=1)
        opt.zero_grad()
        loss.backward()
        opt.step()
        P = lote["goal_foto"].size(0)
        tot += float(loss.item()) * P
        n += P
    return tot / max(n, 1), len(lotes)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--modelo", required=True, choices=list(MODELOS_MEM))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--val-every", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--tope-fotos", type=int, default=512, help="fotos por lote como maximo, secuencias enteras")
    p.add_argument("--root", default=str(ROOT_MEM))
    p.add_argument("--refs2", default=str(REFS2))
    p.add_argument("--device", default="cpu", help="entrenamiento de la gnn, la memoria y la cabeza")
    p.add_argument("--dev-unet", default="mps", help="u-net congelada durante la busqueda, el mismo dispositivo que la validacion de cf_filtro")
    p.add_argument("--dev-mem", default="cpu", help="gnn, memoria y cabeza durante la busqueda")
    p.add_argument("--tag", default=None)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--n-val", type=int, default=0, help="humo: recorta la validacion")
    p.add_argument("--max-secuencias", type=int, default=0, help="humo: recorta las secuencias de entrenamiento")
    a = p.parse_args()
    set_threads(a.threads)
    if a.n_val:
        import experiments.common as cm
        cm.N_VAL = a.n_val
    tag = a.tag or f"{a.modelo}_s{a.seed}"
    out = Path(a.root) / a.modelo / "runs" / a.dataset / tag
    out.mkdir(parents=True, exist_ok=True)
    if (out / "listo").exists():
        print(f"[{a.dataset}/{tag}] ya esta listo", flush=True)
        return
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    datos = Path(a.root) / "datos" / a.dataset / f"s{a.seed}"
    assert (datos / "pre" / "listo").exists(), f"faltan los datos en {datos}"
    ck = cf_ckpt_de(a.refs2, a.dataset, a.seed)
    unet, ucfg, raw_u = cargar(ck, "cpu")
    cfg = MemConfig(modelo=a.modelo, cf_ckpt=str(ck), sigma=ucfg.sigma, every_cf=ucfg.every_cf, every_fs=ucfg.every_fs)
    model = construir_mem(cfg, unet)
    model.unet.to(a.dev_unet)
    model.resto_a(a.device)
    opt = torch.optim.Adam(model.entrenables(), lr=a.lr)
    ref = referencia_val(a.dataset)
    fotos = FotosMem.cargar(datos / "fotos")
    pre = Precalculo.crear if False else Precalculo(datos / "pre")
    if a.max_secuencias:
        secs = fotos.secuencias()[:a.max_secuencias]
        fotos.secuencias = lambda: secs
    meta = {"dataset": a.dataset, "seed": a.seed, "modelo": a.modelo, "tag": tag, "args": vars(a), "config": cfg.to_dict(), "cf_ckpt": str(ck),
            "epoca_cf": raw_u.get("epoca"), "val_cf": raw_u.get("val"), "ref_val": ref, "n_params": sum(p.numel() for p in model.parameters()),
            "n_entrenables": sum(p.numel() for p in model.entrenables()), "n_fotos": len(fotos), "n_secuencias": len(fotos.secuencias()), "epocas": a.epochs}
    (out / "config.json").write_text(json.dumps(meta, indent=1))
    log = CSVLogger(out / "train.csv", COLS)
    estado = {"epoca": 0, "best_epoca": None, "best_val": None, "derrumbes": 0, "corte": False}
    ultimo = out / "last.pt"
    if ultimo.exists():
        raw = torch.load(ultimo, map_location="cpu", weights_only=False)
        model.load_state_dict(raw["model_state"]); model.unet.to(a.dev_unet); model.resto_a(a.device)
        opt.load_state_dict(raw["extra"]["opt"]); estado = raw["extra"]["estado"]
        rng = np.random.default_rng([a.seed, estado["epoca"]])
        print(f"[{a.dataset}/{tag}] sigue desde la epoca {estado['epoca']}", flush=True)

    def validar_y_registrar(epoca, loss, n_fotos, secs_train):
        t0 = time.time()
        v = validar_mem(model, cfg, a.dataset, a.dev_unet, a.dev_mem, a.device)
        ok = elegible(v, ref)
        derrumbe = v["encontrados"] < ref["encontrados"] or v["exp"] > 1.5 * ref["exp"]
        estado["derrumbes"] = estado["derrumbes"] + 1 if derrumbe else 0
        if ok and (estado["best_val"] is None or v["exp"] < estado["best_val"]):
            estado["best_epoca"], estado["best_val"] = epoca, v["exp"]
            guardar_mem(out / "best.pt", model, epoca, v, {"ronda": 0})
        guardar_mem(out / f"epoca_{epoca}.pt", model, epoca, v, {"ronda": 0})
        log.log(epoca=epoca, ronda=0, loss=f(loss), n_fotos=n_fotos, secs_train=round(secs_train, 1), val_exp=round(v["exp"], 1),
                val_camino=f(v["camino"]), val_encontrados=v["encontrados"], elegible=int(ok), derrumbe=int(derrumbe),
                secs_val=round(time.time() - t0, 1), secs_red_val=round(v["secs_red"], 1), best_epoca=estado["best_epoca"], best_val=f(estado["best_val"]))
        print(f"[{a.dataset}/{tag}] epoca {epoca} loss={f(loss)} val_exp={v['exp']:.1f} camino={f(v['camino'])} enc={v['encontrados']} "
              f"elegible={int(ok)} derrumbe={int(derrumbe)} best={estado['best_epoca']} ({time.time() - t0:.0f}s val, {v['secs'] / max(v['n'], 1):.2f}s por mapa)", flush=True)
        if estado["derrumbes"] >= 2:
            estado["corte"] = True
            print(f"[{a.dataset}/{tag}] corte por segundo derrumbe seguido en la epoca {epoca}", flush=True)

    if estado["epoca"] == 0 and not ultimo.exists():
        validar_y_registrar(0, float("nan"), 0, 0.0)
        estado["derrumbes"] = 0
        guardar_mem(ultimo, model, 0, None, {"opt": opt.state_dict(), "estado": estado})
    while estado["epoca"] < a.epochs and not estado["corte"]:
        epoca = estado["epoca"] + 1
        t0 = time.time()
        loss, n_lotes = epoca_entrenamiento(model, fotos, pre, cfg, opt, rng, a.device, a.tope_fotos)
        secs_train = time.time() - t0
        if "mps" in (a.device, a.dev_unet):
            torch.mps.empty_cache()
        estado["epoca"] = epoca
        if epoca % a.val_every == 0 or epoca == a.epochs:
            validar_y_registrar(epoca, loss, len(fotos), secs_train)
        else:
            log.log(epoca=epoca, ronda=0, loss=f(loss), n_fotos=len(fotos), secs_train=round(secs_train, 1),
                    best_epoca=estado["best_epoca"], best_val=f(estado["best_val"]))
            print(f"[{a.dataset}/{tag}] epoca {epoca} loss={f(loss)} ({secs_train:.0f}s, {n_lotes} lotes)", flush=True)
        guardar_mem(ultimo, model, epoca, None, {"opt": opt.state_dict(), "estado": estado})
    if estado["best_epoca"] is None:
        raw = torch.load(out / "epoca_0.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(raw["model_state"])
        guardar_mem(out / "best.pt", model, 0, raw["val"], {"ronda": 0, "sin_elegibles": True})
        estado["best_epoca"] = 0
    meta.update({"best_epoca": estado["best_epoca"], "best_val": estado["best_val"], "corte": estado["corte"], "epoca_final": estado["epoca"]})
    (out / "config.json").write_text(json.dumps(meta, indent=1))
    log.close()
    (out / "listo").write_text(f"epoca {estado['epoca']} best {estado['best_epoca']}\n")
    print(f"[{a.dataset}/{tag}] listo, best epoca {estado['best_epoca']} val {f(estado['best_val'])}", flush=True)


if __name__ == "__main__":
    main()
