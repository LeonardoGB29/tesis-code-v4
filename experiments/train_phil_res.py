# exp_mem: phil_res_cf, PHIL tal cual (algoritmo 1 de PHIL, roll-in mezclado por beta y perdida tbtt de error cuadratico contra h*)
# con la base h_fs / cf_filtro congelado de la misma semilla; copia del ciclo de exp_res de code_v2 con la validacion de code_v3,
# 70 mapas cada val_every iteraciones y en la 0, tope 10 000, camino5 respecto de la voraz con h_fs, derrumbe y corte como train_cf
#   python -m experiments.train_phil_res --dataset mazes --seed 0 [--root results/exp_mem --refs2 results/exp_cf2 --device mps]
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam

from experiments.common import ESCALA, MAX_EXP, grids_de, load_problem, set_threads
from experiments.fotos_mem import REFS2, ROOT_MEM, cf_ckpt_de
from experiments.train_cf import COLS, elegible, f, referencia_val
from src.cf.phil_res import PhilResCF, ResCfConfig
from src.cf.red import cargar
from src.phil.data import build_adjacency, grid_to_graph, load_grid
from src.phil.training import CSVLogger, collect_trajectory, tbtt_loss
from src.search.greedy import greedy_search

MODELO = "phil_res_cf"


def validar_phil(model, dataset, max_exp=MAX_EXP):
    model.eval()
    exps, caminos, secs_red, secs = [], [], 0.0, 0.0
    for gp in grids_de(dataset, "validation"):
        data, adj, v_s, v_g, h_star = load_problem(gp)
        opt = int(h_star[v_s])
        if opt < 0:
            continue
        t0 = time.time()
        res = greedy_search(model, data, v_s, v_g, adj=adj, max_exp=max_exp, return_path=True)
        secs += time.time() - t0
        secs_red += model.secs_red
        exps.append(res.n_expanded if res.found else max_exp)
        if res.found:
            caminos.append((len(res.path) - 1) / max(opt, 1))
    model.train()
    return {"exp": float(np.mean(exps)), "camino": float(np.mean(caminos)) if caminos else float("nan"), "encontrados": len(caminos),
            "secs_red": secs_red, "secs": secs, "n": len(exps)}


def guardar_phil(path, model, opt, iteracion, val, cfg, extra=None):
    torch.save({"iteration": iteracion, "epoca": iteracion, "model_state": model.state_dict(), "optimizer_state": opt.state_dict() if opt else None,
                "val": val, "val_metric": (val or {}).get("exp"), "config": cfg.to_dict(), "extra": extra or {}}, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-iter", type=int, default=108)
    p.add_argument("--val-every", type=int, default=4)
    p.add_argument("--root", default=str(ROOT_MEM))
    p.add_argument("--refs2", default=str(REFS2))
    p.add_argument("--device", default="mps", help="u-net; phil va en cpu")
    p.add_argument("--tag", default=None)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--n-val", type=int, default=0, help="humo: recorta la validacion")
    p.add_argument("--n-train", type=int, default=0, help="humo: recorta los mapas de entrenamiento")
    a = p.parse_args()
    set_threads(a.threads)
    if a.n_val:
        import experiments.common as cm
        cm.N_VAL = a.n_val
    tag = a.tag or f"{MODELO}_s{a.seed}"
    out = Path(a.root) / MODELO / "runs" / a.dataset / tag
    out.mkdir(parents=True, exist_ok=True)
    if (out / "listo").exists():
        print(f"[{a.dataset}/{tag}] ya esta listo", flush=True)
        return
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    ck = cf_ckpt_de(a.refs2, a.dataset, a.seed)
    unet, ucfg, raw_u = cargar(ck, a.device)
    # los mismos hiperparametros de phil_res_s en exp_res: T 256, t_tau 32, beta 0.7, lote 32, adam 1e-3, escala 200, 8 vecinos, h_fs cada 16
    cfg = ResCfConfig(fs_every=ucfg.every_fs, coord_scale=ESCALA, d_enc=2, cf_ckpt=str(ck), every_cf=ucfg.every_cf, N_iter=a.n_iter)
    model = PhilResCF(cfg, unet=unet, device=a.device)
    params = model.parameters_phil()
    opt = Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    ref = referencia_val(a.dataset)
    train_paths = grids_de(a.dataset, "train")
    if a.n_train:
        train_paths = train_paths[:a.n_train]
    meta = {"dataset": a.dataset, "seed": a.seed, "modelo": MODELO, "tag": tag, "args": vars(a), "config": cfg.to_dict(), "cf_ckpt": str(ck),
            "epoca_cf": raw_u.get("epoca"), "val_cf": raw_u.get("val"), "ref_val": ref, "n_params": sum(p.numel() for p in params),
            "n_entrenables": sum(p.numel() for p in params), "epocas": a.n_iter, "val_every": a.val_every}
    (out / "config.json").write_text(json.dumps(meta, indent=1))
    log = CSVLogger(out / "train.csv", COLS)
    estado = {"epoca": 0, "best_epoca": None, "best_val": None, "derrumbes": 0, "corte": False}
    replay = []
    ultimo = out / "last.pt"
    if ultimo.exists():
        raw = torch.load(ultimo, map_location="cpu", weights_only=False)
        model.load_state_dict(raw["model_state"]); model.unet.to(a.device)
        opt.load_state_dict(raw["optimizer_state"]); estado = raw["extra"]["estado"]; replay = raw["extra"]["replay"]
        random.setstate(raw["extra"]["rng"]["random"]); np.random.set_state(raw["extra"]["rng"]["numpy"]); torch.set_rng_state(raw["extra"]["rng"]["torch"])
        print(f"[{a.dataset}/{tag}] sigue desde la iteracion {estado['epoca']} con {len(replay)} trayectorias", flush=True)

    def guardar_ultimo():
        guardar_phil(ultimo, model, opt, estado["epoca"], None, cfg,
                     {"estado": estado, "replay": replay, "rng": {"random": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}})

    def validar_y_registrar(it, loss, n_tray, secs_train):
        t0 = time.time()
        v = validar_phil(model, a.dataset)
        ok = elegible(v, ref)
        derrumbe = v["encontrados"] < ref["encontrados"] or v["exp"] > 1.5 * ref["exp"]
        estado["derrumbes"] = estado["derrumbes"] + 1 if derrumbe else 0
        if ok and (estado["best_val"] is None or v["exp"] < estado["best_val"]):
            estado["best_epoca"], estado["best_val"] = it, v["exp"]
            guardar_phil(out / "best.pt", model, None, it, v, cfg, {"ronda": 0})
        guardar_phil(out / f"epoca_{it}.pt", model, None, it, v, cfg, {"ronda": 0})
        log.log(epoca=it, ronda=0, loss=f(loss), n_fotos=n_tray, secs_train=round(secs_train, 1), val_exp=round(v["exp"], 1),
                val_camino=f(v["camino"]), val_encontrados=v["encontrados"], elegible=int(ok), derrumbe=int(derrumbe),
                secs_val=round(time.time() - t0, 1), secs_red_val=round(v["secs_red"], 1), best_epoca=estado["best_epoca"], best_val=f(estado["best_val"]))
        print(f"[{a.dataset}/{tag}] iter {it} loss={f(loss)} val_exp={v['exp']:.1f} camino={f(v['camino'])} enc={v['encontrados']} "
              f"elegible={int(ok)} derrumbe={int(derrumbe)} best={estado['best_epoca']} ({time.time() - t0:.0f}s val, {v['secs'] / max(v['n'], 1):.2f}s por mapa)", flush=True)
        if estado["derrumbes"] >= 2:
            estado["corte"] = True
            print(f"[{a.dataset}/{tag}] corte por segundo derrumbe seguido en la iteracion {it}", flush=True)

    if estado["epoca"] == 0 and not ultimo.exists():
        validar_y_registrar(0, float("nan"), 0, 0.0)
        estado["derrumbes"] = 0
        guardar_ultimo()
    while estado["epoca"] < cfg.N_iter and not estado["corte"]:
        i = estado["epoca"] + 1
        t0 = time.time()
        beta = cfg.beta_0 ** i
        grid_path = random.choice(train_paths)
        data, adj, v_s, v_g, h_star = load_problem(grid_path)
        model.train()
        for _ in range(cfg.m_trajectories):
            traj, z0 = collect_trajectory(model, data, adj, h_star, v_s, v_g, cfg, beta)
            if not traj:
                continue
            if not cfg.rolled_in_state:
                z0 = torch.zeros_like(z0)
            replay.append({"trajectory": traj, "z_t_init": z0, "grid_path": str(grid_path), "v_g": v_g})
        # una pasada sobre todo lo agregado, lotes de batch_size trayectorias, como en exp_res
        loss_sum, n_samples, n_batch = 0.0, 0, 0
        opt.zero_grad()
        orden = list(replay)
        random.shuffle(orden)
        for s in orden:
            s_data = grid_to_graph(load_grid(s["grid_path"]))
            loss = tbtt_loss(model, s_data, s_data.x[s["v_g"]], s["trajectory"], s["z_t_init"])
            (loss / cfg.batch_size).backward()
            loss_sum += float(loss.item()); n_samples += 1; n_batch += 1
            if n_batch >= cfg.batch_size:
                if cfg.grad_clip:
                    torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
                opt.step(); opt.zero_grad(); n_batch = 0
        if n_batch > 0:
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step(); opt.zero_grad()
        secs_train = time.time() - t0
        avg_loss = loss_sum / max(n_samples, 1) * cfg.coord_scale ** 2            # en saltos al cuadrado, como exp_res
        estado["epoca"] = i
        if i % a.val_every == 0 or i == cfg.N_iter:
            validar_y_registrar(i, avg_loss, len(replay), secs_train)
        else:
            log.log(epoca=i, ronda=0, loss=f(avg_loss), n_fotos=len(replay), secs_train=round(secs_train, 1),
                    best_epoca=estado["best_epoca"], best_val=f(estado["best_val"]))
            print(f"[{a.dataset}/{tag}] iter {i} loss={f(avg_loss)} beta={beta:.3f} |D|={len(replay)} ({secs_train:.1f}s)", flush=True)
        guardar_ultimo()
    if estado["best_epoca"] is None:
        raw = torch.load(out / "epoca_0.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(raw["model_state"])
        guardar_phil(out / "best.pt", model, None, 0, raw["val"], cfg, {"ronda": 0, "sin_elegibles": True})
        estado["best_epoca"] = 0
    meta.update({"best_epoca": estado["best_epoca"], "best_val": estado["best_val"], "corte": estado["corte"], "epoca_final": estado["epoca"]})
    (out / "config.json").write_text(json.dumps(meta, indent=1))
    log.close()
    (out / "listo").write_text(f"iteracion {estado['epoca']} best {estado['best_epoca']}\n")
    print(f"[{a.dataset}/{tag}] listo, best iteracion {estado['best_epoca']} val {f(estado['best_val'])}", flush=True)


if __name__ == "__main__":
    main()
