# exp_escalera. entrena un peldano con el algoritmo 1 de PHIL, el mismo ciclo de train_phil_res y de exp_res,
# roll-in mezclado por beta, perdida tbtt de error cuadratico contra h*, validacion en los 70 mapas cada val_every iteraciones y en la 0
#   python -m experiments.train_escalera --dataset mazes --seed 0 --peldano R2 [--root results/exp_escalera --device cuda --val-every 4]
# R0 es PHIL original. R1 a R4 y C llevan la u-net de cf_filtro de la misma semilla congelada, C ademas phil_res_s congelado de la misma semilla
# seleccion camino5 contra h_fs con la iteracion 0 como candidata. R2 R3 R4 y C ademas tienen que ganar a su iteracion 0, que es cf_filtro,
# y si ninguna gana el brazo queda en cf_filtro. R0 y R1 no tienen vuelta atras ni corte
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam

from experiments.common import ESCALA, EXP_CF2, MAX_EXP, RESULTS, RESULTS_GEO, grids_de, leer_csv, load_phil, load_problem, set_threads
from experiments.train_cf import elegible, f, referencia_val
from src.cf.escalera import D_CNN, PELDANOS, Compuerta, Escalera, EscaleraConfig, estadisticas_rasgos
from src.cf.fotos import Fotos
from src.cf.red import cargar
from src.phil.config import PhilConfig
from src.phil.data import grid_to_graph, load_grid
from src.phil.model import PHIL
from src.phil.training import CSVLogger, collect_trajectory, tbtt_loss
from src.search.greedy import greedy_search

ROOT_ESC = RESULTS_GEO.parent / "exp_escalera"
CON_BASE = ("R2", "R3", "R4", "C")
COLS = ["epoca", "loss", "n_tray", "secs_train", "val_exp", "val_camino", "val_encontrados", "elegible", "gana_cf", "derrumbe",
        "secs_val", "secs_red_val", "best_epoca", "best_val", "mejor_aprendida", "mejor_aprendida_val"]


def cf_ckpt_de(refs2, dataset, seed):
    return Path(refs2) / "cf_filtro" / "runs" / dataset / f"cf_filtro_s{seed}" / "best.pt"


def phil_ckpt_de(refs, dataset, seed):
    return Path(refs) / "phil_res_s" / "runs" / dataset / f"phil_res_s_n108_s{seed}" / "best.pt"


def referencia_cf(refs2, dataset, seed):
    # validacion de la epoca elegida de cf_filtro en exp_cf2, medida en la mac, solo como registro
    d = cf_ckpt_de(refs2, dataset, seed).parent
    meta = json.loads((d / "config.json").read_text())
    fila = next((r for r in leer_csv(d / "train.csv") if r["val_exp"] and int(r["epoca"]) == int(meta["best_epoca"])), None)
    return {"exp": meta.get("best_val"), "epoca": meta.get("best_epoca"), "camino": float(fila["val_camino"]) if fila else None}


def validar(model, dataset, max_exp=MAX_EXP):
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
        secs_red += float(getattr(model, "secs_red", 0.0))
        exps.append(res.n_expanded if res.found else max_exp)
        if res.found:
            caminos.append((len(res.path) - 1) / max(opt, 1))
    model.train()
    return {"exp": float(np.mean(exps)), "camino": float(np.mean(caminos)) if caminos else float("nan"), "encontrados": len(caminos),
            "secs_red": secs_red, "secs": secs, "n": len(exps)}


def estado_modelo(model):
    return model.state_dict_phil() if hasattr(model, "state_dict_phil") else model.state_dict()


def guardar(path, model, opt, it, val, cfg, peldano, extra=None):
    torch.save({"iteration": it, "epoca": it, "model_state": estado_modelo(model), "optimizer_state": opt.state_dict() if opt else None,
                "val": val, "val_metric": (val or {}).get("exp"), "config": {**cfg.to_dict(), "peldano": peldano},
                "extra": {"peldano": peldano, **(extra or {})}}, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--peldano", required=True, choices=PELDANOS)
    p.add_argument("--n-iter", type=int, default=108)
    p.add_argument("--val-every", type=int, default=4, help="cada cuantas iteraciones se valida hasta val_denso")
    p.add_argument("--val-denso", type=int, default=24, help="hasta esta iteracion se valida cada val_every, despues cada val_every_tarde")
    p.add_argument("--val-every-tarde", type=int, default=12)
    p.add_argument("--root", default=str(ROOT_ESC))
    p.add_argument("--refs2", default=str(EXP_CF2))
    p.add_argument("--refs", default=str(RESULTS))
    p.add_argument("--device", default="cuda", help="u-net de cf_filtro, phil va en cpu")
    p.add_argument("--tag", default=None)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--tope-delta", type=float, default=1.0)
    p.add_argument("--n-stats", type=int, default=200, help="fotos para la estandarizacion de los rasgos")
    p.add_argument("--max-exp", type=int, default=MAX_EXP)
    p.add_argument("--n-val", type=int, default=0, help="humo, recorta la validacion")
    p.add_argument("--n-train", type=int, default=0, help="humo, recorta los mapas de entrenamiento")
    a = p.parse_args()
    set_threads(a.threads)
    if a.n_val:
        import experiments.common as cm
        cm.N_VAL = a.n_val
    tag = a.tag or f"{a.peldano}_s{a.seed}"
    out = Path(a.root) / "runs" / a.dataset / tag
    out.mkdir(parents=True, exist_ok=True)
    if (out / "listo").exists():
        print(f"[{a.dataset}/{tag}] ya esta listo", flush=True)
        return
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    cfgj, ultimo = out / "config.json", out / "last.pt"
    previo = json.loads(cfgj.read_text()) if cfgj.exists() and ultimo.exists() else None
    ref = referencia_val(a.dataset)
    con_base = a.peldano in CON_BASE
    info = {}
    if a.peldano == "R0":
        cfg = PhilConfig(coord_scale=ESCALA, N_iter=a.n_iter)
        model = PHIL(cfg)
        params = list(model.parameters())
        n_entrenables = sum(q.numel() for q in params)
    else:
        ck = cf_ckpt_de(a.refs2, a.dataset, a.seed)
        unet, ucfg, raw_u = cargar(ck, a.device)
        if previo and previo.get("rasgos_media"):
            media, std = previo["rasgos_media"], previo["rasgos_std"]
        else:
            t0 = time.time()
            fotos = Fotos.cargar(RESULTS / "fotos" / a.dataset / "hfs")
            media, std = estadisticas_rasgos(unet, fotos, a.device, a.n_stats, seed=a.seed)
            del fotos
            print(f"[{a.dataset}/{tag}] estandarizacion de los rasgos sobre {a.n_stats} fotos base ({time.time() - t0:.0f}s)", flush=True)
        phil_ck = phil_ckpt_de(a.refs, a.dataset, a.seed) if a.peldano == "C" else None
        cfg = EscaleraConfig(peldano=a.peldano, d_enc=D_CNN, fs_every=ucfg.every_fs, every_cf=ucfg.every_cf, coord_scale=ESCALA, cf_ckpt=str(ck),
                             phil_ckpt=str(phil_ck or ""), tope_delta=a.tope_delta, N_iter=a.n_iter, rasgos_media=list(media), rasgos_std=list(std))
        if a.peldano == "C":
            phil, pcfg, raw_p = load_phil(phil_ck)
            model = Compuerta(cfg, phil, unet, a.device)
            info["iteracion_phil"] = raw_p.get("iteration")
        else:
            model = Escalera(cfg, unet, a.device)
        params = model.parameters_phil()
        n_entrenables = model.n_entrenables()
        info.update({"cf_ckpt": str(ck), "epoca_cf": raw_u.get("epoca"), "val_cf_mac": referencia_cf(a.refs2, a.dataset, a.seed), "phil_ckpt": str(phil_ck or ""),
                     "rasgos_media": list(media), "rasgos_std": list(std)})
    opt = Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    train_paths = grids_de(a.dataset, "train")
    if a.n_train:
        train_paths = train_paths[:a.n_train]
    meta = {"dataset": a.dataset, "seed": a.seed, "peldano": a.peldano, "tag": tag, "args": vars(a), "config": {**cfg.to_dict(), "peldano": a.peldano},
            "ref_val": ref, "n_params": sum(q.numel() for q in params), "n_entrenables": n_entrenables, "epocas": a.n_iter, "val_every": a.val_every,
            "val_denso": a.val_denso, "val_every_tarde": a.val_every_tarde, **info}

    def toca_validar(i):
        # denso al principio, donde phil gana y se derrumba, y ralo despues
        if i <= a.val_denso:
            return i % a.val_every == 0
        return i % a.val_every_tarde == 0
    cfgj.write_text(json.dumps(meta, indent=1))
    log = CSVLogger(out / "train.csv", COLS)
    estado = {"epoca": 0, "best_epoca": None, "best_val": None, "derrumbes": 0, "corte": False, "val0": None,
              "mejor_aprendida": None, "mejor_aprendida_val": None, "min_epoca": None, "min_val": None}
    replay = []
    if ultimo.exists():
        raw = torch.load(ultimo, map_location="cpu", weights_only=False)
        model.load_state_dict(raw["model_state"], strict=(a.peldano == "R0"))
        opt.load_state_dict(raw["optimizer_state"]); estado = raw["extra"]["estado"]; replay = raw["extra"]["replay"]
        random.setstate(raw["extra"]["rng"]["random"]); np.random.set_state(raw["extra"]["rng"]["numpy"]); torch.set_rng_state(raw["extra"]["rng"]["torch"])
        print(f"[{a.dataset}/{tag}] sigue desde la iteracion {estado['epoca']} con {len(replay)} trayectorias", flush=True)

    def guardar_ultimo():
        guardar(ultimo, model, opt, estado["epoca"], None, cfg, a.peldano,
                {"estado": estado, "replay": replay, "rng": {"random": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}})

    def validar_y_registrar(it, loss, n_tray, secs_train):
        t0 = time.time()
        v = validar(model, a.dataset, a.max_exp)
        ok = elegible(v, ref)
        if it == 0:
            estado["val0"] = v["exp"]
        gana_cf = int(con_base and it > 0 and estado["val0"] is not None and v["exp"] < estado["val0"] - 1e-9)
        derrumbe = v["encontrados"] < ref["encontrados"] or v["exp"] > 1.5 * ref["exp"]
        estado["derrumbes"] = estado["derrumbes"] + 1 if derrumbe else 0
        if estado["min_val"] is None or v["exp"] < estado["min_val"]:
            estado["min_epoca"], estado["min_val"] = it, v["exp"]
        if it > 0 and ok and (estado["mejor_aprendida_val"] is None or v["exp"] < estado["mejor_aprendida_val"]):
            estado["mejor_aprendida"], estado["mejor_aprendida_val"] = it, v["exp"]
        candidata = (ok and gana_cf) if con_base else ok
        if candidata and (estado["best_val"] is None or v["exp"] < estado["best_val"]):
            estado["best_epoca"], estado["best_val"] = it, v["exp"]
            guardar(out / "best.pt", model, None, it, v, cfg, a.peldano)
        guardar(out / f"epoca_{it}.pt", model, None, it, v, cfg, a.peldano)
        log.log(epoca=it, loss=f(loss), n_tray=n_tray, secs_train=round(secs_train, 1), val_exp=round(v["exp"], 1), val_camino=f(v["camino"]),
                val_encontrados=v["encontrados"], elegible=int(ok), gana_cf=gana_cf, derrumbe=int(derrumbe), secs_val=round(time.time() - t0, 1),
                secs_red_val=round(v["secs_red"], 1), best_epoca=estado["best_epoca"], best_val=f(estado["best_val"]),
                mejor_aprendida=estado["mejor_aprendida"], mejor_aprendida_val=f(estado["mejor_aprendida_val"]))
        print(f"[{a.dataset}/{tag}] iter {it} loss={f(loss)} val_exp={v['exp']:.1f} camino={f(v['camino'])} enc={v['encontrados']} elegible={int(ok)} "
              f"gana_cf={gana_cf} derrumbe={int(derrumbe)} best={estado['best_epoca']} ({time.time() - t0:.0f}s val, {v['secs'] / max(v['n'], 1):.2f}s por mapa)", flush=True)
        if con_base and estado["derrumbes"] >= 2:
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
        # una pasada sobre todo lo acumulado, lotes de batch_size trayectorias, como en exp_res
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
        if toca_validar(i) or i == cfg.N_iter:
            validar_y_registrar(i, avg_loss, len(replay), secs_train)
        else:
            log.log(epoca=i, loss=f(avg_loss), n_tray=len(replay), secs_train=round(secs_train, 1), best_epoca=estado["best_epoca"], best_val=f(estado["best_val"]))
            print(f"[{a.dataset}/{tag}] iter {i} loss={f(avg_loss)} beta={beta:.3f} |D|={len(replay)} ({secs_train:.1f}s)", flush=True)
        guardar_ultimo()
    flag = {}
    if estado["best_epoca"] is None:
        # con base queda en cf_filtro, que es la iteracion 0. sin base se elige la de menos expansiones aunque no sea elegible
        src_it = 0 if con_base else estado["min_epoca"]
        flag = {"queda_cf": True} if con_base else {"sin_elegibles": True}
        raw = torch.load(out / f"epoca_{src_it}.pt", map_location="cpu", weights_only=False)
        torch.save({**raw, "extra": {**raw["extra"], **flag}}, out / "best.pt")
        estado["best_epoca"], estado["best_val"] = src_it, (raw.get("val") or {}).get("exp")
    val_cf = (info.get("val_cf_mac") or {}).get("exp")
    meta.update({"best_epoca": estado["best_epoca"], "best_val": estado["best_val"], "val0": estado["val0"], "corte": estado["corte"], "epoca_final": estado["epoca"],
                 "mejor_aprendida": estado["mejor_aprendida"], "mejor_aprendida_val": estado["mejor_aprendida_val"], "min_epoca": estado["min_epoca"], "min_val": estado["min_val"],
                 "queda_cf": bool(flag.get("queda_cf")), "sin_elegibles": bool(flag.get("sin_elegibles")),
                 "epoca0_igual_cf_mac": (abs(estado["val0"] - val_cf) < 0.05) if (con_base and val_cf is not None and estado["val0"] is not None and not a.n_val) else None})
    cfgj.write_text(json.dumps(meta, indent=1))
    log.close()
    (out / "listo").write_text(f"iteracion {estado['epoca']} best {estado['best_epoca']}\n")
    print(f"[{a.dataset}/{tag}] listo, best iteracion {estado['best_epoca']} val {f(estado['best_val'])} {'queda en cf_filtro' if flag.get('queda_cf') else ''}", flush=True)


if __name__ == "__main__":
    main()
