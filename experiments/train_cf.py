# entrena el factor de correccion con fotos del mapa descubierto, validacion con busqueda cada val_every epocas
#   python -m experiments.train_cf --dataset mazes --seed 0 --modelo cf_global
#   modelos: cf_global, cf_local, cf_global_gnn (--init cf_global), cf_global_dagger (--init cf_global)
#   exp_cf2: cf_filtro y cf_grueso, escala gruesa dentro del modelo, 10 epocas con fotos de h_fs y despues 2 rondas de dagger en la misma corrida
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from experiments.common import ESCALA, MAX_EXP, RESULTS, grids_de, leer_csv, load_problem, set_threads
from experiments.fotos import FOTO_CADA, recolectar, referencia
from src.cf.busqueda import EvaluadorCF, EvaluadorGNN, voraz
from src.cf.fotos import Fotos, lote
from src.cf.red import CfConfig, construir, guardar
from src.phil.training import CSVLogger

TIPOS = {"cf_global": "global", "cf_local": "local", "cf_global_gnn": "gnn", "cf_global_dagger": "global", "cf_filtro": "global", "cf_grueso": "global"}
SALIDAS = {"cf_filtro": "filtro", "cf_grueso": "gruesa"}                 # exp_cf2
DAGGER = {"cf_global_dagger", "cf_filtro", "cf_grueso"}
SIGMA = {"gaps_and_forest": 16.0, "multiple_bugtraps": 40.0, "mazes": 12.0}     # barrido en results/exp_cf/diagnostico/barrido_sigma_<ds>.txt
COLS = ["epoca", "ronda", "loss", "n_fotos", "secs_train", "val_exp", "val_camino", "val_encontrados", "elegible",
        "derrumbe", "secs_val", "secs_red_val", "best_epoca", "best_val"]


def hacer_evaluador(model, cfg, data, adj, v_g, device):
    if cfg.tipo == "gnn":
        return EvaluadorGNN(data, v_g, model, adj, cfg.every_cf, cfg.every_fs, ESCALA, device)
    return EvaluadorCF(data, v_g, model, cfg.every_cf, cfg.every_fs, ESCALA, device)


def validar(model, cfg, dataset, device, max_exp=MAX_EXP):
    model.eval()
    if device == "mps":
        torch.mps.empty_cache()
    exps, caminos, secs_red = [], [], 0.0
    for gp in grids_de(dataset, "validation"):
        data, adj, v_s, v_g, h_star = load_problem(gp)
        opt = int(h_star[v_s])
        if opt < 0:
            continue
        ev = hacer_evaluador(model, cfg, data, adj, v_g, device)
        res = voraz(ev, data, v_s, v_g, adj, max_exp=max_exp)
        exps.append(res.n_expanded if res.found else max_exp)
        secs_red += res.secs_red
        if res.found:
            caminos.append((len(res.path) - 1) / max(opt, 1))
    model.train()
    return {"exp": float(np.mean(exps)), "camino": float(np.mean(caminos)) if caminos else float("nan"),
            "encontrados": len(caminos), "secs_red": secs_red}


def referencia_val(dataset):
    path = RESULTS / "refs" / f"val_hfs_{dataset}.csv"
    if not path.exists():
        referencia(dataset, "validation", path)
    import experiments.common as cm
    rows = leer_csv(path)[:cm.N_VAL]
    enc = [r for r in rows if r["encontrado"] == "1"]
    return {"exp": float(np.mean([float(r["exp"]) for r in rows])),
            "camino": float(np.mean([int(r["camino"]) / max(int(r["optimo"]), 1) for r in enc])), "encontrados": len(enc)}


def elegible(val, ref, tol=0.05):
    # camino5: no pierde mapas y camino a no mas de tol del de la voraz con h_fs
    return val["encontrados"] >= ref["encontrados"] and val["camino"] <= (1 + tol) * ref["camino"]


def vecinos_libres(real, rc):
    # rc [n, 2] celdas; devuelve lista de listas con las vecinas libres del mapa real (hasta 8)
    H, W = real.shape
    out = []
    for r, c in rc:
        v = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if (dr or dc) and 0 <= rr < H and 0 <= cc < W and not real[rr, cc]:
                    v.append((rr, cc))
        out.append(v)
    return out


def perdida_gnn(model, fotos, idx, n_nodos, rng, device):
    # muestrea n_nodos celdas etiquetadas por foto y entrena la gnn con los rasgos congelados de la u-net
    x, y, m = lote(fotos, idx, None, device)
    with torch.no_grad():
        rasgos = model.unet.caracteristicas(x)
    losses = []
    for b, i in enumerate(idx):
        mp = int(fotos.mapa_idx[int(i)])
        celdas = np.argwhere(m[b].cpu().numpy())
        if len(celdas) == 0:
            continue
        sel = celdas[rng.choice(len(celdas), size=min(n_nodos, len(celdas)), replace=False)]
        vec = vecinos_libres(fotos.real[mp], sel)
        rc_n, src, dst, pos = [], [], [], {}
        for j, vs in enumerate(vec):
            for v in vs:
                if v not in pos:
                    pos[v] = len(rc_n)
                    rc_n.append(v)
                src.append(len(sel) + pos[v]); dst.append(j)
        rc_f = torch.tensor(sel, dtype=torch.float, device=device)
        rc_nt = torch.tensor(rc_n, dtype=torch.float, device=device) if rc_n else torch.zeros((0, 2), device=device)
        ei = torch.tensor([src, dst], dtype=torch.long, device=device)
        goal = torch.tensor(fotos.goal[mp], dtype=torch.float, device=device)
        cf = model(rc_f, rc_nt, ei, goal, rasgos[b])
        obj = y[b][sel[:, 0], sel[:, 1]]
        losses.append(((cf - obj) ** 2).mean())
    return torch.stack(losses).mean() if losses else None


def epoca_entrenamiento(model, cfg, fotos, opt, batch, rng, device, n_nodos):
    model.train()
    orden = rng.permutation(len(fotos))
    tot, n = 0.0, 0
    for k in range(0, len(orden), batch):
        idx = orden[k:k + batch]
        if cfg.tipo == "gnn":
            loss = perdida_gnn(model, fotos, idx, n_nodos, rng, device)
            if loss is None:
                continue
        else:
            x, y, m = lote(fotos, idx, rng, device)
            pred = model(x)
            loss = (((pred - y) ** 2) * m).sum() / m.sum().clamp(min=1)
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
    p.add_argument("--modelo", required=True, choices=list(TIPOS))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--epochs-ronda", type=int, default=10, help="dagger: epocas por ronda")
    p.add_argument("--rondas", type=int, default=2)
    p.add_argument("--epochs-inicio", type=int, default=None, help="dagger: epocas con solo fotos de h_fs antes de la ronda 1 (10 en cf_filtro y cf_grueso, 0 en cf_global_dagger)")
    p.add_argument("--sigma", type=float, default=None, help="cf_filtro: desvio del gauss, por defecto el del conjunto")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--val-every", type=int, default=5)
    p.add_argument("--ancho", type=int, default=16)
    p.add_argument("--nodos", type=int, default=128, help="gnn: nodos muestreados por foto")
    p.add_argument("--init", default=None, help="checkpoint inicial (gnn y dagger)")
    p.add_argument("--fotos", default=None)
    p.add_argument("--out-root", default=None)
    p.add_argument("--tag", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--max-fotos", type=int, default=0, help="humo: recorta el conjunto de fotos")
    p.add_argument("--n-val", type=int, default=0, help="humo: recorta la validacion")
    p.add_argument("--n-train", type=int, default=0, help="humo: recorta los mapas de entrenamiento del dagger")
    a = p.parse_args()
    set_threads(a.threads)
    if a.n_val or a.n_train:
        import experiments.common as cm
        cm.N_VAL = a.n_val or cm.N_VAL
        cm.N_TRAIN = a.n_train or cm.N_TRAIN
    tag = a.tag or f"{a.modelo}_s{a.seed}"
    out = Path(a.out_root or RESULTS / a.modelo / "runs") / a.dataset / tag
    out.mkdir(parents=True, exist_ok=True)
    if (out / "listo").exists():
        print(f"[{a.dataset}/{tag}] ya esta listo", flush=True)
        return
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    salida = SALIDAS.get(a.modelo, "fina")
    sigma = (a.sigma if a.sigma is not None else SIGMA[a.dataset]) if salida == "filtro" else 0.0
    cfg = CfConfig(tipo=TIPOS[a.modelo], ancho=a.ancho, salida=salida, sigma=sigma)
    model = construir(cfg)
    if a.init:
        raw = torch.load(a.init, map_location="cpu", weights_only=False)
        if cfg.tipo == "gnn":
            model.unet.load_state_dict(raw["model_state"])
        else:
            model.load_state_dict(raw["model_state"])
    model.to(a.device)
    params = model.entrenables() if cfg.tipo == "gnn" else list(model.parameters())
    opt = torch.optim.Adam(params, lr=a.lr)
    ref = referencia_val(a.dataset)
    fotos_base = Fotos.cargar(a.fotos or RESULTS / "fotos" / a.dataset / "hfs")
    if a.max_fotos:
        fotos_base = Fotos(fotos_base.nombres, fotos_base.real, fotos_base.hstar, fotos_base.goal, fotos_base.start,
                           fotos_base.mapa_idx[:a.max_fotos], fotos_base.visto[:a.max_fotos], fotos_base.hfs[:a.max_fotos], fotos_base.escala)
    dagger = a.modelo in DAGGER
    inicio = (a.epochs_inicio if a.epochs_inicio is not None else (10 if a.modelo in SALIDAS else 0)) if dagger else 0
    rondas = a.rondas if dagger else 1
    epocas_ronda = a.epochs_ronda if dagger else a.epochs
    total = inicio + rondas * epocas_ronda                      # ronda 0 con fotos de h_fs, despues rondas con fotos propias acumuladas
    meta = {"dataset": a.dataset, "seed": a.seed, "modelo": a.modelo, "tag": tag, "args": vars(a), "config": cfg.to_dict(),
            "ref_val": ref, "n_params": sum(p.numel() for p in model.parameters()), "n_entrenables": sum(p.numel() for p in params),
            "n_fotos_base": len(fotos_base), "epocas": total, "epocas_inicio": inicio, "rondas": rondas}
    (out / "config.json").write_text(json.dumps(meta, indent=1))
    log = CSVLogger(out / "train.csv", COLS)
    estado = {"epoca": 0, "best_epoca": None, "best_val": None, "derrumbes": 0, "corte": False}
    ultimo = out / "last.pt"
    if ultimo.exists():                                            # relanzar: sigue desde la ultima epoca guardada
        raw = torch.load(ultimo, map_location="cpu", weights_only=False)
        model.load_state_dict(raw["model_state"]); model.to(a.device)
        opt.load_state_dict(raw["extra"]["opt"]); estado = raw["extra"]["estado"]
        rng = np.random.default_rng([a.seed, estado["epoca"]])
        print(f"[{a.dataset}/{tag}] sigue desde la epoca {estado['epoca']}", flush=True)

    def validar_y_registrar(epoca, ronda, loss, n_fotos, secs_train):
        t0 = time.time()
        v = validar(model, cfg, a.dataset, a.device)
        ok = elegible(v, ref)
        derrumbe = v["encontrados"] < ref["encontrados"] or v["exp"] > 1.5 * ref["exp"]     # pierde mapas o expande 50 % mas que h_fs
        estado["derrumbes"] = estado["derrumbes"] + 1 if derrumbe else 0
        if ok and (estado["best_val"] is None or v["exp"] < estado["best_val"]):
            estado["best_epoca"], estado["best_val"] = epoca, v["exp"]
            guardar(out / "best.pt", model, epoca, v, {"ronda": ronda})
        guardar(out / f"epoca_{epoca}.pt", model, epoca, v, {"ronda": ronda})
        log.log(epoca=epoca, ronda=ronda, loss=f(loss), n_fotos=n_fotos, secs_train=round(secs_train, 1), val_exp=round(v["exp"], 1),
                val_camino=f(v["camino"]), val_encontrados=v["encontrados"], elegible=int(ok), derrumbe=int(derrumbe),
                secs_val=round(time.time() - t0, 1), secs_red_val=round(v["secs_red"], 1), best_epoca=estado["best_epoca"], best_val=f(estado["best_val"]))
        print(f"[{a.dataset}/{tag}] epoca {epoca} loss={f(loss)} val_exp={v['exp']:.1f} camino={f(v['camino'])} enc={v['encontrados']} "
              f"elegible={int(ok)} derrumbe={int(derrumbe)} best={estado['best_epoca']} ({time.time() - t0:.0f}s val)", flush=True)
        if estado["derrumbes"] >= 2:
            estado["corte"] = True
            print(f"[{a.dataset}/{tag}] corte por segundo derrumbe seguido en la epoca {epoca}", flush=True)

    if estado["epoca"] == 0 and not ultimo.exists():
        validar_y_registrar(0, 0, float("nan"), 0, 0.0)
        estado["derrumbes"] = 0                                    # el punto de partida no cuenta como derrumbe (gnn y dagger desde cf_global)
        guardar(ultimo, model, 0, None, {"opt": opt.state_dict(), "estado": estado})
    fotos_rondas = {}
    while estado["epoca"] < total and not estado["corte"]:
        ronda = 0 if estado["epoca"] < inicio else (estado["epoca"] - inicio) // epocas_ronda + 1
        if dagger and ronda >= 1:
            for r in range(1, ronda):
                if r not in fotos_rondas:
                    fotos_rondas[r] = Fotos.cargar(out / f"fotos_ronda_{r}")
            if ronda not in fotos_rondas:
                # fotos de las busquedas del modelo actual sobre los mapas de entrenamiento
                ck = out / f"inicio_ronda_{ronda}.pt"
                if ck.exists():
                    model.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["model_state"]); model.to(a.device)
                else:
                    guardar(ck, model, estado["epoca"], None, {"ronda": ronda})
                fp = out / f"fotos_ronda_{ronda}"
                if not (fp / "listo").exists():
                    t0 = time.time()
                    model.eval()
                    fr, filas = recolectar(a.dataset, lambda data, adj, v_g: hacer_evaluador(model, cfg, data, adj, v_g, a.device), FOTO_CADA, MAX_EXP)
                    model.train()
                    fr.guardar(fp)
                    print(f"[{a.dataset}/{tag}] ronda {ronda}: {len(fr)} fotos propias en {time.time() - t0:.0f}s", flush=True)
                fotos_rondas[ronda] = Fotos.cargar(fp)
            fotos = Fotos.unir([fotos_base] + [fotos_rondas[r] for r in sorted(fotos_rondas) if r <= ronda])
        else:
            fotos = fotos_base
        epoca = estado["epoca"] + 1
        t0 = time.time()
        loss = epoca_entrenamiento(model, cfg, fotos, opt, a.batch, rng, a.device, a.nodos)
        secs_train = time.time() - t0
        if a.device == "mps":
            torch.mps.empty_cache()
        estado["epoca"] = epoca
        if epoca % a.val_every == 0 or epoca == total:
            validar_y_registrar(epoca, ronda, loss, len(fotos), secs_train)
        else:
            log.log(epoca=epoca, ronda=ronda, loss=f(loss), n_fotos=len(fotos), secs_train=round(secs_train, 1),
                    best_epoca=estado["best_epoca"], best_val=f(estado["best_val"]))
            print(f"[{a.dataset}/{tag}] epoca {epoca} loss={f(loss)} ({secs_train:.0f}s)", flush=True)
        guardar(ultimo, model, epoca, None, {"opt": opt.state_dict(), "estado": estado})
    if estado["best_epoca"] is None:                               # nada elegible: queda el inicio
        raw = torch.load(out / "epoca_0.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(raw["model_state"])
        guardar(out / "best.pt", model, 0, raw["val"], {"ronda": 0, "sin_elegibles": True})
        estado["best_epoca"] = 0
    meta.update({"best_epoca": estado["best_epoca"], "best_val": estado["best_val"], "corte": estado["corte"], "epoca_final": estado["epoca"]})
    (out / "config.json").write_text(json.dumps(meta, indent=1))
    log.close()
    (out / "listo").write_text(f"epoca {estado['epoca']} best {estado['best_epoca']}\n")
    print(f"[{a.dataset}/{tag}] listo, best epoca {estado['best_epoca']} val {f(estado['best_val'])}", flush=True)


def f(x, nd=4):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


if __name__ == "__main__":
    main()
