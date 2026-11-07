# exp_escalera. mapas de alpha de la compuerta C en mapas de prueba y alpha medio por tramo de h_fs del nodo y de avance de la busqueda
#   python -m experiments.figuras_escalera --root results/exp_escalera [--seed 0 --seed-ds gaps_and_forest=1 --n 3 --n-tramos 10 --device cpu]
# en gaps_and_forest la semilla 0 de C quedo en la iteracion 0, donde la ultima capa de la compuerta esta en cero y alpha es 0 por construccion,
# asi que ahi se usa la semilla 1, iteracion 12. en los otros dos conjuntos va la semilla 0
import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.common import DATASETS, MAX_EXP, grids_de, load_phil, load_problem, set_threads, write_csv
from experiments.train_escalera import ROOT_ESC
from src.search.greedy import greedy_search

TRAMOS_HFS = [(0, 50), (50, 100), (100, 200), (200, 400), (400, 10 ** 9)]
TRAMOS_AVANCE = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]


def buscar_con_alpha(model, gp):
    data, adj, v_s, v_g, h_star = load_problem(gp)
    model.alphas = []
    res = greedy_search(model, data, v_s, v_g, adj=adj, max_exp=MAX_EXP, return_path=True)
    pasos = model.alphas
    model.alphas = None
    rc = np.concatenate([p[0] for p in pasos]) if pasos else np.zeros((0, 2), int)
    al = np.concatenate([p[1] for p in pasos]) if pasos else np.zeros(0)
    hfs = np.concatenate([p[2] for p in pasos]) * model.coord_scale if pasos else np.zeros(0)
    avance = np.concatenate([np.full(len(p[1]), k / max(len(pasos) - 1, 1)) for k, p in enumerate(pasos)]) if pasos else np.zeros(0)
    return data, v_s, v_g, h_star, res, rc, al, hfs, avance


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(ROOT_ESC))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seed-ds", nargs="*", default=["gaps_and_forest=1"], help="semilla por conjunto, conjunto=semilla")
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--n-tramos", type=int, default=10)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    semillas = {ds: a.seed for ds in DATASETS}
    semillas.update({k: int(v) for k, v in (x.split("=") for x in a.seed_ds)})
    set_threads(1)
    root = Path(a.root)
    (root / "figuras").mkdir(exist_ok=True)
    filas, stats = [], []
    por_ds = {}
    for ds in DATASETS:
        seed = semillas[ds]
        ck = root / "runs" / ds / f"C_s{seed}" / "best.pt"
        if not ck.exists():
            print(f"[{ds}] sin compuerta en {ck}", flush=True)
            continue
        model, cfg, raw = load_phil(ck, a.device)
        grids = grids_de(ds, "test")[:max(a.n, a.n_tramos)]
        acum = {"hfs": {t: [] for t in TRAMOS_HFS}, "avance": {t: [] for t in TRAMOS_AVANCE}, "todo": []}
        fig, axes = plt.subplots(a.n, 2, figsize=(11, 5 * a.n), squeeze=False)
        for k, gp in enumerate(grids):
            data, v_s, v_g, h_star, res, rc, al, hfs, avance = buscar_con_alpha(model, gp)
            if int(h_star[v_s]) < 0:
                continue
            acum["todo"].append(al)
            for t in TRAMOS_HFS:
                acum["hfs"][t].append(al[(hfs >= t[0]) & (hfs < t[1])])
            for t in TRAMOS_AVANCE:
                acum["avance"][t].append(al[(avance >= t[0]) & (avance < t[1])])
            filas.append([ds, gp.name, seed, raw.get("iteration"), res.n_expanded, int(h_star[v_s]), len(res.path) - 1 if res.path else "", len(al),
                          round(float(al.mean()), 4) if len(al) else "", round(float((al > 0).mean()), 4) if len(al) else "", round(float(al.max()), 4) if len(al) else ""])
            if k < a.n:
                ev = model.ev
                visto, real = ev.desc.visto, ev.desc.real
                H, W = real.shape
                rcm = data.x.long().numpy()
                estado = np.full((H, W), 0.5); estado[visto & ~real] = 1.0; estado[visto & real] = 0.0
                ax = axes[k, 0]
                ax.imshow(estado, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                cl = np.array(sorted(res.closed)); exp = np.zeros((H, W), bool); exp[rcm[cl, 0], rcm[cl, 1]] = True
                ax.imshow(np.where(exp, 1.0, np.nan), cmap="autumn", alpha=0.5, vmin=0, vmax=1, interpolation="nearest")
                if res.path:
                    pth = rcm[np.array(res.path)]; ax.plot(pth[:, 1], pth[:, 0], color="lime", lw=1)
                ax.set_title(f"{ds} {gp.name}, C s{seed} iteracion {raw.get('iteration')}, exp {res.n_expanded}, optimo {int(h_star[v_s])}", fontsize=9)
                ax = axes[k, 1]
                ax.imshow(np.where(visto, estado, np.nan), cmap="gray", vmin=0, vmax=1, interpolation="nearest", alpha=0.35)
                amap = np.full((H, W), np.nan)
                if len(al):
                    amap[rc[:, 0], rc[:, 1]] = al
                im = ax.imshow(amap, cmap="viridis", vmin=0, vmax=max(float(np.nanmax(amap)), 1e-3) if len(al) else 1, interpolation="nearest")
                fig.colorbar(im, ax=ax, fraction=0.046)
                ax.set_title(f"alpha por nodo evaluado, media {float(al.mean()) if len(al) else 0:.3f}, nodos con alpha > 0 {float((al > 0).mean()) if len(al) else 0:.2f}", fontsize=9)
                for ax in axes[k]:
                    ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(root / "figuras" / f"alpha_{ds}.png", dpi=110)
        plt.close(fig)
        por_ds[ds] = acum
        todo = np.concatenate(acum["todo"]) if acum["todo"] else np.zeros(0)
        stats.append([ds, "todo", "", len(todo), round(float(todo.mean()), 4) if len(todo) else "", round(float((todo > 0).mean()), 4) if len(todo) else "", seed, raw.get("iteration")])
        for nombre, tr in (("hfs", TRAMOS_HFS), ("avance", TRAMOS_AVANCE)):
            for t in tr:
                v = np.concatenate(acum[nombre][t]) if acum[nombre][t] else np.zeros(0)
                stats.append([ds, nombre, f"{t[0]:g} a {t[1]:g}" if t[1] < 10 ** 8 else f"{t[0]:g} o mas", len(v), round(float(v.mean()), 4) if len(v) else "", round(float((v > 0).mean()), 4) if len(v) else "", seed, raw.get("iteration")])
        print(f"[{ds}] alpha medio {float(todo.mean()) if len(todo) else 0:.4f} en {len(todo)} nodos de {len(grids)} mapas", flush=True)
    write_csv(root / "figuras" / "alpha_mapas.csv", ["dataset", "grid", "seed", "iteracion", "exp", "optimo", "camino", "nodos", "alpha_medio", "frac_alpha_pos", "alpha_max"], filas)
    write_csv(root / "figuras" / "alpha_stats.csv", ["dataset", "tramo_de", "tramo", "nodos", "alpha_medio", "frac_alpha_pos", "seed", "iteracion"], stats)
    if por_ds:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        for ds, col in zip(DATASETS, ["C0", "C1", "C2"]):
            if ds not in por_ds:
                continue
            for ax, nombre, tr in ((axes[0], "hfs", TRAMOS_HFS), (axes[1], "avance", TRAMOS_AVANCE)):
                ys = [float(np.concatenate(por_ds[ds][nombre][t]).mean()) if por_ds[ds][nombre][t] and len(np.concatenate(por_ds[ds][nombre][t])) else np.nan for t in tr]
                ax.plot(range(len(tr)), ys, marker="o", color=col, label=f"{ds}, semilla {semillas[ds]}")
        axes[0].set_xticks(range(len(TRAMOS_HFS))); axes[0].set_xticklabels([f"{t[0]:g} a {t[1]:g}" if t[1] < 10 ** 8 else f"{t[0]:g} o mas" for t in TRAMOS_HFS])
        axes[0].set_xlabel("h_fs del nodo en celdas"); axes[0].set_ylabel("alpha medio"); axes[0].set_title("alpha de la compuerta por tramo de h_fs")
        axes[1].set_xticks(range(len(TRAMOS_AVANCE))); axes[1].set_xticklabels([f"{int(100 * t[0])} a {int(100 * min(t[1], 1))} %" for t in TRAMOS_AVANCE])
        axes[1].set_xlabel("avance de la busqueda"); axes[1].set_ylabel("alpha medio"); axes[1].set_title("alpha de la compuerta por tramo de avance")
        for ax in axes:
            ax.grid(alpha=0.3); ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(root / "figuras" / "alpha_tramos.png", dpi=120)
        plt.close(fig)
    print(f"-> {root / 'figuras'}", flush=True)


if __name__ == "__main__":
    main()
