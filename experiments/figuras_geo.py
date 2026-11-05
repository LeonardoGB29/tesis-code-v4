# figuras de exp_geo. ocupacion predicha contra real en 3 mapas de prueba por conjunto, barrido de lambda, calibracion y expansiones contra camino
#   python -m experiments.figuras_geo --root results/exp_geo [--seed 0 --n 3]
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.common import DATASETS, ESCALA, MAX_EXP, RESULTS_GEO, grids_de, leer_csv, load_problem, set_threads, write_csv
from experiments.resumen_geo import NOMBRES, corridas
from src.cf.busqueda import voraz
from src.geo.hgeo import EvaluadorGeo
from src.geo.metricas import Acumulador
from src.geo.red import cargar_geo


def exp_vs_camino(root, ev):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    marcas = {"voraz_hstar": ("*", "C1"), "voraz_hfs": ("s", "C3"), "cf_filtro": ("<", "crimson"), "geo": ("o", "C0"), "geo_real": ("D", "C2")}
    for ax, ds in zip(axes, DATASETS):
        for m, (mk, col) in marcas.items():
            evs = ev.get((ds, m), {})
            if not evs:
                continue
            xs, ys = [], []
            for s, e in evs.items():
                g = list(e)
                xs.append(np.mean([e[x]["exp"] for x in g]))
                c = [e[x]["camino"] / max(e[x]["opt"], 1) for x in g if e[x]["enc"]]
                ys.append(np.mean(c) if c else np.nan)
            ax.scatter(xs, ys, marker=mk, s=25, alpha=0.4, color=col)
            ax.scatter([np.mean(xs)], [np.nanmean(ys)], marker=mk, s=120, label=NOMBRES[m], edgecolor="k", color=col)
        ax.set_title(ds); ax.set_xlabel("expansiones (media de 100 mapas)"); ax.set_ylabel("camino sobre el optimo"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root / "figuras" / "exp_vs_camino.png", dpi=120)
    plt.close(fig)


def barrido(root):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for j, ds in enumerate(DATASETS):
        p = root / "barrido" / ds / "lambda.json"
        if not p.exists():
            continue
        L = json.loads(p.read_text())
        ref = L["ref_val_hfs"]
        for quien, col in (("geo_s0", "C0"), ("real", "C2")):
            filas = sorted([r for r in L["tabla"] if r["quien"] == quien], key=lambda r: r["lambda"])
            if not filas:
                continue
            xs = [r["lambda"] for r in filas]
            axes[0, j].plot(range(len(xs)), [r["exp"] for r in filas], marker="o", color=col, label="cnn semilla 0" if quien == "geo_s0" else "ocupacion real")
            axes[1, j].plot(range(len(xs)), [r["camino"] for r in filas], marker="o", color=col, label="cnn semilla 0" if quien == "geo_s0" else "ocupacion real")
            for ax in axes[:, j]:
                ax.set_xticks(range(len(xs))); ax.set_xticklabels([f"{x:g}" for x in xs])
        axes[0, j].axhline(ref["exp"], color="gray", ls="--", label="voraz con h_fs")
        axes[1, j].axhline(ref["camino"], color="gray", ls="--", label="voraz con h_fs")
        axes[1, j].axhline(1.05 * ref["camino"], color="gray", ls=":", label="camino5")
        k = [r["lambda"] for r in L["tabla"] if r["quien"] == "geo_s0"]
        if L["lambda"] in k:
            axes[0, j].axvline(sorted(k).index(L["lambda"]), color="C0", alpha=0.3, lw=6)
        axes[0, j].set_title(f"{ds}, expansiones de validacion, lambda elegido {L['lambda']:g}"); axes[1, j].set_title(f"{ds}, camino sobre el optimo")
        for ax in axes[:, j]:
            ax.set_xlabel("lambda"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root / "figuras" / "barrido_lambda.png", dpi=120)
    plt.close(fig)


def calibracion(root, seed):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for j, ds in enumerate(DATASETS):
        p = root / "metricas" / ds / f"calibracion_s{seed}.csv"
        m = root / "metricas" / ds / f"geo_s{seed}.csv"
        if not p.exists():
            continue
        rows = leer_csv(p)
        met = {(r["modelo"], r["estrato"]): r for r in leer_csv(m)} if m.exists() else {}
        for i, estrato in enumerate(("todo", "corredor")):
            ax = axes[i, j]
            for modelo, col in (("mejor", "C0"), ("previo", "C3")):
                fl = [r for r in rows if r["modelo"] == modelo and r["estrato"] == estrato]
                if not fl:
                    continue
                xs = [float(r["p_media"]) for r in fl]; ys = [float(r["frac_obst"]) for r in fl]; ns = np.array([int(r["n"]) for r in fl], float)
                mm = met.get((modelo, estrato), {})
                et = f"{'cnn mejor epoca' if modelo == 'mejor' else 'previo constante'}, bce {mm.get('bce', '')} ece {mm.get('ece', '')}"
                ax.scatter(xs, ys, s=20 + 300 * ns / ns.max(), color=col, alpha=0.7, label=et)
                ax.plot(xs, ys, color=col, alpha=0.5)
            ax.plot([0, 1], [0, 1], color="gray", ls="--")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xlabel("p predicha media del tramo"); ax.set_ylabel("fraccion real de obstaculos")
            ax.set_title(f"{ds}, {'todo lo no visto' if estrato == 'todo' else 'no visto cerca del camino optimo'}, semilla {seed}"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root / "figuras" / "calibracion.png", dpi=120)
    plt.close(fig)


def mapas(root, seed, n):
    filas = []
    for ds in DATASETS:
        ck = root / "runs" / ds / f"geo_s{seed}" / "best.pt"
        pl = root / "barrido" / ds / "lambda.json"
        if not ck.exists():
            print(f"[{ds}] sin checkpoint {ck}"); continue
        lam = json.loads(pl.read_text())["lambda"] if pl.exists() else 4.0
        model, cfg, raw = cargar_geo(ck)
        grids = grids_de(ds, "test")[:n]
        fig, axes = plt.subplots(len(grids), 4, figsize=(20, 5 * len(grids)), squeeze=False)
        for fila, gp in zip(axes, grids):
            data, adj, v_s, v_g, h_star = load_problem(gp)
            ev = EvaluadorGeo(data, v_g, model, lam, 20, 16, ESCALA)
            res = voraz(ev, data, v_s, v_g, adj, max_exp=MAX_EXP)
            visto, real = ev.desc.visto, ev.desc.real
            p = ev.calcular()
            H, W = real.shape
            rc = data.x.long().numpy()
            estado = np.full((H, W), 0.5); estado[visto & ~real] = 1.0; estado[visto & real] = 0.0
            fila[0].imshow(estado, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            exp = np.zeros((H, W), bool); cl = np.array(sorted(res.closed)); exp[rc[cl, 0], rc[cl, 1]] = True
            fila[0].imshow(np.where(exp, 1.0, np.nan), cmap="autumn", alpha=0.5, vmin=0, vmax=1, interpolation="nearest")
            if res.path:
                pth = rc[np.array(res.path)]; fila[0].plot(pth[:, 1], pth[:, 0], color="lime", lw=1)
            fila[0].set_title(f"{ds} {gp.name}\nh_geo s{seed} epoca {raw.get('epoca')} lambda {lam:g}, exp {res.n_expanded}, optimo {int(h_star[v_s])}, camino {len(res.path) - 1 if res.path else '-'}", fontsize=9)
            nv = ~visto
            acc = Acumulador(); acc.agregar(p[nv], real[nv]); r, _ = acc.resultados()
            im = fila[1].imshow(np.where(nv, p, np.nan), cmap="magma", vmin=0, vmax=1, interpolation="nearest")
            fila[1].imshow(np.where(visto, estado, np.nan), cmap="gray", vmin=0, vmax=1, interpolation="nearest", alpha=0.35)
            fila[1].set_title(f"p de obstaculo predicha en lo no visto, bce {r['bce']:.3f} auc {r['auc']:.3f}", fontsize=9); fig.colorbar(im, ax=fila[1], fraction=0.046)
            fila[2].imshow(np.where(nv, real.astype(float), np.nan), cmap="magma", vmin=0, vmax=1, interpolation="nearest")
            fila[2].imshow(np.where(visto, estado, np.nan), cmap="gray", vmin=0, vmax=1, interpolation="nearest", alpha=0.35)
            fila[2].set_title(f"ocupacion real en lo no visto, fraccion {real[nv].mean():.3f}, previo p0 {cfg.p0:.3f}", fontsize=9)
            im2 = fila[3].imshow(np.where(nv, p - real, np.nan), cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
            fila[3].set_title(f"p menos real, media {float((p - real)[nv].mean()):+.3f}, abs {float(np.abs(p - real)[nv].mean()):.3f}", fontsize=9); fig.colorbar(im2, ax=fila[3], fraction=0.046)
            for ax in fila:
                ax.set_xticks([]); ax.set_yticks([])
            filas.append([ds, gp.name, seed, raw.get("epoca"), lam, res.n_expanded, int(h_star[v_s]), len(res.path) - 1 if res.path else "", int(visto.sum()), int(nv.sum()),
                          round(r["frac_obst"], 4), round(r["bce"], 4), round(r["auc"], 4) if r["auc"] == r["auc"] else "", round(r["ece"], 4)])
        fig.tight_layout()
        fig.savefig(root / "figuras" / f"geo_{ds}.png", dpi=110)
        plt.close(fig)
        print(f"[{ds}] figura lista", flush=True)
    write_csv(root / "figuras" / "geo_stats.csv", ["dataset", "grid", "seed", "epoca", "lambda", "exp", "optimo", "camino", "celdas_vistas", "celdas_no_vistas", "frac_obst", "bce", "auc", "ece"], filas)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(RESULTS_GEO))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n", type=int, default=3)
    a = p.parse_args()
    set_threads(1)
    root = Path(a.root)
    (root / "figuras").mkdir(exist_ok=True)
    for nombre, fn in (("exp_vs_camino", lambda: exp_vs_camino(root, corridas(root))), ("barrido", lambda: barrido(root)), ("calibracion", lambda: calibracion(root, a.seed)),
                       ("mapas", lambda: mapas(root, a.seed, a.n))):
        try:
            fn()
            print(f"{nombre} lista", flush=True)
        except Exception as e:
            print(f"FALLO figura {nombre}, {e}", flush=True)


if __name__ == "__main__":
    main()
