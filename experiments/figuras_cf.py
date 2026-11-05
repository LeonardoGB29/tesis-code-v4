# figuras del experimento cf: expansiones contra camino por modelo, y cf predicho contra cf* en 3 mapas de prueba por conjunto
#   python -m experiments.figuras_cf --root results/exp_cf [--refs results/exp_cf] [--modelo cf_global_suave --seed 0 --n 3]
# los mapas de cf usan por defecto cf_global_suave (la red de cf_global en la epoca elegida con suavizado), porque best.pt de cf_global es la epoca 0 (cf constante)
import argparse
from pathlib import Path

import matplotlib
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.common import DATASETS, ESCALA, MAX_EXP, grids_de, hstar_mapa, leer_csv, load_problem, set_threads, write_csv
from experiments.resumen_cf import MODELOS, NOMBRES, corridas
from experiments.suavizado_cf import SIGMA, EvaluadorSuave
from src.cf.busqueda import EvaluadorCF, EvaluadorGNN, voraz
from src.cf.fotos import LADO, canales, etiqueta, hfs_entero
from src.cf.red import cargar


def exp_vs_camino(root, ev):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    # marcador y color fijos por modelo, asi no se repiten colores al dar la vuelta al ciclo
    marcas = {"voraz_hstar": ("*", "C1"), "voraz_hfs": ("s", "C3"), "phil": ("^", "C5"), "phil_res_s": ("v", "C7"), "cf_global": ("o", "C0"), "cf_local": ("D", "C2"),
              "cf_global_gnn": ("P", "C4"), "cf_global_dagger": ("X", "C6"), "cf_global_suave": ("h", "C8"), "cf_global_dagger_suave": ("8", "C9"),
              "cf_filtro": ("<", "crimson"), "cf_grueso": (">", "darkcyan")}
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
        ax.set_title(ds); ax.set_xlabel("expansiones (media de 100 mapas)"); ax.set_ylabel("camino sobre el optimo"); ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    (root / "figuras").mkdir(exist_ok=True)
    fig.savefig(root / "figuras" / "exp_vs_camino.png", dpi=120)
    plt.close(fig)


def mapas_cf(root, modelo, seed, n):
    filas = []
    for ds in DATASETS:
        ck = root / MODELOS[modelo][0] / "runs" / ds / f"{modelo}_s{seed}" / "best.pt"
        if not ck.exists():
            print(f"[{ds}] sin checkpoint {ck}"); continue
        model, cfg, raw = cargar(ck)
        externo = SIGMA[ds] if modelo in ("cf_global_suave", "cf_global_dagger_suave") else 0.0     # gauss por fuera del modelo (exp_cf)
        sigma = externo or (cfg.sigma if cfg.salida == "filtro" else 0.0)
        con_crudo = bool(externo) or cfg.salida == "filtro"                                       # hay un cf de la red antes de suavizar
        grids = grids_de(ds, "test")[:n]
        ncol = 5 if con_crudo else 4
        fig, axes = plt.subplots(len(grids), ncol, figsize=(5 * ncol, 5 * len(grids)), squeeze=False)
        for fila, gp in zip(axes, grids):
            data, adj, v_s, v_g, h_star = load_problem(gp)
            if cfg.tipo == "gnn":
                ev = EvaluadorGNN(data, v_g, model, adj, cfg.every_cf, cfg.every_fs, ESCALA)
            elif externo:
                ev = EvaluadorSuave(data, v_g, model, sigma=externo, every_cf=cfg.every_cf, every_fs=cfg.every_fs, escala=ESCALA, device="cpu")
            else:
                ev = EvaluadorCF(data, v_g, model, cfg.every_cf, cfg.every_fs, ESCALA)
            res = voraz(ev, data, v_s, v_g, adj, max_exp=MAX_EXP)
            # foto al final de la busqueda: lo descubierto, cf predicho y cf*
            hfs = hfs_entero(ev.fs.h)
            x = canales(ev.desc.visto, ev.desc.real, ev.goal, hfs, ESCALA)
            with torch.no_grad():
                xt = torch.from_numpy(x)[None]
                pred, crudo = None, None
                if cfg.tipo != "gnn":
                    pred = model(xt)[0].numpy()                                        # el cf que usa la busqueda (en exp_cf2 ya sale grueso del modelo)
                    if externo:
                        crudo, pred = pred, np.clip(gaussian_filter(pred, externo, mode="nearest"), 0.001, 1.0)
                    elif cfg.salida == "filtro":
                        crudo = model.cabeza(model.caracteristicas(xt))[0].numpy()
            y, m = etiqueta(hfs, hstar_mapa(data, h_star), ev.desc.real, ev.goal)
            H, W = ev.desc.real.shape
            rc = data.x.long().numpy()
            estado = np.full((H, W), 0.5); estado[ev.desc.visto & ~ev.desc.real] = 1.0; estado[ev.desc.visto & ev.desc.real] = 0.0
            fila[0].imshow(estado, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            exp = np.zeros((H, W), bool)
            cl = np.array(sorted(res.closed)); exp[rc[cl, 0], rc[cl, 1]] = True
            fila[0].imshow(np.where(exp, 1.0, np.nan), cmap="autumn", alpha=0.5, vmin=0, vmax=1, interpolation="nearest")
            if res.path:
                p = rc[np.array(res.path)]; fila[0].plot(p[:, 1], p[:, 0], color="lime", lw=1)
            fila[0].set_title(f"{ds} {gp.name}\n{modelo} s{seed} epoca {raw.get('epoca')}{f' sigma {sigma:g}' if sigma else ''}: exp {res.n_expanded}, optimo {int(h_star[v_s])}, camino {len(res.path) - 1 if res.path else '-'}", fontsize=9)
            ys = np.where(m[:H, :W], y[:H, :W], np.nan)
            c_cf, c_err = (3, 4) if con_crudo else (2, 3)
            fila[c_cf].imshow(ys, cmap="viridis", vmin=0, vmax=1, interpolation="nearest"); fila[c_cf].set_title("cf* = h_fs / h* (mapa real)", fontsize=9)
            if pred is not None:
                libre = ~ev.desc.real
                usado = pred                                                    # el cf que divide a h_fs en la busqueda
                red = crudo if crudo is not None else usado
                if con_crudo:
                    im = fila[1].imshow(np.where(libre, crudo[:H, :W], np.nan), cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
                    fila[1].set_title("cf de la red antes de suavizar (sobre lo descubierto)", fontsize=9)
                    fig.colorbar(im, ax=fila[1], fraction=0.046)
                    im_s = fila[2].imshow(np.where(libre, usado[:H, :W], np.nan), cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
                    fila[2].set_title(f"cf suavizado (gauss sigma {sigma:g}{', dentro del modelo' if not externo else ''}), el que usa la busqueda", fontsize=9)
                    fig.colorbar(im_s, ax=fila[2], fraction=0.046)
                else:
                    im = fila[1].imshow(np.where(libre, usado[:H, :W], np.nan), cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
                    fila[1].set_title("cf predicho por la red, el que usa la busqueda" + (" (cabeza 26x26 interpolada)" if cfg.salida == "gruesa" else ""), fontsize=9)
                    fig.colorbar(im, ax=fila[1], fraction=0.046)
                err = np.where(m[:H, :W], usado[:H, :W] - y[:H, :W], np.nan)
                im2 = fila[c_err].imshow(err, cmap="RdBu_r", vmin=-0.5, vmax=0.5, interpolation="nearest")
                e = err[~np.isnan(err)]
                e_red = (red[:H, :W] - y[:H, :W])[m[:H, :W]]
                fila[c_err].set_title(f"cf usado menos cf*: media {e.mean():+.3f}, abs {np.abs(e).mean():.3f}" + (f"\n(red sin suavizar: abs {np.abs(e_red).mean():.3f})" if con_crudo else ""), fontsize=9)
                fig.colorbar(im2, ax=fila[c_err], fraction=0.046)
                filas.append([ds, gp.name, modelo, seed, raw.get("epoca"), sigma, res.n_expanded, int(h_star[v_s]), len(res.path) - 1 if res.path else "",
                              int(ev.desc.visto.sum()), int(m[:H, :W].sum()), round(float(e_red.mean()), 4), round(float(np.abs(e_red).mean()), 4),
                              round(float(np.corrcoef(red[:H, :W][m[:H, :W]], y[:H, :W][m[:H, :W]])[0, 1]), 4),
                              round(float(np.abs(e).mean()), 4), round(float(np.corrcoef(usado[:H, :W][m[:H, :W]], y[:H, :W][m[:H, :W]])[0, 1]), 4)])
            for ax in fila:
                ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(root / "figuras" / f"cf_{ds}.png", dpi=110)
        plt.close(fig)
        print(f"[{ds}] figura lista", flush=True)
    write_csv(root / "figuras" / "cf_stats.csv", ["dataset", "grid", "modelo", "seed", "epoca", "sigma", "exp", "optimo", "camino", "celdas_vistas", "celdas_etiquetadas",
                                                   "error_medio_red", "error_abs_red", "correlacion_red", "error_abs_usado", "correlacion_usado"], filas)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results/exp_cf")
    p.add_argument("--refs", default=None)
    p.add_argument("--modelo", default="cf_global_suave")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n", type=int, default=3)
    a = p.parse_args()
    set_threads(1)
    root = Path(a.root)
    (root / "figuras").mkdir(exist_ok=True)
    exp_vs_camino(root, corridas(root, Path(a.refs) if a.refs else None, Path(a.refs2) if a.refs2 else None))
    mapas_cf(root, a.modelo, a.seed, a.n)


if __name__ == "__main__":
    main()
