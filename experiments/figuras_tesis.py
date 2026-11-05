# figuras para la tesis final: mapas de los tres conjuntos, una foto del mapa descubierto con su etiqueta, y cf_filtro con leyendas claras
# tambien los cinco canales de la entrada, los paneles del pipeline y las medidas de los conjuntos
#   python -m experiments.figuras_tesis --out "../doc_tesis/final_doc/figuras"
#   python -m experiments.figuras_tesis --out "../doc_tesis/final_doc/figuras" --figs medidas canales pipeline
import argparse
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D

from experiments.common import DATASETS, ESCALA, MAX_EXP, grids_de, hstar_mapa, load_problem, md_tabla, set_threads
from src.cf.busqueda import EvaluadorCF, voraz
from src.cf.fotos import H_TOPE, canales, etiqueta, hfs_entero
from src.cf.red import cargar
from src.phil.data import grid_to_graph, load_grid, sample_start_goal
from src.search.oracle import bfs_oracle

FOTOS = Path("results/exp_cf/fotos")
CK = Path("results/exp_cf2/cf_filtro/runs")
LADO_REAL = 201
# estado de la celda: 0 obstaculo visto, 1 no visto, 2 libre visto
CMAP_ESTADO = ListedColormap(["black", "#bdbdbd", "white"])


def visto_de(fila):
    return np.unpackbits(fila)[:LADO_REAL * LADO_REAL].reshape(LADO_REAL, LADO_REAL).astype(bool)


def ejes(ax, x=True, y=True):
    ax.set_xticks([0, 100, 200]); ax.set_yticks([0, 100, 200])
    ax.tick_params(labelsize=8)
    if x: ax.set_xlabel("columna de la grilla", fontsize=9)
    if y: ax.set_ylabel("fila de la grilla", fontsize=9)


def figura_mapas(out):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    for ax, ds in zip(axes, DATASETS):
        d = FOTOS / ds / "hfs"
        real = np.load(d / "real.npy", mmap_mode="r")[0]
        s = np.load(d / "start.npy")[0]; g = np.load(d / "goal.npy")[0]
        ax.imshow(np.where(real, 0, 2), cmap=CMAP_ESTADO, vmin=0, vmax=2, interpolation="nearest")
        ax.plot(s[1], s[0], "o", color="tab:blue", ms=9, mec="k", clip_on=False, zorder=5)
        ax.plot(g[1], g[0], "*", color="tab:red", ms=14, mec="k", clip_on=False, zorder=5)
        ax.set_title(ds.replace("_", r"\_") if False else ds, fontsize=11)
        ejes(ax, y=ax is axes[0])
    h = [Patch(facecolor="white", edgecolor="k", label="celda libre"), Patch(facecolor="black", label="obstáculo"),
         Line2D([], [], marker="o", ls="", color="tab:blue", mec="k", ms=8, label="inicio"),
         Line2D([], [], marker="*", ls="", color="tab:red", mec="k", ms=12, label="meta")]
    fig.legend(handles=h, loc="lower center", ncol=4, fontsize=10, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out / "datos_mapas.png", dpi=150)
    plt.close(fig)


def foto_media(ds):
    # una foto a mitad de busqueda del primer mapa: la del medio de sus fotos
    d = FOTOS / ds / "hfs"
    idx = np.load(d / "mapa_idx.npy")
    sel = np.where(idx == idx[0])[0]
    i = sel[len(sel) // 2]
    m = idx[i]
    real = np.array(np.load(d / "real.npy", mmap_mode="r")[m])
    hstar = np.load(d / "hstar.npy", mmap_mode="r")[m].astype(float)
    hfs = np.array(np.load(d / "hfs.npy", mmap_mode="r")[i])
    g = np.load(d / "goal.npy")[m]; s = np.load(d / "start.npy")[m]
    v = visto_de(np.load(d / "visto.npy", mmap_mode="r")[i])
    return i, m, real, hstar, hfs, g, s, v


def figura_foto(out, ds="gaps_and_forest"):
    i, m, real, hstar, hfs, g, s, v = foto_media(ds)
    hfs = hfs.astype(float)
    estado = np.ones(real.shape); estado[v & ~real] = 2; estado[v & real] = 0
    libre = (~real) & (hstar >= 0) & (hstar < 1e6)
    hs = np.where((hfs >= 0) & ~(v & real), hfs, np.nan)
    hr = np.where(libre, hstar, np.nan)
    cf = np.where(libre & (hstar > 0), hfs / np.maximum(hstar, 1), np.nan)
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6))
    axes[0].imshow(estado, cmap=CMAP_ESTADO, vmin=0, vmax=2, interpolation="nearest")
    axes[0].plot(s[1], s[0], "o", color="tab:blue", ms=8, mec="k", clip_on=False, zorder=5); axes[0].plot(g[1], g[0], "*", color="tab:red", ms=13, mec="k", clip_on=False, zorder=5)
    axes[0].set_title("(a) mapa descubierto, lo que ve el agente", fontsize=10)
    vmax = np.nanmax(hr)
    im = axes[1].imshow(hs, cmap="magma", vmin=0, vmax=vmax, interpolation="nearest"); axes[1].set_title(r"(b) $h_{fs}$ sobre el mapa supuesto, lo no visto libre", fontsize=10)
    c = fig.colorbar(im, ax=axes[1], fraction=0.046); c.set_label("pasos hasta la meta", fontsize=9)
    im = axes[2].imshow(hr, cmap="magma", vmin=0, vmax=vmax, interpolation="nearest"); axes[2].set_title(r"(c) $h^{*}$, distancia real, solo para etiquetar", fontsize=10)
    c = fig.colorbar(im, ax=axes[2], fraction=0.046); c.set_label("pasos hasta la meta", fontsize=9)
    im = axes[3].imshow(cf, cmap="viridis", vmin=0, vmax=1, interpolation="nearest"); axes[3].set_title(r"(d) etiqueta $cf^{*} = h_{fs} / h^{*}$", fontsize=10)
    c = fig.colorbar(im, ax=axes[3], fraction=0.046); c.set_label("factor de corrección", fontsize=9)
    for k, ax in enumerate(axes):
        ejes(ax, y=k == 0)
    h = [Patch(facecolor="white", edgecolor="k", label="libre visto"), Patch(facecolor="black", label="obstáculo visto"),
         Patch(facecolor="#bdbdbd", label="no visto"),
         Line2D([], [], marker="o", ls="", color="tab:blue", mec="k", ms=8, label="inicio"),
         Line2D([], [], marker="*", ls="", color="tab:red", mec="k", ms=12, label="meta")]
    fig.legend(handles=h, loc="lower center", ncol=5, fontsize=10, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out / "datos_foto.png", dpi=150)
    plt.close(fig)
    print("foto", ds, "mapa", m, "foto", i, "vistas", int(v.sum()))


def figura_canales(out, ds="gaps_and_forest"):
    # los cinco canales de la misma foto de datos_foto, en el lado de 208 con el relleno
    i, m, real, hstar, hfs, g, s, v = foto_media(ds)
    x = canales(v, real, g, hfs, ESCALA)
    titulos = ["(a) canal 1, libre visto", "(b) canal 2, obstáculo visto", "(c) canal 3, no visto", "(d) canal 4, meta",
               r"(e) canal 5, $h_{fs}$ dividida por 200"]
    # dos filas para que se lea a lo ancho de la pagina; el sexto lugar lleva la leyenda
    fig, axs = plt.subplots(2, 3, figsize=(9.6, 6.9))
    axes = axs.ravel()
    for k, ax in enumerate(axes[:5]):
        if k < 4:
            ax.imshow(x[k], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        else:
            im = ax.imshow(x[4], cmap="magma", vmin=0, vmax=H_TOPE, interpolation="nearest")
            c = fig.colorbar(im, ax=ax, fraction=0.046); c.set_label(r"$h_{fs}$ / 200, recortada en 2", fontsize=9)
            c.ax.tick_params(labelsize=8)
        ax.axhline(LADO_REAL - 0.5, color="tab:cyan", ls="--", lw=1.3); ax.axvline(LADO_REAL - 0.5, color="tab:cyan", ls="--", lw=1.3)
        ax.set_title(titulos[k], fontsize=10)
        ejes(ax, x=k >= 2, y=k in (0, 3))
    axes[3].plot(g[1], g[0], "o", ms=16, mfc="none", mec="tab:red", mew=2, clip_on=False, zorder=5)
    axes[5].set_axis_off()
    h = [Patch(facecolor="white", edgecolor="k", label="valor 1"), Patch(facecolor="black", label="valor 0"),
         Line2D([], [], color="tab:cyan", ls="--", lw=1.5, label="borde del mapa de 201,\nel resto es relleno hasta 208"),
         Line2D([], [], marker="o", ls="", ms=11, mfc="none", mec="tab:red", mew=2, label="celda de la meta,\nmarcada para que se vea")]
    axes[5].legend(handles=h, loc="center", fontsize=9, frameon=False, labelspacing=1.1)
    fig.tight_layout()
    fig.savefig(out / "datos_canales.png", dpi=170)
    plt.close(fig)
    print("canales", ds, "mapa", m, "foto", i)


def panel(img, cmap, vmin=None, vmax=None, borde=True):
    # imagen sin ejes para el pipeline, con margen para que inicio y meta no se corten
    fig = plt.figure(figsize=(2.4, 2.4))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    cm = plt.get_cmap(cmap).copy() if isinstance(cmap, str) else cmap
    cm.set_bad("white")
    n = img.shape[0]
    ax.imshow(img, cmap=cm, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xlim(-7, n + 6); ax.set_ylim(n + 6, -7)
    if borde:
        ax.add_patch(Rectangle((-0.5, -0.5), n, n, fill=False, ec="k", lw=0.6))
    return fig, ax


def guardar(fig, path):
    fig.savefig(path, dpi=200)
    plt.close(fig)


def marcas(ax, s, g, ms=5):
    ax.plot(s[1], s[0], "o", color="tab:blue", ms=ms, mec="k", mew=0.6, clip_on=False, zorder=5)
    ax.plot(g[1], g[0], "*", color="tab:red", ms=ms * 1.6, mec="k", mew=0.6, clip_on=False, zorder=5)


def panel_estado(path, desc, s, g, closed=None, rc=None, camino=None):
    estado = np.ones(desc.real.shape); estado[desc.visto & ~desc.real] = 2; estado[desc.visto & desc.real] = 0
    fig, ax = panel(estado, CMAP_ESTADO, 0, 2)
    if closed is not None:
        exp = np.zeros(desc.real.shape, bool); cl = np.array(sorted(closed)); exp[rc[cl, 0], rc[cl, 1]] = True
        ax.imshow(np.where(exp, 1.0, np.nan), cmap=ListedColormap(["orange"]), alpha=0.8, interpolation="nearest")
    if camino is not None:
        p = rc[np.array(camino)]; ax.plot(p[:, 1], p[:, 0], color="green", lw=1.6)
    marcas(ax, s, g)
    guardar(fig, path)


def figura_pipeline(out, ds="gaps_and_forest", seed=0, mapa=0):
    # paneles del pipeline: un mapa de prueba con cf_filtro, estado a mitad de la busqueda, busqueda completa y entrenamiento
    d = out / "pipeline"; d.mkdir(parents=True, exist_ok=True)
    for c in DATASETS:
        gp = grids_de(c, "test")[mapa if c == ds else 0]
        data = grid_to_graph(load_grid(gp))
        v_s, v_g = sample_start_goal(data)
        rc = data.x.long().numpy()
        fig, ax = panel(np.where(load_grid(gp), 0, 2), CMAP_ESTADO, 0, 2)
        marcas(ax, rc[v_s], rc[v_g])
        guardar(fig, d / f"conjunto_{c}.png")
    model, cfg, raw = cargar(CK / ds / f"cf_filtro_s{seed}" / "best.pt")
    gp = grids_de(ds, "test")[mapa]
    data, adj, v_s, v_g, h_star = load_problem(gp)
    rc = data.x.long().numpy(); s, g = rc[v_s], rc[v_g]
    nuevo = lambda: EvaluadorCF(data, v_g, model, cfg.every_cf, cfg.every_fs, ESCALA)
    ev = nuevo(); res = voraz(ev, data, v_s, v_g, adj, max_exp=MAX_EXP)
    panel_estado(d / "busqueda.png", ev.desc, s, g, res.closed, rc, res.path)
    n = res.n_expanded
    for k, t in enumerate([n // 4, n // 2, 3 * n // 4]):
        e = nuevo(); r = voraz(e, data, v_s, v_g, adj, max_exp=t)
        panel_estado(d / f"propia_{k}.png", e.desc, s, g, r.closed, rc)
    # estado a mitad de la busqueda
    ev = nuevo(); r = voraz(ev, data, v_s, v_g, adj, max_exp=n // 2)
    panel_estado(d / "descubierto.png", ev.desc, s, g, r.closed, rc)
    hfs = hfs_entero(ev.fs.h)
    x = canales(ev.desc.visto, ev.desc.real, ev.goal, hfs, ESCALA)
    with torch.no_grad():
        xt = torch.from_numpy(x)[None]
        usado = model(xt)[0].numpy()[:LADO_REAL, :LADO_REAL]
        crudo = model.cabeza(model.caracteristicas(xt))[0].numpy()[:LADO_REAL, :LADO_REAL]
    hs_mapa = hstar_mapa(data, h_star)
    alc = hs_mapa >= 0
    vmax = float(hs_mapa[alc].max())
    obst = ev.desc.visto & ev.desc.real
    hf = np.where((hfs >= 0) & ~obst, hfs, np.nan).astype(float)
    guardar(panel(hf, "magma", 0, vmax)[0], d / "hfs.png")
    for k in range(5):
        fig, ax = panel(x[k], "gray" if k < 4 else "magma", 0, 1 if k < 4 else H_TOPE, borde=False)
        if k == 3:
            ax.plot(g[1], g[0], "o", ms=14, mfc="none", mec="tab:red", mew=2, clip_on=False)
        guardar(fig, d / f"canal_{k}.png")
    guardar(panel(np.where(obst, np.nan, crudo), "viridis", 0, 1)[0], d / "cf_red.png")
    guardar(panel(np.where(obst, np.nan, usado), "viridis", 0, 1)[0], d / "cf.png")
    guardar(panel(np.where(obst, np.nan, hf / usado), "magma", 0, vmax)[0], d / "hhat.png")
    guardar(panel(np.where(alc, hs_mapa, np.nan).astype(float), "magma", 0, vmax)[0], d / "hstar.png")
    y, msk = etiqueta(hfs, hs_mapa, ev.desc.real, ev.goal)
    guardar(panel(np.where(msk, y, np.nan)[:LADO_REAL, :LADO_REAL], "viridis", 0, 1)[0], d / "cfstar.png")
    # la leyenda y las barras de color van en tikz; h* max es el tope de la barra de pasos
    print("pipeline", gp.stem, "expansiones", n, "mitad", n // 2, "h* max", vmax, "camino", len(res.path) - 1 if res.path else None)


def medidas_conjuntos():
    # medidas de los 370 mapas usados por conjunto: obstaculos, camino optimo entre esquinas y resolubles por particion
    filas = []
    for ds in DATASETS:
        dens, opt, res = [], [], {}
        for split in ("train", "validation", "test"):
            gs, ok = grids_de(ds, split), 0
            for gp in gs:
                grid = load_grid(gp)
                dens.append(grid.mean())
                data = grid_to_graph(grid)
                v_s, v_g = sample_start_goal(data)
                dist = int(bfs_oracle(data, v_g)[v_s])
                if dist >= 0:
                    ok += 1; opt.append(dist)
            res[split] = f"{ok}/{len(gs)}"
        filas.append([ds, f"{100 * np.mean(dens):.1f}", f"{100 * np.std(dens):.1f}", f"{np.mean(opt):.0f}", f"{np.std(opt):.0f}",
                      f"{min(opt)} a {max(opt)}", res["train"], res["validation"], res["test"]])
    print(md_tabla(["conjunto", "obstaculos %", "desvio", "camino optimo", "desvio", "rango", "resolubles train", "validation", "test"], filas))


def figura_cf(out, ds="gaps_and_forest", seed=0, n=2):
    model, cfg, raw = cargar(CK / ds / f"cf_filtro_s{seed}" / "best.pt")
    grids = grids_de(ds, "test")[:n]
    fig, axes = plt.subplots(n, 5, figsize=(24, 5.0 * n + 0.8), squeeze=False)
    titulos = ["(a) búsqueda con el corrector", "(b) factor de la red antes de suavizar", fr"(c) factor suavizado ($\sigma$ = {cfg.sigma:g}), el que usa la búsqueda",
               r"(d) etiqueta $cf^{*}$ calculada con el mapa real", "(e) error, factor usado menos etiqueta"]
    for r, (fila, gp) in enumerate(zip(axes, grids)):
        data, adj, v_s, v_g, h_star = load_problem(gp)
        ev = EvaluadorCF(data, v_g, model, cfg.every_cf, cfg.every_fs, ESCALA)
        res = voraz(ev, data, v_s, v_g, adj, max_exp=MAX_EXP)
        hfs = hfs_entero(ev.fs.h)
        x = canales(ev.desc.visto, ev.desc.real, ev.goal, hfs, ESCALA)
        with torch.no_grad():
            xt = torch.from_numpy(x)[None]
            usado = model(xt)[0].numpy()
            crudo = model.cabeza(model.caracteristicas(xt))[0].numpy()
        y, msk = etiqueta(hfs, hstar_mapa(data, h_star), ev.desc.real, ev.goal)
        H, W = ev.desc.real.shape
        rc = data.x.long().numpy()
        estado = np.ones((H, W)); estado[ev.desc.visto & ~ev.desc.real] = 2; estado[ev.desc.visto & ev.desc.real] = 0
        fila[0].imshow(estado, cmap=CMAP_ESTADO, vmin=0, vmax=2, interpolation="nearest")
        exp = np.zeros((H, W), bool); cl = np.array(sorted(res.closed)); exp[rc[cl, 0], rc[cl, 1]] = True
        fila[0].imshow(np.where(exp, 1.0, np.nan), cmap=ListedColormap(["orange"]), alpha=0.7, interpolation="nearest")
        if res.path:
            p = rc[np.array(res.path)]; fila[0].plot(p[:, 1], p[:, 0], color="green", lw=2)
        libre = ~ev.desc.real
        for k, (img, cm, lo, hi, lab) in enumerate([(np.where(libre, crudo[:H, :W], np.nan), "viridis", 0, 1, "factor de corrección cf"),
                                                    (np.where(libre, usado[:H, :W], np.nan), "viridis", 0, 1, "factor de corrección cf"),
                                                    (np.where(msk[:H, :W], y[:H, :W], np.nan), "viridis", 0, 1, r"factor exacto $cf^{*}$"),
                                                    (np.where(msk[:H, :W], usado[:H, :W] - y[:H, :W], np.nan), "RdBu_r", -0.5, 0.5, "diferencia")], start=1):
            im = fila[k].imshow(img, cmap=cm, vmin=lo, vmax=hi, interpolation="nearest")
            cb = fig.colorbar(im, ax=fila[k], fraction=0.046); cb.set_label(lab, fontsize=10)
        e = (usado[:H, :W] - y[:H, :W])[msk[:H, :W]]; e_red = (crudo[:H, :W] - y[:H, :W])[msk[:H, :W]]
        for k, ax in enumerate(fila):
            ejes(ax, x=r == n - 1, y=k == 0)
            if r == 0:
                ax.set_title(titulos[k], fontsize=12)
        fila[0].set_ylabel(f"mapa {gp.stem}\n{res.n_expanded} expansiones, óptimo {int(h_star[v_s])}, camino {len(res.path) - 1}\n\nfila de la grilla", fontsize=10)
        fila[4].text(0.02, 0.02, f"error absoluto medio {np.abs(e).mean():.3f}\n(sin suavizar {np.abs(e_red).mean():.3f})", transform=fila[4].transAxes,
                     fontsize=9, va="bottom", bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    h = [Patch(facecolor="white", edgecolor="k", label="libre visto"), Patch(facecolor="black", label="obstáculo visto"), Patch(facecolor="#bdbdbd", label="no visto"),
         Patch(facecolor="orange", label="nodo expandido"), Line2D([], [], color="green", lw=2, label="camino encontrado"),
         Patch(facecolor="white", edgecolor="none", label="blanco en (b) a (e), obstáculo o fuera de la etiqueta")]
    fig.legend(handles=h, loc="lower center", ncol=6, fontsize=11, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "cf_gaps_tesis.png", dpi=110)
    plt.close(fig)


FIGS = {"mapas": figura_mapas, "foto": figura_foto, "cf": figura_cf, "canales": figura_canales}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--figs", nargs="+", default=["mapas", "foto", "cf"], choices=list(FIGS) + ["pipeline", "medidas"])
    p.add_argument("--mapa", type=int, default=0)      # mapa de prueba de gaps del pipeline
    a = p.parse_args()
    set_threads(1)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for k in a.figs:
        if k == "medidas":
            medidas_conjuntos()
        elif k == "pipeline":
            figura_pipeline(out, mapa=a.mapa)
        else:
            FIGS[k](out)


if __name__ == "__main__":
    main()
