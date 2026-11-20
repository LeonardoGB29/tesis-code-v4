# pipeline de la cnn local de 11x11 en una sola imagen, con miniaturas reales de un mapa de prueba
#   python -m new_exp.figura_pipeline --out ablations/cnn_local/pipeline.png
#   modulo 1 entrenamiento fuera de la busqueda, modulo 2 corrector dentro de la busqueda voraz
import argparse
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from experiments.common import ESCALA, MAX_EXP, grids_de, hstar_mapa, load_problem
from new_exp import redes
from src.cf import red
from src.cf.busqueda import Evaluador, EvaluadorCF, voraz
from src.cf.fotos import H_TOPE, canales, etiqueta, hfs_entero

LADO_REAL = 201
CMAP_ESTADO = ListedColormap(["black", "#bdbdbd", "white"])
AZUL, GRIS, NARANJA = "#2b5c8a", "#f3f5f8", "#e07b00"


def estado(desc):
    e = np.ones(desc.real.shape); e[desc.visto & ~desc.real] = 2; e[desc.visto & desc.real] = 0
    return e


def mini(fig, rect, img, cmap, vmin=None, vmax=None, titulo=None, s=None, g=None, closed=None, rc=None, camino=None):
    ax = fig.add_axes(rect); ax.set_xticks([]); ax.set_yticks([])
    cm = plt.get_cmap(cmap).copy() if isinstance(cmap, str) else cmap
    cm.set_bad("white")
    ax.imshow(img, cmap=cm, vmin=vmin, vmax=vmax, interpolation="nearest")
    if closed is not None:
        m = np.zeros(img.shape, bool); cl = np.array(sorted(closed)); m[rc[cl, 0], rc[cl, 1]] = True
        ax.imshow(np.where(m, 1.0, np.nan), cmap=ListedColormap([NARANJA]), alpha=0.85, interpolation="nearest")
    if camino is not None:
        p = rc[np.array(camino)]; ax.plot(p[:, 1], p[:, 0], color="tab:green", lw=1.4)
    if s is not None:
        ax.plot(s[1], s[0], "o", color="tab:blue", ms=8, mec="k", mew=0.7, clip_on=False, zorder=5)
        ax.plot(g[1], g[0], "*", color="tab:red", ms=13, mec="k", mew=0.7, clip_on=False, zorder=5)
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)
    if titulo:
        ax.set_title(titulo, fontsize=8, pad=3)
    return ax


def caja(fig, x, y, w, h, titulo, color=GRIS, ec="#9aa5b1", fs=9.5, letra=None):
    fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.008", transform=fig.transFigure,
                                      fc=color, ec=ec, lw=0.9, zorder=-1))
    t = f"{letra}  {titulo}" if letra else titulo
    fig.text(x + 0.006, y + h - 0.012, t, fontsize=fs, fontweight="bold", va="top", color="#1d2733")


def flecha(fig, a, b, color="#5b6670", estilo="-|>", rad=0.0, lw=1.3):
    fig.patches.append(FancyArrowPatch(a, b, transform=fig.transFigure, arrowstyle=estilo, mutation_scale=12, color=color, lw=lw,
                                       connectionstyle=f"arc3,rad={rad}"))


def texto(fig, x, y, s, fs=7.6, **k):
    fig.text(x, y, s, fontsize=fs, va="top", color="#26313d", linespacing=1.35, **k)


def red_local(fig, x, y, w, h):
    # 5 convoluciones 3x3 sin submuestreo, la imagen no se achica, y el campo de 11x11 de una celda
    ax = fig.add_axes([x, y, w, h]); ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.set_axis_off()
    for k in range(5):
        ax.add_patch(Rectangle((0.3 + 1.25 * k, 1.6), 0.9, 3.2, fc="#9cc3e6", ec=AZUL, lw=0.8))
        ax.text(0.75 + 1.25 * k, 3.2, f"3x3\n32", ha="center", va="center", fontsize=6.3)
    ax.add_patch(Rectangle((6.65, 1.6), 0.6, 3.2, fc="#c8dcb4", ec="#4d7a2a", lw=0.8))
    ax.text(6.95, 3.2, "1x1\n32", ha="center", va="center", fontsize=6.3)
    ax.add_patch(Rectangle((7.7, 1.6), 0.6, 3.2, fc="#f6d3a8", ec=NARANJA, lw=0.8))
    ax.text(8.0, 3.2, "1x1\n1", ha="center", va="center", fontsize=6.3)
    ax.text(4.0, 5.35, "conv, GroupNorm y LeakyReLU", ha="center", fontsize=6.6)
    ax.text(6.95, 5.35, "rasgos", ha="center", fontsize=6.6)
    ax.text(8.0, 5.35, "cabeza", ha="center", fontsize=6.6)
    ax.text(4.9, 0.75, "sin submuestreo, la salida tiene el tamaño de la entrada", ha="center", fontsize=6.3, color="#47525d")
    ax.text(4.9, 0.1, "cabeza en cero, cf = 0.5 en la época 0", ha="center", fontsize=6.3, color="#47525d")
    # campo receptivo
    for i in range(11):
        for j in range(11):
            ax.add_patch(Rectangle((8.75 + j * 0.105, 2.05 + i * 0.21), 0.105, 0.21, fc="#fde3c4" if (i, j) != (5, 5) else NARANJA,
                                   ec="white", lw=0.2))
    ax.text(9.33, 4.65, "campo\n11x11", ha="center", fontsize=6.3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="new_exp/results/cf_local_filtro/runs/multiple_bugtraps/cf_local_filtro_s0/best.pt")
    p.add_argument("--dataset", default="multiple_bugtraps")
    p.add_argument("--mapa", type=int, default=0)
    p.add_argument("--out", default="ablations/cnn_local/pipeline.png")
    a = p.parse_args()
    redes.instalar()
    torch.set_num_threads(1)
    model, cfg, raw = red.cargar(a.ckpt)
    gp = grids_de(a.dataset, "test")[a.mapa]
    data, adj, v_s, v_g, h_star = load_problem(gp)
    rc = data.x.long().numpy(); s, g = rc[v_s], rc[v_g]

    # busqueda con h_fs, de donde salen las fotos del entrenamiento
    base = Evaluador(data, v_g, cfg.every_fs, ESCALA); rb = voraz(base, data, v_s, v_g, adj, max_exp=MAX_EXP)
    eb = Evaluador(data, v_g, cfg.every_fs, ESCALA); rbm = voraz(eb, data, v_s, v_g, adj, max_exp=rb.n_expanded // 2)
    hs_mapa = hstar_mapa(data, h_star)
    hfs_b = hfs_entero(eb.fs.h)
    xb = canales(eb.desc.visto, eb.desc.real, eb.goal, hfs_b, ESCALA)
    y, msk = etiqueta(hfs_b, hs_mapa, eb.desc.real, eb.goal)

    # busqueda con el corrector
    nuevo = lambda: EvaluadorCF(data, v_g, model, cfg.every_cf, cfg.every_fs, ESCALA)
    ev = nuevo(); res = voraz(ev, data, v_s, v_g, adj, max_exp=MAX_EXP)
    em = nuevo(); rm = voraz(em, data, v_s, v_g, adj, max_exp=res.n_expanded // 2)
    hfs = hfs_entero(em.fs.h)
    x = canales(em.desc.visto, em.desc.real, em.goal, hfs, ESCALA)
    with torch.no_grad():
        xt = torch.from_numpy(x)[None]
        crudo = model.cabeza(model.caracteristicas(xt))[0].numpy()[:LADO_REAL, :LADO_REAL]
        cf = model(xt)[0].numpy()[:LADO_REAL, :LADO_REAL]
    obst = em.desc.visto & em.desc.real
    hf = np.where((hfs >= 0) & ~obst, hfs, np.nan).astype(float)
    vmax = float(hs_mapa[hs_mapa >= 0].max())
    R = slice(0, LADO_REAL)

    fig = plt.figure(figsize=(16, 9.6), dpi=170)
    fig.patch.set_facecolor("white")
    fig.text(0.5, 0.975, "Pipeline de la CNN local de 11x11 como corrector de h_fs", ha="center", va="top", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.945, f"Miniaturas reales de un mapa de prueba de {a.dataset} con la semilla 0. Mismo protocolo que cf_filtro, solo cambia la red",
             ha="center", va="top", fontsize=9, color="#47525d")

    # modulo 1, entrenamiento
    caja(fig, 0.015, 0.515, 0.97, 0.405, "1  Entrenamiento por regresión supervisada, fuera de la búsqueda", color="#eaf1f8", ec=AZUL, fs=11)
    W = 0.148; X = [0.025, 0.185, 0.345, 0.505, 0.665, 0.825]; Y, H = 0.53, 0.35
    nombres = ["Fotos de la voraz con h_fs", "Entrada de 5 canales", "Etiqueta cf*", "Regresión", "DAgger", "Validación y elección"]
    for k, (xx, nm) in enumerate(zip(X, nombres)):
        caja(fig, xx, Y, W, H, nm, color="white", letra="abcdef"[k])
        if k < 5:
            flecha(fig, (xx + W + 0.002, Y + H / 2), (X[k + 1] - 0.002, Y + H / 2))
    mini(fig, [X[0] + 0.012, 0.6, 0.124, 0.22], estado(eb.desc), CMAP_ESTADO, 0, 2, s=s, g=g, closed=rbm.closed, rc=rc)
    texto(fig, X[0] + 0.008, 0.595, "una foto del mapa descubierto\ncada 20 expansiones", fs=7)
    nc = ["libre", "obstáculo", "desconocido", "meta", "h_fs"]
    for k in range(5):
        c, f_ = k % 3, k // 3
        ax = mini(fig, [X[1] + 0.008 + c * 0.046, 0.71 - f_ * 0.105, 0.042, 0.075], xb[k][R, R], "gray" if k < 4 else "magma", 0, 1 if k < 4 else H_TOPE)
        ax.set_title(nc[k], fontsize=6.6, pad=2)
        if k == 3:
            ax.plot(g[1], g[0], "o", ms=7, mfc="none", mec="tab:red", mew=1.4)
    texto(fig, X[1] + 0.008, 0.595, "208 x 208, h_fs sobre la escala\ncon tope 2, sin ver el mapa real", fs=7)
    mini(fig, [X[2] + 0.012, 0.6, 0.124, 0.22], np.where(msk, y, np.nan)[R, R], "viridis", 0, 1)
    texto(fig, X[2] + 0.008, 0.595, "cf* = h_fs / h* en celdas libres\nalcanzables, la corrección exacta", fs=7)
    texto(fig, X[3] + 0.008, 0.835,
          "error cuadrático entre cf y cf*\nsolo en celdas con etiqueta\n\nAdam 1e-3, lote de 8\n8 simetrías del cuadrado al azar\n\n10 épocas con las fotos\nde la voraz con h_fs", fs=7.8)
    mini(fig, [X[4] + 0.012, 0.6, 0.124, 0.22], estado(em.desc), CMAP_ESTADO, 0, 2, s=s, g=g, closed=rm.closed, rc=rc)
    texto(fig, X[4] + 0.008, 0.595, "2 rondas de 10 épocas con fotos\nde la búsqueda con su propio cf", fs=7)
    texto(fig, X[5] + 0.008, 0.835,
          "búsqueda en 70 mapas de\nvalidación cada 5 épocas\n\nderrumbe si pierde mapas o\nexpande 50 % más que h_fs\n\ndos derrumbes seguidos\ncortan la corrida\n\nse guarda la mejor época,\nla 0 también compite", fs=7.8)

    # modulo 2, uso dentro de la busqueda
    caja(fig, 0.015, 0.03, 0.97, 0.465, "2  Corrector dentro de la búsqueda voraz", color="#fdf3e7", ec=NARANJA, fs=11)
    Y2, H2 = 0.07, 0.385
    X2 = [0.025, 0.165, 0.305, 0.555, 0.695, 0.835]; W2 = [0.128, 0.128, 0.238, 0.128, 0.128, 0.142]
    nombres = ["Mapa descubierto", "5 canales", "CNN local", "cf de la cabeza", "Gauss fijo", "h = h_fs / cf"]
    for k, (xx, ww, nm) in enumerate(zip(X2, W2, nombres)):
        caja(fig, xx, Y2, ww, H2, nm, color="white", letra="abcdef"[k])
        if k < 5:
            flecha(fig, (xx + ww + 0.002, Y2 + H2 / 2 + 0.03), (X2[k + 1] - 0.002, Y2 + H2 / 2 + 0.03))
    mini(fig, [X2[0] + 0.008, 0.2, 0.112, 0.2], estado(em.desc), CMAP_ESTADO, 0, 2, s=s, g=g, closed=rm.closed, rc=rc)
    texto(fig, X2[0] + 0.006, 0.19, "a mitad de la búsqueda\nnaranja, nodos expandidos", fs=7)
    for k in range(5):
        c, f_ = k % 2, k // 2
        mini(fig, [X2[1] + 0.01 + c * 0.056, 0.33 - f_ * 0.085, 0.05, 0.062], x[k][R, R], "gray" if k < 4 else "magma", 0, 1 if k < 4 else H_TOPE,
             titulo=nc[k])
    texto(fig, X2[1] + 0.006, 0.13, "la misma imagen que\nen el entrenamiento", fs=7)
    red_local(fig, X2[2] + 0.005, 0.11, 0.228, 0.29)
    mini(fig, [X2[3] + 0.008, 0.2, 0.112, 0.2], np.where(obst, np.nan, crudo), "viridis", 0, 1)
    texto(fig, X2[3] + 0.006, 0.19, "un cf por celda en (0, 1]\nsigmoide de la cabeza", fs=7)
    mini(fig, [X2[4] + 0.008, 0.2, 0.112, 0.2], np.where(obst, np.nan, cf), "viridis", 0, 1)
    texto(fig, X2[4] + 0.006, 0.19, f"desvío {cfg.sigma:g} celdas en bugtraps\ny 16 en gaps, sin pesos", fs=7)
    mini(fig, [X2[5] + 0.015, 0.2, 0.112, 0.2], estado(ev.desc), CMAP_ESTADO, 0, 2, s=s, g=g, closed=res.closed, rc=rc, camino=res.path)
    texto(fig, X2[5] + 0.006, 0.19, f"búsqueda completa, {res.n_expanded}\nexpansiones contra {rb.n_expanded} con h_fs\nverde, camino encontrado", fs=7)
    flecha(fig, (X2[5] + 0.07, Y2 - 0.002), (X2[5] + 0.07, Y2 - 0.016), color=NARANJA, estilo="-", lw=1.6)
    flecha(fig, (X2[5] + 0.07, Y2 - 0.016), (X2[0] + 0.064, Y2 - 0.016), color=NARANJA, estilo="-", lw=1.6)
    flecha(fig, (X2[0] + 0.064, Y2 - 0.016), (X2[0] + 0.064, Y2 - 0.001), color=NARANJA, lw=1.6)
    fig.text(0.5, 0.049, "cada 20 expansiones se vuelve a sacar la foto y a calcular cf, cada nodo recibe su h al generarse",
             ha="center", va="top", fontsize=8, color=NARANJA, fontweight="bold")
    flecha(fig, (0.5, 0.515), (0.5, 0.497), color=AZUL, lw=1.8)
    fig.text(0.507, 0.506, "la mejor época se usa en prueba", fontsize=7.5, color=AZUL, va="center")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=170, facecolor="white")
    print("pipeline", a.out, gp.name, "expansiones cf", res.n_expanded, "h_fs", rb.n_expanded)


if __name__ == "__main__":
    main()
