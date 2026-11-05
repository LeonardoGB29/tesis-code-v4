# exp_mem: campo R, cf de cf_filtro y cf corregido de los brazos mem en 3 mapas de validacion por conjunto, con la epoca elegida de la semilla 0
#   python -m experiments.figuras_mem --root results/exp_mem --refs2 results/exp_cf2 [--seed 0 --n 3]
import argparse
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.common import DATASETS, ESCALA, MAX_EXP, grids_de, load_problem, set_threads
from src.cf.busqueda import EvaluadorMem, voraz
from src.cf.memoria import MODELOS_MEM, cargar_mem


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results/exp_mem")
    p.add_argument("--refs2", default="results/exp_cf2")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    set_threads(1)
    root = Path(a.root)
    (root / "figuras").mkdir(parents=True, exist_ok=True)
    for ds in DATASETS:
        modelos = {}
        for m in MODELOS_MEM:
            ck = root / m / "runs" / ds / f"{m}_s{a.seed}" / "best.pt"
            if ck.exists():
                modelos[m] = cargar_mem(ck, a.device, "cpu")
        if not modelos:
            continue
        grids = grids_de(ds, "validation")[:a.n]
        fig, axes = plt.subplots(len(grids), 1 + 2 * len(modelos), figsize=(4 * (1 + 2 * len(modelos)), 4 * len(grids)), squeeze=False)
        for i, gp in enumerate(grids):
            data, adj, v_s, v_g, h_star = load_problem(gp)
            H = int(data.x[:, 0].max()) + 1
            for j, (m, (model, cfg, raw)) in enumerate(modelos.items()):
                ev = EvaluadorMem(data, v_g, model, gp.name, cfg.every_cf, cfg.every_fs, ESCALA, a.device, "cpu"); ev.campos = []
                res = voraz(ev, data, v_s, v_g, adj, max_exp=MAX_EXP)
                R = ev.campos[-1].numpy()[:H, :H] if ev.campos else np.zeros((H, H))
                cf = ev.mapa_cf[:H, :H]
                if j == 0:
                    ax = axes[i, 0]
                    img = np.ones((H, H, 3))
                    img[ev.desc.real] = 0
                    img[~ev.desc.visto] = 0.6
                    img[ev.desc.expandido] = (1.0, 0.6, 0.2)
                    ax.imshow(img); ax.set_title(f"{ds} {gp.name}\nmapa descubierto al final, {res.n_expanded} exp", fontsize=8); ax.axis("off")
                ax = axes[i, 1 + 2 * j]
                v = max(float(np.abs(R).max()), 1e-6)
                im = ax.imshow(R, cmap="coolwarm", vmin=-v, vmax=v); ax.set_title(f"{m} epoca {raw.get('epoca')}: R final, exp {res.n_expanded}", fontsize=8); ax.axis("off")
                plt.colorbar(im, ax=ax, fraction=0.046)
                ax = axes[i, 2 + 2 * j]
                im = ax.imshow(cf, cmap="viridis", vmin=0, vmax=1); ax.set_title("cf corregido que usa la busqueda", fontsize=8); ax.axis("off")
                plt.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(root / "figuras" / f"R_{ds}.png", dpi=110)
        plt.close(fig)
        print(f"-> {root / 'figuras' / f'R_{ds}.png'}", flush=True)


if __name__ == "__main__":
    main()
