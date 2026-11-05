# exp_mem: fotos al ritmo del cf, de la voraz con h_fs y de cf_filtro best.pt de la semilla sobre los 200 mapas de entrenamiento,
# con expandido y abierto, y precalculo con la u-net congelada (cf_filtro, rasgos en la muestra y sus vecinos, aristas, estado, escalares)
#   python -m experiments.fotos_mem --dataset mazes --seed 0 [--root results/exp_mem --refs2 results/exp_cf2 --device mps --n-train 2]
import argparse
import json
import time
from pathlib import Path

from experiments.common import ESCALA, MAX_EXP, RESULTS, grids_de, set_threads, write_csv
from src.cf.busqueda import EvaluadorCFFotosMem, EvaluadorFotosMem
from src.cf.datos_mem import FotosMem, Precalculo, recolectar_mem
from src.cf.memoria import MemConfig
from src.cf.red import cargar

ROOT_MEM = RESULTS.parent / "exp_mem"
REFS2 = RESULTS.parent / "exp_cf2"


def cf_ckpt_de(refs2, dataset, seed):
    return Path(refs2) / "cf_filtro" / "runs" / dataset / f"cf_filtro_s{seed}" / "best.pt"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--root", default=str(ROOT_MEM))
    p.add_argument("--refs2", default=str(REFS2))
    p.add_argument("--device", default="mps")
    p.add_argument("--lote", type=int, default=16)
    p.add_argument("--n-train", type=int, default=0, help="humo: recorta los mapas de entrenamiento")
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args()
    set_threads(1)
    out = Path(a.root) / "datos" / a.dataset / f"s{a.seed}"
    if (out / "pre" / "listo").exists():
        print(f"[{a.dataset}/s{a.seed}] datos ya estan", flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    ck = cf_ckpt_de(a.refs2, a.dataset, a.seed)
    unet, ucfg, raw = cargar(ck, a.device)
    cfg = MemConfig(modelo="mem_no", cf_ckpt=str(ck), sigma=ucfg.sigma, every_cf=ucfg.every_cf, every_fs=ucfg.every_fs)
    grids = grids_de(a.dataset, "train")
    if a.n_train:
        grids = grids[:a.n_train]
    t0 = time.time()
    if not (out / "fotos" / "listo").exists():
        fotos, filas = recolectar_mem(a.dataset, grids, lambda data, adj, v_g: [EvaluadorFotosMem(data, v_g, ucfg.every_cf, ucfg.every_fs, ESCALA),
                                                                                EvaluadorCFFotosMem(data, v_g, unet, ucfg.every_cf, ucfg.every_fs, ESCALA, a.device)],
                                      MAX_EXP, verbose=a.verbose)
        fotos.guardar(out / "fotos")
        write_csv(out / "busquedas.csv", ["dataset", "grid", "origen", "optimo", "exp", "encontrado", "camino", "fotos", "secs"], filas)
        print(f"[{a.dataset}/s{a.seed}] {len(fotos)} fotos de {len(fotos.nombres)} mapas en {time.time() - t0:.0f}s", flush=True)
    fotos = FotosMem.cargar(out / "fotos")
    t1 = time.time()
    pre = Precalculo.crear(out / "pre", fotos, unet, cfg, a.device, a.lote, a.verbose)
    secs = fotos.secuencias()
    meta = {"dataset": a.dataset, "seed": a.seed, "cf_ckpt": str(ck), "epoca_cf": raw.get("epoca"), "n_fotos": len(fotos), "n_mapas": len(fotos.nombres),
            "n_secuencias": len(secs), "largo_min": min(l for _, l in secs), "largo_max": max(l for _, l in secs),
            "fotos_hfs": int((fotos.origen == 0).sum()), "fotos_cf": int((fotos.origen == 1).sum()),
            "n_nodos_medio": float(pre.n_nodos[:].mean()), "secs_fotos": round(t1 - t0, 1), "secs_precalculo": round(time.time() - t1, 1), "config": cfg.to_dict()}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[{a.dataset}/s{a.seed}] precalculo de {len(fotos)} fotos en {time.time() - t1:.0f}s, {len(secs)} secuencias de {meta['largo_min']} a {meta['largo_max']} -> {out}", flush=True)


if __name__ == "__main__":
    main()
