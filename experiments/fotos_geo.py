# fotos para exp_geo. la voraz con h_fs sobre los mapas de validacion y la busqueda con cf_filtro best.pt de una semilla
#   python -m experiments.fotos_geo --dataset mazes --que hfs_val
#   python -m experiments.fotos_geo --dataset mazes --que cf --seed 0 --split train
# si el best.pt de cf_filtro es la epoca 0 o no gana a h_fs en validacion, su busqueda es la de h_fs y no se duplican fotos
import argparse
import json
import time
from pathlib import Path

from experiments.common import ESCALA, EXP_CF2, MAX_EXP, RESULTS_GEO, set_threads, write_csv
from experiments.fotos import FOTO_CADA, recolectar
from src.cf.busqueda import Evaluador, EvaluadorCF
from src.cf.red import cargar

COLS = ["dataset", "grid", "optimo", "exp", "encontrado", "camino", "fotos", "secs"]


def cf_filtro_es_hfs(ckpt):
    raw_cfg = Path(ckpt).parent / "config.json"
    import torch
    raw = torch.load(ckpt, map_location="cpu", weights_only=False)
    if raw.get("epoca", 1) == 0 or raw.get("extra", {}).get("sin_elegibles"):
        return True
    ref = (json.loads(raw_cfg.read_text()).get("ref_val") or {}) if raw_cfg.exists() else {}
    val = raw.get("val") or {}
    return ref.get("exp") is not None and val.get("exp") is not None and val["exp"] > ref["exp"] + 1e-9


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--que", required=True, choices=["hfs_val", "cf"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split", default="train", choices=["train", "validation"])
    p.add_argument("--out-root", default=str(RESULTS_GEO / "fotos"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-grids", type=int, default=0, help="humo")
    a = p.parse_args()
    set_threads(1)
    if a.n_grids:
        import experiments.common as cm
        cm.N_TRAIN = cm.N_VAL = a.n_grids
    nombre = "hfs_val" if a.que == "hfs_val" else f"cf_s{a.seed}_{'train' if a.split == 'train' else 'val'}"
    split = "validation" if a.que == "hfs_val" else a.split
    out = Path(a.out_root) / a.dataset / nombre
    if (out / "listo").exists():
        print(f"[{a.dataset}/{nombre}] ya esta", flush=True)
        return
    t0 = time.time()
    if a.que == "hfs_val":
        hacer = lambda data, adj, v_g: Evaluador(data, v_g, 16, ESCALA)
    else:
        ck = EXP_CF2 / "cf_filtro" / "runs" / a.dataset / f"cf_filtro_s{a.seed}" / "best.pt"
        if cf_filtro_es_hfs(ck):
            out.mkdir(parents=True, exist_ok=True)
            (out / "listo").write_text("0 fotos, cf_filtro sin epoca elegible, su busqueda es la de h_fs y ya esta en las fotos base\n")
            print(f"[{a.dataset}/{nombre}] cf_filtro s{a.seed} es h_fs, sin fotos nuevas", flush=True)
            return
        model, cfg, raw = cargar(ck, a.device)
        hacer = lambda data, adj, v_g: EvaluadorCF(data, v_g, model, cfg.every_cf, cfg.every_fs, ESCALA, a.device)
    fotos, filas = recolectar(a.dataset, hacer, FOTO_CADA, MAX_EXP, split=split)
    fotos.guardar(out)
    write_csv(out / "busquedas.csv", COLS, filas)
    print(f"[{a.dataset}/{nombre}] {len(fotos)} fotos de {len(fotos.nombres)} mapas en {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
