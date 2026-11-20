# copia los numeros de un modelo de new_exp a ablations/<carpeta>, solo csv, la explicacion va a mano en el RESUMEN.md de esa carpeta
#   python -m new_exp.exportar --modelo cf_local_filtro --carpeta ablations/cnn_local
#   resultados.csv     una fila por conjunto para cf_filtro y el modelo, las mismas cuentas que new_exp/resumen.py
#   por_semilla.csv    una fila por corrida, epocas, corte por derrumbe y relaciones de prueba
#   eval/<ds>/         los csv de prueba mapa por mapa
#   train/<ds>/        los train.csv de cada corrida
import argparse
import csv
import shutil
from pathlib import Path

import experiments.resumen_cf as rc
from experiments.common import EXP_CF2, RESULTS
from new_exp.resumen import contra_filtro, num

CONJUNTOS = ["gaps_and_forest", "multiple_bugtraps"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modelo", required=True)
    p.add_argument("--carpeta", required=True)
    p.add_argument("--root", default="new_exp/results")
    a = p.parse_args()
    root, out, m = Path(a.root), Path(a.carpeta), a.modelo
    out.mkdir(parents=True, exist_ok=True)
    ev = rc.corridas(root, RESULTS, EXP_CF2)

    filas = [["conjunto", "modelo", "semillas", "expansiones", "rel_voraz_hfs", "rel_voraz_hfs_por_semilla", "p_voraz_hfs",
              "rel_cf_filtro", "rel_cf_filtro_por_semilla", "p_cf_filtro", "camino", "ms_nodo"]]
    por_semilla = {}
    for ds in CONJUNTOS:
        for mod in ("cf_filtro", m):
            r = rc.resumen_modelo(ev, ds, mod, ev.get((ds, "voraz_hfs")), ev.get((ds, "phil")))
            if not r:
                continue
            c = contra_filtro(ev, ds, mod)
            filas.append([ds, mod, r["seeds"], num(r["exp"], 1), num(r.get("rel_hfs"), 3), r.get("rel_hfs_por_semilla", ""),
                          num(r.get("p_hfs_exp")), num(c["rel"], 3) if c else "", " ".join(f"{x:.3f}" for x in c["por_semilla"]) if c else "",
                          num(c["p"]) if c else "", num(r.get("camino_opt"), 3), num(r.get("ms_nodo"), 3)])
            if mod == m:
                seeds = [int(s) for s in str(r["seeds"]).split()]
                rh = str(r.get("rel_hfs_por_semilla", "")).split()
                rf = c["por_semilla"] if c else [None] * len(seeds)
                for s, x, y in zip(seeds, rh, rf):
                    por_semilla[(ds, s)] = (x, y)
    with open(out / "resultados.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(filas)

    filas = [["conjunto", "semilla", "epocas_corridas", "mejor_epoca", "mejor_val", "corte_por_derrumbe", "rel_voraz_hfs", "rel_cf_filtro"]]
    for ds in CONJUNTOS:
        for run in sorted((root / m / "runs" / ds).glob(f"{m}_s*")):
            s = int(run.name.rsplit("_s", 1)[1])
            tr = list(csv.DictReader(open(run / "train.csv", encoding="utf-8")))
            ult = tr[-1]
            corte = int(int(ult["epoca"]) < 30)
            x, y = por_semilla.get((ds, s), ("", None))
            filas.append([ds, s, ult["epoca"], ult["best_epoca"], ult["best_val"], corte, x, f"{y:.3f}" if y is not None else ""])
            (out / "train" / ds).mkdir(parents=True, exist_ok=True)
            shutil.copy(run / "train.csv", out / "train" / ds / f"{run.name}.csv")
        src = root / m / "eval" / ds
        if src.exists():
            (out / "eval" / ds).mkdir(parents=True, exist_ok=True)
            for f in src.glob("*.csv"):
                shutil.copy(f, out / "eval" / ds / f.name)
    with open(out / "por_semilla.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(filas)
    print((out / "resultados.csv").read_text(encoding="utf-8"))
    print((out / "por_semilla.csv").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
