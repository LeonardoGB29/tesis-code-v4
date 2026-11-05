# exp_mem: resumen de la fase 1 desde la validacion. seleccion por semilla, regla de paso contra cf_filtro (y contra mem_no para las memorias),
# tiempos por mapa y por nodo de los cuatro brazos contra cf_filtro y h_fs, curvas, y borrador del resumen
#   python -m experiments.resumen_mem --root results/exp_mem --refs results/exp_cf --refs2 results/exp_cf2
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.common import DATASETS, f, leer_csv, md_tabla, write_csv
from experiments.resumen_cf import NOMBRES, curvas, seleccion

BRAZOS = ["phil_res_cf", "mem_no", "mem_gru", "mem_mamba"]
UMBRAL = 0.95


def corrida(root, modelo, ds, s):
    d = root / modelo / "runs" / ds / f"{modelo}_s{s}"
    if not (d / "train.csv").exists() or not (d / "config.json").exists():
        return None
    rows = leer_csv(d / "train.csv")
    meta = json.loads((d / "config.json").read_text())
    vals = [r for r in rows if r["val_exp"]]
    if not vals:
        return None
    best_ep = meta.get("best_epoca")
    fila_best = next((r for r in vals if best_ep is not None and int(r["epoca"]) == int(best_ep)), vals[0])
    return {"modelo": modelo, "dataset": ds, "seed": s, "best_epoca": best_ep, "best_val": meta.get("best_val"), "val0": float(vals[0]["val_exp"]),
            "camino_best": fila_best["val_camino"], "enc_best": fila_best["val_encontrados"], "secs_val_best": float(fila_best["secs_val"] or 0),
            "secs_red_best": float(fila_best["secs_red_val"] or 0), "validaciones": len(vals), "elegibles": sum(int(r["elegible"]) for r in vals),
            "epoca_final": meta.get("epoca_final"), "corte": int(bool(meta.get("corte"))), "listo": (d / "listo").exists(),
            "min_train": sum(float(r["secs_train"] or 0) for r in rows) / 60, "min_val": sum(float(r["secs_val"] or 0) for r in rows) / 60,
            "val_cf": (meta.get("val_cf") or {}).get("exp"), "ref_hfs": (meta.get("ref_val") or {}).get("exp")}


def referencia_cf(refs2, ds, s):
    d = refs2 / "cf_filtro" / "runs" / ds / f"cf_filtro_s{s}"
    if not (d / "train.csv").exists():
        return None
    meta = json.loads((d / "config.json").read_text())
    rows = [r for r in leer_csv(d / "train.csv") if r["val_exp"]]
    best = next((r for r in rows if int(r["epoca"]) == int(meta["best_epoca"])), None)
    return {"best_val": meta.get("best_val"), "best_epoca": meta.get("best_epoca"), "secs_val": float(best["secs_val"]) if best else None,
            "camino": best["val_camino"] if best else None}


def referencia_hfs(refs, ds):
    rows = leer_csv(refs / "refs" / f"val_hfs_{ds}.csv")
    rows = [r for r in rows if r["optimo"] and int(r["optimo"]) >= 0]
    return {"n": len(rows), "exp": float(np.mean([float(r["exp"]) for r in rows])), "secs": float(np.sum([float(r["secs"]) for r in rows])),
            "camino": float(np.mean([int(r["camino"]) / max(int(r["optimo"]), 1) for r in rows if r["encontrado"] == "1"]))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results/exp_mem")
    p.add_argument("--refs", default="results/exp_cf")
    p.add_argument("--refs2", default="results/exp_cf2")
    a = p.parse_args()
    root, refs, refs2 = Path(a.root), Path(a.refs), Path(a.refs2)
    seeds = sorted({int(d.name.split("_s")[-1]) for m in BRAZOS for ds in DATASETS for d in (root / m / "runs" / ds).glob(f"{m}_s*")} | {0, 1})
    hfs = {ds: referencia_hfs(refs, ds) for ds in DATASETS}
    cf = {(ds, s): referencia_cf(refs2, ds, s) for ds in DATASETS for s in seeds}
    cor = {(m, ds, s): corrida(root, m, ds, s) for m in BRAZOS for ds in DATASETS for s in seeds}

    # seleccion y paso
    filas_sel, filas_paso = [], []
    for m in BRAZOS:
        for ds in DATASETS:
            for s in seeds:
                c, r = cor[(m, ds, s)], cf[(ds, s)]
                if not c or not r:
                    continue
                rel_cf = c["best_val"] / r["best_val"] if c["best_val"] and r["best_val"] else float("nan")
                c0 = cor[("mem_no", ds, s)]
                rel_no = (c["best_val"] / c0["best_val"]) if (m in ("mem_gru", "mem_mamba") and c0 and c0["best_val"] and c["best_val"]) else float("nan")
                filas_sel.append([ds, m, s, c["best_epoca"], f(c["best_val"], 1), f(c["val0"], 1), f(r["best_val"], 1), f(rel_cf), f(rel_no), c["camino_best"], c["enc_best"],
                                  c["elegibles"], c["validaciones"], c["epoca_final"], c["corte"], int(c["listo"]), round(c["min_train"], 1), round(c["min_val"], 1),
                                  int(abs(c["val0"] - r["best_val"]) < 0.05)])
    head_sel = ["dataset", "modelo", "seed", "epoca_elegida", "val_exp_elegida", "val_exp_epoca0", "val_exp_cf_filtro", "rel_cf_filtro", "rel_mem_no", "camino_elegida",
                "encontrados", "validaciones_elegibles", "validaciones", "epoca_final", "corte", "listo", "min_train", "min_val", "epoca0_igual_cf"]
    write_csv(root / "seleccion_val.csv", head_sel, filas_sel)
    paso = {}
    for m in BRAZOS:
        for ds in DATASETS:
            rels = [c["best_val"] / cf[(ds, s)]["best_val"] for s in seeds if (c := cor[(m, ds, s)]) and cf[(ds, s)] and c["best_val"]]
            ok = len(rels) >= 2 and all(r < UMBRAL for r in rels)
            det = ""
            if m in ("mem_gru", "mem_mamba"):
                rn = [c["best_val"] / cor[("mem_no", ds, s)]["best_val"] for s in seeds if (c := cor[(m, ds, s)]) and cor[("mem_no", ds, s)] and c["best_val"] and cor[("mem_no", ds, s)]["best_val"]]
                det = " ".join(f"{x:.3f}" for x in rn)
                paso[(m, ds, "mem_no")] = len(rn) >= 2 and all(x < UMBRAL for x in rn)
            paso[(m, ds)] = ok
            filas_paso.append([ds, m, " ".join(f"{x:.3f}" for x in rels), int(ok), det, int(paso.get((m, ds, "mem_no"), 0)) if det else ""])
    write_csv(root / "paso.csv", ["dataset", "modelo", "rel_cf_filtro_por_semilla", "pasa_cf_filtro", "rel_mem_no_por_semilla", "pasa_mem_no"], filas_paso)

    # tiempos en validacion a la epoca elegida: por mapa y por nodo, contra cf_filtro y h_fs
    filas_t = []
    for ds in DATASETS:
        h = hfs[ds]
        filas_t.append([ds, "voraz_hfs", "", f(h["exp"], 1), f(h["secs"] / h["n"], 3), f(1000 * h["secs"] / (h["exp"] * h["n"]), 3), f(h["camino"])])
        for s in seeds:
            r = cf[(ds, s)]
            if r and r["secs_val"]:
                filas_t.append([ds, "cf_filtro", s, f(r["best_val"], 1), f(r["secs_val"] / h["n"], 3), f(1000 * r["secs_val"] / (r["best_val"] * h["n"]), 3), r["camino"]])
        for m in BRAZOS:
            for s in seeds:
                c = cor[(m, ds, s)]
                if c and c["secs_val_best"] and c["best_val"]:                    # las corridas en marcha todavia no tienen epoca elegida
                    filas_t.append([ds, m, s, f(c["best_val"], 1), f(c["secs_val_best"] / h["n"], 3), f(1000 * c["secs_val_best"] / (c["best_val"] * h["n"]), 3), c["camino_best"]])
    head_t = ["dataset", "modelo", "seed", "val_exp", "secs_por_mapa", "ms_por_nodo", "camino"]
    write_csv(root / "tiempos.csv", head_t, filas_t)
    seleccion(root)
    curvas(root)

    md = [f"# Tablas de validacion de {root.name}\n",
          "Fase 1, solo validacion, 70 mapas (los resolubles) con tope de 10 000 expansiones. La epoca 0 de cada brazo es cf_filtro best.pt de la misma semilla. "
          f"Relacion es expansiones de la epoca elegida sobre las de la epoca elegida de cf_filtro. Pasa si la relacion es menor a {UMBRAL} en las dos semillas.\n"]
    for ds in DATASETS:
        md.append(f"\n## {ds}\n")
        md.append(md_tabla(head_sel, [r for r in filas_sel if r[0] == ds]))
    md.append("\n## Regla de paso\n")
    md.append(md_tabla(["conjunto", "brazo", "relacion con cf_filtro por semilla", "pasa", "relacion con mem_no por semilla", "pasa contra mem_no"], filas_paso))
    pasan = sorted({m for (m, ds, *k) in paso if not k and paso[(m, ds)]})
    md.append(("\nSiguen a la fase 2: " + ", ".join(pasan) + ".\n") if pasan else f"\nNingun brazo baja mas de {100 * (1 - UMBRAL):.0f} % la validacion de cf_filtro en un conjunto con las dos semillas.\n")
    md.append("\n## Tiempos en validacion a la epoca elegida\n")
    md.append(md_tabla(head_t, filas_t))
    md.append("\n## Curvas\n\n![curvas](curvas_entrenamiento.png)\n")
    (root / "tablas_val.md").write_text("".join(md))
    print("".join(md))
    print(f"-> {root / 'tablas_val.md'}")


if __name__ == "__main__":
    main()
