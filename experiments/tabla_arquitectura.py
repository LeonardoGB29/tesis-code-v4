# tabla de la arquitectura de la cnn. promedios de expansiones, largo del camino y caminos optimos de las 5 redes y la voraz con h_fs
#   python -m experiments.tabla_arquitectura --out ablations
# lee de code_v3 la prueba de la voraz con h_fs y la de cf_filtro de exp_cf2, y de ablations la de las 4 variantes, semillas 0 a 2
import argparse
import csv
from pathlib import Path

import numpy as np

from experiments.alinear_md import alinear
from experiments.common import RESULTS_V3

REPO = Path(__file__).resolve().parent.parent
CONJUNTOS = ["gaps_and_forest", "multiple_bugtraps", "mazes"]
SEMILLAS = [0, 1, 2]
# nombre, plantilla de la ruta con {ds} y {s}, semillas
MODELOS = [
    ("Voraz con h_fs (referencia)", RESULTS_V3 / "exp_cf/refs/eval/{ds}/voraz_hfs.csv", [None]),
    ("U-Net con atención (cf_filtro)", RESULTS_V3 / "exp_cf2/cf_filtro/eval/{ds}/cf_filtro_s{s}.csv", SEMILLAS),
    ("U-Net sin atención", REPO / "ablations/unet_sin_atencion/eval/{ds}/cf_sin_atencion_s{s}.csv", SEMILLAS),
    ("Solo codificador", REPO / "ablations/solo_codificador/eval/{ds}/cf_codificador_s{s}.csv", SEMILLAS),
    ("TransPath", REPO / "ablations/transpath/eval/{ds}/cf_transpath_s{s}.csv", SEMILLAS),
    ("CNN local 11x11", REPO / "ablations/cnn_local/eval/{ds}/cf_local_filtro_s{s}.csv", SEMILLAS),
]
OPTIMO = "Óptimo (BFS)"


def leer(plantilla, ds, s):
    # filas de prueba por mapa
    with open(str(plantilla).format(ds=ds, s=s)) as f:
        return {r["grid"]: r for r in csv.DictReader(f)}


def medidas(ds):
    # mapas, largo optimo medio y por modelo expansiones, camino y caminos optimos, media de las semillas
    corridas = {n: [leer(p, ds, s) for s in sem] for n, p, sem in MODELOS}
    ref = corridas[MODELOS[0][0]][0]
    mapas = sorted(ref)
    for n, rs in corridas.items():
        for r in rs:
            assert sorted(r) == mapas, (ds, n)
            assert all(r[g]["encontrado"] == "1" for g in mapas), (ds, n)   # el camino se promedia en todos los mapas
    opt = np.array([int(ref[g]["optimo"]) for g in mapas])
    filas = {}
    for n, rs in corridas.items():
        exp = np.mean([np.mean([float(r[g]["exp"]) for g in mapas]) for r in rs])
        cam = np.mean([np.mean([float(r[g]["camino"]) for g in mapas]) for r in rs])
        pct = 100 * np.mean([np.mean([int(float(r[g]["camino"])) == o for g, o in zip(mapas, opt)]) for r in rs])
        filas[n] = (exp, cam, pct)
    return len(mapas), float(opt.mean()), filas


def tabla(titulo, filas_md):
    return [f"## {titulo}", "", "| modelo | " + " | ".join(CONJUNTOS) + " |", "|---|---|---|---|"] + filas_md + [""]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="ablations")
    a = p.parse_args()
    out = REPO / a.out
    res = {ds: medidas(ds) for ds in CONJUNTOS}
    nombres = [m[0] for m in MODELOS]
    exp = [f"| {n} | " + " | ".join(f"{res[ds][2][n][0]:.3f}" for ds in CONJUNTOS) + " |" for n in nombres]
    cam = [f"| {OPTIMO} | " + " | ".join(f"{res[ds][1]:.3f}" for ds in CONJUNTOS) + " |"]
    cam += [f"| {n} | " + " | ".join(f"{res[ds][2][n][1]:.3f}" for ds in CONJUNTOS) + " |" for n in nombres]
    pct = [f"| {n} | " + " | ".join(f"{res[ds][2][n][2]:.2f}" for ds in CONJUNTOS) + " |" for n in nombres]
    mapas = ", ".join(f"{res[ds][0]} en {ds}" for ds in CONJUNTOS)
    md = ["# Arquitectura de la CNN, promedios por mapa", "",
          f"Prueba en todos los mapas resolubles, {mapas}. Los 6 modelos resuelven el 100 % de los mapas. "
          "Media de las semillas 0 a 2, con un tope de 10 000 expansiones. "
          "El óptimo sale de una búsqueda en anchura desde la meta sobre el mapa real. Generado por experiments/tabla_arquitectura.py.", ""]
    md += tabla("Nodos expandidos, promedio por mapa", exp)
    md += tabla("Largo medio del camino, en pasos", cam)
    md += tabla("Caminos óptimos, en porcentaje", pct)
    (out / "promedios.md").write_text(alinear("\n".join(md)))
    with open(out / "promedios.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["conjunto", "modelo", "mapas", "expansiones", "camino", "caminos_optimos_pct"])
        for ds in CONJUNTOS:
            w.writerow([ds, "optimo_bfs", res[ds][0], "", f"{res[ds][1]:.4f}", "100.0000"])
            for n in nombres:
                e, c, q = res[ds][2][n]
                w.writerow([ds, n, res[ds][0], f"{e:.4f}", f"{c:.4f}", f"{q:.4f}"])
    print(out / "promedios.md")


if __name__ == "__main__":
    main()
