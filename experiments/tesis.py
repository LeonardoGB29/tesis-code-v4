# tesis. tabla principal, comparaciones nuevas, chequeo del p de exp_cf2, perdida de R0 y R1, tiempos limpios y figuras, todo leido de los csv
#   python -m experiments.tesis --out results/tesis
# lee de code_v3 las pruebas de la voraz con h_fs y con h*, de PHIL viejo y phil_res_s de exp_cf y de cf_filtro de exp_cf2,
# de exp_escalera las pruebas de R0 y R2, los train.csv y config.json de R0 y R1, seleccion.csv, resumen.csv y figuras/alpha_stats.csv,
# y de results/tesis/tiempos los tiempos limpios que deja tiempos_limpios.py
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.common import DATASETS, EXP_CF2, RESULTS, f, leer_csv, md_tabla, p_wilcoxon, write_csv
from scipy.stats import wilcoxon

from experiments.resumen_cf import cargar_eval, medias_por_mapa
from experiments.resumen_cf import resumen_modelo as resumen_cf_modelo
from experiments.resumen_escalera import comparar, resumen_modelo, veredicto
from experiments.train_escalera import ROOT_ESC

TABLA = ["voraz_hstar", "voraz_hfs", "phil_viejo", "phil_res_s", "cf_filtro"]
NOMBRES = {"voraz_hstar": "voraz con h* (techo)", "voraz_hfs": "voraz con h_fs", "phil_viejo": "PHIL de exp_cf", "phil_res_s": "phil_res_s de exp_cf",
           "cf_filtro": "cf_filtro de exp_cf2", "R0": "R0", "R2": "R2"}
ETIQUETA = {"phil_viejo": "PHIL viejo", "cf_filtro": "cf_filtro", "phil_res_s": "phil_res_s", "R0": "R0", "R2": "R2"}
COMPARACIONES = [("R2", "R0", "PHIL con la CNN de base baja a PHIL solo"), ("phil_res_s", "cf_filtro", "phil_res_s baja a cf_filtro"),
                 ("cf_filtro", "phil_viejo", "cf_filtro baja a PHIL viejo")]
# metodo de la tabla principal -> archivo de tiempos limpios en CPU, semilla 0
TIEMPO_DE = {"voraz_hstar": "voraz_hstar", "voraz_hfs": "voraz_hfs", "phil_viejo": "phil_viejo_s0", "phil_res_s": "phil_res_s_s0", "cf_filtro": "cf_filtro_s0"}
TIEMPOS = [("voraz_hstar", "cpu", "voraz con h*"), ("voraz_hfs", "cpu", "voraz con h_fs"), ("phil_viejo_s0", "cpu", "PHIL viejo s0"),
           ("R0_s0", "cpu", "PHIL reentrenado, R0 s0"), ("phil_res_s_s0", "cpu", "phil_res_s s0"), ("cf_filtro_s0", "cpu", "cf_filtro s0, U-Net en CPU"),
           ("cf_filtro_s0", "cuda", "cf_filtro s0, U-Net en CUDA")]


def cargar_todo(refs, refs2, esc):
    ev = defaultdict(dict)
    for ds in DATASETS:
        for ref in ("voraz_hfs", "voraz_hstar"):
            ev[(ds, ref)][0] = cargar_eval(refs / "refs" / "eval" / ds / f"{ref}.csv")
        for s in range(5):
            for m, p in (("phil_viejo", refs / "phil" / "eval" / ds / f"phil_s_s{s}.csv"), ("cf_filtro", refs2 / "cf_filtro" / "eval" / ds / f"cf_filtro_s{s}.csv"),
                         ("phil_res_s", refs / "phil_res_s" / "eval" / ds / f"phil_res_s_s{s}.csv"), ("R0", esc / "eval" / ds / f"R0_s{s}.csv"),
                         ("R2", esc / "eval" / ds / f"R2_s{s}.csv")):
                if p.exists():
                    ev[(ds, m)][s] = cargar_eval(p)
    return ev


def solo(ev, seeds):
    return {k: {s: v for s, v in d.items() if s in seeds} for k, d in ev.items()}


def fp(p):
    if p is None or p != p:
        return ""
    return "< 0.0001" if p < 1e-4 else f"{p:.4f}"


def fg(p):
    return "" if p is None or p != p else f"{p:.3g}"


def comunes(evx, ds, modelos):
    # mapas que encuentran todos los modelos con todas sus semillas
    g = None
    for m in modelos:
        for e in evx[(ds, m)].values():
            enc = {x for x, r in e.items() if r["enc"]}
            g = enc if g is None else g & enc
    return sorted(g or [])


def camino_en(evx, ds, m, grids):
    evs = evx[(ds, m)]
    v = [np.mean([evs[s][g]["camino"] / max(evs[s][g]["opt"], 1) for s in evs]) for g in grids]
    return float(np.mean(v)) if v else float("nan")


def pierde(evx, ds, m):
    evs = evx[(ds, m)]
    pares = sum(not r["enc"] for e in evs.values() for r in e.values())
    mapas = len({g for e in evs.values() for g, r in e.items() if not r["enc"]})
    return pares, mapas


def tiempos_de(tdir, dev, ds, archivo):
    p = tdir / dev / ds / f"{archivo}.csv"
    if not p.exists():
        return None
    rows = leer_csv(p)
    secs = sum(float(r["secs"]) for r in rows)
    exp = sum(float(r["exp"]) for r in rows)
    nr, sr = sum(int(r["n_red"] or 0) for r in rows), sum(float(r["secs_red"] or 0) for r in rows)
    return {"mapas": len(rows), "encontrados": sum(r["encontrado"] == "1" for r in rows), "exp": exp / len(rows), "ms_nodo": 1000 * secs / max(exp, 1),
            "ms_busqueda": 1000 * secs / len(rows), "ms_red": 1000 * sr / nr if nr else None, "exps": {r["grid"]: float(r["exp"]) for r in rows}}


def prueba_de(refs, refs2, esc, ds, archivo):
    return {"voraz_hstar": refs / "refs" / "eval" / ds / "voraz_hstar.csv", "voraz_hfs": refs / "refs" / "eval" / ds / "voraz_hfs.csv",
            "phil_viejo_s0": refs / "phil" / "eval" / ds / "phil_s_s0.csv", "R0_s0": esc / "eval" / ds / "R0_s0.csv",
            "phil_res_s_s0": refs / "phil_res_s" / "eval" / ds / "phil_res_s_s0.csv", "cf_filtro_s0": refs2 / "cf_filtro" / "eval" / ds / "cf_filtro_s0.csv"}[archivo]


def tabla_exp_cf2(path):
    # filas de la tabla principal del RESUMEN de exp_cf2, modelo -> columna -> texto
    out, head = {}, None
    for linea in path.read_text(encoding="utf-8").splitlines():
        if not linea.startswith("|"):
            if head is not None and out:
                break
            continue
        celdas = [c.strip() for c in linea.strip().strip("|").split("|")]
        if head is None:
            if "gaps p" in celdas:
                head = celdas
            continue
        if set(celdas[0]) <= set("-"):
            continue
        out[celdas[0]] = dict(zip(head, celdas))
    return out


def coincide(impreso, p):
    # el p calculado redondeado a las cifras significativas del texto impreso
    try:
        v = float(impreso)
    except ValueError:
        return False
    cifras = impreso.lower().split("e")[0].replace("-", "").replace(".", "").lstrip("0")
    return float(f"{p:.{max(len(cifras), 1)}g}") == v


def bloque(evx, ds, modelos, hfs, tdir=None):
    # filas de una tabla con el camino en los mapas que resuelven todos los modelos y semillas del bloque, y aparte en los mapas propios
    res = {m: resumen_modelo(evx, ds, m, hfs) for m in modelos}
    gc = comunes(evx, ds, modelos)
    filas = []
    for m in modelos:
        r = res[m]
        t = tiempos_de(tdir, "cpu", ds, TIEMPO_DE[m]) if tdir else None
        filas.append({"conjunto": ds, "modelo": m, "busqueda": NOMBRES[m], "semillas": r["seeds"], "mapas": r["n_mapas"], "exp": r["exp"], "rel": r.get("rel_hfs"),
                      "por_semilla": r.get("rel_hfs_por_semilla", ""), "exito": r["exito"], "camino_comun": camino_en(evx, ds, m, gc), "n_comunes": len(gc),
                      "camino_propio": r["camino_opt"], "p": None if m == "voraz_hfs" else r.get("p_hfs_exp"), "ms_nodo": t["ms_nodo"] if t else None,
                      "pierde": pierde(evx, ds, m)})
    return filas, gc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/tesis")
    p.add_argument("--refs", default=str(RESULTS))
    p.add_argument("--refs2", default=str(EXP_CF2))
    p.add_argument("--esc", default=str(ROOT_ESC))
    a = p.parse_args()
    out, refs, refs2, esc = Path(a.out), Path(a.refs), Path(a.refs2), Path(a.esc)
    tdir = out / "tiempos"
    out.mkdir(parents=True, exist_ok=True)
    ev = cargar_todo(refs, refs2, esc)
    ev3 = solo(ev, {0, 1, 2})
    md = ["# Tablas de la tesis\n", "Generadas por experiments/tesis.py a partir de los csv. Prueba en los 100 mapas de cada conjunto, 91 resolubles en gaps_and_forest, "
          "tope de 10 000 expansiones. Relacion es expansiones sobre las de la voraz con h_fs. Wilcoxon pareado por mapa sobre la media de las semillas. "
          "ms por nodo de los tiempos limpios, semilla 0, un proceso y un hilo, red en CPU como en la prueba.\n"]

    # 1. tabla principal con las semillas 0 a 2, camino en mapas comunes y aparte en mapas propios
    head_tp = ["conjunto", "busqueda", "semillas", "mapas", "expansiones", "relacion con h_fs", "por semilla", "exito", "camino sobre optimo en mapas comunes",
               "p contra h_fs", "ms por nodo"]
    filas_tp, filas_cam, filas_com, md_tp = [], [], [], []
    for ds in DATASETS:
        fl, gc = bloque(ev3, ds, TABLA, ev3[(ds, "voraz_hfs")], tdir)
        resolubles = fl[0]["mapas"]
        quien = [f"{NOMBRES[r['modelo']]} {r['pierde'][0]} pares semilla y mapa en {r['pierde'][1]} mapas" for r in fl if r["pierde"][0]]
        filas_com.append([ds, resolubles, len(gc), resolubles - len(gc), ", ".join(quien) or "nadie"])
        md_fl = []
        for r in fl:
            filas_tp.append([ds, r["busqueda"], r["semillas"], r["mapas"], f(r["exp"], 1), f(r["rel"]), r["por_semilla"], f(r["exito"]), f(r["camino_comun"]),
                             fg(r["p"]), f(r["ms_nodo"]) if r["ms_nodo"] is not None else "sin checkpoint"])
            md_fl.append([r["busqueda"], r["semillas"], f(r["exp"], 1), f(r["rel"]), r["por_semilla"], f(r["exito"]), f(r["camino_comun"]), fp(r["p"]),
                          f(r["ms_nodo"]) if r["ms_nodo"] is not None else "sin checkpoint"])
            filas_cam.append([ds, r["busqueda"], r["semillas"], f(r["camino_comun"]), r["n_comunes"], f(r["camino_propio"]), f(r["exito"])])
        md_tp.append((ds, md_fl, len(gc), resolubles))
    write_csv(out / "tabla_principal.csv", head_tp, filas_tp)
    head_cam = ["conjunto", "busqueda", "semillas", "camino en mapas comunes", "mapas comunes", "camino en mapas propios", "exito"]
    write_csv(out / "camino.csv", head_cam, filas_cam)
    head_com = ["conjunto", "mapas resolubles", "mapas comunes", "mapas fuera", "quien pierde mapas"]
    write_csv(out / "mapas_comunes.csv", head_com, filas_com)
    md.append("\n## Tabla principal, semillas 0 a 2\n")
    for ds, md_fl, n_c, n_r in md_tp:
        md.append(f"\n### {ds}, camino en los {n_c} de {n_r} mapas que resuelven todos los metodos y semillas\n\n")
        md.append(md_tabla(["busqueda", "semillas", "expansiones", "relacion con h_fs", "por semilla", "exito", "camino sobre optimo, mapas comunes", "p contra h_fs", "ms por nodo"], md_fl))
    md.append("\n## Camino en mapas comunes y en mapas propios\n\n")
    md.append(md_tabla(head_cam, filas_cam))
    md.append("\n## Mapas que quedan fuera del camino comun\n\n")
    md.append(md_tabla(head_com, filas_com))

    # 2. cifras con 5 semillas donde existen
    head_5 = ["conjunto", "busqueda", "semillas", "mapas", "expansiones", "relacion con h_fs", "por semilla", "exito", "camino en mapas comunes del bloque",
              "mapas comunes del bloque", "camino en mapas propios", "p contra h_fs"]
    filas_5 = []
    for ds in DATASETS:
        con5 = [m for m in ("phil_viejo", "cf_filtro") if len(ev[(ds, m)]) >= 5]
        if not con5:
            continue
        fl, gc = bloque(ev, ds, ["voraz_hstar", "voraz_hfs"] + con5, ev[(ds, "voraz_hfs")])
        for r in fl:
            if r["modelo"] in con5:
                filas_5.append([ds, r["busqueda"], r["semillas"], r["mapas"], f(r["exp"], 1), f(r["rel"]), r["por_semilla"], f(r["exito"]), f(r["camino_comun"]),
                                r["n_comunes"], f(r["camino_propio"]), fg(r["p"])])
    write_csv(out / "tabla_5_semillas.csv", head_5, filas_5)
    md.append("\n## Cifras con 5 semillas, donde existen\n\n")
    md.append(md_tabla(head_5, filas_5))

    # 3. comparaciones nuevas, formato de escalera.csv, criterio baja, semillas 0 a 2 emparejadas
    head_esc = ["a", "b", "conjunto", "criterio", "mapas", "exp a", "exp b", "a sobre b", "mejora %", "p Wilcoxon", "mapas donde gana a", "a sobre b por semilla",
                "semillas < 1", "camino a sobre b", "pasa en el conjunto"]
    filas_esc, veredictos = [], []
    for ma, mb, que in COMPARACIONES:
        n_ok = 0
        for ds in DATASETS:
            c = comparar(resumen_modelo(ev3, ds, ma, ev3[(ds, "voraz_hfs")]), resumen_modelo(ev3, ds, mb, ev3[(ds, "voraz_hfs")]))
            v = veredicto("baja", c)
            n_ok += bool(v)
            filas_esc.append([ETIQUETA[ma], ETIQUETA[mb], ds, "baja", c["n"], f(c["exp_a"], 1), f(c["exp_b"], 1), f(c["rel"]), f(100 * (1 - c["rel"]), 1), f(c["p"], 4),
                              c["gana_a"], " ".join(f"{x:.3f}" for x in c["rel_semillas"]), f"{c['semillas_bajo_1']} de {len(c['rel_semillas'])}", f(c["camino_rel"]), int(bool(v))])
        veredictos.append(f"{ETIQUETA[ma]} contra {ETIQUETA[mb]}, {que}. {'pasa' if n_ok >= 2 else 'no pasa'}, baja mas de 5 % con Wilcoxon y 3 de 3 semillas en {n_ok} de 3 conjuntos")
    write_csv(out / "comparaciones.csv", head_esc, filas_esc)
    md.append("\n## Comparaciones nuevas, criterio baja, semillas 0 a 2 emparejadas\n\n")
    md.append(md_tabla(head_esc, filas_esc))
    md.append("\n" + "\n".join(f"- {v}." for v in veredictos) + "\n")

    # 4. de donde sale el p 0.0018 de mazes en el RESUMEN de exp_cf2. cada celda p de su tabla principal contra el p de expansiones y el de camino,
    # calculados con resumen_cf.py, el mismo codigo que la armo, y con las mismas semillas, 5 en PHIL, 3 en phil_res_s y 5, 5 y 3 en cf_filtro
    impreso = tabla_exp_cf2(refs2 / "RESUMEN.md")
    head_p = ["modelo", "conjunto", "semillas", "p en el RESUMEN de exp_cf2", "p de expansiones", "p de camino", "coincide con", "relacion con h_fs", "expansiones",
              "expansiones voraz con h_fs", "camino", "camino voraz con h_fs", "mapas del p de camino", "mapas con el camino empatado", "p de camino con pratt"]
    filas_p = []
    for nombre, m in (("PHIL", "phil_viejo"), ("phil_res_s", "phil_res_s"), ("cf_filtro", "cf_filtro")):
        for ds, col in (("gaps_and_forest", "gaps p"), ("multiple_bugtraps", "bugtraps p"), ("mazes", "mazes p")):
            r = resumen_cf_modelo(ev, ds, m, ev[(ds, "voraz_hfs")], None)
            hf = ev[(ds, "voraz_hfs")][0]
            p_txt = impreso.get(nombre, {}).get(col, "")
            cual = [n for n, pv in (("expansiones", r["p_hfs_exp"]), ("camino", r["p_hfs_camino"])) if coincide(p_txt, pv)]
            # el p de camino de scipy descarta los mapas empatados, con pratt entran
            _, cam_m = medias_por_mapa(ev[(ds, m)], sorted(ev[(ds, m)]))
            cam_ref = {g: hf[g]["camino"] / max(hf[g]["opt"], 1) for g in hf if hf[g]["enc"]}
            g3 = [g for g in sorted(cam_m) if g in cam_ref]
            d = np.array([cam_m[g] - cam_ref[g] for g in g3])
            d = np.where(np.abs(d) < 1e-9, 0.0, d)
            p_pratt = float(wilcoxon(d, zero_method="pratt").pvalue) if len(d) >= 5 and np.any(d != 0) else 1.0
            filas_p.append([nombre, ds, r["seeds"], p_txt, f"{r['p_hfs_exp']:.4g}", f"{r['p_hfs_camino']:.4g}", " y ".join(cual) or "ninguno", f(r["rel_hfs"]),
                            f(r["exp"], 1), f(np.mean([hf[g]["exp"] for g in hf]), 1), f(r["camino_opt"]), f(r["camino_hfs"]), len(g3), int((d == 0).sum()), f"{p_pratt:.4g}"])
    write_csv(out / "p_exp_cf2.csv", head_p, filas_p)
    md.append("\n## El p 0.0018 de mazes en el RESUMEN de exp_cf2\n\nCada celda p de la tabla principal de ese RESUMEN contra el Wilcoxon de expansiones y el de camino de resumen_cf.py, con sus mismas semillas.\n\n")
    md.append(md_tabla(head_p, filas_p))

    # 5. perdida de entrenamiento de R0 y R1, en saltos al cuadrado, desde train.csv
    sel = {(x["conjunto"], x["peldano"], int(x["semilla"])): x for x in leer_csv(esc / "seleccion.csv")}
    head_l = ["conjunto", "peldano", "semilla", "perdida iteracion 1", "media iteraciones 1 a 12", "media iteraciones 97 a 108", "raiz de la media 97 a 108 en celdas",
              "minima", "iteracion de la minima", "iteracion elegida", "perdida en la elegida", "validacion en la elegida", "validacion en la 108", "val h_fs"]
    filas_l, curvas_l, medias = [], {}, {}
    for ds in DATASETS:
        for s in range(3):
            for pel in ("R0", "R1"):
                rows = leer_csv(esc / "runs" / ds / f"{pel}_s{s}" / "train.csv")
                L = {int(x["epoca"]): float(x["loss"]) for x in rows if x["loss"]}
                V = {int(x["epoca"]): float(x["val_exp"]) for x in rows if x["val_exp"]}
                curvas_l[(ds, pel, s)] = L
                ini = [L[i] for i in range(1, 13) if i in L]
                fin = [L[i] for i in range(97, 109) if i in L]
                imin = min(L, key=L.get)
                x = sel[(ds, pel, s)]
                ie = int(x["iteracion elegida"])
                medias[(ds, pel, s)] = (np.mean(ini), np.mean(fin), V.get(108))
                filas_l.append([ds, pel, s, f(L.get(1), 1), f(np.mean(ini), 1), f(np.mean(fin), 1), f(np.sqrt(np.mean(fin)), 1), f(L[imin], 1), imin, ie,
                                f(L.get(ie), 1) if ie in L else "", x["val elegida"], f(V.get(108), 1), x["val h_fs"]])
    write_csv(out / "perdida_R0_R1.csv", head_l, filas_l)
    head_lr = ["conjunto", "semilla", "perdida R1 sobre R0, iteraciones 1 a 12", "perdida R1 sobre R0, iteraciones 97 a 108", "validacion R1 sobre R0 en la 108"]
    filas_lr = [[ds, s, f(medias[(ds, "R1", s)][0] / medias[(ds, "R0", s)][0]), f(medias[(ds, "R1", s)][1] / medias[(ds, "R0", s)][1]),
                 f(medias[(ds, "R1", s)][2] / medias[(ds, "R0", s)][2])] for ds in DATASETS for s in range(3)]
    write_csv(out / "perdida_R1_sobre_R0.csv", head_lr, filas_lr)
    md.append("\n## Perdida de entrenamiento de R0 y R1\n\nPerdida TBTT contra h* sobre todas las trayectorias acumuladas en la iteracion, en saltos al cuadrado, de train.csv.\n\n")
    md.append(md_tabla(head_l, filas_l))
    md.append("\n")
    md.append(md_tabla(head_lr, filas_lr))

    # 6. tiempos limpios
    head_t = ["conjunto", "busqueda", "dispositivo", "mapas", "encontrados", "expansiones", "ms por nodo", "ms por busqueda", "ms por llamada a la U-Net",
              "mapas con las mismas expansiones que la prueba"]

    def tabla_tiempos(d):
        filas, tt = [], {}
        for ds in DATASETS:
            for archivo, dev, nombre in TIEMPOS:
                t = tiempos_de(d, dev, ds, archivo)
                if not t:
                    continue
                tt[(ds, archivo, dev)] = t
                pr = prueba_de(refs, refs2, esc, ds, archivo)
                ref = cargar_eval(pr) if pr.exists() else {}
                iguales = sum(g in ref and ref[g]["exp"] == e for g, e in t["exps"].items())
                filas.append([ds, nombre, dev, t["mapas"], t["encontrados"], f(t["exp"], 1), f(t["ms_nodo"]), f(t["ms_busqueda"], 1),
                              f(t["ms_red"], 1) if t["ms_red"] is not None else "", f"{iguales} de {t['mapas']}"])
        return filas, tt

    filas_t, tiempos = tabla_tiempos(tdir)
    write_csv(out / "tiempos.csv", head_t, filas_t)
    sin_fijar = out / "tiempos_sin_nucleo_fijo"
    filas_sf = tabla_tiempos(sin_fijar)[0] if sin_fijar.exists() else []
    if filas_sf:
        write_csv(out / "tiempos_sin_nucleo_fijo.csv", head_t, filas_sf)
    nucleos = leer_csv(tdir / "nucleos.csv") if (tdir / "nucleos.csv").exists() else []
    faltan = leer_csv(tdir / "faltantes.csv") if (tdir / "faltantes.csv").exists() else []
    entorno = leer_csv(tdir / "entorno.csv") if (tdir / "entorno.csv").exists() else []
    md.append("\n## Tiempos limpios, semilla 0, proceso fijo en un nucleo de rendimiento\n\n")
    md.append(md_tabla(head_t, filas_t))
    if nucleos:
        md.append("\nCosto de la U-Net de cf_filtro en 2 mapas de gaps_and_forest con el proceso fijo en cada nucleo logico, de tiempos/nucleos.csv\n\n")
        md.append(md_tabla(list(nucleos[0].keys()), [list(x.values()) for x in nucleos]))
    if filas_sf:
        md.append("\nPrimera medicion, sin fijar el nucleo, donde windows movio el proceso entre nucleos de rendimiento y de eficiencia. No se usa en la tabla principal\n\n")
        md.append(md_tabla(head_t, filas_sf))
    if faltan:
        md.append("\nSin checkpoint en esta maquina\n\n" + md_tabla(["conjunto", "metodo", "dispositivo", "motivo", "ruta"], [[x["dataset"], x["metodo"], x["dispositivo"], x["motivo"], x["ruta"]] for x in faltan]))
    if entorno:
        md.append("\nEntorno\n\n" + md_tabla(["clave", "valor"], [[x["clave"], x["valor"]] for x in entorno]))

    # figuras
    fig, ax = plt.subplots(figsize=(9, 5))
    res_esc = {(x["dataset"], x["modelo"]): x for x in leer_csv(esc / "resumen.csv")}
    xs = ["R0", "R1", "R2", "R3", "R4", "R5"]
    for ds, col in zip(DATASETS, ["C0", "C1", "C2"]):
        ys = [float(res_esc[(ds, r)]["rel_hfs"]) if (ds, r) in res_esc else np.nan for r in xs]
        ax.plot(range(len(xs)), ys, marker="o", color=col, label=ds)
        if (ds, "C") in res_esc:
            ax.scatter([len(xs) + 0.3], [float(res_esc[(ds, "C")]["rel_hfs"])], marker="*", s=180, color=col, edgecolor="k", zorder=5)
    ax.axhline(1.0, color="gray", ls="--", label="voraz con h_fs")
    ax.set_xticks(list(range(len(xs))) + [len(xs) + 0.3]); ax.set_xticklabels(["R0\nPHIL", "R1\nCNN de entrada", "R2\nCNN de base", "R3\nsin GRU", "R4\nsin GNN", "R5\ncf_filtro", "C\ncompuerta"], fontsize=8)
    ax.set_yscale("log"); ax.set_ylabel("expansiones sobre la voraz con h_fs"); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    ax.set_title("escalera entre PHIL y la CNN, prueba en 100 mapas, media de 3 semillas", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "escalera.png", dpi=130); plt.close(fig)

    al = leer_csv(esc / "figuras" / "alpha_stats.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, tramo_de, xl in ((axes[0], "hfs", "h_fs del nodo en celdas"), (axes[1], "avance", "avance de la busqueda")):
        for ds, col in zip(DATASETS, ["C0", "C1", "C2"]):
            rows = [x for x in al if x["dataset"] == ds and x["tramo_de"] == tramo_de]
            ys = [float(x["alpha_medio"]) if x["alpha_medio"] else np.nan for x in rows]
            et = f"{ds}, semilla {rows[0]['seed']}, iteracion {rows[0]['iteracion']}" if rows and "seed" in rows[0] else ds
            ax.plot(range(len(rows)), ys, marker="o", color=col, label=et)
            ax.set_xticks(range(len(rows))); ax.set_xticklabels([x["tramo"] if tramo_de == "hfs" else f"{int(100 * float(x['tramo'].split()[0]))} a {min(100, int(100 * float(x['tramo'].split()[-1])))} %" for x in rows], fontsize=8)
        ax.set_xlabel(xl); ax.set_ylabel("alpha medio, peso de phil_res_s"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    axes[0].set_title("alpha de la compuerta por tramo de h_fs", fontsize=10); axes[1].set_title("alpha de la compuerta por tramo de avance", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "alpha.png", dpi=130); plt.close(fig)

    if tiempos:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        etiquetas = [(archivo, dev, nombre) for archivo, dev, nombre in TIEMPOS if any((ds, archivo, dev) in tiempos for ds in DATASETS)]
        ancho = 0.8 / len(etiquetas)
        for ax, clave, yl in ((axes[0], "ms_nodo", "ms por nodo expandido"), (axes[1], "ms_busqueda", "ms por busqueda")):
            for k, (archivo, dev, nombre) in enumerate(etiquetas):
                ys = [tiempos[(ds, archivo, dev)][clave] if (ds, archivo, dev) in tiempos else np.nan for ds in DATASETS]
                ax.bar(np.arange(len(DATASETS)) + (k - (len(etiquetas) - 1) / 2) * ancho, ys, width=ancho, label=nombre)
            ax.set_xticks(range(len(DATASETS))); ax.set_xticklabels(DATASETS); ax.set_yscale("log"); ax.set_ylabel(yl); ax.grid(alpha=0.3, axis="y", which="both")
        fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=len(etiquetas), fontsize=8)
        axes[0].set_title("tiempo por nodo, semilla 0, un proceso y un hilo", fontsize=10); axes[1].set_title("tiempo por busqueda, media de los mapas de prueba", fontsize=10)
        fig.tight_layout(rect=[0, 0.07, 1, 1]); fig.savefig(out / "tiempos.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, ds in zip(axes, DATASETS):
        for pel, col in (("R0", "C0"), ("R1", "C3")):
            for s, ls in zip(range(3), ["-", "--", ":"]):
                L = curvas_l[(ds, pel, s)]
                its = sorted(L)
                ax.plot(its, [L[i] for i in its], color=col, ls=ls, lw=1, label=f"{pel} s{s}")
        ax.set_yscale("log"); ax.set_xlabel("iteracion"); ax.set_ylabel("perdida TBTT, saltos al cuadrado"); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7, ncol=2)
        ax.set_title(f"{ds}, perdida de entrenamiento de R0 y R1", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "perdida_R0_R1.png", dpi=120); plt.close(fig)

    md.append("\n## Figuras\n\n![escalera](escalera.png)\n\n![alpha](alpha.png)\n\n![tiempos](tiempos.png)\n\n![perdida](perdida_R0_R1.png)\n")
    (out / "tablas.md").write_text("".join(md), encoding="utf-8", newline="\n")
    print("".join(md))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
