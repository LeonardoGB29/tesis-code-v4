# exp_escalera. tablas, escalera de comparaciones, seleccion por corrida, curvas de validacion y borrador del resumen
#   python -m experiments.resumen_escalera --root results/exp_escalera
# lee de code_v3 las referencias de h_fs y h* en prueba, los csv de prueba de cf_filtro (R5), de PHIL viejo y de phil_res_s como control,
# y de exp_escalera las corridas de R0 a R4 y C
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.common import DATASETS, EXP_CF2, RESULTS, f, leer_csv, md_tabla, p_wilcoxon, write_csv
from experiments.resumen_cf import cargar_eval, medias_por_mapa
from experiments.train_escalera import ROOT_ESC

PELDANOS = ["R0", "R1", "R2", "R3", "R4", "R5", "C"]
NOMBRES = {"voraz_hstar": "voraz con h* (techo)", "voraz_hfs": "voraz con h_fs", "phil_viejo": "PHIL de exp_cf, 5 semillas", "phil_res_s": "phil_res_s de exp_cf",
           "R0": "R0 PHIL original", "R1": "R1 PHIL con la CNN de entrada", "R2": "R2 f GNN GRU y MLP sobre la base", "R3": "R3 f GNN y MLP sobre la base",
           "R4": "R4 f y MLP sobre la base", "R5": "R5 cf_filtro", "C": "C compuerta phil_res_s y base"}
ORDEN = ["voraz_hstar", "voraz_hfs", "phil_viejo", "phil_res_s"] + PELDANOS
# comparaciones de la escalera. tipo baja pide bajar mas de 5 % en 2 de 3 conjuntos con wilcoxon y 3 de 3 semillas, tipo igual pide no quedar mas de 5 % peor en ninguno
COMPARACIONES = [("R1", "R0", "baja", "la CNN aporta a PHIL como entrada"), ("R2", "R1", "baja", "la CNN aporta mas como base que como entrada"),
                 ("R3", "R2", "igual", "la CNN puede reemplazar a la GRU"), ("R4", "R3", "igual", "la CNN puede reemplazar a la GNN"),
                 ("R5", "R4", "igual", "la CNN puede reemplazar a todo PHIL"), ("C", "R5", "baja", "la compuerta aporta sobre cf_filtro")]


def corridas(root, refs, refs2):
    ev = defaultdict(dict)
    for ds in DATASETS:
        for ref in ("voraz_hfs", "voraz_hstar"):
            p = refs / "refs" / "eval" / ds / f"{ref}.csv"
            if p.exists():
                ev[(ds, ref)][0] = cargar_eval(p)
        for s in range(5):
            p = refs / "phil" / "eval" / ds / f"phil_s_s{s}.csv"
            if p.exists():
                ev[(ds, "phil_viejo")][s] = cargar_eval(p)
        for s in range(3):
            p = refs / "phil_res_s" / "eval" / ds / f"phil_res_s_s{s}.csv"
            if p.exists():
                ev[(ds, "phil_res_s")][s] = cargar_eval(p)
            p = refs2 / "cf_filtro" / "eval" / ds / f"cf_filtro_s{s}.csv"
            if p.exists():
                ev[(ds, "R5")][s] = cargar_eval(p)
            for r in ("R0", "R1", "R2", "R3", "R4", "C"):
                p = root / "eval" / ds / f"{r}_s{s}.csv"
                if p.exists():
                    ev[(ds, r)][s] = cargar_eval(p)
    return ev


def meta_de(root, ds, r, s):
    d = root / "runs" / ds / f"{r}_s{s}"
    if not (d / "config.json").exists():
        return None
    m = json.loads((d / "config.json").read_text())
    rows = leer_csv(d / "train.csv") if (d / "train.csv").exists() else []
    vals = [x for x in rows if x["val_exp"]]
    m["_vals"] = vals
    m["_min_train"] = sum(float(x["secs_train"] or 0) for x in rows) / 60
    m["_min_val"] = sum(float(x["secs_val"] or 0) for x in rows) / 60
    m["_listo"] = (d / "listo").exists()
    return m


def resumen_modelo(ev, ds, modelo, ref_hfs):
    evs = ev.get((ds, modelo), {})
    if not evs:
        return None
    seeds = sorted(evs)
    exp_m, cam_m = medias_por_mapa(evs, seeds)
    grids = sorted(exp_m)
    por_seed = {s: float(np.mean([evs[s][g]["exp"] for g in grids])) for s in seeds}
    tot_exp = sum(evs[s][g]["exp"] for s in seeds for g in grids)
    tot_secs = sum(evs[s][g]["secs"] for s in seeds for g in grids)
    caminos = [evs[s][g]["camino"] / max(evs[s][g]["opt"], 1) for s in seeds for g in grids if evs[s][g]["enc"]]
    r = {"dataset": ds, "modelo": modelo, "seeds": " ".join(map(str, seeds)), "n_mapas": len(grids), "exp": float(np.mean([exp_m[g] for g in grids])),
         "exp_sd_semillas": float(np.std(list(por_seed.values()))) if len(seeds) > 1 else 0.0,
         "exito": float(np.mean([evs[s][g]["enc"] for s in seeds for g in grids])), "camino_opt": float(np.mean(caminos)) if caminos else float("nan"),
         "ms_nodo": 1000 * tot_secs / max(tot_exp, 1), "epocas": " ".join(str(next(iter(evs[s].values()))["epoca"]) for s in seeds), "_exp_m": exp_m, "_cam_m": cam_m, "_por_seed": por_seed}
    if ref_hfs:
        hf = ref_hfs[0]
        g2 = [g for g in grids if g in hf]
        ref_exp = float(np.mean([hf[g]["exp"] for g in g2]))
        r["rel_hfs"] = r["exp"] / ref_exp
        r["rel_hfs_por_semilla"] = " ".join(f"{por_seed[s] / ref_exp:.3f}" for s in seeds)
        r["semillas_mejoran_hfs"] = sum(por_seed[s] < ref_exp for s in seeds)
        r["p_hfs_exp"] = p_wilcoxon([exp_m[g] for g in g2], [hf[g]["exp"] for g in g2])
        cam_ref = {g: hf[g]["camino"] / max(hf[g]["opt"], 1) for g in g2 if hf[g]["enc"]}
        g3 = [g for g in g2 if g in cam_m and g in cam_ref]
        r["camino_vs_hfs_pct"] = 100 * (np.mean([cam_m[g] for g in g3]) / np.mean([cam_ref[g] for g in g3]) - 1) if g3 else float("nan")
    return r


def comparar(ra, rb):
    # a contra b por mapa sobre las medias de semillas, mas la relacion por semilla emparejada
    if not ra or not rb:
        return None
    g = sorted(set(ra["_exp_m"]) & set(rb["_exp_m"]))
    xa, xb = np.array([ra["_exp_m"][x] for x in g]), np.array([rb["_exp_m"][x] for x in g])
    sa, sb = ra["_por_seed"], rb["_por_seed"]
    comunes = sorted(set(sa) & set(sb))
    rel_seed = [sa[s] / sb[s] for s in comunes]
    gc = [x for x in g if x in ra["_cam_m"] and x in rb["_cam_m"]]
    cam = (np.mean([ra["_cam_m"][x] for x in gc]) / np.mean([rb["_cam_m"][x] for x in gc])) if gc else float("nan")
    return {"n": len(g), "exp_a": float(xa.mean()), "exp_b": float(xb.mean()), "rel": float(xa.mean() / xb.mean()), "p": p_wilcoxon(xa, xb),
            "gana_a": int((xa < xb).sum()), "rel_semillas": rel_seed, "semillas_bajo_1": sum(x < 1 for x in rel_seed), "camino_rel": float(cam)}


def veredicto(tipo, c):
    if not c:
        return None
    camino_ok = not (c["camino_rel"] == c["camino_rel"]) or c["camino_rel"] <= 1.05
    if tipo == "baja":
        return bool(c["rel"] < 0.95 and c["p"] < 0.05 and c["rel_semillas"] and c["semillas_bajo_1"] == len(c["rel_semillas"]) and camino_ok)
    return bool(c["rel"] <= 1.05 and camino_ok)


def curvas(root, metas):
    fig, axes = plt.subplots(len(PELDANOS) - 1, 3, figsize=(16, 3.2 * (len(PELDANOS) - 1)), squeeze=False)
    filas = [r for r in PELDANOS if r != "R5"]
    for i, r in enumerate(filas):
        for j, ds in enumerate(DATASETS):
            ax = axes[i, j]
            for s, col in zip(range(3), ["C0", "C1", "C2"]):
                m = metas.get((ds, r, s))
                if not m or not m["_vals"]:
                    continue
                it = [int(x["epoca"]) for x in m["_vals"]]; vx = [float(x["val_exp"]) for x in m["_vals"]]
                ax.plot(it, vx, color=col, marker="o", ms=3, label=f"s{s} elegida {m.get('best_epoca')}")
                der = [int(x["epoca"]) for x in m["_vals"] if x["derrumbe"] == "1"]
                if der:
                    ax.scatter(der, [vx[it.index(k)] for k in der], color=col, marker="x", s=40)
                if m.get("val0") is not None:
                    ax.axhline(m["val0"], color=col, ls=":", alpha=0.4)
                if m.get("ref_val"):
                    ax.axhline(m["ref_val"]["exp"], color="gray", ls="--", alpha=0.6)
            ax.set_title(f"{ds}, {NOMBRES[r]}", fontsize=9); ax.set_xlabel("iteracion"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
            ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(root / "curvas_validacion.png", dpi=110)
    plt.close(fig)


def escalera_fig(root, res):
    fig, ax = plt.subplots(figsize=(9, 5))
    xs = ["R0", "R1", "R2", "R3", "R4", "R5"]
    for ds, col in zip(DATASETS, ["C0", "C1", "C2"]):
        ys = [res.get((ds, r), {}).get("rel_hfs", np.nan) for r in xs]
        ax.plot(range(len(xs)), ys, marker="o", color=col, label=ds)
        c = res.get((ds, "C"), {}).get("rel_hfs")
        if c is not None:
            ax.scatter([len(xs) + 0.3], [c], marker="*", s=180, color=col, edgecolor="k", zorder=5)
    ax.axhline(1.0, color="gray", ls="--", label="voraz con h_fs")
    ax.set_xticks(list(range(len(xs))) + [len(xs) + 0.3]); ax.set_xticklabels(xs + ["C"])
    ax.set_yscale("log"); ax.set_ylabel("expansiones sobre la voraz con h_fs, prueba"); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    ax.set_title("escalera de reemplazo entre PHIL y la CNN, 100 mapas de prueba, media de las semillas")
    (root / "figuras").mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(root / "figuras" / "escalera.png", dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(ROOT_ESC))
    p.add_argument("--refs", default=str(RESULTS))
    p.add_argument("--refs2", default=str(EXP_CF2))
    a = p.parse_args()
    root, refs, refs2 = Path(a.root), Path(a.refs), Path(a.refs2)
    ev = corridas(root, refs, refs2)
    res, filas = {}, []
    for ds in DATASETS:
        for m in ORDEN:
            r = resumen_modelo(ev, ds, m, ev.get((ds, "voraz_hfs")))
            if r:
                res[(ds, m)] = r; filas.append(r)
    cols = ["dataset", "modelo", "seeds", "n_mapas", "exp", "exp_sd_semillas", "rel_hfs", "rel_hfs_por_semilla", "semillas_mejoran_hfs", "p_hfs_exp", "exito", "camino_opt",
            "camino_vs_hfs_pct", "ms_nodo", "epocas"]
    write_csv(root / "resumen.csv", cols, [[f(r.get(c)) if isinstance(r.get(c), float) else r.get(c, "") for c in cols] for r in filas])

    # escalera de comparaciones
    filas_esc, pasa = [], {}
    for (ra, rb, tipo, que) in COMPARACIONES:
        for ds in DATASETS:
            c = comparar(res.get((ds, ra)), res.get((ds, rb)))
            v = veredicto(tipo, c)
            pasa[(ra, rb, ds)] = v
            if c:
                filas_esc.append([ra, rb, ds, tipo, c["n"], f(c["exp_a"], 1), f(c["exp_b"], 1), f(c["rel"]), f(100 * (1 - c["rel"]), 1), f(c["p"], 4), c["gana_a"],
                                  " ".join(f"{x:.3f}" for x in c["rel_semillas"]), f"{c['semillas_bajo_1']} de {len(c['rel_semillas'])}", f(c["camino_rel"]), int(bool(v))])
            else:
                filas_esc.append([ra, rb, ds, tipo, 0, "", "", "", "", "", "", "", "", "", ""])
    head_esc = ["a", "b", "conjunto", "criterio", "mapas", "exp a", "exp b", "a sobre b", "mejora %", "p Wilcoxon", "mapas donde gana a", "a sobre b por semilla", "semillas < 1", "camino a sobre b", "pasa en el conjunto"]
    write_csv(root / "escalera.csv", head_esc, filas_esc)
    veredictos = []
    for (ra, rb, tipo, que) in COMPARACIONES:
        vs = [pasa.get((ra, rb, ds)) for ds in DATASETS]
        n_ok = sum(bool(v) for v in vs)
        hechos = sum(v is not None for v in vs)
        if tipo == "baja":
            ok = n_ok >= 2
            texto = f"{ra} contra {rb}, {que}. {'pasa' if ok else 'no pasa'}, baja mas de 5 % con Wilcoxon y 3 de 3 semillas en {n_ok} de {hechos} conjuntos con datos"
        else:
            ok = hechos > 0 and n_ok == hechos
            texto = f"{ra} contra {rb}, {que}. {'pasa' if ok else 'no pasa'}, no queda mas de 5 % peor en {n_ok} de {hechos} conjuntos con datos"
        veredictos.append(texto)

    # seleccion por corrida
    metas = {(ds, r, s): meta_de(root, ds, r, s) for ds in DATASETS for r in PELDANOS if r != "R5" for s in range(3)}
    filas_sel = []
    for (ds, r, s), m in sorted(metas.items(), key=lambda kv: (PELDANOS.index(kv[0][1]), DATASETS.index(kv[0][0]), kv[0][2])):
        if not m:
            continue
        filas_sel.append([ds, r, s, m.get("best_epoca", ""), f(m.get("best_val"), 1), f(m.get("val0"), 1), f((m.get("ref_val") or {}).get("exp"), 1),
                          f(((m.get("val_cf_mac") or {}).get("exp")), 1) if m.get("val_cf_mac") else "", m.get("mejor_aprendida", ""), f(m.get("mejor_aprendida_val"), 1),
                          int(bool(m.get("queda_cf"))), int(bool(m.get("sin_elegibles"))), int(bool(m.get("corte"))), m.get("epoca_final", ""), len(m["_vals"]),
                          m.get("n_entrenables", ""), round(m["_min_train"], 1), round(m["_min_val"], 1), int(m["_listo"])])
    head_sel = ["conjunto", "peldano", "semilla", "iteracion elegida", "val elegida", "val iteracion 0", "val h_fs", "val cf_filtro mac", "mejor aprendida", "val mejor aprendida",
                "queda en cf_filtro", "sin elegibles", "corte", "iteracion final", "validaciones", "parametros entrenables", "min entrenamiento", "min validacion", "listo"]
    write_csv(root / "seleccion.csv", head_sel, filas_sel)

    # tiempos por nodo en prueba y en la iteracion 0 de las verificaciones
    filas_t = []
    for ds in DATASETS:
        for m in ORDEN:
            r = res.get((ds, m))
            if r:
                filas_t.append([ds, NOMBRES[m], f(r["exp"], 1), f(r["ms_nodo"])])
    for pth in sorted((root / "checks").glob("tiempos_epoca0_*_s*.csv")) if (root / "checks").exists() else []:
        for x in leer_csv(pth):
            filas_t.append([x["dataset"], f"iteracion 0 de {x['brazo']} s{x['seed']}, u-net en el acelerador", x["exp_media"], x["ms_por_nodo"]])
    head_t = ["conjunto", "busqueda", "expansiones", "ms por nodo"]

    try:
        curvas(root, metas)
        escalera_fig(root, res)
    except Exception as e:
        print(f"figuras del resumen fallaron, {e}")

    md = [f"# Tablas de {root.name}\n", "Prueba en los 100 mapas de cada conjunto, 91 resolubles en gaps_and_forest, tope de 10 000 expansiones. Relacion es expansiones sobre las de la voraz con h_fs. "
          "Camino es el largo sobre el optimo en los mapas encontrados. Wilcoxon pareado por mapa sobre la media de las semillas. ms por nodo cuenta la red y depende de la carga de la maquina.\n"]
    head = ["busqueda", "semillas", "expansiones", "sd entre semillas", "relacion con h_fs", "por semilla", "semillas < 1", "p Wilcoxon", "exito", "camino sobre optimo", "camino vs h_fs %", "ms por nodo", "iteracion elegida"]
    for ds in DATASETS:
        md.append(f"\n## {ds}, prueba\n")
        fl = []
        for m in ORDEN:
            r = res.get((ds, m))
            if r:
                fl.append([NOMBRES[m], r["seeds"], f(r["exp"], 1), f(r["exp_sd_semillas"], 1), f(r.get("rel_hfs")), r.get("rel_hfs_por_semilla", ""), r.get("semillas_mejoran_hfs", ""),
                           f(r.get("p_hfs_exp"), 4), f(r["exito"]), f(r["camino_opt"]), f(r.get("camino_vs_hfs_pct"), 2), f(r["ms_nodo"]), r["epocas"]])
        md.append(md_tabla(head, fl))
    md.append("\n## Escalera de comparaciones\n")
    md.append(md_tabla(head_esc, filas_esc))
    md.append("\n" + "\n".join(f"- {v}." for v in veredictos) + "\n")
    md.append("\n## Seleccion por corrida\n")
    md.append(md_tabla(head_sel, filas_sel))
    md.append("\n## Tiempo por nodo\n")
    md.append(md_tabla(head_t, filas_t))
    (root / "tablas.md").write_text("".join(md))

    md = [f"# Resumen de {root.name}\n", f"Carpeta tesis/code_v4, resultados en {root}. Borrador generado por resumen_escalera.py, el RESUMEN.md a mano va aparte.\n",
          "\n## Escalera en prueba, relacion de expansiones con la voraz con h_fs\n"]
    head2 = ["conjunto"] + PELDANOS
    fl = [[ds] + [f(res.get((ds, r), {}).get("rel_hfs")) for r in PELDANOS] for ds in DATASETS]
    md.append(md_tabla(head2, fl))
    md.append("\n![escalera](figuras/escalera.png)\n")
    md.append("\n## Comparaciones\n")
    md.append(md_tabla(head_esc, filas_esc))
    md.append("\n" + "\n".join(f"- {v}." for v in veredictos) + "\n")
    md.append("\n## Seleccion por corrida\n")
    md.append(md_tabla(head_sel, filas_sel))
    md.append("\n## Figuras\n\n![curvas de validacion](curvas_validacion.png)\n\nMapas de alpha en figuras/alpha_<conjunto>.png y alpha por tramo en figuras/alpha_tramos.png si figuras_escalera.py corrio.\n")
    md.append("\n## Que funciono\n\n(pendiente de revisar)\n\n## Que no funciono\n\n(pendiente de revisar)\n\n## Lo que necesita tu decision\n\n(pendiente de revisar)\n")
    (root / "RESUMEN_auto.md").write_text("".join(md))
    print("".join(md))
    print(f"-> {root / 'tablas.md'}\n-> {root / 'RESUMEN_auto.md'}")


if __name__ == "__main__":
    main()
