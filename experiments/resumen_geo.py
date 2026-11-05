# tablas, curvas y borrador del resumen de exp_geo
#   python -m experiments.resumen_geo --root results/exp_geo
# lee de code_v3 las referencias de h_fs y h* en prueba, los csv de prueba y los train.csv de cf_filtro, y de exp_geo todo lo demas
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.common import DATASETS, EXP_CF2, RESULTS, RESULTS_GEO, f, leer_csv, md_tabla, p_wilcoxon, write_csv
from experiments.resumen_cf import cargar_eval, medias_por_mapa

NOMBRES = {"voraz_hstar": "voraz con h* (techo)", "voraz_hfs": "voraz con h_fs", "cf_filtro": "voraz con h_fs / cf_filtro", "cf_filtro_3": "cf_filtro semillas 0 a 2",
           "geo": "voraz con h_geo", "geo_real": "voraz con h_geo sobre la ocupacion real"}
ORDEN = ["voraz_hstar", "voraz_hfs", "cf_filtro", "cf_filtro_3", "geo", "geo_real"]


def corridas(root):
    ev = defaultdict(dict)
    for ds in DATASETS:
        for ref in ("voraz_hfs", "voraz_hstar"):
            p = RESULTS / "refs" / "eval" / ds / f"{ref}.csv"
            if p.exists():
                ev[(ds, ref)][0] = cargar_eval(p)
        for s in range(5):
            p = EXP_CF2 / "cf_filtro" / "eval" / ds / f"cf_filtro_s{s}.csv"
            if p.exists():
                ev[(ds, "cf_filtro")][s] = cargar_eval(p)
                if s < 3:
                    ev[(ds, "cf_filtro_3")][s] = cargar_eval(p)
        for s in range(5):
            p = root / "eval" / ds / f"geo_s{s}.csv"
            if p.exists():
                ev[(ds, "geo")][s] = cargar_eval(p)
        p = root / "eval" / ds / "geo_real.csv"
        if p.exists():
            ev[(ds, "geo_real")][0] = cargar_eval(p)
    return ev


def resumen_modelo(ev, ds, modelo, ref_hfs):
    evs = ev.get((ds, modelo), {})
    if not evs:
        return None
    seeds = sorted(evs)
    exp_m, cam_m = medias_por_mapa(evs, seeds)
    grids = sorted(exp_m)
    exp = float(np.mean([exp_m[g] for g in grids]))
    por_seed = {s: float(np.mean([evs[s][g]["exp"] for g in grids])) for s in seeds}
    tot_exp = sum(evs[s][g]["exp"] for s in seeds for g in grids)
    tot_secs = sum(evs[s][g]["secs"] for s in seeds for g in grids)
    tot_red = sum(evs[s][g]["secs_red"] for s in seeds for g in grids)
    caminos = [evs[s][g]["camino"] / max(evs[s][g]["opt"], 1) for s in seeds for g in grids if evs[s][g]["enc"]]
    r = {"dataset": ds, "modelo": modelo, "seeds": " ".join(map(str, seeds)), "n_mapas": len(grids), "exp": exp,
         "exp_sd_semillas": float(np.std(list(por_seed.values()))) if len(seeds) > 1 else 0.0,
         "exito": float(np.mean([evs[s][g]["enc"] for s in seeds for g in grids])), "camino_opt": float(np.mean(caminos)) if caminos else float("nan"),
         "ms_nodo": 1000 * tot_secs / max(tot_exp, 1), "ms_nodo_red": 1000 * tot_red / max(tot_exp, 1),
         "epocas": " ".join(str(next(iter(evs[s].values()))["epoca"]) for s in seeds)}
    if ref_hfs:
        hf = ref_hfs[0]
        g2 = [g for g in grids if g in hf]
        ref_exp = float(np.mean([hf[g]["exp"] for g in g2]))
        r["rel_hfs"] = exp / ref_exp
        r["rel_hfs_por_semilla"] = " ".join(f"{por_seed[s] / ref_exp:.3f}" for s in seeds)
        r["semillas_mejoran_hfs"] = sum(por_seed[s] < ref_exp for s in seeds)
        r["mejora_hfs_pct"] = 100 * (1 - exp / ref_exp)
        r["p_hfs_exp"] = p_wilcoxon([exp_m[g] for g in g2], [hf[g]["exp"] for g in g2])
        cam_ref = {g: hf[g]["camino"] / max(hf[g]["opt"], 1) for g in g2 if hf[g]["enc"]}
        g3 = [g for g in g2 if g in cam_m and g in cam_ref]
        r["camino_hfs"] = float(np.mean([cam_ref[g] for g in g3])) if g3 else float("nan")
        r["camino_vs_hfs_pct"] = 100 * (np.mean([cam_m[g] for g in g3]) / np.mean([cam_ref[g] for g in g3]) - 1) if g3 else float("nan")
        r["p_hfs_camino"] = p_wilcoxon([cam_m[g] for g in g3], [cam_ref[g] for g in g3]) if g3 else 1.0
    return r


def regla_paso2(res):
    det, pasan = [], 0
    for ds in DATASETS:
        r = res.get((ds, "geo"))
        if not r or "rel_hfs" not in r:
            det.append(f"{ds} sin datos"); continue
        n = len(r["seeds"].split())
        ok = r["mejora_hfs_pct"] > 10 and r["p_hfs_exp"] < 0.05 and r["semillas_mejoran_hfs"] == n and r["camino_vs_hfs_pct"] <= 5
        pasan += ok
        det.append(f"{ds} {'pasa' if ok else 'no pasa'} (mejora {r['mejora_hfs_pct']:.1f} %, p {r['p_hfs_exp']:.3g}, camino {r['camino_vs_hfs_pct']:+.1f} %, "
                   f"semillas < 1 en {r['semillas_mejoran_hfs']} de {n})")
    return pasan >= 2, pasan, det


def comparar(ev, ds, a, b):
    # a contra b por mapa, medias de semillas
    ea, eb = ev.get((ds, a), {}), ev.get((ds, b), {})
    if not ea or not eb:
        return None
    ma, _ = medias_por_mapa(ea, sorted(ea)); mb, _ = medias_por_mapa(eb, sorted(eb))
    g = sorted(set(ma) & set(mb))
    xa, xb = np.array([ma[x] for x in g]), np.array([mb[x] for x in g])
    return {"n": len(g), "exp_a": float(xa.mean()), "exp_b": float(xb.mean()), "rel": float(xa.mean() / xb.mean()), "p": p_wilcoxon(xa, xb), "gana_a": int((xa < xb).sum())}


def lambdas(root):
    out = {}
    for ds in DATASETS:
        p = root / "barrido" / ds / "lambda.json"
        if p.exists():
            out[ds] = json.loads(p.read_text())
    return out


def val_resumen(path):
    rows = leer_csv(path)
    enc = [r for r in rows if r["encontrado"] == "1"]
    return {"exp": float(np.mean([float(r["exp"]) for r in rows])), "camino": float(np.mean([int(r["camino"]) / max(int(r["optimo"]), 1) for r in enc])) if enc else float("nan"),
            "encontrados": len(enc)}


def base_validacion(root, lams):
    # h_geo con lambda fijo, semillas 0 a 2 en validacion, contra la validacion de la epoca elegida de cf_filtro semillas 0 a 2
    filas, base = [], {}
    for ds in DATASETS:
        if ds not in lams:
            continue
        lam = lams[ds]["lambda"]
        geo = []
        p0 = root / "barrido" / ds / f"geo_s0_lam{lam:g}.csv"
        if p0.exists():
            geo.append(val_resumen(p0))
        for s in (1, 2):
            p = root / "val" / ds / f"geo_s{s}.csv"
            if p.exists():
                geo.append(val_resumen(p))
        cf = []
        for s in (0, 1, 2):
            d = EXP_CF2 / "cf_filtro" / "runs" / ds / f"cf_filtro_s{s}"
            if not (d / "config.json").exists():
                continue
            meta = json.loads((d / "config.json").read_text())
            be = meta.get("best_epoca")
            fila = next((r for r in leer_csv(d / "train.csv") if r["epoca"] == str(be) and r["val_exp"]), None)
            if fila:
                cf.append({"exp": float(fila["val_exp"]), "camino": float(fila["val_camino"]), "encontrados": int(fila["val_encontrados"])})
        ref = lams[ds]["ref_val_hfs"]
        if not geo or not cf:
            continue
        g_exp, g_cam = np.mean([x["exp"] for x in geo]), np.mean([x["camino"] for x in geo])
        c_exp, c_cam = np.mean([x["exp"] for x in cf]), np.mean([x["camino"] for x in cf])
        if g_exp < c_exp:
            ganador = "h_geo" if g_cam <= 1.05 * c_cam else "cf_filtro"
        else:
            ganador = "cf_filtro" if c_cam <= 1.05 * g_cam else "h_geo"
        base[ds] = {"base": ganador, "lambda": lam, "h_geo": {"exp": float(g_exp), "camino": float(g_cam), "semillas": len(geo), "por_semilla": [round(x["exp"], 1) for x in geo]},
                    "cf_filtro": {"exp": float(c_exp), "camino": float(c_cam), "semillas": len(cf), "por_semilla": [round(x["exp"], 1) for x in cf]}, "ref_hfs": ref}
        filas.append([ds, f(ref["exp"], 1), f(ref["camino"]), f"{lam:g}", f(g_exp, 1), " ".join(str(x) for x in base[ds]["h_geo"]["por_semilla"]), f(g_cam), f(g_exp / ref["exp"]),
                      f(c_exp, 1), " ".join(str(x) for x in base[ds]["cf_filtro"]["por_semilla"]), f(c_cam), f(c_exp / ref["exp"]), ganador])
    head = ["conjunto", "h_fs exp", "h_fs camino", "lambda", "h_geo exp", "h_geo por semilla", "h_geo camino", "h_geo rel", "cf_filtro exp", "cf_filtro por semilla",
            "cf_filtro camino", "cf_filtro rel", "base elegida"]
    (root / "base.json").write_text(json.dumps(base, indent=1))
    return head, filas, base


def metricas(root):
    # por conjunto y semilla, previo contra mejor epoca en todo lo no visto y en el corredor
    filas, paso1 = [], {}
    for ds in DATASETS:
        for s in range(5):
            p = root / "metricas" / ds / f"geo_s{s}.csv"
            if not p.exists():
                continue
            rows = leer_csv(p)
            por = {(r["modelo"], r["estrato"]): r for r in rows}
            for e in ("todo", "corredor", "todo_hfs", "todo_cf", "frontera_1", "frontera_2_3", "frontera_4_7", "frontera_8_15", "frontera_16_mas"):
                pr, mj = por.get(("previo", e)), por.get(("mejor", e))
                if not pr or not mj:
                    continue
                gana = int(float(mj["bce"]) < float(pr["bce"]) - 1e-9)
                if e == "todo":
                    paso1.setdefault(ds, []).append(gana)
                filas.append([ds, s, e, pr["n_celdas"], f(float(pr["frac_obst"])), f(float(pr["bce"]), 4), f(float(mj["bce"]), 4),
                              f(100 * (1 - float(mj["bce"]) / float(pr["bce"])), 1), mj["auc"], f(float(pr["ece"]), 4), f(float(mj["ece"]), 4), gana])
    head = ["conjunto", "semilla", "estrato", "celdas", "frac obst", "bce previo", "bce cnn", "mejora bce %", "auc cnn", "ece previo", "ece cnn", "gana al previo"]
    return head, filas, paso1


def seleccion(root):
    filas = []
    for ds in DATASETS:
        for s in range(5):
            d = root / "runs" / ds / f"geo_s{s}"
            if not (d / "train.csv").exists():
                continue
            rows = leer_csv(d / "train.csv")
            meta = json.loads((d / "config.json").read_text()) if (d / "config.json").exists() else {}
            vals = [r for r in rows if r["val_bce"]]
            filas.append([ds, s, meta.get("best_epoca", ""), f(meta.get("best_val"), 5), f(meta.get("val_previo"), 5), f(meta.get("p0"), 4), meta.get("n_fotos", ""), meta.get("n_fotos_cf", ""),
                          meta.get("n_fotos_val", ""), meta.get("epoca_final", ""), int(bool(meta.get("corte"))),
                          f(float(rows[1]["loss"]), 5) if len(rows) > 1 and rows[1]["loss"] else "", f(float(rows[-1]["loss"]), 5) if rows[-1]["loss"] else "",
                          round(sum(float(r["secs_train"] or 0) for r in rows) / 60, 1), round(sum(float(r["secs_val"] or 0) for r in rows) / 60, 1)])
    head = ["conjunto", "semilla", "epoca elegida", "bce val elegida", "bce val previo", "p0", "fotos", "fotos de cf_filtro", "fotos val", "epoca final", "corte",
            "loss epoca 1", "loss final", "min entrenamiento", "min validacion"]
    write_csv(root / "seleccion.csv", head, filas)
    return head, filas


def tiempos(root):
    filas = []
    for ds in DATASETS:
        p = root / "checks" / f"tiempos_{ds}.csv"
        if p.exists():
            for r in leer_csv(p):
                filas.append([ds, r["busqueda"], r["mapas"], r["ms_nodo"], r["ms_nodo_red"], r["s_mapa"]])
    return ["conjunto", "busqueda", "mapas", "ms por nodo", "ms red por nodo", "s por mapa"], filas


def curvas(root):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for j, ds in enumerate(DATASETS):
        prev = None
        for s, col in zip(range(5), ["C0", "C1", "C2", "C3", "C4"]):
            p = root / "runs" / ds / f"geo_s{s}" / "train.csv"
            if not p.exists():
                continue
            rows = leer_csv(p)
            ep = [int(r["epoca"]) for r in rows if r["loss"]]; ls = [float(r["loss"]) for r in rows if r["loss"]]
            axes[0, j].plot(ep, ls, color=col, marker="o", ms=3, label=f"s{s}")
            ev = [int(r["epoca"]) for r in rows if r["val_bce"]]; vb = [float(r["val_bce"]) for r in rows if r["val_bce"]]
            axes[1, j].plot(ev, vb, color=col, marker="o", ms=3, label=f"s{s}")
            if vb and prev is None:
                prev = vb[0]
        if prev is not None:
            axes[1, j].axhline(prev, color="gray", ls="--", label="previo constante")
        axes[0, j].set_title(f"{ds}, bce de entrenamiento"); axes[1, j].set_title(f"{ds}, bce de validacion en lo no visto")
        for ax in axes[:, j]:
            ax.set_xlabel("epoca"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root / "curvas_entrenamiento.png", dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(RESULTS_GEO))
    a = p.parse_args()
    root = Path(a.root)
    ev = corridas(root)
    res, filas = {}, []
    for ds in DATASETS:
        for m in ORDEN:
            r = resumen_modelo(ev, ds, m, ev.get((ds, "voraz_hfs")))
            if r:
                res[(ds, m)] = r; filas.append(r)
    cols = ["dataset", "modelo", "seeds", "n_mapas", "exp", "exp_sd_semillas", "rel_hfs", "rel_hfs_por_semilla", "semillas_mejoran_hfs", "mejora_hfs_pct", "p_hfs_exp",
            "exito", "camino_opt", "camino_hfs", "camino_vs_hfs_pct", "p_hfs_camino", "ms_nodo", "ms_nodo_red", "epocas"]
    write_csv(root / "resumen.csv", cols, [[f(r.get(c)) if isinstance(r.get(c), float) else r.get(c, "") for c in cols] for r in filas])
    lams = lambdas(root)
    head_base, filas_base, base = base_validacion(root, lams)
    head_met, filas_met, paso1 = metricas(root)
    head_sel, filas_sel = seleccion(root)
    head_t, filas_t = tiempos(root)
    ok2, n2, det2 = regla_paso2(res)
    try:
        curvas(root)
    except Exception as e:
        print(f"curvas fallaron, {e}")

    md = [f"# Tablas de {root.name}\n", "Prueba en los 100 mapas de cada conjunto, 91 resolubles en gaps_and_forest, tope de 10 000 expansiones. Relacion es expansiones sobre las de la voraz con h_fs. "
          "Camino es el largo sobre el optimo en los mapas encontrados. Wilcoxon pareado por mapa sobre la media de las semillas. ms por nodo cuenta la red y depende de la carga de la maquina.\n"]
    for ds in DATASETS:
        md.append(f"\n## {ds}, prueba\n")
        head = ["busqueda", "semillas", "expansiones", "sd entre semillas", "relacion con h_fs", "por semilla", "semillas < 1", "mejora %", "p Wilcoxon exp", "exito",
                "camino sobre optimo", "camino vs h_fs %", "p camino", "ms por nodo", "ms red", "epoca elegida"]
        fl = []
        for m in ORDEN:
            r = res.get((ds, m))
            if r:
                fl.append([NOMBRES[m], r["seeds"], f(r["exp"], 1), f(r["exp_sd_semillas"], 1), f(r.get("rel_hfs")), r.get("rel_hfs_por_semilla", ""), r.get("semillas_mejoran_hfs", ""),
                           f(r.get("mejora_hfs_pct"), 1), f(r.get("p_hfs_exp"), 4), f(r["exito"]), f(r["camino_opt"]), f(r.get("camino_vs_hfs_pct"), 2), f(r.get("p_hfs_camino"), 4),
                           f(r["ms_nodo"]), f(r["ms_nodo_red"]), r["epocas"]])
        md.append(md_tabla(head, fl))
        c = comparar(ev, ds, "geo", "cf_filtro")
        if c:
            md.append(f"\nh_geo contra cf_filtro por mapa, {c['n']} mapas, {f(c['exp_a'], 1)} contra {f(c['exp_b'], 1)}, relacion {f(c['rel'])}, p {f(c['p'], 4)}, h_geo gana en {c['gana_a']} de {c['n']}.\n")
        c = comparar(ev, ds, "geo", "geo_real")
        if c:
            md.append(f"\nh_geo con la cnn contra h_geo con la ocupacion real, relacion {f(c['rel'])}, p {f(c['p'], 4)}.\n")
    md.append("\n## Regla del paso 2\n")
    md.append(f"- h_geo {'pasa' if ok2 else 'no pasa'}, {n2} de 3 conjuntos. " + ". ".join(det2) + ".\n")
    md.append("\n## Barrido de lambda en validacion, semilla 0, 70 mapas. real es la ocupacion real en vez de la cnn\n")
    for ds, L in lams.items():
        ref = L["ref_val_hfs"]
        md.append(f"\n{ds}. h_fs expande {f(ref['exp'], 1)} con camino {f(ref['camino'])}. lambda elegido {L['lambda']:g}. lambda 0 igual a h_fs por mapa {L.get('lambda_cero_igual_hfs')}.\n\n")
        md.append(md_tabla(["quien", "lambda", "expansiones", "relacion con h_fs", "camino", "encontrados", "elegible", "ms por nodo"],
                           [[r["quien"], f"{r['lambda']:g}", f(r["exp"], 1), f(r["rel_hfs"]), f(r["camino"]), r["encontrados"], r["elegible"], f(r["ms_nodo"], 2)] for r in L["tabla"]]))
    md.append("\n## Base del paso 3, elegida en validacion. Gana la de menos expansiones si su camino no es mas de 5 % peor que el de la otra\n")
    md.append(md_tabla(head_base, filas_base))
    md.append("\n## Paso 1. La cnn de geometria contra el previo constante, fotos de validacion, celdas no vistas\n")
    md.append(md_tabla(head_met, filas_met))
    for ds, g in paso1.items():
        md.append(f"- {ds}. gana al previo en {sum(g)} de {len(g)} semillas.\n")
    md.append("\n## Seleccion por semilla. Epoca de menor bce en validacion\n")
    md.append(md_tabla(head_sel, filas_sel))
    md.append("\n## Tiempo por nodo a solas, 10 mapas de prueba, antes de lanzar\n")
    md.append(md_tabla(head_t, filas_t))
    (root / "tablas.md").write_text("".join(md))

    md = [f"# Resumen de {root.name}\n", f"Carpeta tesis/code_v4, resultados en {root}. Borrador generado por resumen_geo.py, el RESUMEN.md a mano va aparte.\n",
          "\n## Paso 1. La cnn de geometria contra el previo constante\n"]
    fl = [r for r in filas_met if r[2] in ("todo", "corredor")]
    md.append(md_tabla(head_met, fl))
    for ds, g in paso1.items():
        md.append(f"- {ds}. gana al previo en {sum(g)} de {len(g)} semillas.\n")
    md.append("\n## Paso 2. Prueba, 100 mapas, relacion con la voraz con h_fs\n")
    head = ["conjunto"] + [f"{NOMBRES[m]} exp" for m in ("voraz_hstar", "voraz_hfs")] + [f"{m} rel" for m in ("cf_filtro", "geo", "geo_real")] + [f"{m} camino vs h_fs %" for m in ("cf_filtro", "geo", "geo_real")] + ["h_geo por semilla", "p h_geo"]
    fl = []
    for ds in DATASETS:
        fila = [ds] + [f(res.get((ds, m), {}).get("exp"), 1) for m in ("voraz_hstar", "voraz_hfs")]
        fila += [f(res.get((ds, m), {}).get("rel_hfs")) for m in ("cf_filtro", "geo", "geo_real")] + [f(res.get((ds, m), {}).get("camino_vs_hfs_pct"), 1) for m in ("cf_filtro", "geo", "geo_real")]
        fila += [res.get((ds, "geo"), {}).get("rel_hfs_por_semilla", ""), f(res.get((ds, "geo"), {}).get("p_hfs_exp"), 4)]
        fl.append(fila)
    md.append(md_tabla(head, fl))
    md.append(f"\nRegla del paso 2. h_geo {'pasa' if ok2 else 'no pasa'}, {n2} de 3. " + ". ".join(det2) + ".\n")
    md.append("\n## Lambda elegido y base del paso 3\n")
    md.append(md_tabla(head_base, filas_base))
    md.append("\n## Tiempos\n")
    md.append(md_tabla(head_t, filas_t))
    md.append("\n## Figuras\n\n![curvas](curvas_entrenamiento.png)\n\n![barrido](figuras/barrido_lambda.png)\n\n![calibracion](figuras/calibracion.png)\n\n![exp vs camino](figuras/exp_vs_camino.png)\n\nOcupacion predicha contra real en figuras/geo_<conjunto>.png.\n")
    md.append("\n## Que funciono\n\n(pendiente de revisar)\n\n## Que no funciono\n\n(pendiente de revisar)\n\n## Lo que necesita tu decision\n\n(pendiente de revisar)\n")
    (root / "RESUMEN_auto.md").write_text("".join(md))
    print("".join(md))
    print(f"-> {root / 'tablas.md'}\n-> {root / 'RESUMEN_auto.md'}")


if __name__ == "__main__":
    main()
