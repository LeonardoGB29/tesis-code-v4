# tablas y borrador del resumen del experimento cf
#   python -m experiments.resumen_cf --root results/exp_cf [--refs results/exp_cf]   (exp_cf2 lee las referencias y los modelos de exp_cf desde --refs)
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.common import DATASETS, f, leer_csv, md_tabla, p_wilcoxon, write_csv

MODELOS = {"cf_global": ("cf_global", 5), "cf_local": ("cf_local", 3), "cf_global_gnn": ("cf_global_gnn", 3), "cf_global_dagger": ("cf_global_dagger", 3),
           "cf_global_suave": ("cf_global_suave", 5),          # fuera del protocolo: cf_global con cf suavizado y epoca reelegida
           "cf_global_dagger_suave": ("cf_global_dagger_suave", 3),   # fuera del protocolo: lo mismo sobre los checkpoints de dagger
           "cf_filtro": ("cf_filtro", 5), "cf_grueso": ("cf_grueso", 3),   # exp_cf2: escala gruesa dentro del modelo, entrenados con dagger
           "phil_res_cf": ("phil_res_cf", 5), "mem_no": ("mem_no", 5), "mem_gru": ("mem_gru", 5), "mem_mamba": ("mem_mamba", 5)}   # exp_mem: sobre cf_filtro congelado
NOMBRES = {"voraz_hstar": "voraz con h* (techo)", "voraz_hfs": "voraz con h_fs", "phil": "PHIL", "phil_res_s": "phil_res_s",
           "cf_global": "cf_global", "cf_local": "cf_local", "cf_global_gnn": "cf_global_gnn", "cf_global_dagger": "cf_global_dagger", "cf_global_suave": "cf_global suavizado (fuera del protocolo)", "cf_global_dagger_suave": "cf_global_dagger suavizado (fuera del protocolo)",
           "ruido_hfs": "voraz con h_fs mas ruido", "ruido_phil": "PHIL s0 mas ruido",
           "cf_filtro": "cf_filtro (gauss dentro del modelo, dagger)", "cf_grueso": "cf_grueso (cabeza 26x26, dagger)",
           "phil_res_cf": "phil_res_cf (PHIL sobre cf_filtro)", "mem_no": "mem_no (GNN sin memoria sobre cf_filtro)",
           "mem_gru": "mem_gru (GNN y GRU sobre cf_filtro)", "mem_mamba": "mem_mamba (GNN y Mamba sobre cf_filtro)"}


def cargar_eval(path):
    # grid -> fila con numeros
    out = {}
    for r in leer_csv(path):
        out[r["grid"]] = {"exp": float(r["exp"]), "opt": int(r["optimo"]), "enc": r["encontrado"] == "1",
                          "camino": int(r["camino"]) if r["camino"] else None, "secs": float(r["secs"]),
                          "secs_red": float(r.get("secs_red") or 0), "epoca": r.get("epoca", "")}
    return out


def corridas(root, refs=None, refs2=None):
    # (ds, modelo) -> {seed: eval}; las referencias y los modelos que no esten en root se buscan en refs2 y despues en refs
    refs = refs or root
    otras = [r for r in (refs2, refs) if r is not None]
    ev = defaultdict(dict)
    for ds in DATASETS:
        for ref in ("voraz_hfs", "voraz_hstar"):
            p = refs / "refs" / "eval" / ds / f"{ref}.csv"
            if p.exists():
                ev[(ds, ref)][0] = cargar_eval(p)
        for s in range(5):
            p = refs / "phil" / "eval" / ds / f"phil_s_s{s}.csv"
            if p.exists():
                ev[(ds, "phil")][s] = cargar_eval(p)
            p = refs / "ruido" / "eval" / ds / f"voraz_hfs_ruido{s}.csv"
            if p.exists():
                ev[(ds, "ruido_hfs")][s] = cargar_eval(p)
            p = refs / "ruido" / "eval" / ds / f"phil_s_s0_ruido{s}.csv"
            if p.exists():
                ev[(ds, "ruido_phil")][s] = cargar_eval(p)
        for s in range(3):
            p = refs / "phil_res_s" / "eval" / ds / f"phil_res_s_s{s}.csv"
            if p.exists():
                ev[(ds, "phil_res_s")][s] = cargar_eval(p)
        for m, (sub, n) in MODELOS.items():
            for s in range(n):
                p = root / sub / "eval" / ds / f"{m}_s{s}.csv"
                for r in otras:
                    if not p.exists():
                        p = r / sub / "eval" / ds / f"{m}_s{s}.csv"
                if p.exists():
                    ev[(ds, m)][s] = cargar_eval(p)
    return ev


def medias_por_mapa(evs, seeds):
    # grid -> media sobre semillas de exp y camino/opt (camino solo en mapas encontrados por todas)
    grids = set.intersection(*[set(evs[s]) for s in seeds])
    exp = {g: float(np.mean([evs[s][g]["exp"] for s in seeds])) for g in grids}
    cam = {}
    for g in grids:
        cs = [evs[s][g]["camino"] / max(evs[s][g]["opt"], 1) for s in seeds if evs[s][g]["enc"]]
        if len(cs) == len(seeds):
            cam[g] = float(np.mean(cs))
    return exp, cam


def resumen_modelo(ev, ds, modelo, ref_hfs, phil):
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
    enc = np.mean([evs[s][g]["enc"] for s in seeds for g in grids])
    caminos = [evs[s][g]["camino"] / max(evs[s][g]["opt"], 1) for s in seeds for g in grids if evs[s][g]["enc"]]
    optimos = [evs[s][g]["camino"] == evs[s][g]["opt"] for s in seeds for g in grids if evs[s][g]["enc"]]
    r = {"dataset": ds, "modelo": modelo, "seeds": " ".join(map(str, seeds)), "n_mapas": len(grids), "exp": exp,
         "exp_sd_semillas": float(np.std(list(por_seed.values()))) if len(seeds) > 1 else 0.0,
         "exito": float(enc), "camino_opt": float(np.mean(caminos)) if caminos else float("nan"),
         "pct_optimos": 100 * float(np.mean(optimos)) if optimos else float("nan"),
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
    if phil:
        ps = [s for s in seeds if s in phil] or sorted(phil)
        pm, _ = medias_por_mapa(phil, ps)
        g2 = [g for g in grids if g in pm]
        ref_exp = float(np.mean([pm[g] for g in g2]))
        r["rel_phil"] = exp / ref_exp
        r["rel_phil_por_semilla"] = " ".join(f"{por_seed[s] / np.mean([phil[s if s in phil else ps[0]][g]['exp'] for g in g2]):.3f}" for s in seeds)
        r["semillas_mejoran_phil"] = sum(por_seed[s] < np.mean([phil[s if s in phil else ps[0]][g]["exp"] for g in g2]) for s in seeds)
        r["p_phil_exp"] = p_wilcoxon([exp_m[g] for g in g2], [pm[g] for g in g2])
    return r


def regla(res, modelo):
    # funciona si baja mas de 10 % a la voraz con h_fs en 2 de 3 conjuntos con p < 0.05, relacion < 1 en 4 de 5 semillas (n - 1) y camino no mas de 5 % peor
    det, pasan = [], 0
    for ds in DATASETS:
        r = res.get((ds, modelo))
        if not r or "rel_hfs" not in r:
            det.append(f"{ds} sin datos"); continue
        n = len(r["seeds"].split())
        need = 4 if n >= 5 else n - 1
        ok = r["mejora_hfs_pct"] > 10 and r["p_hfs_exp"] < 0.05 and r["semillas_mejoran_hfs"] >= need and r["camino_vs_hfs_pct"] <= 5
        pasan += ok
        det.append(f"{ds} {'pasa' if ok else 'no pasa'} (mejora {r['mejora_hfs_pct']:.1f} %, p {r['p_hfs_exp']:.3g}, camino {r['camino_vs_hfs_pct']:+.1f} %, "
                   f"semillas < 1 en {r['semillas_mejoran_hfs']} de {n})")
    return pasan >= 2, pasan, det


def ruido(ev, ds, clave, ref_key, ref_seed):
    # piso de ruido: relacion de cada semilla de ruido con la referencia sin ruido
    evs = ev.get((ds, clave), {})
    ref = ev.get((ds, ref_key), {}).get(ref_seed)
    if not evs or not ref:
        return None
    rels, ps = [], []
    for s in sorted(evs):
        g = [x for x in evs[s] if x in ref]
        rels.append(np.mean([evs[s][x]["exp"] for x in g]) / np.mean([ref[x]["exp"] for x in g]))
        ps.append(p_wilcoxon([evs[s][x]["exp"] for x in g], [ref[x]["exp"] for x in g]))
    return {"dataset": ds, "modelo": clave, "n": len(rels), "rel_media": float(np.mean(rels)), "rel_min": float(min(rels)),
            "rel_max": float(max(rels)), "desvio_pct": 100 * float(np.std(rels)), "p_min": float(min(ps)), "por_semilla": " ".join(f"{r:.3f}" for r in rels)}


def seleccion(root):
    filas = []
    for m, (sub, n) in MODELOS.items():
        for ds in DATASETS:
            for s in range(n):
                d = root / sub / "runs" / ds / f"{m}_s{s}"
                if not (d / "train.csv").exists():
                    continue
                rows = leer_csv(d / "train.csv")
                if not rows:                                   # corrida recien lanzada o rota
                    continue
                meta = json.loads((d / "config.json").read_text()) if (d / "config.json").exists() else {}
                vals = [r for r in rows if r["val_exp"]]
                ref = meta.get("ref_val", {})
                mejor = min(vals, key=lambda r: float(r["val_exp"])) if vals else None
                filas.append([ds, m, s, meta.get("best_epoca", ""), f(meta.get("best_val"), 1), f(ref.get("exp"), 1), f(ref.get("camino")),
                              sum(int(r["elegible"]) for r in vals), len(vals), mejor["epoca"] if mejor else "", f(float(mejor["val_exp"]), 1) if mejor else "",
                              mejor["elegible"] if mejor else "", meta.get("epoca_final", ""), int(bool(meta.get("corte"))),
                              f(float(rows[1]["loss"]), 4) if len(rows) > 1 and rows[1]["loss"] else "", f(float(rows[-1]["loss"]), 4) if rows[-1]["loss"] else "",
                              round(sum(float(r["secs_train"] or 0) for r in rows) / 60, 1), round(sum(float(r["secs_val"] or 0) for r in rows) / 60, 1)])
    head = ["dataset", "modelo", "seed", "epoca_elegida", "val_exp_elegida", "val_exp_hfs", "val_camino_hfs", "validaciones_elegibles", "validaciones",
            "epoca_menos_exp", "val_exp_min", "min_elegible", "epoca_final", "corte", "loss_epoca1", "loss_final", "min_train", "min_val"]
    write_csv(root / "seleccion.csv", head, filas)
    return head, filas


def curvas(root):
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    colores = {"cf_global": "C0", "cf_local": "C2", "cf_global_gnn": "C4", "cf_global_dagger": "C3", "cf_filtro": "crimson", "cf_grueso": "darkcyan",
               "phil_res_cf": "C5", "mem_no": "C0", "mem_gru": "C2", "mem_mamba": "C1"}
    for j, ds in enumerate(DATASETS):
        ref = None
        for m, (sub, n) in MODELOS.items():
            runs = []
            for s in range(n):
                p = root / sub / "runs" / ds / f"{m}_s{s}" / "train.csv"
                if p.exists():
                    runs.append(leer_csv(p))
                    cfgp = p.parent / "config.json"
                    if ref is None and cfgp.exists():
                        ref = json.loads(cfgp.read_text()).get("ref_val", {}).get("exp")
            if not runs:
                continue
            for k, (col, ax) in enumerate(zip(("loss", "val_exp", "val_camino"), axes[:, j])):
                por_ep = defaultdict(list)
                for rows in runs:
                    for r in rows:
                        if r[col]:
                            por_ep[int(r["epoca"])].append(float(r[col]))
                if not por_ep:
                    continue
                xs = sorted(por_ep)
                ys = [np.mean(por_ep[x]) for x in xs]
                lo = [np.min(por_ep[x]) for x in xs]; hi = [np.max(por_ep[x]) for x in xs]
                ax.plot(xs, ys, color=colores.get(m, "C5"), label=m, marker="o" if k else None, ms=3)
                ax.fill_between(xs, lo, hi, color=colores.get(m, "C5"), alpha=0.15)
        for k, ax in enumerate(axes[:, j]):
            ax.set_title(f"{ds}: " + ["perdida mse sobre cf (entrenamiento)", "expansiones de validacion", "camino de validacion sobre el optimo"][k])
            ax.set_xlabel("epoca"); ax.grid(alpha=0.3)
            if k == 1 and ref:
                ax.axhline(ref, color="gray", ls="--", label="voraz con h_fs")
            if k == 0:
                ax.set_yscale("log")
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root / "curvas_entrenamiento.png", dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results/exp_cf")
    p.add_argument("--refs", default=None, help="raiz con las referencias y los modelos anteriores (exp_cf2 usa results/exp_cf)")
    p.add_argument("--refs2", default=None, help="segunda raiz de modelos anteriores (exp_mem usa results/exp_cf2)")
    a = p.parse_args()
    root = Path(a.root)
    ev = corridas(root, Path(a.refs) if a.refs else None, Path(a.refs2) if a.refs2 else None)
    res, filas = {}, []
    orden = ["voraz_hstar", "voraz_hfs", "phil", "phil_res_s", "cf_global", "cf_local", "cf_global_gnn", "cf_global_dagger", "cf_global_suave", "cf_global_dagger_suave", "cf_filtro", "cf_grueso",
             "phil_res_cf", "mem_no", "mem_gru", "mem_mamba"]
    for ds in DATASETS:
        for m in orden:
            r = resumen_modelo(ev, ds, m, ev.get((ds, "voraz_hfs")), ev.get((ds, "phil")))
            if r:
                res[(ds, m)] = r; filas.append(r)
    cols = ["dataset", "modelo", "seeds", "n_mapas", "exp", "exp_sd_semillas", "rel_hfs", "rel_hfs_por_semilla", "semillas_mejoran_hfs", "mejora_hfs_pct",
            "p_hfs_exp", "rel_phil", "rel_phil_por_semilla", "semillas_mejoran_phil", "p_phil_exp", "exito", "camino_opt", "camino_hfs", "camino_vs_hfs_pct",
            "p_hfs_camino", "pct_optimos", "ms_nodo", "ms_nodo_red", "epocas"]
    write_csv(root / "resumen.csv", cols, [[f(r.get(c)) if isinstance(r.get(c), float) else r.get(c, "") for c in cols] for r in filas])
    pisos = [x for ds in DATASETS for x in (ruido(ev, ds, "ruido_hfs", "voraz_hfs", 0), ruido(ev, ds, "ruido_phil", "phil", 0)) if x]
    write_csv(root / "piso_ruido.csv", ["dataset", "modelo", "n", "rel_media", "rel_min", "rel_max", "desvio_pct", "p_min", "por_semilla"],
              [[x["dataset"], x["modelo"], x["n"], f(x["rel_media"]), f(x["rel_min"]), f(x["rel_max"]), f(x["desvio_pct"], 2), f(x["p_min"], 4), x["por_semilla"]] for x in pisos])
    head_sel, filas_sel = seleccion(root)
    curvas(root)

    md = [f"# Tablas de {root.name}\n", "Todo en los 100 mapas de prueba (91 resolubles en gaps_and_forest) con tope de 10 000 expansiones. Relacion es expansiones sobre las de la referencia. "
          "Camino es el largo sobre el optimo en los mapas encontrados. Wilcoxon pareado por mapa sobre la media de las semillas de cada lado. ms por nodo cuenta la red.\n"]
    for ds in DATASETS:
        md.append(f"\n## {ds}\n")
        head = ["modelo", "semillas", "expansiones", "sd entre semillas", "relacion con voraz h_fs", "por semilla", "semillas < 1", "mejora %", "p Wilcoxon exp",
                "relacion con PHIL", "p vs PHIL", "exito", "camino sobre optimo", "camino vs h_fs %", "p camino", "caminos optimos %", "ms por nodo", "ms red", "epoca elegida"]
        fl = []
        for m in orden:
            r = res.get((ds, m))
            if r:
                fl.append([NOMBRES[m], r["seeds"], f(r["exp"], 1), f(r["exp_sd_semillas"], 1), f(r.get("rel_hfs")), r.get("rel_hfs_por_semilla", ""), r.get("semillas_mejoran_hfs", ""),
                           f(r.get("mejora_hfs_pct"), 1), f(r.get("p_hfs_exp"), 4), f(r.get("rel_phil")), f(r.get("p_phil_exp"), 4), f(r["exito"]), f(r["camino_opt"]),
                           f(r.get("camino_vs_hfs_pct"), 2), f(r.get("p_hfs_camino"), 4), f(r["pct_optimos"], 1), f(r["ms_nodo"]), f(r["ms_nodo_red"]), r["epocas"]])
        md.append(md_tabla(head, fl))
    md.append("\n## Piso de ruido. Ruido uniforme de 0.01 saltos, 5 semillas, relacion con la misma busqueda sin ruido\n")
    md.append(md_tabla(["conjunto", "busqueda", "semillas", "relacion media", "min", "max", "desvio %", "p minimo", "por semilla"],
                       [[x["dataset"], NOMBRES[x["modelo"]], x["n"], f(x["rel_media"]), f(x["rel_min"]), f(x["rel_max"]), f(x["desvio_pct"], 2), f(x["p_min"], 4), x["por_semilla"]] for x in pisos]))
    md.append("\n## cf_global contra cf_local. Semillas 0 a 2, Wilcoxon por mapa sobre medias\n")
    fl = []
    for ds in DATASETS:
        a_, b_ = ev.get((ds, "cf_global"), {}), ev.get((ds, "cf_local"), {})
        if a_ and b_:
            seeds = [s for s in (0, 1, 2) if s in a_ and s in b_]
            ea, _ = medias_por_mapa(a_, seeds); eb, _ = medias_por_mapa(b_, seeds)
            g = sorted(set(ea) & set(eb))
            fl.append([ds, " ".join(map(str, seeds)), f(np.mean([ea[x] for x in g]), 1), f(np.mean([eb[x] for x in g]), 1),
                       f(np.mean([ea[x] for x in g]) / np.mean([eb[x] for x in g])), f(p_wilcoxon([ea[x] for x in g], [eb[x] for x in g]), 4)])
    md.append(md_tabla(["conjunto", "semillas", "cf_global exp", "cf_local exp", "global / local", "p Wilcoxon"], fl))
    md.append("\n## Seleccion por semilla. Epoca elegida con camino5 respecto de la voraz con h_fs\n")
    md.append(md_tabla(head_sel, filas_sel))
    md.append("\n## Regla de decision\n")
    reglas = {m: regla(res, m) for m in MODELOS}
    for m, (ok, n, det) in reglas.items():
        md.append(f"- {NOMBRES[m]}. {'Funciona' if ok else 'No funciona'}, pasa en {n} de 3. " + ". ".join(det) + ".\n")
    (root / "tablas.md").write_text("".join(md))

    md = [f"# Resumen de {root.name}\n", f"Carpeta tesis/code_v3, resultados en {root}. Borrador generado por resumen_cf.py.\n",
          "\n## Tabla principal. Contra la voraz con h_fs y contra PHIL, 100 mapas de prueba\n",
          "Relacion es expansiones sobre las de la referencia. Camino es el largo sobre el optimo.\n"]
    head = ["conjunto", "voraz h* exp", "voraz h_fs exp", "PHIL exp"] + [f"{m} rel h_fs" for m in MODELOS] + [f"{m} camino vs h_fs %" for m in MODELOS]
    fl = []
    for ds in DATASETS:
        fila = [ds] + [f(res.get((ds, m), {}).get("exp"), 1) for m in ("voraz_hstar", "voraz_hfs", "phil")]
        fila += [f(res.get((ds, m), {}).get("rel_hfs")) for m in MODELOS] + [f(res.get((ds, m), {}).get("camino_vs_hfs_pct"), 1) for m in MODELOS]
        fl.append(fila)
    md.append(md_tabla(head, fl))
    md.append("\n## Relacion por semilla con la voraz con h_fs\n")
    fl = [[m, ds, res[(ds, m)]["rel_hfs_por_semilla"], f"{res[(ds, m)]['semillas_mejoran_hfs']} de {len(res[(ds, m)]['seeds'].split())}", f(res[(ds, m)]["p_hfs_exp"], 4), res[(ds, m)]["epocas"]]
          for m in list(MODELOS) + ["phil", "phil_res_s"] for ds in DATASETS if (ds, m) in res]
    md.append(md_tabla(["modelo", "conjunto", "relacion por semilla", "semillas < 1", "p Wilcoxon", "epoca o iteracion elegida"], fl))
    md.append("\n## Regla de decision\n")
    md.append(md_tabla(["modelo", "conjuntos que pasan", "detalle"], [[m, f"{n} de 3", ". ".join(det)] for m, (ok, n, det) in reglas.items()]))
    ganan = [m for m, (ok, n, det) in reglas.items() if ok]
    md.append(("\n" + ", ".join(ganan) + " cumple la regla.\n") if ganan else "\nNingun modelo cumple la regla.\n")
    md.append("\n## Figuras\n\n![curvas](curvas_entrenamiento.png)\n\n![exp vs camino](figuras/exp_vs_camino.png)\n\nMapas de cf en figuras/cf_<conjunto>.png.\n")
    md.append("\n## Que funciono\n\n(pendiente de revisar)\n\n## Que no funciono\n\n(pendiente de revisar)\n\n## Lo que necesita tu decision\n\n(pendiente de revisar)\n")
    out = root / ("RESUMEN_auto.md" if (root / "RESUMEN.md").exists() else "RESUMEN.md")
    out.write_text("".join(md))
    print("".join(md))
    print(f"-> {root / 'tablas.md'}\n-> {out}")


if __name__ == "__main__":
    main()
