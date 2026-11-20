# tablas de new_exp: corrector local, transpath y phil_res_cf en prueba, contra la voraz con h_fs y contra cf_filtro
#   python -m new_exp.resumen --root new_exp/results          (cf_filtro sale de ../code_v3/results/exp_cf2, las referencias de ../code_v3/results/exp_cf)
import argparse
from pathlib import Path

import numpy as np

import experiments.resumen_cf as rc
from experiments.common import DATASETS, EXP_CF2, RESULTS, p_wilcoxon

rc.MODELOS.update({"cf_local_filtro": ("cf_local_filtro", 3), "cf_transpath": ("cf_transpath", 3)})
MODELOS = ["cf_filtro", "cf_local_filtro", "cf_transpath", "phil_res_cf"]
NOMBRES = {"cf_filtro": "corrector global (cf_filtro)", "cf_local_filtro": "corrector local 11x11 (exp 1)",
           "cf_transpath": "arquitectura de transpath (exp 2)", "phil_res_cf": "PHIL sobre el corrector (exp 3)"}


def contra_filtro(ev, ds, m):
    # relacion de expansiones con cf_filtro y wilcoxon pareado por mapa sobre las medias de semillas
    # phil_res_cf se apoya en el cf_filtro de su misma semilla, asi que se empareja por semilla; los otros van contra la media de cf_filtro
    a, b = ev.get((ds, m)), ev.get((ds, "cf_filtro"))
    if not a or not b or m == "cf_filtro":
        return None
    seeds = sorted(set(a) & set(b)) if m == "phil_res_cf" else sorted(a)
    sb = seeds if m == "phil_res_cf" else sorted(b)
    ea, _ = rc.medias_por_mapa(a, seeds)
    eb, _ = rc.medias_por_mapa(b, sb)
    g = sorted(set(ea) & set(eb))
    base = np.mean([eb[x] for x in g])
    por_semilla = [np.mean([a[s][x]["exp"] for x in g]) / (np.mean([b[s][x]["exp"] for x in g]) if m == "phil_res_cf" else base) for s in seeds]
    return {"rel": float(np.mean([ea[x] for x in g]) / base), "p": p_wilcoxon([ea[x] for x in g], [eb[x] for x in g]),
            "por_semilla": por_semilla, "seeds": seeds}


def lectura(res, cf):
    # reglas fijadas en new_exp/PLAN.md antes de correr
    out = []
    ok = [ds for ds in ("gaps_and_forest", "multiple_bugtraps") if cf.get((ds, "cf_local_filtro"))
          and cf[(ds, "cf_local_filtro")]["rel"] > 1.10 and cf[(ds, "cf_local_filtro")]["p"] < 0.05]
    if all(cf.get((ds, "cf_local_filtro")) for ds in ("gaps_and_forest", "multiple_bugtraps")):
        txt = {2: "sostiene la palabra global del titulo", 1: "sostiene lo global solo en parte", 0: "no sostiene lo global"}[len(ok)]
        out.append(f"- Exp 1. El corrector local queda mas de 10 % por encima de cf_filtro con Wilcoxon significativo en {len(ok)} de 2 conjuntos ({', '.join(ok) or 'ninguno'}), {txt}.")
    elif any(cf.get((ds, "cf_local_filtro")) for ds in DATASETS):
        out.append("- Exp 1. Faltan gaps_and_forest o multiple_bugtraps para leer la regla.")
    if any(cf.get((ds, "cf_transpath")) for ds in DATASETS):
        partes = []
        for ds in DATASETS:
            c = cf.get((ds, "cf_transpath"))
            if c:
                v = "gana" if c["rel"] < 0.90 and c["p"] < 0.05 else "pierde" if c["rel"] > 1.10 and c["p"] < 0.05 else "empata"
                partes.append(f"{ds} {v} ({c['rel']:.3f}, p {c['p']:.3g})")
        pasa, n, _ = rc.regla(res, "cf_transpath")
        out.append(f"- Exp 2. Transpath frente a cf_filtro, {'; '.join(partes)}. Contra la voraz con h_fs pasa la regla en {n} de 3 conjuntos.")
    if any(cf.get((ds, "phil_res_cf")) for ds in DATASETS):
        aporta = []
        for ds in DATASETS:
            c = cf.get((ds, "phil_res_cf"))
            if c:
                need = 4 if len(c["seeds"]) >= 5 else len(c["seeds"]) - 1
                if c["rel"] < 0.95 and c["p"] < 0.05 and sum(x < 1 for x in c["por_semilla"]) >= need:
                    aporta.append(ds)
        txt = "la ultima parte de la hipotesis se sostiene" if not aporta else "la ultima parte de la hipotesis no se sostiene"
        out.append(f"- Exp 3. PHIL sobre el corrector baja mas de 5 % a cf_filtro con Wilcoxon significativo y n - 1 semillas por debajo en {len(aporta)} conjuntos ({', '.join(aporta) or 'ninguno'}), {txt}.")
    return out


def num(x, d=4):
    return "" if x is None or x == "" else f"{float(x):.{d}f}"


def main():
    # resultados.csv lleva solo los numeros, la lectura con las reglas del plan va en RESUMEN.md
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="new_exp/results")
    p.add_argument("--refs", default=str(RESULTS))
    p.add_argument("--refs2", default=str(EXP_CF2))
    a = p.parse_args()
    root = Path(a.root)
    ev = rc.corridas(root, Path(a.refs), Path(a.refs2))
    res, cf = {}, {}
    filas = ["conjunto,modelo,semillas,expansiones,rel_voraz_hfs,p_voraz_hfs,rel_cf_filtro,rel_cf_filtro_por_semilla,p_cf_filtro,camino,ms_nodo"]
    for ds in DATASETS:
        for m in MODELOS:
            r = rc.resumen_modelo(ev, ds, m, ev.get((ds, "voraz_hfs")), ev.get((ds, "phil")))
            if not r:
                continue
            res[(ds, m)] = r
            c = contra_filtro(ev, ds, m)
            if c:
                cf[(ds, m)] = c
            filas.append(",".join([ds, m, str(r["seeds"]), num(r["exp"], 1), num(r.get("rel_hfs"), 3), num(r.get("p_hfs_exp")),
                                   num(c["rel"], 3) if c else "", " ".join(f"{x:.3f}" for x in c["por_semilla"]) if c else "",
                                   num(c["p"]) if c else "", num(r.get("camino_opt"), 3), num(r.get("ms_nodo"), 3)]))
    (root / "resultados.csv").write_text("\n".join(filas) + "\n", encoding="utf-8")
    md = ["# Lectura automatica de new_exp\n", "Los numeros estan en resultados.csv. Rel es expansiones sobre las de la referencia en los 100 mapas de prueba, "
          "con Wilcoxon pareado por mapa sobre la media de las semillas.\n", "## Reglas del plan\n"]
    md += lectura(res, cf) or ["- Todavia no hay evaluaciones de prueba."]
    for m in ("cf_local_filtro", "cf_transpath"):
        if any((ds, m) in res for ds in DATASETS):
            pasa, n, det = rc.regla(res, m)
            md.append(f"\nRegla general de {m}, {'pasa' if pasa else 'no pasa'} ({n} de 3).\n")
            md += [f"- {d}" for d in det]
    (root / "RESUMEN_auto.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(filas)); print(); print("\n".join(md))


if __name__ == "__main__":
    main()
