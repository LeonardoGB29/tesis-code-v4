# elige lambda en validacion con camino5 respecto de la voraz con h_fs, con lambda 0 como candidata
#   python -m experiments.lambda_geo --dataset mazes --root results/exp_geo
# lee barrido/<ds>/geo_s0_lam<l>.csv y real_lam<l>.csv y escribe barrido/<ds>/lambda.json
import argparse
import json
import re
from pathlib import Path

import numpy as np

from experiments.common import RESULTS, RESULTS_GEO, leer_csv, write_csv
from experiments.train_cf import elegible

LAMBDAS = [0, 0.5, 1, 2, 4, 8, 16, 32]


def resumen_csv(path):
    rows = leer_csv(path)
    enc = [r for r in rows if r["encontrado"] == "1"]
    exps = [float(r["exp"]) for r in rows]
    return {"exp": float(np.mean(exps)), "camino": float(np.mean([int(r["camino"]) / max(int(r["optimo"]), 1) for r in enc])) if enc else float("nan"),
            "encontrados": len(enc), "n": len(rows), "secs": float(np.sum([float(r["secs"]) for r in rows])), "exp_total": float(np.sum(exps)),
            "secs_red": float(np.sum([float(r.get("secs_red") or 0) for r in rows]))}


def misma_que_hfs(path, dataset):
    # el barrido con lambda 0 tiene que dar las mismas expansiones y caminos por mapa que la referencia de validacion de h_fs
    ref = {r["grid"]: (r["exp"], r["camino"]) for r in leer_csv(RESULTS / "refs" / f"val_hfs_{dataset}.csv")}
    rows = leer_csv(path)
    return all(ref.get(r["grid"]) == (r["exp"], r["camino"]) for r in rows) and len(rows) > 0


def referencia_en(dataset, grids):
    # la validacion de h_fs restringida a los mapas del barrido, en la corrida completa son los 70
    rows = [r for r in leer_csv(RESULTS / "refs" / f"val_hfs_{dataset}.csv") if r["grid"] in grids]
    enc = [r for r in rows if r["encontrado"] == "1"]
    return {"exp": float(np.mean([float(r["exp"]) for r in rows])), "camino": float(np.mean([int(r["camino"]) / max(int(r["optimo"]), 1) for r in enc])), "encontrados": len(enc), "n": len(rows)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--root", default=str(RESULTS_GEO))
    p.add_argument("--quien", default="geo_s0")
    a = p.parse_args()
    root = Path(a.root)
    d = root / "barrido" / a.dataset
    archivos = sorted(d.glob(f"{a.quien}_lam*.csv"))
    if not archivos:
        raise SystemExit(f"[{a.dataset}] sin barrido de {a.quien} en {d}")
    grids = {r["grid"] for r in leer_csv(archivos[0])}
    ref = referencia_en(a.dataset, grids)
    filas, cand = [], []
    for quien in (a.quien, "real"):
        for f in sorted(d.glob(f"{quien}_lam*.csv"), key=lambda q: float(re.search(r"lam([0-9.]+)\.csv", q.name).group(1))):
            lam = float(re.search(r"lam([0-9.]+)\.csv", f.name).group(1))
            r = resumen_csv(f)
            ok = elegible(r, ref)
            fila = {"quien": quien, "lambda": lam, "exp": r["exp"], "camino": r["camino"], "encontrados": r["encontrados"], "elegible": int(ok),
                    "rel_hfs": r["exp"] / ref["exp"], "ms_nodo": 1000 * r["secs"] / max(r["exp_total"], 1), "ms_nodo_red": 1000 * r["secs_red"] / max(r["exp_total"], 1),
                    "igual_hfs": int(misma_que_hfs(f, a.dataset)) if lam == 0 else ""}
            filas.append(fila)
            if quien == a.quien and ok:
                cand.append(fila)
    if not cand:
        raise SystemExit(f"[{a.dataset}] sin barrido de {a.quien} en {d}")
    mejor = min(cand, key=lambda r: (r["exp"], r["lambda"]))
    cero = next((r for r in filas if r["quien"] == a.quien and r["lambda"] == 0), None)
    if cero is None or mejor["exp"] >= ref["exp"] - 1e-9:
        mejor = cero or mejor
    sal = {"dataset": a.dataset, "quien": a.quien, "lambda": mejor["lambda"], "exp": mejor["exp"], "camino": mejor["camino"], "rel_hfs": mejor["rel_hfs"],
           "ref_val_hfs": ref, "lambda_cero_igual_hfs": cero["igual_hfs"] if cero else None, "tabla": filas}
    (d / "lambda.json").write_text(json.dumps(sal, indent=1))
    cols = ["quien", "lambda", "exp", "rel_hfs", "camino", "encontrados", "elegible", "ms_nodo", "ms_nodo_red", "igual_hfs"]
    write_csv(d / "barrido.csv", cols, [[r[c] if not isinstance(r[c], float) else round(r[c], 4) for c in cols] for r in filas])
    print(f"[{a.dataset}] lambda elegido {mejor['lambda']:g}, val exp {mejor['exp']:.1f} contra {ref['exp']:.1f} de h_fs, camino {mejor['camino']:.3f}, "
          f"lambda 0 igual a h_fs {sal['lambda_cero_igual_hfs']}", flush=True)
    for r in filas:
        print(f"  {r['quien']:8s} lam {r['lambda']:5g} exp {r['exp']:8.1f} rel {r['rel_hfs']:.3f} camino {r['camino']:.3f} enc {r['encontrados']} elegible {r['elegible']}")


if __name__ == "__main__":
    main()
