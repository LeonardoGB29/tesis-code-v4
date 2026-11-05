# verificaciones previas de exp_cf2
#   1 la epoca 0 de cf_filtro y cf_grueso reproduce la voraz con h_fs en los mapas de prueba, expansion por expansion
#   2 el filtro dentro del modelo es el de afuera: la epoca 10 de cf_global s0 cargada en cf_filtro da lo mismo que cf_global_suave s0 de exp_cf
#   3 una constante sale exacta del filtro y de la cabeza gruesa, en cpu y en mps
#   python -m experiments.check_cf2 --root results/exp_cf2 --refs results/exp_cf [--dataset gaps_and_forest] [--n 100]
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from experiments.common import DATASETS, ESCALA, MAX_EXP, grids_de, leer_csv, load_problem, set_threads, write_csv
from experiments.suavizado_cf import EvaluadorSuave
from experiments.train_cf import SIGMA
from src.cf.busqueda import EvaluadorCF, voraz
from src.cf.fotos import canales, hfs_entero
from src.cf.red import CfConfig, construir
from src.mapa_oculto.search import greedy_fs


def modelos_epoca0(ds):
    return {"cf_filtro": construir(CfConfig(tipo="global", ancho=16, salida="filtro", sigma=SIGMA[ds])).eval(),
            "cf_grueso": construir(CfConfig(tipo="global", ancho=16, salida="gruesa")).eval()}


def check_epoca_cero(root, ds, n):
    filas, ok = [], True
    modelos = modelos_epoca0(ds)
    t0 = time.time()
    for gp in grids_de(ds, "test")[:n]:
        data, adj, v_s, v_g, h_star = load_problem(gp)
        if int(h_star[v_s]) < 0:
            continue
        c1, f1, p1, _ = greedy_fs(data, v_s, v_g, adj, 16, MAX_EXP)
        for nombre, m in modelos.items():
            r = voraz(EvaluadorCF(data, v_g, m, 20, 16, ESCALA), data, v_s, v_g, adj, max_exp=MAX_EXP)
            igual = r.found == f1 and r.closed == c1 and r.path == p1
            ok &= igual
            filas.append([ds, gp.name, nombre, len(c1), r.n_expanded, int(igual)])
    write_csv(root / "checks" / f"epoca_cero_{ds}.csv", ["dataset", "grid", "modelo", "exp_hfs", "exp_modelo", "igual"], filas)
    print(f"[{ds}] epoca 0: {sum(f[-1] for f in filas)} de {len(filas)} iguales a h_fs ({time.time() - t0:.0f}s)", flush=True)
    return ok


def check_equivalencia(root, refs, ds, n_fotos=5):
    # la misma red con el filtro adentro (cf_filtro) y afuera (EvaluadorSuave de exp_cf) tiene que dar el mismo cf y la misma validacion
    ck = refs / "cf_global" / "runs" / ds / "cf_global_s0" / "epoca_10.pt"
    tcsv = refs / "cf_global_suave" / "runs" / ds / "cf_global_suave_s0" / "train.csv"
    if not ck.exists() or not tcsv.exists():
        print(f"[{ds}] equivalencia: faltan {ck} o {tcsv}", flush=True)
        return False
    raw = torch.load(ck, map_location="cpu", weights_only=False)
    cfg = CfConfig.from_dict(raw["config"])
    fino = construir(cfg); fino.load_state_dict(raw["model_state"]); fino.eval()
    cfg2 = CfConfig.from_dict({**raw["config"], "salida": "filtro", "sigma": SIGMA[ds]})
    filtro = construir(cfg2); faltan = filtro.load_state_dict(raw["model_state"], strict=False); filtro.eval()
    assert all(k.startswith("suave.") for k in faltan.missing_keys) and not faltan.unexpected_keys, faltan
    sigma = SIGMA[ds]
    ref_row = next((r for r in leer_csv(tcsv) if r["epoca"] == "10"), None)
    ref_val = float(ref_row["val_exp"]) if ref_row else float("nan")
    grids = grids_de(ds, "validation")
    dif_max, exps_in, exps_out, iguales, tot, caminos = 0.0, [], [], 0, 0, []
    t0 = time.time()
    for k, gp in enumerate(grids):
        data, adj, v_s, v_g, h_star = load_problem(gp)
        opt = int(h_star[v_s])
        if opt < 0:
            continue
        ev_in = EvaluadorCF(data, v_g, filtro, cfg.every_cf, cfg.every_fs, ESCALA)
        r_in = voraz(ev_in, data, v_s, v_g, adj, max_exp=MAX_EXP)
        ev_out = EvaluadorSuave(data, v_g, fino, sigma=sigma, every_cf=cfg.every_cf, every_fs=cfg.every_fs, escala=ESCALA, device="cpu")
        r_out = voraz(ev_out, data, v_s, v_g, adj, max_exp=MAX_EXP)
        exps_in.append(r_in.n_expanded if r_in.found else MAX_EXP); exps_out.append(r_out.n_expanded if r_out.found else MAX_EXP)
        iguales += int(r_in.closed == r_out.closed and r_in.path == r_out.path); tot += 1
        if r_in.found:
            caminos.append((len(r_in.path) - 1) / max(opt, 1))
        if k < n_fotos:                                                     # el cf de la foto final, adentro contra afuera
            x = torch.from_numpy(canales(ev_in.desc.visto, ev_in.desc.real, ev_in.goal, hfs_entero(ev_in.fs.h), ESCALA))[None]
            with torch.no_grad():
                a = filtro(x)[0].numpy()
                b = np.clip(gaussian_filter(fino(x)[0].numpy(), sigma, mode="nearest"), 0.001, 1.0)
            dif_max = max(dif_max, float(np.abs(a - b).max()))
    m_in, m_out = float(np.mean(exps_in)), float(np.mean(exps_out))
    ok = dif_max < 1e-4 and abs(m_in - m_out) <= 0.01 * m_out and abs(m_in - ref_val) <= 0.01 * ref_val
    write_csv(root / "checks" / f"equivalencia_{ds}.csv", ["dataset", "mapas", "exp_filtro_adentro", "exp_filtro_afuera", "val_exp_cf_global_suave_s0_e10", "camino_adentro",
                                                            "busquedas_identicas", "dif_max_cf", "pasa"],
              [[ds, tot, round(m_in, 1), round(m_out, 1), ref_val, round(float(np.mean(caminos)), 4), iguales, f"{dif_max:.2e}", int(ok)]])
    print(f"[{ds}] equivalencia: adentro {m_in:.1f}, afuera {m_out:.1f}, exp_cf {ref_val:.1f}, identicas {iguales} de {tot}, dif cf {dif_max:.2e} ({time.time() - t0:.0f}s)", flush=True)
    return ok


def check_constante(root, ds):
    filas, ok = [], True
    devs = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    x = torch.rand(2, 5, 208, 208)
    for nombre, m in modelos_epoca0(ds).items():
        for dev in devs:
            with torch.no_grad():
                out = m.to(dev)(x.to(dev)).cpu()
            m.to("cpu")
            exacto = torch.equal(out, torch.full_like(out, 0.5))
            ok &= exacto
            filas.append([ds, nombre, dev, float(out.min()), float(out.max()), int(exacto)])
    write_csv(root / "checks" / f"constante_{ds}.csv", ["dataset", "modelo", "device", "min", "max", "pasa"], filas)
    print(f"[{ds}] constante: {'exacta' if ok else 'NO exacta'} en {devs}", flush=True)
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results/exp_cf2")
    p.add_argument("--refs", default="results/exp_cf")
    p.add_argument("--dataset", default=None)
    p.add_argument("--n", type=int, default=100)
    a = p.parse_args()
    set_threads(1)
    torch.manual_seed(0)
    root, refs = Path(a.root), Path(a.refs)
    (root / "checks").mkdir(parents=True, exist_ok=True)
    dss = [a.dataset] if a.dataset else DATASETS
    lineas = []
    for ds in dss:
        lineas.append(f"constante_{ds} {int(check_constante(root, ds))}")
        lineas.append(f"epoca_cero_{ds} {int(check_epoca_cero(root, ds, a.n))}")
        lineas.append(f"equivalencia_{ds} {int(check_equivalencia(root, refs, ds))}")
    out = root / "checks" / (f"estado_{a.dataset}.txt" if a.dataset else "estado.txt")
    out.write_text("\n".join(lineas) + "\n")
    print("\n".join(lineas))


if __name__ == "__main__":
    main()
