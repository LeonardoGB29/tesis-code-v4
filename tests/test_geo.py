# pruebas de exp_geo
#   .venv/bin/python -m tests.test_geo
import math
import random
import tempfile

import numpy as np
import torch

from experiments.common import ESCALA, EXP_CF2, grids_de, hstar_mapa, load_problem
from src.cf.busqueda import Evaluador, voraz
from src.cf.fotos import Fotos, canales, hfs_entero
from src.cf.red import cargar
from src.geo.fotos import CANALES_GEO, canales_geo, cargar_opcional, etiqueta_geo, lote_geo, previo, unir_opcional
from src.geo.hgeo import EvaluadorGeo, EvaluadorGeoReal, distancia_geo
from src.geo.metricas import Acumulador, corredor, distancia_frontera, estratos
from src.geo.red import GeoConfig, cargar_geo, construir_geo, guardar_geo
from src.mapa_oculto.fs import free_space_distance
from src.mapa_oculto.search import greedy_fs

torch.set_num_threads(1)


def problemas(ds="mazes", n=3):
    for gp in grids_de(ds, "test")[:n]:
        yield (gp,) + load_problem(gp)


def estado_de(ds="mazes", n_exp=120):
    gp, data, adj, v_s, v_g, h_star = next(problemas(ds, 1))
    ev = Evaluador(data, v_g, 16, ESCALA)
    voraz(ev, data, v_s, v_g, adj, max_exp=n_exp)
    return data, adj, v_s, v_g, h_star, ev


def modelo_al_azar(seed=0, p0=0.3):
    torch.manual_seed(seed)
    m = construir_geo(GeoConfig(p0=p0))
    with torch.no_grad():
        m.logit.weight.normal_(0, 0.5)
    return m.eval()


def test_canales_y_etiqueta():
    data, adj, v_s, v_g, h_star, ev = estado_de("gaps_and_forest", 200)
    x = canales_geo(ev.desc.visto, ev.desc.real)
    y, m = etiqueta_geo(ev.desc.visto, ev.desc.real)
    assert x.shape == (CANALES_GEO, 208, 208) and y.shape == (208, 208) and m.shape == (208, 208)
    assert np.array_equal(x[:, :201, :201].sum(0), np.ones((201, 201))), "cada celda del mapa esta en un solo canal"
    assert x[1, 201:, :].all() and x[1, :, 201:].all() and not m[201:, :].any() and not m[:, 201:].any()
    assert m.sum() == (~ev.desc.visto).sum() and (y[:201, :201] == ev.desc.real).all()
    assert not (m[:201, :201] & ev.desc.visto).any()
    print("ok canales de 3 y etiqueta de ocupacion en lo no visto")


def test_previo_constante_exacto():
    m = construir_geo(GeoConfig(p0=0.3)).eval()
    with torch.no_grad():
        out = m(torch.rand(2, 3, 208, 208))
    assert out.shape == (2, 208, 208)
    assert torch.all(out == out.flatten()[0]) and abs(float(out.flatten()[0]) - math.log(0.3 / 0.7)) < 1e-6
    assert abs(float(torch.sigmoid(out).mean()) - 0.3) < 1e-6
    print("ok la cnn arranca en el previo constante exacto")


def test_lambda_cero_igual_hfs():
    m = modelo_al_azar()
    for gp, data, adj, v_s, v_g, h_star in problemas("mazes", 3):
        c1, f1, p1, _ = greedy_fs(data, v_s, v_g, adj, 16, 10000)
        r = voraz(EvaluadorGeo(data, v_g, m, 0.0, 20, 16, ESCALA), data, v_s, v_g, adj, max_exp=10000)
        assert r.found == f1 and r.closed == c1 and r.path == p1, gp.name
        assert r.n_red >= r.n_expanded // 20, (r.n_red, r.n_expanded)
        rr = voraz(EvaluadorGeoReal(data, v_g, 0.0, 16, ESCALA), data, v_s, v_g, adj, max_exp=10000)
        assert rr.found == f1 and rr.closed == c1 and rr.path == p1, gp.name
    print("ok con lambda 0 h_geo con la cnn y con la ocupacion real reproduce la voraz con h_fs")


def test_distancia_geo():
    data, adj, v_s, v_g, h_star, ev = estado_de("gaps_and_forest", 300)
    known, visto, goal = ev.desc.known_inmap(), ev.desc.visto, ev.goal
    rng = np.random.default_rng(0)
    p = rng.random(known.shape).astype(np.float32)
    h0 = distancia_geo(known, visto, p, goal, 0.0)
    hf = free_space_distance(known, goal)
    assert np.array_equal(np.isfinite(h0), np.isfinite(hf)) and np.array_equal(h0[np.isfinite(h0)], hf[np.isfinite(hf)])
    h4 = distancia_geo(known, visto, p, goal, 4.0)
    fin = np.isfinite(hf)
    assert np.array_equal(np.isfinite(h4), fin), "h_geo es finito donde h_fs es finito"
    assert (h4[fin] >= h0[fin] - 1e-9).all() and (h4[fin] > h0[fin]).any()
    assert h4[goal] == 0.0
    # con p 0 en todo lado da h_fs aunque lambda sea grande, y con p 1 cada celda no vista cuesta 1 + lambda
    assert np.array_equal(distancia_geo(known, visto, np.zeros_like(p), goal, 8.0)[fin], hf[fin])
    h1 = distancia_geo(known, visto, np.ones_like(p), goal, 2.0)
    # la arista que entra a una celda cuesta esa celda y la meta cuesta 1, asi que todo vecino de la meta vale 1
    assert (h1[fin & (hf == 1)] == 1.0).all()
    # a dos saltos vale 2 si hay un paso intermedio visto y libre, o 2 + lambda si todos los intermedios son no vistos
    assert set(np.unique(h1[fin & (hf == 2)]).tolist()) <= {2.0, 4.0}
    assert (h1[fin & (hf == 2) & (~visto)] >= 2.0).all()
    print("ok distancia_geo, lambda 0 es h_fs, los costos suben con lambda y nunca bloquean")


def test_entrena_y_guarda():
    gp, data, adj, v_s, v_g, h_star = next(problemas("mazes", 1))
    ev = Evaluador(data, v_g, 16, ESCALA)
    res = voraz(ev, data, v_s, v_g, adj, max_exp=10000, foto_cada=50)
    vistos, hfss = zip(*res.fotos)
    fotos = Fotos([gp.name], np.stack([ev.desc.real]), np.stack([hstar_mapa(data, h_star)]), np.array([ev.goal]), np.array([(int(ev.rc[v_s, 0]), int(ev.rc[v_s, 1]))]),
                  np.zeros(len(vistos), dtype=np.int32), np.stack(vistos), np.stack(hfss), ESCALA)
    p0 = previo(fotos)
    assert 0 < p0 < 1
    rng = np.random.default_rng(0)
    x, y, m = lote_geo(fotos, np.arange(min(8, len(fotos))), rng)
    assert x.shape[1:] == (3, 208, 208) and m.any() and set(np.unique(y.numpy())) <= {0.0, 1.0}
    torch.manual_seed(0)
    model = construir_geo(GeoConfig(p0=p0))
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    l0 = None
    for i in range(8):
        logits = model(x)
        mf = m.float()
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none") * mf).sum() / mf.sum()
        opt.zero_grad(); loss.backward(); opt.step()
        l0 = l0 if l0 is not None else loss.item()
    assert model.logit.weight.grad is not None and model.logit.weight.grad.abs().sum() > 0
    assert loss.item() < l0, (l0, loss.item())
    bce_previo = -(p0 * math.log(p0) + (1 - p0) * math.log(1 - p0))
    model.eval()
    with tempfile.TemporaryDirectory() as d:
        guardar_geo(f"{d}/m.pt", model, 3, {"bce": 0.1})
        m2, cfg2, raw = cargar_geo(f"{d}/m.pt")
        assert cfg2.canales_entrada == 3 and abs(cfg2.p0 - p0) < 1e-9 and raw["epoca"] == 3
        with torch.no_grad():
            assert torch.equal(model(x), m2(x))
        fotos.guardar(f"{d}/f")
        f2 = cargar_opcional(f"{d}/f")
        assert len(f2) == len(fotos)
        from pathlib import Path
        Path(f"{d}/vacio").mkdir(); Path(f"{d}/vacio/listo").write_text("0 fotos, prueba\n")
        assert cargar_opcional(f"{d}/vacio") is None and len(unir_opcional([f2, None])) == len(fotos)
    print(f"ok entrena {l0:.4f} -> {loss.item():.4f} (previo {bce_previo:.4f}), guarda y carga")


def test_fuga():
    data, adj, v_s, v_g, h_star, ev = estado_de("mazes", 150)
    m = modelo_al_azar()
    cf, cfg, _ = cargar(EXP_CF2 / "cf_filtro" / "runs" / "mazes" / "cf_filtro_s0" / "best.pt")
    visto, real, goal = ev.desc.visto, ev.desc.real, ev.goal
    rng = np.random.default_rng(0)
    real2 = real.copy()
    nv = np.argwhere(~visto)
    for r, c in nv[rng.choice(len(nv), 200, replace=False)]:
        real2[r, c] = ~real2[r, c]
    assert np.array_equal(canales_geo(visto, real), canales_geo(visto, real2))
    with torch.no_grad():
        assert torch.equal(m(torch.from_numpy(canales_geo(visto, real))[None]), m(torch.from_numpy(canales_geo(visto, real2))[None]))
        hfs = hfs_entero(ev.fs.h)
        assert torch.equal(cf(torch.from_numpy(canales(visto, real, goal, hfs, ESCALA))[None]), cf(torch.from_numpy(canales(visto, real2, goal, hfs, ESCALA))[None]))
    # el mismo cambio sobre un estado de la busqueda con h_geo, con las celdas no vistas de ese estado
    e1 = EvaluadorGeo(data, v_g, m, 4.0, 20, 16, ESCALA)
    voraz(e1, data, v_s, v_g, adj, max_exp=150)
    real2 = e1.desc.real.copy()
    nv = np.argwhere(~e1.desc.visto)
    for r, c in nv[rng.choice(len(nv), 200, replace=False)]:
        real2[r, c] = ~real2[r, c]
    import copy
    e2 = copy.copy(e1); e2.desc = e1.desc.copia(); e2.desc.real = real2
    p1, p2 = e1.calcular(), e2.calcular()
    assert np.array_equal(p1, p2)
    assert np.array_equal(distancia_geo(e1.desc.known_inmap(), e1.desc.visto, p1, goal, 4.0), distancia_geo(e2.desc.known_inmap(), e2.desc.visto, p2, goal, 4.0))
    print("ok la cnn de geometria, cf_filtro y h_geo no ven las celdas no vistas")


def test_metricas():
    rng = np.random.default_rng(0)
    p = rng.random(50000); y = rng.random(50000) < p
    a = Acumulador(); a.agregar(p[:20000], y[:20000]); a.agregar(p[20000:], y[20000:])
    r, tabla = a.resultados()
    from scipy.stats import rankdata
    rk = rankdata(p); n1 = y.sum(); n0 = (~y).sum()
    auc = (rk[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    pc = np.clip(p, 1e-7, 1 - 1e-7)
    bce = -(np.log(pc[y]).sum() + np.log(1 - pc[~y]).sum()) / len(p)
    assert abs(r["auc"] - auc) < 2e-3 and abs(r["bce"] - bce) < 1e-9 and r["n"] == 50000 and len(tabla) == 10
    a = Acumulador(); a.agregar(np.full(len(y), 0.4), y)
    r, _ = a.resultados()
    assert abs(r["auc"] - 0.5) < 1e-12 and abs(r["ece"] - abs(0.4 - y.mean())) < 1e-9
    r0, _ = Acumulador().resultados()
    assert r0["n"] == 0
    # corredor y frontera en un mapa real
    data, adj, v_s, v_g, h_star, ev = estado_de("gaps_and_forest", 300)
    hs = hstar_mapa(data, h_star)
    start = (int(ev.rc[v_s, 0]), int(ev.rc[v_s, 1]))
    corr = corredor(ev.desc.real, hs, start)
    assert corr[start] and corr[ev.goal] and 0 < corr.sum() < corr.size
    d = distancia_frontera(ev.desc.visto)
    assert (d[ev.desc.visto] == 0).all() and (d[~ev.desc.visto] >= 1).all()
    es = estratos(ev.desc.visto, ev.desc.real, hs, start)
    assert es["todo"].sum() == (~ev.desc.visto).sum()
    assert sum(es[k].sum() for k in es if k.startswith("frontera")) == es["todo"].sum()
    assert not (es["corredor"] & ev.desc.visto).any()
    print(f"ok auc {r['auc']:.3f}, calibracion, corredor de {int(corr.sum())} celdas y tramos de frontera")


def test_lambda_cambia_la_busqueda():
    # con la ocupacion real y lambda grande la busqueda no deberia expandir mas que con h_fs en la mayoria de los mapas
    mejor, total = 0, 0
    for gp, data, adj, v_s, v_g, h_star in problemas("gaps_and_forest", 4):
        c1, f1, p1, _ = greedy_fs(data, v_s, v_g, adj, 16, 10000)
        r = voraz(EvaluadorGeoReal(data, v_g, 8.0, 16, ESCALA), data, v_s, v_g, adj, max_exp=10000)
        assert r.found
        total += 1; mejor += int(r.n_expanded < len(c1))
    assert mejor >= total // 2, (mejor, total)
    print(f"ok la ocupacion real con lambda 8 expande menos que h_fs en {mejor} de {total} mapas")


if __name__ == "__main__":
    random.seed(0); np.random.seed(0)
    tests = [test_canales_y_etiqueta, test_previo_constante_exacto, test_lambda_cero_igual_hfs, test_distancia_geo, test_entrena_y_guarda,
             test_fuga, test_metricas, test_lambda_cambia_la_busqueda]
    for t in tests:
        t()
    print(f"{len(tests)}/{len(tests)} pruebas ok")
