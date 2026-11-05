# pruebas del factor de correccion
#   .venv/bin/python -m tests.test_cf
import random

import numpy as np
import torch

from experiments.common import ESCALA, grids_de, hstar_mapa, load_problem
from src.cf.busqueda import Evaluador, EvaluadorCF, EvaluadorGNN, EvaluadorOraculo, voraz
from src.cf.fotos import Fotos, canales, diedral, diedral_inversa, empaquetar, etiqueta, hfs_entero, lote
from src.cf.red import CfConfig, construir
from src.mapa_oculto.search import greedy_fs
from src.search.baselines import _greedy_best_first

torch.set_num_threads(1)


def problemas(ds="mazes", n=3):
    for gp in grids_de(ds, "test")[:n]:
        yield (gp,) + load_problem(gp)


def test_cf_uno_igual_voraz_hfs():
    for gp, data, adj, v_s, v_g, h_star in problemas("multiple_bugtraps", 4):
        c1, f1, p1, _ = greedy_fs(data, v_s, v_g, adj, 16, 10000)
        res = voraz(Evaluador(data, v_g, 16, ESCALA), data, v_s, v_g, adj, max_exp=10000)
        assert res.found == f1 and res.closed == c1 and res.path == p1, gp.name
    print("ok cf = 1 reproduce la voraz con h_fs")


def test_oraculo_igual_voraz_hstar():
    for gp, data, adj, v_s, v_g, h_star in problemas("gaps_and_forest", 4):
        hs = h_star.numpy().astype(np.float64); hs[hs < 0] = np.inf
        c1, f1, p1 = _greedy_best_first(hs, v_s, v_g, adj, 10000)
        res = voraz(EvaluadorOraculo(data, v_g, hstar_mapa(data, h_star), 16, ESCALA), data, v_s, v_g, adj, max_exp=10000)
        assert res.found == f1 and res.closed == c1 and res.path == p1, (gp.name, len(c1), res.n_expanded)
    print("ok cf* reproduce la voraz con h*")


def foto_de(ds="mazes", n_exp=120):
    gp, data, adj, v_s, v_g, h_star = next(problemas(ds, 1))
    ev = Evaluador(data, v_g, 16, ESCALA)
    voraz(ev, data, v_s, v_g, adj, max_exp=n_exp)
    return data, adj, v_s, v_g, h_star, ev


def test_etiqueta_en_rango():
    data, adj, v_s, v_g, h_star, ev = foto_de("gaps_and_forest", 300)
    hs = hstar_mapa(data, h_star)
    y, m = etiqueta(hfs_entero(ev.fs.h), hs, ev.desc.real, ev.goal)
    assert m.sum() > 1000 and y[m].min() > 0 and y[m].max() <= 1.0 + 1e-6, (y[m].min(), y[m].max())
    assert y[ev.goal] == 1.0 and m[ev.goal]
    print(f"ok cf* en (0, 1], {int(m.sum())} celdas, min {y[m].min():.3f}")


def test_solo_ve_lo_descubierto():
    data, adj, v_s, v_g, h_star, ev = foto_de("mazes", 150)
    torch.manual_seed(0)
    model = construir(CfConfig(tipo="global", ancho=16)).eval()
    with torch.no_grad():
        for p in model.cabeza.conv.parameters():
            p.normal_()
    x0 = canales(ev.desc.visto, ev.desc.real, ev.goal, hfs_entero(ev.fs.h), ESCALA)
    real2 = ev.desc.real.copy()
    rng = np.random.default_rng(0)
    no_vistas = np.argwhere(~ev.desc.visto)
    for r, c in no_vistas[rng.choice(len(no_vistas), 200, replace=False)]:
        real2[r, c] = ~real2[r, c]
    x1 = canales(ev.desc.visto, real2, ev.goal, hfs_entero(ev.fs.h), ESCALA)
    assert np.array_equal(x0, x1)
    with torch.no_grad():
        o0, o1 = model(torch.from_numpy(x0)[None]), model(torch.from_numpy(x1)[None])
    assert torch.equal(o0, o1)
    vis = ev.desc.visto.copy()
    vistas = np.argwhere(vis & ~ev.desc.real)
    real3 = ev.desc.real.copy()
    for r, c in vistas[rng.choice(len(vistas), 20, replace=False)]:
        real3[r, c] = True
    x2 = canales(vis, real3, ev.goal, hfs_entero(ev.fs.h), ESCALA)
    with torch.no_grad():
        o2 = model(torch.from_numpy(x2)[None])
    assert not torch.equal(o0, o2)
    print("ok la red solo ve lo descubierto")


def test_epoca_cero_igual_hfs():
    model = construir(CfConfig(tipo="global", ancho=16)).eval()
    for gp, data, adj, v_s, v_g, h_star in problemas("mazes", 3):
        c1, f1, p1, _ = greedy_fs(data, v_s, v_g, adj, 16, 10000)
        res = voraz(EvaluadorCF(data, v_g, model, 20, 16, ESCALA), data, v_s, v_g, adj, max_exp=10000)
        assert res.found == f1 and res.closed == c1 and res.path == p1, gp.name
        assert res.n_red >= res.n_expanded // 20
    local = construir(CfConfig(tipo="local", ancho=16)).eval()
    gp, data, adj, v_s, v_g, h_star = next(problemas("mazes", 1))
    c1, f1, p1, _ = greedy_fs(data, v_s, v_g, adj, 16, 10000)
    res = voraz(EvaluadorCF(data, v_g, local, 20, 16, ESCALA), data, v_s, v_g, adj, max_exp=10000)
    assert res.closed == c1 and res.path == p1
    print("ok la epoca 0 de cf_global y cf_local reproduce la voraz con h_fs")


def test_diedral():
    a = np.random.rand(5, 208, 208)
    for k in range(8):
        assert np.array_equal(diedral_inversa(diedral(a, k), k), a)
    print("ok simetrias")


def fotos_chicas(ds="mazes", n_mapas=2):
    nombres, reales, hstars, goals, starts, idx, vistos, hfss = [], [], [], [], [], [], [], []
    for k, (gp, data, adj, v_s, v_g, h_star) in enumerate(problemas(ds, n_mapas)):
        ev = Evaluador(data, v_g, 16, ESCALA)
        res = voraz(ev, data, v_s, v_g, adj, max_exp=10000, foto_cada=50)
        nombres.append(gp.name); reales.append(ev.desc.real); hstars.append(hstar_mapa(data, h_star))
        goals.append(ev.goal); starts.append((0, 0))
        for v, h in res.fotos:
            idx.append(k); vistos.append(v); hfss.append(h)
    return Fotos(nombres, np.stack(reales), np.stack(hstars), np.array(goals), np.array(starts), np.array(idx), np.stack(vistos), np.stack(hfss), ESCALA)


def test_entrena_baja_loss(tmp=None):
    fotos = fotos_chicas()
    assert len(fotos) >= 4
    rng = np.random.default_rng(0)
    x, y, m = lote(fotos, np.arange(min(8, len(fotos))), rng)
    assert x.shape[1:] == (5, 208, 208) and m.any()
    for tipo in ("global", "local"):
        torch.manual_seed(0)
        model = construir(CfConfig(tipo=tipo, ancho=16))
        opt = torch.optim.Adam(model.parameters(), 1e-3)
        l0 = None
        for i in range(6):
            pred = model(x)
            loss = (((pred - y) ** 2) * m).sum() / m.sum()
            opt.zero_grad(); loss.backward(); opt.step()
            l0 = l0 if l0 is not None else loss.item()
        assert loss.item() < l0, (tipo, l0, loss.item())
        print(f"ok {tipo} entrena: {l0:.4f} -> {loss.item():.4f}")
    return fotos


def test_gnn():
    from experiments.train_cf import perdida_gnn
    fotos = fotos_chicas()
    torch.manual_seed(0)
    model = construir(CfConfig(tipo="gnn", ancho=16))
    rng = np.random.default_rng(0)
    opt = torch.optim.Adam(model.entrenables(), 1e-3)
    assert all(not n.startswith("unet.") for n, p in model.named_parameters() if any(p is q for q in model.entrenables()))
    l0 = None
    for i in range(5):
        loss = perdida_gnn(model, fotos, np.arange(min(4, len(fotos))), 64, rng, "cpu")
        opt.zero_grad(); loss.backward(); opt.step()
        l0 = l0 if l0 is not None else loss.item()
    assert loss.item() < l0, (l0, loss.item())
    gp, data, adj, v_s, v_g, h_star = next(problemas("mazes", 1))
    model.eval()
    res = voraz(EvaluadorGNN(data, v_g, model, adj, 20, 16, ESCALA), data, v_s, v_g, adj, max_exp=2000)
    assert res.n_expanded > 10 and res.n_red >= 1
    print(f"ok gnn entrena {l0:.4f} -> {loss.item():.4f} y busca ({res.n_expanded} exp)")


def test_guardar_cargar():
    import tempfile
    from src.cf.fotos import Fotos
    fotos = fotos_chicas()
    with tempfile.TemporaryDirectory() as d:
        fotos.guardar(f"{d}/f")
        f2 = Fotos.cargar(f"{d}/f")
        assert len(f2) == len(fotos) and np.array_equal(f2.hfs, fotos.hfs)
        u = Fotos.unir([fotos, f2])
        assert len(u) == 2 * len(fotos)
    print("ok guardar, cargar y unir")


def test_filtro_igual_scipy():
    from scipy.ndimage import gaussian_filter
    from src.cf.red import Suavizado
    rng = np.random.default_rng(0)
    for sigma in (12.0, 40.0):
        cf = torch.from_numpy(rng.random((2, 208, 208), dtype=np.float32))
        ref = np.stack([gaussian_filter(c, sigma, mode="nearest") for c in cf.numpy()])
        out = Suavizado(sigma)(cf).numpy()
        assert np.abs(out - ref).max() < 1e-4, (sigma, np.abs(out - ref).max())
    print("ok el filtro dentro del modelo es gaussian_filter con mode nearest")


def test_constante_y_epoca_cero_exp_cf2():
    for salida, sigma in (("filtro", 16.0), ("gruesa", 0.0)):
        model = construir(CfConfig(tipo="global", ancho=16, salida=salida, sigma=sigma)).eval()
        with torch.no_grad():
            out = model(torch.rand(2, 5, 208, 208))
        assert out.shape == (2, 208, 208) and torch.equal(out, torch.full_like(out, 0.5)), salida
        for gp, data, adj, v_s, v_g, h_star in problemas("mazes", 2):
            c1, f1, p1, _ = greedy_fs(data, v_s, v_g, adj, 16, 10000)
            res = voraz(EvaluadorCF(data, v_g, model, 20, 16, ESCALA), data, v_s, v_g, adj, max_exp=10000)
            assert res.found == f1 and res.closed == c1 and res.path == p1, (salida, gp.name)
    print("ok cf_filtro y cf_grueso arrancan en 0.5 exacto y su epoca 0 reproduce la voraz con h_fs")


def test_gradiente_exp_cf2():
    fotos = fotos_chicas()
    rng = np.random.default_rng(0)
    x, y, m = lote(fotos, np.arange(min(4, len(fotos))), rng)
    for salida, sigma in (("filtro", 16.0), ("gruesa", 0.0)):
        torch.manual_seed(0)
        model = construir(CfConfig(tipo="global", ancho=16, salida=salida, sigma=sigma))
        opt = torch.optim.Adam(model.parameters(), 1e-3)
        l0 = None
        for i in range(6):
            pred = model(x)
            loss = (((pred - y) ** 2) * m).sum() / m.sum()
            opt.zero_grad(); loss.backward(); opt.step()
            l0 = l0 if l0 is not None else loss.item()
        cab = model.cabeza_gruesa if salida == "gruesa" else model.cabeza
        assert cab.conv.weight.grad is not None and cab.conv.weight.grad.abs().sum() > 0, salida
        assert loss.item() < l0, (salida, l0, loss.item())
        print(f"ok {salida} entrena {l0:.4f} -> {loss.item():.4f} y el gradiente llega a la cabeza")


def test_guardar_cargar_exp_cf2():
    import tempfile
    from src.cf.red import cargar, guardar
    x = torch.rand(1, 5, 208, 208)
    for salida, sigma in (("filtro", 40.0), ("gruesa", 0.0)):
        torch.manual_seed(1)
        model = construir(CfConfig(tipo="global", ancho=16, salida=salida, sigma=sigma))
        with torch.no_grad():
            for p in (model.cabeza_gruesa if salida == "gruesa" else model.cabeza).conv.parameters():
                p.normal_()
        model.eval()
        with tempfile.TemporaryDirectory() as d:
            guardar(f"{d}/m.pt", model, 3, None)
            m2, cfg2, raw = cargar(f"{d}/m.pt")
        assert cfg2.salida == salida and cfg2.sigma == sigma
        with torch.no_grad():
            assert torch.equal(model(x), m2(x))
        assert not torch.equal(m2(x), torch.full((1, 208, 208), 0.5))
    print("ok guardar y cargar conservan la salida filtro y gruesa")


if __name__ == "__main__":
    random.seed(0); np.random.seed(0)
    tests = [test_cf_uno_igual_voraz_hfs, test_oraculo_igual_voraz_hstar, test_etiqueta_en_rango, test_solo_ve_lo_descubierto,
             test_epoca_cero_igual_hfs, test_diedral, test_entrena_baja_loss, test_gnn, test_guardar_cargar,
             test_filtro_igual_scipy, test_constante_y_epoca_cero_exp_cf2, test_gradiente_exp_cf2, test_guardar_cargar_exp_cf2]
    for t in tests:
        t()
    print(f"{len(tests)}/{len(tests)} pruebas ok")
