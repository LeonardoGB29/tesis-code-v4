# pruebas de exp_escalera
#   .venv/bin/python -m tests.test_escalera
import copy
import random
import tempfile

import numpy as np
import torch

from experiments.common import ESCALA, EXP_CF2, RESULTS, grids_de, load_phil, load_problem
from src.cf.busqueda import EvaluadorCF, voraz
from src.cf.escalera import D_CNN, D_RASGOS, Compuerta, Escalera, EscaleraConfig, RecorteRecto, columnas, construir_desde
from src.cf.red import cargar
from src.phil.config import PhilConfig
from src.phil.data import pack_neighborhood, sample_neighbors
from src.phil.model import PHIL
from src.phil.training import collect_trajectory, tbtt_loss
from src.search.greedy import greedy_search

torch.set_num_threads(1)
DS = "mazes"
CK = EXP_CF2 / "cf_filtro" / "runs" / DS / "cf_filtro_s0" / "best.pt"
PHIL_CK = RESULTS / "phil_res_s" / "runs" / DS / "phil_res_s_n108_s0" / "best.pt"


def problemas(split="validation", n=2):
    for gp in grids_de(DS, split)[:n]:
        yield (gp,) + load_problem(gp)


def unet_y_cfg(peldano, **k):
    unet, ucfg, _ = cargar(CK, "cpu")
    cfg = EscaleraConfig(peldano=peldano, d_enc=D_CNN, fs_every=ucfg.every_fs, every_cf=ucfg.every_cf, coord_scale=ESCALA, cf_ckpt=str(CK),
                         phil_ckpt=str(PHIL_CK), rasgos_media=[0.1] * D_RASGOS, rasgos_std=[0.5] * D_RASGOS, **k)
    return unet, ucfg, cfg


def modelo(peldano, seed=0, **k):
    torch.manual_seed(seed)
    unet, ucfg, cfg = unet_y_cfg(peldano, **k)
    if peldano == "C":
        return Compuerta(cfg, load_phil(PHIL_CK)[0], unet, "cpu").eval(), ucfg
    return Escalera(cfg, unet, "cpu").eval(), ucfg


def estado_de(model, n_exp=120):
    gp, data, adj, v_s, v_g, h_star = next(problemas("test", 1))
    greedy_search(model, data, v_s, v_g, adj=adj, max_exp=n_exp)
    return data, adj, v_s, v_g


def test_columnas():
    m, ucfg = modelo("R2")
    data, adj, v_s, v_g = estado_de(m, 150)
    ev = m.ev
    libres = np.argwhere(~ev.desc.real)
    rc = libres[np.random.default_rng(0).choice(len(libres), 64, replace=False)]
    base, ent = columnas(ev, rc, m.media, m.std, ESCALA)
    assert base.shape == (64, 2) and base.dtype == np.float64 and ent.shape == (64, D_CNN) and ent.dtype == np.float32
    hfs = ev.fs.h[rc[:, 0], rc[:, 1]]
    fin = np.isfinite(hfs)
    assert np.allclose(base[fin, 1] * ESCALA, hfs[fin], rtol=1e-12, atol=0) and np.allclose(base[fin, 0] * ESCALA, hfs[fin] / ev.mapa_cf[rc[fin, 0], rc[fin, 1]].astype(np.float64), rtol=1e-12, atol=0)
    assert np.array_equal(base[:, 1], hfs / ESCALA) and np.array_equal(base[:, 0], hfs / ev.mapa_cf[rc[:, 0], rc[:, 1]].astype(np.float64) / ESCALA), "la base se calcula igual que en phil_res_cf"
    assert np.isfinite(ent).all() and (ent[:, 2] > 0).all() and (ent[:, 2] <= 1).all()
    assert (base[:, 0] >= base[:, 1] - 1e-12).all(), "b nunca es menor que h_fs porque cf esta entre 0 y 1"
    assert np.allclose(ent[fin, 1], base[fin, 1]) and np.allclose(ent[fin, 0], np.minimum(base[fin, 0], ev.fs.h.size / ESCALA))
    print("ok columnas, base exacta en float64 y 43 entradas finitas")


def test_iteracion_cero_igual_cf_filtro():
    unet, ucfg, _ = unet_y_cfg("R2")
    brazos = {r: modelo(r)[0] for r in ("R2", "R3", "R4", "C")}
    for gp, data, adj, v_s, v_g, h_star in problemas("validation", 2):
        ref = voraz(EvaluadorCF(data, v_g, unet, ucfg.every_cf, ucfg.every_fs, ESCALA), data, v_s, v_g, adj, max_exp=10000)
        for r, m in brazos.items():
            res = greedy_search(m, data, v_s, v_g, adj=adj, max_exp=10000, return_path=True)
            assert res.found == ref.found and res.closed == ref.closed and res.path == ref.path, (r, gp.name, res.n_expanded, ref.n_expanded)
            assert m.n_red >= res.n_expanded // ucfg.every_cf
    print("ok la iteracion 0 de R2, R3, R4 y C reproduce la busqueda de cf_filtro expansion por expansion")


def test_alpha_cero_y_gradiente_derecho():
    z = torch.tensor([-0.5, 0.0, 0.3, 1.7], requires_grad=True)
    a = RecorteRecto.apply(z)
    assert torch.equal(a, torch.tensor([0.0, 0.0, 0.3, 1.0]))
    a.sum().backward()
    assert torch.equal(z.grad, torch.ones(4)), "el gradiente pasa derecho tambien fuera de [0, 1]"
    m, _ = modelo("C")
    gp, data, adj, v_s, v_g, h_star = next(problemas("test", 1))
    m.train()
    cfg = m.cfg
    traj, z0 = collect_trajectory(m, data, adj, h_star, v_s, v_g, cfg, 1.0)
    assert traj
    for V, N, t, st in traj[:3]:
        assert torch.equal(st[0][:len(V), 0], st[0][:len(V), 0]) and st[1].shape[1] == D_CNN
    loss = tbtt_loss(m, data, data.x[v_g], traj, z0)
    loss.backward()
    g = m.compuerta[-1].weight.grad
    assert g is not None and g.abs().sum() > 0, "el gradiente llega a la compuerta aunque alpha sea 0"
    assert all(p.grad is None for p in m.phil.parameters()) and all(p.grad is None for p in m.unet.parameters())
    print(f"ok alpha arranca en 0 exacto y el gradiente llega a la compuerta ({float(g.abs().sum()):.2e})")


def test_delta_acotado():
    m, ucfg = modelo("R3", tope_delta=0.25)
    with torch.no_grad():
        m.heuristic.mlp[-1].weight.normal_(0, 3.0); m.heuristic.mlp[-1].bias.fill_(2.0)
    data, adj, v_s, v_g = estado_de(m, 100)
    ev = m.ev
    libres = np.argwhere(~ev.desc.real)
    rc = libres[np.random.default_rng(1).choice(len(libres), 40, replace=False)]
    x_f, x_n = torch.tensor(rc[:8], dtype=torch.float), torch.tensor(rc[8:], dtype=torch.float)
    ei = torch.tensor([list(range(8, 40)), [i % 8 for i in range(32)]], dtype=torch.long)
    with torch.no_grad():
        h, _ = m(x_f, x_n, ei, data.x[v_g], torch.zeros(m.cfg.d_hidden))
    base, _ = m._last
    b, hfs = base[:8, 0], base[:8, 1]
    assert h.dtype == torch.float64 and (h >= hfs - 1e-12).all() and (h <= b + 0.25 + 1e-9).all() and (h >= torch.minimum(b, b - 0.25) - 1e-9).all()
    assert (h - b).abs().max() > 0.05, "con pesos al azar la correccion se mueve"
    print("ok la correccion queda dentro de mas menos tope y nunca baja de h_fs")


def test_fuga():
    for r in ("R2", "C"):
        m, _ = modelo(r)
        data, adj, v_s, v_g = estado_de(m, 150)
        ev = m.ev
        visto, real = ev.desc.visto, ev.desc.real
        rng = np.random.default_rng(0)
        nv = np.argwhere(~visto)
        real2 = real.copy()
        for rr, cc in nv[rng.choice(len(nv), 200, replace=False)]:
            real2[rr, cc] = ~real2[rr, cc]
        ev2 = copy.copy(ev); ev2.desc = ev.desc.copia(); ev2.desc.real = real2
        for e in (ev, ev2):                      # los dos recalculan la cnn en el mismo estado de la busqueda
            e.mapa_cf = None; e.mapa_rasgos = None
        libres = np.argwhere(~real & ~real2)
        rc = libres[rng.choice(len(libres), 40, replace=False)]
        b1, e1 = columnas(ev, rc, m.media, m.std, ESCALA)
        b2, e2 = columnas(ev2, rc, m.media, m.std, ESCALA)
        assert np.array_equal(b1, b2) and np.array_equal(e1, e2)
        x_f, x_n = torch.tensor(rc[:8], dtype=torch.float), torch.tensor(rc[8:], dtype=torch.float)
        ei = torch.tensor([list(range(8, 40)), [i % 8 for i in range(32)]], dtype=torch.long)
        with torch.no_grad():
            m.ev = ev; h1, _ = m(x_f, x_n, ei, data.x[v_g], torch.zeros(m.cfg.d_hidden))
            m.ev = ev2; h2, _ = m(x_f, x_n, ei, data.x[v_g], torch.zeros(m.cfg.d_hidden))
        assert torch.equal(h1, h2), r
        # control, tapar celdas vistas libres si cambia las columnas
        vistas = np.argwhere(visto & ~real)
        real3 = real.copy()
        for rr, cc in vistas[rng.choice(len(vistas), 20, replace=False)]:
            real3[rr, cc] = True
        ev3 = copy.copy(ev); ev3.desc = ev.desc.copia(); ev3.desc.real = real3; ev3.mapa_cf = None; ev3.mapa_rasgos = None
        _, e3 = columnas(ev3, rc, m.media, m.std, ESCALA)
        assert not np.array_equal(e1, e3)
    print("ok las 43 columnas y las salidas de R2 y C no ven las celdas no vistas")


def test_entrena():
    for r in ("R1", "R2", "R4"):
        m, _ = modelo(r)
        m.train()
        gp, data, adj, v_s, v_g, h_star = next(problemas("test", 1))
        opt = torch.optim.Adam(m.parameters_phil(), 1e-3)
        traj, z0 = collect_trajectory(m, data, adj, h_star, v_s, v_g, m.cfg, 1.0)
        assert traj and all(st[1].shape[1] == D_CNN for _, _, _, st in traj)
        l0 = None
        for _ in range(6):
            loss = tbtt_loss(m, data, data.x[v_g], traj, z0)
            opt.zero_grad(); loss.backward(); opt.step()
            l0 = l0 if l0 is not None else loss.item()
        assert loss.item() < l0, (r, l0, loss.item())
        assert all(p.grad is None for p in m.unet.parameters())
        print(f"ok {r} entrena {l0:.4f} -> {loss.item():.4f} con la u-net quieta")


def test_guardar_cargar():
    gp, data, adj, v_s, v_g, h_star = next(problemas("test", 1))
    with tempfile.TemporaryDirectory() as d:
        for r in ("R0", "R2", "C"):
            if r == "R0":
                torch.manual_seed(0)
                m = PHIL(PhilConfig(coord_scale=ESCALA)).eval()
                estado, cfg = m.state_dict(), m.cfg
            else:
                m, _ = modelo(r, seed=3)
                with torch.no_grad():
                    for q in (m.compuerta if r == "C" else m.heuristic).parameters():
                        q.normal_(0, 0.1)
                estado, cfg = m.state_dict_phil(), m.cfg
            torch.save({"iteration": 5, "model_state": estado, "config": {**cfg.to_dict(), "peldano": r}, "val": {"exp": 1.0}, "extra": {"peldano": r}}, f"{d}/{r}.pt")
            m2, cfg2, raw = load_phil(f"{d}/{r}.pt")
            assert type(m2) is type(m) and raw["iteration"] == 5
            res1 = greedy_search(m, data, v_s, v_g, adj=adj, max_exp=300, return_path=True)
            random.seed(0)
            res2 = greedy_search(m2, data, v_s, v_g, adj=adj, max_exp=300, return_path=True)
            random.seed(0)
            res1 = greedy_search(m, data, v_s, v_g, adj=adj, max_exp=300, return_path=True)
            assert res1.closed == res2.closed and res1.n_expanded == res2.n_expanded, r
            if r != "R0":
                assert not any(k.startswith("unet.") or k.startswith("phil.") for k in estado)
    print("ok guardar y cargar R0, R2 y C sin la u-net ni phil_res_s adentro")


if __name__ == "__main__":
    random.seed(0); np.random.seed(0)
    tests = [test_columnas, test_iteracion_cero_igual_cf_filtro, test_alpha_cero_y_gradiente_derecho, test_delta_acotado, test_fuga, test_entrena, test_guardar_cargar]
    for t in tests:
        t()
    print(f"{len(tests)}/{len(tests)} pruebas ok")
