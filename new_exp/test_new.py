# pruebas de las redes de new_exp: forma, epoca 0 constante, guardado y carga, gradiente y epoca 0 igual a la voraz con h_fs
#   .venv/bin/python -m new_exp.test_new
import tempfile
from pathlib import Path

import numpy as np
import torch

from experiments.common import ESCALA, grids_de, load_problem, set_threads
from new_exp import redes
from src.cf import red
from src.cf.busqueda import Evaluador, EvaluadorCF, voraz
from src.cf.fotos import CANALES, LADO
from src.cf.red import CfConfig

redes.instalar()


def modelo(tipo):
    return red.construir(CfConfig(tipo=tipo, ancho=16, salida="filtro", sigma=16.0))


def test_forma_y_epoca_cero():
    x = torch.rand(2, CANALES, LADO, LADO)
    for tipo in ("local", "transpath"):
        m = modelo(tipo).eval()
        with torch.no_grad():
            y = m(x)
        assert y.shape == (2, LADO, LADO), (tipo, y.shape)
        assert torch.allclose(y, torch.full_like(y, 0.5), atol=1e-6), (tipo, float(y.min()), float(y.max()))


def test_config_transpath():
    m = modelo("transpath")
    c = m.cfg
    assert (c.ancho, c.niveles, c.capas_atencion, c.cabezas) == (64, 3, 4, 4)
    n = sum(p.numel() for p in m.parameters())
    assert 900_000 < n < 1_000_000, n
    print(f"  transpath {n} parametros, local {sum(p.numel() for p in modelo('local').parameters())}")


def test_guardar_y_cargar():
    x = torch.rand(1, CANALES, LADO, LADO)
    for tipo in ("local", "transpath"):
        m = modelo(tipo)
        with torch.no_grad():
            for p in m.parameters():
                p.add_(0.01 * torch.randn_like(p))
        m.eval()
        with tempfile.TemporaryDirectory() as d:
            red.guardar(Path(d) / "m.pt", m, 1, None)
            m2, cfg, _ = red.cargar(Path(d) / "m.pt")
        with torch.no_grad():
            assert torch.allclose(m(x), m2(x), atol=1e-6), tipo
        assert type(m2) is type(m), tipo


def test_gradiente():
    x, y = torch.rand(2, CANALES, LADO, LADO), torch.rand(2, LADO, LADO)
    for tipo in ("local", "transpath"):
        m = modelo(tipo).train()
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        loss = ((m(x) - y) ** 2).mean()
        loss.backward()
        opt.step()
        m.eval()
        with torch.no_grad():
            out = m(x)
        assert float(out.std()) > 0, tipo                      # despues de un paso el factor ya no es constante


def test_suavizado_igual():
    # SuavizadoMat da lo mismo que el Suavizado de src, en salida y en gradiente, con los 3 desvios de la tesis
    for sigma in (16.0, 40.0, 12.0):
        a, b = red.Suavizado(sigma), redes.SuavizadoMat(sigma)
        x = torch.rand(3, LADO, LADO, dtype=torch.float64)
        a, b = a.double(), b.double()
        xa, xb = x.clone().requires_grad_(), x.clone().requires_grad_()
        g = torch.rand(3, LADO, LADO, dtype=torch.float64)
        ya, yb = a(xa), b(xb)
        (ya * g).sum().backward(); (yb * g).sum().backward()
        assert torch.allclose(ya, yb, atol=1e-10), (sigma, float((ya - yb).abs().max()))
        assert torch.allclose(xa.grad, xb.grad, atol=1e-10), sigma
        assert set(a.state_dict()) == set(b.state_dict()), sigma


def test_epoca_cero_es_hfs():
    # con la cabeza en cero el orden de la voraz es el de h_fs, expansion por expansion
    gp = grids_de("mazes", "validation")[0]
    data, adj, v_s, v_g, h_star = load_problem(gp)
    base = voraz(Evaluador(data, v_g, 16, ESCALA), data, v_s, v_g, adj)
    for tipo in ("local", "transpath"):
        m = modelo(tipo).eval()
        res = voraz(EvaluadorCF(data, v_g, m, 20, 16, ESCALA), data, v_s, v_g, adj)
        assert res.n_expanded == base.n_expanded and res.path == base.path, (tipo, res.n_expanded, base.n_expanded)


if __name__ == "__main__":
    set_threads(1)
    torch.manual_seed(0); np.random.seed(0)
    pruebas = [test_forma_y_epoca_cero, test_config_transpath, test_guardar_y_cargar, test_gradiente, test_suavizado_igual, test_epoca_cero_es_hfs]
    for t in pruebas:
        t()
        print(f"ok {t.__name__}", flush=True)
    print(f"{len(pruebas)} pruebas ok")
