# pruebas de exp_mem
#   .venv/bin/python -m tests.test_mem
import random
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

from experiments.common import ESCALA, grids_de, load_problem
from src.cf.busqueda import EvaluadorCF, EvaluadorCFFotosMem, EvaluadorFotosMem, EvaluadorMem, voraz
from src.cf.datos_mem import ESTADO_CLOSED, ESTADO_OPEN, FotosMem, Precalculo, armar_lote, muestra_open, recolectar_mem
from src.cf.fotos import LADO
from src.cf.memoria import MODELOS_MEM, CampoR, MemConfig, MemoriaGRU, MemoriaMamba, bloques_de, construir_mem
from src.cf.phil_res import PhilResCF, ResCfConfig
from src.cf.red import CfConfig, Suavizado, construir
from src.search.greedy import greedy_search

torch.set_num_threads(1)
SIGMA = 12.0
_CACHE = {}


def unet_prueba():
    # cf_filtro chico con la cabeza al azar, asi el cf no es constante
    if "unet" not in _CACHE:
        torch.manual_seed(0)
        u = construir(CfConfig(tipo="global", ancho=16, salida="filtro", sigma=SIGMA))
        with torch.no_grad():
            u.cabeza.conv.weight.normal_(0, 0.3); u.cabeza.conv.bias.fill_(0.2)
        _CACHE["unet"] = u.eval()
    return _CACHE["unet"]


def datos_prueba(n=2):
    # fotos de h_fs y de la u-net de prueba sobre n mapas de entrenamiento de mazes, con precalculo
    if "datos" not in _CACHE:
        unet = unet_prueba()
        grids = grids_de("mazes", "train")[:n]
        fotos, _ = recolectar_mem("mazes", grids, lambda data, adj, v_g: [EvaluadorFotosMem(data, v_g, 20, 16, ESCALA),
                                                                           EvaluadorCFFotosMem(data, v_g, unet, 20, 16, ESCALA)], 10000)
        d = tempfile.TemporaryDirectory()
        fotos.guardar(f"{d.name}/fotos")
        fotos = FotosMem.cargar(f"{d.name}/fotos")
        pre = Precalculo.crear(f"{d.name}/pre", fotos, unet, MemConfig(sigma=SIGMA), "cpu", lote=8)
        _CACHE["datos"] = (grids, fotos, pre, d)
    return _CACHE["datos"][:3]


def modelo_prueba(modelo, azar=False):
    torch.manual_seed(1)
    m = construir_mem(MemConfig(modelo=modelo, sigma=SIGMA), unet_prueba())
    if azar:
        with torch.no_grad():
            m.cabeza[-1].weight.normal_(0, 0.5); m.cabeza[-1].bias.fill_(0.1)
    return m.eval()


def test_campo_igual_interp_suavizado():
    torch.manual_seed(0)
    for sigma in (12.0, 40.0):
        campo, suave = CampoR(sigma, 8), Suavizado(sigma)
        b = torch.randn(3, 26, 26)
        with torch.no_grad():
            ref = suave(F.interpolate(b.unsqueeze(1), size=(LADO, LADO), mode="bilinear", align_corners=False).squeeze(1))
            out = campo(b)
            cero = campo(torch.zeros(2, 26, 26))
        assert (out - ref).abs().max() < 1e-4, (sigma, (out - ref).abs().max())
        assert torch.equal(cero, torch.zeros_like(cero)), sigma
    print("ok el campo R es interpolacion mas suavizado, y bloques en cero dan campo en cero exacto")


def test_bloques_vacios_cero():
    r = torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0])
    celdas = torch.tensor([[0, 0], [7, 7], [8, 0], [100, 100], [103, 101]])          # bloques (0,0) (0,0) (1,0) (12,12) (12,12)
    foto = torch.tensor([0, 0, 0, 1, 1])
    b = bloques_de(r, celdas, foto, 2, 8, 26)
    assert b.shape == (2, 26, 26)
    assert abs(b[0, 0, 0].item() - 2.0) < 1e-6 and abs(b[0, 1, 0].item() - 5.0) < 1e-6 and abs(b[1, 12, 12].item() - 8.0) < 1e-6
    nz = (b != 0).sum().item()
    assert nz == 3, nz
    print("ok la media por bloque es correcta y los bloques sin nodos quedan en 0 exacto")


def test_epoca_cero_igual_cf_filtro():
    unet = unet_prueba()
    phil = PhilResCF(ResCfConfig(fs_every=16, coord_scale=ESCALA, d_enc=2, every_cf=20), unet=unet).eval()
    modelos = {m: modelo_prueba(m) for m in MODELOS_MEM}
    for gp in grids_de("mazes", "test")[:2]:
        data, adj, v_s, v_g, h_star = load_problem(gp)
        ref = voraz(EvaluadorCF(data, v_g, unet, 20, 16, ESCALA), data, v_s, v_g, adj, max_exp=10000)
        for nombre, m in modelos.items():
            r = voraz(EvaluadorMem(data, v_g, m, gp.name, 20, 16, ESCALA), data, v_s, v_g, adj, max_exp=10000)
            assert r.found == ref.found and r.closed == ref.closed and r.path == ref.path, (nombre, gp.name)
        r = greedy_search(phil, data, v_s, v_g, adj=adj, max_exp=10000, return_path=True)
        assert r.found == ref.found and r.closed == ref.closed and r.path == ref.path, ("phil_res_cf", gp.name)
    print("ok la epoca 0 de mem_no, mem_gru, mem_mamba y phil_res_cf reproduce la busqueda de cf_filtro expansion por expansion")


def test_secuencia_igual_paso():
    torch.manual_seed(0)
    devs = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    for dev in devs:
        for cls in (MemoriaGRU, MemoriaMamba):
            m = (cls(64, 64) if cls is MemoriaGRU else cls(64, 64, 16, 4)).to(dev).eval()
            u = torch.randn(1, 41, 64, device=dev)
            with torch.no_grad():
                Z = m.secuencia(u)
                est, zs = m.inicio(dev), []
                for t in range(41):
                    z, est = m.paso(u[:, t], est); zs.append(z)
            dif = (torch.stack(zs, 1) - Z).abs().max().item()
            assert dif < 1e-5, (dev, cls.__name__, dif)
    print(f"ok gru y mamba dan lo mismo en secuencia y paso a paso en {devs}")


def test_muestra_reproducible():
    grids, fotos, pre = datos_prueba()
    i = max(range(len(fotos)), key=lambda j: (fotos.bits("abierto", j) & ~fotos.bits("expandido", j)).sum())
    ab, ex = fotos.bits("abierto", i), fotos.bits("expandido", i)
    assert (ab & ~ex).sum() > 128, "la prueba necesita una foto con OPEN grande"
    a, b, c = muestra_open(ab, ex, "x.png", 3, 128), muestra_open(ab, ex, "x.png", 3, 128), muestra_open(ab, ex, "x.png", 4, 128)
    assert np.array_equal(a, b) and len(a) == 128 and not np.array_equal(a, c)
    # la busqueda con el modelo reproduce las muestras y los nodos guardados con las fotos del mismo mapa
    m = modelo_prueba("mem_no")
    data, adj, v_s, v_g, h_star = load_problem(grids[0])
    ev = EvaluadorMem(data, v_g, m, grids[0].name, 20, 16, ESCALA); ev.muestras = []
    voraz(ev, data, v_s, v_g, adj, max_exp=10000)
    idx = [j for j in range(len(fotos)) if fotos.mapa(j) == 0 and fotos.origen[j] == 1]
    assert len(idx) == len(ev.muestras), (len(idx), len(ev.muestras))
    for k, j in enumerate(idx):
        assert np.array_equal(pre.nodos[j, :int(pre.n_nodos[j])], ev.muestras[k][1]), k
        assert int(pre.n_front[j]) == len(ev.muestras[k][0])
    print("ok la muestra de OPEN es reproducible y la busqueda da las mismas muestras y nodos que las fotos")


def test_vecinos_solo_vistos():
    grids, fotos, pre = datos_prueba()
    n_nodos = 0
    for i in range(len(fotos)):
        m = fotos.mapa(i)
        visto, ex, ab = fotos.bits("visto", i), fotos.bits("expandido", i), fotos.bits("abierto", i)
        libre = visto & ~fotos.real[m]
        n, nf = int(pre.n_nodos[i]), int(pre.n_front[i])
        if n == 0:
            continue
        nod = pre.nodos[i, :n].astype(int)
        assert libre[nod[:, 0], nod[:, 1]].all(), i
        est = pre.estado[i, :n]
        assert (est[:nf] == ESTADO_OPEN).all() and (est[ex[nod[:, 0], nod[:, 1]]] == ESTADO_CLOSED).all(), i
        assert (ab[nod[:nf, 0], nod[:nf, 1]] & ~ex[nod[:nf, 0], nod[:nf, 1]]).all(), i
        assert 0.0 <= pre.escalares[i, 1] <= 1.0
        n_nodos += n
    assert n_nodos > 0
    print(f"ok {n_nodos} nodos de la muestra y sus vecinos son celdas vistas libres con el estado correcto")


def test_recorte_entrenamiento_igual_busqueda_y_gradiente():
    # con una cabeza al azar la busqueda cambia desde la primera llamada, asi que las fotos se toman con el mismo modelo (grabar) y se precalculan
    grids, _, _ = datos_prueba()
    unet = unet_prueba()
    data, adj, v_s, v_g, h_star = load_problem(grids[0])
    for modelo in MODELOS_MEM:
        m = modelo_prueba(modelo, azar=True)
        cfg = m.cfg
        evs = []
        def hacer(data_, adj_, v_g_):
            ev = EvaluadorMem(data_, v_g_, m, grids[0].name, 20, 16, ESCALA, grabar=True); ev.campos = []; evs.append(ev)
            return [ev]
        fotos, _ = recolectar_mem("mazes", [grids[0]], hacer, 10000)
        ev = evs[0]
        with tempfile.TemporaryDirectory() as d:
            fotos.guardar(f"{d}/fotos"); fotos = FotosMem.cargar(f"{d}/fotos")
            pre = Precalculo.crear(f"{d}/pre", fotos, unet, cfg, "cpu", lote=8)
            secs = fotos.secuencias()
            assert len(secs) == 1 and secs[0][1] == len(ev.campos), (secs, len(ev.campos))
            with torch.no_grad():
                lote = armar_lote(fotos, pre, secs, cfg, "cpu")
                R_ent = m.campo_lote(lote)
                cf_ent = m.cf_de(lote["cf0"], R_ent)
            dif = max((ev.campos[k] - R_ent[k]).abs().max().item() for k in range(len(ev.campos)))
            assert dif < 1e-4, (modelo, dif)
            assert (R_ent != 0).any(), modelo
            assert cf_ent.min() >= 1e-3 and cf_ent.max() <= 1.0
            # el cf que uso la busqueda en la ultima llamada es el del entrenamiento, salvo el redondeo del cf_filtro guardado en float16
            assert np.abs(ev.mapa_cf - cf_ent[-1].numpy()).max() < 2e-3, (modelo, np.abs(ev.mapa_cf - cf_ent[-1].numpy()).max())
            # entrenar baja la perdida y el gradiente llega a la cabeza
            m.train()
            opt = torch.optim.Adam(m.entrenables(), 1e-3)
            l0 = None
            for _ in range(5):
                lote = armar_lote(fotos, pre, secs, cfg, "cpu")
                cf = m.cf_de(lote["cf0"], m.campo_lote(lote))
                loss = (((cf - lote["y"]) ** 2) * lote["m"]).sum() / lote["m"].sum()
                opt.zero_grad(); loss.backward(); opt.step()
                l0 = l0 if l0 is not None else loss.item()
            assert m.cabeza[-1].weight.grad.abs().sum() > 0 and loss.item() < l0, (modelo, l0, loss.item())
        print(f"ok {modelo}: R y cf de la busqueda iguales a los del entrenamiento (dif {dif:.1e}), cf recortado, entrena {l0:.4f} -> {loss.item():.4f}")


def test_lote_igual_secuencia_por_secuencia():
    grids, fotos, pre = datos_prueba()
    secs = fotos.secuencias()
    for modelo in ("mem_gru", "mem_mamba"):
        m = modelo_prueba(modelo, azar=True)
        with torch.no_grad():
            R_lote = m.campo_lote(armar_lote(fotos, pre, secs, m.cfg, "cpu"))
            partes = [m.campo_lote(armar_lote(fotos, pre, [s], m.cfg, "cpu")) for s in secs]
        R_sep = torch.cat(partes)
        dif = (R_lote - R_sep).abs().max().item()
        assert dif < 1e-4, (modelo, dif)
    print("ok el lote con relleno da el mismo R que secuencia por secuencia en gru y mamba")


if __name__ == "__main__":
    random.seed(0); np.random.seed(0)
    tests = [test_campo_igual_interp_suavizado, test_bloques_vacios_cero, test_epoca_cero_igual_cf_filtro, test_secuencia_igual_paso,
             test_muestra_reproducible, test_vecinos_solo_vistos, test_recorte_entrenamiento_igual_busqueda_y_gradiente, test_lote_igual_secuencia_por_secuencia]
    for t in tests:
        t()
    print(f"{len(tests)}/{len(tests)} pruebas ok")
