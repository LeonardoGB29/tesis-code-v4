# voraz con h = h_fs / cf; misma mecanica que la voraz con h_fs y que la de phil
# cada nodo recibe su h una vez al generarse; cf se recalcula cada every_cf expansiones
import heapq
import time
from dataclasses import dataclass

import numpy as np
import torch

from src.enc_phil.encoder import obstacle_grid
from src.mapa_oculto.fs import FreeSpace
from src.phil.data import pack_neighborhood, sample_neighbors

from .estado import Descubierto
from .fotos import canales, empaquetar, hfs_entero


@dataclass
class Resultado:
    closed: set
    found: bool
    path: list | None
    n_expanded: int
    secs_red: float = 0.0
    n_red: int = 0
    fotos: list | None = None


class Evaluador:
    # cf = 1: reproduce la voraz con h_fs
    def __init__(self, data, v_g, every_fs=16, escala=200.0):
        self.rc = data.x.long().numpy()
        self.desc = Descubierto(obstacle_grid(data).numpy().astype(bool))
        self.goal = (int(self.rc[v_g, 0]), int(self.rc[v_g, 1]))
        self.fs = FreeSpace(self.desc, self.goal, every_fs)
        self.escala = float(escala)
        self.secs_red, self.n_red = 0.0, 0

    def observar(self, v):
        self.desc.observar(int(self.rc[v, 0]), int(self.rc[v, 1]))
        self.fs.update(self.desc)

    def h_fs(self, nodos):
        return self.fs.h[self.rc[nodos, 0], self.rc[nodos, 1]].astype(np.float64)

    def cf(self, nodos):
        return np.ones(len(nodos))

    def h(self, nodos):
        return self.h_fs(nodos) / self.cf(nodos)

    def foto(self):
        return empaquetar(self.desc.visto), hfs_entero(self.fs.h)


class EvaluadorOraculo(Evaluador):
    # cf* = h_fs / h*: la busqueda debe reproducir la voraz con h*
    def __init__(self, data, v_g, hstar_mapa, every_fs=16, escala=200.0):
        super().__init__(data, v_g, every_fs, escala)
        self.hstar = hstar_mapa.astype(np.float64)

    def h(self, nodos):
        hf = self.h_fs(nodos)
        hs = self.hstar[self.rc[nodos, 0], self.rc[nodos, 1]]
        cf = np.where(hs > 0, hf / np.where(hs > 0, hs, 1.0), np.where(hs == 0, 1.0, 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            h = np.where(cf > 0, hf / cf, np.inf)
        return np.round(h, 6)                 # quita el error de punto flotante de h_fs / (h_fs / h*)


class EvaluadorCF(Evaluador):
    # cf denso de la u-net o la cnn local sobre la foto del momento
    def __init__(self, data, v_g, model, every_cf=20, every_fs=16, escala=200.0, device="cpu", cf_fijo=None):
        super().__init__(data, v_g, every_fs, escala)
        self.model, self.every_cf, self.device, self.cf_fijo = model, every_cf, device, cf_fijo
        self.mapa_cf, self.ultimo = None, None

    def entrada(self):
        x = canales(self.desc.visto, self.desc.real, self.goal, hfs_entero(self.fs.h), self.escala)
        return torch.from_numpy(x)[None].to(self.device)

    @torch.no_grad()
    def calcular(self):
        t0 = time.time()
        out = self.model(self.entrada())[0].cpu().numpy()
        self.secs_red += time.time() - t0
        self.n_red += 1
        return out

    def actualizar(self):
        if self.mapa_cf is None or self.desc.n_expanded - self.ultimo >= self.every_cf:
            self.mapa_cf = self.calcular()
            self.ultimo = self.desc.n_expanded

    def cf(self, nodos):
        if self.cf_fijo is not None:
            return np.full(len(nodos), self.cf_fijo)
        self.actualizar()
        return self.mapa_cf[self.rc[nodos, 0], self.rc[nodos, 1]].astype(np.float64)


class EvaluadorGNN(EvaluadorCF):
    # rasgos de la u-net cada every_cf expansiones y gnn de un salto por frontera
    def __init__(self, data, v_g, model, adj, every_cf=20, every_fs=16, escala=200.0, device="cpu"):
        super().__init__(data, v_g, model, every_cf, every_fs, escala, device)
        self.data, self.adj = data, adj
        self.goal_t = data.x[v_g].to(device)

    @torch.no_grad()
    def calcular(self):
        t0 = time.time()
        out = self.model.unet.caracteristicas(self.entrada())[0]
        self.secs_red += time.time() - t0
        self.n_red += 1
        return out

    @torch.no_grad()
    def cf(self, nodos):
        self.actualizar()
        t0 = time.time()
        x_f, x_n, ei = pack_neighborhood(self.data, nodos, sample_neighbors(self.adj, nodos, self.model.cfg.n_neighbors))
        cf = self.model(x_f.to(self.device), x_n.to(self.device), ei.to(self.device), self.goal_t, self.mapa_cf)
        self.secs_red += time.time() - t0
        return cf.cpu().numpy().astype(np.float64)


def voraz(ev, data, v_s, v_g, adj, max_exp=10000, ruido=0.0, rng=None, foto_cada=0):
    closed, opened, parent = {v_s}, set(), {v_s: None}
    pq, counter = [], 0
    fotos = [] if foto_cada else None

    def expand(u):
        nonlocal counter
        ev.observar(u)
        if fotos is not None and (ev.desc.n_expanded == 1 or ev.desc.n_expanded % foto_cada == 0):
            fotos.append(ev.foto())
        fr = [v for v in adj[u] if v not in closed and v not in opened]
        if not fr:
            return
        hs = ev.h(fr)
        if ruido:
            hs = hs + rng.uniform(-ruido, ruido, size=len(fr))
        for h, v in zip(hs.tolist(), fr):
            heapq.heappush(pq, (h, counter, v))
            parent[v] = u
            opened.add(v)
            counter += 1

    expand(v_s)
    found = False
    while pq and len(closed) < max_exp:
        u = heapq.heappop(pq)[-1]
        if u in closed:
            continue
        closed.add(u)
        if u == v_g:
            found = True
            break
        expand(u)
    path = None
    if found:
        path, cur = [], v_g
        while cur is not None:
            path.append(cur)
            cur = parent.get(cur)
        path.reverse()
    return Resultado(closed, found, path, len(closed), ev.secs_red, ev.n_red, fotos)


class PhilConRuido:
    # envuelve a phil y suma ruido uniforme a cada h predicha, en saltos sobre la escala
    def __init__(self, model, ruido, rng):
        self.model, self.ruido, self.rng = model, ruido / float(model.coord_scale), rng

    def __getattr__(self, k):
        return getattr(self.model, k)

    def __call__(self, x_f, x_n, ei, x_goal, z_t):
        h, z = self.model(x_f, x_n, ei, x_goal, z_t)
        return h + torch.from_numpy(self.rng.uniform(-self.ruido, self.ruido, size=h.shape[0])).float(), z


class RastreoMem:
    # exp_mem: mapa de bits de abierto (nodos que recibieron h) y fotos en cada refresco de cf, antes de marcar los nodos nuevos
    def iniciar_rastreo(self):
        self.abierto = np.zeros_like(self.desc.real)
        self.fotos_mem = []

    def marcar(self, nodos):
        self.abierto[self.rc[nodos, 0], self.rc[nodos, 1]] = True

    def foto_mem(self):
        return {"visto": empaquetar(self.desc.visto), "expandido": empaquetar(self.desc.expandido), "abierto": empaquetar(self.abierto),
                "hfs": hfs_entero(self.fs.h), "n_exp": int(self.desc.n_expanded)}


class EvaluadorFotosMem(Evaluador, RastreoMem):
    # voraz con h_fs que saca fotos al ritmo del cf: primera llamada y cada every_cf expansiones, igual que EvaluadorCF.actualizar
    def __init__(self, data, v_g, every_cf=20, every_fs=16, escala=200.0):
        super().__init__(data, v_g, every_fs, escala)
        self.iniciar_rastreo()
        self.every_cf, self.ultimo = every_cf, None

    def h(self, nodos):
        if self.ultimo is None or self.desc.n_expanded - self.ultimo >= self.every_cf:
            self.fotos_mem.append(self.foto_mem())
            self.ultimo = self.desc.n_expanded
        out = super().h(nodos)
        self.marcar(nodos)
        return out


class EvaluadorCFFotosMem(EvaluadorCF, RastreoMem):
    # cf_filtro que saca una foto en cada refresco del mapa de cf
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.iniciar_rastreo()

    def calcular(self):
        out = super().calcular()
        self.fotos_mem.append(self.foto_mem())
        return out

    def h(self, nodos):
        out = super().h(nodos)
        self.marcar(nodos)
        return out


class EvaluadorMem(EvaluadorCF, RastreoMem):
    # brazos mem: en cada refresco, u-net congelada, gnn sobre una muestra de OPEN, memoria y residuo grueso; cf = recorte(cf_filtro * exp(R))
    def __init__(self, data, v_g, modelo, nombre, every_cf=20, every_fs=16, escala=200.0, device="cpu", dev_mem="cpu", grabar=False):
        super().__init__(data, v_g, modelo.unet, every_cf, every_fs, escala, device)
        self.iniciar_rastreo()
        self.mem, self.nombre, self.dev_mem, self.grabar = modelo, nombre, dev_mem, grabar
        self.estado_mem = modelo.memoria.inicio(dev_mem)
        self.k = 0
        self.muestras = None                      # lista de (muestra, nodos) si se quiere comprobar contra las fotos
        self.campos = None                        # lista de R por llamada, para comprobar contra el entrenamiento

    @torch.no_grad()
    def calcular(self):
        from .datos_mem import armar_llamada
        t0 = time.time()
        if self.grabar:                                            # fotos propias para dagger, en el mismo instante que las de cf_filtro
            self.fotos_mem.append(self.foto_mem())
        cf0, f = self.model.cf_y_rasgos(self.entrada())
        cf0, f = cf0[0], f[0]
        ent = armar_llamada(self.desc.visto, self.desc.real, self.desc.expandido, self.abierto, hfs_entero(self.fs.h), self.goal,
                            self.nombre, self.k, self.desc.n_expanded, self.mem.cfg, self.escala)
        if self.muestras is not None:
            self.muestras.append((ent["muestra"], ent["nodos"]))
        R, self.estado_mem = self.mem.llamada(f, ent, self.estado_mem, self.dev_mem)
        if self.campos is not None:
            self.campos.append(R.cpu().clone())
        cf = self.mem.cf_de(cf0, R.to(cf0.device))
        self.k += 1
        self.secs_red += time.time() - t0
        self.n_red += 1
        return cf.cpu().numpy()

    def h(self, nodos):
        out = super().h(nodos)
        self.marcar(nodos)
        return out
