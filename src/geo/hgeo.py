# h_geo, dijkstra desde la meta sobre el mapa descubierto con costo 1 en lo libre conocido, bloqueo en lo conocido obstaculo
# y 1 + lam p en lo no visto. con lam 0 es h_fs exacto. se recalcula al mismo ritmo que h_fs, con la ultima prediccion de la cnn
import time

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from src.cf.busqueda import Evaluador
from src.mapa_oculto.fs import full_grid

from .fotos import canales_geo


def distancia_geo(known, visto, p, goal, lam):
    # known [H, W] obstaculos conocidos, visto [H, W], p [H, W] probabilidad de obstaculo, goal (fila, col)
    # la arista que entra a una celda cuesta lo que cuesta esa celda. dijkstra desde la meta pesa cada arista con su celda de origen
    H, W = known.shape
    indptr, indices, rows = full_grid(H, W)
    costo = np.ones(H * W, dtype=np.float64)
    if lam:
        nv = ~visto.ravel()
        costo[nv] = 1.0 + float(lam) * p.ravel()[nv].astype(np.float64)
        costo[int(goal[0]) * W + int(goal[1])] = 1.0
    data = costo[rows]
    blk = known.ravel()
    data[blk[rows] | blk[indices]] = 0.0
    g = csr_matrix((data, indices.copy(), indptr.copy()), shape=(H * W, H * W))
    g.eliminate_zeros()
    d = dijkstra(g, directed=True, indices=int(goal[0]) * W + int(goal[1]), min_only=True)
    return d.reshape(H, W)


class GeoSpace:
    # misma regla que FreeSpace. recalcula si hay obstaculos nuevos y pasaron every expansiones, y toma la p del momento
    def __init__(self, desc, goal, every, lam, p):
        self.goal, self.every, self.lam = goal, every, float(lam)
        self.h = distancia_geo(desc.known_inmap(), desc.visto, p, goal, self.lam)
        self.last = 0
        self.n_updates = 0

    def update(self, desc, p):
        if desc.dirty and desc.n_expanded - self.last >= self.every:
            self.h = distancia_geo(desc.known_inmap(), desc.visto, p, self.goal, self.lam)
            self.last, desc.dirty = desc.n_expanded, False
            self.n_updates += 1


class EvaluadorGeo(Evaluador):
    # voraz con h_geo. la cnn se calcula sobre la foto vacia al inicio y despues cada every_cf expansiones
    def __init__(self, data, v_g, model, lam, every_cf=20, every_fs=16, escala=200.0, device="cpu"):
        super().__init__(data, v_g, every_fs, escala)
        self.model, self.lam, self.every_cf, self.device = model, float(lam), every_cf, device
        self.secs_red, self.n_red = 0.0, 0
        self.p = self.calcular()
        self.ultimo = 0
        self.geo = GeoSpace(self.desc, self.goal, every_fs, self.lam, self.p)

    def entrada(self):
        return torch.from_numpy(canales_geo(self.desc.visto, self.desc.real))[None].to(self.device)

    @torch.no_grad()
    def calcular(self):
        t0 = time.time()
        out = torch.sigmoid(self.model(self.entrada()))[0].cpu().numpy()[:self.desc.H, :self.desc.W]
        self.secs_red += time.time() - t0
        self.n_red += 1
        return out

    def observar(self, v):
        self.desc.observar(int(self.rc[v, 0]), int(self.rc[v, 1]))
        if self.desc.n_expanded - self.ultimo >= self.every_cf:
            self.p = self.calcular()
            self.ultimo = self.desc.n_expanded
        self.geo.update(self.desc, self.p)

    def h(self, nodos):
        return self.geo.h[self.rc[nodos, 0], self.rc[nodos, 1]].astype(np.float64)


class EvaluadorGeoReal(Evaluador):
    # lo que daria una cnn perfecta con el mismo costo, p es la ocupacion real de las celdas no vistas
    def __init__(self, data, v_g, lam, every_fs=16, escala=200.0):
        super().__init__(data, v_g, every_fs, escala)
        self.lam = float(lam)
        self.p = self.desc.real.astype(np.float32)
        self.geo = GeoSpace(self.desc, self.goal, every_fs, self.lam, self.p)

    def observar(self, v):
        self.desc.observar(int(self.rc[v, 0]), int(self.rc[v, 1]))
        self.geo.update(self.desc, self.p)

    def h(self, nodos):
        return self.geo.h[self.rc[nodos, 0], self.rc[nodos, 1]].astype(np.float64)
