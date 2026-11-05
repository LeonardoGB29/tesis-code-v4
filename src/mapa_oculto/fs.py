# h_fs: distancia a la meta en el mapa optimista, lo no descubierto se supone libre
# costo 1 en los ocho movimientos, igual que el oraculo del repo
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

_GRIDS = {}


def full_grid(H, W):
    # grafo 8-conexo de todas las celdas, se arma una vez por tamano
    if (H, W) not in _GRIDS:
        idx = np.arange(H * W).reshape(H, W)
        src, dst = [], []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                a = idx[max(0, -dr):H - max(0, dr), max(0, -dc):W - max(0, dc)]
                b = idx[max(0, dr):H + min(0, dr), max(0, dc):W + min(0, dc)]
                src.append(a.ravel()); dst.append(b.ravel())
        src, dst = np.concatenate(src), np.concatenate(dst)
        g = csr_matrix((np.ones(len(src), dtype=np.float32), (src, dst)), shape=(H * W, H * W))
        g.sort_indices()
        rows = np.repeat(np.arange(H * W), np.diff(g.indptr))
        _GRIDS[(H, W)] = (g.indptr, g.indices, rows)
    return _GRIDS[(H, W)]


def free_space_distance(known, goal_rc):
    # known: [H, W] bool con los obstaculos conocidos; devuelve [H, W] con saltos a la meta
    H, W = known.shape
    indptr, indices, rows = full_grid(H, W)
    blk = known.ravel()
    data = np.ones(len(indices), dtype=np.float32)
    data[blk[rows] | blk[indices]] = 0.0
    g = csr_matrix((data, indices.copy(), indptr.copy()), shape=(H * W, H * W))   # copias: eliminate_zeros edita en sitio
    g.eliminate_zeros()
    d = dijkstra(g, directed=True, unweighted=True, indices=int(goal_rc[0]) * W + int(goal_rc[1]))
    return d.reshape(H, W)


class FreeSpace:
    def __init__(self, disc, goal_rc, every):
        self.goal, self.every = goal_rc, every
        self.h = free_space_distance(disc.known_inmap(), goal_rc)
        self.last = 0                # expansiones al ultimo recalculo
        self.n_updates = 0

    def update(self, disc):
        # recalcula si hay obstaculos nuevos, como maximo una vez cada `every` expansiones
        if disc.dirty and disc.n_expanded - self.last >= self.every:
            self.h = free_space_distance(disc.known_inmap(), self.goal)
            self.last, disc.dirty = disc.n_expanded, False
            self.n_updates += 1
