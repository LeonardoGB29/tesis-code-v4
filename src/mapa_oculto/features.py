# caracteristicas simples de lo explorado, al estilo de sail, con costo constante por nodo
# todas miran una ventana de 7x7 del mapa descubierto alrededor del nodo
import numpy as np

from .state import PAD

N_FEAT = 8
NAMES = ["dist_obst", "obst_dfila", "obst_dcol", "frac_obst", "frac_expandidas", "linea_bloqueada",
         "profundidad", "expansiones"]
_OFF = np.arange(-PAD, PAD + 1)
_D = np.maximum(np.abs(_OFF)[:, None], np.abs(_OFF)[None, :])        # chebyshev al centro, 0..3
_NONE = PAD + 1                                                      # no hay obstaculo en la ventana


def local_features(disc, rc, goal, scale):
    # rc: [n, 2] enteros (fila, col); devuelve [n, 8] en rangos cercanos a 0..1
    n = len(rc)
    r, c = rc[:, 0] + PAD, rc[:, 1] + PAD
    rows = r[:, None, None] + _OFF[None, :, None]
    cols = c[:, None, None] + _OFF[None, None, :]
    kn = disc.known[rows, cols]
    ex = disc.expanded[rows, cols]
    dist = np.where(kn, _D[None], _NONE).reshape(n, -1)
    j = dist.argmin(axis=1)
    d = dist[np.arange(n), j]
    has = d < _NONE
    drow = np.where(has, _OFF[j // len(_OFF)], 0)
    dcol = np.where(has, _OFF[j % len(_OFF)], 0)
    # linea recta hacia la meta: hasta 3 pasos, sin pasarse de la meta
    v = np.stack([goal[0] - rc[:, 0], goal[1] - rc[:, 1]], axis=1).astype(np.float64)
    far = np.abs(v).max(axis=1)
    u = v / np.maximum(far, 1e-9)[:, None]
    hit = np.zeros(n, dtype=bool)
    for t in (1, 2, 3):
        tt = np.minimum(t, far)
        rr = np.rint(rc[:, 0] + tt * u[:, 0]).astype(np.int64) + PAD
        cc = np.rint(rc[:, 1] + tt * u[:, 1]).astype(np.int64) + PAD
        hit |= disc.known[rr, cc]
    depth = disc.depth[r, c]
    depth = np.where(depth >= 0, depth, disc.cur_depth + 1)          # aun sin abrir: seria hijo del nodo actual
    return np.stack([d / _NONE, drow / PAD, dcol / PAD, kn.sum(axis=(1, 2)) / kn[0].size,
                     ex.sum(axis=(1, 2)) / ex[0].size, hit, depth / scale,
                     np.full(n, disc.n_expanded / (50.0 * scale))], axis=1).astype(np.float32)
