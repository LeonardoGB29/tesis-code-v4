# metricas de la cnn de geometria sobre celdas no vistas. bce, auc por histogramas y calibracion, mas los estratos
import numpy as np
from scipy.ndimage import distance_transform_cdt, maximum_filter

from src.mapa_oculto.fs import free_space_distance

NB = 2000                 # tramos del histograma de p para el auc
TRAMOS_CAL = 10
FRONTERA = [(1, 1), (2, 3), (4, 7), (8, 15), (16, 10**9)]
NOMBRES_FRONTERA = ["frontera_1", "frontera_2_3", "frontera_4_7", "frontera_8_15", "frontera_16_mas"]


class Acumulador:
    def __init__(self):
        self.n0 = np.zeros(NB); self.n1 = np.zeros(NB)
        self.sp0 = np.zeros(NB); self.sp1 = np.zeros(NB)
        self.bce = 0.0
        self.n = 0

    def agregar(self, p, y):
        # p probabilidades, y bool, los dos en 1d
        if len(p) == 0:
            return
        p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1.0 - 1e-7)
        y = np.asarray(y, dtype=bool)
        b = np.minimum((p * NB).astype(np.int64), NB - 1)
        self.n1 += np.bincount(b[y], minlength=NB)
        self.n0 += np.bincount(b[~y], minlength=NB)
        self.sp1 += np.bincount(b[y], weights=p[y], minlength=NB)
        self.sp0 += np.bincount(b[~y], weights=p[~y], minlength=NB)
        self.bce += -(np.log(p[y]).sum() + np.log1p(-p[~y]).sum())
        self.n += len(p)

    def resultados(self):
        N1, N0 = self.n1.sum(), self.n0.sum()
        N = N1 + N0
        if N == 0:
            return {"n": 0, "frac_obst": float("nan"), "bce": float("nan"), "auc": float("nan"), "ece": float("nan")}, []
        # auc, cada positivo gana contra los negativos de tramos menores y empata en su tramo
        abajo = np.cumsum(self.n0) - self.n0
        auc = float((self.n1 * (abajo + 0.5 * self.n0)).sum() / (N1 * N0)) if N1 > 0 and N0 > 0 else float("nan")
        # calibracion en TRAMOS_CAL tramos iguales de p
        k = NB // TRAMOS_CAL
        n_t = (self.n0 + self.n1).reshape(TRAMOS_CAL, k).sum(1)
        sp_t = (self.sp0 + self.sp1).reshape(TRAMOS_CAL, k).sum(1)
        n1_t = self.n1.reshape(TRAMOS_CAL, k).sum(1)
        tabla, ece = [], 0.0
        for t in range(TRAMOS_CAL):
            if n_t[t] > 0:
                mp, fr = sp_t[t] / n_t[t], n1_t[t] / n_t[t]
                ece += n_t[t] / N * abs(mp - fr)
                tabla.append({"tramo": t, "p_min": t / TRAMOS_CAL, "p_max": (t + 1) / TRAMOS_CAL, "n": int(n_t[t]), "p_media": float(mp), "frac_obst": float(fr)})
        return {"n": int(N), "frac_obst": float(N1 / N), "bce": float(self.bce / N), "auc": auc, "ece": float(ece)}, tabla


def corredor(real, hstar_mapa, start, K=10, r=3):
    # celdas libres a no mas de K saltos del camino optimo, dilatadas r celdas con un maximo local
    d_start = free_space_distance(real, (int(start[0]), int(start[1])))
    hs = hstar_mapa.astype(np.float64)
    hs[hstar_mapa < 0] = np.inf
    opt = hs[int(start[0]), int(start[1])]
    casi = np.isfinite(d_start) & np.isfinite(hs) & (d_start + hs <= opt + K)
    return maximum_filter(casi, size=2 * r + 1, mode="constant", cval=0)


def distancia_frontera(visto):
    # distancia de tablero de cada celda no vista a la celda vista mas cercana, 0 en lo visto
    if visto.all():
        return np.zeros(visto.shape, dtype=np.int64)
    return distance_transform_cdt(~visto, metric="chessboard").astype(np.int64)


def estratos(visto, real, hstar_mapa, start):
    # mascaras por nombre, todas dentro de lo no visto
    nv = ~visto
    out = {"todo": nv, "corredor": nv & corredor(real, hstar_mapa, start)}
    d = distancia_frontera(visto)
    for (a, b), nombre in zip(FRONTERA, NOMBRES_FRONTERA):
        out[nombre] = nv & (d >= a) & (d <= b)
    return out
