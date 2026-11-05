# fotos para la cnn de geometria, entrada de 3 canales sin meta ni h_fs y etiqueta de ocupacion real en lo no visto
from pathlib import Path

import numpy as np
import torch

from src.cf.fotos import LADO, Fotos, diedral

CANALES_GEO = 3      # libre visto, obstaculo visto, no visto


def canales_geo(visto, real):
    # visto, real [H, W] bool. fuera del mapa cuenta como obstaculo visto, igual que en cf_filtro
    H, W = real.shape
    x = np.zeros((CANALES_GEO, LADO, LADO), dtype=np.float32)
    x[1] = 1.0
    x[0, :H, :W] = visto & ~real
    x[1, :H, :W] = visto & real
    x[2, :H, :W] = ~visto
    return x


def etiqueta_geo(visto, real):
    # y es la ocupacion real, la mascara es lo no visto dentro del mapa
    H, W = real.shape
    y = np.zeros((LADO, LADO), dtype=np.float32)
    m = np.zeros((LADO, LADO), dtype=bool)
    y[:H, :W] = real
    m[:H, :W] = ~visto
    return y, m


def muestra_geo(fotos, i, k=0):
    mi = int(fotos.mapa_idx[i])
    real, visto = fotos.real[mi], fotos.visto_de(i)
    x = canales_geo(visto, real)
    y, m = etiqueta_geo(visto, real)
    if k:
        x, y, m = diedral(x, k), diedral(y, k), diedral(m, k)
    return np.ascontiguousarray(x), np.ascontiguousarray(y), np.ascontiguousarray(m)


def lote_geo(fotos, idx, rng=None, device="cpu"):
    xs, ys, ms = [], [], []
    for i in idx:
        k = int(rng.integers(8)) if rng is not None else 0
        x, y, m = muestra_geo(fotos, int(i), k)
        xs.append(x); ys.append(y); ms.append(m)
    return (torch.from_numpy(np.stack(xs)).to(device), torch.from_numpy(np.stack(ys)).to(device),
            torch.from_numpy(np.stack(ms)).to(device))


def previo(fotos):
    # fraccion de obstaculos en las celdas no vistas de todas las fotos, el previo constante
    n1, n = 0, 0
    for i in range(len(fotos)):
        mi = int(fotos.mapa_idx[i])
        nv = ~fotos.visto_de(i)
        n1 += int((nv & fotos.real[mi]).sum())
        n += int(nv.sum())
    return n1 / max(n, 1)


def cargar_opcional(path):
    # un conjunto de fotos puede estar vacio a proposito, listo dice 0 fotos
    path = Path(path)
    listo = path / "listo"
    assert listo.exists(), f"fotos incompletas en {path}"
    if listo.read_text().startswith("0 fotos"):
        return None
    return Fotos.cargar(path)


def unir_opcional(partes):
    partes = [p for p in partes if p is not None]
    assert partes, "sin fotos"
    return partes[0] if len(partes) == 1 else Fotos.unir(partes)
