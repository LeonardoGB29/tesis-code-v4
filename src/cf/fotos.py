# fotos del mapa descubierto: entrada de la red, etiqueta cf* y aumento diedral
from pathlib import Path

import numpy as np
import torch

LADO = 208           # 201 rellenado a multiplo de 16
CANALES = 5          # libre conocido, obstaculo conocido, desconocido, meta, h_fs normalizada
H_TOPE = 2.0         # h_fs sobre la escala se recorta aqui, tambien lo inalcanzable
CF_MIN = 1e-3


def hfs_entero(h):
    # h_fs float con inf -> int16 con -1
    out = np.where(np.isfinite(h), h, -1).astype(np.int16)
    return out


def canales(visto, real, goal, hfs, escala):
    # visto, real [H, W] bool; hfs [H, W] int16 con -1 inalcanzable; devuelve [5, LADO, LADO] float32
    H, W = real.shape
    x = np.zeros((CANALES, LADO, LADO), dtype=np.float32)
    x[1] = 1.0                                           # fuera del mapa: obstaculo conocido
    x[4] = H_TOPE
    x[0, :H, :W] = visto & ~real
    x[1, :H, :W] = visto & real
    x[2, :H, :W] = ~visto
    x[3, goal[0], goal[1]] = 1.0
    h = hfs.astype(np.float32) / escala
    h[hfs < 0] = H_TOPE
    x[4, :H, :W] = np.minimum(h, H_TOPE)
    return x


def etiqueta(hfs, hstar, real, goal):
    # cf* = h_fs / h* en celdas libres alcanzables, 1 en la meta; devuelve y [LADO, LADO] y mascara
    H, W = real.shape
    y = np.zeros((LADO, LADO), dtype=np.float32)
    m = np.zeros((LADO, LADO), dtype=bool)
    ok = (~real) & (hstar > 0) & (hfs >= 0)
    y[:H, :W][ok] = hfs[ok].astype(np.float32) / hstar[ok].astype(np.float32)
    m[:H, :W] = ok
    y[goal[0], goal[1]] = 1.0
    m[goal[0], goal[1]] = True
    return y, m


def diedral(a, k):
    # las 8 simetrias del cuadrado sobre los dos ultimos ejes; k en 0..7
    if k >= 4:
        a = np.flip(a, axis=-1)
    return np.rot90(a, k % 4, axes=(-2, -1))


def diedral_inversa(a, k):
    a = np.rot90(a, -(k % 4), axes=(-2, -1))
    if k >= 4:
        a = np.flip(a, axis=-1)
    return a


class Fotos:
    # conjunto de fotos en memoria: por mapa real, h* y meta; por foto visto empaquetado y h_fs
    def __init__(self, nombres, real, hstar, goal, start, mapa_idx, visto, hfs, escala):
        self.nombres, self.real, self.hstar, self.goal, self.start = nombres, real, hstar, goal, start
        self.mapa_idx, self.visto, self.hfs, self.escala = mapa_idx, visto, hfs, float(escala)

    def __len__(self):
        return len(self.mapa_idx)

    def visto_de(self, i):
        H, W = self.real.shape[1:]
        return np.unpackbits(self.visto[i])[:H * W].reshape(H, W).astype(bool)

    def muestra(self, i, k=0):
        # entrada, etiqueta y mascara de la foto i con la simetria k
        m = int(self.mapa_idx[i])
        real, goal = self.real[m], self.goal[m]
        visto, hfs = self.visto_de(i), self.hfs[i]
        x = canales(visto, real, goal, hfs, self.escala)
        y, msk = etiqueta(hfs, self.hstar[m], real, goal)
        if k:
            x, y, msk = diedral(x, k), diedral(y, k), diedral(msk, k)
        return np.ascontiguousarray(x), np.ascontiguousarray(y), np.ascontiguousarray(msk)

    def guardar(self, path):
        # carpeta con un .npy por arreglo; visto y hfs se leen con mmap y los procesos comparten las paginas
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for k, v in [("nombres", np.array(self.nombres)), ("real", self.real), ("hstar", self.hstar), ("goal", self.goal), ("start", self.start),
                     ("mapa_idx", self.mapa_idx), ("visto", self.visto), ("hfs", self.hfs), ("escala", np.array(self.escala))]:
            np.save(path / f"{k}.npy", v)
        (path / "listo").write_text(f"{len(self.mapa_idx)} fotos\n")

    @classmethod
    def cargar(cls, path):
        path = Path(path)
        assert (path / "listo").exists(), f"fotos incompletas en {path}"
        ld = lambda k, mm=None: np.load(path / f"{k}.npy", mmap_mode=mm, allow_pickle=False)
        return cls([str(n) for n in ld("nombres")], ld("real"), ld("hstar"), ld("goal"), ld("start"), ld("mapa_idx"),
                   ld("visto", "r"), ld("hfs", "r"), float(ld("escala")))

    @classmethod
    def unir(cls, partes):
        # concatena conjuntos del mismo dataset; si las listas de mapas difieren, concatena mapas y corrige los indices
        base = partes[0]
        if all(p.nombres == base.nombres for p in partes[1:]):
            return cls(base.nombres, base.real, base.hstar, base.goal, base.start,
                       np.concatenate([p.mapa_idx for p in partes]), np.concatenate([p.visto for p in partes]),
                       np.concatenate([p.hfs for p in partes]), base.escala)
        nombres, idxs, off = [], [], 0
        for p in partes:
            nombres += p.nombres; idxs.append(p.mapa_idx + off); off += len(p.nombres)
        return cls(nombres, np.concatenate([p.real for p in partes]), np.concatenate([p.hstar for p in partes]),
                   np.concatenate([p.goal for p in partes]), np.concatenate([p.start for p in partes]),
                   np.concatenate(idxs), np.concatenate([p.visto for p in partes]), np.concatenate([p.hfs for p in partes]), base.escala)


def empaquetar(visto):
    return np.packbits(visto.ravel())


def lote(fotos, idx, rng=None, device="cpu"):
    # tensores de un lote con simetria al azar por foto
    xs, ys, ms = [], [], []
    for i in idx:
        k = int(rng.integers(8)) if rng is not None else 0
        x, y, m = fotos.muestra(int(i), k)
        xs.append(x); ys.append(y); ms.append(m)
    return (torch.from_numpy(np.stack(xs)).to(device), torch.from_numpy(np.stack(ys)).to(device),
            torch.from_numpy(np.stack(ms)).to(device))
