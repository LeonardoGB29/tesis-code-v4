# mapa descubierto para el factor de correccion: expandir una celda revela sus ocho vecinas
# compatible con FreeSpace de mapa_oculto (dirty, n_expanded, known_inmap)
import numpy as np


class Descubierto:
    def __init__(self, obstaculos):
        self.real = np.asarray(obstaculos, dtype=bool)
        self.H, self.W = self.real.shape
        self.visto = np.zeros_like(self.real)
        self.expandido = np.zeros_like(self.real)
        self.n_expanded = 0
        self.dirty = False                      # obstaculos nuevos desde el ultimo h_fs

    def observar(self, r, c):
        sl = (slice(max(r - 1, 0), min(r + 2, self.H)), slice(max(c - 1, 0), min(c + 2, self.W)))
        nuevo = ~self.visto[sl]
        if (nuevo & self.real[sl]).any():
            self.dirty = True
        self.visto[sl] = True
        self.expandido[r, c] = True
        self.n_expanded += 1

    def known_inmap(self):
        # obstaculos conocidos, lo que usa h_fs
        return self.visto & self.real

    def copia(self):
        d = Descubierto.__new__(Descubierto)
        d.real, d.H, d.W = self.real, self.H, self.W
        d.visto, d.expandido = self.visto.copy(), self.expandido.copy()
        d.n_expanded, d.dirty = self.n_expanded, self.dirty
        return d
