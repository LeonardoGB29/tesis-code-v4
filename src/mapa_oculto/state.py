# mapa descubierto durante la busqueda: solo se conoce lo que revelan los nodos expandidos
import numpy as np

PAD = 3      # relleno para ventanas de 7x7


class Discovered:
    def __init__(self, obstacles):
        # obstacles: [H, W] bool con el mapa real; solo se lee alrededor de nodos expandidos
        H, W = obstacles.shape
        self.H, self.W = H, W
        P = PAD
        self.inmap = np.zeros((H + 2 * P, W + 2 * P), dtype=bool)
        self.inmap[P:P + H, P:P + W] = True
        self.true = np.ones((H + 2 * P, W + 2 * P), dtype=bool)     # fuera del mapa cuenta como pared
        self.true[P:P + H, P:P + W] = obstacles
        self.known = np.zeros_like(self.true)                       # obstaculos ya vistos
        self.expanded = np.zeros_like(self.true)
        self.depth = np.full(self.true.shape, -1, dtype=np.int32)   # profundidad en el arbol de busqueda
        self.n_expanded = 0
        self.dirty = False                                          # hay obstaculos nuevos dentro del mapa
        self.cur_depth = 0

    def observe(self, r, c):
        # expandir (r, c) revela las celdas bloqueadas vecinas
        P = PAD
        sl = (slice(r + P - 1, r + P + 2), slice(c + P - 1, c + P + 2))
        new = self.true[sl] & ~self.known[sl]
        if new.any():
            self.known[sl] |= new
            if (new & self.inmap[sl]).any():
                self.dirty = True
        self.expanded[r + P, c + P] = True
        if self.depth[r + P, c + P] < 0:
            self.depth[r + P, c + P] = 0                            # el inicio
        self.cur_depth = int(self.depth[r + P, c + P])
        self.n_expanded += 1

    def known_inmap(self):
        P = PAD
        return self.known[P:P + self.H, P:P + self.W]
