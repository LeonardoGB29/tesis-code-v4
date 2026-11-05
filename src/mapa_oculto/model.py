# phil con columnas extra que salen de lo que la busqueda ha descubierto, nunca del mapa completo
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.enc_phil.encoder import obstacle_grid
from src.phil.config import PhilConfig
from src.phil.model import PHIL

from .features import N_FEAT, local_features
from .fs import FreeSpace
from .state import PAD, Discovered


@dataclass
class FsConfig(PhilConfig):
    d_enc: int = 1             # una columna: h_fs en la escala de la etiqueta
    fs_every: int = 16         # h_fs se recalcula como maximo cada tantas expansiones


@dataclass
class ResConfig(FsConfig):
    residual: bool = True      # la salida es h_fs mas una correccion no negativa


@dataclass
class FeatConfig(PhilConfig):
    d_enc: int = N_FEAT        # caracteristicas simples de lo explorado
    feat_window: int = 2 * PAD + 1


class PhilSearch(PHIL):
    # base: lleva el mapa descubierto y guarda las columnas de cada paso para el entrenamiento
    def __init__(self, cfg):
        super().__init__(cfg)
        self.disc = None
        self._last = self._replay = None
        self._n_fringe = 0

    def begin_search(self, data, x_goal):
        self.disc = Discovered(obstacle_grid(data).numpy().astype(bool))
        self.goal = (int(x_goal[0]), int(x_goal[1]))
        self._rc = data.x.long().numpy()
        self._replay = None

    def observe(self, data, v):
        self.disc.observe(int(self._rc[v, 0]), int(self._rc[v, 1]))

    def extra(self, rc):
        raise NotImplementedError

    def forward(self, x_fringe, x_neigh, edge_index_local, x_goal, z_t, edge_attr=None):
        self._n_fringe = x_fringe.size(0)
        return super().forward(x_fringe, x_neigh, edge_index_local, x_goal, z_t, edge_attr)

    def node_input(self, x, x_g):
        if self._replay is not None:               # entrenamiento: columnas guardadas al recolectar
            e = self._replay
        else:
            e = torch.from_numpy(self.extra(x.long().numpy()))
            self._last = e
        return torch.cat([super().node_input(x, x_g), e], dim=-1)

    def step_state(self):
        return self._last

    def set_step_state(self, state):
        self._replay = state


class PhilFS(PhilSearch):
    # h_fs: distancia a la meta sobre lo descubierto, tomada cuando el nodo entra a la frontera
    def begin_search(self, data, x_goal):
        super().begin_search(data, x_goal)
        self.fs = FreeSpace(self.disc, self.goal, self.cfg.fs_every)

    def observe(self, data, v):
        super().observe(data, v)
        self.fs.update(self.disc)

    def extra(self, rc):
        h = self.fs.h[rc[:, 0], rc[:, 1]]
        h = np.where(np.isfinite(h), h, self.fs.h.size)
        return (h / self.coord_scale).astype(np.float32)[:, None]


class PhilRes(PhilFS):
    # corrige a h_fs en lugar de predecir h* completo: salida = h_fs + softplus(cabeza)
    def __init__(self, cfg):
        super().__init__(cfg)
        nn.init.zeros_(self.heuristic.mlp[-1].weight)      # correccion constante al inicio: orden de h_fs

    def forward(self, x_fringe, x_neigh, edge_index_local, x_goal, z_t, edge_attr=None):
        g, z = super().forward(x_fringe, x_neigh, edge_index_local, x_goal, z_t, edge_attr)
        e = self._replay if self._replay is not None else self._last
        return e[:x_fringe.size(0), 0] + F.softplus(g), z


class PhilFeat(PhilSearch):
    def extra(self, rc):
        f = local_features(self.disc, rc, self.goal, self.coord_scale)
        # los nodos de la frontera quedan abiertos con la profundidad del padre mas 1
        r, c = rc[:self._n_fringe, 0] + PAD, rc[:self._n_fringe, 1] + PAD
        new = self.disc.depth[r, c] < 0
        self.disc.depth[r[new], c[new]] = self.disc.cur_depth + 1
        return f
