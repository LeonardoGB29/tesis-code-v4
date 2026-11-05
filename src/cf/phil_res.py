# exp_mem: phil_res_cf, PHIL tal cual (phil_res_s) con la base h_fs / cf_filtro en vez de h_fs
# entrada: dos columnas, base y h_fs sobre la escala; salida: max(h_fs, base + delta), delta con signo y ultima capa en cero
# la base se calcula en float64 como en EvaluadorCF y la salida sale en float64, asi la iteracion 0 es cf_filtro exacto
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from src.mapa_oculto.model import FsConfig, PhilFS
from src.phil.model import PHIL

from .busqueda import EvaluadorCF
from .red import cargar


@dataclass
class ResCfConfig(FsConfig):
    d_enc: int = 2
    cf_ckpt: str = ""          # best.pt de cf_filtro de la misma semilla
    every_cf: int = 20
    residual_cf: bool = True


class PhilResCF(PhilFS):
    def __init__(self, cfg, unet=None, device="cpu"):
        super().__init__(cfg)
        nn.init.zeros_(self.heuristic.mlp[-1].weight)
        nn.init.zeros_(self.heuristic.mlp[-1].bias)
        self.unet = unet if unet is not None else cargar(cfg.cf_ckpt, device)[0]
        for p in self.unet.parameters():
            p.requires_grad_(False)
        self.unet.eval()
        self.device_unet = device
        self.ev = None
        self.escala = float(cfg.coord_scale)

    def train(self, mode=True):
        super().train(mode)
        self.unet.eval()
        return self

    def parameters_phil(self):
        return [p for n, p in self.named_parameters() if not n.startswith("unet.")]

    def state_dict_phil(self):
        return {k: v for k, v in self.state_dict().items() if not k.startswith("unet.")}

    def begin_search(self, data, x_goal):
        # el evaluador se crea al primer uso: en el replay del entrenamiento no hace falta
        self._data, self._goal_t = data, x_goal
        self.goal = (int(x_goal[0]), int(x_goal[1]))
        self._rc = data.x.long().numpy()
        self._replay = None
        self.ev = None

    def evaluador(self):
        if self.ev is None:
            v_g = int(np.flatnonzero((self._rc[:, 0] == self.goal[0]) & (self._rc[:, 1] == self.goal[1]))[0])
            self.ev = EvaluadorCF(self._data, v_g, self.unet, self.cfg.every_cf, self.cfg.fs_every, self.escala, self.device_unet)
        return self.ev

    def observe(self, data, v):
        self.evaluador().observar(int(v))

    def extra(self, rc):
        # columnas float64 [n, 2]: base = h_fs / cf_filtro y h_fs, las dos sobre la escala
        ev = self.evaluador()
        ev.actualizar()
        hfs = ev.fs.h[rc[:, 0], rc[:, 1]].astype(np.float64)
        cf = ev.mapa_cf[rc[:, 0], rc[:, 1]].astype(np.float64)
        return np.stack([hfs / cf, hfs], axis=1) / self.escala

    def node_input(self, x, x_g):
        if self._replay is not None:
            e = self._replay
        else:
            e = torch.from_numpy(self.extra(x.long().numpy()))
            self._last = e
        return torch.cat([PHIL.node_input(self, x, x_g), e.float()], dim=-1)

    def forward(self, x_fringe, x_neigh, edge_index_local, x_goal, z_t, edge_attr=None):
        delta, z = super().forward(x_fringe, x_neigh, edge_index_local, x_goal, z_t, edge_attr)
        e = self._replay if self._replay is not None else self._last
        n = x_fringe.size(0)
        return torch.maximum(e[:n, 1], e[:n, 0] + delta.double()), z

    @property
    def secs_red(self):
        return self.ev.secs_red if self.ev is not None else 0.0

    @property
    def n_red(self):
        return self.ev.n_red if self.ev is not None else 0
