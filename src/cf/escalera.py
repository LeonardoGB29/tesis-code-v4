# exp_escalera. peldanos entre PHIL y la cnn de cf_filtro
#   R0 PHIL original, la clase PHIL de src/phil, sin base y sin cnn
#   R1 PHIL con las columnas de la cnn de entrada y salida directa, como PHIL original
#   R2 R3 R4 correccion acotada sobre la base b = h_fs / cf de cf_filtro, con f, gnn, gru y mlp, sin gru, y sin gnn ni gru
#   C compuerta entre la heuristica de phil_res_s congelado y la base, alpha arranca en 0 exacto
# la u-net de cf_filtro va congelada y entrega la base y los 32 rasgos del decodificador en la celda de cada nodo
# la base se calcula en float64 como en phil_res_cf, asi la iteracion 0 de R2 R3 R4 y C es cf_filtro exacto
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.mapa_oculto.model import FsConfig, PhilFS
from src.phil.config import PhilConfig
from src.phil.model import PHIL

from .busqueda import EvaluadorCF
from .red import cargar

PELDANOS = ("R0", "R1", "R2", "R3", "R4", "C")
D_RASGOS = 32
OFFS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
D_CNN = 2 + 1 + D_RASGOS + len(OFFS)        # b y h_fs sobre la escala, cf, rasgos estandarizados, b(vecino) - b(v)


@dataclass
class EscaleraConfig(FsConfig):
    peldano: str = "R2"
    d_enc: int = D_CNN
    cf_ckpt: str = ""              # best.pt de cf_filtro de la misma semilla
    phil_ckpt: str = ""            # C, best.pt de phil_res_s de la misma semilla
    every_cf: int = 20
    tope_delta: float = 1.0        # R2 R3 R4, delta = tope * tanh(cabeza), en unidades de la escala
    rasgos_media: list = field(default_factory=list)
    rasgos_std: list = field(default_factory=list)


class EvaluadorCFRasgos(EvaluadorCF):
    # cf_filtro y los rasgos de la u-net en la misma pasada, el cf es el mismo que el de EvaluadorCF
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.mapa_rasgos = None

    @torch.no_grad()
    def calcular(self):
        t0 = time.time()
        cf, f = self.model.cf_y_rasgos(self.entrada())
        self.mapa_rasgos = f[0].cpu().numpy()
        out = cf[0].cpu().numpy()
        self.secs_red += time.time() - t0
        self.n_red += 1
        return out


def estadisticas_rasgos(unet, fotos, device, n=200, lote=16, seed=0):
    # media y desvio por canal de los rasgos de la u-net sobre las celdas del mapa, en n fotos de entrenamiento
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(fotos), size=min(n, len(fotos)), replace=False)
    H, W = fotos.real.shape[1:]
    s1 = torch.zeros(D_RASGOS, dtype=torch.float64)
    s2 = torch.zeros(D_RASGOS, dtype=torch.float64)
    cnt = 0
    with torch.no_grad():
        for k in range(0, len(idx), lote):
            xs = np.stack([fotos.muestra(int(i))[0] for i in idx[k:k + lote]])
            _, f = unet.cf_y_rasgos(torch.from_numpy(xs).to(device))
            f = f[:, :, :H, :W].double()
            s1 += f.sum(dim=(0, 2, 3)).cpu()
            s2 += (f ** 2).sum(dim=(0, 2, 3)).cpu()
            cnt += f.shape[0] * H * W
    media = s1 / cnt
    std = (s2 / cnt - media ** 2).clamp(min=0).sqrt().clamp(min=1e-3)
    return media.float().tolist(), std.float().tolist()


def columnas(ev, rc, media, std, escala):
    # rc [n, 2] enteros. devuelve la base exacta en float64, b y h_fs sobre la escala con inf donde no se llega,
    # y las entradas en float32, b y h_fs recortadas al tamano del mapa, cf, rasgos estandarizados y b(vecino) - b(v) de los 8 vecinos
    ev.actualizar()
    hmap = ev.fs.h
    H, W = hmap.shape
    tope = float(hmap.size)
    r, c = rc[:, 0], rc[:, 1]
    hfs = hmap[r, c].astype(np.float64)
    cf = ev.mapa_cf[r, c].astype(np.float64)
    base = np.stack([hfs / cf, hfs], axis=1) / escala
    # el mapa de b recortado y su relleno se rehacen solo cuando cambia el cf o el h_fs, no en cada llamada
    clave = (ev.n_red, ev.fs.n_updates)        # llamadas a la u-net y recalculos de h_fs, no id() porque se reutilizan direcciones
    cache = getattr(ev, "_bmap", None)
    if cache is None or cache[0] != clave:
        hfin = np.where(np.isfinite(hmap), hmap, tope)
        bmap = np.minimum(hfin / ev.mapa_cf[:H, :W].astype(np.float64), tope)
        pad = np.full((H + 2, W + 2), tope)
        pad[1:-1, 1:-1] = bmap
        ev._bmap = (clave, bmap, pad)
    _, bmap, pad = ev._bmap
    bv = bmap[r, c]
    difs = np.stack([pad[r + 1 + dr, c + 1 + dc] - bv for dr, dc in OFFS], axis=1) / escala
    ras = (ev.mapa_rasgos[:, r, c].T - media) / std
    ent = np.concatenate([np.stack([bv, np.minimum(hfs, tope)], axis=1) / escala, cf[:, None], ras, difs], axis=1).astype(np.float32)
    return base, ent


class _ConUNet:
    # lo comun a los peldanos con cnn, el evaluador de cf_filtro con rasgos y las columnas por nodo

    def _preparar_unet(self, cfg, unet, device):
        self.unet = unet if unet is not None else cargar(cfg.cf_ckpt, device)[0]
        for p in self.unet.parameters():
            p.requires_grad_(False)
        self.unet.eval()
        self.device_unet = device
        self.escala = float(cfg.coord_scale)
        self.media = np.asarray(cfg.rasgos_media or [0.0] * D_RASGOS, dtype=np.float32)
        self.std = np.asarray(cfg.rasgos_std or [1.0] * D_RASGOS, dtype=np.float32)
        self.ev = None
        self._last = self._replay = None

    def begin_search(self, data, x_goal):
        self._data, self._goal_t = data, x_goal
        self.goal = (int(x_goal[0]), int(x_goal[1]))
        self._rc = data.x.long().numpy()
        self._replay = None
        self.ev = None

    def evaluador(self):
        if self.ev is None:
            v_g = int(np.flatnonzero((self._rc[:, 0] == self.goal[0]) & (self._rc[:, 1] == self.goal[1]))[0])
            self.ev = EvaluadorCFRasgos(self._data, v_g, self.unet, self.cfg.every_cf, self.cfg.fs_every, self.escala, self.device_unet)
        return self.ev

    def observe(self, data, v):
        self.evaluador().observar(int(v))

    def extra(self, rc):
        base, ent = columnas(self.evaluador(), rc, self.media, self.std, self.escala)
        return torch.from_numpy(base), torch.from_numpy(ent)

    def columnas_de(self, x):
        # las columnas del paso, guardadas al recolectar y reproducidas al entrenar
        if self._replay is not None:
            return self._replay
        cols = self.extra(x.long().numpy())
        self._last = cols
        return cols

    def step_state(self):
        return self._last

    def set_step_state(self, state):
        self._replay = state

    def parameters_phil(self):
        return [p for n, p in self.named_parameters() if not n.startswith("unet.") and not n.startswith("phil.")]

    def state_dict_phil(self):
        return {k: v for k, v in self.state_dict().items() if not k.startswith("unet.") and not k.startswith("phil.")}

    @property
    def secs_red(self):
        return self.ev.secs_red if self.ev is not None else 0.0

    @property
    def n_red(self):
        return self.ev.n_red if self.ev is not None else 0


class Escalera(_ConUNet, PhilFS):
    # R1 a R4. PHIL con las columnas de la cnn de entrada, salida directa en R1 y correccion acotada sobre la base en R2 R3 R4
    def __init__(self, cfg, unet=None, device="cpu"):
        PhilFS.__init__(self, cfg)
        assert cfg.peldano in ("R1", "R2", "R3", "R4"), cfg.peldano
        self.peldano = cfg.peldano
        self._preparar_unet(cfg, unet, device)
        if self.peldano != "R1":
            nn.init.zeros_(self.heuristic.mlp[-1].weight)
            nn.init.zeros_(self.heuristic.mlp[-1].bias)

    def train(self, mode=True):
        super().train(mode)
        self.unet.eval()
        return self

    def modulos_usados(self):
        usados = [self.f, self.heuristic]
        if self.peldano != "R4":
            usados.append(self.gnn)
        if self.peldano in ("R1", "R2"):
            usados.append(self.memory)
        return usados

    def n_entrenables(self):
        return sum(p.numel() for m in self.modulos_usados() for p in m.parameters())

    def node_input(self, x, x_g):
        _, ent = self.columnas_de(x)
        return torch.cat([PHIL.node_input(self, x, x_g), ent], dim=-1)

    def forward(self, x_fringe, x_neigh, edge_index_local, x_goal, z_t, edge_attr=None):
        n = x_fringe.size(0)
        self._n_fringe = n
        x_all = torch.cat([x_fringe, x_neigh], dim=0)
        x_proj = self.project(x_all, x_goal)
        if self.peldano == "R4":
            g_prime, z_t1 = x_proj[:n], z_t
        else:
            g = self.gnn(x_proj, edge_index_local, edge_attr)[:n]
            if self.peldano == "R3":
                g_prime, z_t1 = g, z_t
            else:
                g_prime, z_t1 = self.memory(g, z_t)
        salida = self.heuristic(g_prime, x_goal / self.coord_scale)
        if self.peldano == "R1":
            return salida, z_t1
        base, _ = self._replay if self._replay is not None else self._last
        delta = self.cfg.tope_delta * torch.tanh(salida)
        return torch.maximum(base[:n, 1], base[:n, 0] + delta.double()), z_t1


class RecorteRecto(torch.autograd.Function):
    # recorte a [0, 1] con gradiente derecho, asi alpha es 0 exacto al inicio y la compuerta puede volver si z baja de 0
    @staticmethod
    def forward(ctx, z):
        return z.clamp(0.0, 1.0)

    @staticmethod
    def backward(ctx, g):
        return g


class Compuerta(_ConUNet, nn.Module):
    # C. h = alpha * h_phil + (1 - alpha) * b, con phil_res_s congelado y la base de cf_filtro. solo la compuerta entrena
    def __init__(self, cfg, phil, unet=None, device="cpu"):
        nn.Module.__init__(self)
        assert cfg.peldano == "C"
        self.cfg = cfg
        self.peldano = "C"
        self.coord_scale = cfg.coord_scale
        self.phil = phil
        for p in self.phil.parameters():
            p.requires_grad_(False)
        self.phil.eval()
        self._preparar_unet(cfg, unet, device)
        d_in = cfg.d_hidden + D_RASGOS + 3
        self.compuerta = nn.Sequential(nn.Linear(d_in, 128), nn.LeakyReLU(), nn.Linear(128, 128), nn.LeakyReLU(), nn.Linear(128, 1))
        nn.init.zeros_(self.compuerta[-1].weight)
        nn.init.zeros_(self.compuerta[-1].bias)
        self.alphas = None                                          # lista de (rc, alpha) si se quiere grabar

    def train(self, mode=True):
        super().train(mode)
        self.unet.eval()
        self.phil.eval()
        return self

    @property
    def memory(self):
        # z_t de la busqueda es el de phil_res_s. propiedad y no atributo, asi no se registra dos veces en el state_dict
        return self.phil.memory

    def n_entrenables(self):
        return sum(p.numel() for p in self.compuerta.parameters())

    def parameters_phil(self):
        return list(self.compuerta.parameters())

    def state_dict_phil(self):
        return {k: v for k, v in self.state_dict().items() if k.startswith("compuerta.")}

    def forward(self, x_fringe, x_neigh, edge_index_local, x_goal, z_t, edge_attr=None):
        n = x_fringe.size(0)
        x_all = torch.cat([x_fringe, x_neigh], dim=0)
        base, ent = self.columnas_de(x_all)
        hfs_col = ent[:, 1:2]                                       # h_fs sobre la escala, recortada, como PhilFS.extra
        with torch.no_grad():
            x_proj = self.phil.f(torch.cat([PHIL.node_input(self.phil, x_all, x_goal), hfs_col], dim=-1))
            g = self.phil.gnn(x_proj, edge_index_local, edge_attr)[:n]
            g_prime, z_t1 = self.phil.memory(g, z_t)
            h_phil = hfs_col[:n, 0] + F.softplus(self.phil.heuristic(g_prime, x_goal / self.coord_scale))
        entrada = torch.cat([g_prime, ent[:n, 3:3 + D_RASGOS], ent[:n, 2:3], ent[:n, 1:2], ent[:n, 0:1]], dim=-1)
        alpha = RecorteRecto.apply(self.compuerta(entrada).squeeze(-1)).double()
        if self.alphas is not None:
            self.alphas.append((x_fringe.long().numpy().copy(), alpha.detach().numpy().copy(), base[:n, 1].numpy().copy()))
        return alpha * h_phil.double() + (1.0 - alpha) * base[:n, 0], z_t1


def construir_desde(raw, device="cpu"):
    # arma el peldano guardado en un checkpoint, la u-net y phil_res_s se recargan desde sus rutas
    c = raw["config"]
    peldano = c.get("peldano") or raw.get("extra", {}).get("peldano")
    if peldano == "R0":
        m = PHIL(PhilConfig.from_dict(c))
        m.load_state_dict(raw["model_state"])
        return m.eval(), m.cfg
    cfg = EscaleraConfig.from_dict(c)
    if peldano == "C":
        from experiments.common import load_phil
        phil = load_phil(cfg.phil_ckpt)[0]
        m = Compuerta(cfg, phil, device=device)
    else:
        m = Escalera(cfg, device=device)
    faltan, sobran = m.load_state_dict(raw["model_state"], strict=False)
    propios = [k for k in faltan if not k.startswith("unet.") and not k.startswith("phil.")]
    assert not propios and not sobran, (propios, sobran)
    return m.eval(), cfg
