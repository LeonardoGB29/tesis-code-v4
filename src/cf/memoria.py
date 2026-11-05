# exp_mem: gnn de phil sobre una muestra de la frontera, memoria temporal (ninguna, gru, mamba) sobre la secuencia comun
# y residuo grueso suavizado sobre cf_filtro congelado; cf = recorte(cf_filtro * exp(R), CF_MIN, 1), R = 0 al inicio
from dataclasses import asdict, dataclass, fields

import torch
import torch.nn as nn
import torch.nn.functional as F
from mambapy.mamba import Mamba, MambaConfig
from torch_geometric.nn import GENConv

from .fotos import CF_MIN, LADO
from .red import Suavizado, cargar

N_ESTADO = 3            # open, closed, resto
N_ESCALARES = 3         # expansiones / 1000, fraccion vista, minimo h_fs de la muestra sobre la escala
MODELOS_MEM = ("mem_no", "mem_gru", "mem_mamba")


@dataclass
class MemConfig:
    modelo: str = "mem_no"
    cf_ckpt: str = ""             # best.pt de cf_filtro de la misma semilla, congelado
    sigma: float = 16.0
    d_rasgos: int = 32
    d_emb: int = 64
    f_width: int = 128
    mlp_width: int = 128
    d_mem: int = 64
    d_state: int = 16
    d_conv: int = 4
    n_muestra: int = 128
    bloque: int = 8
    every_cf: int = 20
    every_fs: int = 16
    coord_scale: float = 200.0

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


def entrada_nodos(rc, goal, rasgos, estado, escalares, escala):
    # rc [n, 2] float, goal [n, 2] float, rasgos [n, d], estado [n] long, escalares [n, 3] -> [n, 6 + d + 3 + 3]
    x, gb = rc / escala, goal / escala
    d_euc = (x - gb).norm(dim=-1, keepdim=True)
    d_cos = 1.0 - F.cosine_similarity(x, gb, dim=-1).unsqueeze(-1)
    return torch.cat([x, gb, d_euc, d_cos, rasgos, F.one_hot(estado, N_ESTADO).to(rasgos.dtype), escalares], dim=-1)


class GnnMem(nn.Module):
    # f de phil y genconv de un salto, la misma de cf_global_gnn
    def __init__(self, cfg):
        super().__init__()
        d_in = 6 + cfg.d_rasgos + N_ESTADO + N_ESCALARES
        self.f = nn.Sequential(nn.Linear(d_in, cfg.f_width), nn.LeakyReLU(), nn.Linear(cfg.f_width, cfg.f_width), nn.LeakyReLU(),
                               nn.Linear(cfg.f_width, cfg.d_emb), nn.LeakyReLU())
        self.conv = GENConv(cfg.d_emb, cfg.d_emb, aggr="softmax", learn_t=True, num_layers=2, norm="layer", msg_norm=True, learn_msg_scale=True)

    def forward(self, x, ei):
        return self.conv(self.f(x), ei)


class SinMemoria(nn.Module):
    d_out = 0

    def secuencia(self, u):
        return None

    def inicio(self, device="cpu"):
        return None

    def paso(self, u, estado):
        return None, None


class MemoriaGRU(nn.Module):
    # gru sobre la secuencia comun u_t, z_0 = 0
    def __init__(self, d_in, d_mem):
        super().__init__()
        self.gru = nn.GRU(d_in, d_mem, batch_first=True)
        self.d_out = d_mem

    def secuencia(self, u):
        # u [B, L, d_in] -> z [B, L, d_mem]; causal, el relleno al final no afecta a lo anterior
        return self.gru(u)[0]

    def inicio(self, device="cpu"):
        return torch.zeros(1, 1, self.d_out, device=device)

    def paso(self, u, h):
        out, h = self.gru(u.unsqueeze(1), h)
        return out[:, 0], h


class MemoriaMamba(nn.Module):
    # bloque de mamba (mamba.py, pytorch puro) sobre la misma secuencia, con su propio estado y cache de convolucion
    def __init__(self, d_in, d_mem, d_state, d_conv):
        super().__init__()
        self.proy = nn.Identity() if d_in == d_mem else nn.Linear(d_in, d_mem)
        self.cfg = MambaConfig(d_model=d_mem, n_layers=1, d_state=d_state, d_conv=d_conv, expand_factor=2)
        self.mamba = Mamba(self.cfg)
        self.d_out = d_mem

    def secuencia(self, u):
        return self.mamba(self.proy(u))

    def inicio(self, device="cpu"):
        return [(None, torch.zeros(1, self.cfg.d_inner, self.cfg.d_conv - 1, device=device))]

    def paso(self, u, caches):
        return self.mamba.step(self.proy(u), caches)


def hacer_memoria(cfg):
    if cfg.modelo == "mem_no":
        return SinMemoria()
    if cfg.modelo == "mem_gru":
        return MemoriaGRU(cfg.d_emb, cfg.d_mem)
    if cfg.modelo == "mem_mamba":
        return MemoriaMamba(cfg.d_emb, cfg.d_mem, cfg.d_state, cfg.d_conv)
    raise ValueError(cfg.modelo)


class CampoR(nn.Module):
    # bloques [B, lb, lb] -> campo [B, LADO, LADO] con interpolacion bilineal y el suavizado de cf_filtro
    # los dos son lineales, asi que es una matriz fija; bloques en cero dan campo en cero exacto
    def __init__(self, sigma, bloque):
        super().__init__()
        lb = LADO // bloque
        self.lb = lb
        suave = Suavizado(sigma)
        ojo = torch.eye(lb * lb).view(-1, 1, lb, lb)
        filas = []
        with torch.no_grad():
            for k in range(0, lb * lb, 52):
                e = ojo[k:k + 52]
                up = F.interpolate(e, size=(LADO, LADO), mode="bilinear", align_corners=False).squeeze(1)
                filas.append(suave(up).reshape(e.size(0), -1))
        self.register_buffer("M", torch.cat(filas))                 # [lb*lb, LADO*LADO]

    def forward(self, bloques):
        B = bloques.size(0)
        return (bloques.reshape(B, -1) @ self.M).view(B, LADO, LADO)


def bloques_de(r, celdas, foto, n_fotos, bloque, lb):
    # r [n] residuos de nodos de la frontera, celdas [n, 2] long, foto [n] long -> media por bloque [n_fotos, lb, lb], bloques sin nodos en 0
    idx = foto * lb * lb + (celdas[:, 0] // bloque) * lb + celdas[:, 1] // bloque
    suma = torch.zeros(n_fotos * lb * lb, device=r.device, dtype=r.dtype).index_add_(0, idx, r)
    cnt = torch.zeros(n_fotos * lb * lb, device=r.device, dtype=r.dtype).index_add_(0, idx, torch.ones_like(r))
    return (suma / cnt.clamp(min=1)).view(n_fotos, lb, lb)


class ResidualMem(nn.Module):
    def __init__(self, cfg, unet=None):
        super().__init__()
        self.cfg = cfg
        if unet is None:
            unet = cargar(cfg.cf_ckpt, "cpu")[0]
        self.unet = unet
        for p in self.unet.parameters():
            p.requires_grad_(False)
        self.unet.eval()
        self.gnn = GnnMem(cfg)
        self.memoria = hacer_memoria(cfg)
        self.cabeza = nn.Sequential(nn.Linear(cfg.d_emb + self.memoria.d_out + 2, cfg.mlp_width), nn.LeakyReLU(),
                                    nn.Linear(cfg.mlp_width, cfg.mlp_width), nn.LeakyReLU(), nn.Linear(cfg.mlp_width, 1))
        nn.init.zeros_(self.cabeza[-1].weight)
        nn.init.zeros_(self.cabeza[-1].bias)
        self.campo = CampoR(cfg.sigma, cfg.bloque)

    def train(self, mode=True):
        super().train(mode)
        self.unet.eval()                                             # congelada, sin dropout ni nada que cambie
        return self

    def entrenables(self):
        return [p for n, p in self.named_parameters() if not n.startswith("unet.")]

    def partes(self):
        return [self.gnn, self.memoria, self.cabeza, self.campo]

    def resto_a(self, device):
        for m in self.partes():
            m.to(device)
        return self

    @staticmethod
    def cf_de(cf0, R):
        # el mismo recorte en la perdida y en la busqueda
        return (cf0 * torch.exp(R)).clamp(CF_MIN, 1.0)

    def residuos(self, g_front, z_front, goal_front):
        gb = goal_front / self.cfg.coord_scale
        ent = [g_front, z_front, gb] if z_front is not None else [g_front, gb]
        return self.cabeza(torch.cat(ent, dim=-1)).squeeze(-1)

    def campo_lote(self, lote):
        # lote: x_nodos [n, d_in], ei [2, m], es_front [n] bool, foto [n] long, celdas [n, 2] long, goal_foto [P, 2], seq [P] long, paso [P] long, B, L
        g = self.gnn(lote["x_nodos"], lote["ei"])
        front = lote["es_front"]
        gf, fi = g[front], lote["foto"][front]
        P = lote["goal_foto"].size(0)
        z_f = None
        if self.memoria.d_out:
            u = torch.zeros(P, g.size(1), device=g.device, dtype=g.dtype).index_add_(0, fi, gf)
            cnt = torch.zeros(P, device=g.device, dtype=g.dtype).index_add_(0, fi, torch.ones(fi.size(0), device=g.device, dtype=g.dtype))
            u = u / cnt.clamp(min=1).unsqueeze(1)
            U = torch.zeros(lote["B"], lote["L"], g.size(1), device=g.device, dtype=g.dtype)
            U[lote["seq"], lote["paso"]] = u
            Z = self.memoria.secuencia(U)
            z_f = Z[lote["seq"][fi], lote["paso"][fi]]
        r = self.residuos(gf, z_f, lote["goal_foto"][fi])
        bloques = bloques_de(r, lote["celdas"][front], fi, P, self.cfg.bloque, self.campo.lb)
        return self.campo(bloques)

    @torch.no_grad()
    def llamada(self, f, ent, estado, dev_mem):
        # una llamada de la busqueda: f [d, LADO, LADO] rasgos de la u-net en su dispositivo, ent de datos_mem.armar_llamada
        nodos = ent["nodos"]
        R = torch.zeros(LADO, LADO, device=dev_mem)
        if len(nodos) == 0:
            u = torch.zeros(1, self.cfg.d_emb, device=dev_mem)
            z, estado = self.memoria.paso(u, estado)
            return R, estado
        nt = torch.from_numpy(nodos.astype("int64"))
        ras = f[:, nt[:, 0].to(f.device), nt[:, 1].to(f.device)].t().to(dev_mem)
        rc = nt.float().to(dev_mem)
        goal = torch.tensor(ent["goal"], dtype=torch.float, device=dev_mem).unsqueeze(0).expand(rc.size(0), -1)
        estado_n = torch.from_numpy(ent["estado"].astype("int64")).to(dev_mem)
        esc = torch.tensor(ent["escalares"], dtype=torch.float, device=dev_mem).unsqueeze(0).expand(rc.size(0), -1)
        x = entrada_nodos(rc, goal, ras, estado_n, esc, self.cfg.coord_scale)
        ei = torch.from_numpy(ent["aristas"].astype("int64")).to(dev_mem)
        g = self.gnn(x, ei)
        nf = ent["n_front"]
        gf = g[:nf]
        z_f = None
        if self.memoria.d_out:
            u = gf.mean(dim=0, keepdim=True)
            z, estado = self.memoria.paso(u, estado)
            z_f = z.expand(nf, -1)
        r = self.residuos(gf, z_f, goal[:nf])
        bloques = bloques_de(r, rc[:nf].long(), torch.zeros(nf, dtype=torch.long, device=dev_mem), 1, self.cfg.bloque, self.campo.lb)
        return self.campo(bloques)[0], estado


def construir_mem(cfg, unet=None):
    return ResidualMem(cfg, unet)


def guardar_mem(path, model, epoca, val, extra=None):
    torch.save({"epoca": epoca, "model_state": model.state_dict(), "config": model.cfg.to_dict(), "val": val, "extra": extra or {}}, path)


def cargar_mem(path, device="cpu", dev_mem="cpu"):
    raw = torch.load(path, map_location="cpu", weights_only=False)
    cfg = MemConfig.from_dict(raw["config"])
    model = construir_mem(cfg, cargar(cfg.cf_ckpt, "cpu")[0])
    model.load_state_dict(raw["model_state"])
    model.unet.to(device)
    model.resto_a(dev_mem).eval()
    return model, cfg, raw
