# redes del factor de correccion: u-net con autoatencion en el cuello, cnn local y gnn de un salto
# exp_cf2: salida filtro (gauss fijo despues de la cabeza) o gruesa (cabeza a 26x26 e interpolacion), la escala gruesa dentro del modelo
from dataclasses import asdict, dataclass, fields

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GENConv

from .fotos import CANALES, CF_MIN, LADO


@dataclass
class CfConfig:
    tipo: str = "global"          # global, local, gnn
    ancho: int = 32               # canales del primer nivel
    niveles: int = 4              # bajadas de la u-net
    capas_atencion: int = 2
    cabezas: int = 4
    d_rasgos: int = 32            # rasgos por celda que salen de la u-net
    capas_local: int = 5          # 5 convs 3x3: campo receptivo 11x11
    coord_scale: float = 200.0
    d_emb: int = 64               # gnn
    f_width: int = 128
    mlp_width: int = 128
    n_neighbors: int = 8
    every_cf: int = 20            # cf se recalcula cada tantas expansiones
    every_fs: int = 16            # h_fs igual que la voraz con h_fs
    congelar_unet: bool = True    # gnn: la u-net viene de cf_global y no se entrena
    salida: str = "fina"          # fina, filtro (gauss fijo despues de la cabeza) o gruesa (cabeza en baja resolucion e interpolacion)
    sigma: float = 0.0            # filtro: desvio del gauss en celdas
    nivel_grueso: int = 3         # gruesa: nivel del decodificador donde va la cabeza, lado LADO / 2**nivel (26 con 3)
    canales_entrada: int = CANALES  # exp_geo usa 3 canales sin meta ni h_fs, los checkpoints viejos quedan en 5

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


def bloque(c_in, c_out):
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1), nn.GroupNorm(8, c_out), nn.LeakyReLU(),
                         nn.Conv2d(c_out, c_out, 3, padding=1), nn.GroupNorm(8, c_out), nn.LeakyReLU())


class Atencion(nn.Module):
    # bloque transformer pre-norm sobre los tokens del cuello
    def __init__(self, d, cabezas):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, cabezas, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, t):
        h = self.n1(t)
        t = t + self.att(h, h, h, need_weights=False)[0]
        return t + self.mlp(self.n2(t))


class Cabeza(nn.Module):
    # cf = sigmoide recortada; arranca en cf = 0.5 constante, asi la epoca 0 ordena igual que h_fs y el gradiente no esta saturado
    def __init__(self, c_in):
        super().__init__()
        self.conv = nn.Conv2d(c_in, 1, 1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, f):
        return torch.sigmoid(self.conv(f)).clamp(CF_MIN, 1.0).squeeze(1)


class Suavizado(nn.Module):
    # gauss fijo separable con relleno por borde, el mismo que gaussian_filter de scipy con mode nearest
    # se filtra cf - c con c la media de la foto, asi una constante sale exacta y la epoca 0 sigue siendo h_fs
    def __init__(self, sigma, truncar=4.0):
        super().__init__()
        r = int(truncar * sigma + 0.5)
        x = torch.arange(-r, r + 1, dtype=torch.float64)
        k = torch.exp(-0.5 * (x / sigma) ** 2)
        k = (k / k.sum()).float()
        self.r = r
        self.register_buffer("kh", k.view(1, 1, 1, -1))
        self.register_buffer("kv", k.view(1, 1, -1, 1))

    def forward(self, cf):
        # cf [B, H, W] -> [B, H, W]
        B, H, W = cf.shape
        c = cf.mean(dim=(1, 2), keepdim=True)
        x = (cf - c).unsqueeze(1)
        ih = torch.arange(-self.r, H + self.r, device=cf.device).clamp(0, H - 1)
        iw = torch.arange(-self.r, W + self.r, device=cf.device).clamp(0, W - 1)
        x = x[:, :, ih][:, :, :, iw]                                        # relleno por borde con indices, igual en cpu y mps
        x = F.conv2d(F.conv2d(x, self.kh), self.kv)
        return x.squeeze(1) + c


class CabezaGruesa(nn.Module):
    # cabeza conv 1x1 en cero sobre los rasgos de baja resolucion, sigmoide e interpolacion bilineal al lado de la foto
    # interpola cf - c con c la media, asi una constante sale exacta
    def __init__(self, c_in):
        super().__init__()
        self.conv = nn.Conv2d(c_in, 1, 1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, f):
        cf = torch.sigmoid(self.conv(f)).clamp(CF_MIN, 1.0)
        c = cf.mean(dim=(2, 3), keepdim=True)
        cf = c + F.interpolate(cf - c, size=(LADO, LADO), mode="bilinear", align_corners=False)
        return cf.clamp(CF_MIN, 1.0).squeeze(1)


class UNetCF(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        w = [cfg.ancho * 2 ** i for i in range(cfg.niveles + 1)]       # 32 64 128 256 512 con 4 niveles
        self.enc = nn.ModuleList([bloque(cfg.canales_entrada if i == 0 else w[i - 1], w[i]) for i in range(cfg.niveles + 1)])
        self.pool = nn.MaxPool2d(2)
        lado = LADO // 2 ** cfg.niveles
        self.pos = nn.Parameter(torch.zeros(1, lado * lado, w[-1]))
        nn.init.normal_(self.pos, std=0.02)
        self.att = nn.ModuleList([Atencion(w[-1], cfg.cabezas) for _ in range(cfg.capas_atencion)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(w[i + 1], w[i], 2, stride=2) for i in reversed(range(cfg.niveles))])
        self.dec = nn.ModuleList([bloque(2 * w[i], w[i]) for i in reversed(range(cfg.niveles))])
        self.rasgos = nn.Conv2d(w[0], cfg.d_rasgos, 1)
        self.cabeza = Cabeza(cfg.d_rasgos)
        if cfg.salida == "filtro":
            self.suave = Suavizado(cfg.sigma)
        elif cfg.salida == "gruesa":
            self.cabeza_gruesa = CabezaGruesa(w[cfg.nivel_grueso])

    def rasgos_y_grueso(self, x):
        # x [B, 5, LADO, LADO] -> rasgos [B, d_rasgos, LADO, LADO] y la salida del decodificador en el nivel grueso
        saltos = []
        for i, e in enumerate(self.enc):
            x = e(x if i == 0 else self.pool(x))
            saltos.append(x)
        B, C, Hc, Wc = x.shape
        t = x.flatten(2).transpose(1, 2) + self.pos
        for a in self.att:
            t = a(t)
        x = t.transpose(1, 2).reshape(B, C, Hc, Wc)
        grueso = None
        for up, dec, s in zip(self.up, self.dec, reversed(saltos[:-1])):
            x = dec(torch.cat([up(x), s], dim=1))
            if x.shape[-1] == LADO // 2 ** self.cfg.nivel_grueso:
                grueso = x
        return F.leaky_relu(self.rasgos(x)), grueso

    def caracteristicas(self, x):
        return self.rasgos_y_grueso(x)[0]

    def cf_y_rasgos(self, x):
        # exp_mem: el cf que usa la busqueda y los rasgos por celda en una sola pasada
        f, g = self.rasgos_y_grueso(x)
        if self.cfg.salida == "gruesa":
            return self.cabeza_gruesa(g), f
        cf = self.cabeza(f)
        if self.cfg.salida == "filtro":
            cf = self.suave(cf).clamp(CF_MIN, 1.0)
        return cf, f

    def forward(self, x):
        return self.cf_y_rasgos(x)[0]


class LocalCF(nn.Module):
    # solo convoluciones 3x3 sin bajadas: campo receptivo 2 * capas + 1
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        capas, c = [], cfg.canales_entrada
        for _ in range(cfg.capas_local):
            capas += [nn.Conv2d(c, cfg.ancho, 3, padding=1), nn.GroupNorm(8, cfg.ancho), nn.LeakyReLU()]
            c = cfg.ancho
        self.net = nn.Sequential(*capas)
        self.rasgos = nn.Conv2d(c, cfg.d_rasgos, 1)
        self.cabeza = Cabeza(cfg.d_rasgos)

    def caracteristicas(self, x):
        return F.leaky_relu(self.rasgos(self.net(x)))

    def forward(self, x):
        return self.cabeza(self.caracteristicas(x))


class GnnCF(nn.Module):
    # la gnn de un salto de phil (f + genconv + mlp) sin gru, con los rasgos de la u-net en la celda del nodo
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.unet = UNetCF(cfg)
        d_in = 2 + 2 + 2 + cfg.d_rasgos                  # coords, meta, d_euc, d_cos, rasgos
        self.f = nn.Sequential(nn.Linear(d_in, cfg.f_width), nn.LeakyReLU(), nn.Linear(cfg.f_width, cfg.f_width), nn.LeakyReLU(),
                               nn.Linear(cfg.f_width, cfg.d_emb), nn.LeakyReLU())
        self.conv = GENConv(cfg.d_emb, cfg.d_emb, aggr="softmax", learn_t=True, num_layers=2, norm="layer",
                            msg_norm=True, learn_msg_scale=True)
        self.mlp = nn.Sequential(nn.Linear(cfg.d_emb + 2, cfg.mlp_width), nn.LeakyReLU(),
                                 nn.Linear(cfg.mlp_width, cfg.mlp_width), nn.LeakyReLU(), nn.Linear(cfg.mlp_width, 1))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def entrada(self, rc, goal, rasgos):
        # rc [n, 2] float (fila, col); rasgos [d, LADO, LADO]
        s = self.cfg.coord_scale
        x, g = rc / s, goal / s
        gb = g.unsqueeze(0).expand(x.size(0), -1)
        d_euc = (x - gb).norm(dim=-1, keepdim=True)
        d_cos = 1.0 - F.cosine_similarity(x, gb, dim=-1).unsqueeze(-1)
        r = rasgos[:, rc[:, 0].long(), rc[:, 1].long()].t()
        return torch.cat([x, gb, d_euc, d_cos, r], dim=-1)

    def forward(self, rc_f, rc_n, ei, goal, rasgos):
        # rc_f frontera, rc_n vecinos, ei aristas vecino -> frontera en indices locales
        z = self.f(self.entrada(torch.cat([rc_f, rc_n], dim=0), goal, rasgos))
        g = self.conv(z, ei)[:rc_f.size(0)]
        gb = (goal / self.cfg.coord_scale).unsqueeze(0).expand(g.size(0), -1)
        return torch.sigmoid(self.mlp(torch.cat([g, gb], dim=-1)).squeeze(-1)).clamp(CF_MIN, 1.0)

    def entrenables(self):
        if self.cfg.congelar_unet:
            return [p for n, p in self.named_parameters() if not n.startswith("unet.")]
        return list(self.parameters())


def construir(cfg):
    return {"global": UNetCF, "local": LocalCF, "gnn": GnnCF}[cfg.tipo](cfg)


def guardar(path, model, epoca, val, extra=None):
    torch.save({"epoca": epoca, "model_state": model.state_dict(), "config": model.cfg.to_dict(), "val": val,
                "extra": extra or {}}, path)


def cargar(path, device="cpu"):
    raw = torch.load(path, map_location="cpu", weights_only=False)
    cfg = CfConfig.from_dict(raw["config"])
    model = construir(cfg)
    model.load_state_dict(raw["model_state"])
    model.to(device).eval()
    return model, cfg, raw
