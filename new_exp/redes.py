# redes de new_exp: cnn local con el suavizado de cf_filtro y la arquitectura de transpath con la misma cabeza en cero y el mismo suavizado
#   transpath segun el repositorio AIRI-Institute/TransPath (models/autoencoder.py y modules/): tallo 5x5, 3 bajadas con bloque residual,
#   4 bloques transformer de 4 cabezas con posiciones aprendidas, 3 subidas, groupnorm de 32 grupos, silu y salida tanh llevada a (0, 1)
#   unico cambio: la ultima conv arranca en cero, asi cf = 0.5 constante y la epoca 0 es la voraz con h_fs, igual que en cf_filtro
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.cf import red
from src.cf.fotos import CANALES, CF_MIN, LADO
from src.cf.red import LocalCF, Suavizado

_construir_original = red.construir
TRANSPATH = {"ancho": 64, "niveles": 3, "capas_atencion": 4, "cabezas": 4}     # valores por defecto del autoencoder de transpath
DROP_CNN, DROP_ATT = 0.15, 0.15                                                 # cnn_dropout y attn_dropout por defecto


class SuavizadoMat(Suavizado):
    # el mismo gauss separable con relleno por borde que src.cf.red.Suavizado, escrito como M_v (cf - c) M_h^T
    # en la rtx 4070 cudnn elige para el gradiente de la conv de 1x129 (sigma 16, gaps) un algoritmo de 620 ms por paso de lote 8,
    # 18 veces todo el resto del paso. el producto de matrices da lo mismo y tarda menos de 1 ms. mismos buffers, los checkpoints son intercambiables
    def matriz(self, n, k, device):
        # M[i, j] = suma de los pesos k[t] con clamp(i + t - r, 0, n - 1) = j
        i = torch.arange(n, device=device).view(-1, 1)
        t = torch.arange(2 * self.r + 1, device=device).view(1, -1)
        j = (i + t - self.r).clamp(0, n - 1)
        M = torch.zeros(n, n, device=device, dtype=k.dtype)
        return M.scatter_add_(1, j, k.view(1, -1).expand(n, -1).contiguous())

    def forward(self, cf):
        B, H, W = cf.shape
        c = cf.mean(dim=(1, 2), keepdim=True)
        mv = self.matriz(H, self.kv.flatten(), cf.device)
        mh = self.matriz(W, self.kh.flatten(), cf.device)
        return mv @ (cf - c) @ mh.T + c


class LocalFiltro(LocalCF):
    # 5 convs 3x3 sin bajadas (campo 11x11), cabeza en cero y el gauss fijo de cf_filtro despues de la cabeza
    def __init__(self, cfg):
        super().__init__(cfg)
        self.suave = SuavizadoMat(cfg.sigma)

    def forward(self, x):
        return self.suave(self.cabeza(self.caracteristicas(x))).clamp(CF_MIN, 1.0)


def normalizar(c):
    return nn.GroupNorm(32, c, eps=1e-6, affine=True)


class BloqueRes(nn.Module):
    # resnetblock: gn, silu, conv 3x3, gn, silu, dropout, conv 3x3 y suma con la entrada
    def __init__(self, c, dropout):
        super().__init__()
        self.norm1, self.conv1 = normalizar(c), nn.Conv2d(c, c, 3, padding=1)
        self.norm2, self.drop, self.conv2 = normalizar(c), nn.Dropout(dropout), nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        return x + self.conv2(self.drop(F.silu(self.norm2(h))))


class Bajada(nn.Module):
    # relleno de un pixel a la derecha y abajo, conv 3x3 con paso 2
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, stride=2)

    def forward(self, x):
        return self.conv(F.pad(x, (0, 1, 0, 1)))


class Subida(nn.Module):
    # vecino mas cercano por 2 y conv 3x3
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class Posiciones(nn.Module):
    # posembeds: rejilla fija (fila, columna, 1 - fila, 1 - columna) en [0, 1] llevada a c canales por una lineal aprendida y sumada
    def __init__(self, c, lado):
        super().__init__()
        t = torch.linspace(0, 1, lado)
        fila, col = torch.meshgrid(t, t, indexing="ij")
        self.register_buffer("rejilla", torch.stack([fila, col, 1 - fila, 1 - col], dim=-1)[None])
        self.lin = nn.Linear(4, c)

    def forward(self, x):
        return x + self.lin(self.rejilla).permute(0, 3, 1, 2)


class Atencion(nn.Module):
    # crossattention sin contexto: q, k, v lineales sin sesgo, escala d_cabeza^-0.5, salida lineal con dropout
    def __init__(self, dim, cabezas, d_cabeza, dropout):
        super().__init__()
        inner = cabezas * d_cabeza
        self.h = cabezas
        self.q = nn.Linear(dim, inner, bias=False)
        self.k = nn.Linear(dim, inner, bias=False)
        self.v = nn.Linear(dim, inner, bias=False)
        self.out = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout))

    def forward(self, x):
        B, N, _ = x.shape
        q, k, v = (t(x).view(B, N, self.h, -1).transpose(1, 2) for t in (self.q, self.k, self.v))
        return self.out(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, -1))


class BloqueTransformer(nn.Module):
    # basictransformerblock: atencion, segunda atencion (sin contexto tambien es autoatencion) y mlp x4 con gelu, cada una con layernorm previa y residuo
    def __init__(self, dim, cabezas, d_cabeza, dropout):
        super().__init__()
        self.n1, self.n2, self.n3 = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.at1 = Atencion(dim, cabezas, d_cabeza, dropout)
        self.at2 = Atencion(dim, cabezas, d_cabeza, dropout)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.at1(self.n1(x))
        x = x + self.at2(self.n2(x))
        return x + self.ff(self.n3(x))


class TransformerEspacial(nn.Module):
    # spatialtransformer: groupnorm, conv 1x1 a los tokens, bloques, conv 1x1 de vuelta y residuo
    def __init__(self, c, cabezas, d_cabeza, profundidad, dropout):
        super().__init__()
        inner = cabezas * d_cabeza
        self.norm = normalizar(c)
        self.entrada = nn.Conv2d(c, inner, 1)
        self.bloques = nn.ModuleList([BloqueTransformer(inner, cabezas, d_cabeza, dropout) for _ in range(profundidad)])
        self.salida = nn.Conv2d(inner, c, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        t = self.entrada(self.norm(x)).flatten(2).transpose(1, 2)
        for b in self.bloques:
            t = b(t)
        return x + self.salida(t.transpose(1, 2).reshape(B, -1, H, W))


class TransPathCF(nn.Module):
    # autoencoder de transpath sobre la imagen de 5 canales del mapa descubierto; cf = (tanh + 1) / 2 recortado y suavizado
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        c, k = cfg.ancho, cfg.niveles
        lado = LADO // 2 ** k                                       # 26 con 208 y 3 bajadas
        self.tallo = nn.Conv2d(CANALES, c, 5, padding=2)
        self.bajadas = nn.ModuleList([nn.Sequential(BloqueRes(c, DROP_CNN), Bajada(c)) for _ in range(k)])
        self.pos_enc = Posiciones(c, lado)
        self.transformer = TransformerEspacial(c, cfg.cabezas, c // cfg.cabezas, cfg.capas_atencion, DROP_ATT)
        self.pos_dec = Posiciones(c, lado)
        self.subidas = nn.ModuleList([nn.Sequential(BloqueRes(c, DROP_CNN), Subida(c)) for _ in range(k)])
        self.norm = normalizar(c)
        self.salida = nn.Conv2d(c, 1, 3, padding=1)
        nn.init.zeros_(self.salida.weight)
        nn.init.zeros_(self.salida.bias)
        if cfg.salida == "filtro":
            self.suave = SuavizadoMat(cfg.sigma)

    def cf_crudo(self, x):
        h = self.tallo(x)
        for b in self.bajadas:
            h = b(h)
        h = self.pos_dec(self.transformer(self.pos_enc(h)))
        for s in self.subidas:
            h = s(h)
        h = self.salida(F.silu(self.norm(h)))
        return ((torch.tanh(h) + 1) / 2).clamp(CF_MIN, 1.0).squeeze(1)

    def forward(self, x):
        cf = self.cf_crudo(x)
        if self.cfg.salida == "filtro":
            cf = self.suave(cf).clamp(CF_MIN, 1.0)
        return cf


def construir(cfg):
    if cfg.tipo == "local" and cfg.salida == "filtro":
        return LocalFiltro(cfg)
    if cfg.tipo == "transpath":
        for k, v in TRANSPATH.items():
            setattr(cfg, k, v)
        return TransPathCF(cfg)
    return _construir_original(cfg)


def instalar():
    # train_cf y eval_cf arman y cargan modelos con red.construir; se reemplaza por esta version, que conoce los tipos nuevos
    red.construir = construir
