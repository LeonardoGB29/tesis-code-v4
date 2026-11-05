# cnn de geometria, la misma u-net de cf_filtro con 3 canales de entrada y una cabeza que da el logit de obstaculo por celda
# la cabeza arranca en cero con el sesgo en logit(p0), asi la epoca 0 es el previo constante
import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from src.cf.red import CfConfig, UNetCF

from .fotos import CANALES_GEO


@dataclass
class GeoConfig(CfConfig):
    tipo: str = "global"
    ancho: int = 16
    canales_entrada: int = CANALES_GEO
    salida: str = "fina"
    p0: float = 0.5                 # previo constante de obstaculo en lo no visto


class UNetGeo(UNetCF):
    def __init__(self, cfg):
        super().__init__(cfg)
        del self.cabeza
        self.logit = nn.Conv2d(cfg.d_rasgos, 1, 1)
        nn.init.zeros_(self.logit.weight)
        p = min(max(float(cfg.p0), 1e-6), 1.0 - 1e-6)
        nn.init.constant_(self.logit.bias, math.log(p / (1.0 - p)))

    def forward(self, x):
        # x [B, 3, LADO, LADO] -> logits [B, LADO, LADO]
        return self.logit(self.rasgos_y_grueso(x)[0]).squeeze(1)

    def prob(self, x):
        return torch.sigmoid(self(x))


def construir_geo(cfg):
    return UNetGeo(cfg)


def guardar_geo(path, model, epoca, val, extra=None):
    torch.save({"epoca": epoca, "model_state": model.state_dict(), "config": model.cfg.to_dict(), "val": val,
                "extra": extra or {}}, path)


def cargar_geo(path, device="cpu"):
    raw = torch.load(path, map_location="cpu", weights_only=False)
    cfg = GeoConfig.from_dict(raw["config"])
    model = construir_geo(cfg)
    model.load_state_dict(raw["model_state"])
    model.to(device).eval()
    return model, cfg, raw
