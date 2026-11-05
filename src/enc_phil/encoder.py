# cnn local: el vector de una celda solo depende de su ventana k x k
import torch
import torch.nn as nn


def obstacle_grid(data):
    # mascara de obstaculos desde las coords de los nodos libres, 1 = obstaculo
    rc = data.x.long()
    size = int(rc.max()) + 1
    grid = torch.ones(size, size)
    grid[rc[:, 0], rc[:, 1]] = 0.0
    return grid


def local_fields(obstacles, goal_rc, pad):
    # 3 canales por celda: obstaculo, direccion al goal (fila, col)
    # fuera del mapa: obstaculo y direccion cero
    H, W = obstacles.shape
    rows = torch.arange(H, dtype=torch.float).view(H, 1).expand(H, W)
    cols = torch.arange(W, dtype=torch.float).view(1, W).expand(H, W)
    d = torch.stack([goal_rc[0] - rows, goal_rc[1] - cols])
    d = d / d.norm(dim=0, keepdim=True).clamp(min=1e-6)
    out = torch.zeros(3, H + 2 * pad, W + 2 * pad)
    out[0] = 1.0
    out[0, pad:pad + H, pad:pad + W] = obstacles
    out[1:, pad:pad + H, pad:pad + W] = d
    return out


class LocalEncoder(nn.Module):
    def __init__(self, window, width, d_out):
        super().__init__()
        self.pad = window // 2
        layers, c = [], 3
        for _ in range(self.pad):              # cada conv 3x3 sin relleno quita 2: k -> 1
            layers += [nn.Conv2d(c, width, 3), nn.LeakyReLU()]
            c = width
        layers.append(nn.Conv2d(c, d_out, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obstacles, goal_rc):
        # [H, W] -> [d, H, W], una pasada para todas las ventanas del mapa
        return self.net(local_fields(obstacles, goal_rc, self.pad)[None])[0]
