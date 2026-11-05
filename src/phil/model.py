# phil, algorithm 2 del paper
#   x_i  <- f(x_i, x_g, d_euc, d_cos)    igual para los vecinos
#   g_i  <- gnn de 1 capa sobre sus vecinos
#   g'_i <- gru(g_i, z_t), z nuevo = media
#   h_i  <- mlp(g'_i, x_g)
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gnn import PhilGNN
from .heuristic import PhilHeuristic
from .memory import PhilMemory


class PHIL(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.coord_scale = cfg.coord_scale
        d_proj_in = cfg.d_node * 2 + 2 + getattr(cfg, "d_enc", 0)   # x_i, x_g, d_euc, d_cos (+ encoder)

        # f: mlp 3x128 leakyrelu
        if cfg.f_layers <= 1:
            self.f = nn.Sequential(nn.Linear(d_proj_in, cfg.d_emb), nn.LeakyReLU())
        else:
            layers = [nn.Linear(d_proj_in, cfg.f_width), nn.LeakyReLU()]
            for _ in range(cfg.f_layers - 2):
                layers += [nn.Linear(cfg.f_width, cfg.f_width), nn.LeakyReLU()]
            layers += [nn.Linear(cfg.f_width, cfg.d_emb), nn.LeakyReLU()]
            self.f = nn.Sequential(*layers)

        self.gnn = PhilGNN(cfg.d_emb, cfg.d_emb, edge_dim=cfg.d_edge or None)
        self.memory = PhilMemory(cfg.d_emb, cfg.d_hidden)
        self.heuristic = PhilHeuristic(cfg.d_hidden, cfg.d_node, cfg.mlp_width, cfg.mlp_layers)

    def begin_search(self, data, x_goal):
        # gancho al inicio de cada busqueda; phil no necesita nada
        pass

    def observe(self, data, v):
        # gancho al expandir el nodo v
        pass

    def step_state(self):
        # lo que hay que guardar de un paso para repetirlo al entrenar
        return None

    def set_step_state(self, state):
        pass

    def node_input(self, x, x_g):
        # entrada de f: coords, goal, dist euclidiana, dist coseno
        x = x / self.coord_scale
        x_g = x_g / self.coord_scale
        x_g_b = x_g.unsqueeze(0).expand(x.size(0), -1)
        d_euc = (x - x_g_b).norm(dim=-1, keepdim=True)
        d_cos = 1.0 - F.cosine_similarity(x, x_g_b, dim=-1).unsqueeze(-1)   # distancia coseno
        return torch.cat([x, x_g_b, d_euc, d_cos], dim=-1)

    def project(self, x, x_g):
        return self.f(self.node_input(x, x_g))

    def forward(self, x_fringe, x_neigh, edge_index_local, x_goal, z_t, edge_attr=None):
        x_all = torch.cat([x_fringe, x_neigh], dim=0)
        x_proj = self.project(x_all, x_goal)
        n = x_fringe.size(0)
        g = self.gnn(x_proj, edge_index_local, edge_attr)[:n]   # solo la frontera
        g_prime, z_t1 = self.memory(g, z_t)
        h = self.heuristic(g_prime, x_goal / self.coord_scale)
        return h, z_t1
