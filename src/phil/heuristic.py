# cabeza: h_i = mlp(g'_i, x_g)
import torch
import torch.nn as nn


class PhilHeuristic(nn.Module):
    def __init__(self, d_in, d_goal, width=128, depth=3):
        super().__init__()
        layers = [nn.Linear(d_in + d_goal, width), nn.LeakyReLU()]
        for _ in range(depth - 2):
            layers += [nn.Linear(width, width), nn.LeakyReLU()]
        layers.append(nn.Linear(width, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, g_prime, x_goal):
        x_g_b = x_goal.unsqueeze(0).expand(g_prime.size(0), -1)
        return self.mlp(torch.cat([g_prime, x_g_b], dim=-1)).squeeze(-1)
