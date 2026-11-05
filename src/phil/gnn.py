# conv de grafo: deepergcn (genconv) con agregacion softmax, una sola capa
import torch.nn as nn
from torch_geometric.nn import GENConv


class PhilGNN(nn.Module):
    def __init__(self, d_in, d_out, edge_dim=None):
        super().__init__()
        self.conv = GENConv(
            in_channels=d_in, out_channels=d_out,
            aggr="softmax", learn_t=True,     # temperatura aprendible
            num_layers=2,                     # mlp interno
            norm="layer",                     # layernorm, batchnorm falla con batches chicos
            msg_norm=True, learn_msg_scale=True,
            edge_dim=edge_dim,                # solo si hay features de arista
        )

    def forward(self, x, edge_index, edge_attr=None):
        return self.conv(x, edge_index, edge_attr)
