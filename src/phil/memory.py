# memoria: gru por nodo, z nuevo = media sobre la frontera
import torch.nn as nn


class PhilMemory(nn.Module):
    def __init__(self, d_emb, d_hidden):
        super().__init__()
        self.gru = nn.GRUCell(input_size=d_emb, hidden_size=d_hidden)

    def forward(self, g, z_t):
        z_batch = z_t.unsqueeze(0).expand(g.size(0), -1)   # mismo z_t para toda la frontera
        z_i_t1 = self.gru(g, z_batch)                       # g'_i
        z_t1 = z_i_t1.mean(dim=0)                           # media, no depende del orden
        return z_i_t1, z_t1
