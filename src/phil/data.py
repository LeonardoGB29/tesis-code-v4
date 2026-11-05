# datos: grid png -> grafo 8-conexo, start/goal, adyacencia y muestreo de vecinos
import random

import numpy as np
import torch
from PIL import Image
from torch_geometric.data import Data


def load_grid(path):
    # png -> matriz bool, true = obstaculo
    return np.array(Image.open(path).convert("L")) < 128


def grid_to_graph(obstacle_mask):
    # celdas libres = nodos, vecinos 8-conexos = aristas
    # nodos en orden por filas, x = (fila, col)
    free = ~np.asarray(obstacle_mask, dtype=bool)
    H, W = free.shape
    ys, xs = np.nonzero(free)
    N = len(ys)
    idx = -np.ones((H, W), dtype=np.int64)
    idx[ys, xs] = np.arange(N)
    node_ids = np.arange(N)
    src_parts, dst_parts = [], []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ny, nx = ys + dy, xs + dx
            ok = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
            nb = np.full(N, -1, dtype=np.int64)
            nb[ok] = idx[ny[ok], nx[ok]]
            valid = nb >= 0
            src_parts.append(node_ids[valid])
            dst_parts.append(nb[valid])
    src = np.concatenate(src_parts) if src_parts else np.zeros(0, dtype=np.int64)
    dst = np.concatenate(dst_parts) if dst_parts else np.zeros(0, dtype=np.int64)
    if len(src):
        order = np.lexsort((dst, src))
        edge_index = torch.from_numpy(np.stack([src[order], dst[order]]).astype(np.int64))
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    x = torch.tensor(np.stack([ys, xs], axis=1), dtype=torch.float)
    return Data(x=x, edge_index=edge_index)


class Adjacency:
    # adyacencia en csr, adj[v] -> lista de vecinos

    def __init__(self, edge_index, num_nodes):
        ei = edge_index.numpy() if torch.is_tensor(edge_index) else np.asarray(edge_index)
        src, dst = ei[0].astype(np.int64), ei[1].astype(np.int64)
        order = np.lexsort((dst, src))
        self.indices = dst[order]
        counts = np.bincount(src, minlength=num_nodes)
        self.indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        self.num_nodes = int(num_nodes)

    def __getitem__(self, v):
        return self.indices[self.indptr[v]:self.indptr[v + 1]].tolist()

    def __len__(self):
        return self.num_nodes

    def degree(self, v):
        return int(self.indptr[v + 1] - self.indptr[v])


def build_adjacency(data):
    return Adjacency(data.edge_index, data.x.size(0))


def sample_start_goal(data):
    # benchmark: start abajo-izquierda, goal arriba-derecha
    c = data.x.numpy()
    diff = c[:, 0] - c[:, 1]
    return int(np.argmax(diff)), int(np.argmin(diff))


def sample_start_for_goal(data, h_star, min_frac=0.4):
    # start alcanzable y lejos del goal
    h = h_star.numpy() if hasattr(h_star, "numpy") else np.asarray(h_star)
    reachable = np.where(h >= 1)[0]
    if len(reachable) == 0:
        return random.randrange(data.x.size(0))
    far = reachable[h[reachable] >= min_frac * h[reachable].max()]
    pool = far if len(far) > 0 else reachable
    return int(random.choice(pool.tolist()))


def sample_neighbors(adj, fringe_nodes, n=8):
    # hasta n vecinos al azar del 1-hop por cada nodo de la frontera
    sampled = []
    for v in fringe_nodes:
        nbrs = adj[v]
        sampled.append(random.sample(nbrs, min(n, len(nbrs))))
    return sampled


def pack_neighborhood(data, fringe_nodes, sampled):
    # arma la entrada del modelo: una arista vecino -> frontera
    # indices locales: frontera primero, vecinos despues
    fringe_nodes = list(fringe_nodes)
    F = len(fringe_nodes)
    seen = {v: i for i, v in enumerate(fringe_nodes)}       # global -> local
    neigh_list, src, dst = [], [], []
    for local_i, nbrs in enumerate(sampled):
        for nb in nbrs:
            if nb not in seen:
                seen[nb] = F + len(neigh_list)
                neigh_list.append(nb)
            src.append(seen[nb])
            dst.append(local_i)
    x_fringe = data.x[fringe_nodes]
    x_neigh = data.x[neigh_list] if neigh_list else data.x.new_zeros((0, data.x.size(1)))
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return x_fringe, x_neigh, edge_index
