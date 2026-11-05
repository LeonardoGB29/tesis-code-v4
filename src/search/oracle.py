# oraculo h*: saltos reales de cada nodo al goal, bfs desde el goal
from collections import deque

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


def bfs_oracle(data, v_goal):
    # con scipy, -1 si no se llega
    N = data.x.size(0)
    ei = data.edge_index.numpy()
    A = csr_matrix((np.ones(ei.shape[1], dtype=np.float32), (ei[0], ei[1])), shape=(N, N))
    d = shortest_path(A, method="D", directed=True, unweighted=True, indices=v_goal)
    dist = np.where(np.isinf(d), -1, d).astype(np.int64)
    return torch.from_numpy(dist)


def bfs_oracle_reference(data, v_goal):
    # version python pura, solo para tests
    N = data.x.size(0)
    dist = torch.full((N,), -1, dtype=torch.long)
    dist[v_goal] = 0
    adj = [[] for _ in range(N)]
    for s, d in zip(*data.edge_index.tolist()):
        adj[s].append(d)
    q = deque([v_goal])
    while q:
        u = q.popleft()
        for w in adj[u]:
            if dist[w] == -1:
                dist[w] = dist[u] + 1
                q.append(w)
    return dist
