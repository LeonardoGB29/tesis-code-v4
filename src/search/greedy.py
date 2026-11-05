# greedy best-first con la heuristica aprendida
# solo evalua los nodos nuevos de la frontera, cada nodo recibe h una vez
# la memoria z_t corre toda la busqueda
import heapq
from dataclasses import dataclass, field

import torch

from src.phil.data import build_adjacency, pack_neighborhood, sample_neighbors


@dataclass
class SearchResult:
    closed: set
    found: bool
    path: list | None
    n_expanded: int       # nodos en closed al terminar
    order: list = field(default_factory=list)


@torch.no_grad()
def greedy_search(model, data, v_start, v_goal, adj=None, max_exp=50000,
                  return_path=False, return_order=False):
    if adj is None:
        adj = build_adjacency(data)
    closed = {v_start}
    opened = set()
    parent = {v_start: None}
    order = [v_start] if return_order else []
    open_pq = []
    counter = 0
    z_t = torch.zeros(model.memory.gru.hidden_size)
    n_neighbors = model.cfg.n_neighbors
    x_goal = data.x[v_goal]
    model.begin_search(data, x_goal)
    model.observe(data, v_start)

    def evaluate(parent_node, fringe):
        nonlocal z_t, counter
        if not fringe:
            return
        x_f, x_n, ei = pack_neighborhood(data, fringe, sample_neighbors(adj, fringe, n_neighbors))
        h_pred, z_t = model(x_f, x_n, ei, x_goal, z_t)
        for h, n in zip(h_pred.tolist(), fringe):
            heapq.heappush(open_pq, (h, counter, n))
            parent.setdefault(n, parent_node)
            opened.add(n)
            counter += 1

    evaluate(v_start, [n for n in adj[v_start] if n not in closed and n not in opened])

    found = False
    while open_pq and len(closed) < max_exp:
        v = heapq.heappop(open_pq)[-1]
        if v in closed:
            continue
        closed.add(v)
        if return_order:
            order.append(v)
        if v == v_goal:
            found = True
            break
        model.observe(data, v)
        evaluate(v, [n for n in adj[v] if n not in closed and n not in opened])

    path = None
    if return_path and found:
        path, cur = [], v_goal
        while cur is not None:
            path.append(cur)
            cur = parent.get(cur)
        path.reverse()
    return SearchResult(closed, found, path, len(closed), order)
