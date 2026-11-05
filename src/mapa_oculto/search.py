# linea base sin aprendizaje: voraz guiada por h_fs tomada al entrar a la frontera
import heapq

from src.enc_phil.encoder import obstacle_grid

from .fs import FreeSpace
from .state import Discovered


def greedy_fs(data, v_start, v_goal, adj, every, max_exp=200000):
    # misma mecanica que la busqueda de phil: cada nodo recibe su valor una vez y no se actualiza
    rc = data.x.long().numpy()
    disc = Discovered(obstacle_grid(data).numpy().astype(bool))
    fs = FreeSpace(disc, (int(rc[v_goal, 0]), int(rc[v_goal, 1])), every)
    closed, opened, parent = {v_start}, set(), {v_start: None}
    pq, counter = [], 0

    def expand(u):
        nonlocal counter
        disc.observe(int(rc[u, 0]), int(rc[u, 1]))
        fs.update(disc)
        for v in adj[u]:
            if v not in closed and v not in opened:
                heapq.heappush(pq, (float(fs.h[rc[v, 0], rc[v, 1]]), counter, v))
                parent[v] = u
                opened.add(v)
                counter += 1

    expand(v_start)
    found = False
    while pq and len(closed) < max_exp:
        u = heapq.heappop(pq)[-1]
        if u in closed:
            continue
        closed.add(u)
        if u == v_goal:
            found = True
            break
        expand(u)
    path = None
    if found:
        path, cur = [], v_goal
        while cur is not None:
            path.append(cur)
            cur = parent.get(cur)
        path.reverse()
    return closed, found, path, fs.n_updates
