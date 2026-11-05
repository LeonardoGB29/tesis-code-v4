# baselines clasicos: bfs, a* euclidiano, greedy con h euclidiana y manhattan
# todos devuelven (closed, found, path)
import heapq
from collections import deque

import numpy as np


def _reconstruct(parent, goal):
    path, cur = [], goal
    while cur is not None:
        path.append(cur)
        cur = parent.get(cur)
    path.reverse()
    return path


def _euclid_to_goal(data, v_goal):
    coords = data.x.numpy()
    return np.linalg.norm(coords - coords[v_goal], axis=1)


def _manhattan_to_goal(data, v_goal):
    coords = data.x.numpy()
    return np.abs(coords - coords[v_goal]).sum(axis=1)


def bfs(data, v_start, v_goal, adj, max_exp=200000):
    closed, parent, q, found = set(), {v_start: None}, deque([v_start]), False
    while q and len(closed) < max_exp:
        u = q.popleft()
        if u in closed:
            continue
        closed.add(u)
        if u == v_goal:
            found = True
            break
        for v in adj[u]:
            if v not in parent:
                parent[v] = u
                q.append(v)
    return closed, found, (_reconstruct(parent, v_goal) if found else None)


def astar(data, v_start, v_goal, adj, max_exp=200000, h=None):
    # a* con aristas de costo 1; h euclidiana si no se pasa otra
    if h is None:
        h = _euclid_to_goal(data, v_goal)
    g = {v_start: 0.0}
    parent = {v_start: None}
    pq = [(float(h[v_start]), 0, v_start)]
    counter, closed, found = 1, set(), False
    while pq and len(closed) < max_exp:
        _, _, u = heapq.heappop(pq)
        if u in closed:
            continue
        closed.add(u)
        if u == v_goal:
            found = True
            break
        gu = g[u] + 1.0
        for v in adj[u]:
            if gu < g.get(v, float("inf")):
                g[v] = gu
                parent[v] = u
                heapq.heappush(pq, (gu + float(h[v]), counter, v))
                counter += 1
    return closed, found, (_reconstruct(parent, v_goal) if found else None)


def _greedy_best_first(h, v_start, v_goal, adj, max_exp):
    # misma busqueda que phil pero con h fija
    parent = {v_start: None}
    pq = [(float(h[v_start]), 0, v_start)]
    counter, closed, found = 1, set(), False
    while pq and len(closed) < max_exp:
        _, _, u = heapq.heappop(pq)
        if u in closed:
            continue
        closed.add(u)
        if u == v_goal:
            found = True
            break
        for v in adj[u]:
            if v in closed:
                continue
            if v not in parent:
                parent[v] = u
                heapq.heappush(pq, (float(h[v]), counter, v))
                counter += 1
    return closed, found, (_reconstruct(parent, v_goal) if found else None)


def heuc(data, v_start, v_goal, adj, max_exp=200000):
    # greedy con h euclidiana
    return _greedy_best_first(_euclid_to_goal(data, v_goal), v_start, v_goal, adj, max_exp)


def hman(data, v_start, v_goal, adj, max_exp=200000):
    # greedy con h manhattan
    return _greedy_best_first(_manhattan_to_goal(data, v_goal), v_start, v_goal, adj, max_exp)
