# algorithm 1: recoleccion de trayectorias (roll-in + roll-out) y perdida tbtt
import csv
import heapq
import random
from pathlib import Path

import torch

from .data import pack_neighborhood, sample_neighbors


def _pop_unclosed(pq, closed):
    while pq:
        _, _, v = heapq.heappop(pq)
        if v not in closed:
            return v
    return None


@torch.no_grad()
def collect_trajectory(model, data, adj, h_star, v_s, v_g, cfg, beta):
    # roll-in: t pasos con el modelo
    # roll-out: t_tau pasos mezclando oraculo (prob beta) y modelo
    # cada paso guarda frontera nueva, vecinos muestreados, h* y el estado extra del modelo
    t = random.randint(0, cfg.T - cfg.t_tau)
    closed = {v_s}
    opened = set()                     # cada nodo se evalua una sola vez
    open_model, open_oracle = [], []   # dos colas: modelo y oraculo
    counter = 0
    z_t = torch.zeros(cfg.d_hidden)
    x_goal = data.x[v_g]
    model.begin_search(data, x_goal)

    def new_fringe_of(v):
        return [n for n in adj[v] if n not in closed and n not in opened]

    def evaluate(fringe):
        nonlocal z_t, counter
        if not fringe:
            return None, None
        sampled = sample_neighbors(adj, fringe, cfg.n_neighbors)
        x_f, x_n, ei = pack_neighborhood(data, fringe, sampled)
        h_pred, z_t = model(x_f, x_n, ei, x_goal, z_t)
        for h_m, n in zip(h_pred.tolist(), fringe):
            hs = h_star[n].item()
            heapq.heappush(open_model, (h_m, counter, n))
            heapq.heappush(open_oracle, (hs if hs >= 0 else float("inf"), counter, n))
            opened.add(n)
            counter += 1
        return sampled, model.step_state()

    model.observe(data, v_s)
    evaluate(new_fringe_of(v_s))

    goal_reached = False
    for _ in range(t):                                   # roll-in
        v = _pop_unclosed(open_model, closed)
        if v is None:
            break
        closed.add(v)
        if v == v_g:
            goal_reached = True
            if not cfg.continue_after_goal:
                break
        model.observe(data, v)
        evaluate(new_fringe_of(v))

    z_t_init = z_t.detach().clone()
    if goal_reached and not cfg.continue_after_goal:
        return [], z_t_init

    trajectory = []
    for _ in range(cfg.t_tau):                           # roll-out
        pq = open_oracle if random.random() < beta else open_model
        v = _pop_unclosed(pq, closed)
        if v is None:
            break
        closed.add(v)
        if v == v_g and not cfg.continue_after_goal:
            break
        model.observe(data, v)
        new_fringe = new_fringe_of(v)
        if new_fringe:
            targets = h_star[new_fringe].float() / cfg.coord_scale
            sampled, state = evaluate(new_fringe)
            trajectory.append((new_fringe, sampled, targets, state))
    return trajectory, z_t_init


def tbtt_loss(model, data, x_goal, trajectory, z_t_init):
    # desenrolla la trayectoria con gradiente, mse contra h* por paso
    # un solo backward pasa por toda la gru
    if not trajectory:
        return torch.tensor(0.0, requires_grad=True)
    model.begin_search(data, x_goal)
    z_t = z_t_init
    losses = []
    for V_new, N_new, h_star_new, state in trajectory:
        model.set_step_state(state)
        x_f, x_n, ei = pack_neighborhood(data, V_new, N_new)
        h_pred, z_t = model(x_f, x_n, ei, x_goal, z_t)
        mask = h_star_new >= 0          # -1 = inalcanzable
        if mask.any():
            losses.append(((h_pred[mask] - h_star_new[mask]) ** 2).mean())
    model.set_step_state(None)
    if not losses:
        return torch.tensor(0.0, requires_grad=True)
    return torch.stack(losses).mean()


def save_checkpoint(path, model, optimizer, iteration, val_metric, cfg, extra=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "iteration": iteration,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer else None,
        "val_metric": val_metric,
        "config": cfg.to_dict(),
        "extra": extra or {},
    }, path)


def load_checkpoint(path, model, optimizer=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and ckpt.get("optimizer_state"):
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt


class CSVLogger:
    def __init__(self, path, fieldnames):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        is_new = (not self.path.exists()) or self.path.stat().st_size == 0
        self.file = open(self.path, "a", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
        if is_new:
            self.writer.writeheader()

    def log(self, **kwargs):
        self.writer.writerow({f: kwargs.get(f, "") for f in self.writer.fieldnames})
        self.file.flush()

    def close(self):
        self.file.close()
