# rutas, carga de problemas y utilidades compartidas
import csv
from pathlib import Path

import numpy as np
import torch

from src.mapa_oculto.model import FsConfig, PhilFS, PhilRes, ResConfig
from src.phil.config import PhilConfig
from src.phil.data import build_adjacency, grid_to_graph, load_grid, sample_start_goal
from src.phil.model import PHIL
from src.search.oracle import bfs_oracle

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data" / "motion_planning_datasets"
RESULTS = REPO / "results" / "exp_cf"
DATASETS = ["gaps_and_forest", "multiple_bugtraps", "mazes"]
N_TRAIN, N_VAL, N_TEST = 200, 70, 100
MAX_EXP = 10000
ESCALA = 200.0


def set_threads(n):
    torch.set_num_threads(max(1, int(n)))


def list_grids(dataset, split):
    d = DATA_DIR / dataset / split
    if not d.exists():
        raise FileNotFoundError(d)
    return sorted(d.glob("*.png"))


def grids_de(dataset, split):
    n = {"train": N_TRAIN, "validation": N_VAL, "test": N_TEST}[split]
    return list_grids(dataset, split)[:n]


_ORACULO = {}


def load_problem(grid_path):
    # esquina a esquina, h* con cache chica
    data = grid_to_graph(load_grid(grid_path))
    adj = build_adjacency(data)
    v_s, v_g = sample_start_goal(data)
    key = (str(grid_path), v_g)
    if key not in _ORACULO:
        if len(_ORACULO) >= 256:
            _ORACULO.clear()
        _ORACULO[key] = bfs_oracle(data, v_g)
    return data, adj, v_s, v_g, _ORACULO[key]


def hstar_mapa(data, h_star):
    # h* por celda, -1 en obstaculos e inalcanzables
    rc = data.x.long().numpy()
    H = int(rc.max()) + 1
    m = -np.ones((H, H), dtype=np.int32)
    m[rc[:, 0], rc[:, 1]] = h_star.numpy()
    return m


def load_phil(ckpt_path, device="cpu"):
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    c = raw["config"]
    if "cf_ckpt" in c:                                              # exp_mem: phil_res_cf, PHIL sobre cf_filtro congelado
        from src.cf.phil_res import PhilResCF, ResCfConfig
        cfg = ResCfConfig.from_dict(c)
        model = PhilResCF(cfg, device=device)
        model.load_state_dict(raw["model_state"], strict=False)
        model.eval()
        return model, cfg, raw
    cls = ResConfig if "residual" in c else FsConfig if "fs_every" in c else PhilConfig
    cfg = cls.from_dict(c)
    model = PhilRes(cfg) if cls is ResConfig else PhilFS(cfg) if cls is FsConfig else PHIL(cfg)
    model.load_state_dict(raw["model_state"])
    model.eval()
    return model, cfg, raw


def write_csv(path, header, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    tmp.replace(path)                       # aparece completo o no aparece


def leer_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def f(x, nd=3):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


def md_tabla(head, filas):
    filas = [[str(x) for x in r] for r in filas]
    anchos = [max([len(h)] + [len(r[j]) for r in filas if j < len(r)] + [3]) for j, h in enumerate(head)]
    linea = lambda r: "| " + " | ".join(c.ljust(w) for c, w in zip(r, anchos)) + " |"
    return "\n".join([linea(head), "|" + "|".join("-" * (w + 2) for w in anchos) + "|"] + [linea(r) for r in filas]) + "\n"


def p_wilcoxon(a, b):
    from scipy.stats import wilcoxon
    d = np.asarray(a, float) - np.asarray(b, float)
    d = np.where(np.abs(d) < 1e-9, 0.0, d)            # la media de semillas iguales difiere de la referencia en un ulp
    if len(d) < 5 or np.all(d == 0):
        return 1.0
    return float(wilcoxon(d).pvalue)
