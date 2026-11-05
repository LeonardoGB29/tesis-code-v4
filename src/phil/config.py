# hiperparametros de phil, defaults = experimento de grids del paper
from dataclasses import asdict, dataclass, fields


@dataclass
class PhilConfig:
    # arquitectura
    d_node: int = 2            # coordenadas 2d crudas
    d_edge: int = 0            # los grids no tienen features de arista
    d_emb: int = 64            # embedding g_i
    d_hidden: int = 64         # memoria z_t
    f_layers: int = 3          # f: mlp 3x128 leakyrelu
    f_width: int = 128
    mlp_layers: int = 3        # cabeza: mlp 3x128 leakyrelu
    mlp_width: int = 128
    coord_scale: float = 1.0   # 1.0 = sin escalar

    # vecindario
    n_neighbors: int = 8       # max vecinos del 1-hop

    # algorithm 1
    T: int = 256                        # horizonte, app c dice 256 y sec 5.1 dice 128
    t_tau: int = 32
    beta_0: float = 0.7
    N_iter: int = 36
    m_trajectories: int = 1
    rolled_in_state: bool = True        # tbtt arranca del z_t del roll-in
    continue_after_goal: bool = True    # sigue hasta t_tau aunque llegue al goal
    corner_goal_prob: float = 1.0       # 1.0 = siempre esquina a esquina

    # optimizacion
    lr: float = 1e-3
    batch_size: int = 32
    weight_decay: float = 0.0
    grad_clip: float = 0.0

    def __post_init__(self):
        self.n_neighbors = int(self.n_neighbors)
        assert self.n_neighbors >= 0, f"n_neighbors inválido: {self.n_neighbors}"

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})
