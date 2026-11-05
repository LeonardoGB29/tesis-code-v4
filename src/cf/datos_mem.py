# exp_mem: fotos al ritmo del cf con expandido y abierto, muestra reproducible de OPEN, vecinos vistos, precalculo con la u-net congelada y lotes por secuencia
import json
import time
import zlib
from pathlib import Path

import numpy as np
import torch

from .fotos import H_TOPE, LADO, canales, etiqueta, hfs_entero
from .memoria import N_ESCALARES, entrada_nodos

ESTADO_OPEN, ESTADO_CLOSED, ESTADO_RESTO = 0, 1, 2
VECINOS = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]


def semilla_muestra(nombre, k):
    return [zlib.crc32(str(nombre).encode()) & 0xFFFFFFFF, int(k)]


def muestra_open(abierto, expandido, nombre, k, n):
    # n celdas de OPEN al azar con semilla por mapa y llamada; si hay menos, todas; ordenadas por fila y columna
    celdas = np.argwhere(abierto & ~expandido)
    if len(celdas) > n:
        rng = np.random.default_rng(semilla_muestra(nombre, k))
        celdas = celdas[np.sort(rng.choice(len(celdas), n, replace=False))]
    return celdas.astype(np.int16).reshape(-1, 2)


def vecinos_vistos(muestra, visto, real, expandido, abierto):
    # nodos = muestra y despues sus vecinos libres ya vistos sin repetir; aristas vecino -> frontera en indices locales; estado por nodo
    H, W = real.shape
    libre = visto & ~real
    pos = {(int(r), int(c)): i for i, (r, c) in enumerate(muestra)}
    nodos = [(int(r), int(c)) for r, c in muestra]
    src, dst = [], []
    for j, (r, c) in enumerate(nodos[:len(muestra)]):
        for dr, dc in VECINOS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W and libre[rr, cc]:
                if (rr, cc) not in pos:
                    pos[(rr, cc)] = len(nodos)
                    nodos.append((rr, cc))
                src.append(pos[(rr, cc)]); dst.append(j)
    nodos = np.array(nodos, dtype=np.int16).reshape(-1, 2)
    estado = np.full(len(nodos), ESTADO_RESTO, dtype=np.int8)
    if len(nodos):
        e = expandido[nodos[:, 0], nodos[:, 1]]
        a = abierto[nodos[:, 0], nodos[:, 1]]
        estado[a & ~e] = ESTADO_OPEN
        estado[e] = ESTADO_CLOSED
    return nodos, np.array([src, dst], dtype=np.int16).reshape(2, -1), estado


def escalares_de(n_exp, visto, hfs, muestra, escala):
    if len(muestra):
        h = hfs[muestra[:, 0], muestra[:, 1]].astype(np.float32)
        h[h < 0] = H_TOPE * escala
        hmin = min(float(h.min()) / escala, H_TOPE)
    else:
        hmin = H_TOPE
    return np.array([n_exp / 1000.0, float(visto.mean()), hmin], dtype=np.float32)


def armar_llamada(visto, real, expandido, abierto, hfs, goal, nombre, k, n_exp, cfg, escala):
    # lo que necesita el modelo en una llamada de la busqueda, lo mismo que se precalcula por foto
    m = muestra_open(abierto, expandido, nombre, k, cfg.n_muestra)
    nodos, aristas, estado = vecinos_vistos(m, visto, real, expandido, abierto)
    return {"muestra": m, "nodos": nodos, "aristas": aristas, "estado": estado, "n_front": len(m), "goal": (int(goal[0]), int(goal[1])),
            "escalares": escalares_de(n_exp, visto, hfs, m, escala)}


def desempaquetar(bits, H, W):
    return np.unpackbits(bits)[:H * W].reshape(H, W).astype(bool)


class FotosMem:
    # fotos de varias busquedas sobre los mapas de entrenamiento, en orden de busqueda; cada (mapa, origen) es una secuencia
    CAMPOS_FOTO = ("mapa_idx", "origen", "paso", "visto", "expandido", "abierto", "hfs", "n_exp")

    def __init__(self, nombres, real, hstar, goal, start, escala, **fotos):
        self.nombres, self.real, self.hstar, self.goal, self.start, self.escala = nombres, real, hstar, goal, start, float(escala)
        for k in self.CAMPOS_FOTO:
            setattr(self, k, fotos[k])

    def __len__(self):
        return len(self.mapa_idx)

    def mapa(self, i):
        return int(self.mapa_idx[i])

    def bits(self, campo, i):
        H, W = self.real.shape[1:]
        return desempaquetar(getattr(self, campo)[i], H, W)

    def secuencias(self):
        # (inicio, largo) de cada tramo con el mismo mapa y origen, en el orden guardado
        claves = self.mapa_idx.astype(np.int64) * 4 + self.origen.astype(np.int64)
        cortes = np.flatnonzero(np.diff(claves)) + 1
        ini = np.concatenate([[0], cortes])
        fin = np.concatenate([cortes, [len(claves)]])
        return list(zip(ini.tolist(), (fin - ini).tolist()))

    def guardar(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "nombres.npy", np.array(self.nombres))
        for k in ("real", "hstar", "goal", "start") + self.CAMPOS_FOTO:
            np.save(path / f"{k}.npy", getattr(self, k))
        np.save(path / "escala.npy", np.array(self.escala))
        (path / "listo").write_text(f"{len(self)} fotos de {len(self.nombres)} mapas\n")

    @classmethod
    def cargar(cls, path):
        path = Path(path)
        assert (path / "listo").exists(), f"fotos incompletas en {path}"
        ld = lambda k, mm=None: np.load(path / f"{k}.npy", mmap_mode=mm, allow_pickle=False)
        return cls([str(n) for n in ld("nombres")], ld("real"), ld("hstar"), ld("goal"), ld("start"), float(ld("escala")),
                   **{k: ld(k, "r" if k in ("visto", "expandido", "abierto", "hfs") else None) for k in cls.CAMPOS_FOTO})


def recolectar_mem(dataset, grids, hacer_evaluadores, max_exp, verbose=False):
    # hacer_evaluadores(data, adj, v_g) -> lista de evaluadores con fotos_mem, uno por origen (0 h_fs, 1 cf_filtro)
    from experiments.common import hstar_mapa, load_problem
    from .busqueda import voraz
    nombres, reales, hstars, goals, starts = [], [], [], [], []
    cols = {k: [] for k in FotosMem.CAMPOS_FOTO}
    filas = []
    for gp in grids:
        data, adj, v_s, v_g, h_star = load_problem(gp)
        opt = int(h_star[v_s])
        if opt < 0:
            continue
        evs = hacer_evaluadores(data, adj, v_g)
        k = len(nombres)
        for origen, ev in enumerate(evs):
            t0 = time.time()
            res = voraz(ev, data, v_s, v_g, adj, max_exp=max_exp)
            if origen == 0:
                nombres.append(gp.name); reales.append(ev.desc.real); hstars.append(hstar_mapa(data, h_star))
                goals.append(ev.goal); starts.append((int(ev.rc[v_s, 0]), int(ev.rc[v_s, 1])))
            for paso, fm in enumerate(ev.fotos_mem):
                cols["mapa_idx"].append(k); cols["origen"].append(origen); cols["paso"].append(paso)
                for c in ("visto", "expandido", "abierto", "hfs", "n_exp"):
                    cols[c].append(fm[c])
            filas.append([dataset, gp.name, origen, opt, res.n_expanded if res.found else max_exp, int(res.found),
                          (len(res.path) - 1) if res.path else "", len(ev.fotos_mem), round(time.time() - t0, 3)])
            if verbose:
                print(f"[{k + 1:3d}] {gp.name} origen {origen} exp={filas[-1][4]} fotos={len(ev.fotos_mem)} ({filas[-1][-1]}s)", flush=True)
    escala = evs[0].escala
    fotos = FotosMem(nombres, np.stack(reales), np.stack(hstars), np.array(goals, dtype=np.int16), np.array(starts, dtype=np.int16), escala,
                     mapa_idx=np.array(cols["mapa_idx"], dtype=np.int32), origen=np.array(cols["origen"], dtype=np.int8),
                     paso=np.array(cols["paso"], dtype=np.int16), visto=np.stack(cols["visto"]), expandido=np.stack(cols["expandido"]),
                     abierto=np.stack(cols["abierto"]), hfs=np.stack(cols["hfs"]), n_exp=np.array(cols["n_exp"], dtype=np.int32))
    return fotos, filas


class Precalculo:
    # por foto, con la u-net congelada: mapa de cf_filtro, rasgos en los nodos de la muestra y sus vecinos, aristas, estado y escalares
    CAMPOS = ("cf", "rasgos", "nodos", "n_front", "n_nodos", "estado", "aristas", "n_aristas", "escalares")

    def __init__(self, path):
        self.path = Path(path)
        assert (self.path / "listo").exists(), f"precalculo incompleto en {self.path}"
        for k in self.CAMPOS:
            setattr(self, k, np.load(self.path / f"{k}.npy", mmap_mode="r", allow_pickle=False))
        self.meta = json.loads((self.path / "meta.json").read_text())

    @staticmethod
    def nmax(cfg):
        return cfg.n_muestra * 9, cfg.n_muestra * 8

    @classmethod
    def crear(cls, path, fotos, unet, cfg, device, lote=16, verbose=False):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        N, (NMAX, AMAX) = len(fotos), cls.nmax(cfg)
        H, W = fotos.real.shape[1:]
        mm = lambda k, shape, dt: np.lib.format.open_memmap(path / f"{k}.npy", mode="w+", dtype=dt, shape=shape)
        arr = {"cf": mm("cf", (N, LADO, LADO), np.float16), "rasgos": mm("rasgos", (N, NMAX, cfg.d_rasgos), np.float16),
               "nodos": mm("nodos", (N, NMAX, 2), np.int16), "n_front": mm("n_front", (N,), np.int16), "n_nodos": mm("n_nodos", (N,), np.int16),
               "estado": mm("estado", (N, NMAX), np.int8), "aristas": mm("aristas", (N, AMAX, 2), np.int16), "n_aristas": mm("n_aristas", (N,), np.int16),
               "escalares": mm("escalares", (N, N_ESCALARES), np.float32)}
        arr["nodos"][:] = -1; arr["aristas"][:] = -1
        unet.eval()
        t0 = time.time()
        for k in range(0, N, lote):
            idx = list(range(k, min(k + lote, N)))
            xs, ents = [], []
            for i in idx:
                m = fotos.mapa(i)
                visto, expandido, abierto = fotos.bits("visto", i), fotos.bits("expandido", i), fotos.bits("abierto", i)
                hfs = fotos.hfs[i]
                xs.append(canales(visto, fotos.real[m], fotos.goal[m], hfs, fotos.escala))
                ents.append(armar_llamada(visto, fotos.real[m], expandido, abierto, hfs, fotos.goal[m], fotos.nombres[m], int(fotos.paso[i]),
                                          int(fotos.n_exp[i]), cfg, fotos.escala))
            with torch.no_grad():
                cf, f = unet.cf_y_rasgos(torch.from_numpy(np.stack(xs)).to(device))
                cf = cf.cpu().numpy()
                for b, (i, ent) in enumerate(zip(idx, ents)):
                    n, a = len(ent["nodos"]), ent["aristas"].shape[1]
                    assert n <= NMAX and a <= AMAX, (n, a)
                    arr["cf"][i] = cf[b]
                    if n:
                        nt = torch.from_numpy(ent["nodos"].astype(np.int64)).to(device)
                        arr["rasgos"][i, :n] = f[b][:, nt[:, 0], nt[:, 1]].t().cpu().numpy()
                        arr["nodos"][i, :n] = ent["nodos"]
                        arr["estado"][i, :n] = ent["estado"]
                    if a:
                        arr["aristas"][i, :a] = ent["aristas"].T
                    arr["n_front"][i], arr["n_nodos"][i], arr["n_aristas"][i] = ent["n_front"], n, a
                    arr["escalares"][i] = ent["escalares"]
            if verbose and (k // lote) % 50 == 0:
                print(f"  precalculo {k + len(idx)}/{N} ({time.time() - t0:.0f}s)", flush=True)
        for v in arr.values():
            v.flush()
        (path / "meta.json").write_text(json.dumps({"n": N, "config": cfg.to_dict(), "segundos": round(time.time() - t0, 1)}, indent=1))
        (path / "listo").write_text(f"{N} fotos\n")
        return cls(path)


def lotes_por_fotos(secuencias, rng, tope):
    # agrupa secuencias barajadas hasta tope fotos por lote; una secuencia mas larga que el tope va sola
    orden = rng.permutation(len(secuencias))
    lotes, actual, n = [], [], 0
    for j in orden:
        ini, largo = secuencias[j]
        if actual and n + largo > tope:
            lotes.append(actual); actual, n = [], 0
        actual.append((ini, largo)); n += largo
    if actual:
        lotes.append(actual)
    return lotes


def armar_lote(fotos, pre, seqs, cfg, device):
    # tensores de un lote de secuencias: grafo disjunto de todas las fotos, indices de secuencia y paso por foto, cf_filtro, etiqueta y mascara
    idx = [i for ini, largo in seqs for i in range(ini, ini + largo)]
    P, L = len(idx), max(largo for _, largo in seqs)
    seq = np.concatenate([np.full(largo, b) for b, (_, largo) in enumerate(seqs)])
    paso = np.concatenate([np.arange(largo) for _, largo in seqs])
    xs_rc, xs_ras, xs_est, xs_esc, xs_goal, fotos_n, celdas, es_front, eis, ys, ms, cfs = [], [], [], [], [], [], [], [], [], [], [], []
    off = 0
    for p, i in enumerate(idx):
        n, a, nf = int(pre.n_nodos[i]), int(pre.n_aristas[i]), int(pre.n_front[i])
        m = fotos.mapa(i)
        goal = fotos.goal[m]
        if n:
            nod = pre.nodos[i, :n].astype(np.int64)
            xs_rc.append(nod); xs_ras.append(pre.rasgos[i, :n].astype(np.float32)); xs_est.append(pre.estado[i, :n].astype(np.int64))
            xs_esc.append(np.broadcast_to(pre.escalares[i], (n, N_ESCALARES))); xs_goal.append(np.broadcast_to(goal.astype(np.float32), (n, 2)))
            fotos_n.append(np.full(n, p)); ef = np.zeros(n, dtype=bool); ef[:nf] = True; es_front.append(ef)
            if a:
                eis.append(pre.aristas[i, :a].astype(np.int64).T + off)
            off += n
        y, msk = etiqueta(fotos.hfs[i], fotos.hstar[m], fotos.real[m], goal)
        ys.append(y); ms.append(msk); cfs.append(pre.cf[i])
    cat = lambda xs, dt: torch.from_numpy(np.ascontiguousarray(np.concatenate(xs))).to(dt) if xs else torch.zeros(0, dtype=dt)
    rc = cat(xs_rc, torch.long)
    ras, est, esc, goal_n = cat(xs_ras, torch.float), cat(xs_est, torch.long), cat(xs_esc, torch.float), cat(xs_goal, torch.float)
    x_nodos = entrada_nodos(rc.float(), goal_n, ras, est, esc, cfg.coord_scale) if rc.size(0) else torch.zeros(0, 6 + cfg.d_rasgos + 3 + N_ESCALARES)
    ei = torch.from_numpy(np.concatenate(eis, axis=1)) if eis else torch.zeros(2, 0, dtype=torch.long)
    lote = {"x_nodos": x_nodos, "ei": ei, "es_front": cat(es_front, torch.bool), "foto": cat(fotos_n, torch.long), "celdas": rc,
            "goal_foto": torch.from_numpy(fotos.goal[[fotos.mapa(i) for i in idx]].astype(np.float32)), "seq": torch.from_numpy(seq).long(),
            "paso": torch.from_numpy(paso).long(), "B": len(seqs), "L": L,
            "cf0": torch.from_numpy(np.stack(cfs).astype(np.float32)), "y": torch.from_numpy(np.stack(ys)), "m": torch.from_numpy(np.stack(ms))}
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in lote.items()}
