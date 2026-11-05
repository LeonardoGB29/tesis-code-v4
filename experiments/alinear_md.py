# alinea las tablas markdown de un archivo para leerlas en texto plano
#   python -m experiments.alinear_md results/exp_inyeccion/RESUMEN.md ...
import sys
from pathlib import Path


def alinear_tabla(lineas):
    filas = [[c.strip() for c in l.strip().strip("|").split("|")] for l in lineas]
    n = max(len(f) for f in filas)
    filas = [f + [""] * (n - len(f)) for f in filas]
    sep = 1 if len(filas) > 1 and all(set(c) <= set("-: ") for c in filas[1]) else -1
    anchos = [max(len(f[j]) for i, f in enumerate(filas) if i != sep) for j in range(n)]
    anchos = [max(w, 3) for w in anchos]
    out = []
    for i, f in enumerate(filas):
        if i == sep:
            out.append("|" + "|".join("-" * (w + 2) for w in anchos) + "|")
        else:
            out.append("| " + " | ".join(c.ljust(w) for c, w in zip(f, anchos)) + " |")
    return out


def alinear(texto):
    out, bloque = [], []
    for l in texto.splitlines():
        if l.lstrip().startswith("|"):
            bloque.append(l)
            continue
        if bloque:
            out += alinear_tabla(bloque)
            bloque = []
        out.append(l)
    if bloque:
        out += alinear_tabla(bloque)
    return "\n".join(out) + "\n"


def main():
    for p in sys.argv[1:]:
        p = Path(p)
        p.write_text(alinear(p.read_text()))
        print(f"-> {p}")


if __name__ == "__main__":
    main()
