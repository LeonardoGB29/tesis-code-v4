# evalua en los 100 mapas de prueba los modelos de new_exp con experiments.eval_cf, que los carga con red.construir
#   python -m new_exp.eval --dataset gaps_and_forest --ckpt new_exp/results/cf_transpath/runs/gaps_and_forest/cf_transpath_s0/best.pt --out <csv>
import experiments.eval_cf as ec
from new_exp import redes

redes.instalar()

if __name__ == "__main__":
    ec.main()
