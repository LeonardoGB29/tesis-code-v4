# entrena los modelos de new_exp con el mismo protocolo que cf_filtro, experiments.train_cf con dos modelos mas
#   python -m new_exp.train --dataset gaps_and_forest --seed 0 --modelo cf_local_filtro --ancho 32 --device mps --out-root new_exp/results/cf_local_filtro/runs
#   cf_local_filtro: 5 convs 3x3 sin bajadas (campo 11x11); cf_transpath: autoencoder de transpath; los dos con el gauss fijo de cf_filtro y dagger
import experiments.train_cf as tc
from new_exp import redes

tc.TIPOS.update({"cf_local_filtro": "local", "cf_transpath": "transpath"})
tc.SALIDAS.update({"cf_local_filtro": "filtro", "cf_transpath": "filtro"})
tc.DAGGER.update({"cf_local_filtro", "cf_transpath"})
tc.construir = redes.construir
redes.instalar()

if __name__ == "__main__":
    tc.main()
