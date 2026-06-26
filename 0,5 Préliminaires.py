
#calculs préliminaires 
import pandas as pd
import numpy as np
from Donnees import ret_dc, ret_ref, ret_ia, PERIODES


def stats_descriptives(ret, d1, d2):

   
   
    r = ret.loc[d1:d2]

    rdt_ann = r.mean().mean() * 252

    vol_ann = r.std().mean() * np.sqrt(252)

    prix_cum = (1 + r.fillna(0)).cumprod() #pour calculer le prix cumulé

    # pour chaque titre, drawdown = prix / max_historique - 1
    drawdowns = prix_cum / prix_cum.cummax() - 1

    # Drawdown moyen sur tous les titres
    dd_moy = drawdowns.min().mean()

    # Drawdown absolu = le pire titre
    dd_max = drawdowns.min().min()

    return rdt_ann, vol_ann, dd_moy, dd_max



print("STATISTIQUES DESCRIPTIVES")

print(f"\n{'Période':<42} {'Rdt ann%':>8} {'Vol ann%':>9} {'DD moy%':>8} {'DD max%':>8}")


for label, (d1, d2, ret, _spx) in PERIODES.items():
    rdt, vol, dd_moy, dd_max = stats_descriptives(ret, d1, d2)
    print(f"{label:<42} {rdt*100:>8.1f} {vol*100:>9.1f} {dd_moy*100:>8.1f} {dd_max*100:>8.1f}")

#la période bulle internet présente les chiffres les plus volatiles avec des drawdowns maximum, qui s'explique par l'effondrement des valorisations.