#on retire le facteur marché, en l'occurence le S&P500
#un outil qui nous sert de proxy de lecture mimétique, et non pas une mesure directe

import pandas as pd
import numpy as np
from Donnees import ret_dc, ret_ref, ret_ia, PERIODES


def corr_residuelle_capm(ret, spx, d1, d2):

    

    
    #1. Filtrer la période pour les titres et le S&P 500
    #2. Pour chaque titre, régresser ses rendements sur le S&P 500
       #par moindres carrés ordinaires (OLS)
    #3. Récupérer le résidu = ce qui reste après l'effet de marché
    #4. Calculer la corrélation moyenne entre tous les résidus
    
    
    r = ret.loc[d1:d2]
    s = spx.loc[d1:d2]

    # Aligner les dates entre les titres et le S&P 500
    s = s.reindex(r.index).ffill() #sert à remplir les valeurs manquantes par la dernière valeur connue (forward fill)

    
    residus = pd.DataFrame(index=r.index) #sert à créer un base vide pour stocker les résidus de chaque titre après la régression sur le S&P 500.

    for col in r.columns: #
        y = r[col].values      # rendements du titre i
        x = s.values           # rendements du S&P 500

        # Garder uniquement les observations où les deux ont des données
        masque = ~(np.isnan(y) | np.isnan(x)) #isnan est une fonction qui renvoie True si l'élément est NaN (Not a Number) et False sinon. 
        #le masque est donc un tableau booléen qui indique quelles observations sont valides (ni y ni x ne sont NaN).
        y_clean = y[masque]
        x_clean = x[masque]

        if len(y_clean) < 30:  # on exclus les données qui ont moins de 30 observations communes avec le S&P 500
            residus[col] = np.nan
            continue

        # OLS manuel : beta = Cov(y,x) / Var(x)
        beta = np.cov(y_clean, x_clean)[0, 1] / np.var(x_clean)
        alpha = np.mean(y_clean) - beta * np.mean(x_clean)

        # Résidu = rendement observé - rendement prédit par le marché
        residu_complet = np.full(len(y), np.nan)
        residu_complet[masque] = y_clean - (alpha + beta * x_clean)
        residus[col] = residu_complet

  
    corr_res = residus.corr()
    n = corr_res.shape[0] 
    masque_diag = ~np.eye(n, dtype=bool) #.eye crée une matrice identité (1 sur la diagonale, 0 ailleurs). ~ inverse les valeurs, donc on obtient True pour toutes les positions hors diagonale et False pour la diagonale. Cela permet d'exclure la corrélation d'un titre avec lui-même (qui est toujours 1) lors du calcul de la moyenne.
    corr_moy = np.nanmean(corr_res.values[masque_diag])

    # PC1 sur les résidus
    vp = np.linalg.eigvalsh(corr_res.values)
    vp = np.sort(vp)[::-1]
    pc1_res = vp[0] / vp.sum()

    return corr_moy, pc1_res



print("CORRÉLATION RÉSIDUELLE POST-CAPM")

print(f"\n{'Période':<42} {'ρ CAPM':>8} {'PC1 rés%':>9}")


resultats = {}
for label, (d1, d2, ret, spx) in PERIODES.items():
    corr, pc1 = corr_residuelle_capm(ret, spx, d1, d2)
    resultats[label] = (corr, pc1)
    print(f"{label:<42} {corr:>8.4f} {pc1*100:>9.1f}%") 

