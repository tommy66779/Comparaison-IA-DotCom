#calcul de ratio de diversification (DR)

import pandas as pd
import numpy as np
from Donnees import ret_dc, ret_ref, ret_ia, PERIODES



def calcul_dr(ret, d1, d2): #le principe reste le même que les fonctions précédentes.
    """
    
    #1. Filtrer la période
    #2. Calculer la matrice de covariance annualisée (x252)
    #3. Extraire les volatilités individuelles (racine de la diagonale)
    #4. Définir les poids équipondérés (1/n)
    #5. Calculer le numérateur : somme pondérée des volatilités
    #6. Calculer le dénominateur : volatilité du portefeuille
   
    """
 
    r = ret.loc[d1:d2]

    # 2. Matrice de covariance annualisée
    # On multiplie par 252 car on a des rendements journaliers
    # et on veut des volatilités annualisées
    cov = r.cov() * 252

    # 3. Volatilités individuelles = racine carrée de la diagonale
    vols = np.sqrt(np.diag(cov.values))

    # 4. Poids équipondérés
    n = len(vols)
    w = np.ones(n) / n

    # 5. Numérateur : somme pondérée des volatilités individuelles
    # C'est ce que serait la volatilité si tout était décorrélé
    numerateur = w @ vols #le @ est l'opérateur de produit matriciel en Python (équivalent à np.dot(w, vols))
                          #np.dot(w, vols) calcule le produit scalaire entre les deux vecteurs w et vols, ce qui revient à faire la somme des produits des éléments correspondants.

    # 6. Dénominateur : volatilité réelle du portefeuille
    # w @ cov @ w = variance du portefeuille
    # On prend la racine carrée pour avoir la volatilité
    denominateur = np.sqrt(w @ cov.values @ w) 

    # 7. DR
    dr = numerateur / denominateur

    return dr



print("RATIO DE DIVERSIFICATION (DR)")

print(f"\n{'Période':<42} {'DR':>8}")


resultats = {}
for label, (d1, d2, ret, _spx) in PERIODES.items():
    dr = calcul_dr(ret, d1, d2)
    resultats[label] = dr
    print(f"{label:<42} {dr:>8.4f}")
