#Ce fichier calcule la corrélation moyenne de Pearson sur les différentes périodes définies dans Donnees.py. Il sert à vérifier que les données sont bien chargées et que les rendements sont bien calculés.

import pandas as pd
import numpy as np
from Donnees import (
    titres_dc,
    titres_ref,
    titres_ia,
    spx_dc,
    spx_ref,
    spx_ia,
    ret_dc,
    ret_ref,
    ret_ia,
    ret_spx_dc,
    ret_spx_ref,
    ret_spx_ia,
    PERIODES,
)


#Fonction de calcul de la corrélation moyenne de Pearson, entre les périodes d1 et d2, pour les rendements ret. Elle retourne la corrélation moyenne, le nombre d'observations et le nombre de titres.
def corr_moyenne(ret, d1, d2):
    

    # On filtre la période avec .loc, .loc sert à sélectionner les lignes et les colonnes d'une base de données en utilisant des étiquettes (labels) plutôt que des indices numériques. Ici, on sélectionne toutes les lignes entre d1 et d2 (inclus) pour toutes les colonnes.
    #.corr() calcule toutes les corrélations pairwise
    # On exclut la diagonale (corrélation d'un titre avec lui-même = 1)
    # On prend la moyenne des valeurs restantes
    
    # Filtrer la période
    r = ret.loc[d1:d2]

    # Calculer la matrice de corrélation
    # pandas utilise automatiquement les observations communes
    # pour chaque paire de titres (gestion des NaN au cas par cas)
    corr_matrix = r.corr()

    # Créer un masque pour exclure la diagonale
    n = corr_matrix.shape[0]
    masque = ~np.eye(n, dtype=bool) #masque est une matrice booléenne de la même taille que corr_matrix, où les éléments de la diagonale sont False et tous les autres éléments sont True. Cela permet d'exclure la diagonale lors du calcul de la moyenne.

    # Moyenne des valeurs hors diagonale
    corr_moy = np.nanmean(corr_matrix.values[masque]) # np.nanmean calcule la moyenne en ignorant les NaN. corr_matrix.values[masque] sélectionne uniquement les éléments de corr_matrix qui ne sont pas sur la diagonale.

    return corr_moy, len(r), corr_matrix.shape[0] #renvoie la corrélation moyenne, le nombre d'observations (lignes) et le nombre de titres (colonnes) pour la période donnée.



print("CORRÉLATION MOYENNE DE PEARSON")

print(f"\n{'Période':<42} {'ρ̄ Pearson':>10} {'Obs':>6} {'Titres':>7}") #sert à afficher les en-têtes de colonnes pour le tableau des résultats. <42 signifie que la chaîne "Période" sera alignée à gauche et occupera 42 caractères, >10 signifie que "ρ̄ Pearson" sera aligné à droite et occupera 10 caractères, >6 signifie que "Obs" sera aligné à droite et occupera 6 caractères, et >7 signifie que "Titres" sera aligné à droite et occupera 7 caractères.


resultats = {} #affiche un dictionnaire vide pour stocker les résultats de la corrélation moyenne pour chaque période. Les clés seront les étiquettes des périodes et les valeurs seront les corrélations moyennes correspondantes.
for label, (d1, d2, ret, spx) in PERIODES.items(): #.items() renvoie une vue des paires clé-valeur du dictionnaire PERIODES. Chaque élément de cette vue est un tuple (clé, valeur), où la clé est le label de la période et la valeur est un tuple contenant les dates de début et de fin, les rendements et l'indice S&P 500 pour cette période.
    corr, n_obs, n_titres = corr_moyenne(ret, d1, d2) 
    resultats[label] = corr #resultats[label] = corr stocke la corrélation moyenne calculée pour la période actuelle dans le dictionnaire resultats, en utilisant le label de la période comme clé.
    print(f"{label:<42} {corr:>10.4f} {n_obs:>6} {n_titres:>7}")
#je trouve d1 grace à la fonction PERIODES
#nous trouvons que la corrélation moyenne est souvent plus élevée pendant la période IA
#les corrélations brutes restent tous similaires à la période de référence

