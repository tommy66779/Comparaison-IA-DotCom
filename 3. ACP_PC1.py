#Calcul des PC1

import pandas as pd
import numpy as np
from Donnees import ret_dc, ret_ref, ret_ia, PERIODES


def calcul_pc1(ret, d1, d2): #fonction qui calcule le PC1 sur la matrice de corrélation pour les rendements ret entre les dates d1 et d2. Elle retourne la part de variance expliquée par le premier facteur commun (PC1).
    
    #Calcule le PC1 sur la matrice de corrélation.

    #on calcule la matrice des corrélation car
    # Sur les rendements bruts, un titre très volatil comme NVDA
    #dominerait l'analyse juste parce qu'il bouge beaucoup.
    #Sur la matrice de corrélation, chaque titre a le même poids.
    #On mesure uniquement les co-mouvements.
   
    
    # 1. Filtrer la période
    r = ret.loc[d1:d2]

    # 2. Calculer la matrice de corrélation
    # Les NaN sont gérés au cas par cas entre chaque paire de titres
    corr_matrix = r.corr().values #.values convertit les données en un tableau numpy, ce qui est nécessaire pour les calculs de valeurs propres.

    # 3. Extraire les valeurs propres
    # eigvalsh est fait pour les matrices symétriques (comme la matrice de corrélation)
   
    valeurs_propres = np.linalg.eigvalsh(corr_matrix) #np.linalg est un module de numpy qui contient des fonctions pour l'algèbre linéaire.

    # 4. Trier par ordre décroissant
    # eigvalsh retourne par ordre croissant donc on inverse
    valeurs_propres = np.sort(valeurs_propres)[::-1]

    # 5. PC1 = plus grande valeur propre / somme de toutes les valeurs propres
    pc1 = valeurs_propres[0] / valeurs_propres.sum() #a noter que nous pouvons le diviser par n (nombre de titres) car la somme des valeurs propres = n pour une matrice de corrélation.

    return pc1

#calcul sur toute la période


print("PC1 (ANALYSE EN COMPOSANTES PRINCIPALES)")

print(f"\n{'Période':<42} {'PC1 (%)':>8}")


resultats = {}
for label, (d1, d2, ret, _spx) in PERIODES.items(): #même fonctionnement que pour les corrélations
    pc1 = calcul_pc1(ret, d1, d2)
    resultats[label] = pc1
    print(f"{label:<42} {pc1*100:>8.1f}%") #8.1f signifie que le nombre sera affiché avec 8 caractères au total, dont 1 chiffre après la virgule. Le reste sera rempli d'espaces pour aligner correctement les colonnes.

#on remarque que le PC1 réagit comme les corrélation brute dans la mesure où elle est au plus haut lors de la période IA.
