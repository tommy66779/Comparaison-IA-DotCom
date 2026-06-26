#ce fichier calcule les rendements, il sert à vérifier que les données sont bien chargées et que les rendements sont bien calculés.
import pandas as pd
import numpy as np
# Donnees import contient toutes les données nécessaires pour les calculs préalables, y compris les périodes.
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
)



print("RENDEMENTS JOURNALIERS")


for label, ret, spx in [ #sert a comparer les rendements des différentes périodes et des différents indices
    ("DOT-COM 1995-2002",   ret_dc,  ret_spx_dc),
    ("RÉFÉRENCE 2003-2019", ret_ref, ret_spx_ref),
    ("IT-AI 2019-2026",     ret_ia,  ret_spx_ia),
]:
    print(f"\n--- {label} ---") #sert à afficher le label de la période pour laquelle on calcule les rendements
    print(f"  Période         : {ret.index[0].date()} → {ret.index[-1].date()}") #sert à afficher la période pour laquelle on calcule les rendements
    print(f"  Nb observations : {len(ret)}")
    print(f"  Rendement moy.  : {ret.mean().mean()*100:.4f}% par jour") #.4f sert à afficher 4 chiffres après la virgule
    print(f"  Volatilité moy. : {ret.std().mean()*100:.4f}% par jour")
    print(f"  SPX rendement   : {spx.mean()*100:.4f}% par jour")


    
