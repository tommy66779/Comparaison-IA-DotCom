#Donnnees.py correspond au livre de références de données pour les différents calculs du projet.
# Il contient les prix, les rendements, les facteurs Fama-French et les sous-périodes partagées.

import pandas as pd
import numpy as np

#la plupart du temps, j'ai priorisé l'utilisation de pandas pour les calculs et la manipulation des données.
# Pandas offre des meilleures fonctions pour gérer les dates, les Nan et mes séries temporelles.


# PRIX
# _____
dotcom = pd.read_csv('DotCom 1995_2002.csv', sep=';', decimal=',', index_col=0) 
dotcom.index = pd.to_datetime(dotcom.index, format='%m/%d/%Y')  #pour convertir les dates en datetime
dotcom = dotcom.sort_index() #Assurer que les dates sont bien en ordre croissante

ref = pd.read_csv('Référence 2003_2019.csv', sep=';', decimal=',', index_col=0)
ref.index = pd.to_datetime(ref.index, format='%m/%d/%Y')
ref = ref.sort_index()

ia = pd.read_csv('IT-AI 2019_2026.csv', sep=';', decimal=',', index_col=0)
ia.index = pd.to_datetime(ia.index, format='%m/%d/%Y')
ia = ia.sort_index()


# FACTEURS FAMA-FRENCH + MOMENTUM
# _____
ff = pd.read_csv('F-F_Research_Data_Factors_daily.csv', skiprows=4) #me permet de lire les données à partir de la 5ème ligne car le début est du texte.
ff = ff[ff['Unnamed: 0'].astype(str).str.match(r'^\d{8}$')].copy() #filtre pour ne garder que les lignes avec une date valide (8 chiffres)
ff.index = pd.to_datetime(ff['Unnamed: 0'].astype(str), format='%Y%m%d') 
ff = ff[['Mkt-RF', 'SMB', 'HML', 'RF']].astype(float) / 100 #diviser par cent pour obtenir des décimaux et float pour les calculs
ff = ff.sort_index()

mom = pd.read_csv('F-F_Momentum_Factor_daily.csv', skiprows=13)
mom = mom[mom['Unnamed: 0'].astype(str).str.match(r'^\d{8}$')].copy()
mom.index = pd.to_datetime(mom['Unnamed: 0'].astype(str), format='%Y%m%d')
mom = mom[['Mom']].astype(float) / 100 #mom veut dire momentum
mom = mom.sort_index()

factors = ff.join(mom, how='inner') #Je fusionne les deux fichiers pour avoir les 4 facteurs dans une seule base de données
                                    #inner correspond à l'intersection des deux fichiers pour garder que les dates communes.


# SÉPARATION SPX ET TITRES
#_____
spx_dc  = dotcom['S&P500']
spx_ref = ref['SP500']
spx_ia  = ia['SP500']

titres_dc  = dotcom.drop(columns=['S&P500']) #drop sert à retirer la colonne SP500 
titres_ref = ref.drop(columns=['SP500'])
titres_ia  = ia.drop(columns=['SP500'])


# RENDEMENTS JOURNALIERS avec ret = Returns, ou rendements en français
#____
ret_dc  = titres_dc.pct_change().dropna(how='all') #dropna(how='all') supprime les lignes où toutes les valeurs sont NaN
ret_ref = titres_ref.pct_change().dropna(how='all')
ret_ia  = titres_ia.pct_change().dropna(how='all')

ret_spx_dc  = spx_dc.pct_change().dropna() #pas besoin de 'all' car c'est une seule colonne
ret_spx_ref = spx_ref.pct_change().dropna()
ret_spx_ia  = spx_ia.pct_change().dropna()


# Définition des SOUS-PÉRIODES PARTAGÉES
#________________________________________
PERIODES = {
    'P1 - Dot-com complet (1995-2002)':      ('1995-01-01', '2002-12-31', ret_dc,  ret_spx_dc),
    'P2 - Dot-com pré-bulle (1995-1999)':    ('1995-01-01', '1999-12-31', ret_dc,  ret_spx_dc),
    'P3 - Dot-com bulle et crash (2000-2002)':  ('2000-01-01', '2002-12-31', ret_dc,  ret_spx_dc),
    'P4 - Référence (2003-2018)':            ('2003-01-01', '2018-12-31', ret_ref, ret_spx_ref),
    'P5 - Référence post-crise (2010-2018)': ('2010-01-01', '2018-12-31', ret_ref, ret_spx_ref),
    'P6 - IA complet (2019-2026)':           ('2019-01-01', '2026-04-30', ret_ia,  ret_spx_ia),
    'P7 - IA pré-ChatGPT (2019-2022)':       ('2019-01-01', '2022-10-31', ret_ia,  ret_spx_ia),
    'P8 - IA post-ChatGPT (2022-2026)':      ('2022-11-01', '2026-04-30', ret_ia,  ret_spx_ia),
    
}
