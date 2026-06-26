#calcul de carhart


import pandas as pd
import numpy as np
from Donnees import ret_dc, ret_ref, ret_ia, ret_spx_dc, ret_spx_ref, ret_spx_ia, factors, PERIODES


def corr_moyenne(ret, d1, d2): #corrélation brute
    
    r = ret.loc[d1:d2]
    corr_matrix = r.corr()
    n = corr_matrix.shape[0]
    masque = ~np.eye(n, dtype=bool)
    return np.nanmean(corr_matrix.values[masque])


def corr_residuelle_capm(ret, spx, d1, d2): #corrélation CAPM
   
    r = ret.loc[d1:d2]
    s = spx.loc[d1:d2]
    s = s.reindex(r.index).ffill()
    residus = pd.DataFrame(index=r.index)

    for col in r.columns:
        y = r[col].values
        x = s.values
        masque = ~(np.isnan(y) | np.isnan(x))
        y_clean = y[masque]
        x_clean = x[masque]

        if len(y_clean) < 30:
            residus[col] = np.nan
            continue

        beta = np.cov(y_clean, x_clean)[0, 1] / np.var(x_clean)
        alpha = np.mean(y_clean) - beta * np.mean(x_clean)

        residu_complet = np.full(len(y), np.nan)
        residu_complet[masque] = y_clean - (alpha + beta * x_clean)
        residus[col] = residu_complet

    corr_res = residus.corr()
    n = corr_res.shape[0]
    masque_diag = ~np.eye(n, dtype=bool)
    return np.nanmean(corr_res.values[masque_diag])



def corr_residuelle_carhart(ret, d1, d2): #on calcule la corrélation résiduelle post-Carhart (4 facteurs)

    
    #1. Filtrer la période pour les titres et les facteurs
    #2. Pour chaque titre, calculer l'excès de rendement (ret - RF)
    #3. Régresser cet excès sur les 4 facteurs par OLS
    #4. Récupérer les résidus
    #5. Calculer la corrélation moyenne entre tous les résidus

  
    r = ret.loc[d1:d2]

    # Aligner les facteurs sur la même période
    # On limite aussi à avril 2026 car c'est la fin des données auquel j'ai accès à la date de création de ce code
    d2_ff = min(d2, '2026-04-30')
    fac = factors.loc[d1:d2_ff]

    # Aligner les titres sur les dates des facteurs
    r_align = r.reindex(fac.index)

    mkt = fac['Mkt-RF'].values
    smb = fac['SMB'].values
    hml = fac['HML'].values
    wml = fac['Mom'].values
    rf  = fac['RF'].values

    # 2, 3 et 4. Régression OLS sur 4 facteurs pour chaque titre
    residus = pd.DataFrame(index=fac.index) #reindex permet de créer une base vide avec les mêmes index que les facteurs, pour stocker les résidus de chaque titre après la régression sur les 4 facteurs.

    for col in r_align.columns:
        y_brut = r_align[col].values

        # Excès de rendement = rendement du titre - taux sans risque
        y = y_brut - rf

        # Garder uniquement les observations complètes
        masque = ~np.isnan(y)
        y_clean = y[masque]
        mkt_c = mkt[masque]
        smb_c = smb[masque]
        hml_c = hml[masque]
        wml_c = wml[masque]
        rf_c  = rf[masque]

        if len(y_clean) < 30:
            residus[col] = np.nan
            continue

        # Matrice X avec constante + 4 facteurs
        X = np.column_stack([
            np.ones(len(y_clean)),  # constante alpha
            mkt_c,                  # facteur marché
            smb_c,                  # facteur taille
            hml_c,                  # facteur valeur
            wml_c                   # facteur momentum
        ])

    
        beta = np.linalg.lstsq(X, y_clean, rcond=None)[0] #leastsquares me permet de tout calculer en une seule fois, sans avoir à calculer la covariance et la variance séparément. Cela me donne directement les coefficients de régression (alpha et les betas pour chaque facteur).

        # Résidu = excès de rendement observé - excès prédit
        residu_complet = np.full(len(y), np.nan)
        residu_complet[masque] = y_clean - X @ beta
        residus[col] = residu_complet

    # 5. Corrélation moyenne entre les résidus
    corr_res = residus.corr()
    n = corr_res.shape[0]
    masque_diag = ~np.eye(n, dtype=bool)
    corr_moy = np.nanmean(corr_res.values[masque_diag])

    # PC1 sur les résidus
    vp = np.linalg.eigvalsh(corr_res.values)
    vp = np.sort(vp)[::-1]
    pc1_res = vp[0] / vp.sum()

    return corr_moy, pc1_res

# 

print("TABLEAU DE LA CORRÉLATION RÉSIDUELLE")

print(
    f"{'Période':<42} {'Brute':>8} {'CAPM':>8} {'Carhart':>8} {'Delta':>8} {'Diff %':>10}"
)
print("-" * 100)

resultats = {}
for label, (d1, d2, ret, spx) in PERIODES.items():
    corr_brute = corr_moyenne(ret, d1, d2)
    corr_capm = corr_residuelle_capm(ret, spx, d1, d2)
    corr_carhart, _ = corr_residuelle_carhart(ret, d1, d2)
    delta = corr_capm - corr_carhart
    diff_pct = np.nan if corr_capm == 0 else 100 * delta / corr_capm
    resultats[label] = {
        'brute': corr_brute,
        'capm': corr_capm,
        'carhart': corr_carhart,
        'delta': delta,
        'diff_pct': diff_pct,
    }
    print(
        f"{label:<42} {corr_brute:>8.4f} {corr_capm:>8.4f} {corr_carhart:>8.4f} {delta:>8.4f} {diff_pct:>10.1f}%"
    )

