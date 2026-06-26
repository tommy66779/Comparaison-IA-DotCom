#j'ai trouvé que les corrélations résiduelles post carhart 4F était tres significantes, donc j'ai voulu voir si le facteur momentum (WML) était vraiment important. Pour cela, j'ai comparé les corrélations résiduelles post-Carhart 4F avec celles post-Carhart 3F (sans WML). J'ai aussi calculé la différence entre les deux et le pourcentage de la corrélation post-CAPM qui est absorbée par l'ajout du facteur WML. 

import pandas as pd
import numpy as np
from Donnees import ret_dc, ret_ref, ret_ia, ret_spx_dc, ret_spx_ref, ret_spx_ia, factors, PERIODES


def corr_residuelle_capm(ret, spx, d1, d2):
    
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


def corr_residuelle_carhart_3f(ret, d1, d2):  #Modèle : Rᵢ,t − RF,t = αᵢ + β₁ MKT,t + β₂ SMB,t + β₃ HML,t + εᵢ,t

    r = ret.loc[d1:d2]

    d2_ff = min(d2, '2026-04-30')
    fac = factors.loc[d1:d2_ff]

    r_align = r.reindex(fac.index)

    mkt = fac['Mkt-RF'].values
    smb = fac['SMB'].values
    hml = fac['HML'].values
    rf  = fac['RF'].values

    residus = pd.DataFrame(index=fac.index)

    for col in r_align.columns:
        y_brut = r_align[col].values
        y = y_brut - rf

        masque = ~np.isnan(y)
        y_clean = y[masque]
        mkt_c = mkt[masque]
        smb_c = smb[masque]
        hml_c = hml[masque]

        if len(y_clean) < 30:
            residus[col] = np.nan
            continue

        X = np.column_stack([
            np.ones(len(y_clean)),  
            mkt_c,                  
            smb_c,                  
            hml_c                   
        ])

        beta = np.linalg.lstsq(X, y_clean, rcond=None)[0]

        # Résidu
        residu_complet = np.full(len(y), np.nan)
        residu_complet[masque] = y_clean - X @ beta
        residus[col] = residu_complet

    corr_res = residus.corr()
    n = corr_res.shape[0]
    masque_diag = ~np.eye(n, dtype=bool)
    corr_moy = np.nanmean(corr_res.values[masque_diag])

    vp = np.linalg.eigvalsh(corr_res.values)
    vp = np.sort(vp)[::-1]
    pc1_res = vp[0] / vp.sum()

    return corr_moy, pc1_res


def corr_residuelle_carhart_4f(ret, d1, d2):
    #copie de la fonction Carhart 
    r = ret.loc[d1:d2]
    d2_ff = min(d2, '2026-04-30')
    fac = factors.loc[d1:d2_ff]
    r_align = r.reindex(fac.index)

    mkt = fac['Mkt-RF'].values
    smb = fac['SMB'].values
    hml = fac['HML'].values
    wml = fac['Mom'].values
    rf  = fac['RF'].values

    residus = pd.DataFrame(index=fac.index)

    for col in r_align.columns:
        y_brut = r_align[col].values
        y = y_brut - rf

        masque = ~np.isnan(y)
        y_clean = y[masque]
        mkt_c = mkt[masque]
        smb_c = smb[masque]
        hml_c = hml[masque]
        wml_c = wml[masque]

        if len(y_clean) < 30:
            residus[col] = np.nan
            continue

        X = np.column_stack([
            np.ones(len(y_clean)),
            mkt_c,
            smb_c,
            hml_c,
            wml_c
        ])

        beta = np.linalg.lstsq(X, y_clean, rcond=None)[0]
        residu_complet = np.full(len(y), np.nan)
        residu_complet[masque] = y_clean - X @ beta
        residus[col] = residu_complet

    corr_res = residus.corr()
    n = corr_res.shape[0]
    masque_diag = ~np.eye(n, dtype=bool)
    corr_moy = np.nanmean(corr_res.values[masque_diag])

    vp = np.linalg.eigvalsh(corr_res.values)
    vp = np.sort(vp)[::-1]
    pc1_res = vp[0] / vp.sum()

    return corr_moy, pc1_res



print("ANALYSE DE LA CONTRIBUTION DU MOMENTUM (WML)")

print()

resultats = []

for label, (d1, d2, ret, spx) in PERIODES.items():
    # Déterminer quel spx utiliser selon la période
    if 'Dot-com' in label:
        spx_to_use = ret_spx_dc
    elif 'Référence' in label:
        spx_to_use = ret_spx_ref
    else:  # IA
        spx_to_use = ret_spx_ia

    corr_capm = corr_residuelle_capm(ret, spx_to_use, d1, d2)
    corr_3f, _ = corr_residuelle_carhart_3f(ret, d1, d2)
    corr_4f, _ = corr_residuelle_carhart_4f(ret, d1, d2)

    delta_wml = corr_3f - corr_4f
    pct_absorbe = (delta_wml / corr_capm * 100) if corr_capm != 0 else 0

    resultats.append({
        'Période': label,
        'ρ post-CAPM': corr_capm,
        'ρ post-FF3': corr_3f,
        'ρ post-Carhart': corr_4f,
        'Δ FF3→Carhart': delta_wml,
        '% absorbé': pct_absorbe
    })

# Affichage table formatée
print(f"{'Période':<35} {'ρ post-CAPM':>12} {'ρ post-FF3':>12} {'ρ post-Carh':>12} {'Δ FF3→Cart':>12} {'% absorbé':>12}")
print("-" * 100)

for row in resultats:
    print(f"{row['Période']:<35} "
          f"{row['ρ post-CAPM']:>12.4f} "
          f"{row['ρ post-FF3']:>12.4f} "
          f"{row['ρ post-Carhart']:>12.4f} "
          f"{row['Δ FF3→Carhart']:>12.4f} "
          f"{row['% absorbé']:>11.2f}%")

print()
