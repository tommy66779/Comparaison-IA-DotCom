#créer notre graphiques rolling sur 60jours

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # pour sauvegarder sans afficher
import matplotlib.pyplot as plt
from Donnees import ret_dc, ret_ref, ret_ia


def corr_rolling(ret, window=60): #pose les bases de mon rolling window = 60 jours
    
    resultats = []

    for i in range(window, len(ret)):
        # fenêtre glissante les 60 jours avant la date i
        sous_periode = ret.iloc[i-window:i]

       
        corr = sous_periode.corr()

       
        n = corr.shape[0]
        masque = ~np.eye(n, dtype=bool)

        
        corr_moy = np.nanmean(corr.values[masque])

        
        resultats.append((ret.index[i], corr_moy)) # on stocke la date et la valeur

    return pd.DataFrame(resultats, columns=['Date', 'Corr']).set_index('Date')


#calcul des séries rolling
print("Calcul des corrélations rolling...")
print("  Dot-com...", end=" ", flush=True)
roll_dc  = corr_rolling(ret_dc, 60)
print("ok")

print("  Référence...", end=" ", flush=True)
roll_ref = corr_rolling(ret_ref, 60)
print("ok")

print("  IA...", end=" ", flush=True)
roll_ia  = corr_rolling(ret_ia, 60)
print("ok")




print("  STATISTIQUES ROLLING")


for label, roll in [
    ("Dot-com 1995-2002",   roll_dc),
    ("Référence 2003-2018", roll_ref),
    ("IA 2019-2026",        roll_ia),
]:
    idx_max = roll['Corr'].idxmax()
    idx_min = roll['Corr'].idxmin()
    print(f"\n{label}")
    print(f"  Moyenne : {roll['Corr'].mean():.4f}")
    print(f"  Max     : {roll['Corr'].max():.4f} le {idx_max.strftime('%d/%m/%Y')}")
    print(f"  Min     : {roll['Corr'].min():.4f} le {idx_min.strftime('%d/%m/%Y')}")

#création du graphique rolling
fig, axes = plt.subplots(3, 1, figsize=(38.25, 19.5), sharex=False)

# bulle Dot-com
ax = axes[0]
ax.plot(roll_dc.index, roll_dc['Corr'], color='#1a5276', lw=1.2)
ax.fill_between(roll_dc.index, roll_dc['Corr'], alpha=0.12, color='#1a5276')
ax.axhline(roll_dc['Corr'].mean(), color='red', ls='--', lw=2,
           label=f"Moyenne = {roll_dc['Corr'].mean():.3f}")
ax.set_title('Dot-com 1995–2002', fontsize=33, fontweight='bold')
ax.set_ylabel('Corrélation moyenne', fontsize=24, fontweight='bold')
ax.set_xlabel('Date', fontsize=24, fontweight='bold')
ax.tick_params(axis='both', labelsize=18)
ax.legend(fontsize=27)
ax.grid(alpha=0.3)

# Période de référence
ax = axes[1]
ax.plot(roll_ref.index, roll_ref['Corr'], color='#7d6608', lw=1.2)
ax.fill_between(roll_ref.index, roll_ref['Corr'], alpha=0.12, color='#7d6608')
ax.axhline(roll_ref['Corr'].mean(), color='red', ls='--', lw=2,
           label=f"Moyenne = {roll_ref['Corr'].mean():.3f}")
ax.set_title('Période neutre — Survivantes 2003–2018', fontsize=33, fontweight='bold')
ax.set_ylabel('Corrélation moyenne', fontsize=24, fontweight='bold')
ax.set_xlabel('Date', fontsize=24, fontweight='bold')
ax.tick_params(axis='both', labelsize=18)
ax.legend(fontsize=27)
ax.grid(alpha=0.3)

# période IA
ax = axes[2]
ax.plot(roll_ia.index, roll_ia['Corr'], color='#196f3d', lw=1.2)
ax.fill_between(roll_ia.index, roll_ia['Corr'], alpha=0.12, color='#196f3d')
ax.axhline(roll_ia['Corr'].mean(), color='red', ls='--', lw=2,
           label=f"Moyenne = {roll_ia['Corr'].mean():.3f}")
ax.set_title('IA 2019–2026', fontsize=33, fontweight='bold')
ax.set_ylabel('Corrélation moyenne', fontsize=24, fontweight='bold')
ax.set_xlabel('Date', fontsize=24, fontweight='bold')
ax.tick_params(axis='both', labelsize=18)
ax.legend(fontsize=27)
ax.grid(alpha=0.3)

fig.suptitle(
    "Corrélation moyenne rolling 60 jours — Trois périodes",
    fontsize=39, fontweight='bold', y=1.01
)
plt.tight_layout()
plt.savefig('graphique_rolling.png', dpi=150, bbox_inches='tight')
plt.close()

