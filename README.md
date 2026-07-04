# Asteroid Lightcurve

Outils Python pour analyser des mesures CCD d'asteroides :

- lecture des fichiers texte au format `FMT xDVvx`,
- combinaison de plusieurs nuits/fichiers,
- recherche de periode par GLS,
- recherche du couple periode / ordre de Fourier par BIC,
- production de periodogrammes, courbes repliees en phase et residus.

## Installation

```powershell
uv sync
```

ou, avec un environnement Python classique :

```powershell
python -m pip install -e .
```

## Recherche de periode

```powershell
asteroid-lc search data\*.txt --min-period 0.083333 --max-period 0.833333 --out output
```

Pour la recherche, `--min-period`, `--max-period` et `--period` sont en jours. Les resultats et graphiques affichent la periode en heures et en jours. Les dates des mesures sont converties vers le milieu de pose quand la ligne `POS` fournit un temps de pose. Le premier champ de `POS` indique la position temporelle du JD : `-1` debut de pose, `0` milieu de pose, `1` fin de pose.

Si la periode est deja connue, on peut la fournir en jours et sauter toute la recherche de periode :

```powershell
asteroid-lc search data\*.txt --period 0.2106178 --out output
```

Dans ce mode, le programme ajuste seulement le modele Fourier a cette periode imposee, puis produit les courbes repliees et les residus.

Par defaut, le programme interroge aussi JPL Horizons pour produire `ephemeris_by_file.csv`. Pour chaque fichier de mesures, il prend le milieu de l'intervalle d'observation, puis recupere la position geocentrique RA/DEC ICRF de l'asteroide a cette date. Ces positions sont ensuite utilisees pour convertir les dates `JD` en `HJD` geocentrique, et toute l'analyse de periode est faite sur les `HJD`. Le fichier `residuals.csv` conserve `jd_utc`, `hjd_utc` et `hjd_correction_days`.

Pour travailler hors-ligne ou rester en `JD` non heliocentrique :

```powershell
asteroid-lc search data\*.txt --period 0.2106178 --no-ephemeris --out output
```

Options utiles :

```powershell
asteroid-lc search data\*.txt --min-period 0.083333 --max-period 0.833333 --orders 1:12 --samples 12000 --out output
```

La recherche automatique suit une strategie robuste pour les courbes double-pic :

```text
GLS -> meilleurs pics -> test de P/2, P et 2P -> ordres Fourier -> choix du BIC minimal
```

Par defaut, les 20 meilleurs pics GLS sont testes avec les multiplicateurs `0.5,1,2`. On peut ajuster ces parametres :

```powershell
asteroid-lc search data\*.txt --min-period 0.083333 --max-period 0.833333 --gls-candidates 30 --gls-multipliers 0.5,1,2 --orders 2:8 --out output
```

Le fichier `period_order_candidates.csv` liste tous les couples periode/ordre testes, avec la periode GLS d'origine, le multiplicateur, le chi2, l'AIC et le BIC. Il permet de verifier pourquoi une periode double de la meilleure periode GLS peut etre retenue.

## Cas des asteroides binaires

Pour rechercher une periode de rotation plus stable quand des eclipses, occultations ou signatures binaires creent de forts residus, on peut activer une deuxieme passe :

```powershell
asteroid-lc search data\*.txt --min-period 0.083333 --max-period 0.833333 --binary-filter --out output
```

Le programme ajuste d'abord le meilleur modele, mesure les residus, rejette les points trop ecartes, puis relance la recherche de periode sur les points conserves. Par defaut, le seuil est robuste :

```text
|residu - mediane(residus)| > 3.5 x 1.4826 x MAD(residus)
```

Ce seuil s'adapte au bruit de la courbe au lieu d'imposer une valeur fixe en magnitude. On peut le modifier, ou imposer un seuil absolu :

```powershell
asteroid-lc search data\*.txt --min-period 0.083333 --max-period 0.833333 --binary-filter --binary-filter-sigma 4.0 --out output
asteroid-lc search data\*.txt --min-period 0.083333 --max-period 0.833333 --binary-filter --binary-filter-threshold-mag 0.08 --out output
```

Par securite, le filtrage ne rejette pas plus de 25 % des points par defaut (`--binary-filter-max-reject-fraction`) et conserve au moins 30 mesures (`--binary-filter-min-points`). Le graphe `binary_filter_rejected_points.png` montre les points rejetes et les lignes de seuil. Les sorties de cette deuxieme passe sont separees avec le prefixe `binary_filtered_`, par exemple `binary_filtered_folded_lightcurve.png`, `binary_filtered_residuals.csv` et `binary_filtered_period_summary.csv`. Le fichier `binary_filter_summary.csv` resume le seuil et le nombre de points rejetes.

Pour afficher seulement un resume des fichiers :

```powershell
asteroid-lc inspect data\*.txt
```

## Notes de modelisation

Les ajustements incluent par defaut un offset de magnitude par fichier. C'est important quand les donnees viennent de plusieurs nuits, instruments, observateurs ou filtres : les decalages photometriques changent le zero point, mais pas la periode de rotation recherchee.
