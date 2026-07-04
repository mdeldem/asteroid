# Asteroid Lightcurve

Outils Python pour analyser des mesures CCD d'asteroides :

- lecture des fichiers texte au format `FMT xDVvx`,
- combinaison de plusieurs nuits/fichiers,
- recherche de periode par GLS,
- recherche de periode par serie de Fourier avec choix de l'ordre 1 a 12,
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

Pour afficher seulement un resume des fichiers :

```powershell
asteroid-lc inspect data\*.txt
```

## Notes de modelisation

Les ajustements incluent par defaut un offset de magnitude par fichier. C'est important quand les donnees viennent de plusieurs nuits, instruments, observateurs ou filtres : les decalages photometriques changent le zero point, mais pas la periode de rotation recherchee.
