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
asteroid-lc search data\*.txt --min-period 2 --max-period 20 --out output
```

Les periodes sont en heures. Les dates des mesures sont converties du debut de pose vers le milieu de pose quand la ligne `POS` fournit un temps de pose.

Options utiles :

```powershell
asteroid-lc search data\*.txt --orders 1:12 --samples 12000 --top 10 --out output
```

Pour afficher seulement un resume des fichiers :

```powershell
asteroid-lc inspect data\*.txt
```

## Notes de modelisation

Les ajustements incluent par defaut un offset de magnitude par fichier. C'est important quand les donnees viennent de plusieurs nuits, instruments, observateurs ou filtres : les decalages photometriques changent le zero point, mais pas la periode de rotation recherchee.

