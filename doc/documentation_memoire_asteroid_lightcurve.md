# Mémoire technique du projet `asteroid-lightcurve`

Documentation destinée à un agent de codage chargé d'étendre le projet.

---

## 1. Objectif général du projet

Le projet `asteroid-lightcurve` est une application Python en ligne de commande destinée à analyser des mesures photométriques CCD d'astéroïdes afin d'estimer leur période de rotation. Le cas scientifique visé est celui des courbes de lumière d'astéroïdes, souvent non sinusoïdales et fréquemment **double-pic** sur une rotation complète : deux maxima et deux minima apparaissent parce qu'un astéroïde irrégulier présente deux grandes sections apparentes par tour.

L'application sait actuellement :

- lire des fichiers texte de mesures au format `FMT xDVvx` ;
- combiner plusieurs fichiers ou nuits d'observation ;
- corriger les dates vers le milieu de pose ;
- interroger JPL Horizons pour obtenir RA/DEC, distance héliocentrique `r` et distance observateur `Δ` ;
- convertir les dates JD UTC en HJD géocentriques via une correction héliocentrique ;
- appliquer une correction géométrique obligatoire de magnitude `5 log10(rΔ)` avant l'ajustement ;
- mettre en cache par fichier les corrections indépendantes du fit ;
- rechercher une période par une stratégie hybride GLS + Fourier ;
- tester explicitement les ambiguïtés `P/2`, `P` et `2P` issues des courbes double-pic ;
- ajuster une série de Fourier pondérée par les erreurs photométriques ;
- choisir un couple période / ordre de Fourier avec une stratégie prudente contre le sur-ajustement ;
- estimer une incertitude de période par profil de `χ²` ;
- produire des graphiques de périodogramme, de courbe repliée, de résidus et de filtrage robuste ;
- produire des fichiers CSV et JSON de synthèse.

Le point d'entrée CLI défini dans `pyproject.toml` est :

```bash
asteroid-lc = asteroid_lightcurve.cli:main
```

Le package nécessite Python `>=3.9` et dépend de `numpy`, `scipy`, `matplotlib` et `astropy`.

---

## 2. Structure du dépôt

```text
asteroid-lightcurve/
├── README.md
├── pyproject.toml
├── uv.lock
└── src/
    └── asteroid_lightcurve/
        ├── __init__.py
        ├── cli.py
        ├── ephemeris.py
        ├── models.py
        ├── parser.py
        ├── period.py
        └── plotting.py
```

### Rôle des modules

| Module | Rôle principal |
|---|---|
| `models.py` | Structures de données : fichier d'observation, courbe combinée, sous-échantillonnage. |
| `parser.py` | Lecture des fichiers `FMT`, expansion des motifs glob, construction d'une courbe multi-fichiers. |
| `ephemeris.py` | Requête JPL Horizons, extraction RA/DEC/r/Δ, correction HJD, correction géométrique et cache par fichier. |
| `period.py` | Cœur algorithmique : grille de périodes, GLS simplifié, ajustement Fourier pondéré, critères AIC/AICc/BIC, sélection stable, incertitude. |
| `plotting.py` | Production des figures PNG : périodogrammes, courbes repliées, résidus, filtrage. |
| `cli.py` | Interface utilisateur, orchestration complète, génération des fichiers de sortie. |

---

## 3. Modèle de données

### 3.1 `ObservationFile`

Un `ObservationFile` représente un fichier source individuel.

Attributs principaux :

```python
path: Path
fmt: str
keywords: dict[str, list[str]]
jd: np.ndarray
jd_decimals: np.ndarray
magnitude: np.ndarray
mag_error: np.ndarray
ignored: list[list[str]]
```

Les mots-clés reconnus sont :

```python
KEYWORDS = {"FMT", "NOM", "MES", "POS", "FIL", "CAT", "TEL", "CAP"}
```

Propriétés utiles :

- `object_name` : reconstruit le champ `NOM` ;
- `observer` : reconstruit le champ `MES` ;
- `observer_name` : partie de `MES` avant `@` ;
- `filter_name` : premier élément du champ `FIL` ;
- `exposure_seconds` : durée de pose extraite de `POS` ;
- `exposure_time_position` : position temporelle du JD dans la pose :
  - `-1` = début de pose ;
  - `0` = milieu de pose ;
  - `1` = fin de pose.

La méthode `mid_exposure_jd()` corrige les JD si le fichier indique que le temps est donné au début ou à la fin de la pose :

\[
JD_\mathrm{mid} = JD_\mathrm{start} + \frac{t_\mathrm{exp}}{2 \times 86400}
\]

ou

\[
JD_\mathrm{mid} = JD_\mathrm{end} - \frac{t_\mathrm{exp}}{2 \times 86400}
\]

Si `POS` indique déjà le milieu de pose, les JD sont conservés.

### 3.2 `LightCurve`

Une `LightCurve` représente la combinaison de plusieurs fichiers :

```python
jd: np.ndarray
jd_decimals: np.ndarray
magnitude: np.ndarray
mag_error: np.ndarray
group: np.ndarray
group_names: list[str]
files: list[ObservationFile]
time_label: str = "JD"
mag_observed: np.ndarray | None = None
mag_reduced: np.ndarray | None = None
geometry_correction: np.ndarray | None = None
r_au: np.ndarray | None = None
delta_au: np.ndarray | None = None
jd_utc: np.ndarray | None = None
light_time_correction_days: np.ndarray | None = None
```

Le champ `group` associe chaque mesure à son fichier d'origine. C'est essentiel parce que le modèle Fourier inclut un **offset de magnitude par fichier**, afin de corriger les décalages de zéro photométrique entre nuits, observateurs, instruments ou filtres.

Après la lecture initiale, `magnitude` et `mag_observed` valent les magnitudes mesurées. Après correction géométrique, `magnitude` est remplacée par `mag_reduced` et devient la colonne utilisée par `weighted_fit()`. La magnitude observée originale reste disponible dans `mag_observed`.

Les champs `r_au`, `delta_au`, `geometry_correction`, `jd_utc` et `light_time_correction_days` sont des corrections indépendantes du fit global. Ils peuvent être sauvegardés dans le cache par fichier. En revanche les offsets `Zi`, les magnitudes alignées, les résidus et le modèle Fourier ne doivent pas être stockés dans ce cache.

Propriétés :

- `n_points` : nombre de mesures ;
- `baseline_days` : durée totale couverte par les observations.

### 3.3 Sous-échantillonnage

La fonction `subset_lightcurve(curve, mask)` construit une courbe filtrée en conservant seulement les points où `mask=True`.

Elle recompacte les groupes afin que les fichiers conservés aient des identifiants consécutifs `0, 1, 2, ...`. Cette fonction est utilisée par le filtrage robuste des résidus et conserve aussi les champs optionnels de correction.

---

## 4. Lecture des fichiers photométriques

### 4.1 Format attendu

L'application attend des fichiers texte structurés ainsi :

```text
FMT xDVvx
NOM (3669) VERTINSKIJ
MES Observateur@email
POS 0 120.00
FIL C
...
001 2460000.123456 14.123 0.015 14.567
```

Les lignes vides et les lignes commençant par `;` sont ignorées.

Le champ `FMT` décrit la signification des colonnes de mesure. Les marqueurs actuellement exploités sont :

| Marqueur | Signification |
|---|---|
| `D` | Date en jour julien. |
| `V` | Magnitude mesurée de l'objet. |
| `v` | Erreur de magnitude. |
| `x` | Champ ignoré. |

Exemple `FMT xDVvx` :

| Colonne | Marqueur | Usage |
|---|---:|---|
| 1 | `x` | ignorée, typiquement numéro de séquence. |
| 2 | `D` | JD. |
| 3 | `V` | magnitude astéroïde. |
| 4 | `v` | erreur de magnitude. |
| 5 | `x` | ignorée, typiquement magnitude de contrôle. |

### 4.2 Encodage

La fonction `read_observation_file()` utilise par défaut :

```python
encoding="mbcs"
errors="replace"
```

`mbcs` est un encodage spécifique Windows. Cela correspond au contexte de fichiers ANSI/Windows déjà utilisé dans le projet. Point important pour Codex : sous Linux ou macOS, `mbcs` peut ne pas être disponible. Si le projet doit devenir portable, il faudra prévoir une stratégie `cp1252` / `latin-1` / `utf-8-sig` ou un paramètre CLI `--encoding`.

### 4.3 Tri temporel

`read_lightcurve()` lit tous les fichiers, concatène les tableaux, puis trie toutes les mesures par JD croissant. Les groupes restent associés aux fichiers d'origine.

---

## 5. Éphémérides, correction temporelle et correction géométrique

### 5.1 Requête JPL Horizons

Par défaut, la commande `search` interroge JPL Horizons, sauf si `--no-ephemeris` est fourni. Cette étape sert à deux corrections distinctes :

1. correction temporelle JD UTC vers HJD UTC ;
2. correction géométrique des magnitudes par les distances Soleil-astéroïde et observateur-astéroïde.

L'identifiant de l'astéroïde est extrait du champ `NOM` avec l'expression régulière :

```python
r"\((\d+)\)"
```

Donc le format attendu est par exemple :

```text
NOM (3669) VERTINSKIJ
```

La commande Horizons envoyée devient :

```text
3669;
```

Pour chaque fichier, l'application calcule le JD moyen :

\[
JD_\mathrm{mid,file} = \frac{\min(JD_i) + \max(JD_i)}{2}
\]

Puis elle demande à Horizons la position géocentrique et les distances de l'objet avec :

- `EPHEM_TYPE=OBSERVER` ;
- `CENTER=500@399`, c'est-à-dire géocentre terrestre ;
- `QUANTITIES='1,19,20'`, position astrométrique RA/DEC, distances héliocentrique et observateur ;
- `ANG_FORMAT=DEG` ;
- `CSV_FORMAT=YES`.

Le code évite une requête par point. Pour chaque fichier d'observation, il interroge Horizons au début et à la fin du fichier, puis interpole linéairement `r_au` et `delta_au` pour chaque mesure. Sur des fichiers de quelques heures, cette interpolation est suffisante pour l'objectif du projet.

### 5.2 Extraction RA/DEC/r/Δ

`parse_horizons_geometry()` extrait la table située entre `$$SOE` et `$$EOE`, parcourt les lignes, sépare sur les virgules, puis retourne pour chaque époque :

- le JD Horizons ;
- RA et DEC en degrés ;
- `r_au`, distance Soleil-astéroïde en unités astronomiques ;
- `delta_au`, distance observateur-astéroïde en unités astronomiques.

### 5.3 Correction héliocentrique

La correction HJD est calculée avec `astropy` :

```python
Time(..., format="jd", scale="utc", location=geocenter)
SkyCoord(ra=ra_deg, dec=dec_deg, frame="icrs")
times.light_travel_time(target, kind="heliocentric")
```

Puis :

\[
HJD_\mathrm{UTC} = JD_\mathrm{UTC} + \Delta t_\mathrm{helio}
\]

L'objet `LightCurve` est modifié en place : `curve.jd` devient `hjd_utc`, et `curve.time_label` devient `"HJD"`.

Les champs `jd_utc`, `hjd_utc` et `light_time_correction_days` sont conservés dans les sorties point par point et dans le cache de correction.

### 5.4 Correction géométrique de magnitude

Pour chaque mesure, le programme calcule :

\[
g_i = 5\log_{10}(r_i\Delta_i)
\]

puis :

\[
m_{\mathrm{reduced},i} = m_{\mathrm{obs},i} - g_i
\]

Cette correction est obligatoire en mode normal. Elle supprime la dérive lente due aux variations de distances pendant la campagne, sans chercher à estimer une magnitude absolue `H`.

Point important : `mag_reduced` peut être très différente de `mag_obs`. Par exemple une magnitude observée autour de `11.7` peut devenir environ `7.8`. C'est normal physiquement. Les graphes finaux ne sont toutefois pas centrés sur cette valeur brute : ils sont recentrés globalement autour de la magnitude observée.

### 5.5 Cache de correction par fichier

Pour éviter de refaire les mêmes requêtes Horizons, `ephemeris.py` écrit un cache dans un sous-dossier `cache` du répertoire de données :

```text
data/
  obs_001.txt
  cache/
    obs_001.<hash>.correction-cache.csv
    obs_001.<hash>.correction-cache.json
```

Le hash court est calculé à partir du chemin absolu normalisé, afin d'éviter les collisions entre fichiers de même nom dans des dossiers différents.

Le CSV de cache contient uniquement des valeurs indépendantes du fit :

```text
jd_utc
hjd_utc
light_time_correction_days
mag_obs
mag_error
r_au
delta_au
geometry_correction
mag_reduced
```

Le JSON adjacent contient les métadonnées de validation :

- fichier source ;
- chemin absolu ;
- taille ;
- `mtime_ns` ;
- commande Horizons de l'objet ;
- centre observateur ;
- version de correction.

Le cache est invalidé si le fichier source change, si l'objet change ou si `correction_version` change.

Le cache ne doit jamais contenir :

- `Zi` ;
- `block_offset` ;
- `mag_aligned` ;
- `mag_plot` ;
- résidus ;
- modèle Fourier ;
- période ou ordre Fourier.

Ces valeurs dépendent du fit global et peuvent changer si l'utilisateur modifie les fichiers inclus, les ordres Fourier, les bornes de période, les pondérations ou le filtrage.

### 5.6 Commande de purge du cache

La commande dédiée est :

```bash
asteroid-lc clear-cache data/
```

Avec :

```bash
asteroid-lc clear-cache data/ --dry-run
```

elle compte les fichiers qui seraient supprimés sans toucher au cache. La commande ne supprime jamais les fichiers source d'observation.

### 5.7 Limites scientifiques de cette correction

Le calcul est une correction héliocentrique en direction de la position RA/DEC de l'astéroïde. Or un astéroïde se déplace pendant la campagne. Le projet approxime sa position par **une RA/DEC par fichier**, évaluée au milieu de l'intervalle du fichier. C'est généralement raisonnable pour des fichiers couvrant une nuit courte, mais ce n'est pas une correction topocentrique complète observatoire par observatoire.

Pour une évolution future, Codex pourrait ajouter :

- une correction topocentrique si les coordonnées de l'observatoire sont connues ;
- une RA/DEC interpolée pour chaque point au lieu d'une RA/DEC constante par fichier ;
- une option BJD_TDB pour des analyses plus précises.

Le projet n'implémente volontairement pas de modèle de phase `H,G`, `H,G1,G2`, `H,G12`, de coefficient `β`, de magnitude absolue `H`, ni de combinaison multi-opposition. L'objectif reste la période synodique issue des données fournies.

---

## 6. Modèle photométrique ajusté

### 6.1 Modèle Fourier

Le cœur du modèle est une série de Fourier tronquée à l'ordre `M`, ajustée sur les magnitudes réduites :

\[
m_{\mathrm{reduced},i} = c_0 + Z_{g_i} + \sum_{k=1}^{M}\left[a_k \cos\left(2\pi k \frac{t_i-t_0}{P}\right) + b_k \sin\left(2\pi k \frac{t_i-t_0}{P}\right)\right] + \epsilon_i
\]

où :

- `m_reduced,i` est la magnitude observée corrigée de `5 log10(rΔ)` ;
- `P` est la période testée ;
- `M` est l'ordre de Fourier ;
- `c0` est l'offset global ;
- `Z_g` est l'offset photométrique du fichier/groupe `g` ;
- le groupe `0` sert de référence, donc il n'a pas de colonne offset dédiée ;
- `a_k` et `b_k` sont les coefficients harmoniques.

Dans le code, `t0` est implicitement :

\[
t_0 = \min(t_i)
\]

Le choix de `t0=min(jd)` n'affecte pas la qualité de l'ajustement pour une période donnée ; il change seulement la phase des coefficients sinus/cosinus.

### 6.2 Matrice de conception

La fonction `design_matrix()` construit les colonnes :

1. constante globale ;
2. offsets de groupes pour les groupes `1..n_groups-1` ;
3. colonnes cosinus et sinus pour chaque harmonique `k=1..M`.

Nombre de paramètres libres :

\[
K = 1 + (N_\mathrm{groups}-1) + 2M
\]

si les offsets par groupe sont activés.

### 6.3 Ajustement pondéré

L'ajustement est fait par moindres carrés pondérés. Les poids sont :

\[
w_i = \frac{1}{\sigma_i^2}
\]

Le code multiplie la matrice et le vecteur par \(\sqrt{w_i}\), puis résout :

\[
\min_\beta \left\| W^{1/2}(X\beta-y) \right\|^2
\]

avec `scipy.linalg.lstsq`.

Si certaines erreurs ne sont pas finies ou non positives, elles sont remplacées par la médiane des erreurs valides. S'il n'existe aucune erreur valide, la valeur `1.0` est utilisée.

### 6.4 Résidus et `χ²`

Après ajustement :

\[
r_i = y_i - \hat{y_i}
\]

\[
\chi^2 = \sum_i \left(\frac{r_i}{\sigma_i}\right)^2
\]

\[
\chi^2_\nu = \frac{\chi^2}{\max(N-K,1)}
\]

Le code conserve :

- `chi2` ;
- `reduced_chi2` ;
- `model` ;
- `residuals` ;
- `coefficients`.

### 6.5 Critères d'information

Le projet calcule :

\[
AIC = \chi^2 + 2K
\]

\[
AICc = AIC + \frac{2K(K+1)}{N-K-1}
\]

si `N > K + 1`, sinon `AICc = inf`.

\[
BIC = \chi^2 + K\ln(N)
\]

Le BIC pénalise plus fortement la complexité que l'AIC. C'est cohérent avec l'objectif : éviter que des ordres Fourier trop élevés absorbent le bruit ou des artefacts inter-nuits.

---

## 7. Significativité des harmoniques

Pour chaque harmonique `k`, le code calcule l'amplitude :

\[
A_k = \sqrt{a_k^2+b_k^2}
\]

Il estime ensuite l'incertitude de cette amplitude par propagation linéaire de la covariance des coefficients.

La covariance est approximée par :

\[
\mathrm{Cov}(\beta) = (X_W^T X_W)^+ \; \chi^2_\nu
\]

avec `+` la pseudo-inverse de Moore-Penrose.

Pour :

\[
A(a,b)=\sqrt{a^2+b^2}
\]

le gradient vaut :

\[
\nabla A = \left(\frac{a}{A}, \frac{b}{A}\right)
\]

et la variance propagée :

\[
\sigma_A^2 = \nabla A^T \; \mathrm{Cov}(a,b) \; \nabla A
\]

La significativité stockée est alors :

\[
S_k = \frac{A_k}{\sigma_{A_k}}
\]

Cette valeur est utilisée dans la sélection d'ordre : un ordre supplémentaire n'est accepté que si son harmonique ajouté est significatif au-dessus d'un seuil, actuellement `3.0`.

---

## 8. Recherche de période

L'application possède deux modes :

1. période inconnue : recherche automatique ;
2. période imposée : ajustement Fourier à une période fournie.

### 8.1 Grille de périodes

`period_grid(min_period_days, max_period_days, samples)` construit une grille uniforme en fréquence :

\[
f \in \left[\frac{1}{P_\max}, \frac{1}{P_\min}\right]
\]

puis retourne :

\[
P = \frac{1}{f}
\]

C'est un meilleur choix qu'une grille uniforme en période, car le repliement de phase dépend linéairement de la fréquence sur une baseline donnée.

### 8.2 GLS simplifié

La fonction `gls_power()` calcule un score de type Lomb-Scargle généralisé en comparant :

- un modèle constant, c'est-à-dire Fourier ordre `0` ;
- un modèle sinusoïdal ordre `1` à chaque période.

Le score est :

\[
Power(P) = \max\left(0, 1 - \frac{\chi^2_{M=1}(P)}{\chi^2_\mathrm{const}}\right)
\]

Ce n'est pas un appel à `astropy.timeseries.LombScargle`, mais une implémentation cohérente avec le modèle interne : mêmes poids, mêmes offsets de groupes, même convention de temps.

Interprétation : plus le modèle sinusoïdal réduit le `χ²` par rapport au modèle constant, plus le score est élevé.

### 8.3 Problème astrophysique : astéroïdes double-pic

Une courbe de lumière d'astéroïde allongé a souvent deux maxima par rotation. Si les deux maxima/minima sont proches en amplitude, un modèle sinusoïdal simple peut accrocher la demi-période :

\[
P_\mathrm{GLS} \approx \frac{P_\mathrm{rot}}{2}
\]

C'est une difficulté classique : le périodogramme sinusoïdal favorise la fréquence dominante, qui peut être le deuxième harmonique de la rotation réelle.

### 8.4 Génération des candidats GLS

`gls_peak_candidates()` procède ainsi :

1. repérer les maxima locaux du score GLS ;
2. trier les pics par puissance décroissante ;
3. prendre les `top_n` meilleurs pics ;
4. pour chaque pic, générer des périodes candidates par multiplicateurs.

Par défaut :

```python
multipliers = (0.5, 1.0, 2.0)
```

Donc pour un pic GLS `P_g`, le programme teste :

\[
0.5P_g, \quad P_g, \quad 2P_g
\]

Cette stratégie permet de traiter :

- les ambiguïtés demi-période ;
- les cas où GLS trouve déjà la période complète ;
- les cas où un alias ou une harmonique inverse demande de tester `P/2`.

Le code évite les doublons avec une tolérance relative d'environ `1 / n_samples`.

Les offsets par fichier ne sont jamais pré-calculés par moyenne ou médiane de fichier. Ils sont des paramètres libres de la même résolution linéaire que les coefficients Fourier. Cette contrainte est importante : un fichier peut couvrir préférentiellement un maximum, un minimum ou une branche de la courbe, et sa médiane peut alors contenir une partie réelle du signal de rotation.

### 8.5 Raffinement local des candidats

Chaque période candidate est raffinée autour de sa valeur initiale avec `refine_period()`.

Pour une période initiale `P0`, l'intervalle est :

\[
[P_0(1-w), P_0(1+w)]
\]

avec `w = refine_width`, valeur par défaut `0.01`, soit ±1 %. Les bornes globales `min_period` et `max_period` sont respectées.

La recherche locale teste une grille de `candidate_refine_samples` échantillons par couple période/ordre lors de la phase candidat, puis `refine_samples` échantillons pour le meilleur modèle final.

### 8.6 Recherche période + ordre

`search_period_order_candidates()` évalue pour chaque candidat GLS et chaque ordre Fourier :

```text
période candidate × ordre Fourier → raffinement → FitResult
```

Par défaut les ordres sont :

```text
2:6
```

c'est-à-dire ordres 2, 3, 4, 5 et 6.

Les résultats sont triés par BIC brut et écrits dans `period_order_candidates.csv`.

---

## 9. Sélection prudente période / ordre

La fonction `select_stable_period()` est l'une des parties les plus importantes du projet.

### 9.1 Pourquoi ne pas prendre simplement le meilleur BIC ?

En théorie, le BIC pénalise la complexité. En pratique, une courbe multi-nuits, bruitée, avec offsets et alias temporels peut produire un meilleur BIC sur une période fausse si l'ordre Fourier élevé absorbe des structures non physiques.

Le projet adopte donc une sélection hiérarchique plus conservatrice.

### 9.2 Étape 1 : meilleur BIC global

Le code calcule quand même :

```python
bic_best = min(candidate_fits, key=lambda fit: fit.bic)
```

Ce résultat est conservé pour audit, mais pas forcément retenu.

### 9.3 Étape 2 : période de référence à ordre bas

L'ordre de référence est :

```python
reference_order = 2 if 2 in orders else min(orders)
```

Le meilleur ajustement à cet ordre bas fournit une période de référence :

\[
P_\mathrm{ref}
\]

Raison scientifique : un ordre 2 est souvent le minimum réaliste pour une courbe d'astéroïde double-pic, mais reste assez peu flexible pour limiter le sur-ajustement.

### 9.4 Étape 3 : famille stable en période

Pour chaque ordre testé, le programme ne retient que les fits dont la période est proche de `P_ref` :

\[
|P - P_\mathrm{ref}| \leq \epsilon P_\mathrm{ref}
\]

avec :

```python
stability_tolerance = 0.02
```

soit ±2 %.

Pour chaque ordre, le meilleur BIC dans cette famille stable est conservé.

### 9.5 Étape 4 : augmentation progressive de l'ordre

Le programme part du premier ordre stable, puis accepte l'ordre suivant seulement si :

1. le BIC diminue ;
2. le `χ²` réduit ne se dégrade pas ;
3. le nouvel harmonique ajouté a une significativité ≥ `3.0`.

En pseudo-code :

```python
selected = ordre_stable_le_plus_bas
previous = selected
for fit in ordres_stables_suivants:
    if fit.bic < previous.bic \
       and fit.reduced_chi2 <= previous.reduced_chi2 \
       and significance_last_harmonic >= 3:
        selected = fit
        previous = fit
    else:
        break
```

Cette stratégie encode une idée physique : on augmente la complexité seulement si le nouvel harmonique apporte une structure robuste et statistiquement défendable.

### 9.6 Fichier de synthèse

`period_selection_summary.csv` contient :

- le fit sélectionné ;
- le fit de référence à ordre bas ;
- le meilleur BIC global ;
- les meilleurs fits de chaque ordre stable ;
- la tolérance de stabilité ;
- le seuil de significativité harmonique.

C'est un fichier critique pour diagnostiquer pourquoi une période a été retenue au lieu d'une autre.

---

## 10. Mode période imposée

L'option :

```bash
--period <jours>
```

court-circuite toute la recherche GLS/Fourier en période. Le programme ajuste simplement les ordres Fourier demandés à cette période imposée, puis choisit le meilleur ordre par BIC via `search_fourier()`.

Restrictions :

- `--period` doit être strictement positif ;
- `--residual-filter` est interdit avec `--period`, car le filtrage relance une recherche de période.

Ce mode est utile si une période est connue par la littérature ou par une analyse externe, et que l'on veut seulement :

- ajuster le modèle ;
- mesurer l'amplitude ;
- produire une courbe repliée ;
- inspecter les résidus.

---

## 11. Estimation de l'incertitude sur la période

La fonction `estimate_period_uncertainty()` utilise un profil de `χ²` autour de la meilleure période.

Elle cherche les bornes où :

\[
\chi^2(P) = \chi^2(P_\mathrm{best}) + \Delta\chi^2
\]

Deux estimations sont produites :

1. `delta_chi2_1` : avec `Δχ² = 1` ;
2. `scaled_delta_chi2` : avec `Δχ² = max(1, χ²_ν)`.

La deuxième est plus prudente lorsque le `χ²` réduit est supérieur à 1, ce qui indique que les erreurs photométriques sont probablement sous-estimées ou que le modèle ne capture pas toute la structure.

### 11.1 Recherche numérique

Pour chaque côté de la période :

- le code avance par pas multiplicatif croissant ;
- dès que le seuil est dépassé, il utilise `scipy.optimize.brentq` pour trouver la racine ;
- la recherche est limitée à une fraction maximale de la période, par défaut `25 %`.

Si la borne n'est pas trouvée, elle est indiquée comme non bornée.

---

## 12. Filtrage robuste des résidus

L'option :

```bash
--residual-filter
```

active une seconde passe :

1. recherche initiale de période ;
2. calcul des résidus ;
3. rejet robuste des points aberrants ;
4. nouvelle recherche de période sur les points conservés ;
5. production de sorties préfixées `residual_filtered_`.

### 12.1 Seuil robuste MAD

Par défaut, le centre des résidus est la médiane :

\[
r_0 = \mathrm{median}(r_i)
\]

La dispersion robuste est :

\[
\sigma_\mathrm{robuste} = 1.4826 \times \mathrm{MAD}
\]

avec :

\[
MAD = \mathrm{median}(|r_i-r_0|)
\]

Le seuil par défaut est :

\[
|r_i-r_0| > 3.5\sigma_\mathrm{robuste}
\]

On peut remplacer ce seuil par une valeur absolue en magnitude :

```bash
--residual-filter-threshold-mag 0.08
```

### 12.2 Garde-fous

Le filtrage est volontairement conservateur :

- `--residual-filter-max-reject-fraction`, défaut `0.25`, empêche de rejeter plus de 25 % des points ;
- `--residual-filter-min-points`, défaut `30`, impose un nombre minimal de points conservés ;
- si trop de points dépassent le seuil, le code ne rejette que les plus grands écarts jusqu'à la fraction maximale.

### 12.3 Sorties dédiées

Le filtrage produit notamment :

- `residual_filter_summary.csv` ;
- `residual_filter_rejected_points.png` ;
- `residual_filtered_gls_periodogram.png` ;
- `residual_filtered_fourier_period_search.png` ;
- `residual_filtered_folded_lightcurve.png` ;
- `residual_filtered_residuals.csv`.

---

## 13. Courbes, phase et amplitude

### 13.1 Phase

La phase utilisée pour les graphiques est :

\[
\phi_i = \left(\frac{t_i - \min(t)}{P}\right) \bmod 1
\]

Les graphiques de courbe repliée affichent généralement deux cycles :

\[
\phi \in [0,2]
\]

Cela améliore la lisibilité des courbes double-pic.

### 13.2 Magnitudes alignées et affichage recentré

Pour retirer les offsets par fichier, le projet calcule :

\[
m_{i,\mathrm{aligned,reduced}} = m_{\mathrm{reduced},i} - Z_{g_i}
\]

Le groupe 0 a un offset nul par convention.

Le fit travaille sur `mag_reduced`, mais cette grandeur peut être visuellement déroutante car elle est ramenée aux distances unitaires. Pour les graphes principaux, le code calcule donc une constante globale :

\[
s = \mathrm{median}(m_\mathrm{obs}) - \mathrm{median}(m_\mathrm{aligned,reduced})
\]

puis affiche :

\[
m_\mathrm{plot} = m_\mathrm{aligned,reduced} + s
\]

et applique le même `s` au modèle aligné. Ce décalage est unique pour toute la courbe : il ne dépend pas du fichier. Les résidus sont inchangés, car le même décalage global est appliqué aux points et au modèle.

À ne pas faire :

- recentrer fichier par fichier ;
- soustraire une moyenne ou médiane par fichier avant le fit ;
- calculer `mag_plot` avant que les offsets `Zi` aient été ajustés.

### 13.3 Amplitude

L'amplitude photométrique affichée est calculée sur le modèle Fourier, pas directement sur les points :

1. échantillonnage du modèle sur 1000 phases ;
2. amplitude :

\[
A = \max(m_\mathrm{model}) - \min(m_\mathrm{model})
\]

Comme les magnitudes sont inversées, l'amplitude est positive mais ne donne pas directement un sens maximum/minimum en flux.

---

## 14. Commandes CLI

### 14.1 `inspect`

Syntaxe :

```bash
asteroid-lc inspect FILES... [--keep-start-time]
```

But : résumer rapidement les fichiers sans lancer de recherche de période.

Sorties console :

- nombre de fichiers ;
- nombre total de mesures ;
- baseline en jours ;
- pour chaque fichier :
  - nom du fichier ;
  - nombre de mesures ;
  - objet ;
  - filtre ;
  - temps de pose.

Option :

| Option | Effet |
|---|---|
| `--keep-start-time` | Ne pas convertir les JD vers le milieu de pose. |

### 14.2 `search`

Syntaxe générale :

```bash
asteroid-lc search FILES... --min-period MIN --max-period MAX [options]
```

ou :

```bash
asteroid-lc search FILES... --period PERIOD [options]
```

#### Options de période

| Option | Type | Défaut | Description |
|---|---:|---:|---|
| `--min-period` | float | aucun | Période minimale en jours. Requise sans `--period`. |
| `--max-period` | float | aucun | Période maximale en jours. Requise sans `--period`. |
| `--period` | float | aucun | Période imposée en jours. Saute la recherche de période. |
| `--samples` | int | `8000` | Nombre d'échantillons de la grille initiale en fréquence. |

#### Options Fourier / GLS

| Option | Type | Défaut | Description |
|---|---:|---:|---|
| `--orders` | str | `2:6` | Ordres Fourier testés. Format `2:6` ou `4`. |
| `--gls-candidates` | int | `20` | Nombre de pics GLS conservés. |
| `--gls-multipliers` | liste float | `0.5,1,2` | Multiplicateurs appliqués à chaque pic GLS. |
| `--refine-width` | float | `0.01` | Demi-largeur relative de raffinement autour d'un candidat. |
| `--candidate-refine-samples` | int | `300` | Échantillons de raffinement pour chaque couple candidat période/ordre. |
| `--refine-samples` | int | `2000` | Échantillons du raffinement final. |

#### Options de filtrage robuste

| Option | Type | Défaut | Description |
|---|---:|---:|---|
| `--residual-filter` | flag | `False` | Active la seconde passe avec rejet robuste. |
| `--residual-filter-sigma` | float | `3.5` | Seuil MAD en sigma robuste. |
| `--residual-filter-threshold-mag` | float | aucun | Seuil absolu en magnitude, remplace le seuil sigma. |
| `--residual-filter-max-reject-fraction` | float | `0.25` | Fraction maximale de points rejetés. |
| `--residual-filter-min-points` | int | `30` | Nombre minimal de points conservés. |

#### Options temporelles et éphémérides

| Option | Type | Défaut | Description |
|---|---:|---:|---|
| `--keep-start-time` | flag | `False` | Ne pas convertir au milieu de pose. |
| `--no-ephemeris` | flag | `False` | Désactive Horizons et reste en JD/magnitude observée. |
| `--no-geometric-correction` | flag | `False` | Diagnostic : conserve HJD/r/Δ mais n'applique pas `5 log10(rΔ)`. |
| `--no-correction-cache` | flag | `False` | Ne lit ni n'écrit le cache de corrections Horizons. |
| `--horizons-timeout` | float | `30.0` | Timeout des appels Horizons en secondes. |

#### Options de sortie

| Option | Type | Défaut | Description |
|---|---:|---:|---|
| `--out` | str | `output` | Répertoire de sortie. |

### 14.3 `clear-cache`

Syntaxe :

```bash
asteroid-lc clear-cache DATA_DIR [--dry-run]
```

But : supprimer les fichiers du sous-dossier `DATA_DIR/cache` créés pour les corrections Horizons/géométriques. Cette commande ne supprime pas les fichiers d'observation source.

Option :

| Option | Effet |
|---|---|
| `--dry-run` | Compte les fichiers de cache sans les supprimer. |

---

## 15. Fichiers produits

### 15.1 Graphiques principaux

| Fichier | Produit quand | Contenu |
|---|---|---|
| `gls_periodogram.png` | recherche automatique | Score GLS en fonction de la période. |
| `fourier_period_search.png` | recherche automatique | Score `-BIC` à l'ordre Fourier retenu. |
| `folded_lightcurve.png` | toujours | Courbe repliée corrigée géométriquement, alignée puis recentrée globalement. |
| `folded_lightcurve_by_file.png` | toujours | Courbe repliée corrigée, avec style/couleur par fichier. |
| `folded_lightcurve_by_file_with_residuals.png` | toujours | Courbe repliée + panneau de résidus en phase. |
| `residuals.png` | toujours | Résidus en fonction du temps et de la phase. |
| `residual_filter_rejected_points.png` | si filtrage | Points rejetés sur résidus vs temps et phase. |

### 15.2 CSV principaux

| Fichier | Contenu |
|---|---|
| `period_summary.csv` | Période, amplitude, incertitudes `Δχ²=1` et rééchelonnée. |
| `file_summary.csv` | Résumé par fichier : date, observateur, nombre de points, scatter, offset. |
| `period_order_candidates.csv` | Tous les couples candidat période/ordre testés, triés par BIC. |
| `fourier_order_summary.csv` | Meilleur candidat par ordre Fourier. |
| `period_selection_summary.csv` | Décision de sélection stable période/ordre. |
| `residuals.csv` | Table point par point : fichier, JD/HJD, phase, `mag_obs`, `r`, `Δ`, correction géométrique, `mag_reduced`, `Zi`, `mag_plot`, modèle, résidu. |
| `ephemeris_by_file.csv` | RA/DEC, `r`, `Δ` Horizons par fichier et statistiques de correction HJD. |
| `residual_filter_summary.csv` | Nombre de points rejetés, seuils et dispersion robuste. |

Les CSV sont écrits en `utf-8-sig`, séparateur `;`, ce qui facilite l'ouverture dans Excel en environnement français.

### 15.3 `run_metadata.json`

Ce fichier est très important pour la reproductibilité. Il contient :

- date de génération UTC ;
- commande exécutée ;
- version Python ;
- version du package si installée ;
- paramètres CLI ;
- fichiers d'entrée résolus ;
- nombre de fichiers et de mesures ;
- baseline ;
- période finale ;
- ordre Fourier ;
- AIC/AICc/BIC ;
- période filtrée éventuelle ;
- stratégie de sélection ;
- liste des fichiers produits.

Pour un agent de codage, c'est le bon endroit pour ajouter des métadonnées futures : méthode de normalisation, version de l'algorithme, coordonnées observatoire, version Horizons, etc.

---

## 16. Chaîne d'exécution complète de `search`

### 16.1 Cas période inconnue

```text
Entrées CLI
  ↓
expand_inputs()
  ↓
read_lightcurve()
  ↓
correction milieu de pose, sauf --keep-start-time
  ↓
si pas --no-ephemeris :
    apply_ephemeris_corrections()
    charge ou écrit cache par fichier
    jd devient hjd_utc
    magnitude devient mag_reduced
  ↓
period_grid(min,max,samples)
  ↓
gls_power()
  ↓
gls_peak_candidates()
  ↓
pour chaque candidat GLS et chaque ordre :
    refine_period()
    weighted_fit()
  ↓
select_stable_period()
  ↓
refine_period() final autour de la période retenue
  ↓
estimate_period_uncertainty()
  ↓
model_amplitude()
  ↓
graphiques + CSV + JSON
  ↓
si --residual-filter :
    residual_filter_from_fit()
    subset_lightcurve()
    relance complète de la recherche
```

### 16.2 Cas période imposée

```text
Entrées CLI
  ↓
expand_inputs()
  ↓
read_lightcurve()
  ↓
correction HJD éventuelle
correction géométrique éventuelle, sauf --no-ephemeris ou diagnostic --no-geometric-correction
  ↓
search_fourier(curve, [period], orders)
  ↓
estimate_period_uncertainty()
  ↓
graphiques + CSV + JSON
```

---

## 17. Interprétation astrophysique des paramètres

### 17.1 Période retenue

La période finale est censée représenter la période de rotation sidérale apparente projetée dans les données photométriques, sous réserve des effets suivants :

- alias liés à l'échantillonnage jour/nuit ;
- évolution de phase solaire pendant la campagne ;
- changement de géométrie d'observation ;
- bruit photométrique et offsets inter-nuits ;
- courbe double-pic pouvant favoriser `P/2` dans un périodogramme sinusoidal.

### 17.2 Ordre Fourier

L'ordre Fourier représente la complexité morphologique de la courbe :

- ordre 1 : sinus simple, souvent insuffisant pour un astéroïde ;
- ordre 2 : première représentation naturelle d'une courbe double-pic ;
- ordres 3 à 6 : asymétries, maxima/minima de hauteurs différentes, formes anguleuses ;
- ordre trop élevé : risque de capturer le bruit, les offsets imparfaits ou les erreurs de phase.

La stratégie actuelle force implicitement une préférence pour un ordre bas stable, ce qui est physiquement raisonnable.

### 17.3 Amplitude

L'amplitude modèle est une amplitude crête-à-crête en magnitude. Elle peut être reliée grossièrement à l'allongement apparent si l'effet d'albédo est négligé :

\[
\frac{a}{b} \gtrsim 10^{0.4A}
\]

Cette relation n'est pas implémentée actuellement, mais pourrait être ajoutée comme indicateur astrophysique.

### 17.4 Résidus

Les résidus permettent de diagnostiquer :

- points aberrants ;
- erreur de période ;
- ordre Fourier insuffisant ;
- offsets inter-nuits mal contraints ;
- photométrie contaminée ;
- possible signature binaire si des événements mutuels produisent des creux systématiques ;
- changement de forme ou d'amplitude avec la géométrie.

---

## 18. Points importants pour les astéroïdes binaires

Le projet actuel ne détecte pas explicitement les binarités. Il fournit toutefois plusieurs briques utiles :

- courbe repliée propre ;
- modèle rotationnel principal ;
- résidus en phase et en temps ;
- filtrage robuste ;
- export `residuals.csv`.

Pour un astéroïde binaire, les événements mutuels peuvent apparaître comme :

- des creux additionnels non modélisés par la rotation principale ;
- une dispersion accrue dans certaines phases ;
- des résidus structurés en temps ;
- une seconde période orbitale dans les résidus.

Fonctionnalités futures pertinentes :

1. recherche de périodicité dans les résidus ;
2. modèle à deux périodes : rotation primaire + période orbitale ;
3. détection d'événements de type eclipse/occultation ;
4. BLS ou box-fitting sur les résidus ;
5. séparation `P1`, `P2`, `P_orb` selon convention utilisée dans les publications de binaires ;
6. export d'un rapport spécifique binaire.

---

## 19. Limites et points d'attention du code actuel

### 19.1 Portabilité de l'encodage `mbcs`

`mbcs` est Windows-spécifique. Pour une exécution Linux, il faudra probablement remplacer ou paramétrer l'encodage.

### 19.2 GLS implémenté comme Fourier ordre 1

Le score GLS est interne et cohérent avec le modèle, mais ce n'est pas une implémentation complète avec toutes les statistiques de Lomb-Scargle classique. Il sert de pré-sélecteur de candidats, pas de décision finale.

### 19.3 Coût computationnel

La recherche finale `fourier_period_search.png` recalcule un ajustement Fourier pour chaque période de la grille à l'ordre retenu. Pour `samples=8000`, c'est acceptable sur un petit jeu de données, mais cela peut devenir coûteux si :

- beaucoup de points ;
- beaucoup de fichiers ;
- beaucoup d'ordres ;
- beaucoup de candidats GLS.

Optimisations possibles : vectorisation partielle, cache de matrices, réduction adaptative de grille, parallélisation.

### 19.4 Offsets par fichier

Les offsets par fichier sont très utiles, mais peuvent absorber une partie du signal si chaque fichier couvre une fraction très courte de la période. C'est un problème classique : si les nuits sont peu recouvrantes en phase, les offsets et la courbe périodique deviennent partiellement dégénérés.

### 19.5 HJD géocentrique et non topocentrique

La correction est géocentrique. Pour des astéroïdes proches et rapides, ou des exigences très précises, une correction topocentrique serait meilleure.

### 19.6 Une position RA/DEC par fichier

La position de l'astéroïde est supposée constante dans chaque fichier pour la correction HJD. Cela peut être insuffisant pour des objets rapides ou des fichiers couvrant une longue durée.

### 19.7 Correction de distance sans modèle de phase

Le projet corrige maintenant la variation de magnitude due :

- à la distance héliocentrique `r` ;
- à la distance observateur-astéroïde `Δ`.

La correction appliquée est :

\[
m_\mathrm{reduced} = m_\mathrm{obs} - 5\log_{10}(r\Delta)
\]

Le projet ne corrige pas l'angle de phase solaire `α`. C'est volontaire : l'objectif est la recherche de période synodique sur une seule opposition, pas l'estimation d'une magnitude absolue ou d'un modèle physique de phase.

Pour des campagnes très longues à l'intérieur d'une même opposition, une évolution future pourrait ajouter une tendance lente optionnelle en temps ou en angle de phase. Elle devrait rester séparée des offsets par fichier et ne pas introduire de modèle `H-G`.

### 19.8 Petite anomalie d'affichage console

Dans la sortie console de l'incertitude brute, le code imprime deux fois la partie `minus` :

```python
f"-{format_uncertainty(raw_uncertainty.minus_days)} / "
f"-{format_uncertainty(raw_uncertainty.minus_days)} / "
f"+{format_uncertainty(raw_uncertainty.plus_days)}"
```

La deuxième occurrence devrait probablement être supprimée. Ce n'est qu'un problème d'affichage console, pas de calcul ni de CSV.

### 19.9 `period_selection_summary.csv`

Dans la version inspectée, l'en-tête est correct dans le fichier réel, mais il faut surveiller toute duplication de colonne lors d'évolutions futures. L'archive actuelle contient une seule colonne `harmonic_significance_threshold` dans la section écrite effectivement par le code après inspection complète.

---

## 20. Extension recommandée : détection binaire

Une suite logique pour le projet serait :

### 20.1 Étape 1 : modèle rotationnel robuste

Utiliser l'existant pour obtenir :

- période principale `P_rot` ;
- modèle Fourier principal ;
- résidus propres ;
- points éventuellement rejetés.

### 20.2 Étape 2 : analyse des résidus

Ajouter une commande ou une option :

```bash
asteroid-lc binary-search data/*.txt --primary-period 0.2106178 --out output
```

ou intégrer à `search` :

```bash
--search-secondary
--secondary-method bls|gls|pdm
```

Sorties possibles :

- périodogramme des résidus ;
- BLS sur les creux ;
- candidats `P_orb` ;
- phase-fold des résidus sur `P_orb` ;
- table des événements candidats.

### 20.3 Étape 3 : modèle combiné

Modèle possible :

\[
m(t) = m_\mathrm{rot}(t; P_\mathrm{rot}) + m_\mathrm{orb}(t; P_\mathrm{orb}) + offsets
\]

Avec `m_orb` pouvant être :

- une série de Fourier basse fréquence ;
- un modèle à boîtes pour événements mutuels ;
- un modèle paramétrique d'éclipse simplifié.

### 20.4 Étape 4 : validation astrophysique

Critères à ajouter :

- cohérence des événements sur plusieurs nuits ;
- profondeur en magnitude ;
- durée relative ;
- stabilité du timing ;
- réduction du `χ²` et pénalité BIC ;
- rejet des événements isolés liés à la photométrie.

---

## 21. Extension recommandée : tendance de phase locale optionnelle

La correction géométrique en `5 log10(rΔ)` est déjà implémentée. Pour améliorer l'analyse de campagnes longues sur une même opposition, une extension possible serait une tendance de phase locale indépendante de la rotation :

1. récupérer depuis Horizons l'angle de phase `α` en plus de `r` et `Δ` ;
2. conserver la magnitude réduite actuelle :

\[
m(1,1,\alpha) = m - 5\log_{10}(r\Delta)
\]

3. ajuster éventuellement une tendance locale :

\[
m_\mathrm{corr} = m(1,1,\alpha) - \beta \alpha
\]

ou ajouter une tendance linéaire directement au modèle :

\[
m(t) = Fourier(P) + offsets + c_1(t-t_0)
\]

Cette extension ne doit pas transformer le projet en logiciel de magnitude absolue. Les modèles `H-G`, `H-G1,G2`, `H-G12`, les corrections d'aspect et les combinaisons multi-opposition restent hors périmètre.

---

## 22. Extension recommandée : choix automatique plus rigoureux de l'ordre Fourier

La stratégie actuelle est déjà prudente. Pour aller plus loin :

- ajouter une validation croisée par nuit ;
- utiliser un test de vraisemblance pénalisé entre ordres imbriqués ;
- imposer une amplitude minimale par harmonique en mmag ;
- contrôler les oscillations non physiques du modèle ;
- ajouter un score de stabilité par bootstrap ;
- comparer les périodes retenues sur sous-échantillons de nuits.

Un indicateur utile serait :

```text
order_stability_score = fraction de bootstrap retrouvant même période ± tolérance
```

---

## 23. Résumé opérationnel pour Codex

### Ce que fait déjà l'application

```text
Lire fichiers CCD → combiner nuits → Horizons/cache → HJD + correction géométrique → GLS ordre 1 → pics candidats → tester P/2,P,2P → ajuster Fourier pondéré avec offsets par fichier → recentrer l'affichage sur mag_obs → sélectionner période stable et ordre significatif → exporter graphes/CSV/JSON.
```

### Ce qu'il faut préserver

- Les offsets par fichier ;
- La correction géométrique `mag_reduced = mag_obs - 5 log10(rΔ)` avant le fit ;
- Le recentrage d'affichage global, jamais par fichier ;
- Le cache sans valeurs dépendantes du fit ;
- La grille uniforme en fréquence ;
- La stratégie explicite contre l'ambiguïté double-pic ;
- La sélection hiérarchique ordre bas → famille stable → harmonique significatif ;
- Les exports CSV/JSON reproductibles ;
- Les sorties préfixées pour la passe filtrée.

### Ce qu'il faut améliorer en priorité

1. rendre l'encodage portable ;
2. élargir les tests unitaires ;
3. ajouter une tendance de phase locale optionnelle, sans modèle `H-G` ;
4. ajouter analyse secondaire des résidus pour binaires ;
5. ajouter une option de coordonnées observatoire pour correction topocentrique ;
6. ajouter des diagnostics d'alias ;
7. documenter formellement le format d'entrée.

---

## 24. Exemples d'utilisation

### Recherche standard

```bash
asteroid-lc search data/*.txt --min-period 0.083333 --max-period 0.833333 --out output
```

### Recherche avec plus de candidats GLS

```bash
asteroid-lc search data/*.txt \
  --min-period 0.083333 \
  --max-period 0.833333 \
  --gls-candidates 30 \
  --gls-multipliers 0.5,1,2 \
  --orders 2:8 \
  --out output
```

### Période imposée

```bash
asteroid-lc search data/*.txt --period 0.2106178 --orders 2:6 --out output
```

### Travail hors ligne sans Horizons

```bash
asteroid-lc search data/*.txt \
  --min-period 0.083333 \
  --max-period 0.833333 \
  --no-ephemeris \
  --out output
```

### Diagnostic sans correction géométrique

```bash
asteroid-lc search data/*.txt \
  --min-period 0.083333 \
  --max-period 0.833333 \
  --no-geometric-correction \
  --out output_no_geom
```

### Purge du cache de corrections

```bash
asteroid-lc clear-cache data/ --dry-run
asteroid-lc clear-cache data/
```

### Filtrage robuste des résidus

```bash
asteroid-lc search data/*.txt \
  --min-period 0.083333 \
  --max-period 0.833333 \
  --residual-filter \
  --out output
```

### Seuil de résidu absolu

```bash
asteroid-lc search data/*.txt \
  --min-period 0.083333 \
  --max-period 0.833333 \
  --residual-filter \
  --residual-filter-threshold-mag 0.08 \
  --out output
```
