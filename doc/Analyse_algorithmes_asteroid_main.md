Voici l’analyse de la branche **`main`**, dans l’état du commit [`c4381ce32cb127d52874e71a5f0f11bdbbc3b628`](https://github.com/mdeldem/asteroid/tree/c4381ce32cb127d52874e71a5f0f11bdbbc3b628).

J’ai relu l’ensemble des modules Python et confronté leur fonctionnement au `README.md` et au mémoire technique. La description ci-dessous prend **le code exécuté comme référence**. Les explications mathématiques explicitent ses opérations ; les limites signalées permettent de préciser la portée de ses résultats.

**Le programme recherche une périodicité photométrique à l’aide d’un modèle de Fourier pondéré, commun à toutes les observations, avec un décalage de magnitude ajusté pour chaque fichier.** La recherche automatique utilise d’abord un périodogramme sinusoïdal pour produire des candidats, puis une sélection hiérarchique de période et d’ordre de Fourier.

Une distinction commande toute la suite : **le résultat final n’est pas nécessairement le couple période–ordre ayant le plus petit BIC parmi tous les candidats.**

Les principales références au code sont :

| Partie algorithmique | Source |
|---|---|
| Ajustement pondéré, GLS, candidats, sélection, incertitudes | [period.py](https://github.com/mdeldem/asteroid/blob/c4381ce32cb127d52874e71a5f0f11bdbbc3b628/src/asteroid_lightcurve/period.py) |
| Enchaînement des traitements, paramètres, filtrage des résidus | [cli.py](https://github.com/mdeldem/asteroid/blob/c4381ce32cb127d52874e71a5f0f11bdbbc3b628/src/asteroid_lightcurve/cli.py) |
| Correction héliocentrique des temps | [ephemeris.py](https://github.com/mdeldem/asteroid/blob/c4381ce32cb127d52874e71a5f0f11bdbbc3b628/src/asteroid_lightcurve/ephemeris.py) |
| Milieu de pose et constitution d’un sous-ensemble de mesures | [models.py](https://github.com/mdeldem/asteroid/blob/c4381ce32cb127d52874e71a5f0f11bdbbc3b628/src/asteroid_lightcurve/models.py) |
| Phase, amplitude, alignement et dispersion des résidus | [plotting.py](https://github.com/mdeldem/asteroid/blob/c4381ce32cb127d52874e71a5f0f11bdbbc3b628/src/asteroid_lightcurve/plotting.py) |

**1. Les grandeurs manipulées et le problème que le programme cherche à résoudre**

L’analyse part d’un ensemble de mesures photométriques déjà disponibles. Chaque mesure comporte quatre informations utiles au calcul :

$$
(t_i,\;m_i,\;\sigma_i,\;g_i).
$$

| Symbole | Définition |
|---|---|
| $i$ | Indice d’une observation. |
| $N$ | Nombre total d’observations utilisées dans l’ajustement. |
| $t_i$ | Date de l’observation, exprimée en jours, après les corrections temporelles activées. |
| $m_i$ | Magnitude mesurée. |
| $\sigma_i$ | Incertitude photométrique annoncée pour cette magnitude. |
| $g_i$ | Groupe auquel appartient l’observation. Dans ce projet, un groupe correspond à un fichier. |
| $G$ | Nombre de groupes présents dans le jeu analysé. |
| $P$ | Période testée, en jours. |
| $f=1/P$ | Fréquence fondamentale, en cycles par jour. |
| $H$ | Ordre maximal de la série de Fourier. |
| $K$ | Nombre de coefficients linéaires ajustés. |

Le programme travaille **directement en magnitude**. Il ne transforme pas les magnitudes en flux avant l’ajustement.

Il cherche une fonction périodique commune aux différents groupes, malgré leurs éventuels décalages de zéro photométrique.

Le modèle suppose notamment :

- une période constante sur toute la campagne ;
- une même forme de courbe de lumière pour tous les groupes ;
- des différences entre groupes représentables par des décalages constants de magnitude ;
- des erreurs traitées individuellement, sans matrice de corrélation temporelle.

Il n’ajuste donc pas une amplitude différente par nuit, une évolution de la forme de la courbe ou une dérive de période.

La durée totale couverte par les observations est :

$$
T=\max_i(t_i)-\min_i(t_i).
$$

Cette durée est calculée et affichée. **Elle ne sert pas à déterminer automatiquement la densité de la grille de recherche.**

Enfin, le résultat directement établi par le calcul est une **période du signal photométrique dans les temps utilisés**. Le programme ne réalise aucune conversion explicite entre période synodique et période sidérale.

La formulation du mémoire existant évoquant une « période de rotation sidérale apparente » est donc trop affirmative pour décrire ce que calcule réellement cette version.

**2. L’ordre général d’exécution**

En mode de recherche automatique, l’enchaînement effectif est le suivant :

| Étape | Traitement | Résultat utilisé ensuite |
|---:|---|---|
| 1 | Correction éventuelle des dates vers le milieu de pose | Dates représentatives des poses |
| 2 | Correction héliocentrique éventuelle | Dates utilisées dans tous les calculs périodiques |
| 3 | Construction d’une grille uniforme en fréquence | Première liste de périodes à tester |
| 4 | Calcul du périodogramme GLS interne | Puissance associée à chaque période |
| 5 | Extraction des meilleurs pics GLS | Périodes de départ |
| 6 | Multiplication de chaque période de départ par les facteurs autorisés | Ensemble de périodes candidates |
| 7 | Raffinement local de chaque candidat pour chaque ordre de Fourier demandé | Meilleur ajustement de chaque couple candidat–ordre |
| 8 | Sélection d’une période de référence à bas ordre | Famille de périodes privilégiée |
| 9 | Sélection progressive de l’ordre dans cette famille | Couple période–ordre retenu |
| 10 | Raffinement supplémentaire de la période, à ordre fixé | Ajustement final de la première passe |
| 11 | Balayage Fourier de la grille initiale, à l’ordre retenu | Graphique de diagnostic |
| 12 | Estimation des incertitudes, amplitude et résidus | Caractérisation du résultat |
| 13 | Si demandé, filtrage des observations à partir des résidus | Sous-ensemble de mesures |
| 14 | Si au moins un point est rejeté, nouvelle recherche complète sur ce sous-ensemble | Résultat supplémentaire de la passe filtrée |

Deux précisions sont essentielles.

Premièrement, le balayage Fourier de l’étape 11 est effectué **après** la sélection. Il ne remplace pas le résultat sélectionné.

Deuxièmement, la seconde passe analyse **les magnitudes des observations conservées**. Elle ne recherche pas une nouvelle période directement dans les valeurs des résidus.

Le mode à période imposée emprunte un autre chemin : après les corrections temporelles, il compare les ordres de Fourier à cette seule période, puis calcule les diagnostics et les incertitudes.

Sources : [`run_period_search`](https://github.com/mdeldem/asteroid/blob/c4381ce32cb127d52874e71a5f0f11bdbbc3b628/src/asteroid_lightcurve/cli.py#L557-L613) et [`cmd_search`](https://github.com/mdeldem/asteroid/blob/c4381ce32cb127d52874e71a5f0f11bdbbc3b628/src/asteroid_lightcurve/cli.py#L679-L954).

**3. Les corrections temporelles précèdent toute recherche de période**

**La correction au milieu de pose associe chaque mesure au centre temporel de son exposition.**

Pour une durée de pose $E$, exprimée en secondes, la demi-durée en jours vaut :

$$
d_E=\frac{E}{2\times86400}.
$$

La transformation dépend de la position temporelle indiquée pour la date :

| Position de la date fournie | Correction |
|---|---|
| Début de pose | $t_{\mathrm{milieu}}=t_{\mathrm{fourni}}+d_E$ |
| Milieu de pose | $t_{\mathrm{milieu}}=t_{\mathrm{fourni}}$ |
| Fin de pose | $t_{\mathrm{milieu}}=t_{\mathrm{fourni}}-d_E$ |

Si la durée de pose n’est pas positive, aucune correction n’est appliquée.

L’option `--keep-start-time` désactive cette conversion. Son nom est un peu restrictif : concrètement, elle conserve les dates fournies, même si celles-ci correspondent à une fin de pose.

Cette opération corrige **la date représentative** de la mesure. Elle ne modélise pas l’intégration du signal pendant la pose. Le modèle de Fourier sera évalué à un instant unique, sans moyenner sa valeur sur la durée d’exposition.

Source : [`mid_exposure_jd`](https://github.com/mdeldem/asteroid/blob/c4381ce32cb127d52874e71a5f0f11bdbbc3b628/src/asteroid_lightcurve/models.py#L59-L66).

**La correction héliocentrique est activée par défaut.**

Pour chaque groupe $g$, le programme calcule d’abord :

$$
t_{\mathrm{milieu},g}
=
\frac{
\min_{i:g_i=g}(t_i)
+
\max_{i:g_i=g}(t_i)
}{2}.
$$

Il s’agit du **milieu de l’intervalle temporel**, et non de la moyenne arithmétique des dates de toutes les mesures.

À cette date, il obtient auprès de JPL Horizons :

- l’ascension droite $\alpha_g$ ;
- la déclinaison $\delta_g$ ;

pour une position d’observation géocentrique.

Ces deux angles définissent une direction vers l’astéroïde. Une seule direction est utilisée pour l’ensemble des mesures du groupe.

Astropy calcule ensuite une correction héliocentrique pour chaque date du groupe :

$$
\Delta t_{\mathrm{helio},i}.
$$

Le programme forme :

$$
t_i^{\mathrm{HJD}}
=
t_i^{\mathrm{JD}}
+
\Delta t_{\mathrm{helio},i}.
$$

L’idée géométrique de cette correction peut se représenter par :

$$
\Delta t_{\mathrm{helio},i}
\approx
\frac{
\boldsymbol{r}_{\odot\rightarrow\oplus}(t_i)
\cdot
\boldsymbol{n}_g
}{c},
$$

où :

- $\boldsymbol{r}_{\odot\rightarrow\oplus}$ est le vecteur du Soleil vers le géocentre ;
- $\boldsymbol{n}_g$ est le vecteur unitaire dans la direction retenue ;
- $c$ est la vitesse de la lumière.

Les positions terrestres évoluent avec les dates : **la correction n’est donc pas une constante ajoutée à tout le fichier**, même si la direction de l’objet est maintenue constante à l’intérieur de celui-ci.

La précision de la description impose de distinguer plusieurs choses :

- le point d’observation utilisé est le géocentre, pas les coordonnées réelles de l’observatoire ;
- la direction de l’objet est évaluée une fois par groupe ;
- le programme traite les dates d’entrée comme des dates UTC ;
- il ajoute numériquement la correction aux JD UTC et nomme le résultat `hjd_utc` ;
- il n’effectue pas de conversion explicite des dates de travail vers l’échelle TDB.

La documentation d’Astropy précise également que `light_travel_time` suppose une source à grande distance et néglige les effets de distance finie. **Il ne faut donc pas présenter ce traitement comme un calcul complet du temps de propagation entre un astéroïde mobile et l’observateur.** Le code ne récupère d’ailleurs pas la distance nécessaire à un tel calcul. [Documentation Astropy — Time.light_travel_time](https://docs.astropy.org/en/stable/api/astropy.time.Time.html)

L’option `--no-ephemeris` désactive à la fois l’interrogation Horizons et cette correction. La correction au milieu de pose reste indépendante et peut toujours être appliquée.

Source : [ephemeris.py](https://github.com/mdeldem/asteroid/blob/c4381ce32cb127d52874e71a5f0f11bdbbc3b628/src/asteroid_lightcurve/ephemeris.py).

**4. Le modèle ajusté associe une série de Fourier et des décalages par groupe**

Pour une période $P$ et un ordre $H$, le modèle est :

$$
\boxed{
\widehat m_i
=
c_0+d_{g_i}
+
\sum_{h=1}^{H}
\left[
a_h\cos\left(2\pi h\frac{t_i-t_0}{P}\right)
+
b_h\sin\left(2\pi h\frac{t_i-t_0}{P}\right)
\right]
}
$$

avec :

$$
t_0=\min_i(t_i).
$$

Les paramètres ont les significations suivantes :

| Paramètre | Signification |
|---|---|
| $c_0$ | Niveau constant de référence du modèle, en magnitude |
| $d_g$ | Décalage constant du groupe $g$, en magnitude |
| $a_h$ | Coefficient du cosinus de l’harmonique $h$ |
| $b_h$ | Coefficient du sinus de l’harmonique $h$ |
| $P$ | Période fondamentale testée |
| $H$ | Dernière harmonique incluse dans le modèle |

Le groupe $0$ sert de référence :

$$
d_0=0.
$$

Les autres groupes possèdent chacun un décalage libre.

Cette convention évite une indétermination : si tous les $d_g$ étaient libres en plus de $c_0$, on pourrait augmenter $c_0$ et diminuer tous les décalages de la même quantité sans changer aucune prédiction.

Le nombre de coefficients linéaires est donc :

$$
\boxed{
K=1+(G-1)+2H=G+2H.
}
$$

Par exemple, avec quatre groupes et un ordre $H=3$ :

$$
K=4+6=10.
$$

Le modèle comprend alors un niveau global, trois décalages relatifs et six coefficients de Fourier.

**Les harmoniques représentent des composantes d’une même périodicité.**

L’harmonique $h$ possède :

$$
f_h=\frac{h}{P},
\qquad
P_h=\frac{P}{h}.
$$

Ainsi :

| Harmonique | Fréquence | Période de la composante |
|---:|---:|---:|
| 1 | $1/P$ | $P$ |
| 2 | $2/P$ | $P/2$ |
| 3 | $3/P$ | $P/3$ |

La présence de l’harmonique 2 ne représente pas, dans ce programme, un second objet ou une seconde rotation indépendante. Elle permet de décrire une structure répétée deux fois pendant la période fondamentale.

Un ordre 2 inclut **les harmoniques 1 et 2**. Il n’impose ni deux maxima identiques ni une symétrie particulière.

Le choix de $t_0$ sert d’origine de phase. Il ne correspond pas nécessairement à un maximum ou à un minimum de lumière. Changer cette origine transforme les coefficients sinus et cosinus, mais ne change pas l’espace des courbes que le modèle peut représenter à une période donnée.

Source : [`design_matrix` et `weighted_fit`](https://github.com/mdeldem/asteroid/blob/c4381ce32cb127d52874e71a5f0f11bdbbc3b628/src/asteroid_lightcurve/period.py#L142-L215).

**5. Les décalages photométriques sont ajustés simultanément avec la courbe périodique**

Le but de $d_g$ est de permettre à plusieurs groupes de partager une même modulation périodique malgré des niveaux photométriques différents.

Ces différences peuvent notamment représenter des décalages de zéro entre séries.

Le traitement ne consiste pas à soustraire une fois pour toutes la moyenne de chaque groupe avant la recherche.

Pour **chaque période et chaque ordre testés**, le programme recalcule ensemble :

- le niveau $c_0$ ;
- tous les décalages $d_g$ ;
- tous les coefficients de Fourier.

Les décalages dépendent donc du modèle périodique essayé.

C’est important lorsqu’un groupe couvre seulement une portion du cycle : sa moyenne observée contient à la fois son zéro photométrique et la contribution de la portion de courbe échantillonnée.

Pour illustrer ce couplage, si l’on note :

$$
F_i(P)
=
\sum_{h=1}^{H}
\left[
a_h\cos\left(2\pi h\frac{t_i-t_0}{P}\right)
+
b_h\sin\left(2\pi h\frac{t_i-t_0}{P}\right)
\right],
$$

le niveau constant propre au groupe se règle à partir de la moyenne pondérée de :

$$
m_i-F_i(P).
$$

Il ne se règle donc pas uniquement à partir de la moyenne des $m_i$.

La contrepartie est une possible **dégénérescence** : si les observations couvrent peu de phase, une variation attribuable au signal périodique peut aussi être partiellement absorbée par les décalages.

Le programme ne mesure pas explicitement la solidité du recouvrement en phase entre groupes et ne rejette pas une solution pour ce motif.

Il ne corrige pas non plus une différence de forme ou d’amplitude entre filtres : un décalage constant ne peut représenter qu’une différence de niveau.

**6. L’ajustement à période fixée est un problème de moindres carrés pondérés**

Une fois $P$ et $H$ fixés, tous les coefficients inconnus apparaissent linéairement.

On rassemble les observations dans :

$$
\boldsymbol y=
\begin{pmatrix}
m_1\\
\vdots\\
m_N
\end{pmatrix},
$$

et les coefficients dans :

$$
\boldsymbol\beta=
\begin{pmatrix}
c_0&
d_1&\cdots&d_{G-1}&
a_1&b_1&\cdots&a_H&b_H
\end{pmatrix}^{T}.
$$

La matrice de conception $X$, de dimensions $N\times K$, contient :

1. une colonne constante égale à 1 ;
2. une colonne indicatrice pour chaque groupe autre que le groupe de référence ;
3. les colonnes cosinus et sinus des harmoniques.

Le modèle s’écrit :

$$
\widehat{\boldsymbol y}=X\boldsymbol\beta.
$$

**Les poids proviennent des incertitudes photométriques.**

Normalement :

$$
w_i=\frac{1}{\sigma_i^2}.
$$

Une incertitude deux fois plus petite produit donc un poids quatre fois plus grand.

Avant ce calcul, le programme remplace les incertitudes invalides, c’est-à-dire non finies ou non strictement positives :

- par la médiane des incertitudes valides, s’il en existe ;
- par $1.0$, sinon.

Notons $\widetilde\sigma_i$ l’incertitude effectivement utilisée après ce remplacement :

$$
w_i=\frac{1}{\widetilde\sigma_i^2}.
$$

Ce remplacement concerne le calcul des poids. Il ne réécrit pas le tableau original des incertitudes utilisé ailleurs.

Si toutes les erreurs sont remplacées par $1.0$, l’ajustement revient à donner le même poids à tous les points. En revanche, l’échelle absolue du $\chi^2$ devient dépendante de cette convention arbitraire.

Le programme ne normalise pas les poids pour donner une influence identique à chaque groupe. Un groupe comportant davantage de points précis peut contribuer beaucoup plus fortement au résultat.

**Le critère minimisé est :**

$$
\boxed{
\chi^2(P,H,\boldsymbol\beta)
=
\sum_{i=1}^{N}
\frac{
\left(m_i-(X\boldsymbol\beta)_i\right)^2
}{
\widetilde\sigma_i^2
}.
}
$$

Le code forme :

$$
X_w=W^{1/2}X,
\qquad
\boldsymbol y_w=W^{1/2}\boldsymbol y,
$$

où $W$ est la matrice diagonale des $w_i$, puis résout :

$$
\widehat{\boldsymbol\beta}
=
\operatorname*{argmin}_{\boldsymbol\beta}
\left\|
X_w\boldsymbol\beta-\boldsymbol y_w
\right\|_2^2.
$$

Ici, $\operatorname*{argmin}$ désigne **la valeur des coefficients pour laquelle le critère atteint son minimum**.

La résolution est confiée à `scipy.linalg.lstsq`. Le code ne calcule pas les coefficients par inversion explicite des équations normales ; il utilise le solveur de moindres carrés de SciPy. [Documentation SciPy — scipy.linalg.lstsq](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lstsq.html)

Il n’y a ni transformée de Fourier rapide, ni contrainte de positivité des coefficients, ni pénalité de régularisation sur leur taille.

La non-linéarité liée à $P$ est traitée à l’extérieur : on essaie plusieurs périodes et on résout ce problème linéaire pour chacune.

**Les résidus sont ensuite définis par :**

$$
r_i=m_i-\widehat m_i.
$$

En magnitude :

- $r_i>0$ signifie que l’objet mesuré est plus faible que le modèle ;
- $r_i<0$ signifie qu’il est plus brillant que le modèle.

Le programme recalcule explicitement :

$$
\chi^2=\sum_i w_i r_i^2.
$$

Source : [`weighted_fit`](https://github.com/mdeldem/asteroid/blob/c4381ce32cb127d52874e71a5f0f11bdbbc3b628/src/asteroid_lightcurve/period.py#L160-L215).

**7. Les indicateurs statistiques évaluent la qualité et la complexité des ajustements**

Le nombre de degrés de liberté utilisé par le code est :

$$
\nu=\max(N-K,1).
$$

Le $\chi^2$ réduit vaut :

$$
\boxed{
\chi^2_\nu=\frac{\chi^2}{\nu}.
}
$$

Si le modèle est adéquat et les erreurs correctement calibrées, un $\chi^2$ réduit voisin de 1 est une situation attendue dans le cadre statistique usuel.

Un résultat élevé peut cependant avoir plusieurs causes : erreurs sous-estimées, points atypiques, forme de courbe insuffisamment décrite, période incorrecte ou structure non représentée par le modèle.

**La convention de comptage du programme mérite d’être explicite :**

$$
K=G+2H
$$

compte seulement les coefficients de la matrice de conception. **La période recherchée n’est pas ajoutée à ce nombre**, même en mode automatique.

De plus, $K$ correspond au nombre de colonnes, pas au rang numérique effectif de la matrice.

La limitation $\nu\geq1$ évite une division par zéro ou par un nombre négatif. Elle ne rend pas statistiquement fiable un ajustement comportant trop de coefficients pour les données disponibles.

Le programme calcule trois critères d’information :

$$
\mathrm{AIC}=\chi^2+2K,
$$

$$
\mathrm{AICc}
=
\mathrm{AIC}
+
\frac{2K(K+1)}{N-K-1},
\qquad N>K+1,
$$

et :

$$
\boxed{
\mathrm{BIC}
=
\chi^2+K\ln\bigl(\max(N,2)\bigr).
}
$$

Lorsque $N\leq K+1$, l’AICc est fixé à $+\infty$.

Les sigles désignent :

- **AIC** : critère d’information d’Akaike ;
- **AICc** : correction de l’AIC pour les petits échantillons ;
- **BIC** : critère d’information bayésien.

Ces expressions combinent un terme d’erreur d’ajustement et une pénalité de complexité. Les valeurs les plus petites sont préférées.

Pour $N\geq8$, la pénalité par coefficient du BIC, $\ln N$, dépasse celle de l’AIC, qui vaut 2.

**Dans cette version, l’AIC et l’AICc sont des diagnostics. Le programme ne les utilise pas pour choisir le résultat.**

Le BIC intervient dans les comparaisons de périodes et d’ordres, mais la sélection automatique ajoute d’autres règles.

À ordre fixé, $K$ ne change pas. On a donc :

$$
\operatorname*{argmin}_{P}\mathrm{BIC}(P,H)
=
\operatorname*{argmin}_{P}\chi^2(P,H).
$$

Le raffinement de période par BIC est ainsi exactement équivalent à un raffinement par $\chi^2$, à cet ordre.

En passant de l’ordre $H$ à l’ordre $H+1$, on ajoute deux coefficients. Pour des données identiques, le nouveau modèle améliore le BIC si :

$$
\chi^2_H-\chi^2_{H+1}
>
2\ln\bigl(\max(N,2)\bigr).
$$

Cette relation explicite la diminution d’erreur nécessaire pour compenser la pénalité des deux coefficients supplémentaires.

Ces critères sont utilisés dans leur forme adaptée à un $\chi^2$ construit à partir des erreurs fournies. Le programme n’utilise pas une formulation alternative dans laquelle une variance de bruit inconnue serait réestimée dans la vraisemblance pour chaque modèle.

**8. La significativité des harmoniques sert à limiter l’augmentation de l’ordre**

Pour chaque harmonique $h$, les coefficients $a_h$ et $b_h$ définissent une amplitude :

$$
A_h=\sqrt{a_h^2+b_h^2}.
$$

C’est l’amplitude de la composante sinusoïdale. Sa variation crête-à-crête, si elle était seule, serait $2A_h$.

Cette amplitude harmonique ne doit pas être confondue avec l’amplitude totale de la courbe, calculée plus loin.

Le programme estime d’abord une matrice de covariance des coefficients :

$$
\boxed{
C_\beta
=
\left(X_w^TX_w\right)^+
\chi^2_\nu.
}
$$

Le symbole $+$ désigne la pseudo-inverse de Moore–Penrose, qui généralise l’inverse aux matrices pouvant être singulières.

La covariance décrit les variances estimées des coefficients et leurs covariations.

Le facteur $\chi^2_\nu$ ajuste globalement l’échelle de cette estimation. Il est appliqué même si $\chi^2_\nu<1$.

Pour l’amplitude $A_h$, le gradient par rapport aux deux coefficients est :

$$
\boldsymbol u_h
=
\begin{pmatrix}
a_h/A_h\\
b_h/A_h
\end{pmatrix}.
$$

En extrayant de $C_\beta$ la sous-matrice $C_h$ correspondant à $(a_h,b_h)$, le programme estime :

$$
\sigma_{A_h}^2
=
\boldsymbol u_h^T C_h\boldsymbol u_h.
$$

La significativité enregistrée est :

$$
\boxed{
S_h=\frac{A_h}{\sigma_{A_h}}.
}
$$

Elle mesure l’amplitude par rapport à son incertitude propagée.

Le calcul contient deux conventions particulières :

- si $A_h=0$, alors $S_h=0$ ;
- si la variance calculée est non finie ou non positive, alors $S_h=+\infty$.

Cette seconde convention est importante : une valeur infinie peut provenir d’un cas numérique dégénéré. Elle ne constitue pas nécessairement une preuve physique particulièrement forte.

L’estimation est également **conditionnelle à la période testée**. Elle ne propage pas l’incertitude sur la période dans celle des amplitudes.

Le seuil utilisé par la sélection automatique vaut :

$$
S_{\mathrm{seuil}}=3.
$$

Il ne représente pas une probabilité globale de fausse détection après toutes les périodes et tous les ordres essayés. Le programme ne calcule pas une telle probabilité.

Source : [`weighted_fit`, calcul des significativités](https://github.com/mdeldem/asteroid/blob/c4381ce32cb127d52874e71a5f0f11bdbbc3b628/src/asteroid_lightcurve/period.py#L184-L202).

**9. La première grille est uniforme en fréquence**

L’utilisateur fournit :

$$
0<P_{\min}<P_{\max}.
$$

Les unités sont les jours.

Pour $S$ échantillons, le programme construit :

$$
f_j
=
\frac{1}{P_{\max}}
+
\frac{j}{S-1}
\left(
\frac{1}{P_{\min}}-\frac{1}{P_{\max}}
\right),
$$

avec :

$$
j=0,\ldots,S-1,
$$

puis :

$$
P_j=\frac{1}{f_j}.
$$

Les fréquences sont croissantes ; les périodes correspondantes sont décroissantes, de $P_{\max}$ vers $P_{\min}$.

Le pas fréquentiel est :

$$
\Delta f
=
\frac{
1/P_{\min}-1/P_{\max}
}{
S-1
}.
$$

Le pas en période n’est pas constant. Localement :

$$
|\Delta P|\approx P^2\Delta f.
$$

L’intérêt de travailler en fréquence se voit dans le déphasage accumulé pendant une durée $T$ :

$$
\Delta\phi\approx T\Delta f.
$$

Pour l’harmonique $h$, ce déphasage devient :

$$
\Delta\phi_h\approx hT\Delta f.
$$

À densité de grille identique, une campagne plus longue ou des harmoniques plus élevées peuvent donc exiger une résolution fréquentielle plus fine.

**Le code n’utilise pas ces relations pour adapter automatiquement la grille.** Le nombre de points reste celui demandé par `--samples`, soit 8000 par défaut.

La fonction vérifie les bornes de période, mais n’impose pas elle-même une validation complète du nombre d’échantillons. Pour une véritable grille couvrant les deux bornes, il faut au moins deux points.

Source : [`period_grid`](https://github.com/mdeldem/asteroid/blob/c4381ce32cb127d52874e71a5f0f11bdbbc3b628/src/asteroid_lightcurve/period.py#L126-L130).

**10. Le périodogramme GLS compare une sinusoïde à des niveaux constants par groupe**

Le terme GLS désigne ici une implémentation interne de type **Lomb–Scargle généralisé**.

Le programme n’appelle pas `astropy.timeseries.LombScargle`. Il construit le score à partir de son propre ajustement pondéré.

Il commence par ajuster un modèle d’ordre zéro :

$$
\widehat m_i^{(0)}=c_0+d_{g_i}.
$$

Le modèle de référence n’est donc pas une constante unique pour toutes les observations. Il comporte **un niveau constant libre par groupe**.

À cet ordre, les niveaux ajustés correspondent aux moyennes pondérées propres aux groupes.

On obtient :

$$
\chi^2_0.
$$

Puis, pour chaque période $P_j$, on ajuste le modèle d’ordre 1 :

$$
\widehat m_i^{(1)}
=
c_0+d_{g_i}
+
a_1\cos\left(2\pi\frac{t_i-t_0}{P_j}\right)
+
b_1\sin\left(2\pi\frac{t_i-t_0}{P_j}\right),
$$

d’où :

$$
\chi^2_1(P_j).
$$

La puissance GLS est :

$$
\boxed{
\mathcal P(P_j)
=
\max\left(
0,\;
1-\frac{\chi^2_1(P_j)}{\chi^2_0}
\right)
}
$$

si $\chi^2_0>0$.

Lorsque $\chi^2_0=0$, le programme affecte une puissance nulle à toutes les périodes.

La puissance exprime la fraction de l’erreur pondérée du modèle de référence éliminée par l’ajout de la sinusoïde.

Par exemple :

$$
\mathcal P=0.8
$$

signifie que le $\chi^2$ est réduit de 80 % par rapport aux seuls niveaux constants par groupe.

Ce n’est pas une probabilité de 80 % que la période soit correcte.

Le meilleur point GLS affiché est simplement :

$$
P_{\mathrm{GLS,max}}
=
\operatorname*{argmax}_{P_j}\mathcal P(P_j).
$$

À ce stade, le programme ne calcule ni probabilité de fausse alarme, ni seuil statistique de détection, ni diagnostic explicite de fenêtre spectrale.

Le GLS sert avant tout à **localiser des zones prometteuses pour la recherche Fourier plus complète**.

Source : [`gls_power`](https://github.com/mdeldem/asteroid/blob/c4381ce32cb127d52874e71a5f0f11bdbbc3b628/src/asteroid_lightcurve/period.py#L218-L225).

**11. Les pics GLS deviennent des candidats, avec traitement explicite des ambiguïtés de période**

Un indice intérieur $j$ est considéré comme un maximum local si :

$$
\mathcal P_j\geq\mathcal P_{j-1}
$$

et :

$$
\mathcal P_j\geq\mathcal P_{j+1}.
$$

Les égalités sont acceptées. Un plateau peut donc fournir plusieurs maxima locaux.

Les extrémités de la grille ne sont pas incluses dans cette recherche de maxima intérieurs.

Si aucun maximum intérieur n’est trouvé, tous les points deviennent admissibles au classement.

Les maxima sont ensuite triés par puissance décroissante et les premiers sont conservés, dans la limite de :

```text
--gls-candidates
```

dont la valeur par défaut est 20.

**Une conséquence précise de cette règle :** si un maximum global se trouve à une borne et qu’il existe au moins un maximum intérieur, ce maximum de bord peut être absent de la liste des candidats, même s’il est affiché comme meilleur point GLS.

Pour chaque pic retenu de période $P_g$, le programme applique les multiplicateurs demandés.

Par défaut :

$$
\mu\in\{0.5,\;1,\;2\}.
$$

Les candidats sont donc :

$$
P_c=\mu P_g.
$$

Cela permet de tester :

$$
\frac{P_g}{2},\qquad P_g,\qquad 2P_g.
$$

L’intention est notamment de traiter les courbes présentant deux motifs photométriques par rotation.

Mathématiquement, si le signal est dominé par :

$$
\cos\left(4\pi\frac{t}{P}\right),
$$

un modèle d’ordre 1 peut l’ajuster avec la période $P/2$. Tester $2P_g$ réintroduit alors la possibilité d’une période fondamentale deux fois plus longue.

Cela ne garantit pas que l’ambiguïté puisse être levée. Si toutes les harmoniques impaires sont nulles :

$$
F(\phi+1/2)=F(\phi),
$$

et la courbe photométrique se répète exactement après une demi-période. La décomposition seule ne peut imposer une interprétation physique unique.

**Les candidats sont ensuite bornés et dédupliqués.**

Un candidat situé hors de :

$$
[P_{\min},P_{\max}]
$$

est éliminé.

Deux périodes $P_a$ et $P_b$ sont considérées comme des doublons si :

$$
|P_a-P_b|
\leq
\varepsilon_{\mathrm{dup}}\max(P_a,P_b),
$$

avec :

$$
\varepsilon_{\mathrm{dup}}
=
\max\left(10^{-8},\frac{1}{S}\right).
$$

Cette tolérance dépend du nombre d’échantillons de la grille initiale.

Le premier candidat rencontré est conservé. L’ordre des pics et l’ordre des multiplicateurs peuvent donc déterminer l’origine attribuée à un candidat dédupliqué.

Avec les réglages par défaut, le nombre maximal théorique est :

$$
20\times3=60
$$

candidats, avant élimination des doublons et des valeurs hors bornes.

La sélection des 20 pics intervient avant ces éliminations : le programme ne va pas chercher des pics supplémentaires pour remplacer ceux dont tous les candidats disparaissent.

Enfin, si aucun candidat ne subsiste, il reprend le meilleur point global du GLS avec un multiplicateur égal à 1.

Source : [`gls_peak_candidates`](https://github.com/mdeldem/asteroid/blob/c4381ce32cb127d52874e71a5f0f11bdbbc3b628/src/asteroid_lightcurve/period.py#L228-L287).

**12. Chaque candidat est raffiné séparément pour chaque ordre de Fourier**

Pour chaque candidat $P_c$, le programme parcourt tous les ordres demandés.

Avec les valeurs par défaut :

$$
H\in\{2,3,4,5,6\}.
$$

Il ne choisit donc pas d’abord un ordre unique pour raffiner toutes les périodes.

Chaque couple :

$$
(P_c,H)
$$

fait l’objet d’une recherche locale indépendante.

La demi-largeur relative du voisinage est :

$$
w=\texttt{--refine-width},
$$

avec :

$$
w=0.01
$$

par défaut.

Avant prise en compte des bornes globales, l’intervalle est :

$$
P_{\mathrm{bas}}
=
\max\left(P_c(1-w),\frac{P_c}{2}\right),
$$

$$
P_{\mathrm{haut}}=P_c(1+w).
$$

Il est ensuite tronqué à l’intervalle global demandé :

$$
P_{\mathrm{bas}}\leftarrow
\max(P_{\mathrm{bas}},P_{\min}),
$$

$$
P_{\mathrm{haut}}\leftarrow
\min(P_{\mathrm{haut}},P_{\max}).
$$

Pour $w=0.01$, on recherche donc normalement dans un voisinage de $\pm1\,\%$.

Si la borne supérieure n’est pas strictement supérieure à la borne inférieure, le programme ajuste directement la période initiale sans créer de grille locale.

Sinon, il construit une nouvelle grille uniforme en fréquence, comportant :

```text
--candidate-refine-samples
```

points, soit 300 par défaut.

Chaque période locale donne lieu à un ajustement complet : offsets, coefficients, résidus et statistiques.

Le résultat du couple candidat–ordre est :

$$
P_{c,H}^{*}
=
\operatorname*{argmin}_{P\in\mathcal G_c}
\chi^2(P,H),
$$

où $\mathcal G_c$ est la grille locale.

Le programme conserve **un seul meilleur ajustement par couple candidat–ordre**, pas les 300 essais individuels dans la table des candidats.

Tous ces meilleurs ajustements sont ensuite triés par BIC.

Ce procédé est un raffinement **discret**. Il n’utilise pas un minimiseur continu de la période, ni une interpolation parabolique autour du meilleur point.

Une limite en découle : l’optimum réel peut se situer entre deux points de la grille.

Sources : [`refine_period`](https://github.com/mdeldem/asteroid/blob/c4381ce32cb127d52874e71a5f0f11bdbbc3b628/src/asteroid_lightcurve/period.py#L313-L339) et [`search_period_order_candidates`](https://github.com/mdeldem/asteroid/blob/c4381ce32cb127d52874e71a5f0f11bdbbc3b628/src/asteroid_lightcurve/period.py#L342-L377).

**13. La sélection automatique privilégie une famille de périodes définie à bas ordre**

Cette sélection est l’élément le plus spécifique du projet.

Elle poursuit une intention conservatrice : éviter qu’un modèle très flexible soit préféré uniquement parce qu’il obtient un bon ajustement dans une autre famille de périodes.

**Le meilleur BIC global est d’abord mémorisé.**

Parmi tous les ajustements candidats raffinés :

$$
(P_{\mathrm{BIC}},H_{\mathrm{BIC}})
=
\operatorname*{argmin}_{c,H}\mathrm{BIC}_{c,H}.
$$

Ce résultat est conservé à des fins de comparaison.

Il n’est pas automatiquement adopté.

**Une période de référence est ensuite choisie.**

L’ordre de référence est :

$$
H_{\mathrm{ref}}
=
\begin{cases}
2,&\text{si l’ordre 2 fait partie des ordres demandés},\\
\min(H_{\mathrm{demandés}}),&\text{sinon}.
\end{cases}
$$

À cet ordre, on retient le candidat possédant le plus petit BIC.

Sa période devient :

$$
P_{\mathrm{ref}}.
$$

Avec les paramètres par défaut, **l’ordre 2 détermine donc la famille de périodes dans laquelle le programme poursuivra sa sélection d’ordre**.

Cela ne signifie pas que le résultat final sera nécessairement d’ordre 2.

**Le programme construit ensuite une famille dite stable.**

Pour chaque ordre demandé, seuls les ajustements respectant :

$$
\boxed{
|P-P_{\mathrm{ref}}|
\leq
\varepsilon_{\mathrm{stab}}P_{\mathrm{ref}}
}
$$

sont admissibles, avec :

$$
\varepsilon_{\mathrm{stab}}=0.02.
$$

La fenêtre vaut donc $\pm2\,\%$ autour de la référence.

Pour chaque ordre, le meilleur BIC à l’intérieur de cette fenêtre est conservé.

Le mot « stable » a ici une définition strictement algorithmique : **proche de la référence à moins de 2 %**.

Il ne signifie pas que la période a été retrouvée par rééchantillonnage, sur plusieurs sous-ensembles de nuits ou dans un intervalle d’incertitude statistique commun.

La tolérance de 2 % est indépendante :

- de la durée des observations ;
- des incertitudes de période ;
- de la largeur des pics ;
- de la précision photométrique.

**L’ordre est alors augmenté progressivement.**

Les ajustements admissibles sont parcourus par ordre croissant.

Le premier est retenu initialement.

Un ordre suivant n’est accepté que si les trois conditions suivantes sont simultanément satisfaites :

$$
\mathrm{BIC}_{\mathrm{nouveau}}
<
\mathrm{BIC}_{\mathrm{précédent}},
$$

$$
\chi^2_{\nu,\mathrm{nouveau}}
\leq
\chi^2_{\nu,\mathrm{précédent}},
$$

$$
S_{H_{\mathrm{nouveau}}}\geq3.
$$

Le « précédent » est le dernier ajustement accepté.

Le programme demande donc :

1. une amélioration du BIC ;
2. une absence de dégradation du $\chi^2$ réduit ;
3. une amplitude suffisamment significative pour la dernière harmonique du nouvel ordre.

**Au premier échec, la progression s’arrête définitivement.**

Supposons que :

- l’ordre 2 soit retenu initialement ;
- l’ordre 3 améliore le BIC, mais présente $S_3=2$ ;
- l’ordre 4 ait un BIC encore meilleur.

Le résultat reste d’ordre 2. L’ordre 4 n’est pas examiné pour une acceptation ultérieure.

Ce n’est donc pas la recherche du minimum du BIC parmi tous les ordres stables.

Plusieurs détails complètent cette définition :

- le premier ordre retenu n’est pas soumis au seuil $S_H\geq3$ ;
- si l’ordre 1 est demandé et présent dans la famille, il peut être le point de départ, même si la référence est définie à l’ordre 2 ;
- si un ordre n’a aucun ajustement admissible, il est absent de la liste ;
- en cas de saut entre ordres disponibles, seul le dernier harmonique du nouvel ordre est testé, pas tous ceux ajoutés depuis le précédent ;
- les périodes comparées peuvent légèrement différer d’un ordre à l’autre, puisque chaque ajustement a été raffiné séparément.

La stratégie peut donc volontairement retenir une solution dont le BIC est supérieur au meilleur BIC global.

Elle dépend aussi fortement de la référence : si l’ordre 2 choisit une mauvaise famille de périodes, les ordres supérieurs situés hors de la fenêtre de 2 % ne peuvent pas la remplacer.

Source : [`select_stable_period`](https://github.com/mdeldem/asteroid/blob/c4381ce32cb127d52874e71a5f0f11bdbbc3b628/src/asteroid_lightcurve/period.py#L380-L437).

**14. Le raffinement final conserve l’ordre choisi, puis un balayage global sert au diagnostic**

Après la sélection hiérarchique, le programme conserve l’ordre choisi $H_*$ et raffine une nouvelle fois la période.

Ce second raffinement utilise :

- la même demi-largeur relative `--refine-width` ;
- les mêmes bornes globales ;
- `--refine-samples`, soit 2000 points par défaut.

Il produit :

$$
P_*
=
\operatorname*{argmin}_{P\in\mathcal G_{\mathrm{finale}}}
\chi^2(P,H_*).
$$

Tous les coefficients sont réajustés à chacune de ces périodes.

**L’ordre n’est pas resélectionné après ce raffinement.**

Le programme ne réapplique pas non plus les trois conditions de progression ni le contrôle de proximité à la période de référence.

La nouvelle grille n’inclut pas nécessairement exactement la période initiale. Comme l’algorithme ne conserve pas explicitement celle-ci comme solution de secours, une très légère dégradation numérique de l’ajustement est possible malgré une grille plus dense.

Pour donner un ordre de grandeur du pas local, loin des bornes et pour une faible largeur $w$ :

$$
|\Delta P|
\approx
\frac{2wP}{R-1},
$$

où $R$ est le nombre de points du raffinement.

Ce pas caractérise la résolution de la recherche, **pas l’incertitude statistique sur la période**.

Le résultat final est ensuite injecté dans la synthèse de sélection. En revanche, les ajustements de la table des candidats restent ceux du premier raffinement. Il peut donc exister un petit écart entre la période finale et celle du candidat correspondant.

**Le programme effectue ensuite un nouveau balayage de la grille initiale.**

À chaque période $P_j$, il ajuste le modèle d’ordre $H_*$ et calcule :

$$
Q(P_j)=-\mathrm{BIC}(P_j,H_*).
$$

Ce score est représenté dans `fourier_period_search.png`.

Le signe négatif permet de représenter les meilleurs ajustements sous forme de maxima.

À cet ordre fixé :

$$
Q(P_j)
=
-\chi^2(P_j,H_*)-\text{constante}.
$$

Le graphique équivaut donc, à un décalage vertical près, à un tracé de $-\chi^2$.

**Ce balayage ne réouvre pas la sélection.** Un maximum plus élevé ailleurs dans le graphique ne provoque pas le remplacement automatique de $P_*$.

Source : [`run_period_search`](https://github.com/mdeldem/asteroid/blob/c4381ce32cb127d52874e71a5f0f11bdbbc3b628/src/asteroid_lightcurve/cli.py#L557-L613).

**15. Le mode à période imposée utilise une autre règle de choix d’ordre**

L’option :

```text
--period
```

fournit une période $P_{\mathrm{fixe}}>0$, toujours en jours.

Dans ce mode, il n’y a :

- ni grille initiale de recherche ;
- ni sélection de pics GLS ;
- ni multiplication des candidats ;
- ni raffinement de la période choisie ;
- ni sélection dans une famille stable.

Chaque ordre demandé est ajusté à $P_{\mathrm{fixe}}$.

Le choix est simplement :

$$
\boxed{
H_*
=
\operatorname*{argmin}_{H}
\mathrm{BIC}(P_{\mathrm{fixe}},H).
}
$$

Le seuil de significativité de 3 et la règle d’arrêt au premier ordre refusé ne sont pas utilisés.

Ainsi, deux analyses effectuées à des périodes presque identiques peuvent choisir des ordres différents selon qu’elles proviennent de la recherche automatique ou du mode à période imposée.

Si un seul ordre est demandé, il n’y a plus de sélection d’ordre : le programme ajuste directement ce modèle.

Les options de bornes et de grille de recherche n’interviennent pas dans ce choix lorsque `--period` est fourni.

L’association de `--period` et `--residual-filter` est explicitement refusée.

Une nuance concerne l’incertitude : le programme la calcule tout de même par exploration de périodes voisines. Il conserve la période imposée comme résultat, mais évalue d’autres périodes pour construire le profil décrit ci-dessous.

Sources : [`search_fourier`](https://github.com/mdeldem/asteroid/blob/c4381ce32cb127d52874e71a5f0f11bdbbc3b628/src/asteroid_lightcurve/period.py#L290-L310) et [`cmd_search`](https://github.com/mdeldem/asteroid/blob/c4381ce32cb127d52874e71a5f0f11bdbbc3b628/src/asteroid_lightcurve/cli.py#L679-L728).

**16. L’incertitude de période est estimée par un profil de $\chi^2$**

L’ordre retenu $H_*$ reste fixé.

Pour chaque période voisine, tous les coefficients linéaires sont réajustés. Le profil est donc :

$$
\boxed{
q(P)
=
\min_{\boldsymbol\beta}
\chi^2(P,H_*,\boldsymbol\beta).
}
$$

Cette minimisation inclut les décalages de groupes.

Le programme cherche des bornes telles que :

$$
q(P)=q(P_*)+\Delta\chi^2.
$$

Il réalise deux estimations.

La première utilise :

$$
\Delta\chi^2=1.
$$

Ce seuil correspond à la convention habituelle d’une incertitude à un écart-type pour un paramètre, sous les hypothèses statistiques appropriées et autour d’un minimum correctement déterminé. Ce n’est pas une garantie indépendante de la qualité du modèle ou des erreurs d’entrée. [Documentation CERN ROOT — Minuit 2](https://root.cern.ch/root/htmldoc/guides/minuit2/Minuit2.html)

La seconde utilise :

$$
\boxed{
\Delta\chi^2_{\mathrm{rééchelonné}}
=
\max(1,\chi^2_\nu).
}
$$

Elle élargit normalement l’intervalle lorsque le $\chi^2$ réduit dépasse 1.

L’idée peut s’interpréter ainsi : si toutes les incertitudes étaient multipliées par :

$$
s=\sqrt{\max(1,\chi^2_\nu)},
$$

le nouveau $\chi^2$ serait divisé par $s^2$. Un seuil de 1 dans cette nouvelle échelle correspond au seuil $\max(1,\chi^2_\nu)$ dans l’échelle initiale.

**Le code ne modifie toutefois pas les incertitudes des observations.** Il modifie seulement le seuil recherché dans le profil.

Cette opération ne corrige pas une mauvaise période, un alias ou une structure absente du modèle.

**La recherche des bornes s’effectue séparément à gauche et à droite.**

Le premier écart testé vaut :

$$
s_0=\max(P_*\times10^{-6},10^{-8}\ \text{jour}).
$$

Les distances successives au centre sont :

$$
s_j=s_0\,1.5^j.
$$

Les périodes évaluées sont :

$$
P_j=P_*-s_j
$$

à gauche, et :

$$
P_j=P_*+s_j
$$

à droite.

Les déplacements sont calculés depuis $P_*$. Ce ne sont pas des pas ajoutés successivement à la dernière période.

Dès que le profil dépasse le niveau cible, le dernier point sous le seuil et le premier point au-dessus encadrent une racine de :

$$
F(P)=q(P)-q(P_*)-\Delta\chi^2.
$$

Le programme appelle alors `scipy.optimize.brentq`, avec au maximum 100 itérations.

La méthode de Brent combine un encadrement de la racine, des étapes de bissection et des interpolations pour résoudre $F(P)=0$. Elle intervient ici **pour trouver les bornes**, pas pour minimiser la période centrale. [Documentation SciPy — scipy.optimize.brentq](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brentq.html)

La recherche extérieure est limitée à :

- 80 essais par côté ;
- un déplacement de 25 % de la période centrale.

Les limites sont donc :

$$
0.75P_*
\quad\text{et}\quad
1.25P_*.
$$

Le code s’arrête avant d’évaluer un point atteignant ou dépassant ces limites.

Si aucune traversée n’a été détectée, la borne correspondante est absente.

**Une borne absente signifie donc : “non trouvée par cette procédure dans sa zone d’exploration”.** Cela ne prouve pas que le profil ne possède aucune borne à plus grande distance.

Autres conséquences de la procédure :

- les bornes globales `--min-period` et `--max-period` ne limitent pas cette exploration ;
- les pas croissants peuvent manquer une excursion étroite au-dessus du seuil dans un profil très irrégulier ;
- le profil ne combine pas plusieurs ordres de Fourier ;
- il ne décrit pas l’incertitude de choix entre plusieurs familles de périodes éloignées ;
- il ne remplace pas une analyse des alias.

**Les erreurs gauche et droite sont conservées séparément :**

$$
\sigma_P^{-}=P_*-P_{\mathrm{bas}},
$$

$$
\sigma_P^{+}=P_{\mathrm{haut}}-P_*.
$$

Pour les titres des graphiques, le programme utilise une valeur dite symétrique :

$$
\sigma_{P,\mathrm{sym}}
=
\frac{\sigma_P^-+\sigma_P^+}{2}
$$

si les deux côtés sont disponibles.

Si un seul côté est disponible, il utilise ce seul côté comme valeur symétrique. Si aucun ne l’est, il n’affiche pas d’incertitude.

Les graphiques emploient la version rééchelonnée.

Enfin, en mode à période imposée, $P_*$ peut ne pas être un minimum du profil. Dans ce cas, les traversées calculées autour du niveau $q(P_*)+\Delta\chi^2$ ne doivent pas être interprétées automatiquement comme un intervalle de confiance centré sur un estimateur optimal.

Source : [`estimate_period_uncertainty`](https://github.com/mdeldem/asteroid/blob/c4381ce32cb127d52874e71a5f0f11bdbbc3b628/src/asteroid_lightcurve/period.py#L440-L476).

**17. La phase, l’amplitude et les dispersions sont calculées à partir de l’ajustement final**

La phase d’une observation est :

$$
\boxed{
\phi_i
=
\left(
\frac{t_i-t_0}{P_*}
\right)\bmod1.
}
$$

Elle peut aussi s’écrire :

$$
\phi_i
=
x_i-\lfloor x_i\rfloor,
\qquad
x_i=\frac{t_i-t_0}{P_*}.
$$

Elle appartient à :

$$
[0,1[.
$$

Le modèle de régression utilise directement les arguments des sinus et cosinus. Le calcul explicite de cette phase sert principalement aux représentations repliées.

Pour afficher deux cycles, les observations sont dupliquées graphiquement :

$$
(\phi_i,m_i)
\quad\text{et}\quad
(\phi_i+1,m_i).
$$

Cette duplication ne double ni le nombre de points ajustés ni leur poids.

**L’alignement retire les décalages ajustés.**

Les magnitudes alignées sont :

$$
m_{i,\mathrm{aligné}}=m_i-d_{g_i}.
$$

Le modèle aligné est :

$$
\widehat m_{i,\mathrm{aligné}}
=
\widehat m_i-d_{g_i}.
$$

Tous les groupes sont ainsi rapportés au niveau du groupe de référence.

Les résidus restent identiques :

$$
m_{i,\mathrm{aligné}}
-
\widehat m_{i,\mathrm{aligné}}
=
m_i-\widehat m_i
=
r_i.
$$

L’alignement est donc une transformation de présentation cohérente avec le modèle ajusté.

**L’amplitude totale est mesurée sur la courbe de Fourier.**

On définit :

$$
F(\phi)
=
c_0+
\sum_{h=1}^{H_*}
\left[
a_h\cos(2\pi h\phi)
+
b_h\sin(2\pi h\phi)
\right].
$$

Les décalages de groupes n’interviennent pas.

Le programme évalue $F$ sur 1000 phases régulièrement espacées entre 0 et 1, bornes incluses.

L’amplitude publiée est :

$$
\boxed{
A_{\mathrm{modèle}}
=
\max_j F(\phi_j)
-
\min_j F(\phi_j).
}
$$

C’est une amplitude crête-à-crête en magnitude.

Elle n’est :

- ni l’écart entre les mesures extrêmes ;
- ni la somme des amplitudes $A_h$ ;
- ni une mesure directe d’allongement physique ;
- ni accompagnée d’une incertitude propagée.

Les extrema sont ceux de la grille de 1000 points. Le programme ne résout pas analytiquement l’équation $F'(\phi)=0$.

**La dispersion par groupe est un écart-type non pondéré des résidus.**

Pour un groupe contenant $N_g\geq2$ mesures :

$$
s_g
=
\sqrt{
\frac{
\sum_{i:g_i=g}(r_i-\overline r_g)^2
}{
N_g-1
}
},
$$

où $\overline r_g$ est la moyenne arithmétique de ses résidus.

Pour un seul point, la dispersion est fixée à zéro.

Cette grandeur :

- n’utilise pas les poids photométriques ;
- n’est pas le $\chi^2$ réduit du groupe ;
- ne remplace pas les incertitudes individuelles ;
- n’est pas réinjectée dans les ajustements ultérieurs.

Elle est affichée en millimagnitudes dans les légendes :

$$
1\ \mathrm{mmag}=10^{-3}\ \mathrm{mag}.
$$

Source : [calculs de phase, alignement, amplitude et dispersion](https://github.com/mdeldem/asteroid/blob/c4381ce32cb127d52874e71a5f0f11bdbbc3b628/src/asteroid_lightcurve/plotting.py#L24-L110).

**18. Le filtrage robuste rejette des observations d’après leur écart au premier modèle**

Ce traitement est optionnel et activé par :

```text
--residual-filter
```

Il intervient après la recherche initiale et utilise les résidus de son ajustement final.

Son objectif est de limiter l’influence des observations très éloignées de cette première courbe.

**Le centre des résidus est leur médiane :**

$$
r_c=\operatorname{médiane}(r_i).
$$

Le programme calcule ensuite :

$$
D_i=|r_i-r_c|.
$$

Puis :

$$
\mathrm{MAD}
=
\operatorname{médiane}(D_i).
$$

MAD signifie *Median Absolute Deviation*, soit médiane des écarts absolus à la médiane.

L’échelle robuste est :

$$
\boxed{
s_r=1.4826\,\mathrm{MAD}.
}
$$

Le facteur 1.4826 rend cette mesure comparable à l’écart-type pour une distribution normale, dans le sens de la cohérence asymptotique. [Documentation R — Median Absolute Deviation](https://www.stat.ethz.ch/R-manual/R-devel/RHOME/library/stats/help/mad.html)

Si cette échelle est non finie ou non positive, le code utilise l’écart-type usuel des résidus avec le diviseur $N-1$, lorsque suffisamment de points existent.

Si cette solution de remplacement est également non exploitable, il emploie :

$$
s_r=10^{-12}\ \mathrm{mag}.
$$

**Le seuil demandé peut être relatif ou absolu.**

En fonctionnement normal :

$$
T_{\mathrm{demandé}}=q\,s_r,
$$

avec :

$$
q=\texttt{--residual-filter-sigma}=3.5
$$

par défaut.

Si `--residual-filter-threshold-mag` est fourni :

$$
T_{\mathrm{demandé}}
=
\text{valeur fournie en magnitude}.
$$

Ce seuil absolu remplace le seuil fondé sur la MAD.

Le rejet initial est défini par :

$$
\boxed{
D_i>T_{\mathrm{demandé}}.
}
$$

L’inégalité est stricte.

Trois propriétés sont importantes :

- le filtrage est symétrique autour de la médiane ;
- il est calculé globalement, pas séparément pour chaque groupe ;
- il porte sur des écarts en magnitude, pas sur les résidus normalisés $r_i/\sigma_i$.

Un point peu précis et un point très précis sont donc comparés au même seuil absolu une fois celui-ci établi.

**Le nombre de rejets est plafonné.**

Avec :

$$
\rho=
\texttt{--residual-filter-max-reject-fraction},
$$

le nombre maximal est :

$$
N_{\mathrm{rejet,max}}
=
\lfloor N\rho\rfloor.
$$

Par défaut :

$$
\rho=0.25.
$$

Si le nombre de points dépassant le seuil excède cette limite, le programme rejette seulement les :

$$
\lfloor N\rho\rfloor
$$

plus grands écarts $D_i$.

Cela signifie que des points dépassant le seuil demandé peuvent être conservés parce que le quota maximal est atteint.

Si le quota entier vaut zéro, aucun point n’est rejeté.

Le programme impose ensuite un minimum de points conservés :

$$
N_{\mathrm{conservé}}
\geq
\texttt{--residual-filter-min-points},
$$

soit 30 par défaut.

Si cette condition échoue, il abandonne avec une erreur. Il ne réduit pas automatiquement le nombre de rejets pour atteindre ce minimum.

Ce contrôle est global. Il ne garantit ni un nombre minimal de points par groupe, ni un nombre suffisant par rapport à :

$$
K=G+2H.
$$

**Le seuil effectif enregistré a une définition particulière.**

S’il existe au moins un rejet :

$$
T_{\mathrm{effectif}}
=
\min_{i\ \mathrm{rejeté}}D_i.
$$

Sinon :

$$
T_{\mathrm{effectif}}=T_{\mathrm{demandé}}.
$$

Le seuil effectif est donc le plus petit écart parmi les points effectivement rejetés. Ce n’est pas toujours le seuil qui avait été demandé.

En présence d’ex æquo et d’un plafonnement, une ligne horizontale placée à ce niveau ne suffit pas nécessairement à reproduire exactement le masque de rejet.

Source : [`residual_filter_from_fit`](https://github.com/mdeldem/asteroid/blob/c4381ce32cb127d52874e71a5f0f11bdbbc3b628/src/asteroid_lightcurve/cli.py#L398-L447).

**19. La seconde passe refait toute la recherche sur les observations conservées**

Si aucun point n’a été rejeté, aucune nouvelle recherche n’est lancée.

Sinon, le programme construit un nouveau jeu avec :

- les dates conservées ;
- les magnitudes originales conservées ;
- les incertitudes originales conservées ;
- les groupes encore présents.

Les groupes devenus vides sont supprimés et les identifiants restants sont renumérotés.

La recherche complète recommence ensuite avec les mêmes paramètres :

- mêmes bornes de période ;
- même nombre de points de grille ;
- mêmes multiplicateurs ;
- mêmes ordres ;
- mêmes paramètres de raffinement.

Les corrections temporelles ne sont pas recalculées : les dates ont déjà été corrigées.

En revanche, le nouveau minimum temporel peut être différent si les premières observations ont été rejetées. L’origine $t_0$ est alors recalculée sur le sous-ensemble.

Le modèle de la seconde passe est toujours :

$$
m_i
\approx
c_0+d_{g_i}+F(t_i;P,H).
$$

Ce n’est pas :

$$
r_i\approx F_{\mathrm{secondaire}}(t_i).
$$

Il faut donc interpréter ce traitement comme une **nouvelle estimation de la modulation principale sur un échantillon filtré**.

Il n’y a qu’une seconde passe :

- pas de boucle de rejet jusqu’à convergence ;
- pas de troisième recherche automatique ;
- pas de rejet itératif réévalué sur les nouveaux résidus.

Les résultats initiaux sont conservés et ceux de la seconde passe sont produits séparément avec le préfixe `residual_filtered_`.

Le programme ne décide pas automatiquement que le résultat filtré remplace scientifiquement le premier.

Comparer directement les BIC des deux passes comme s’il s’agissait de modèles concurrents ajustés aux mêmes observations serait incorrect : le nombre de points et éventuellement le nombre de groupes ont changé.

Enfin, ce filtrage ne distingue pas un défaut de mesure d’un événement astrophysique réel. Un événement mutuel produisant un fort résidu peut être rejeté comme n’importe quel autre écart important.

Sources : [`subset_lightcurve`](https://github.com/mdeldem/asteroid/blob/c4381ce32cb127d52874e71a5f0f11bdbbc3b628/src/asteroid_lightcurve/models.py#L91-L113) et [seconde passe dans `cmd_search`](https://github.com/mdeldem/asteroid/blob/c4381ce32cb127d52874e71a5f0f11bdbbc3b628/src/asteroid_lightcurve/cli.py#L768-L858).

**20. Les paramètres accessibles permettent de contrôler des étapes bien distinctes**

Les options de recherche sont :

| Option | Défaut | Rôle exact |
|---|---:|---|
| `--min-period` | Aucun | Période minimale, en jours. Requise en recherche automatique. |
| `--max-period` | Aucun | Période maximale, en jours. Requise en recherche automatique. |
| `--period` | Aucun | Période imposée, en jours ; active le mode d’ajustement à période fixée. |
| `--samples` | `8000` | Nombre de points de la grille initiale uniforme en fréquence. Sert aussi au balayage Fourier de diagnostic. |
| `--orders` | `2:6` | Ordre unique ou intervalle inclusif des ordres de Fourier à tester. |
| `--gls-candidates` | `20` | Nombre maximal de pics GLS retenus avant multiplication et déduplication. |
| `--gls-multipliers` | `0.5,1,2` | Facteurs appliqués aux périodes des pics GLS. |
| `--refine-width` | `0.01` | Demi-largeur relative des raffinements locaux, candidats et final. |
| `--candidate-refine-samples` | `300` | Nombre de points par raffinement d’un couple candidat–ordre. |
| `--refine-samples` | `2000` | Nombre de points du raffinement final à ordre fixé. |

Le format `2:6` inclut les ordres 2, 3, 4, 5 et 6. Le format `4` n’en demande qu’un.

**Il n’existe pas de maximum scientifique d’ordre fixé à 6 dans le code.** Six est seulement la borne supérieure par défaut.

Le parseur ne met pas en place de règle automatique reliant l’ordre maximal au nombre de mesures, à la couverture en phase ou au rapport signal sur bruit.

Un ordre zéro est représentable par le moteur et utilisé pour le modèle de référence du GLS. Mais à cet ordre le modèle est indépendant de la période : il ne permet pas d’estimer une périodicité.

Les paramètres de filtrage sont :

| Option | Défaut | Rôle exact |
|---|---:|---|
| `--residual-filter` | Désactivé | Active le rejet puis, si nécessaire, une seconde recherche complète. |
| `--residual-filter-sigma` | `3.5` | Multiplicateur de l’échelle robuste des résidus. |
| `--residual-filter-threshold-mag` | Aucun | Seuil absolu en magnitude, prioritaire sur le seuil relatif. |
| `--residual-filter-max-reject-fraction` | `0.25` | Fraction maximale de points rejetés. Doit appartenir à $]0,1[$. |
| `--residual-filter-min-points` | `30` | Nombre minimal global de points à conserver. |

Le multiplicateur sigma et le seuil absolu, lorsqu’il est fourni, doivent être strictement positifs. Le multiplicateur sigma est vérifié même si un seuil absolu est utilisé.

Les options temporelles sont :

| Option | Défaut | Rôle exact |
|---|---:|---|
| `--keep-start-time` | Désactivé | Conserve les dates fournies sans conversion au milieu de pose. |
| `--no-ephemeris` | Désactivé | Supprime l’interrogation Horizons et la correction héliocentrique. |
| `--horizons-timeout` | `30.0` s | Délai maximal des requêtes Horizons ; ne règle pas la précision du calcul. |

L’option `--out`, par défaut `output`, choisit uniquement la destination des résultats.

**Les effets des paramètres de recherche ne sont pas interchangeables.**

Augmenter `--samples` affine l’exploration initiale et peut faire apparaître des pics jusque-là mal échantillonnés. Cela modifie aussi la tolérance de déduplication.

Augmenter `--gls-candidates` permet d’examiner davantage de pics déjà présents dans le périodogramme.

Augmenter `--candidate-refine-samples` améliore la localisation des solutions comparées **avant** la sélection de famille et d’ordre.

Augmenter `--refine-samples` améliore seulement le raffinement **après** cette sélection. Cela ne récupère pas un candidat exclu auparavant.

Augmenter `--refine-width` explore un voisinage plus large. À nombre de points constant, cela rend la grille locale moins fine.

Élargir `--orders` autorise des formes plus complexes, mais ne force pas la sélection automatique à atteindre les ordres les plus élevés.

Source : [définition des options](https://github.com/mdeldem/asteroid/blob/c4381ce32cb127d52874e71a5f0f11bdbbc3b628/src/asteroid_lightcurve/cli.py#L957-L1034).

**21. Plusieurs réglages importants restent internes au programme**

Ces valeurs influencent le comportement sans être accessibles par une option de la commande :

| Réglage interne | Valeur ou règle actuelle | Portée |
|---|---|---|
| Décalages par groupe | Activés dans tous les appels de la chaîne principale | Un niveau constant libre par groupe |
| Ordre de référence | 2 si demandé, sinon le plus petit ordre demandé | Ancre la famille de périodes |
| Tolérance de stabilité | `0.02` | Fenêtre de $\pm2\,\%$ autour de la référence |
| Seuil de significativité | `3.0` | Condition d’acceptation du dernier harmonique |
| Tolérance de déduplication | $\max(10^{-8},1/S)$ | Regroupement des périodes candidates proches |
| Seuil du profil brut | $\Delta\chi^2=1$ | Première estimation d’incertitude |
| Seuil du profil rééchelonné | $\max(1,\chi^2_\nu)$ | Deuxième estimation d’incertitude |
| Étendue maximale du profil | $\pm25\,\%$ | Limite de recherche des bornes |
| Facteur de croissance des pas | `1.5` | Exploration du profil |
| Nombre maximal d’essais | `80` par côté | Recherche d’un encadrement |
| Maximum d’itérations de Brent | `100` | Résolution d’une borne encadrée |
| Facteur de calibration MAD | `1.4826` | Échelle robuste des résidus |
| Échelle de dernier recours | $10^{-12}$ mag | Cas de dispersion nulle ou invalide |
| Échantillonnage de l’amplitude | `1000` phases | Estimation des extrema du modèle |

Certaines fonctions Python possèdent des arguments permettant de changer ces valeurs par appel direct.

Par exemple, `weighted_fit` peut fonctionner sans offsets de groupes. Mais la chaîne principale ne fournit pas d’option pour cela et utilise les offsets.

De même, la fonction de sélection possède des paramètres de stabilité et de significativité, mais la commande utilise les valeurs actuelles sans les exposer à l’utilisateur.

Les deux valeurs de 25 % ont des rôles entièrement différents :

- 25 % de déplacement maximal en période pour l’incertitude ;
- 25 % de points rejetés au maximum pour le filtrage.

**22. Les sorties ne représentent pas toutes le même niveau de sélection**

Cette distinction est utile pour relire les résultats sans leur attribuer une signification erronée.

| Sortie | Signification algorithmique |
|---|---|
| `gls_periodogram.png` | Puissance du modèle d’ordre 1 par rapport aux niveaux constants par groupe |
| `fourier_period_search.png` | Score $-\mathrm{BIC}$ sur la grille initiale, à l’ordre finalement retenu ; diagnostic après sélection |
| `period_order_candidates.csv` | Meilleur ajustement de chaque couple candidat–ordre après le premier raffinement, classé par BIC |
| `fourier_order_summary.csv` | Meilleur candidat par ordre, parmi l’ensemble des candidats, sans restriction à la famille stable |
| `period_selection_summary.csv` | Résultat sélectionné, référence à bas ordre, meilleur BIC global et meilleurs ajustements de la famille stable |
| `period_summary.csv` | Période finale, amplitude et deux estimations d’incertitude |
| `file_summary.csv` | Décalages et dispersions résiduelles par groupe |
| `residuals.csv` | Modèle et résidu de chaque observation |
| `residual_filter_summary.csv` | Nombre de points conservés et rejetés, centre, échelle robuste, seuil demandé et seuil effectif |
| Sorties `residual_filtered_*` | Résultats propres à la seconde passe |
| `run_metadata.json` | Paramètres d’exécution et principaux résultats, avec conservation distincte du résultat filtré |

En particulier, **la première ligne de la table classée par BIC peut différer du résultat retenu**, puisque cette table n’applique pas à elle seule les règles de sélection hiérarchique.

De même, le meilleur résultat d’un ordre dans `fourier_order_summary.csv` peut appartenir à une famille éloignée de $P_{\mathrm{ref}}$, tandis que le résultat du même ordre dans la synthèse de sélection est limité à la fenêtre de stabilité.

**23. Le coût du calcul provient surtout des ajustements répétés**

Chaque ajustement reconstruit la matrice de conception, résout les moindres carrés, calcule les statistiques et estime les significativités.

Notons :

- $S$ : nombre de points de la grille initiale ;
- $C$ : nombre de candidats après déduplication ;
- $Q$ : nombre d’ordres ;
- $L$ : nombre de points de raffinement par candidat–ordre ;
- $R$ : nombre de points du raffinement final.

Hors calcul des incertitudes, une passe automatique effectue approximativement :

$$
\boxed{
1+2S+CQL+R
}
$$

ajustements :

- 1 pour le modèle constant du GLS ;
- $S$ pour le GLS ;
- $CQL$ pour les candidats ;
- $R$ pour le raffinement final ;
- $S$ pour le graphique Fourier final.

Avec les valeurs par défaut et le maximum de 60 candidats :

$$
1+2\times8000+60\times5\times300+2000
=
108001.
$$

Cela représente **108 001 ajustements**, avant l’exploration des profils d’incertitude.

Le coût de chacun augmente avec le nombre de points $N$ et avec :

$$
K=G+2H.
$$

Le programme recalcule également la covariance et les significativités lors des nombreux essais intermédiaires, même lorsque le seul score nécessaire à cette étape est le $\chi^2$ ou le BIC.

La seconde passe filtrée peut ajouter un volume de calcul comparable, sur un jeu de données plus petit.

**24. Les limites à conserver dans l’interprétation scientifique**

Plusieurs résultats peuvent être calculés très précisément numériquement sans être déterminés aussi sûrement physiquement.

**La recherche automatique n’est pas exhaustive sur tous les couples période–ordre.** Elle explore les voisinages des candidats issus du GLS et de leurs multiples. Une solution mal représentée par l’ordre 1 peut ne pas être retenue parmi les pics de départ.

**La sélection dépend de l’ordre de référence.** La proximité à une solution d’ordre 2 est une règle de décision du programme, pas une démonstration que cette famille est physiquement correcte.

**Le choix d’ordre s’arrête au premier échec.** Il peut écarter un ordre supérieur contenant une structure significative si un ordre intermédiaire ne satisfait pas les conditions.

**La stabilité de période est une tolérance fixe.** Elle n’est pas validée par bootstrap, validation croisée ou comparaison systématique de sous-campagnes.

**Les ajustements utilisent un nombre nominal de paramètres.** Le rang renvoyé par le solveur n’est pas exploité pour invalider une solution ou ajuster les degrés de liberté. Une mauvaise couverture en phase peut donc produire des coefficients peu contraints malgré une résolution numérique réussie.

**Les incertitudes de période sont locales et conditionnelles à l’ordre choisi.** Elles ne mesurent pas l’ambiguïté entre des alias éloignés ni l’incertitude sur la structure du modèle.

**La correction temporelle implémentée reste une correction héliocentrique directionnelle calculée au géocentre.** Elle ne constitue pas un modèle complet de propagation de lumière pour un astéroïde à distance finie.

**Les magnitudes ne sont pas corrigées explicitement des distances et de l’angle de phase solaire.** Le programme n’ajuste ni loi de phase photométrique, ni tendance temporelle additionnelle. Des offsets par groupe peuvent absorber une partie des différences de niveau, mais ne remplacent pas une modélisation de leur évolution.

**Le filtrage peut supprimer un signal réel absent du premier modèle.** Il ne sait pas reconnaître un événement mutuel, une éclipse ou une occultation.

Enfin, les fonctionnalités suivantes, évoquées comme pistes dans le mémoire existant, **ne sont pas implémentées sur ce commit de `main`** :

- recherche d’une période secondaire dans les résidus ;
- modèle simultané à deux périodes indépendantes ;
- détection automatique d’événements mutuels ;
- recherche BLS ou PDM ;
- estimation d’une période orbitale ;
- détermination d’un caractère binaire ;
- inversion de forme ou d’axe de rotation ;
- conversion explicite vers une période sidérale.

Le résidu produit par cette version reste :

$$
\boxed{
r_i
=
m_i-
\left[
c_0+d_{g_i}+F(t_i;P_*,H_*)
\right].
}
$$

Il fournit une base pour examiner ce que le modèle principal n’explique pas. Le programme actuel n’attribue pas automatiquement cette structure résiduelle à une cause astrophysique particulière.
