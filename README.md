# Prédiction du risque de crédit
Construire une chaîne de scoring bancaire complète, puis comparer une grille de score interprétable à plusieurs modèles de Machine Learning.

## Objectif

Le projet cherche à fiabiliser la décision d’octroi de crédit tout en limitant :

- les défauts de paiement ;
- les refus de clients solvables ;
- les décisions opaques ;
- l’écart entre performance statistique et rentabilité réelle.

## Données

- 307 511 clients ;
- 22 variables ;
- données personnelles, financières et professionnelles ;
- cible `GOOD_PAYER` :
  - 1 : aucun retard ;
  - 0 : retard de remboursement.

La cible est déséquilibrée : environ 91 % de bons payeurs contre 9 % de mauvais payeurs.

## Pipeline

1. analyse des valeurs manquantes ;
2. imputations conditionnelles, catégorielles et KNN ;
3. traitement des valeurs extrêmes ;
4. analyse statistique, WOE, IV, Gini et tests de significativité ;
5. découpage stratifié train / validation / test ;
6. encodage, scaling et binning ;
7. construction d’une grille de score logistique ;
8. comparaison avec plusieurs modèles de Machine Learning ;
9. interprétation par SHAP et permutation importance ;
10. évaluation économique des seuils de décision.

## Modèles comparés

| Modèle | AUC | Gini |
|---|---:|---:|
| XGBoost | **0,7088** | **0,4175** |
| Gradient Boosting | 0,7083 | 0,4167 |
| Random Forest | 0,7066 | 0,4132 |

Les variables les plus influentes sont `EXT_SOURCE_2`, `EXT_SOURCE_3`, `YEARS_EMPLOYED` et `AGE_YEARS`.

Après réentraînement sur ces quatre variables, la performance reste presque inchangée. Le modèle devient donc plus simple sans perte significative de discrimination.

## Enseignement métier

Avec le seuil et les coûts retenus, les modèles produisent une perte économique proche d’environ **-3 684,72**.

Une meilleure AUC ne garantit donc pas automatiquement une meilleure décision métier. Le seuil, les coûts d’erreur et la politique commerciale doivent être calibrés ensemble.

## Technologies

`Python` · `pandas` · `NumPy` · `SciPy` · `statsmodels` · `scikit-learn` · `XGBoost` · `Optuna` · `SHAP`

## Ce que ce projet démontre

- data cleaning à grande échelle ;
- scoring bancaire ;
- sélection et simplification de variables ;
- optimisation de modèles ;
- explicabilité ;
- évaluation statistique et économique.

## Limites et améliorations

Une version industrialisée devrait intégrer le suivi de dérive, le recalibrage périodique, une API de scoring, un tableau de bord de contrôle et des tests de biais.

## Auteurs

Projet réalisé par **Benjamin BAILLET**, **Alexandra MILLOT** et **Ismail OUAZZANI CHAHDI**.

---

