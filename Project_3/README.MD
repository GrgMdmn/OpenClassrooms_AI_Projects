# Projet 03 : Préparez des données pour un organisme de santé publique

## Objectif du projet
Préparer et explorer les données de la base OpenFoodFacts afin de faciliter la création d’un système d’aide à la saisie de produits par les utilisateurs.  
Les objectifs spécifiques sont :
- Nettoyer et filtrer les données ;
- Traiter les valeurs manquantes et aberrantes ;
- Réaliser des analyses statistiques univariées, bi-variées et multivariées ;
- Produire des visualisations pour comprendre la distribution et les relations entre les variables ;
- Respecter les principes RGPD.

## Contenu du projet

- `data_analysis_functions.py` : fonctions pour le nettoyage des données, identification et traitement des valeurs aberrantes et des doublons.
- `pandas_df_management.py` : fonctions d’automatisation pour la manipulation et la gestion des DataFrames pandas.
- `strings_management.py` : fonctions pour le nettoyage et la normalisation des chaînes de caractères dans le dataset.
- `translated_allergens.json` et `translated_ingredients.json` : fichiers de correspondance pour harmoniser les allergènes et ingrédients.
- `openfoodfacts_analysis_clean.ipynb` : notebook principal qui couvre les étapes suivantes :
  1. **Chargement et exploration initiale** : taille du dataset, types de données, statistiques descriptives.
  2. **Nettoyage et filtrage** : sélection des caractéristiques pertinentes, suppression des doublons et lignes sans valeurs cibles (allergens).
  3. **Gestion des valeurs manquantes et aberrantes** : traitement des valeurs nutritionnelles, estimation des valeurs manquantes d’`energy_100g` et du NutriScore via NutriScoreCalculator.
  4. **Création de nouvelles caractéristiques** : nombre d’allergènes (`allergens_count`), caractéristiques dérivées des ingrédients et des additifs.
  5. **Analyse exploratoire** : visualisations univariées, bivariées et multivariées ; tests statistiques (corrélation, ANOVA, Khi², V de Cramer) ; ACP et heatmaps pour résumer les relations entre variables.
  6. **Export des données raffinées** : sauvegarde du DataFrame nettoyé et enrichi (`df_refined_allergens_based.csv`) pour usage ultérieur.
- `OpenFoodFacts_presentation.pdf` : présentation synthétisant la démarche et les résultats pour la soutenance.

## Données utilisées
- Dataset source : [OpenFoodFacts CSV](https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/parcours-data-scientist/P2/fr.openfoodfacts.org.products.csv.zip)
- Compléments via la librairie [NutriScoreCalculator](https://github.com/MalteIwanicki/NutriScoreCalculator) pour compléter certaines valeurs nutritionnelles manquantes.

## Notes
Le projet met l’accent sur la **qualité des données** et la préparation d’un dataset exploitable pour de futures analyses ou modélisations, tout en respectant les principes du RGPD.
