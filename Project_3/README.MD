# Projet de Préparation de Données - OpenFoodFacts

## 📋 Contexte du projet

Ce projet a été réalisé pour un organisme de santé publique souhaitant améliorer la **base de données collaborative OpenFoodFacts**. L'objectif est de préparer et d'analyser les données alimentaires pour faciliter la création d'un système d'aide à la saisie de produits par les utilisateurs, tout en respectant les principes du RGPD.

OpenFoodFacts est une base de données ouverte sur les produits alimentaires du monde entier, contenant des informations nutritionnelles, des ingrédients, des allergènes et des labels. Cependant, ces données nécessitent un nettoyage approfondi et une exploration statistique pour être exploitables.

## 🎯 Objectifs

Les objectifs principaux du projet sont :

1. **Nettoyer et filtrer les données** : Éliminer les doublons, les incohérences et sélectionner les variables pertinentes
2. **Traiter les valeurs manquantes et aberrantes** : Identifier et gérer les données manquantes, estimer les valeurs nutritionnelles
3. **Réaliser des analyses statistiques** : Analyses univariées, bivariées et multivariées pour comprendre les relations entre variables
4. **Créer de nouvelles caractéristiques** : Feature engineering pour enrichir le dataset
5. **Produire des visualisations** : Graphiques et analyses pour faciliter la compréhension des données
6. **Respecter le RGPD** : Garantir la conformité avec les réglementations sur la protection des données

## 📊 Données

Les données proviennent de la base **OpenFoodFacts**, une base collaborative mondiale sur les produits alimentaires contenant :
- Des informations nutritionnelles (énergie, lipides, glucides, protéines, etc.)
- Des listes d'ingrédients et d'allergènes
- Des scores nutritionnels (Nutri-Score)
- Des labels et certifications
- Des informations géographiques et de packaging

**⚠️ Note importante** : Le dataset original est volumineux et n'est pas inclus dans ce dépôt. Il est téléchargeable à l'adresse suivante :
- **URL** : https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/parcours-data-scientist/P2/fr.openfoodfacts.org.products.csv.zip

**Complément** : La librairie [NutriScoreCalculator](https://github.com/MalteIwanicki/NutriScoreCalculator) a été utilisée pour compléter certaines valeurs nutritionnelles et calculer le Nutri-Score manquant.

## 📁 Structure du projet

```
├── openfoodfacts_analysis_clean.ipynb
├── OpenFoodFacts_presentation.pdf
├── data_analysis_functions.py
├── pandas_df_management.py
├── strings_management.py
├── translated_allergens.json
└── translated_ingredients.json
```

### 1. `openfoodfacts_analysis_clean.ipynb`

**Notebook Principal d'Analyse et de Préparation des Données**

Ce notebook constitue le cœur du projet et couvre l'ensemble du processus de data cleaning et d'analyse :

**Chargement et exploration initiale** :
- Importation du dataset OpenFoodFacts
- Analyse de la taille, des types de données et des statistiques descriptives
- Identification des colonnes pertinentes pour l'analyse

**Nettoyage et filtrage** :
- Sélection des caractéristiques pertinentes pour l'objectif métier
- Suppression des doublons (produits identiques)
- Filtrage des lignes sans valeurs cibles (notamment les allergènes)
- Harmonisation des formats et des unités

**Gestion des valeurs manquantes et aberrantes** :
- Identification des valeurs manquantes par variable
- Stratégies d'imputation adaptées à chaque type de variable
- Détection et traitement des valeurs nutritionnelles aberrantes
- Estimation des valeurs manquantes d'`energy_100g` via des corrélations
- Calcul du Nutri-Score manquant avec NutriScoreCalculator

**Création de nouvelles caractéristiques (Feature Engineering)** :
- `allergens_count` : nombre d'allergènes par produit
- Caractéristiques dérivées des ingrédients (nombre, présence de certains ingrédients)
- Variables binaires pour les additifs
- Ratios nutritionnels (lipides/énergie, sucres/glucides, etc.)

**Analyse exploratoire approfondie** :
- **Analyses univariées** : distributions des variables nutritionnelles, Nutri-Score, catégories de produits
- **Analyses bivariées** : corrélations entre variables numériques, relations entre Nutri-Score et compositions
- **Analyses multivariées** : ACP (Analyse en Composantes Principales) pour visualiser les relations globales
- **Tests statistiques** : corrélations de Pearson, ANOVA, tests du Khi², V de Cramer
- **Visualisations** : histogrammes, boxplots, heatmaps de corrélation, scatter plots

**Export des données raffinées** :
- Sauvegarde du DataFrame nettoyé et enrichi (`df_refined_allergens_based.csv`)
- Documentation des transformations effectuées

### 2. `OpenFoodFacts_presentation.pdf`

**Présentation de Soutenance**

Cette présentation synthétise :
- Le contexte et les enjeux du projet
- La méthodologie de nettoyage et de préparation des données
- Les principales analyses statistiques réalisées
- Les insights clés découverts dans les données
- Les recommandations pour l'amélioration du système de saisie
- Les perspectives d'amélioration et d'exploitation des données

### 3. `data_analysis_functions.py`

**Module de Fonctions d'Analyse de Données**

Ce module Python contient des fonctions réutilisables pour :
- Le nettoyage des données (suppression de doublons, filtrage)
- L'identification des valeurs aberrantes (méthodes statistiques, IQR)
- Le traitement des outliers
- La détection d'incohérences dans les données
- La génération de rapports de qualité des données

### 4. `pandas_df_management.py`

**Module de Gestion des DataFrames Pandas**

Ce module automatise les tâches courantes de manipulation de DataFrames :
- Fonctions d'inspection rapide des DataFrames
- Automatisation de la gestion des types de données
- Fonctions de transformation et de reshaping
- Utilitaires pour l'export et l'import de données
- Fonctions de validation de la structure des données

### 5. `strings_management.py`

**Module de Nettoyage et Normalisation de Chaînes de Caractères**

Ce module contient des fonctions spécialisées pour :
- Le nettoyage des chaînes de caractères (suppression d'espaces, caractères spéciaux)
- La normalisation des textes (majuscules/minuscules, accents)
- Le parsing des listes d'ingrédients et d'allergènes
- La standardisation des noms de produits et de catégories
- La gestion des formats multilingues

### 6. `translated_allergens.json`

**Fichier de Correspondance des Allergènes**

Fichier JSON contenant les traductions et correspondances des allergènes :
- Harmonisation des noms d'allergènes en différentes langues
- Standardisation des appellations (gluten, lactose, fruits à coque, etc.)
- Regroupement des variantes orthographiques
- Facilite l'analyse multilingue de la base OpenFoodFacts

### 7. `translated_ingredients.json`

**Fichier de Correspondance des Ingrédients**

Fichier JSON contenant les traductions et correspondances des ingrédients :
- Harmonisation des noms d'ingrédients communs
- Standardisation des appellations en différentes langues
- Regroupement des synonymes et variantes
- Support de l'analyse multilingue des compositions

## 🛠️ Technologies utilisées

- **Python** : Langage principal pour l'analyse de données
  - pandas, numpy : manipulation et analyse de données
  - matplotlib, seaborn, plotly : visualisation de données
  - scipy, statsmodels : tests statistiques et analyses
  - scikit-learn : ACP et analyses multivariées
  - NutriScoreCalculator : calcul du Nutri-Score
- **Jupyter Notebook** : développement et documentation
- **JSON** : stockage des fichiers de correspondance

## 📈 Méthodologie

1. **Exploration initiale** : Compréhension de la structure et du contenu du dataset
2. **Nettoyage** : Élimination des doublons, incohérences et données non pertinentes
3. **Traitement des valeurs manquantes** : Stratégies d'imputation adaptées à chaque variable
4. **Feature Engineering** : Création de variables dérivées pertinentes
5. **Analyses statistiques** : Tests et visualisations pour comprendre les relations
6. **Validation** : Vérification de la qualité et de la cohérence des données nettoyées
7. **Documentation** : Présentation des résultats et recommandations

## 🎯 Résultats attendus

- ✅ Dataset nettoyé, enrichi et prêt pour des analyses avancées
- ✅ Compréhension approfondie des données OpenFoodFacts
- ✅ Identification des variables clés pour la prédiction du Nutri-Score
- ✅ Insights sur les relations entre composition nutritionnelle et qualité alimentaire
- ✅ Modules Python réutilisables pour le traitement de données similaires
- ✅ Documentation complète du processus de préparation

## 💡 Points clés du projet

### Qualité des données
OpenFoodFacts étant une base collaborative, elle présente de nombreux défis :
- **Valeurs manquantes importantes** : certaines colonnes ont plus de 50% de données manquantes
- **Incohérences** : variations dans les formats, les unités, les langues
- **Doublons** : produits saisis plusieurs fois avec des variations
- **Valeurs aberrantes** : erreurs de saisie dans les valeurs nutritionnelles

### Respect du RGPD
Le projet garantit la conformité avec les réglementations :
- Anonymisation des données personnelles potentielles
- Traitement conforme des données européennes
- Documentation des traitements effectués

### Analyses statistiques approfondies
Le projet va au-delà du simple nettoyage en proposant :
- Des analyses univariées pour comprendre les distributions
- Des analyses bivariées pour identifier les relations entre variables
- Des analyses multivariées (ACP) pour une vision globale
- Des tests statistiques pour valider les observations

### Feature Engineering orienté métier
Création de variables pertinentes pour la santé publique :
- Comptage d'allergènes (aide à l'identification des produits à risque)
- Ratios nutritionnels (sucres/glucides, lipides/énergie)
- Indicateurs de qualité nutritionnelle

## 🎓 Compétences développées

- Nettoyage et préparation de données volumineuses et imparfaites
- Gestion des valeurs manquantes et aberrantes avec stratégies adaptées
- Analyses statistiques univariées, bivariées et multivariées
- Feature engineering orienté métier santé
- Visualisation de données pour la communication d'insights
- Développement de modules Python réutilisables
- Respect des contraintes réglementaires (RGPD)
- Documentation et présentation de résultats techniques

## 👤 Auteur

**Grégoire Mureau**

---

*Ce projet illustre la démarche complète de préparation de données en data science : de l'exploration initiale au dataset raffiné, en passant par le nettoyage, l'enrichissement et l'analyse statistique.*
