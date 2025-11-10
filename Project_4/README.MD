# Projet 04 : Construction d’un Modèle de Scoring Crédit

## Contexte du projet
Dans le cadre de cette mission, nous avons développé un **modèle de scoring crédit** pour la société "Prêt à dépenser", qui souhaite évaluer la probabilité qu’un client rembourse son prêt.  
L’objectif est de fournir un outil **interprétable**, permettant aux chargés de relation client de comprendre les facteurs influençant chaque décision et de classer les demandes de crédit (accordé ou refusé).

Les aspects clés de la mission sont :
- Gestion des classes déséquilibrées et prise en compte du coût métier des erreurs (faux positifs et faux négatifs).
- Création d’un score métier optimisé pour minimiser les pertes financières.
- Interprétabilité globale et locale des modèles via SHAP et LIME.

## Contenu du projet

- **Notebook 1 : Analyse Exploratoire des Données (EDA)**
  - Chargement et exploration des fichiers CSV (train et test, bureau, previous_application, credit_card_balance, installments_payments, POS_CASH_balance).
  - Analyse des valeurs manquantes et atypiques.
  - Étude des corrélations entre les variables et la cible.
  - Fusion et enrichissement des données par agrégation d’informations provenant de sources multiples.
  - Création de nouvelles caractéristiques (feature engineering) basées sur des ratios métiers.
  - Pré-traitement des données : encodage, imputation, normalisation.
  - Préparation des données pour des modèles tolérant les valeurs manquantes (ex : LightGBM).

- **Notebook 2 : Modélisation et Interprétation**
  - Évaluation de modèles de base : régression logistique et Random Forest.
  - Entraînement de modèles avancés : XGBoost, LightGBM et CatBoost avec gestion du déséquilibre des classes.
  - Optimisation des hyperparamètres avec GridSearchCV.
  - Détermination du **seuil optimal** pour minimiser le coût métier combinant faux positifs et faux négatifs.
  - Interprétation globale : analyse de l’importance des variables avec **SHAP**.
  - Interprétation locale : explication des prédictions pour des clients individuels avec **LIME**.
  - Option de simplification du modèle en réduisant le nombre de variables tout en conservant l’AUC.

- **Rapports et Présentation**
  - `explanation_global.html` et `explanation_local.html` : rapports générés pour l’interprétation globale et locale.
  - `soutenance_projet4_presentation.pdf` : support de présentation pour la soutenance.

## Fichiers volumineux
Les fichiers bruts issus du concours Kaggle ne sont pas inclus pour des raisons de taille. Ils peuvent être téléchargés ici :  
[Home Credit Default Risk – Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk/data)

## Résultat attendu
- Dataset prétraité, enrichi et prêt pour la modélisation.
- Modèles robustes de scoring crédit, optimisés selon un **score métier**.
- Visualisations et rapports expliquant :
  - L’importance globale des variables.
  - Les raisons des décisions pour chaque client individuel.
- Présentation synthétique pour les utilisateurs finaux mettant en avant :
  - Méthodologie de nettoyage et d’enrichissement.
  - Comparaison des modèles.
  - Interprétabilité globale et locale.
  - Recommandations métier basées sur les résultats.

## Notes
Ce projet illustre la démarche complète de **data science appliquée au scoring crédit** : préparation des données, création de nouvelles variables, modélisation supervisée, optimisation et interprétation métier.
