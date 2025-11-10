# Projet de Scoring Crédit - Prêt à dépenser

## 📋 Contexte du projet

Ce projet a été réalisé pour **"Prêt à dépenser"**, une société financière spécialisée dans l'octroi de crédits à la consommation. L'entreprise souhaite mettre en place un **outil de scoring crédit** permettant d'évaluer la probabilité qu'un client rembourse son prêt et ainsi automatiser le processus de décision d'octroi de crédit.

## 🎯 Objectifs

Les objectifs principaux du projet sont :

1. **Développer un modèle de scoring** : Créer un modèle prédictif robuste capable d'évaluer le risque de défaut de paiement
2. **Optimiser le coût métier** : Minimiser les pertes financières en tenant compte du coût réel des faux positifs et faux négatifs
3. **Assurer l'interprétabilité** : Fournir des explications claires sur les décisions du modèle pour les chargés de relation client
4. **Gérer le déséquilibre des classes** : Traiter efficacement les données déséquilibrées caractéristiques du domaine du crédit

## 📊 Données

Les données proviennent du concours Kaggle **"Home Credit Default Risk"** et comprennent :
- L'historique de crédit des clients
- Les demandes de prêt précédentes
- Les informations sur les cartes de crédit
- Les données de paiement par acomptes
- Les soldes POS (Point of Sale) et cash

**⚠️ Note importante** : Les fichiers de données bruts ne sont pas inclus dans ce dépôt en raison de leur taille importante. Ils sont téléchargeables à l'adresse suivante :
- **URL** : https://www.kaggle.com/competitions/home-credit-default-risk/data

Les fichiers nécessaires incluent :
- `application_train.csv` / `application_test.csv`
- `bureau.csv`
- `previous_application.csv`
- `credit_card_balance.csv`
- `installments_payments.csv`
- `POS_CASH_balance.csv`

## 📁 Structure du projet

```
├── P4_01_notebook_analyse_exploratoire_feature_engineering.ipynb
├── P4_02_notebook_modelisation.ipynb
├── P4_03_presentation.pptx
├── explication_client_0.html
└── explication_client_2.html
```

### 1. `P4_01_notebook_analyse_exploratoire_feature_engineering.ipynb`

**Notebook d'Analyse Exploratoire des Données (EDA) et Feature Engineering**

Ce notebook contient l'ensemble de la phase de préparation des données :

**Exploration des données** :
- Chargement et analyse des différents fichiers CSV
- Étude de la distribution de la variable cible (déséquilibre des classes)
- Analyse des valeurs manquantes et des valeurs atypiques
- Étude des corrélations entre variables et avec la cible

**Enrichissement des données** :
- Fusion intelligente des différentes sources de données (bureau, previous_application, etc.)
- Agrégation d'informations provenant de multiples tables
- Création de variables métier pertinentes (ratios, moyennes, statistiques)

**Feature Engineering** :
- Création de nouvelles caractéristiques basées sur des ratios métiers
- Encodage des variables catégorielles
- Imputation des valeurs manquantes
- Normalisation et standardisation des variables
- Préparation spécifique pour les modèles tolérant les valeurs manquantes (LightGBM)

### 2. `P4_02_notebook_modelisation.ipynb`

**Notebook de Modélisation, Optimisation et Interprétation**

Ce notebook présente l'intégralité du processus de modélisation :

**Modélisation de base** :
- Régression logistique (baseline)
- Random Forest
- Évaluation avec métriques standards (AUC-ROC, précision, recall)

**Modélisation avancée** :
- Entraînement de modèles gradient boosting : XGBoost, LightGBM, CatBoost
- Gestion du déséquilibre des classes (class_weight, scale_pos_weight)
- Optimisation des hyperparamètres avec GridSearchCV

**Optimisation métier** :
- Définition d'un **score métier** intégrant le coût des faux positifs et faux négatifs
- Détermination du **seuil de décision optimal** pour minimiser les pertes financières
- Analyse de la courbe coût-seuil

**Interprétabilité** :
- **Interprétation globale** : Importance des variables avec SHAP (TreeExplainer)
- **Interprétation locale** : Explication des prédictions individuelles avec LIME
- Analyse de l'impact de chaque variable sur les décisions

**Simplification du modèle** :
- Sélection des variables les plus importantes
- Réduction du nombre de features tout en maintenant les performances (AUC)

### 3. `P4_03_presentation.pptx`

**Présentation de Soutenance**

Cette présentation synthétise :
- Le contexte métier et les enjeux du scoring crédit
- La méthodologie de préparation et d'enrichissement des données
- La comparaison des différents modèles testés
- Le choix du modèle final et son optimisation métier
- L'approche d'interprétabilité globale et locale (SHAP et LIME)
- Les recommandations et perspectives d'amélioration

### 4. `explication_client_0.html` et `explication_client_2.html`

**Rapports d'Interprétation Locale des Prédictions**

Ces rapports HTML interactifs illustrent des exemples concrets d'explications pour des clients individuels :
- Les prédictions du modèle pour chaque client (probabilité de défaut)
- Les explications SHAP et/ou LIME détaillant pourquoi le modèle a pris telle décision
- La contribution de chaque variable à la prédiction finale pour ce client spécifique
- Des visualisations permettant aux chargés de clientèle de comprendre et justifier les décisions
- Des exemples de cas d'acceptation et de refus avec leurs justifications

Ces fichiers servent de démonstration du système d'interprétabilité locale qui pourrait être déployé en production pour expliquer chaque décision de crédit.

## 🛠️ Technologies utilisées

- **Python** : Développement et analyse
  - pandas, numpy : manipulation de données
  - scikit-learn : modèles de base, preprocessing, métriques
  - XGBoost, LightGBM, CatBoost : modèles de gradient boosting
  - SHAP : interprétation globale des modèles
  - LIME : interprétation locale des prédictions
  - matplotlib, seaborn, plotly : visualisation
- **Jupyter Notebook** : développement et documentation

## 📈 Méthodologie

1. **Exploration et compréhension** : Analyse approfondie des données multi-sources
2. **Feature Engineering** : Création de variables métier pertinentes par agrégation et transformation
3. **Modélisation** : Test de plusieurs algorithmes avec gestion du déséquilibre
4. **Optimisation métier** : Définition et optimisation d'un score métier intégrant les coûts réels
5. **Interprétabilité** : Mise en place d'outils d'explication globale (SHAP) et locale (LIME)
6. **Validation** : Évaluation sur données test et vérification de la généralisation

## 🎯 Résultats attendus

- ✅ Dataset enrichi et prétraité avec nouvelles features métier
- ✅ Modèle de scoring optimisé selon un critère métier (minimisation des coûts)
- ✅ Seuil de décision optimal déterminé
- ✅ Rapports d'interprétabilité globale et locale
- ✅ Documentation complète pour les utilisateurs finaux

## 💡 Points clés du projet

### Gestion du déséquilibre des classes
Le dataset présente un fort déséquilibre entre les bons et mauvais payeurs. Plusieurs techniques ont été mises en œuvre :
- Utilisation de paramètres `class_weight` et `scale_pos_weight`
- Optimisation basée sur des métriques appropriées (AUC-ROC, recall, score métier)
- Ajustement du seuil de décision post-modélisation

### Score métier personnalisé
Plutôt que de se fier uniquement aux métriques classiques, un **score métier** a été défini prenant en compte :
- Le coût d'un faux positif (refus à tort d'un bon client → perte d'opportunité)
- Le coût d'un faux négatif (acceptation d'un mauvais payeur → perte financière importante)
- L'optimisation du seuil de décision pour minimiser le coût total

### Interprétabilité
Deux niveaux d'interprétation ont été implémentés :
- **Globale (SHAP)** : Quelles variables sont les plus importantes pour le modèle en général ?
- **Locale (LIME)** : Pourquoi le modèle a-t-il pris cette décision pour ce client spécifique ?

Cette double approche permet aux chargés de relation client de comprendre et d'expliquer les décisions aux demandeurs de crédit.

## 🎓 Compétences développées

- Préparation et enrichissement de données multi-sources complexes
- Feature engineering orienté métier dans le domaine financier
- Modélisation avec gestion du déséquilibre des classes
- Optimisation basée sur des critères métier (coût/bénéfice)
- Interprétabilité des modèles de machine learning (SHAP, LIME)
- Communication des résultats techniques à des audiences métier

## 👤 Auteur

**Grégoire Mureau**

---

*Ce projet illustre la démarche complète de data science appliquée au scoring crédit : de la préparation des données à l'optimisation métier, en passant par l'interprétabilité des modèles.*
