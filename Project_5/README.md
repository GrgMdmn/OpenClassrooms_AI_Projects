# Projet de Segmentation Client - Olist

## 📋 Contexte du projet

Ce projet a été réalisé dans le cadre d'une mission de consulting pour **Olist**, une entreprise brésilienne proposant une solution de vente sur les marketplaces en ligne. L'objectif principal est d'accompagner Olist dans la mise en place de leur équipe Data et de réaliser leur premier cas d'usage Data Science autour de la segmentation client.

## 🎯 Objectifs

Le projet se divise en deux missions principales :

1. **Mission SQL** : Développer des requêtes SQL pour alimenter le dashboard Customer Experience de l'équipe
2. **Mission Segmentation** : Créer une segmentation client exploitable pour l'équipe Marketing, basée sur des algorithmes de clustering non-supervisé

## 📊 Données

Les données utilisées proviennent d'une base de données anonymisée fournie par Olist, contenant :
- L'historique des commandes depuis janvier 2017
- Les informations sur les produits achetés
- Les commentaires de satisfaction des clients
- La localisation des clients
- Les données sur les vendeurs

**⚠️ Note importante** : La base de données n'est pas incluse dans ce dépôt en raison de sa taille. Elle est téléchargeable à l'adresse suivante :
- **URL** : https://course.oc-static.com/projects/olist.db

## 📁 Structure du projet

```
├── P5_01_script_052025.sql
├── P5_02_notebook_exploration_052025.ipynb
├── P5_03_notebook_essais_052025.ipynb
├── P5_04_notebook_simulation_052025.ipynb
└── P5_05_presentation_052025.pptx
```

### 1. `P5_01_script_052025.sql`

**Script SQL pour le Dashboard Customer Experience**

Ce fichier contient 4 requêtes SQL essentielles pour alimenter le dashboard du service client :

1. **Commandes en retard** : Identification des commandes récentes (moins de 3 mois) reçues avec au moins 3 jours de retard (hors commandes annulées)

2. **Top vendeurs par chiffre d'affaires** : Liste des vendeurs ayant généré un chiffre d'affaires supérieur à 100 000 Real sur les commandes livrées

3. **Nouveaux vendeurs performants** : Identification des vendeurs récents (moins de 3 mois d'ancienneté) ayant déjà vendu plus de 30 produits

4. **Zones géographiques problématiques** : Les 5 codes postaux avec plus de 30 avis et les pires scores moyens de satisfaction sur les 12 derniers mois

### 2. `P5_02_notebook_exploration_052025.ipynb`

**Notebook d'exploration et de feature engineering**

Ce notebook contient :
- L'analyse exploratoire des données (EDA)
- La création et la transformation des features clients
- L'implémentation de la méthode RFM (Recency, Frequency, Monetary)
- Le preprocessing des données (normalisation, encodage)
- Les premières visualisations et insights sur les comportements clients

### 3. `P5_03_notebook_essais_052025.ipynb`

**Notebook de modélisation et clustering**

Ce notebook présente :
- Les différentes approches de clustering testées (K-means, DBSCAN, etc.)
- L'optimisation du nombre de clusters (méthode du coude, silhouette score)
- L'évaluation des performances des modèles
- La caractérisation détaillée de chaque segment client
- La validation métier des segments identifiés

### 4. `P5_04_notebook_simulation_052025.ipynb`

**Notebook de simulation pour le contrat de maintenance**

Ce notebook analyse :
- La stabilité des clusters dans le temps
- L'évolution de l'Adjusted Rand Index (ARI) sur différentes périodes
- La distribution temporelle des features
- Les recommandations sur la fréquence de mise à jour du modèle de segmentation
- Les tests de Kolmogorov-Smirnov pour détecter les drifts

### 5. `P5_05_presentation_052025.pptx`

**Présentation des résultats**

Cette présentation synthétise :
- La démarche méthodologique adoptée
- Les segments clients identifiés et leurs caractéristiques
- Les insights actionnables pour l'équipe Marketing
- Les recommandations sur le contrat de maintenance
- Les perspectives d'amélioration

## 🛠️ Technologies utilisées

- **SQL** : Requêtage de base de données
- **Python** : Analyse de données et machine learning
  - pandas, numpy : manipulation de données
  - scikit-learn : algorithmes de clustering
  - matplotlib, seaborn : visualisation
  - yellowbrick : évaluation des clusters
- **Jupyter Notebook** : développement et documentation

## 📈 Méthodologie

1. **Analyse exploratoire** : Compréhension approfondie des données et des comportements clients
2. **Feature engineering** : Création de variables pertinentes (RFM et autres métriques comportementales)
3. **Modélisation** : Test et sélection d'algorithmes de clustering
4. **Évaluation** : Validation technique (silhouette, ARI) et métier des segments
5. **Simulation** : Analyse de la stabilité temporelle pour définir la fréquence de maintenance

## 📝 Livrables

- ✅ Script SQL avec 4 requêtes pour le dashboard
- ✅ Notebook d'exploration et feature engineering
- ✅ Notebook de modélisation avec différents essais de clustering
- ✅ Notebook de simulation pour le contrat de maintenance
- ✅ Présentation de la segmentation et recommandations

## 🎓 Compétences développées

- Requêtage SQL avancé avec agrégations et jointures
- Apprentissage non-supervisé (clustering)
- Feature engineering orienté métier
- Évaluation de la qualité et de la stabilité des modèles
- Communication des résultats à des équipes métier

## 👤 Auteur

**Grégoire Mureau**  
Date de réalisation : Mai 2025

---

*Ce projet a été réalisé dans le cadre du parcours AI Engineer d'OpenClassrooms*
