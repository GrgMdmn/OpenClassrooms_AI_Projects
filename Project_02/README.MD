# Projet d'Analyse des Arbres Parisiens

## 📋 Contexte du projet

Ce projet a été réalisé dans le cadre d'une mission d'analyse de données pour la **Ville de Paris**. L'objectif est d'explorer et d'analyser le patrimoine arboré parisien à partir d'un jeu de données public recensant les arbres présents sur le territoire de la capitale.

Le projet met l'accent sur les fondamentaux de la data science : le nettoyage des données, la gestion des anomalies, l'analyse exploratoire et la visualisation statistique pour préparer le dataset à des analyses ultérieures ou à des applications métier (gestion urbaine, biodiversité, aménagement).

## 🎯 Objectifs

Les objectifs principaux du projet sont :

1. **Explorer le dataset** : Comprendre la structure, les types de variables et les caractéristiques des arbres parisiens
2. **Nettoyer les données** : Identifier et traiter les valeurs manquantes, aberrantes et incohérentes
3. **Analyser les distributions** : Réaliser des analyses univariées et bivariées pour comprendre les variables
4. **Visualiser les données** : Créer des graphiques et des cartes interactives pour faciliter la compréhension
5. **Préparer les données** : Fournir un dataset propre et exploitable pour des analyses futures

## 📊 Données

Les données proviennent de l'**Open Data de la Ville de Paris** et contiennent des informations détaillées sur les arbres :
- Localisation géographique (coordonnées GPS, arrondissement, adresse)
- Caractéristiques botaniques (genre, espèce, variété)
- Dimensions (hauteur, circonférence)
- Statut (arbre remarquable, domanialité)
- Date de plantation

Le dataset est inclus dans le projet sous le nom `p2-trees-fr.csv`.

**Source** : [Open Data Ville de Paris - Les arbres](https://opendata.paris.fr/explore/dataset/les-arbres/)

## 📁 Structure du projet

```
├── Paris_trees_analysis.ipynb
├── soutenance_projet2.pptx
├── data_analysis_functions.py
├── map_arrondissement.html
├── map_lieu.html
└── p2-trees-fr.csv
```

### 1. `Paris_trees_analysis.ipynb`

**Notebook Principal d'Analyse des Arbres Parisiens**

Ce notebook constitue le cœur du projet et couvre l'ensemble du processus d'analyse exploratoire :

**Chargement et aperçu des données** :
- Import du fichier CSV contenant les données des arbres
- Affichage des premières lignes pour une compréhension rapide
- Inspection de la structure du DataFrame (types, dimensions)

**Analyse exploratoire initiale** :
- Utilisation des méthodes pandas (`info()`, `head()`, `describe()`, `value_counts()`)
- Identification de la taille du dataset
- Analyse des types de colonnes et des valeurs manquantes
- Statistiques descriptives pour les variables numériques

**Identification et nettoyage des anomalies** :
- **Valeurs aberrantes** : 
  - Détection pour la circonférence et la hauteur des arbres
  - Utilisation de méthodes statistiques (IQR, z-score)
  - Correction ou remplacement des valeurs extrêmes par `NaN`
- **Valeurs manquantes** :
  - Analyse approfondie des colonnes avec NaN
  - Imputation intelligente pour certaines variables (`domanialite`, `libelle_francais`)
  - Utilisation de colonnes corrélées (`genre`, `espece`, `variete`) pour l'imputation

**Analyse univariée** :
- **Variables quantitatives** : distribution de la hauteur, circonférence, année de plantation
- **Variables catégorielles** : répartition par genre, espèce, arrondissement, statut remarquable
- Visualisations : histogrammes, boxplots, diagrammes circulaires, barplots

**Analyse bivariée** :
- Relations entre le statut `remarquable` et les variables quantitatives (`hauteur_m`, `circonference_cm`)
- Relations entre `remarquable` et variables catégorielles (`espece`, `genre`)
- Corrélations entre variables numériques
- Tests statistiques pour valider les observations

**Analyse géographique** :
- Distribution des arbres par arrondissement
- Cartographie interactive des arbres parisiens
- Identification des zones à forte densité arborée

**Export des données nettoyées** :
- Sauvegarde du DataFrame préparé pour analyses ultérieures

### 2. `soutenance_projet2.pptx`

**Présentation de Soutenance**

Cette présentation synthétise :
- Le contexte et les objectifs du projet
- La méthodologie d'exploration et de nettoyage des données
- Les principales découvertes sur le patrimoine arboré parisien
- Les analyses statistiques et visualisations clés
- Les insights métier (arbres remarquables, distribution par arrondissement, espèces dominantes)
- Les recommandations pour la gestion du patrimoine arboré
- Les perspectives d'amélioration et d'exploitation des données

### 3. `data_analysis_functions.py`

**Module de Fonctions Utilitaires d'Analyse**

Ce module Python contient des fonctions réutilisables pour :
- Le nettoyage des données (suppression de doublons, gestion des NaN)
- L'identification des valeurs aberrantes (méthodes IQR, z-score, percentiles)
- Le traitement des outliers (remplacement, suppression, cap)
- La transformation des variables (normalisation, encodage)
- La génération de rapports de qualité des données
- Des fonctions de visualisation standardisées

Ces fonctions ont été développées pour être réutilisables sur d'autres projets d'analyse de données.

### 4. `map_arrondissement.html`

**Carte Interactive des Arbres par Arrondissement**

Fichier HTML contenant une carte interactive (probablement Folium ou Plotly) qui visualise :
- La distribution des arbres par arrondissement de Paris
- Des statistiques agrégées par zone géographique
- Des visualisations interactives permettant de zoomer et d'explorer les données
- Potentiellement des indicateurs sur la densité arborée, les espèces dominantes par arrondissement

### 5. `map_lieu.html`

**Carte Interactive des Arbres par Lieu**

Fichier HTML contenant une carte interactive détaillée qui présente :
- La localisation précise des arbres individuels sur le territoire parisien
- Des marqueurs interactifs avec informations détaillées (espèce, hauteur, circonférence)
- Des filtres pour explorer les arbres par caractéristiques
- Des clusters pour améliorer la lisibilité dans les zones à forte densité

### 6. `p2-trees-fr.csv`

**Dataset des Arbres Parisiens**

Fichier CSV contenant les données brutes ou légèrement prétraitées sur les arbres de Paris, incluant :
- Identifiants et références
- Localisation (coordonnées GPS, arrondissement, adresse)
- Caractéristiques botaniques (genre, espèce, variété, nom français)
- Dimensions (hauteur en mètres, circonférence en cm)
- Statut (remarquable ou non, domanialité)
- Date de plantation

## 🛠️ Technologies utilisées

- **Python** : Langage principal pour l'analyse de données
  - pandas, numpy : manipulation et analyse de données
  - matplotlib, seaborn : visualisation de données
  - folium ou plotly : cartographie interactive
  - scipy : tests statistiques
  - scikit-learn : détection d'outliers
- **Jupyter Notebook** : développement et documentation
- **HTML/JavaScript** : cartes interactives

## 📈 Méthodologie

1. **Exploration initiale** : Compréhension de la structure et du contenu du dataset
2. **Nettoyage** : Identification et traitement des valeurs manquantes et aberrantes
3. **Analyses univariées** : Étude de la distribution de chaque variable
4. **Analyses bivariées** : Exploration des relations entre variables
5. **Visualisation géographique** : Cartographie interactive des arbres
6. **Documentation** : Présentation des résultats et insights

## 🎯 Résultats attendus

- ✅ Compréhension approfondie du patrimoine arboré parisien
- ✅ Dataset nettoyé et prêt pour des analyses avancées
- ✅ Identification des arbres remarquables et de leurs caractéristiques
- ✅ Insights sur la répartition géographique des arbres
- ✅ Cartes interactives pour explorer les données visuellement
- ✅ Module Python réutilisable pour des analyses similaires

## 💡 Points clés du projet

### Qualité des données
Comme tout jeu de données réel, le dataset présente des défis :
- **Valeurs manquantes** : certaines caractéristiques botaniques incomplètes
- **Valeurs aberrantes** : erreurs de saisie dans les mesures (hauteurs et circonférences impossibles)
- **Incohérences** : variations dans les formats et les appellations

### Arbres remarquables
Une attention particulière a été portée aux arbres remarquables :
- Caractéristiques différenciant les arbres remarquables des autres
- Relations entre dimensions (hauteur, circonférence) et statut remarquable
- Répartition géographique des arbres remarquables

### Analyse géographique
Le projet intègre une dimension spatiale importante :
- Distribution des arbres par arrondissement
- Identification des zones à forte densité arborée
- Cartes interactives pour une exploration intuitive
- Possibilité d'identifier des patterns géographiques

### Biodiversité urbaine
Les analyses révèlent des informations sur la biodiversité :
- Espèces d'arbres les plus fréquentes à Paris
- Diversité botanique par arrondissement
- Âge moyen des arbres (via les dates de plantation)

## 🎓 Compétences développées

- Chargement et exploration de données tabulaires
- Nettoyage de données réelles avec anomalies
- Gestion des valeurs manquantes avec stratégies d'imputation
- Détection et traitement des valeurs aberrantes
- Analyses statistiques univariées et bivariées
- Visualisation de données (graphiques et cartes)
- Développement de fonctions Python réutilisables
- Communication de résultats d'analyse de données

## 👤 Auteur

**Grégoire Mureau**

---

*Ce projet illustre les fondamentaux de l'analyse exploratoire de données : exploration, nettoyage, visualisation et communication d'insights, appliqués à un cas concret d'open data urbain.*
