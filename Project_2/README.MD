# Projet 02 : Analyse et Exploration des Données sur les Arbres Parisiens

## Objectif du projet
Explorer et analyser un jeu de données sur les arbres de Paris afin de comprendre leurs caractéristiques et identifier des relations pertinentes entre variables.  
Ce projet met l’accent sur le **nettoyage des données**, la **gestion des anomalies** et la **visualisation statistique** pour préparer le dataset à des analyses ultérieures.

## Contenu du projet

- `Mureau_Grégoire_functions_012025.py` : fonctions utilitaires pour le nettoyage, la transformation et l’analyse des données.
- `paris_trees_analysis.py` : script principal pour exécuter les traitements sur le dataset.
- `paris_trees_analysis_notebook.ipynb` : notebook détaillant toutes les étapes de l’exploration et du nettoyage.
- `soutenance_projet2_presentation.pdf` : support de présentation pour la soutenance.

## Étapes principales du notebook

1. **Chargement et Aperçu des Données**
   - Import des données à partir du fichier CSV.
   - Affichage des premières lignes pour un aperçu rapide.
   - Inspection de la structure et des types de données du DataFrame.

2. **Analyse Exploratoire Initiale**
   - Utilisation des méthodes pandas (`info()`, `head()`, `describe()`) pour obtenir :
     - Taille du dataset
     - Types de colonnes
     - Valeurs manquantes
     - Statistiques descriptives pour les variables numériques.

3. **Identification et Nettoyage des Anomalies**
   - **Valeurs aberrantes** : identification pour la circonférence et la hauteur des arbres ; certaines valeurs extrêmes remplacées par `NaN` ou corrigées.
   - **Valeurs manquantes** : analyse approfondie des colonnes avec NaN ; imputation partielle pour certaines variables (`domanialite`, `libelle_francais`) en utilisant des colonnes corrélées (`genre`, `espece`, `variete`).

4. **Analyse Univariée et Bivariée (après nettoyage)**
   - **Analyse univariée** : distribution des variables individuelles (quantitatives et catégorielles) visualisée via histogrammes, boxplots et diagrammes circulaires.
   - **Analyse bivariée** : relations entre paires de variables, notamment :
     - `remarquable` vs variables quantitatives (`hauteur_m`, `circonference_cm`)
     - `remarquable` vs variables catégorielles (`espece`).

## Résultat attendu
- Compréhension globale du dataset sur les arbres parisiens.
- Dataset nettoyé et prêt pour des analyses plus avancées ou des modélisations futures.
- Visualisations et analyses statistiques documentant les distributions et les relations clés.

## Notes
Ce projet met en lumière les étapes essentielles du **nettoyage de données** et de l’**exploration statistique**, qui sont fondamentales pour tout projet de data analysis.
