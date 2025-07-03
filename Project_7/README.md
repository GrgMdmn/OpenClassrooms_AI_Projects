# Projet Air Paradis - Analyse de Sentiments

Ce dépôt contient les notebooks et le code source pour le projet d’analyse de sentiments sur des tweets, dans le cadre du projet OpenClassrooms "Air Paradis".

## Structure du dépôt

- `notebooks/` : notebooks d’analyse exploratoire et modélisation (modèles simples, avancés, BERT)  
- `deployment/` : code de l’API, tests, Dockerfile et fichiers liés au déploiement (qui sera dans un premier temps effectué sur un NAS)  
- `docs/` : documentation et article de blog (à compléter)  
- `.github/` : workflows GitHub Actions pour CI/CD (à configurer)  

## Usage

Les notebooks contiennent les étapes d’analyse et d’entraînement des modèles. L’API de prédiction sera développée dans `deployment/api`.

## Contexte et contraintes personnelles

Un défi personnel important de ce projet a été de privilégier une installation locale afin d’assurer une souveraineté complète sur les données et la chaîne de traitement. Le serveur NAS personnel héberge MLflow, MinIO et l’API FastAPI en local.

Cependant, faute de disposer d’une carte graphique Nvidia à domicile, la phase d’entraînement des modèles Deep Learning, notamment BERT, a nécessité l’usage de Google Colab.

## Remarques complémentaires

- Les fichiers `requirements.txt` présents dans `notebooks/` proviennent d’environnements Google Colab et contiennent des dépendances non optimales, nécessitant un nettoyage pour un usage local plus léger. Ce travail sera éventuellement effectué ultérieurement mais ne constitue pas en soi une priorité.
- Le fichier `.env` contenant des variables d’environnement sensibles est exclu du dépôt Git pour des raisons de sécurité.

---

Ce README sera mis à jour au fur et à mesure de l’avancement du projet.
