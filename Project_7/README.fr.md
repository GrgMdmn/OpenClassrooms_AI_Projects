# Projet Air Paradis – Analyse de sentiments

📘 Ce projet est également disponible en [anglais 🇬🇧](./README.md)

Ce dépôt contient les notebooks et le code source du projet d'analyse de sentiments sur des tweets, réalisé dans le cadre du programme OpenClassrooms **"Air Paradis"**.

---

## Contexte métier

L'objectif est de prédire si un tweet (ou une courte expression écrite) est perçu comme **positif ou négatif**, dans le contexte d'une compagnie aérienne.  
Le projet repose sur le jeu de données public **Sentiment140**.

---

## Description du projet

Le projet vise à développer un système complet d'analyse de sentiments, comprenant :

- Une analyse exploratoire des données textuelles,
- Une comparaison rigoureuse de 10+ modèles de classification (classiques ML, LSTM + word embeddings, DistilBERT),
- Le déploiement d'une API REST FastAPI exposant le meilleur modèle avancé (LSTM + Word2Vec),
- La mise en place d'un pipeline MLOps complet : MLFlow, CI/CD (GitHub Actions), monitoring,
- Un hébergement local sur serveur NAS pour garantir la souveraineté des données.

📖 **Pour des informations détaillées sur les expérimentations menées et les résultats obtenus, consultez [docs/blog.fr.md](./docs/blog.fr.md).**

---

## Structure du dépôt

- `notebooks/` : notebooks d'analyse exploratoire et de modélisation,
- `deployment/` : code de l'API FastAPI, tests unitaires, Dockerfile et configuration déploiement,
- `docs/` : documentation détaillée (blog.fr.md), diagrammes, captures d'écran,
- `presentation/` : support de présentation du projet (PDF),
- `../.github/workflows/` : pipelines CI/CD (tests automatisés, build Docker, push DockerHub).

---

## Entraînement des modèles

Les phases d'entraînement ont principalement été réalisées sur **Google Colab**, afin de bénéficier de ressources GPU nécessaires aux modèles de deep learning.

---

## Utilisation

Les notebooks détaillent les étapes d'analyse et d'entraînement.  
L'API est disponible dans le dossier `deployment/api` avec interface Streamlit de test.

Pour tester l'API en production : https://sentiment-api.greg-madman-nas.duckdns.org

---

## Remarques

- Les fichiers `requirements.txt` dans `notebooks/` proviennent des environnements Google Colab et peuvent contenir des dépendances peu optimisées pour un usage local.
- Le fichier `.env`, exclu du dépôt, contient les variables d'environnement nécessaires (configuration MLFlow, MinIO, alerting).

---

**Projet réalisé dans le cadre du parcours "Data Scientist – Projet 7" d'OpenClassrooms**