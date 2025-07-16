# Projet Air Paradis – Analyse de sentiments

📘 Ce projet est également disponible en [anglais 🇬🇧](./README.md)

Ce dépôt contient les notebooks et le code source du projet d’analyse de sentiments sur des tweets, réalisé dans le cadre du programme OpenClassrooms **"Air Paradis"**.

---

## Contexte métier

L’objectif est de prédire si un tweet (ou une courte expression écrite) est perçu comme **positif ou négatif**, dans le contexte d’une compagnie aérienne.  
Le projet repose sur le jeu de données public **Sentiment140**.

---

## Description du projet

Le projet vise à développer un système complet d’analyse de sentiments, comprenant :

- Une analyse exploratoire des données textuelles,
- Une comparaison de plusieurs modèles de classification (classiques, avancés et à l’état de l’art),
- Le déploiement d’une API REST exposant un modèle avancé (embeddings + LSTM),
- La mise en place d’un pipeline CI/CD pour automatiser les tests et le déploiement,
- Un hébergement local sur un serveur NAS pour garantir la souveraineté des données.

---

## Structure du dépôt

- `notebooks/` : notebooks d’analyse et de modélisation,
- `deployment/` : code de l’API, tests, Dockerfile et fichiers liés au déploiement,
- `docs/` : documentation et ressources associées,
- `../.github/workflows/` : pipelines CI/CD.

---

## Entraînement des modèles

Les phases d’entraînement ont principalement été réalisées sur **Google Colab**, afin de bénéficier de ressources GPU nécessaires aux modèles de deep learning.

---

## Utilisation

Les notebooks détaillent les étapes d’analyse et d’entraînement.  
L’API est disponible dans le dossier `deployment/api`.

---

## Remarques

- Les fichiers `requirements.txt` dans `notebooks/` proviennent des environnements Google Colab et peuvent contenir des dépendances peu optimisées pour un usage local.
- Le fichier `.env`, exclu du dépôt, contient les variables d’environnement nécessaires.

---

Ce README sera mis à jour progressivement au fur et à mesure de l’avancement du projet.  
Certains commentaires de code peuvent rester en français (ainsi que les cellules Markdown des notebooks). Tout sera traduit en anglais ultérieurement.

---

**Projet réalisé dans le cadre du parcours "Data Scientist – Projet 7" d’OpenClassrooms**