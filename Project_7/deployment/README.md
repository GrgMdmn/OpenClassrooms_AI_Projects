# Déploiement de l'API - Projet Air Paradis

Ce dossier contient le code et la configuration nécessaires pour déployer l'API de prédiction de sentiments.

## 🎯 Objectif

Déployer une API REST qui permet de:
- prédire le sentiment d’un tweet (positif ou négatif) en s’appuyant sur un modèle LSTM entraîné à partir d’embeddings (Word2Vec ou GloVe).
- reporter les erreurs de prédictions


L’API utilise le modèle le plus performant, enregistré sur MLflow, et le met à disposition via FastAPI.

---

## 📦 Architecture Globale

```
┌──────────────────────────────┐
│  Utilisateur final (UI)      │
│  (interface Streamlit)       │
└────────────┬─────────────────┘
             │
  [Appel HTTP à /predict]
             │
┌────────────▼─────────────┐
│       API FastAPI        │
│  (serveur lancé via      │
│   Uvicorn)               │
└────────────┬─────────────┘
             │
Chargement du modèle depuis MLflow
             │
   Prétraitement du tweet
             │
     Prédiction (inférence)
             │
    Retour du sentiment (JSON)
```

---

## 🧪 Fonctionnement de l’API

1. **Chargement du modèle**
   - Le modèle le plus performant est enregistré dans MLflow sous un nom fixe (`SentimentAnalysisLSTM`) et une étape (stage) `Production`.
   - L’API récupère ce modèle au démarrage grâce à une URI stable :  
     ```
     models:/SentimentAnalysisLSTM/Production
     ```

2. **Prétraitement du texte**
   - Le texte envoyé par l’utilisateur est nettoyé, tokenisé, lemmatisé et transformé en vecteurs (embeddings).

3. **Prédiction**
   - Le modèle LSTM (chargé depuis MLflow) effectue l’inférence.
   - Le résultat (`positif` ou `négatif`) est renvoyé en réponse JSON.

4. **Interface utilisateur (optionnelle)**
   - Une interface Streamlit permet d’interagir avec l’API de manière visuelle.

---

## 🐳 Déploiement avec Docker

Un `Dockerfile` est fourni pour faciliter le déploiement de l’API sur n’importe quel hôte compatible Docker (NAS, cloud, etc.).

Ce docker devra être déploigné sur DockerHub à chaque versionning validé (voir section CI/CD ci-dessous)

Dans un souci de souveraineté des données, le déploiement sera préférentiellement déployé sur un NAS équippé d'un processeur Intel N100 (peu énergivore, mais probablement suffisant pour un calcul d'inférence LSTM).

Pour le lancement du docker en local (développement), il sera important de donner en entrée le fichier `.env` contenant les variables d'environnement ainsi que qu'ouvrir les ports de FastAPI et Streamlit pour pouvoir tester l'application.

`docker build -t sentiment-api`
`docker run --env-file ../../.env -p 8000:8000 -p 8501:8501 sentiment-api`

Pour un déploiement sur un cloud personnel ou grand public (avec docker-compose, kubernetes), il sera préférable de renseigner les variables d'environnement nécessaires dans `docker-compose.yml` ou dans les secrets.

Dans notre cas, il a été décidé d'utiliser Google Cloud. Une problématique classique se pose ici pour nous : on ne peut exposer  plusieurs ports (contrairement à l'instruction ci-dessus) sur un service cloud.
2 options sont donc possibles:
- Séparer le service en deux docker (un pour le backend fastAPI et un pour le frontend streamlit)
- Configurer un reverse proxy de type nginx pour qui aura pour but de rediriger les requêtes sur différents ports internes

Nous avons décidé de retenir la deuxième solution afin de pouvoir installer l'API à partir d'un seul docker.

---

## 🔁 Intégration continue / Déploiement Continu

Un pipeline GitHub Actions (CI/CD) est prévu pour :
- **Lancer automatiquement les tests unitaires** à chaque push sur le dépôt GitHub.
- **Garantir la stabilité de l’API** avant tout déploiement (NAS ou cloud).
- **Mettre à jour le Dockerfile** et le pousser sur **DockerHub**.

Les tests unitaires seront définis dans `./deployment/api/tests/`
