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

Dans un souci de souveraineté des données, le déploiement sera préférentiellement déployé sur un NAS équippé d'un processeur Intel N100 (peu énergivore, mais probablement suffisant pour un calcul d'inférence LSTM).
Un déploiement sur cloud grand public pourra éventuellement être effectué en fonction des demandes du jury du projet.

Exemple de build et run :
```bash
docker build -t air-paradis-api .
docker run -p 8000:8000 --env-file ../.env air-paradis-api

---

## 🔁 Intégration continue

Un pipeline GitHub Actions (CI/CD) est prévu pour :
- **Lancer automatiquement les tests unitaires** à chaque push sur le dépôt GitHub.
- **Garantir la stabilité de l’API** avant tout déploiement (NAS ou cloud).

Les tests seront définis dans `test_api.py`, avec des cas simples de requêtes POST sur `/predict`.
