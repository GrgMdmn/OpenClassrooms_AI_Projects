# Déploiement de l’API – Projet Air Paradis

📘 Cette documentation est aussi disponible en [anglais 🇬🇧](./README.md)

Ce dossier contient le code et la configuration nécessaires au déploiement de l’API de prédiction de sentiment.

---

## 🎯 Objectif

Déployer une API REST permettant de :

- Prédire le **sentiment d’un tweet** (positif ou négatif) à partir d’un modèle LSTM entraîné sur des embeddings (Word2Vec ou GloVe),
- Signaler les erreurs de prédiction.

L’API utilise le **meilleur modèle enregistré** dans MLflow et le sert via **FastAPI**.

---

## 📦 Architecture globale

Schéma :

```
    ┌──────────────────────────────┐
    │    Utilisateur final (UI)    │
    │    (interface Streamlit)     │
    └────────────┬─────────────────┘
                 │
      [Appel HTTP vers /predict]
                 │
    ┌────────────▼─────────────┐
    │         API FastAPI      │
    │  (serveur lancé via      │
    │   Uvicorn)               │
    └────────────┬─────────────┘
                 │
    Chargement du modèle (MLflow)
                 │
     Prétraitement du tweet
                 │
        Prédiction (inférence)
                 │
     Réponse JSON avec le sentiment
```

---

## 🧪 Fonctionnement de l’API

1. **Chargement du modèle**
   - Le meilleur modèle est enregistré dans MLflow sous le nom `SentimentAnalysisLSTM`, avec le stage `Production`.
   - L’API le charge au démarrage via l’URI :

     ```
     models:/SentimentAnalysisLSTM/Production
     ```

2. **Prétraitement du texte**
   - Nettoyage, tokenisation, lemmatisation, puis conversion en vecteurs d’embedding.

3. **Prédiction**
   - Le modèle LSTM prédit le sentiment.
   - Le résultat (`positive` ou `negative`) est retourné en JSON.

4. **Interface utilisateur (optionnelle)**
   - Une interface Streamlit permet une interaction visuelle avec l’API.

---

## 🐳 Déploiement via Docker

Un `Dockerfile` est fourni pour simplifier le déploiement sur tout hôte compatible Docker (NAS, cloud, etc.).

L’image Docker est poussée sur DockerHub à chaque version validée (voir CI/CD plus bas).

Par souci de **souveraineté des données**, le déploiement est préféré sur un **NAS équipé d’un processeur Intel N100** (peu énergivore et suffisant pour l’inférence LSTM).

Le redéploiement peut être automatisé via :
- une tâche `cron` simple, ou
- un conteneur **Watchtower** (dans docker-compose ou kubernetes).

**Commande pour test local :**

```bash
docker build -t sentiment-api .
docker run --rm --env-file ../../.env -p 8000:8000 -p 8080:8080 sentiment-api
```

En production, les variables d’environnement doivent être définies dans le `docker-compose.yml` ou via un gestionnaire de secrets.

---

## 🛠 Problèmes AVX2 / Déploiement cloud

**Mise à jour :** au départ, le NAS ne supportait pas les instructions AVX2 nécessaires à TensorFlow.  
En réalité, les CPU N100 les prennent en charge, mais certains fabricants désactivent cette option dans le BIOS.

En attendant un accès physique au NAS, nous avons opté pour un déploiement sur **Google Cloud**.  
Problème : un seul port peut être exposé publiquement sur ces services.

Deux solutions possibles :
- Séparer le backend (FastAPI) et le frontend (Streamlit) en deux conteneurs,
- Utiliser un **reverse proxy** (ex. : nginx) pour rediriger les ports.

➡️ Nous avons choisi la **deuxième solution**, pour garder une **seule image Docker**.

**Mise à jour 2 :** AVX2 a été activé via le BIOS. Le déploiement est donc opérationnel **à la fois en local (NAS)** et **sur le cloud**, avec déploiement automatique pour le NAS (voir section suivante).

---

## 🔁 CI/CD – GitHub Actions

Un pipeline est défini dans `.github/workflows/` pour :

- Lancer les tests unitaires à chaque `push`,
- Vérifier la stabilité de l’API avant déploiement (NAS ou cloud),
- Construire et pousser l’image Docker sur DockerHub.

Les tests sont situés dans :

```
deployment/api/tests/
```

Le redéploiement automatique sur le NAS est géré via **Watchtower**, qui surveille les mises à jour de l’image Docker (fonctionne comme un `cron`, avec logs).

---
