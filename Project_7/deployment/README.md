# API Deployment – Air Paradis Project

📘 This documentation is also available in [French 🇫🇷](./README.fr.md)

This folder contains the code and configuration necessary to deploy the sentiment prediction API.

---

## 🎯 Objective

Deploy a REST API that allows to:

- Predict the sentiment of a tweet (positive or negative) using an LSTM model trained on embeddings (Word2Vec or GloVe),
- Report prediction errors.

The API uses the best-performing model registered on MLflow and serves it via FastAPI.

---

## 📦 Overall Architecture

Diagram:

```
    ┌──────────────────────────────┐
    │    End user (UI)             │
    │    (Streamlit interface)     │
    └────────────┬─────────────────┘
                 │
      [HTTP call to /predict]
                 │
    ┌────────────▼─────────────┐
    │       FastAPI API        │
    │  (server launched via    │
    │   Uvicorn)               │
    └────────────┬─────────────┘
                 │
    Model loading from MLflow
                 │
      Tweet preprocessing
                 │
          Prediction (inference)
                 │
       Sentiment response (JSON)
```

---

## 🧪 API Workflow

1. **Model loading**
   - The best performing model is registered in MLflow under a fixed name (`SentimentAnalysisLSTM`) and a `Production` stage.
   - The API loads this model at startup using the URI:
     
     ```
     models:/SentimentAnalysisLSTM/Production
     ```

2. **Text preprocessing**
   - Text is cleaned, tokenized, lemmatized, and converted into vectors (embeddings).

3. **Prediction**
   - The LSTM model (loaded from MLflow) makes a prediction.
   - The result (`positive` or `negative`) is returned as JSON.

4. **User interface (optional)**
   - A Streamlit interface allows visual interaction with the API.

---

## 🐳 Deployment with Docker

A `Dockerfile` is provided for easy deployment on any Docker-compatible host (NAS, cloud, etc.).

This Docker image should be pushed to DockerHub with every validated version (see CI/CD below).

For data sovereignty, deployment is preferably done on a **NAS with Intel N100** CPU (energy-efficient, sufficient for inference).  
Docker can be automatically redeployed using:
- `cron` (simple), or  
- `Watchtower` in docker-compose/kubernetes setups.

**Development command (local test):**

```bash
docker build -t sentiment-api .
docker run --rm --env-file ../../.env -p 8000:8000 -p 8080:8080 sentiment-api
```

In production (e.g. via docker-compose or cloud services), environment variables must be defined in the compose file or secret manager.

---

## 🛠 Notes on AVX2 / Cloud Deployment

**Update:** the NAS initially failed due to lack of AVX2 support. Some N100-based NAS motherboards disable AVX2 in BIOS.

When not accessible, fallback deployment is done on **Google Cloud**.  
Issue: public cloud services do **not allow multiple ports** to be exposed easily.

Two solutions:
- Split into two containers: FastAPI + Streamlit,
- Use a **reverse proxy** (e.g., Nginx) to map ports.

➡️ We chose the second option to keep a **single Docker image**.

**Update 2:** AVX2 was enabled via BIOS. Deployment is now active **both locally (NAS)** and **on cloud**, with continuous deployment on NAS (see section below).

---

## 🔁 CI/CD – GitHub Actions

A pipeline is defined in `.github/workflows/` to:

- Run unit tests on every push,
- Check API health before deployment (local/cloud),
- Build and push Docker image to DockerHub.

Tests are defined in:

```
deployment/api/tests/
```

Docker image auto-redeployment on NAS is handled by **Watchtower**, which checks DockerHub updates (like a cron job, with logs).

---
