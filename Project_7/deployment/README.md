# API Deployment - Air Paradis Project

This folder contains the code and configuration necessary to deploy the sentiment prediction API.

## 🎯 Objective

Deploy a REST API that allows to:
- predict the sentiment of a tweet (positive or negative) based on an LSTM model trained on embeddings (Word2Vec or GloVe).
- report prediction errors

The API uses the best-performing model, registered on MLflow, and serves it via FastAPI.

---

## 📦 Overall Architecture

Diagram:

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

---

## 🧪 API Workflow

1. Model loading
   - The best performing model is registered in MLflow under a fixed name (`SentimentAnalysisLSTM`) and a stage `Production`.
   - The API loads this model at startup using a stable URI:
     
     models:/SentimentAnalysisLSTM/Production

2. Text preprocessing
   - The text sent by the user is cleaned, tokenized, lemmatized, and transformed into vectors (embeddings).

3. Prediction
   - The LSTM model (loaded from MLflow) performs inference.
   - The result (`positive` or `negative`) is returned as a JSON response.

4. User interface (optional)
   - A Streamlit interface allows visual interaction with the API.

---

## 🐳 Deployment with Docker

A Dockerfile is provided to facilitate deploying the API on any Docker-compatible host (NAS, cloud, etc.).

This Docker image should be pushed to DockerHub at each validated versioning (see the CI/CD section below).

For data sovereignty reasons, deployment will preferably be done on a NAS equipped with an Intel N100 processor (energy efficient but probably sufficient for LSTM inference).
Docker can be redeployed each time the docker image on DockerHub is updated. Man can use `cron` instruction (simple) or `Watchtower` docker (docker-compose / kubernetes context).

For running the Docker image locally (development), it is important to provide the `.env` file containing environment variables and to open the FastAPI and Streamlit ports for testing the application.

Example commands to build and run:

    docker build -t sentiment-api .
    docker run --env-file ../../.env -p 8000:8000 -p 8501:8501 sentiment-api

For deployment on a personal or public cloud (with docker-compose, Kubernetes), it is preferable to specify the necessary environment variables in `docker-compose.yml` or in secrets.

Add the `--rm` argument so that the container is removed after stopping.

**UPDATE** : in our case, we encountered some issues with the NAS which CPU is not able to run AVX2 instructions so we are obliged to deploy it on a "standard" public cloud service. To be precise, N100 CPU is normally able to run AXV2 instructions but some NAS motherboard manufacturers sometimes disable this feature. Maybe is it possible to renable it on the BIOS, but I am away from my NAS so it will have to wait.

So, we decided to use Google Cloud. A common issue arises here: we cannot expose multiple ports (contrary to the instruction above) on a cloud service.

Two options are possible:

- Separate the service into two dockers (one for FastAPI backend and one for Streamlit frontend)
- Configure a reverse proxy like nginx to redirect requests to different internal ports

We decided to keep the second solution to install the API from a single Docker container.

---

## 🔁 Continuous Integration / Continuous Deployment

A GitHub Actions pipeline (CI/CD) is planned at the very root of the repo to:

- Automatically run unit tests at every push on the GitHub repository.
- Ensure API stability before any deployment (NAS or cloud).
- Update the Dockerfile and push it to DockerHub.

Unit tests will be defined in `./deployment/api/tests/`
