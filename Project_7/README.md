# Air Paradis Project – Sentiment Analysis

📘 This project is also available in [French 🇫🇷](./README.fr.md)

This repository contains the notebooks and source code for the sentiment analysis project on tweets, conducted as part of the OpenClassrooms **"Air Paradis"** program.

---

## Business Context

The objective is to predict whether a tweet (or short written expression) is perceived as **positive or negative**, in the context of an airline company.  
The project is based on the public **Sentiment140** dataset.

---

## Project Description

The project aims to develop a complete sentiment analysis system, including:

- Exploratory analysis of textual data,
- Rigorous comparison of 10+ classification models (classical ML, LSTM + word embeddings, DistilBERT),
- Deployment of a FastAPI REST API exposing the best advanced model (LSTM + Word2Vec),
- Implementation of a complete MLOps pipeline: MLFlow, CI/CD (GitHub Actions), monitoring,
- Local hosting on NAS server to ensure data sovereignty.

📖 **For detailed information on experiments conducted and results obtained, see [docs/blog.md](./docs/blog.md).**

---

## Repository Structure

- `notebooks/`: exploratory analysis and modeling notebooks,
- `deployment/`: FastAPI API code, unit tests, Dockerfile and deployment configuration,
- `docs/`: detailed documentation (blog.md), diagrams, screenshots,
- `presentation/`: project presentation slides (PDF),
- `../.github/workflows/`: CI/CD pipelines (automated tests, Docker build, DockerHub push).

---

## Model Training

Training phases were primarily conducted on **Google Colab** to leverage GPU resources required for deep learning models.

---

## Usage

Notebooks detail the analysis and training steps.  
The API is available in the `deployment/api` folder with Streamlit testing interface.

To test the production API: https://sentiment-api.greg-madman-nas.duckdns.org

---

## Notes

- The `requirements.txt` files in `notebooks/` come from Google Colab environments and may contain dependencies not optimized for local use.
- The `.env` file, excluded from the repository, contains necessary environment variables (MLFlow, MinIO, alerting configuration).

---

**Project completed as part of OpenClassrooms "Data Scientist – Project 7" program**