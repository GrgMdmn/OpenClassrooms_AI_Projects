# Air Paradis Project – Sentiment Analysis

📘 This project is also available in [French 🇫🇷](./README.fr.md)

This repository contains the notebooks and source code for the sentiment analysis project on tweets, carried out as part of the OpenClassrooms "Air Paradis" program.

---

## Business Context

The goal is to predict whether a tweet, or a short written expression, is perceived as **positive or negative**, in the context of an airline company.  
The project is based on the public **Sentiment140** dataset.

---

## Project Description

The project aims to develop a complete sentiment analysis system, including:

- Exploratory data analysis of text data,
- Comparison of several classification models (classical, advanced, and state-of-the-art),
- Deployment of a REST API exposing an advanced model (based on embeddings and LSTM),
- Implementation of a CI/CD pipeline to automate testing and deployment,
- Local hosting on a NAS server to ensure data sovereignty.

---

## Repository Structure

- `notebooks/`: analysis and modeling notebooks,
- `deployment/`: API code, tests, Dockerfile and deployment-related files,
- `docs/`: documentation and related materials,
- `../.github/workflows/`: CI/CD pipelines.

---

## Model Training

Training phases were mainly conducted on Google Colab to leverage GPU resources required for deep learning models.

---

## Usage

The notebooks detail the analysis and training steps. The API is available in `deployment/api`.

---

## Notes

- The `requirements.txt` files in `notebooks/` come from Google Colab environments and may contain dependencies that are not optimal for local use.
- The `.env` file, excluded from the repository, contains the necessary environment variables.

---

This README will be updated progressively as the project advances.  
Some code comments can remain in French (as well as notebook markdowns). All will be translated into English later.

---

**Project completed as part of the OpenClassrooms "Data Scientist – Project 7" program**
