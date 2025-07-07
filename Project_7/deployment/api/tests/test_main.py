import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "API Sentiment - Air Paradis - en ligne"}

def test_predict_sentiment():
    response = client.post("/predict", json={"tweet": "J'adore cette compagnie aérienne !"})
    assert response.status_code == 200
    data = response.json()
    assert "sentiment" in data
    assert "probability" in data
    assert data["sentiment"] in ["positif", "negatif"]

def test_missing_tweet():
    response = client.post("/predict", json={})
    assert response.status_code == 422  # Unprocessable Entity (FastAPI)

def test_empty_tweet():
    response = client.post("/predict", json={"tweet": ""})
    assert response.status_code == 400
    assert response.json()["detail"] == "Tweet vide ou invalide après prétraitement."