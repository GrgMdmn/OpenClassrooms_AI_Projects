import os
import sys
from fastapi.testclient import TestClient

# Ajouter le répertoire parent au path pour importer email_service
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from main import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "API Sentiment - Air Paradis - en ligne"}

def test_predict_sentiment():
    response = client.post("/predict", json={"tweet": "I love this airline !"})
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
    assert response.json()["detail"] == "Empty or invalid tweet after pre-processing."