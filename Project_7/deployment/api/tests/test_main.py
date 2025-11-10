import os
import sys
from fastapi.testclient import TestClient

# Add parent directory to path to import email_service
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from main import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Sentiment API - Air Paradis - online"}

def test_predict_sentiment():
    response = client.post("/predict", json={"tweet": "I love this airline !"})
    assert response.status_code == 200
    data = response.json()
    assert "sentiment" in data
    assert "probability" in data
    assert data["sentiment"] in ["positive", "negative"]

def test_missing_tweet():
    response = client.post("/predict", json={})
    assert response.status_code == 422  # Unprocessable Entity (FastAPI)

def test_empty_tweet():
    response = client.post("/predict", json={"tweet": ""})
    assert response.status_code == 400
    assert response.json()["detail"] == "Tweet is empty or invalid after preprocessing."