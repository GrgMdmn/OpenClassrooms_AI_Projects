import os
import dill
import pickle
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
from mlflow.tracking import MlflowClient
import mlflow.keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Charger les variables d'environnement
load_dotenv(dotenv_path='../../.env')

#%%
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if mlflow_tracking_uri:
    mlflow.set_tracking_uri(mlflow_tracking_uri)
else:
    print("⚠️ Attention : MLFLOW_TRACKING_URI non défini dans .env")

# Modèle et étape par défaut
MODEL_NAME = "SentimentAnalysisLSTM"
STAGE = "Production"

class TweetRequest(BaseModel):
    tweet: str

def get_latest_version(client, model_name, stage):
    versions = client.search_model_versions(f"name='{model_name}'")
    filtered = [v for v in versions if v.current_stage == stage]
    if not filtered:
        return None
    latest = max(filtered, key=lambda v: int(v.version))
    return latest

def load_model_and_artifacts(model_name="SentimentAnalysisLSTM", stage="Production"):
    client = MlflowClient()
    versions = client.get_latest_versions(model_name, [stage])
    if not versions:
        raise RuntimeError(f"Aucune version du modèle '{model_name}' en stage '{stage}'")
    version = versions[0]
    run_id = version.run_id
    if run_id is None:
        raise RuntimeError("'run_id' introuvable.")

    # Récupérer les tags du modèle pour trouver embedding_type
    model_version_info = client.get_model_version(model_name, version.version)
    embedding_type = model_version_info.tags.get("embedding_type", "glove").lower()

    local_dir = "./downloaded_artifacts"
    os.makedirs(local_dir, exist_ok=True)

    base_path = "local_artifacts/logs"

    if embedding_type == "glove":
        keras_file = "final_model_LSTM_GloVe-300d-Fige.keras"
        tokenizer_file = "tokenizer.pickle"
        preprocess_file = "preprocess_function.dill"
    elif embedding_type == "word2vec":
        keras_file = "final_model_LSTM_Word2Vec.keras"
        tokenizer_file = "tokenizer_word2vec.pickle"
        preprocess_file = "preprocess_function_word2vec.dill"
    else:
        raise ValueError(f"Type d'embedding inconnu: {embedding_type}")

    def dl(path):
        return client.download_artifacts(run_id=run_id, path=f"{base_path}/{path}", dst_path=local_dir)

    model_path = dl(keras_file)
    tokenizer_path = dl(tokenizer_file)
    preprocess_path = dl(preprocess_file)

    model = mlflow.keras.load_model(model_path)
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    with open(preprocess_path, "rb") as f:
        preprocess_fn = dill.load(f)

    return model, tokenizer, preprocess_fn, embedding_type


app = FastAPI(title="API Prédiction Sentiment - Air Paradis")

try:
    model, tokenizer, preprocess_tweet = load_model_and_artifacts()
except Exception as e:
    print(f"❌ Erreur lors du chargement des artefacts : {e}")
    model, tokenizer, preprocess_tweet = None, None, None

@app.get("/")
def root():
    return {"message": "API Prédiction Sentiment Air Paradis - OK"}

@app.post("/predict")
def predict_sentiment(request: TweetRequest):
    if model is None or tokenizer is None or preprocess_tweet is None:
        raise HTTPException(status_code=503, detail="Modèle ou artefacts non disponibles.")

    processed_tweet = preprocess_tweet(request.tweet)
    if not processed_tweet.strip():
        raise HTTPException(status_code=400, detail="Tweet vide ou invalide après prétraitement.")

    sequence = tokenizer.texts_to_sequences([processed_tweet])
    sequence_padded = pad_sequences(sequence, maxlen=100)  # Ajuster maxlen si besoin

    try:
        prediction_prob = model.predict(sequence_padded)[0][0]
        sentiment = "positif" if prediction_prob >= 0.5 else "negatif"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'inférence : {e}")

    return {"sentiment": sentiment, "probability": float(prediction_prob)}
