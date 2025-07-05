import os
import sys
import pickle
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
from mlflow.tracking import MlflowClient
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 🔁 Import de la fonction de prétraitement depuis ../../functions.py
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(PARENT_DIR)
from functions import preprocess_tweet

# 📧 Import du service email
from email_service import send_error_report_email

# 🔐 Charger les variables d'environnement (.env à deux niveaux au-dessus)
load_dotenv(dotenv_path=os.path.join(PARENT_DIR, ".env"))

# 🔗 Configurer MLflow
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if mlflow_tracking_uri:
    mlflow.set_tracking_uri(mlflow_tracking_uri)
else:
    print("⚠️ MLFLOW_TRACKING_URI non défini dans .env")

# 📌 Nom du modèle et stage
MODEL_NAME = "SentimentAnalysisLSTM"
STAGE = "Production"

# 🧾 Structures des requêtes POST
class TweetRequest(BaseModel):
    tweet: str

class ReportRequest(BaseModel):
    tweet: str
    prediction: str

# 🧠 Stockage des signalements d'erreurs
error_reports = {}

# 🔍 Fonction pour charger modèle + tokenizer
def load_model_and_artifacts(model_name=MODEL_NAME, stage=STAGE):
    client = MlflowClient()
    versions = client.get_latest_versions(model_name, [stage])
    
    if not versions:
        raise RuntimeError(f"Aucune version du modèle '{model_name}' en stage '{stage}'")
    
    version_info = versions[0]
    run_id = version_info.run_id
    
    if not run_id:
        raise RuntimeError("Run ID introuvable.")
    
    # Extraire le tag d'embedding
    model_version_info = client.get_model_version(model_name, version_info.version)
    embedding_type = model_version_info.tags.get("embedding_type", "glove").lower()
    
    # Déterminer les chemins des artefacts
    local_dir = "./downloaded_artifacts"
    os.makedirs(local_dir, exist_ok=True)
    base_path = "local_artifacts"
    tokenizer_file = "tokenizer.pickle"
    
    if "glove" in embedding_type:
        keras_file = "final_model_LSTM_GloVe-300d-Fige.keras"
    elif "word2vec" in embedding_type:
        keras_file = "final_model_LSTM_Word2Vec-Fige.keras"
    else:
        raise ValueError(f"Embedding type inconnu : {embedding_type}")
    
    def dl(path):
        return client.download_artifacts(run_id=run_id, path=f"{base_path}/{path}", dst_path=local_dir)
    
    # 🔽 Télécharger les artefacts
    model_path = dl(keras_file)
    tokenizer_path = dl(tokenizer_file)
    
    # 📦 Chargement
    model = load_model(model_path)
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    
    return model, tokenizer, embedding_type

# 🚀 Lancer l'API
app = FastAPI(title="API Prédiction Sentiment - Air Paradis")

try:
    model, tokenizer, embedding_type = load_model_and_artifacts()
except Exception as e:
    print(f"\n❌ Erreur lors du chargement des artefacts : {e}")
    model, tokenizer = None, None

@app.get("/")
def root():
    return {"message": "API Sentiment - Air Paradis - en ligne"}

@app.post("/predict")
def predict_sentiment(request: TweetRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Modèle ou artefacts indisponibles.")
    
    try:
        processed_tweet = preprocess_tweet(request.tweet)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prétraitement : {e}")
    
    if not processed_tweet.strip():
        raise HTTPException(status_code=400, detail="Tweet vide ou invalide après prétraitement.")
    
    try:
        sequence = tokenizer.texts_to_sequences([processed_tweet])
        sequence_padded = pad_sequences(sequence, maxlen=model.input_shape[1])
        prediction_prob = model.predict(sequence_padded)[0][0]
        sentiment = "positif" if prediction_prob >= 0.5 else "negatif"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur d'inférence : {e}")
    
    return {
        "sentiment": sentiment,
        "probability": float(prediction_prob)
    }

@app.post("/report_error")
def report_error(request: ReportRequest):
    tweet = request.tweet.strip()
    prediction = request.prediction.strip().lower()
    
    if not tweet or not prediction:
        raise HTTPException(status_code=400, detail="Tweet ou prédiction manquants.")
    
    # Ajouter ou mettre à jour le signalement du tweet
    error_reports[tweet] = prediction
    
    report_sent = False
    
    # Condition : à chaque multiple de 3 signalements totaux
    if len(error_reports) % 3 == 0:
        report_sent = True
        
        # 📧 NOUVEAU : Envoyer l'email de rapport
        try:
            print(f"📧 Tentative d'envoi d'email pour {len(error_reports)} signalements...")
            email_success = send_error_report_email(error_reports.copy())
            
            if email_success:
                print(f"✅ Email envoyé avec succès pour {len(error_reports)} signalements")
            else:
                print(f"❌ Échec de l'envoi de l'email pour {len(error_reports)} signalements")
                
        except Exception as e:
            print(f"❌ Erreur lors de l'envoi de l'email : {e}")
            # On continue même si l'email échoue pour ne pas bloquer l'API
    
    return {"report_sent": report_sent}
