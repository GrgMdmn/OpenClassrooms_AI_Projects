import os
from dotenv import load_dotenv
import re
import html
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.keras
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Téléchargements nécessaires NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# Charger les variables d'environnement depuis le fichier .env à la racine ou accessible
load_dotenv(dotenv_path='../../.env')

# Configuration de MLflow avec les variables d'environnement
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
mlflow_s3_endpoint_url = os.getenv("MLFLOW_S3_ENDPOINT_URL")
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")

# Définition du modèle de données pour la requête
class TweetRequest(BaseModel):
    tweet: str

# Fonction de prétraitement pour les tweets
def preprocess_tweet(tweet: str) -> str:
    """
    Prétraite un tweet en appliquant plusieurs transformations :
    - Conversion en minuscules
    - Remplacement des URLs, mentions et hashtags par des tokens spéciaux
    - Suppression des caractères spéciaux
    - Tokenisation et lemmatisation
    - Suppression des stopwords
    """
    if not isinstance(tweet, str):
        return ""

    tweet = tweet.lower()
    tweet = re.sub(r'https?://\S+|www\.\S+', '<URL>', tweet)
    tweet = re.sub(r'@\w+', '<MENTION>', tweet)
    tweet = re.sub(r'#(\w+)', r'# \1', tweet)
    tweet = html.unescape(tweet)
    tweet = re.sub(r'[^\w\s<>@#!?]', '', tweet)

    tokens = word_tokenize(tweet)

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    stop_words = set(stopwords.words('english'))
    important_words = {'no', 'not', 'nor', 'neither', 'never', 'nobody', 'none', 'nothing', 'nowhere'}
    stop_words = stop_words - important_words
    tokens = [token for token in tokens if token not in stop_words]

    return ' '.join(tokens)

# Initialisation FastAPI
app = FastAPI(title="API Prédiction Sentiment - Air Paradis")

# Chargement du modèle MLflow (URI fixe du modèle en stage Production)
MODEL_URI = "models:/SentimentAnalysisLSTM/Production"
try:
    model = mlflow.keras.load_model(MODEL_URI)
    print("Modèle chargé depuis MLflow avec succès.")
except Exception as e:
    print(f"Erreur lors du chargement du modèle MLflow : {e}")
    model = None

@app.get("/")
def root():
    return {"message": "API Prédiction Sentiment Air Paradis - OK"}

@app.post("/predict")
def predict_sentiment(request: TweetRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non disponible.")

    processed_tweet = preprocess_tweet(request.tweet)
    if processed_tweet == "":
        raise HTTPException(status_code=400, detail="Tweet vide ou invalide après prétraitement.")

    # Ici, il faudra adapter selon la manière dont ton modèle attend l'entrée
    # Par exemple, si ton modèle attend une séquence d'indices, il faut intégrer la vectorisation / tokenisation
    # Pour l'exemple, on fait un dummy wrap en liste
    input_data = [processed_tweet]

    # Modèle Keras attend généralement un array numpy, ou séquence tokenisée
    # À adapter selon ta pipeline exacte
    import numpy as np
    try:
        # Exemple naïf : on prédit directement sur la liste textuelle (à adapter !)
        prediction_prob = model.predict(input_data)[0][0]  # supposons sortie [prob_neg]
        sentiment = "positif" if prediction_prob >= 0.5 else "negatif"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'inférence : {e}")

    return {"sentiment": sentiment, "probability": float(prediction_prob)}

