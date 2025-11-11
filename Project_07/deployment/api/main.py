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
import pandas as pd
from datetime import datetime, timedelta

# # TO UNCOMMENT FOR DEVELOPMENT. OTHERWISE, LEAVE AS IS
# # Access the .env file
# PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
# sys.path.append(PARENT_DIR)
# # 🔐 Load environment variables (.env two levels above)
# load_dotenv()

# 🔁 Import the preprocessing function (LOCAL)
from preprocessing import preprocess_tweet

# 📧 Import the email service
from email_service import send_error_report_email

# 🔗 Configure MLflow
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if mlflow_tracking_uri:
    mlflow.set_tracking_uri(mlflow_tracking_uri)
else:
    print("⚠️ MLFLOW_TRACKING_URI not defined in .env")

# 📌 Model name and stage
MODEL_NAME = "SentimentAnalysisLSTM"
STAGE = "Production"

# 🧾 Structure of POST requests
class TweetRequest(BaseModel):
    tweet: str

class ReportRequest(BaseModel):
    tweet: str
    prediction: str
    probability: float

# 🧠 Storage for error reports
error_reports = {}

# 🔍 Function to load model + tokenizer
def load_model_and_artifacts(model_name=MODEL_NAME, stage=STAGE):
    client = MlflowClient()
    versions = client.get_latest_versions(model_name, [stage])
    
    if not versions:
        raise RuntimeError(f"No version found for model '{model_name}' in stage '{stage}'")
    
    version_info = versions[0]
    run_id = version_info.run_id
    
    if not run_id:
        raise RuntimeError("Run ID not found.")
    
    # Extract embedding tag
    model_version_info = client.get_model_version(model_name, version_info.version)
    # glove is used as default if embedding_type is not found
    embedding_type = model_version_info.tags.get("embedding_type", "glove").lower()
    
    # Determine paths to artifacts
    local_dir = "./downloaded_artifacts"
    os.makedirs(local_dir, exist_ok=True)
    base_path = "local_artifacts"
    tokenizer_file = "tokenizer.pickle"
    
    if "glove" in embedding_type:
        keras_file = "final_model_LSTM_GloVe-300d-Fige.keras"
    elif "word2vec" in embedding_type:
        keras_file = "final_model_LSTM_Word2Vec-Fige.keras"
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")
        
    print(f"{embedding_type} model found. Downloading model associated with run_id: {run_id}")
    
    def dl(path):
        return client.download_artifacts(run_id=run_id, path=f"{base_path}/{path}", dst_path=local_dir)
    
    # 🔽 Download artifacts
    model_path = dl(keras_file)
    tokenizer_path = dl(tokenizer_file)
    print("Contents of downloaded_artifacts:", os.listdir(local_dir))
    
    # 📦 Load
    model = load_model(model_path)
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    
    return model, tokenizer, embedding_type

# 🚀 Launch the API
app = FastAPI(title="Sentiment Prediction API - Air Paradis")
# app = FastAPI(title="Sentiment Prediction API - Air Paradis",
#               root_path="/api")

try:
    model, tokenizer, embedding_type = load_model_and_artifacts()
except Exception as e:
    print(f"\n❌ Error loading artifacts: {e}")
    model, tokenizer = None, None

@app.get("/")
def root():
    return {"message": "Sentiment API - Air Paradis - online"}

@app.post("/predict")
def predict_sentiment(request: TweetRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model or artifacts unavailable.")
    
    try:
        processed_tweet = preprocess_tweet(request.tweet)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessing error: {e}")
    
    if not processed_tweet.strip():
        raise HTTPException(status_code=400, detail="Tweet is empty or invalid after preprocessing.")
    
    try:
        sequence = tokenizer.texts_to_sequences([processed_tweet])
        sequence_padded = pad_sequences(sequence, maxlen=model.input_shape[1])
        prediction_prob = model.predict(sequence_padded)[0][0]
        if prediction_prob >= 0.5:
            sentiment = "positive"
        else:
            sentiment = "negative"
            # prediction_prob = 1 - prediction_prob
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
    
    return {
        "sentiment": sentiment,
        "probability": float(prediction_prob)
    }

# Replace the dict with a global empty DataFrame initially
error_reports_df = pd.DataFrame(columns=["tweet", "prediction", "probability", "timestamp"])

@app.post("/report_error")
def report_error(request: ReportRequest):
    global error_reports_df

    tweet = request.tweet.strip()
    prediction = request.prediction.strip().lower()
    probability = request.probability

    if not tweet or not prediction:
        raise HTTPException(status_code=400, detail="Missing tweet or prediction.")
    if probability < 0 or probability > 1:
        raise HTTPException(status_code=400, detail="Invalid probability. Must be between 0 and 1.")

    now = datetime.utcnow()

    # Check if tweet already exists to avoid duplicates
    existing_idx = error_reports_df.index[error_reports_df["tweet"] == tweet].tolist()
    if existing_idx:
        # Update existing row
        idx = existing_idx[0]
        error_reports_df.at[idx, "prediction"] = prediction + f' (p = {probability:.2f} ) '
        error_reports_df.at[idx, "probability"] = probability
        error_reports_df.at[idx, "timestamp"] = now
    else:
        # Add new row
        new_row = {
            "tweet": tweet,
            "prediction": prediction + f' (p = {probability:.2f} ) ',
            "probability": probability,
            "timestamp": now
        }
        error_reports_df = pd.concat([error_reports_df, pd.DataFrame([new_row])], ignore_index=True)

    report_sent = False

    # Condition: every 3 reports and max 5 min interval between the last 3
    try:
        if len(error_reports_df) % 3 == 0:
            last_three = error_reports_df.sort_values("timestamp", ascending=False).head(3)
            times = last_three["timestamp"].sort_values()
            delta = times.iloc[-1] - times.iloc[0]
            if delta <= timedelta(minutes=5):
                reports_dict = dict(zip(
                    last_three["tweet"],
                    last_three["prediction"]
                ))
                print(f"📧 Attempting to send email for {len(error_reports_df)} reports...")
                email_success = send_error_report_email(reports_dict)
                if email_success:
                    print(f"✅ Email successfully sent for {len(error_reports_df)} reports")
                    report_sent = True
                    # DELETE the last 3 reports sent
                    # Identify indexes to delete
                    to_remove_idx = last_three.index
                    error_reports_df = error_reports_df.drop(to_remove_idx).reset_index(drop=True)
                else:
                    print(f"❌ Failed to send email for {len(error_reports_df)} reports")

    except Exception as e:
        print(f"❌ Error while sending email: {e}")
        # Continue even if it fails

    return {"report_sent": report_sent}
    