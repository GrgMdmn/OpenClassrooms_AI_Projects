import os
import streamlit as st
import requests

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8080") + "/api"


st.set_page_config(
    page_title="Air Paradis - Tweet Prediction",
    page_icon="✈️",  # optionnel, tu peux mettre un emoji ou un fichier image
)
st.title("Sentiment Analysis - Air Paradis")


tweet = st.text_area("Prompt your tweet here :", height=150)

# Initialisation des états
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "show_report_button" not in st.session_state:
    st.session_state.show_report_button = False

if "report_sent" not in st.session_state:
    st.session_state.report_sent = False

if "last_report_sent" not in st.session_state:
    st.session_state.last_report_sent = False

def reset_state():
    st.session_state.prediction_result = None
    st.session_state.show_report_button = False
    st.session_state.report_sent = False
    st.session_state.last_report_sent = False

# Bouton d'analyse
if st.button("Analyze the sentiment"):
    if not tweet.strip():
        st.warning("Please type a valid tweet.")
        reset_state()
    else:
        try:
            response = requests.post(
                f"{API_BASE_URL}/predict",
                json={"tweet": tweet}
            )
            if response.status_code == 200:
                data = response.json()
                st.session_state.prediction_result = data
                st.session_state.show_report_button = True
                st.session_state.report_sent = False
                st.session_state.last_report_sent = False
            else:
                st.error(f"Erreur API : {response.status_code} - {response.text}")
                reset_state()
        except Exception as e:
            st.error(f"Error during API request : {e}")
            reset_state()

# Affichage du résultat
if st.session_state.prediction_result:
    sentiment = st.session_state.prediction_result["sentiment"].capitalize()
    prob = st.session_state.prediction_result["probability"]
    st.success(f"Sentiment : {sentiment}")
    st.info(f"Positive tweet probability: {prob:.2f}")

# Gestion du bouton signalement
if st.session_state.show_report_button and not st.session_state.report_sent:
    if st.button("Report a wrong prediction"):
        try:
            response = requests.post(
                f"{API_BASE_URL}/report_error",
                json={
                    "tweet": tweet,
                    "prediction": sentiment,
                    "probability": st.session_state.prediction_result["probability"]  # Transmettre la probabilité
                }
            )

            if response.status_code == 200:
                data = response.json()
                # Mettre à jour les états
                st.session_state.show_report_button = False
                st.session_state.report_sent = True
                st.session_state.last_report_sent = data.get("report_sent", False)
                # Forcer un re-run
                st.rerun()
            else:
                st.error(f"API Error : {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Error during API request : {e}")

# Message si signalement déjà fait
elif st.session_state.report_sent:
    st.success("✅ Thanks for your report.")
    
    if st.session_state.last_report_sent:
        st.info("📩 A complete report has been sent to the website administrator.")