import os
import streamlit as st
import requests

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8080") + "/api"

st.title("Analyse de Sentiment - Air Paradis")


tweet = st.text_area("Entrez votre tweet ici :", height=150)

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
if st.button("Analyser le sentiment"):
    if not tweet.strip():
        st.warning("Veuillez entrer un tweet non vide.")
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
            st.error(f"Erreur lors de la requête vers l'API : {e}")
            reset_state()

# Affichage du résultat
if st.session_state.prediction_result:
    sentiment = st.session_state.prediction_result["sentiment"].capitalize()
    prob = st.session_state.prediction_result["probability"]
    st.success(f"Sentiment : {sentiment}")
    st.info(f"Probabilité que le tweet soit positif: {prob:.2f}")

# Gestion du bouton signalement
if st.session_state.show_report_button and not st.session_state.report_sent:
    if st.button("Signaler une mauvaise prédiction"):
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
                st.error(f"Erreur API : {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Erreur lors de la requête vers l'API : {e}")

# Message si signalement déjà fait
elif st.session_state.report_sent:
    st.success("✅ Merci pour votre signalement.")
    
    if st.session_state.last_report_sent:
        st.info("📩 Un rapport a été envoyé à l'administrateur du site.")