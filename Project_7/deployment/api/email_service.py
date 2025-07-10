import smtplib
import os
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

def send_error_report_email(error_reports):
    """
    Envoie un email avec le rapport d'erreurs et le JSON intégré
    """
    smtp_server = os.getenv("SMTP_SERVER")
    smtp_port = int(os.getenv("SMTP_PORT", 587))
    smtp_email = os.getenv("SMTP_EMAIL") # main email address. If smt_from_alias is not, it will also be the mail sender
    smtp_password = os.getenv("SMTP_PASSWORD")
    smtp_from_alias = os.getenv("SMTP_FROM_ALIAS", smtp_email) # mail sender if not none (alias)
    admin_email = os.getenv("ADMIN_EMAIL") # this one will receive the mail

    
    if not all([smtp_server, smtp_email, smtp_password, admin_email]):
        print("❌ Configuration email manquante dans .env")
        return False
    
    try:
        # Créer le message
        msg = MIMEMultipart()
        if smtp_from_alias is not None:
            smtp_from = smtp_from_alias
        else:
            smtp_from = smtp_email
        msg['From'] = f"Air Paradis Monitor <{smtp_from}>"
        msg['To'] = admin_email
        msg['Subject'] = f"🚨 Rapport d'erreurs - Prédictions sentiments ({datetime.now().strftime('%d/%m/%Y %H:%M')})"
        
        # Préparer les données JSON
        json_data = {
            "rapport_genere_le": datetime.now().isoformat(),
            "nombre_signalements": len(error_reports),
            "signalements": [
                {
                    "id": i,
                    "tweet": tweet,
                    "prediction_incorrecte": prediction,
                    "longueur_tweet": len(tweet)
                }
                for i, (tweet, prediction) in enumerate(error_reports.items(), 1)
            ],
            "metadata": {
                "modele": "SentimentAnalysisLSTM",
                "version_api": "1.0",
                "seuil_rapport": 3
            }
        }
        
        # Corps du message avec JSON intégré
        body = f"""
Bonjour,

Un nouveau rapport d'erreurs a été généré pour l'API de prédiction de sentiments Air Paradis.

📊 Nombre total de signalements : {len(error_reports)}
📅 Date du rapport : {datetime.now().strftime('%d/%m/%Y à %H:%M')}

📋 Résumé des signalements :

"""
        
        for i, (tweet, prediction) in enumerate(error_reports.items(), 1):
            body += f"{i}. Tweet: \"{tweet[:100]}{'...' if len(tweet) > 100 else ''}\"\n"
            body += f"   Prédiction signalée comme incorrecte: {prediction.upper()}\n\n"
        
        body += """
📈 Actions recommandées :
- Analyser les tweets signalés pour identifier des patterns
- Considérer un réentraînement du modèle si nécessaire
- Vérifier la qualité des données de training

📊 DONNÉES COMPLÈTES (JSON) :
═══════════════════════════════════════════════════════════════════════════════

"""
        
        # ✅ Ajouter le JSON directement dans le corps
        json_string = json.dumps(json_data, ensure_ascii=False, indent=2)
        body += f"```json\n{json_string}\n```"
        
        body += """

═══════════════════════════════════════════════════════════════════════════════

---
Cordialement,
🤖 API Air Paradis - Système de monitoring automatique
📧 Envoyé depuis : """ + smtp_from
        
        # Ajouter le corps du message
        msg.attach(MIMEText(body, 'plain', 'utf-8'))
        
        # Envoyer l'email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_email, smtp_password)
            server.send_message(msg)
        
        print(f"✅ Email de rapport envoyé avec succès depuis {smtp_from}")
        print(f"📊 JSON intégré dans le corps du message")
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de l'envoi de l'email : {e}")
        return False