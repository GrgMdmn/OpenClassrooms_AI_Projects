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
        # Create message
        msg = MIMEMultipart()
        if smtp_from_alias is not None:
            smtp_from = smtp_from_alias
        else:
            smtp_from = smtp_email
        msg['From'] = f"Air Paradis Monitor <{smtp_from}>"
        msg['To'] = admin_email
        msg['Subject'] = f"🚨 Rapport d'erreurs - Prédictions sentiments ({datetime.now().strftime('%d/%m/%Y %H:%M')})"
        
        # Prepare JSON data
        json_data = {
            "report_generated_on": datetime.now().isoformat(),
            "reports_amount": len(error_reports),
            "reports": [
                {
                    "id": i,
                    "tweet": tweet,
                    "incorrect_prediction": prediction,
                    "tweet_length": len(tweet)
                }
                for i, (tweet, prediction) in enumerate(error_reports.items(), 1)
            ],
            "metadata": {
                "model": "SentimentAnalysisLSTM",
                "api_version": "1.0",
                "reporting_threshold": 3
            }
        }
        
        # Message body with embedded JSON
        body = f"""
Hello,

A new error report has been generated for the Air Paradis sentiment prediction API.

📊 Total number of reports : {len(error_reports)}
📅 Report date : {datetime.now().strftime('%d/%m/%Y à %H:%M')}

📋 Report summary :

"""
        
        for i, (tweet, prediction) in enumerate(error_reports.items(), 1):
            body += f"{i}. Tweet: \"{tweet[:100]}{'...' if len(tweet) > 100 else ''}\"\n"
            body += f"   Prediction reported as incorrect: {prediction.upper()}\n\n"
        
        body += """
📈 Recommended actions:
- Analyze reported tweets to identify patterns
- Consider re-training the model if necessary
- Check quality of training data

📊 COMPLETE DATA (JSON) :
═══════════════════════════════════════════════════════════════════════════════

"""
        
        # ✅ Directly add JSON inside the body
        json_string = json.dumps(json_data, ensure_ascii=False, indent=2)
        body += f"```json\n{json_string}\n```"
        
        body += """

═══════════════════════════════════════════════════════════════════════════════

---
Sincerely,
🤖 API Air Paradis - Automatic monitoring system
📧 Sent from : """ + smtp_from
        
        # Add message body
        msg.attach(MIMEText(body, 'plain', 'utf-8'))
        
        # Sending email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_email, smtp_password)
            server.send_message(msg)
        
        print(f"✅ Report email successfully sent from {smtp_from}")
        print("📊 JSON embedded in message body")
        return True
        
    except Exception as e:
        print(f"❌ Error during email sending : {e}")
        return False