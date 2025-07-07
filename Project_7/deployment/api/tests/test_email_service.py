# Utilisation de mock pour simuler l'envoi de mails mais sans spammer une boîte.

from unittest.mock import patch, MagicMock
import os
import sys
from dotenv import load_dotenv

# Ajouter le répertoire parent au path pour importer email_service
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from email_service import send_error_report_email

# Charger les variables d'environnement
dotenv_path = '../../../.env'
load_dotenv(dotenv_path)

@patch("smtplib.SMTP")
def test_email_service_connection(mock_smtp):
    """
    Teste la connexion SMTP sans envoyer réellement l'email
    """
    # 🔧 Configuration du mock
    mock_server = MagicMock()
    mock_smtp.return_value.__enter__.return_value = mock_server

    test_reports = {
        "Vol annulé sans prévenir": "positif"
    }

    # Appel de la fonction
    result = send_error_report_email(test_reports)

    # ✅ Assertions
    assert result is True
    mock_smtp.assert_called_with(os.getenv("SMTP_SERVER"), int(os.getenv("SMTP_PORT", 587)))
    mock_server.starttls.assert_called_once()
    mock_server.login.assert_called_once_with(os.getenv("SMTP_EMAIL"), os.getenv("SMTP_PASSWORD"))
    mock_server.send_message.assert_called_once()
