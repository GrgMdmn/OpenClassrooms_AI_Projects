#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  5 15:03:24 2025

@author: madfuckinman
"""

# # test_alias_email.py
# import os
# from dotenv import load_dotenv
# from email_service import send_error_report_email

# load_dotenv()

# print("🔧 Configuration Email:")
# print(f"Serveur SMTP: {os.getenv('SMTP_SERVER')}")
# print(f"Authentification: {os.getenv('SMTP_EMAIL')}")
# print(f"Expéditeur (alias): {os.getenv('SMTP_FROM_ALIAS')}")
# print(f"Destinataire: {os.getenv('ADMIN_EMAIL')}")

# # Test
# test_reports = {
#     "Service client décevant": "positif",
#     "J'adore cette compagnie": "negatif",
#     "Vol annulé sans prévenir": "positif"
# }

# print("\n📧 Test d'envoi depuis l'alias...")
# result = send_error_report_email(test_reports)
# print(f"Résultat : {'✅ Succès' if result else '❌ Échec'}")


import os
import sys
from dotenv import load_dotenv

# Ajouter le répertoire parent au path pour importer email_service
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from email_service import send_error_report_email

# Charger le fichier .env situé un niveau au-dessus
dotenv_path = '../../../.env'
load_dotenv(dotenv_path)

def test_email_service_with_alias():
    """
    Teste l'envoi d'un email avec la fonction send_error_report_email
    """
    # ⚙️ Vérifier si les variables d'environnement nécessaires sont présentes
    smtp_server = os.getenv("SMTP_SERVER")
    smtp_email = os.getenv("SMTP_EMAIL")
    admin_email = os.getenv("ADMIN_EMAIL")
    
    assert smtp_server, "SMTP_SERVER non défini dans le fichier .env"
    assert smtp_email, "SMTP_EMAIL non défini dans le fichier .env"
    assert admin_email, "ADMIN_EMAIL non défini dans le fichier .env"

    # Configuration de test
    print("🔧 Configuration Email :")
    print(f"Serveur SMTP : {smtp_server}")
    print(f"Authentification : {smtp_email}")
    print(f"Destinataire : {admin_email}")

    # Cas de test
    test_reports = {
        "Service client décevant": "positif",
        "Erreur à l'embarquement": "negatif",
        "Vol annulé sans prévenir": "positif"
    }

    # 📧 Tester la fonction d'envoi
    result = send_error_report_email(test_reports)

    # ✅ Assertion pour vérifier que la fonction fonctionne correctement
    assert result is True, "L'email n'a pas été envoyé avec succès"