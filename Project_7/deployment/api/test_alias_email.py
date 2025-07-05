#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  5 15:03:24 2025

@author: madfuckinman
"""

# test_alias_email.py
import os
from dotenv import load_dotenv
from email_service import send_error_report_email

load_dotenv()

print("🔧 Configuration Email:")
print(f"Serveur SMTP: {os.getenv('SMTP_SERVER')}")
print(f"Authentification: {os.getenv('SMTP_EMAIL')}")
print(f"Expéditeur (alias): {os.getenv('SMTP_FROM_ALIAS')}")
print(f"Destinataire: {os.getenv('ADMIN_EMAIL')}")

# Test
test_reports = {
    "Service client décevant": "positif",
    "J'adore cette compagnie": "negatif",
    "Vol annulé sans prévenir": "positif"
}

print("\n📧 Test d'envoi depuis l'alias...")
result = send_error_report_email(test_reports)
print(f"Résultat : {'✅ Succès' if result else '❌ Échec'}")