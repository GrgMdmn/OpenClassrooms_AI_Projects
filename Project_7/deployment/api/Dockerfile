# Utiliser une image légère Python 3.12
FROM python:3.12-slim

# Définir le dossier de travail
WORKDIR /app

# Copier uniquement requirements.txt d'abord (pour optimiser le cache Docker)
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste du code source
COPY . .

# Exposer les ports 8000 (fastAPI) et 8501 (streamlit)
EXPOSE 8000 8501

RUN chmod +x /app/start.sh

CMD ["/app/start.sh"]
