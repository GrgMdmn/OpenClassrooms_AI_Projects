FROM python:3.12-slim

WORKDIR /app

# Mettre à jour les packages et installer nginx
RUN apt-get update && \
    apt-get install -y nginx && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copier les fichiers de requirements et installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code de l'application
COPY . .

# Copier la configuration nginx
COPY nginx.conf /etc/nginx/nginx.conf

# Créer les répertoires nécessaires pour nginx
RUN mkdir -p /var/log/nginx && \
    mkdir -p /var/lib/nginx && \
    mkdir -p /run

# Donner les droits d'exécution au script de démarrage
RUN chmod +x /app/start.sh

# Exposer le port 8080
EXPOSE 8080

# Lancer le script de démarrage
CMD ["/app/start.sh"]
