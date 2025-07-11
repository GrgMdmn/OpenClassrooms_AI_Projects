FROM python:3.12-slim

WORKDIR /app

# # Install nginx and system packages
RUN apt-get update && \
    apt-get install -y nginx curl && \
    # Créer l'utilisateur et le groupe www-data si ce n'est pas déjà fait par l'image de base
    # Assurer que l'utilisateur nginx (souvent www-data) a un ID consistent
    adduser --system --no-create-home --group www-data || true && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install nltk resources at build time
RUN python3 -m nltk.downloader punkt \
    && python3 -m nltk.downloader punkt_tab \
    && python3 -m nltk.downloader stopwords \
    && python3 -m nltk.downloader wordnet \
    && python3 -m nltk.downloader omw-1.4


# Copy app code and configs
COPY . .

# Copy nginx config - l'emplacement par défaut de Nginx est /etc/nginx/nginx.conf
COPY nginx.conf /etc/nginx/nginx.conf

# Pas besoin de créer les dossiers ici, le start.sh s'en chargera avec les bonnes permissions
RUN mkdir -p /var/log/nginx /var/lib/nginx /run

# Ensure script is executable
RUN chmod +x /app/start.sh

# Expose only NGINX port (Cloud Run utilise 8080 par défaut)
EXPOSE 8080

# Start everything via script
CMD ["/app/start.sh"]
