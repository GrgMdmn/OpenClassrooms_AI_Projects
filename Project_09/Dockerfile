FROM python:3.12-slim

WORKDIR /app

# Install system dependencies (ajout de git pour segmentation_models)
RUN apt-get update && \
    apt-get install -y nginx curl gettext-base git && \
    # Pour OpenCV si nécessaire
    apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 && \
    adduser --system --no-create-home --group www-data || true && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy app/ d'abord pour avoir requirements.txt
COPY app/ ./app/

# Install requirements depuis le bon chemin
RUN pip install --no-cache-dir -r app/requirements.txt

# ✅ CORRECTION: Copy start.sh AVANT les permissions
COPY start.sh ./start.sh

# Copy du reste après installation (optimise les layers Docker)
COPY utils/ ./utils/
COPY cityscapes_config.json .
COPY notebooks/content/data/test_images_sample/ ./notebooks/content/data/test_images_sample/
COPY notebooks/content/data/train_images_sample/ ./notebooks/content/data/train_images_sample/

# Copy nginx config
COPY nginx.conf /etc/nginx/nginx.conf.template

# ✅ CORRECTION: Setup permissions APRÈS avoir copié start.sh + correction fins de ligne
RUN mkdir -p /var/log/nginx /var/lib/nginx /run && \
    sed -i 's/\r$//' ./start.sh && \
    chmod +x ./start.sh

# Environment variables
ENV NGINX_PORT=8080
ENV MULTISEG_API_BASE_URL=http://localhost:8080/api

EXPOSE $NGINX_PORT

# ✅ CORRECTION: Chemin cohérent (./start.sh au lieu de /app/start.sh)
CMD ["./start.sh"]
