#!/bin/bash

set -e

# Set default port si non fourni
export NGINX_PORT=${NGINX_PORT:-8080}
export MULTISEG_API_BASE_URL=${MULTISEG_API_BASE_URL:-http://localhost:$NGINX_PORT/api}

echo "=== Multi-Segmentation API Docker Startup ==="
echo "NGINX_PORT: $NGINX_PORT"
echo "API_BASE_URL: $MULTISEG_API_BASE_URL"

# Substitute environment variables dans nginx config
envsubst '${NGINX_PORT}' < /etc/nginx/nginx.conf.template > /etc/nginx/nginx.conf

# S'assurer que les répertoires nginx sont accessibles
mkdir -p /var/cache/nginx /var/run/nginx /var/log/nginx
chown -R www-data:www-data /var/cache/nginx /var/run/nginx /var/log/nginx

# Créer le dossier uploads temporaire
mkdir -p /tmp/uploads

echo "Starting FastAPI server..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 --loop asyncio &

echo "Starting Streamlit app..."
streamlit run app/streamlit_app.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.enableXsrfProtection false \
  --server.enableCORS false \
  --server.headless true \
  &

# Vérifier que FastAPI est prêt
echo "Waiting for FastAPI to be ready..."
until curl -s http://127.0.0.1:8000/health > /dev/null; do
  echo "  FastAPI not ready yet, waiting..."
  sleep 2
done
echo "✅ FastAPI is ready!"

# Vérifier que Streamlit est prêt
echo "Waiting for Streamlit to be ready..."
until curl -s http://127.0.0.1:8501 > /dev/null; do
  echo "  Streamlit not ready yet, waiting..."
  sleep 2
done
echo "✅ Streamlit is ready!"

echo "🚀 Launching NGINX reverse proxy on port $NGINX_PORT..."
nginx -c /etc/nginx/nginx.conf -g 'daemon off;'
