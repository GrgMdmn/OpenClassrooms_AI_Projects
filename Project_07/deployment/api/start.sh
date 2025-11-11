#!/bin/bash

set -e

# S'assurer que les répertoires de cache et de log Nginx sont accessibles en écriture
mkdir -p /var/cache/nginx /var/run/nginx /var/log/nginx
chown -R www-data:www-data /var/cache/nginx /var/run/nginx /var/log/nginx

echo "Starting Streamlit..."
streamlit run app.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.enableXsrfProtection false \
  --server.enableCORS false \
  & # Lance en arrière-plan

echo "Starting FastAPI..."
uvicorn main:app --host 0.0.0.0 --port 8000 --loop asyncio & # Lance en arrière-plan

# MODIFICATION IMPORTANTE ICI: Vérifier Streamlit sur le bon chemin
echo "Waiting for Streamlit to be ready..."
until curl -s http://127.0.0.1:8501 > /dev/null; do
  sleep 1
done
echo "Streamlit is ready!"

# MODIFICATION: Vérifier FastAPI sur un endpoint existant, par exemple /docs ou /health
echo "Waiting for FastAPI to be ready..."
until curl -s http://127.0.0.1:8000/docs > /dev/null; do # Assurez-vous que /docs est un endpoint léger
  sleep 1
done
echo "FastAPI is ready!"

echo "Launching NGINX..."
nginx -c /etc/nginx/nginx.conf -g 'daemon off;'
