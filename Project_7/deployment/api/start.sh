#!/bin/bash

# Lancer Streamlit en arrière-plan avec un baseUrlPath dédié
streamlit run app.py \
  --server.port 8501 \
  --server.address 127.0.0.1 \
  --server.baseUrlPath "/app" \
  --server.enableXsrfProtection false \
  &

# Lancer Uvicorn en arrière-plan
uvicorn main:app --host 127.0.0.1 --port 8000 --loop asyncio &

# Lancer nginx au premier plan (Docker suit ce processus)
nginx -g 'daemon off;'
