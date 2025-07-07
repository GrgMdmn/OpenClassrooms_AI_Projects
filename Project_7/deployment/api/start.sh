#!/bin/bash

# Lancer Streamlit en arrière-plan
streamlit run app.py --server.port 8501 --server.address 0.0.0.0 &

# Lancer Uvicorn en premier plan (Docker "suit" ce processus)
uvicorn main:app --host 0.0.0.0 --port 8000
