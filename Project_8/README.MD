# 🚗 Segmentation Multi-Classes pour Véhicules Autonomes

## 📋 Description du Projet

Ce projet développe un système de segmentation sémantique d'images de rue destiné aux véhicules autonomes. Il permet l'identification et la segmentation automatique de différents éléments urbains : véhicules, piétons, mobilier urbain, infrastructure routière, etc.

Le système s'inscrit dans une chaîne complète de vision par ordinateur embarquée pour l'automobile autonome, traitant les images en temps réel pour alimenter les systèmes de décision.

## 🗂️ Structure du Projet

```
├── notebooks/
│   ├── p8_notebook.ipynb       # Notebook principal d'entraînement des modèles
│   └── content/
│       └── data/
│           ├── new_images_to_predict/     # Images diverses pour tests d'inférence
│           └── test_images_sample/        # Échantillon Cityscapes avec masques
├── utils/
│   └── utils.py                # Fonctions communes (prétraitement, chargement, inférence)
├── app/                        # API et Interface Web
│   ├── main.py                 # API FastAPI avec routes de segmentation
│   ├── streamlit_app.py        # Interface web interactive
│   └── requirements.txt        # Dépendances pour l'API et l'interface
├── cityscapes_config.json      # Configuration des classes Cityscapes
├── .env.example                # Template de variables d'environnement
├── Dockerfile                  # Configuration Docker
├── docker-compose.yml.example  # Template Docker Compose
├── nginx.conf                  # Configuration reverse proxy
├── start.sh                    # Script de démarrage des services
├── .dockerignore              # Optimisation du build Docker
└── README.md
```

> **Note** : Les modèles entraînés sont stockés et versionnés via **MLFlow** avec artefacts sur **MinIO**, pas en local.

**État actuel du développement :**
- ✅ **Entraînement des modèles** : Complet (notebook + MLFlow + MinIO)
- ✅ **Fonctions d'inférence** : Développées et prêtes
- ✅ **API FastAPI** : API complète avec endpoints de segmentation
- ✅ **Interface Web** : Interface Streamlit pour upload et visualisation
- ✅ **Déploiement** : Conteneurisation Docker complète + reverse proxy nginx

## 🏷️ Classes de Segmentation

Le modèle identifie **8 catégories principales** :

1. **Route** - Chaussée et surfaces roulantes
2. **Trottoir** - Zones piétonnes
3. **Bâtiments** - Structures architecturales
4. **Végétation** - Arbres, buissons, espaces verts
5. **Véhicules** - Voitures, camions, bus
6. **Personnes** - Piétons, cyclistes
7. **Signalisation** - Panneaux, feux de circulation
8. **Mobilier urbain** - Poteaux, barrières, abribus

## 🧠 Architecture Technique

### Modèles Implémentés
Utilisation de la bibliothèque **[segmentation_models](https://github.com/qubvel/segmentation_models)** de qubvel pour l'implémentation des architectures.

**Architectures principales :**
- **U-Net** : Architecture encoder-decoder classique
- **FPN** (Feature Pyramid Network) : Réseau à pyramide de caractéristiques

**Backbones testés :**
```python
# Modèles avec backbones par défaut
('U-Net', None)     # Backbone U-Net intégré
('FPN', None)       # Backbone VGG16 par défaut

# Modèles avec backbones pré-entraînés (encodeurs gelés)
('U-Net', 'mobilenetv2')
('U-Net', 'efficientnetb0') 
('FPN', 'mobilenetv2')
('FPN', 'efficientnetb0')
('FPN', 'resnet34')

# Fine-tuning sélectif (encodeurs entraînables)
('U-Net', 'mobilenetv2')
('FPN', 'efficientnetb0') 
('FPN', 'resnet34')
```

**Stratégies d'entraînement :**
- **Encodeurs gelés** : Transfer learning avec features pré-entraînées figées
- **Fine-tuning** : Ajustement fin des encodeurs pré-entraînés
- **Entraînement from scratch** : Sans pré-entraînement (backbones par défaut)

### Stack Technologique
- **Framework** : Keras/TensorFlow
- **Architectures** : [segmentation_models](https://github.com/qubvel/segmentation_models) (qubvel)
- **Gestion des expériences** : MLFlow
- **Stockage des artefacts** : MinIO (S3-compatible)
- **API** : FastAPI avec validation Pydantic
- **Interface** : Streamlit pour l'interaction utilisateur
- **Reverse Proxy** : nginx pour routage API/Interface
- **Déploiement** : Docker + NAS personnel *(solutions souveraines)*

## 📊 Dataset

- **Source** : Cityscapes Dataset
- **Organisation locale** :
  - `notebooks/content/data/test_images_sample/` : Échantillon avec masques de vérité pour tests d'inférence
  - `notebooks/content/data/new_images_to_predict/` : Images de test diverses pour validation

## 🚀 Installation et Configuration

### Option 1: Déploiement Docker (Recommandé)

#### Configuration rapide
```bash
# Cloner le projet
git clone https://github.com/GrgMdmn/OpenClassrooms_AI_Projects.git
cd OpenClassrooms_AI_Projects/Project_8

# Copier et adapter la configuration
cp .env.example .env
cp docker-compose.yml.example docker-compose.yml

# Éditer les variables d'environnement
nano .env
```

#### Variables d'environnement requises
```bash
# Configuration du serveur
NGINX_PORT=8080

# Configuration MLflow
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key

# URL de base de l'API (pour Streamlit)
MULTISEG_API_BASE_URL=http://localhost:8080/api
```

#### Déploiement
```bash
# Construction et lancement
docker-compose up --build

# Ou en arrière-plan
docker-compose up -d --build

# Accès à l'application
# Interface web: http://localhost:8080
# API docs: http://localhost:8080/api/docs
# Health check: http://localhost:8080/health
```

#### Test local sans docker-compose
```bash
# Lancement avec port personnalisé
docker build -t multiseg-api .
docker run -p 3000:3000 \
  -e NGINX_PORT=3000 \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
  -e MULTISEG_API_BASE_URL=http://localhost:3000/api \
  multiseg-api
```

### Option 2: Installation Locale (Développement)

#### Pour l'expérimentation et l'entraînement
```bash
cd notebooks
pip install -r requirements.txt
```

#### Pour l'API uniquement
```bash
cd app
pip install -r requirements.txt

# Lancement manuel des services
# Terminal 1: FastAPI
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Terminal 2: Streamlit  
streamlit run app/streamlit_app.py --server.port 8501
```

## 🌐 Utilisation de l'API

### Endpoints Principaux

#### Segmentation d'image
```bash
# Upload et segmentation via curl
curl -X POST "http://localhost:8080/api/segment" \
  -F "file=@image.jpg" \
  -F "model_name=unet_mobilenetv2" \
  -F "confidence_threshold=0.5"
```

#### Informations sur les modèles
```bash
# Liste des modèles disponibles
curl "http://localhost:8080/api/models"

# Détails d'un modèle
curl "http://localhost:8080/api/models/unet_mobilenetv2"
```

#### Monitoring
```bash
# Health check
curl "http://localhost:8080/health"

# Métriques API
curl "http://localhost:8080/api/metrics"
```

### Interface Web

L'interface Streamlit accessible sur `http://localhost:8080` permet :
- **Upload d'images** par glisser-déposer ou sélection
- **Sélection du modèle** parmi les modèles disponibles
- **Ajustement des paramètres** (seuil de confiance, etc.)
- **Visualisation** des résultats avec overlay des segmentations
- **Téléchargement** des masques de segmentation

## 🔧 Architecture de Déploiement

### Structure des Services
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   nginx:8080    │────│  FastAPI:8000    │────│   MLFlow/MinIO  │
│  (Reverse Proxy)│    │  (API Backend)   │    │ (Model Storage) │
└─────────┬───────┘    └──────────────────┘    └─────────────────┘
          │
┌─────────▼───────┐
│  Streamlit:8501 │
│ (Web Interface) │
└─────────────────┘
```

### Volumes et Persistance
- **Modèles** : Téléchargés depuis MLFlow/MinIO au démarrage
- **Uploads temporaires** : Stockage en mémoire (pas de persistance)
- **Logs** : Accessibles via `docker-compose logs`

### Monitoring et Logs
```bash
# Logs en temps réel
docker-compose logs -f

# Logs d'un service spécifique
docker-compose logs -f multiseg-api

# Métriques de performance
curl http://localhost:8080/api/metrics
```

## 🔮 Améliorations Futures

### Modèles et Performance
- **Prétraitement adaptatif** : Prétraitement spécifique à chaque backbone au lieu de la normalisation générique
- **Optimisation temps réel** : Amélioration des performances d'inférence pour déploiement embarqué
- **Résolution d'entraînement améliorée** : L'API accepte toutes les résolutions d'images, mais l'entraînement a été effectué sur des images redimensionnées à 224x224. Un réentraînement sur des résolutions supérieures (512x512, 1024x1024) pourrait améliorer la précision en capturant plus de détails fins, au prix d'une augmentation des besoins en puissance de calcul
- **Transfer learning ciblé** : Utilisation de backbones pré-entraînés sur Mapillary Vistas

### Infrastructure et Déploiement
- **Scaling horizontal** : Support Kubernetes pour déploiement multi-instances
- **Cache intelligent** : Mise en cache des prédictions pour images similaires
- **API Gateway** : Intégration d'un gateway pour authentification et rate limiting
- **Monitoring avancé** : Intégration Prometheus/Grafana pour métriques détaillées

### Données et Généralisation
- **Généralisation des conditions** : Extension au-delà des conditions urbaines de Cityscapes
- **Segmentation des personnes** : Amélioration de l'IoU via data augmentation (IA générative)
- **Datasets additionnels** : Intégration de datasets complémentaires pour robustesse

---

*Projet développé dans une approche de souveraineté technologique, privilégiant les solutions open-source et l'hébergement sur infrastructure personnelle.*
