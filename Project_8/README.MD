# 🚗 Segmentation Multi-Classes pour Véhicules Autonomes

## 📋 Description du Projet

Ce projet développe un système de segmentation sémantique d'images de rue destiné aux véhicules autonomes. Il permet l'identification et la segmentation automatique de différents éléments urbains : véhicules, piétons, mobilier urbain, infrastructure routière, etc.

Le système s'inscrit dans une chaîne complète de vision par ordinateur embarquée pour l'automobile autonome, traitant les images en temps réel pour alimenter les systèmes de décision.

## 🗂️ Structure du Projet

```
├── notebooks/
│   ├── p8_notebook.py          # Notebook principal d'entraînement des modèles
│   ├── requirements.txt        # Dépendances pour l'entraînement et expérimentation
│   └── content/
│       └── data/
│           ├── new_images_to_predict/     # Images diverses pour tests d'inférence
│           └── test_images_sample/        # Échantillon Cityscapes avec masques
├── utils/
│   └── utils.py                # Fonctions communes (prétraitement, chargement, inférence)
├── api/                        # [En développement] API FastAPI
│   └── requirements.txt        # Dépendances légères pour l'API uniquement
├── .env                        # Variables d'environnement (MLFlow, MinIO)
└── README.md
```

> **Note** : Les modèles entraînés sont stockés et versionnés via **MLFlow** avec artefacts sur **MinIO**, pas en local.

**État actuel du développement :**
- ✅ **Entraînement des modèles** : Complet (notebook + MLFlow + MinIO)
- ✅ **Fonctions d'inférence** : Développées et prêtes
- 🔄 **API FastAPI** : En cours de développement
- 🔄 **Déploiement** : Conteneurisation Docker + NAS personnel

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
- **API** : FastAPI *(en développement)*
- **Déploiement** : Docker + NAS personnel *(solutions souveraines)*

## 📊 Dataset

- **Source** : Cityscapes Dataset
- **Organisation locale** :
  - `notebooks/content/data/test_images_sample/` : Échantillon avec masques de vérité pour tests d'inférence
  - `notebooks/content/data/new_images_to_predict/` : Images de test diverses pour validation

## 🚀 Installation et Configuration

### Installation
```bash
git clone https://github.com/GrgMdmn/OpenClassrooms_AI_Projects.git
cd OpenClassrooms_AI_Projects/Project_8

# Pour l'expérimentation et l'entraînement
cd notebooks
pip install -r requirements.txt

# Pour l'API uniquement (plus léger)
cd api  
pip install -r requirements.txt
```

### Configuration MLFlow et MinIO
Créer un fichier `.env` à la racine du projet :
```bash
MLFLOW_TRACKING_URI=http://localhost:5000 # mlflow url
MLFLOW_S3_ENDPOINT_URL=http://localhost:9000 # minio url
AWS_ACCESS_KEY_ID=your_access_key # minio login
AWS_SECRET_ACCESS_KEY=your_secret_key # minio password
```

Le chargement des variables se fait via `python-dotenv` dans le notebook et l'API.

## 🌐 API et Déploiement

### API FastAPI (En développement)
L'API utilisera les fonctions de `utils.py` et sera déployée via Docker sur NAS personnel pour maintenir la souveraineté des données.

## 🔮 Améliorations Futures

- **Prétraitement adaptatif** : Prétraitement spécifique à chaque backbone au lieu de la normalisation générique (division par 255)
- **Généralisation des conditions** : Extension au-delà des conditions urbaines de Cityscapes (météo, environnements variés)
- **Optimisation temps réel** : Évaluation et amélioration des performances d'inférence
- **Résolution variable** : Support de tailles d'images variables au-delà du format fixe 224x224
- **Transfer learning ciblé** : Utilisation de backbones pré-entraînés sur Mapillary Vistas plutôt qu'ImageNet pour un meilleur alignement domaine
- **Segmentation des personnes** : Amélioration de l'IoU sur les humains via data augmentation (IA générative)
- **Limitation du dataset** : Les performances similaires entre modèles suggèrent une limite liée à la qualité/diversité des données d'entraînement

---

*Projet développé dans une approche de souveraineté technologique, privilégiant les solutions open-source et l'hébergement sur infrastructure personnelle.*