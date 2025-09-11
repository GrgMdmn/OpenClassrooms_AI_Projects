# Système de Recommandation de Contenu - My Content

Projet de développement d'un système de recommandation d'articles basé sur une architecture serverless Azure Functions avec interface utilisateur Streamlit.

## Vue d'ensemble

Ce projet implémente un système de recommandation de contenu utilisant des algorithmes de filtrage collaboratif et basé sur le contenu. L'architecture serverless permet une mise à l'échelle automatique et un coût optimisé.

### Fonctionnalités principales

- **API de recommandation** : Azure Function retournant le top 5 des articles recommandés pour un utilisateur
- **Interface utilisateur** : Application Streamlit pour tester et démontrer le système
- **Algorithmes implémentés** : Content-Based Filtering avec embeddings pré-calculés
- **Architecture serverless** : Déploiement sur Azure Functions avec plan Consumption

## Architecture

### MVP Actuel
```
[Interface Streamlit] → [Azure Functions API] → [Algorithme de recommandation]
                                ↓
                        [Données embeddings + historique utilisateur]
```

### Composants
- **Frontend** : Streamlit (interface de test)
- **Backend** : Azure Functions (Python 3.12, Linux)
- **Algorithme** : Content-Based Filtering avec similarité cosinus
- **Données** : Embeddings articles (50D via PCA vs 250D embeddings originaux) + historique des clics

## Structure du Projet

```
Project_10/
├── azure-app/                  # Azure Function
│   ├── function_app.py         # Point d'entrée Azure Function
│   ├── requirements.txt        # Dépendances Python
│   ├── host.json              # Configuration Azure Functions
│   ├── local.settings.json    # Configuration locale (non versionnée)
│   ├── data/
│   │   ├── embeddings_azure_50d.pkl    # Embeddings articles (PCA 50D)
│   │   └── clicks_sample.csv           # Historique des clics utilisateurs
│   └── utils/
│       ├── data_loader.py      # Chargement et cache des données
│       └── content_based.py    # Algorithme de recommandation
├── streamlit_app.py           # Interface utilisateur
├── streamlit_requirements.txt # Dépendances Streamlit
└── README.md
```

## Installation et Déploiement

### Prérequis
- Python 3.12+
- Azure CLI
- Azure Functions Core Tools
- Compte Azure

### Déploiement Azure Functions

1. **Connexion à Azure**
```bash
az login
```

2. **Création de l'infrastructure**
```bash
# Groupe de ressources
az group create --name rg-content-recommender --location francecentral

# Compte de stockage
az storage account create --name [votre-nom-stockage-unique] \
  --resource-group rg-content-recommender \
  --location francecentral \
  --sku Standard_LRS

# Function App Linux avec Python 3.12
az functionapp create \
  --resource-group rg-content-recommender \
  --consumption-plan-location francecentral \
  --runtime python \
  --runtime-version 3.12 \
  --functions-version 4 \
  --name contentrecommender-linux \
  --storage-account stcontentrecommend2025 \
  --os-type Linux
```

3. **Déploiement du code**
```bash
# Depuis le dossier azure-app
func azure functionapp publish contentrecommender-linux
```

### Interface Streamlit

1. **Installation des dépendances**
```bash
pip install -r streamlit_requirements.txt
```

2. **Lancement de l'interface**
```bash
streamlit run streamlit_app.py
```

## Utilisation

### API Azure Functions

**Endpoint** : `https://contentrecommender-linux.azurewebsites.net/api/recommend`

**Paramètres** :
- `user_id` (int) : Identifiant de l'utilisateur (0-706)

**Exemple d'appel** :
```bash
curl "https://contentrecommender-linux.azurewebsites.net/api/recommend?user_id=0"
```

**Réponse** :
```json
{
  "user_id": 0,
  "recommendations": [157519, 157944, 159495, 156690, 162857]
}
```

### Interface Streamlit

1. Configurer l'URL de l'API (local ou Azure)
2. Sélectionner un ID utilisateur
3. Obtenir les recommandations
4. Tester les fonctionnalités avancées

## Algorithme de Recommandation

### Content-Based Filtering

1. **Chargement des données** : Embeddings pré-calculés et historique utilisateur
2. **Calcul du profil utilisateur** : Moyenne des embeddings des articles consultés
3. **Similarité cosinus** : Calcul avec tous les articles du catalogue
4. **Filtrage** : Exclusion des articles déjà vus
5. **Sélection** : Top 5 des articles les plus similaires

### Optimisations

- **Réduction dimensionnelle** : PCA 250D → 50D pour optimiser les performances Azure
- **Cache en mémoire** : Embeddings et données utilisateur chargés une seule fois
- **Architecture serverless** : Mise à l'échelle automatique selon la demande

## Tests et Validation

### Tests locaux
```bash
# Démarrage Azure Functions local
func start

# Test de l'algorithme
python test_local.py
```

### Métriques de performance
- **Temps de réponse** : ~200ms par recommandation
- **Utilisateurs supportés** : 707 dans le dataset actuel
- **Articles disponibles** : 364,047 avec embeddings

## Problèmes Connus et Solutions

### Déploiement Azure
- **Python sur Windows** : Non supporté, utiliser Linux obligatoirement
- **Authentification** : Configurer `http_auth_level=func.AuthLevel.ANONYMOUS`
- **Erreurs "Service unavailable"** : Problèmes temporaires Azure, réessayer plus tard

### Limitations actuelles
- **Cold start** : Première requête lente après période d'inactivité
- **Nouveaux utilisateurs** : Pas de gestion du problème de démarrage à froid
- **Mise à jour des données** : Processus manuel

## Architecture Cible (Future)

### Améliorations prévues
- **Gestion cold start** : Recommandations populaires pour nouveaux utilisateurs
- **Pipeline de données** : Mise à jour automatique des embeddings
- **Base de données** : Stockage persistant des profils utilisateur
- **API hybride** : Combinaison Content-Based + Collaborative Filtering
- **Cache distribué** : Précalcul des recommandations populaires

## Technologies Utilisées

- **Backend** : Azure Functions, Python 3.12
- **Frontend** : Streamlit
- **ML/Data** : scikit-learn, pandas, numpy
- **Infrastructure** : Azure CLI, Azure Storage
- **Déploiement** : Azure Functions Core Tools

## Auteur

Projet réalisé dans le cadre d'une formation OpenClassrooms - Ingénieur IA.

## Licence

Ce projet est à des fins éducatives uniquement.