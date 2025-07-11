# ✨ Analyse de sentiments et déploiement MLOps : retour d’expérience complet

Dans le cadre du projet *Air Paradis*, nous avons mis en œuvre une chaîne complète d’analyse de sentiments à partir de tweets, depuis la modélisation jusqu’au déploiement MLOps. Ce billet de blog revient sur l’ensemble des étapes réalisées, les choix méthodologiques, la comparaison des modèles et la mise en production d’une API. 🧠🔧

---

## 🔍 Trois approches de modélisation supervisée

Nous avons testé trois approches complémentaires pour prédire le sentiment d’un tweet. Chacune s’appuie sur un type de représentation des textes et un algorithme différent :

### 1. 🧱 Approche simple : Bag-of-Words (BoW) + modèles de machine learning

Cette approche utilise des vecteurs BoW ou BoW + TF-IDF, qui capturent la fréquence des mots dans les tweets. On fait le choix d’éviter des techniques non supervisées comme **LDA**, car elles sont plus adaptées à des tâches de topic modeling et non de classification de sentiments (binaire qui plus est).

➡️ **Modèles testés :**  
- Régression Logistique  
- Naive Bayes  
- SVM  
- Random Forest  

Ces modèles ont été entraînés avec **GridSearchCV**, en optimisant la **précision (`precision` et non `accuracy`)**, notre KPI principal.

📸 *[Insérer ici une capture d’écran MLFlow avec les scores des modèles simples]*

---

### 2. 🔥 Approche avancée : Word Embeddings (Word2Vec & GloVe 300d) + LSTM

On représente les mots non plus comme des fréquences mais comme des vecteurs denses, porteurs de signification.  
Nous avons utilisé :
- **Word2Vec**, entraîné sur un grand corpus (skip-gram),
- **GloVe 300d**, embeddings précalculés par Stanford.

Ces représentations sont **statistiquement proches** entre mots au sens sémantique.  
Elles **ne tiennent cependant pas compte du contexte exact dans la phrase** :  
> « banque » aura le même vecteur que ce soit pour une banque d'argent ou une banque de données.

La classification se fait via un **LSTM** (Long Short-Term Memory), une architecture de **deep learning supervisée** adaptée aux séquences de mots.

🧠 **Le modèle retenu pour le déploiement est sélectionné automatiquement dans MLflow parmi les deux (Word2Vec ou GloVe), en fonction de celui qui obtient la précision maximale.**  
Cela permet d’avoir un compromis optimal entre légèreté (CPU-friendly) et performance.

📸 *[Insérer ici une capture d’écran du graphe de précision sur validation pour GloVe+LSTM]*

---

### 3. 🤖 BERT : Bidirectional Encoder Representations from Transformers

BERT est un modèle de langage avancé, préentraîné par Google, capable de **comprendre le contexte bidirectionnel** d’un mot dans une phrase. Contrairement aux word embeddings classiques, BERT :
- **encode dynamiquement les mots** selon leur contexte d’usage (ex. « banque » n’aura pas le même vecteur selon la phrase),
- est entraîné par **masked language modeling** et **next sentence prediction**, puis fine-tuné pour la classification supervisée.

Cependant, il a ses **limites** :
> ⚠️ BERT **ne comprend pas l’ironie, le sarcasme ou les références culturelles implicites**.  
> Il reste un modèle basé sur des cooccurrences statistiques, sans connaissance du monde ou intentionnalité.  
> Or, **Twitter est un terrain propice au second degré**, aux détournements ou aux trolls : ce type de contenu reste un **véritable défi pour les modèles classiques de NLP.**

📸 *[Insérer ici un graphique montrant la précision de BERT sur le jeu de validation]*

---

## 📐 Choix du KPI : pourquoi la précision (precision) ?

Nous avons fait le choix de nous focaliser sur la **précision** plutôt que l’**accuracy** :

> 🎯 **Objectif métier :** éviter de **classer un tweet comme positif alors qu’il est négatif**, car cela pourrait induire une mauvaise communication ou une mauvaise interprétation dans un contexte sensible.

### Pour chaque type de modèle :
- Modèles classiques → optimisation via **GridSearchCV sur la précision**  
- Modèles DL → **EarlyStopping sur val_loss**, mais sélection du **meilleur modèle selon la précision** sur l’ensemble de validation.

---

## ⚙️ Mise en place de l’environnement MLOps maison

Nous avons choisi une **infrastructure MLOps souveraine et autonome** pour orchestrer les expérimentations.

### 🏠 Serveur personnel (NAS) + OpenMediaVault
Nous avons déployé MLFlow sur un NAS personnel tournant sous **OpenMediaVault** (basé sur Debian).

### 📦 Artefacts dans MinIO
Les modèles, métriques, visualisations et paramètres sont stockés dans **MinIO**, un système compatible S3.  
➡️ MinIO est open-source, léger, et permet une **gestion souveraine des artefacts**, contrairement à AWS S3.

> Les notebooks (en particulier ceux entraînés sur Google Colab, pour bénéficier de GPU) envoient automatiquement leurs logs vers MLFlow et leurs artefacts vers MinIO, via des **variables d’environnement** pour sécuriser les identifiants.

📸 *[Insérer ici une capture du dashboard MLFlow]*

---

## 🧪 Intégration continue, tests, versioning

Une fois les modèles validés, nous avons implémenté toute la chaîne MLOps pour le déploiement API :

### ✅ Tests automatisés
- **Tests unitaires** pour valider le prétraitement des tweets, les prédictions et la gestion des erreurs.
- Ces tests sont intégrés dans **GitHub Actions**, et **empêchent tout déploiement Docker si les tests échouent**.

### 🐳 Conteneurisation & CI/CD
- L’API est packagée via **Docker**, puis poussée automatiquement sur **DockerHub** en cas de succès.
- Cela permet un déploiement cohérent et rapide sur n’importe quelle plateforme.

---

## 🚀 Déploiement de l’API FastAPI

### 🛠️ Backend : FastAPI
- Fournit un endpoint `/predict`, avec chargement dynamique du modèle via MLFlow et récupération des embeddings (BoW, GloVe ou Word2Vec) selon le tag `embedding_type`.

🧠 Le modèle de type Word Embeddings + LSTM avec la **meilleure précision** est automatiquement récupéré depuis le *Model Registry* MLFlow.

### 💻 Frontend local : Streamlit
- Sert d’**interface utilisateur** pour tester les prédictions en local, simuler des requêtes API et visualiser les résultats.  
- C’est aussi un outil utile pour les démonstrations ou le debug.

📸 *[Insérer capture Streamlit en action]*

---

### 🌐 Problème matériel sur le NAS
Le NAS utilise un processeur **Intel N100**, théoriquement compatible AVX2.  
> ❗ Malheureusement, **certaines cartes mères désactivent cette instruction** au niveau BIOS, ce qui empêche l’exécution de TensorFlow, nécessaire pour le LSTM et BERT.

🛠️ Solution : **déploiement sur Google Cloud Run**, à l’adresse suivante :  
🔗 [https://sentiment-api-service-7772256003.europe-west1.run.app/](https://sentiment-api-service-7772256003.europe-west1.run.app/)

📸 *[Insérer capture console Google Cloud / endpoint]*

---

## 📊 Monitoring maison en production

L'API embarque une **logique d’alerte interne** en cas de dérive.

🎯 **Comportement :**
- Si plus de **3 prédictions incorrectes** sont identifiées dans une **fenêtre de 5 minutes**, un **rapport d’erreur** est généré automatiquement.
- Ces erreurs peuvent être loggées ou transmises via webhook selon la configuration future.

Cela permet de garder un **œil léger mais réactif** sur les dérives en production, sans surcharger l’infrastructure.

---

## 🧠 Schéma global du pipeline MLOps

      ┌────────────┐
      │  Données   │
      │ (tweets)   │
      └────┬───────┘
           │
    Prétraitement (regex, tokenisation, lemmatisation, stopwords)
           │
 ┌─────────▼──────────┐
 │  Entraînement ML   │
 └─────────┬──────────┘
           │
    Log via MLFlow + artefacts MinIO
           │
 Sélection du meilleur modèle "avancé" (hors BERT) selon précision
           │
 ┌─────────▼──────────┐
 │    API FastAPI     │
 │    + Streamlit     │
 └─────────┬──────────┘
           │
 Déploiement sur Google Cloud Run
           │
┌──────────▼─────────────┐
│ Monitoring léger maison│
│ (alertes sur mauvaises │
│       prédictions)     │
└────────────────────────┘


---

## ✨ Conclusion

Ce projet a permis d’explorer en profondeur les enjeux de classification de sentiments et les **limites des approches standards face à la richesse du langage naturel**, en particulier sur Twitter.  
Il a aussi démontré l’intérêt d’un **pipeline MLOps souverain**, reproductible et automatisé.  

> ✅ En alliant des modèles puissants, des outils robustes comme MLFlow et FastAPI, et un déploiement maîtrisé, on pose les bases d’un produit de NLP industrialisable.

---

💬 Merci pour votre lecture !  
Pour tester l’API : [https://sentiment-api-service-7772256003.europe-west1.run.app/](https://sentiment-api-service-7772256003.europe-west1.run.app/)

