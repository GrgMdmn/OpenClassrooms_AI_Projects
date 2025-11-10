# Projet de Traitement Big Data sur le Cloud - Fruits!

## 📋 Contexte du projet

Ce projet a été réalisé dans le cadre d'une mission de consulting pour **Fruits!**, une jeune start-up AgriTech qui développe des solutions innovantes pour la récolte des fruits. L'entreprise souhaite préserver la biodiversité des fruits en permettant des traitements spécifiques pour chaque espèce grâce au développement de robots cueilleurs intelligents.

## 🎯 Objectifs

Compléter une chaîne de traitement d'extraction de features dans un environnement Big Data sur le cloud AWS :

1. **Reprise des travaux** : S'approprier le notebook incomplet d'un alternant précédent
2. **Complétion de la pipeline** : Implémenter les parties manquantes (broadcast des poids, PCA distribuée)
3. **Mise en production** : Déployer la solution sur un cluster AWS EMR conforme au RGPD
4. **Optimisation des coûts** : Maintenir les coûts d'exécution sous 10€

L'objectif à long terme est de mettre en place un moteur de classification des images de fruits pour une application mobile de sensibilisation du grand public.

## 📊 Données

Le jeu de données utilisé est **Fruits-360**, disponible sur Kaggle :
- **147 691 images** de fruits de différentes variétés
- Format : 100x100 pixels, couleur
- Jeu de test spécifique : 103 images de fruits multiples
- **⚠️ Note importante** : Le jeu de données n'est pas inclus dans ce dépôt en raison de sa taille
- **URL** : https://www.kaggle.com/moltean/fruits

## 📁 Structure du projet

```
├── P11_01_notebook.ipynb
├── P11_02_images/
│   ├── Results_PCA/                    # Résultats PCA au format Parquet
│   │   ├── part-00000-*.parquet
│   │   ├── part-00001-*.parquet
│   │   └── ...
│   └── test-multiple-fruits/           # Images de test (103 images)
│       ├── apple.jpg
│       ├── apples1.jpg
│       └── ...
└── P11_03_présentation.pdf
```

### 1. `P11_01_notebook.ipynb`

**Notebook principal de traitement Big Data PySpark**

Ce notebook contient la chaîne complète de traitement :

- **Initialisation** : Configuration de la SparkSession et connexion à S3
- **Chargement des données** : Lecture des images depuis S3 en format binaire
- **Transfer Learning** : Extraction de features via MobileNetV2 pré-entraîné sur ImageNet
  - Suppression de la couche de classification finale
  - Extraction de vecteurs de 1280 dimensions
  - Broadcast des poids du modèle aux workers (ajout critique)
- **Réduction de dimensionnalité** : PCA distribuée avec Spark ML
  - Réduction de 1280 → 70 dimensions
  - Conservation de 99,94% de la variance
  - Optimisation du stockage et des performances
- **Export des résultats** : Sauvegarde au format Parquet sur S3

**Optimisations clés implémentées** :
- Pandas UDF Scalar Iterator pour traitement par batch
- Broadcast des poids du modèle pour éviter les chargements répétitifs
- PCA distribuée avec calcul parallélisé sur le cluster

### 2. `P11_02_images/`

**Dossier des images et résultats**

- **Results_PCA/** : Résultats de la réduction de dimensionnalité stockés au format Parquet distribué (20 fichiers)
- **test-multiple-fruits/** : 103 images de test variées utilisées pour valider la pipeline

### 3. `P11_03_présentation.pdf`

**Présentation des résultats**

Cette présentation synthétise :
- Le contexte et les enjeux du projet
- L'architecture cloud mise en place (AWS EMR, S3, EC2)
- La configuration du cluster et les optimisations Spark
- Le pipeline algorithmique complet
- Les résultats techniques obtenus
- Les perspectives d'amélioration et de déploiement

## 🛠️ Technologies utilisées

### Cloud & Infrastructure
- **AWS EMR** : Cluster de calcul distribué (Spark, Hadoop)
- **AWS S3** : Stockage des données et résultats
- **AWS EC2** : Instances m5.xlarge (1 Master + 2 Workers)
- **Région EU-West-3** (Paris) : Conformité RGPD

### Big Data & Traitement
- **PySpark** : Framework de calcul distribué
- **Hadoop** : Système de fichiers distribué
- **Spark ML** : Bibliothèque de Machine Learning distribuée

### Machine Learning
- **TensorFlow** : Framework de deep learning
- **MobileNetV2** : Modèle de transfer learning pré-entraîné sur ImageNet
- **PCA** : Algorithme de réduction de dimensionnalité

### Outils de développement
- **JupyterHub** : Environnement de développement sur le cluster
- **SSH Tunneling** : Accès sécurisé au cluster
- **Python 3** : Langage de programmation

## 📈 Architecture technique

### Configuration du Cluster EMR

**Instances** :
- 1 nœud Master (driver Spark) - m5.xlarge
- 2 nœuds Workers (executors Spark) - m5.xlarge
- Région : eu-west-3 (Paris)

**Logiciels installés** :
- Hadoop 3.2.1
- Spark 3.1.2
- JupyterHub 1.4.1
- TensorFlow 2.4.1

**Bootstrap** : Installation automatique des packages Python
```bash
sudo python3 -m pip install numpy pandas pillow pyarrow fsspec s3fs
```

### Accès Sécurisé

**Tunneling SSH avec Port Forwarding** :
```bash
ssh -i ./emr-keypair.pem -L 8890:localhost:8890 -L 9443:localhost:9443 hadoop@[IP-EMR]
```
- Port 9443 : JupyterHub (développement)
- Port 8890 : Interface monitoring Spark

**Avantages** :
- Connexion chiffrée de bout en bout
- Pas de proxy externe nécessaire
- Contrôle précis des accès

### Pipeline de Traitement

1. **Chargement des images** : Format binaire depuis S3, extraction des labels depuis les chemins
2. **Transfer Learning** : MobileNetV2 → features 1280 dimensions
   - Broadcast des poids pour distribution efficace
   - Pandas UDF Scalar Iterator pour traitement par batch
3. **PCA Distribuée** : Réduction 1280 → 70 dimensions
   - Calcul distribué de la matrice de covariance
   - Extraction des composantes principales
   - Transformation des features
4. **Export** : Sauvegarde au format Parquet sur S3

## 📝 Méthodologie

1. **Compréhension** : Analyse du notebook existant et identification des manques
2. **Développement local** : Tests et validation sur échantillon réduit
3. **Migration cloud** : Déploiement sur cluster EMR
4. **Optimisation** : Ajout du broadcast et de la PCA distribuée
5. **Validation** : Tests sur 103 images, vérification de la scalabilité

## 💰 Gestion des coûts

**Objectif** : < 10€ pour la validation

**Stratégies d'optimisation** :
- Utilisation d'un cluster modeste (1 Master + 2 Workers)
- Instances m5.xlarge (compromis performance/coût)
- Tests sur échantillon réduit avant passage à l'échelle
- Arrêt du cluster après utilisation
- Serveur local pour développement et tests

**Résultat** : Coût de validation < 5€ (moins d'1h de cluster)

## 🎯 Résultats

### Techniques

- ✅ Réduction de dimensionnalité : 1280 → 70 dimensions
- ✅ Variance conservée : **99,94%**
- ✅ Facteur de compression : **x4,1**
- ✅ Pipeline complète et fonctionnelle
- ✅ Architecture scalable validée

### Conformité

- ✅ **RGPD** : Région EU-West-3 (Paris)
- ✅ **Sécurité** : Accès SSH, tunneling sécurisé
- ✅ **Coûts** : < 10€ pour la validation

### Fonctionnalités implémentées

- ✅ Broadcast des poids MobileNetV2 (manquant dans version initiale)
- ✅ PCA distribuée avec Spark ML (manquant dans version initiale)
- ✅ Optimisation mémoire avec Pandas UDF Scalar Iterator
- ✅ Export des résultats au format Parquet

## 🚀 Perspectives

### Court terme

**Optimisation de la PCA** :
- 70 composantes valables pour 103 images de test
- Avec dataset complet (~150k images), ajuster le nombre de composantes k
- Tester différentes valeurs pour optimiser variance/dimensions

**Classification finale** :
- Entraîner un modèle de classification sur les features PCA
- Évaluer les performances sur le jeu de test
- Fine-tuning du nombre de composantes

### Moyen terme

**Déploiement production** :
- Intégration du modèle dans l'application mobile
- Pipeline CI/CD automatisée
- Monitoring des performances en temps réel
- Gestion des versions du modèle

### Long terme

**Extensions futures** :
- Utilisation de l'application comme MVP marketing et technique
- Collecte de nouvelles données terrain
- Amélioration continue du modèle
- Intégration avec les robots cueilleurs intelligents

## 🎓 Compétences développées

- Architecture Big Data sur le cloud (AWS EMR, S3, EC2)
- Traitement distribué avec PySpark
- Transfer Learning et extraction de features
- Réduction de dimensionnalité à grande échelle
- Optimisation des performances Spark
- Gestion des coûts cloud
- Conformité RGPD pour le stockage de données

## 👤 Auteur

**Grégoire Mureau**  
Date de réalisation : Octobre 2025

---

*Ce projet a été réalisé dans le cadre du parcours AI Engineer*
