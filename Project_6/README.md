# Projet de Classification Automatique de Produits - Place de marché

## 📋 Contexte du projet

Ce projet a été réalisé pour **Place de marché**, une entreprise qui souhaite lancer une marketplace e-commerce anglophone. L'objectif est d'automatiser la classification des articles vendus sur la plateforme pour améliorer l'expérience utilisateur, tant pour les vendeurs que pour les acheteurs.

Actuellement, l'attribution des catégories est effectuée manuellement par les vendeurs, ce qui est peu fiable et difficile à passer à l'échelle. Ce projet vise à développer un moteur de classification automatique basé sur les descriptions textuelles (en anglais) et les images des produits.

## 🎯 Objectifs

Le projet se divise en plusieurs missions principales :

1. **Étude de faisabilité** : Analyser la capacité à classifier automatiquement des produits à partir de leurs descriptions textuelles et de leurs images
2. **Extraction de features** : Mettre en œuvre diverses techniques d'extraction de caractéristiques pour le texte et les images
3. **Analyse visuelle** : Réduire les données en 2D et visualiser la séparabilité des catégories
4. **Classification supervisée** : Entraîner un modèle de deep learning pour classifier les images avec data augmentation
5. **Collecte de données** : Tester l'API OpenFoodFacts pour enrichir la base de produits

## 📊 Données

Le projet utilise un dataset d'articles avec :
- Des descriptions textuelles en anglais
- Des images de produits
- Des catégories de produits (7 catégories principales)
- Environ 150 produits par catégorie

**Note** : Les données utilisées ne présentent aucune contrainte de propriété intellectuelle.

## 📁 Structure du projet

```
├── P6_01_NLP_Basic_methods.ipynb
├── P6_02_NLP_Doc2Vec.ipynb
├── P6_03_NLP_transfer_learning.ipynb
├── P6_04_IMAGES_SIFT_ORB.ipynb
├── P6_05_IMAGES_CNN_Transfer_learning.ipynb
├── P6_06_Classification_CNN_machine_learning.ipynb
├── P6_07_OpenFoodFacts_API.ipynb
├── P6_Classifiez_automatiquement_des_produits.pptx
└── test_images/
```

### 1. `P6_01_NLP_Basic_methods.ipynb`

**Notebook de traitement NLP - Méthodes classiques**

Ce notebook implémente les approches traditionnelles de traitement du texte :
- Prétraitement des textes (nettoyage, lemmatisation, suppression stopwords)
- Extraction de features avec Bag-of-Words (CountVectorizer)
- Extraction de features avec TF-IDF
- Réduction de dimension (ACP)
- Visualisation T-SNE en 2D
- Clustering K-means et calcul de l'Adjusted Rand Index (ARI)
- Évaluation de la faisabilité de classification automatique

### 2. `P6_02_NLP_Doc2Vec.ipynb`

**Notebook de traitement NLP - Word/Sentence Embedding classique**

Ce notebook explore les techniques d'embedding de mots :
- Implémentation de Word2Vec ou Doc2Vec
- Création de vecteurs de phrases/documents
- Réduction de dimension et visualisation T-SNE
- Comparaison avec les approches bag-of-words
- Analyse de la séparabilité des catégories

### 3. `P6_03_NLP_transfer_learning.ipynb`

**Notebook de traitement NLP - Transfer Learning**

Ce notebook met en œuvre des techniques NLP avancées :
- BERT (Bidirectional Encoder Representations from Transformers)
- USE (Universal Sentence Encoder)
- Extraction de features contextuelles
- Visualisation et comparaison des performances
- Évaluation comparative de toutes les approches NLP

### 4. `P6_04_IMAGES_SIFT_ORB.ipynb`

**Notebook de traitement d'images - Descripteurs classiques**

Ce notebook traite l'extraction de features d'images traditionnelles :
- Prétraitement des images (niveaux de gris, égalisation, filtrage)
- Extraction de descripteurs avec SIFT
- Extraction de descripteurs avec ORB (alternative à SURF)
- Création de Bag of Visual Words
- Réduction de dimension (ACP)
- Visualisation T-SNE et évaluation ARI

### 5. `P6_05_IMAGES_CNN_Transfer_learning.ipynb`

**Notebook de traitement d'images - CNN Transfer Learning**

Ce notebook utilise des réseaux de neurones pré-entraînés :
- Utilisation de CNN pré-entraînés (VGG16, ResNet, MobileNet, etc.)
- Extraction de features via Transfer Learning
- Comparaison avec les approches SIFT/ORB
- Visualisation T-SNE des embeddings
- Évaluation de la faisabilité avec des features CNN

### 6. `P6_06_Classification_CNN_machine_learning.ipynb`

**Notebook de classification supervisée d'images**

Ce notebook implémente la classification supervisée :
- Construction d'un modèle CNN de classification
- Séparation train/validation/test
- Data augmentation pour améliorer les performances
- Entraînement du modèle avec Transfer Learning
- Évaluation des performances (accuracy, matrice de confusion)
- Comparaison de différentes architectures CNN

### 7. `P6_07_OpenFoodFacts_API.ipynb`

**Notebook de test de l'API OpenFoodFacts**

Ce notebook démontre la collecte de données via API :
- Configuration de l'API OpenFoodFacts
- Requête pour extraire des produits à base de champagne
- Filtrage des données pertinentes
- Extraction des champs : foodId, label, category, foodContentsLabel, image
- Export des 10 premiers produits en format CSV
- Respect des normes RGPD

### 8. `P6_Classifiez_automatiquement_des_produits.pptx`

**Présentation des résultats**

Cette présentation (max 30 slides) synthétise :
- Le contexte et les enjeux du projet
- La méthodologie adoptée pour le texte et les images
- Les résultats comparatifs des différentes approches
- L'analyse de faisabilité de la classification automatique
- Les résultats de la classification supervisée
- Les recommandations pour la mise en production
- La démonstration de l'API de collecte de données

### 9. `test_images/`

**Dossier d'images de test**

Contient des images de test pour valider les modèles :
- Crème visage anti-âge
- Meuble 2 tiroirs
- Microphone filaire
- PC de bureau Asus
- Rideau occultant
- Tasse à café

## 🛠️ Technologies utilisées

- **Python** : Langage principal
- **NLP** :
  - NLTK : Prétraitement de texte
  - Scikit-learn : CountVectorizer, TF-IDF
  - Gensim : Word2Vec, Doc2Vec
  - Transformers : BERT
  - TensorFlow Hub : Universal Sentence Encoder
- **Computer Vision** :
  - OpenCV : Traitement d'images, SIFT, ORB
  - TensorFlow/Keras : CNN, Transfer Learning
  - PIL : Manipulation d'images
- **Machine Learning** :
  - Scikit-learn : ACP, K-means, métriques
  - Yellowbrick : Visualisation
- **Visualisation** :
  - Matplotlib, Seaborn : Graphiques
  - t-SNE : Réduction de dimension
- **API** :
  - Requests : Appels API
  - Pandas : Manipulation de données

## 📈 Méthodologie

### Phase 1 : Analyse de faisabilité - Texte
1. Prétraitement des descriptions textuelles
2. Extraction de features (Bag-of-Words, TF-IDF, Word2Vec, BERT, USE)
3. Réduction de dimension (ACP)
4. Visualisation T-SNE en 2D
5. Clustering K-means et calcul ARI
6. Analyse de la séparabilité des catégories

### Phase 2 : Analyse de faisabilité - Images
1. Prétraitement des images
2. Extraction de features (SIFT/ORB et CNN Transfer Learning)
3. Création de Bag of Visual Words (pour SIFT/ORB)
4. Réduction de dimension (ACP)
5. Visualisation T-SNE en 2D
6. Évaluation et comparaison des approches

### Phase 3 : Classification supervisée
1. Préparation des datasets (train/val/test)
2. Construction de modèles CNN avec Transfer Learning
3. Implémentation de data augmentation
4. Entraînement et optimisation
5. Évaluation des performances

### Phase 4 : Extension - Collecte de données
1. Test de l'API OpenFoodFacts
2. Requête et filtrage des produits
3. Export des données en CSV
4. Validation de la conformité RGPD

## 📝 Livrables

- ✅ 7 notebooks d'analyse et de modélisation
- ✅ Présentation de la démarche et des résultats (format .pptx)
- ✅ Fichier CSV d'extraction de produits via API
- ✅ Dossier d'images de test

## 🎓 Compétences développées

- Prétraitement de données textuelles (NLP)
- Prétraitement de données images (Computer Vision)
- Feature engineering pour texte et images
- Techniques d'embedding avancées (BERT, USE)
- Transfer Learning avec CNN pré-entraînés
- Réduction de dimensionnalité (ACP, t-SNE)
- Clustering non-supervisé (K-means)
- Classification supervisée avec deep learning
- Data augmentation pour l'optimisation de modèles
- Collecte de données via API
- Respect des normes RGPD
- Visualisation et communication de résultats

## 🔍 Principaux résultats attendus

- **ARI pour texte** : 0.4-0.5 (faisabilité confirmée)
- **ARI pour images SIFT/ORB** : 0.05-0.1 (résultats peu concluants)
- **ARI pour images CNN** : 0.4-0.6 (faisabilité confirmée)
- **Classification supervisée** : Amélioration significative avec data augmentation

## 👤 Auteur

**Grégoire Mureau**  
Date de réalisation : 2025

---

*Ce projet a été réalisé dans le cadre du parcours Data Scientist d'OpenClassrooms*
