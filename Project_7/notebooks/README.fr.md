# Notebooks

Ce dossier contient les notebooks Jupyter utilisés pour le développement des modèles et les analyses du projet.

## Contenu

- Analyse exploratoire des données
- Modèles classiques (régression logistique, SVM, etc.)
- Modèles avancés (LSTM avec embeddings GloVe)
- Modèle BERT

## Note concernant les fichiers requirements

Les fichiers `*requirements.txt` ont été générés depuis un environnement Google Colab. Cet environnement inclut de nombreuses librairies préinstallées, dont toutes ne sont pas nécessaires au projet.

Un nettoyage manuel ou une génération spécifique des fichiers requirements sera nécessaire pour alléger ces fichiers avant un usage en production ou un déploiement.

---

## Résumé des performances des modèles

### Modèles classiques (entraînés sur 200 000 tweets)

| Modèle                         | Précision | Temps d'entraînement|
|--------------------------------|-----------|---------------------|
| Random Forest + BoW            | 0,7798    | ~5 heures           |
| Régression logistique + TF-IDF | 0,7796    | <2 minutes          |

### Modèles avancés basés sur des embeddings (entraînés sur 200 000 tweets, accélérés GPU sur Google Colab)

| Modèle                 | Précision | Temps d'entraînement|
|------------------------|-----------|---------------------|
| Word2Vec + LSTM        | 0,797     | 57 secondes         |
| GloVe (300d) + LSTM    | 0,792     | 47 secondes         |

### Modèle de pointe (entraîné sur 100 000 tweets en raison des limites RAM)

| Modèle  | Précision | Temps d'entraînement|
|---------|-----------|---------------------|
| BERT    | 0,882     | 1,3 minutes         |

---

Ce résumé met en lumière les compromis entre précision des modèles et temps d’entraînement. Si les modèles classiques sont plus rapides à entraîner, les modèles avancés utilisant des embeddings et LSTM offrent une meilleure précision, avec BERT qui atteint la meilleure précision malgré un jeu de données plus restreint.

L’écart de précision entre BERT et les modèles à embeddings peut s’expliquer principalement par la capacité bidirectionnelle de BERT à comprendre le contexte.
