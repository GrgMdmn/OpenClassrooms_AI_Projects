
# Dashboard de Comparaison de Modèles

## Conformité aux Exigences

### Analyse Exploratoire des Données Non Structurées (Images)

**Présentation d'exemples d'images selon les catégories :**
- Visualisation d'images du dataset Cityscapes organisées par 8 méta-classes de segmentation
- Sélection interactive d'images d'entraînement et de test avec prévisualisation
- Interface de navigation par catégories avec noms d'affichage conviviaux

**Comptage et répartition des données :**
- Graphiques en secteurs (pie charts) détaillant la répartition des 25,000 images Cityscapes
- Visualisation de l'utilisation projet : 80% train / 20% validation / 100% test
- Graphique de distribution des classes dans le dataset d'entraînement
- Tableaux récapitulatifs avec statistiques détaillées par split

**Transformations des images :**
- Démonstration interactive du preprocessing : redimensionnement et normalisation
- Visualisation des augmentations de données aléatoires (rotation, flip, luminosité, etc.)
- Comparaison avant/après transformations avec alignement visuel
- Documentation complète des 6 types d'augmentations possibles

### Sélection de Données d'Entrée

**Interface de sélection multiple :**
- Liste déroulante pour images test Cityscapes avec informations sur la vérité terrain
- Upload de fichiers personnels (JPEG, PNG) avec validation de format
- Prévisualisation des images sélectionnées avant traitement
- Gestion des erreurs et feedback utilisateur en temps réel

### Résultats de Prédiction

**Comparaison complète de modèles :**
- Affichage côte à côte des prédictions des deux meilleurs modèles
- Visualisations avec image originale, masques prédits et vérité terrain
- Détection et mise en évidence des erreurs de segmentation
- Statistiques détaillées par classe avec pourcentages et nombre de pixels

**Métriques de performance :**
- Temps d'inférence comparatif avec analyse de vitesse
- Scores de performance (mIoU, Accuracy) par modèle
- IoU par classe pour analyse fine des performances
- Graphiques comparatifs interactifs des résultats

### Déploiement Cloud

**Architecture déployée :**
- Application FastAPI pour l'API de comparaison de modèles
- Interface Streamlit pour le dashboard utilisateur
- Déploiement sur infrastructure cloud avec haute disponibilité
- Gestion automatique des ressources et montée en charge

### Accessibilité WCAG

**Conformité aux critères essentiels :**

#### Contraste des Couleurs (WCAG AA)
- Palette de couleurs avec ratios de contraste ≥ 4.1:1 pour éléments graphiques
- Ratios de contraste ≥ 4.5:1 pour le texte sur fond
- 8 couleurs principales validées : bleu (#0066CC), rouge (#CC0000), vert (#006600), etc.
- Remplacement des dégradés par des couleurs unies à contraste élevé

#### Alternatives Textuelles
- Descriptions alternatives détaillées pour tous les graphiques générées automatiquement
- Textes explicatifs pour chaque visualisation (pie charts, bar charts)
- Labels ARIA appropriés pour les éléments interactifs
- Documentation textuelle complète des transformations d'images

#### Indication Non-Couleur Uniquement
- Motifs et hachures complémentaires aux couleurs (optionnel via checkbox)
- 6 motifs différents : diagonales, verticales, horizontales, points, croix
- Différenciation par forme et texture en plus des couleurs
- Bordures et contours renforcés pour la distinction visuelle

#### Navigation Clavier
- CSS focus visible avec contours bleus (#0066CC) de 2px
- Support complet de la navigation par tabulation
- Indicateurs visuels clairs pour les éléments focusés
- Respect de l'ordre logique de navigation

#### Tableaux de Données Accessibles
- Tableaux structurés complémentaires à tous les graphiques
- En-têtes appropriés avec scope et descriptions
- Données exportables et lisibles par lecteurs d'écran
- Configuration responsive pour tous les écrans

**Fonctionnalités d'assistance intégrées :**
- Checkbox "Activer les motifs (accessibilité visuelle)" pour personnalisation
- Descriptions automatiques des graphiques avec statistiques complètes
- Interface adaptable aux besoins des utilisateurs en situation de handicap
- Documentation complète des fonctionnalités d'accessibilité via endpoint `/accessibility-info`

## Architecture Technique

**Stack technologique :**
- Backend : FastAPI + PyTorch pour l'inférence des modèles
- Frontend : Streamlit avec composants accessibles
- Visualisations : Matplotlib avec palettes WCAG
- API REST documentée avec validation Pydantic
- Gestion d'état optimisée pour les comparaisons de modèles

**Performance et fiabilité :**
- Temps d'inférence optimisés (< 1 seconde par image)
- Gestion mémoire efficace avec fermeture automatique des figures
- Validation des formats et tailles de fichiers
- Nettoyage automatique des fichiers temporaires
