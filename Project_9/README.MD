# Street Vision PoC - Segmentation Sémantique Urbaine

## 🎯 Description du Projet

**Street Vision PoC** est une preuve de concept démontrant l'amélioration des performances de segmentation sémantique urbaine en comparant les architectures CNN traditionnelles (FPN) aux approches Transformer modernes (SegFormer).

Ce projet évalue empiriquement le potentiel des **Vision Transformers** pour la compréhension de scènes urbaines complexes, avec un focus sur les classes critiques pour la sécurité (piétons, véhicules, signalétique).

### 🔬 Contexte Technique

- **Dataset** : Cityscapes (2,975 images haute résolution annotées pixel-par-pixel)
- **Architectures comparées** : 
  - **CNN traditionnel** : FPN + EfficientNet-B0/ResNet34
  - **Vision Transformer** : SegFormer-B0/B1
- **Résolutions d'évaluation** : 512×512 et 768×768 pixels
- **Métrique principale** : Mean Intersection over Union (mIoU)

## 📊 Résultats Principaux

| **Modèle** | **mIoU** | **Paramètres** | **Taille** | **Performance** |
|------------|----------|----------------|------------|-----------------|
| **SegFormer-B1** | **79.98%** | 13.68M | 52MB | ⭐ **Meilleur** |
| **FPN+ResNet34** | 79.22% | 23.16M | 88MB | 🔄 Baseline forte |
| **FPN+EfficientNet-B0** | 78.67% | 5.76M | 22MB | 💡 Compact |
| **SegFormer-B0** | 77.91% | 3.72M | 14MB | ⚡ Ultra-léger |

**Analyse comparative** : SegFormer-B1 et FPN+ResNet34 atteignent des performances similaires (~79-80% mIoU), mais SegFormer-B1 présente l'avantage d'être plus compact et probablement plus rapide sur GPU. Les versions SegFormer-B2 à B5 promettraient des gains significatifs, mais mes contraintes matérielles n'ont pas permis leur évaluation.

### 🎯 Gains Significatifs par Classes Critiques

- **Human** : +1.78% IoU (sécurité critique)
- **Vehicle** : +1.05% IoU (obstacles mobiles)
- **Object** : +0.56% IoU (signalétique urbaine)

## 🚀 Déploiement Rapide

### Docker Hub

L'application est containerisée et disponible sur Docker Hub :

```bash
docker pull grgmdmn/street-vision-poc_api
```

### Configuration avec Docker Compose

1. Copiez le fichier d'exemple de configuration :
```bash
cp docker-compose.yml.example docker-compose.yml
```

2. Configurez les variables d'environnement dans `docker-compose.yml`.

3. Lancez l'application :
```bash
docker-compose up -d
```

4. Accédez à l'interface : `http://localhost:$(port)`

## 🎮 Fonctionnalités de l'Interface

### Dashboard Interactif

- **Upload d'images** : Glisser-déposer d'images urbaines
- **Double inférence** : Comparaison simultanée FPN vs SegFormer
- **Visualisation** : Masques de segmentation colorés par classe
- **Métriques temps réel** : Temps d'inférence et scores de confiance
- **Export** : Téléchargement des résultats de segmentation
- **Accessibilité WCAG** : Interface conforme aux standards d'accessibilité

### Analyse Exploratoire des Données

- **Statistiques Cityscapes** : Distribution des classes, résolutions natives
- **Graphiques interactifs** : Répartition pixel par méta-classe
- **Exemples annotés** : Visualisation échantillons du dataset

## 📈 Méthodologie d'Évaluation

### Approche Comparative

1. **Baseline CNN** : Réentraînement FPN+EfficientNet/ResNet aux résolutions supérieures
2. **Nouvelle approche** : Évaluation SegFormer avec attention globale
3. **Protocole rigoureux** : Validation croisée, métriques standardisées
4. **Analyse par classe** : Performance détaillée sur 8 méta-classes urbaines

### Limites Identifiées

- **Résolution fixe** : Évaluation sans sliding window (vs protocole original)
- **Ressources limitées** : Tests uniquement sur SegFormer-B0/B1
- **Conditions contrôlées** : Dataset Cityscapes uniquement (météo favorable)

## 🔬 Insights Techniques

### Avantages SegFormer

- **Attention globale** : Chaque pixel "voit" toute l'image simultanément
- **Architecture plus légère** : Modèle compact avec moins de paramètres
- **Robustesse contextuelle** : Meilleure gestion occlusions et objets complexes
- **Potentiel d'amélioration** : Versions B2-B5 promettraient des gains significatifs

### Trade-offs Identifiés

- **Vitesse CPU classique** : ~2× plus lent que FPN sur processeurs haute performance (Xeon, 9600X)
- **Vitesse CPU économe** : 3× plus rapide sur processeurs économes en énergie (N100)
- **Optimisation GPU** : Architecture Transformer favorisée par accélération matérielle

## 📚 Références

- **Xie, E. et al.** (2021). *SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers*. NeurIPS 2021.
- **Cordts, M. et al.** (2016). *The Cityscapes Dataset for Semantic Urban Scene Understanding*. CVPR 2016.
- **Segmentation Models PyTorch** - [Framework d'implémentation utilisé](https://github.com/qubvel/segmentation_models.pytorch)

---

**Note** : Cette preuve de concept démontre le potentiel des architectures Transformer pour la segmentation urbaine, tout en soulignant l'importance des ressources computationnelles pour exploiter pleinement leurs capacités.