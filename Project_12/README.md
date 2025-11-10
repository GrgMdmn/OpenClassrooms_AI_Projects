# Projet de Cadrage d'un Projet IA - Fashion-Insta

## 📋 Contexte du projet

Ce projet a été réalisé dans le cadre d'une mission d'**AI Product Manager** chez **Fashion-Insta**, une entreprise du monde de la mode qui commercialise des articles vestimentaires via un réseau de magasins physiques et un site e-commerce.

L'objectif était de réaliser le cadrage complet d'un projet IA stratégique : le développement d'une application mobile de recommandation d'articles vestimentaires basée sur des photos, en vue de présenter le projet au comité exécutif (COMEX) pour validation.

## 🎯 Objectifs

Le projet vise à développer une **application mobile de recommandation** permettant aux utilisateurs de :
1. Se prendre en photo avec leurs habits favoris
2. Obtenir des recommandations d'articles du même style vestimentaire
3. Booster les ventes en ligne et en magasin

**Mission confiée** : Réaliser l'intégralité du cadrage du projet selon une méthodologie agile SCRUM et convaincre le COMEX de lancer le projet.

## 📊 Démarche de cadrage

Le cadrage du projet s'est déroulé en plusieurs phases sur 3 semaines :

### Phase 1 : Expression des besoins et backlog
- Formalisation des user stories à partir des besoins métier
- Priorisation avec la méthode MoSCoW (Must have, Should have, Could have, Won't have)
- Identification du MVP (Minimum Viable Product)
- Revue avec le Product Owner

### Phase 2 : Estimation et rentabilité
- Estimation des charges par user story (en jours/homme)
- Répartition par profil (Data Scientist, Développeur, DevOps, etc.)
- Calcul des coûts de développement
- Estimation des coûts Azure (infrastructure initiale et production)
- Analyse de rentabilité (ROI, break-even point)
- Planification des sprints

### Phase 3 : Conformité RGPD et risques
- Analyse des données personnelles (registre de traitements CNIL)
- Identification des enjeux légaux et éthiques
- Analyse des biais potentiels du modèle
- Analyse complète des risques projet
- Plan d'action de mitigation

### Phase 4 : Présentation au COMEX
- Synthèse des objectifs et gains attendus
- Présentation de la méthodologie agile SCRUM
- Planning des sprints et organisation
- Enjeux RGPD et éthiques
- Mitigation des risques

## 📁 Structure du projet

```
├── P12_01_backlog.xlsx
├── P12_02_tableur.xlsx
├── P12_03_présentation.pdf (ou .pptx)
├── P12_04_ExportedEstimate_initial.xlsx
└── P12_05_ExportedEstimate_production.xlsx
```


### 1. `P12_01_backlog.xlsx`

**Backlog du projet avec user stories priorisées**

Ce fichier contient :
- **User stories** : Expression des besoins fonctionnels et techniques du point de vue utilisateur
- **Priorisation MoSCoW** :
  - **Must have** : Fonctionnalités indispensables au MVP
  - **Should have** : Fonctionnalités importantes mais non bloquantes
  - **Could have** : Fonctionnalités souhaitables
  - **Won't have** : Fonctionnalités exclues du périmètre actuel
- **Estimation des charges** : Temps de développement par user story
- **Critères d'acceptation** : Conditions de validation de chaque story
- **Dépendances** : Relations entre les user stories

**Focus** : User stories IA (recommandation, traitement d'images, modèles ML)

### 2. `P12_02_tableur.xlsx`

**Tableur d'estimation des charges et coûts**

Ce fichier comprend plusieurs onglets :

**Onglet "Charges par profil"** :
- Répartition des charges par user story
- Profils : Data Scientist, ML Engineer, Développeur Full-Stack, DevOps, Product Owner, Scrum Master
- Calcul des jours/homme par profil
- Application des coûts journaliers (TJM)

**Onglet "Coûts de développement"** :
- Coût total de développement initial
- Répartition par profil et par sprint
- Coûts fixes vs variables

**Onglet "Coûts récurrents"** :
- Maintenance annuelle : 15% du coût de développement
- Coûts d'infrastructure Azure de production (annuels)
- Évolution des coûts sur 3-5 ans

**Onglet "Analyse de rentabilité"** :
- Gains annuels estimés (augmentation des ventes)
- Coûts cumulés année après année
- Calcul du ROI et du break-even point
- Graphique de rentabilité

**Onglet "Planning des sprints"** :
- Découpage en sprints (durée, contenu)
- Affectation des user stories par sprint
- Jalons et livrables

### 3. `P12_03_présentation.pdf` (ou .pptx)

**Présentation pour le COMEX**

Cette présentation comprend :

**Slide 1 : Résumé exécutif**
- Synthèse des points clés (non présentée oralement)
- Vision du projet en une slide

**Slides 2-3 : Objectifs et gains attendus**
- Augmentation des ventes (quantifiée)
- Amélioration de l'expérience client
- KPIs de succès (taux d'adoption, taux de conversion, satisfaction)

**Slides 4-5 : Ressources requises**
- Humaines : Équipe et profils nécessaires
- Techniques : Infrastructure Azure
- Financières : Investissement initial et coûts récurrents

**Slides 6-8 : Méthodologie agile SCRUM**
- Principes de l'agilité
- Rôles (Product Owner, Scrum Master, Development Team)
- Avantages pour le projet

**Slides 9-11 : Organisation et suivi**
- Daily Scrum (point quotidien)
- Sprint Review (démo en fin de sprint)
- Rétrospective (amélioration continue)
- Burndown chart (suivi de l'avancement)

**Slides 12-13 : Planning des sprints**
- Découpage temporel
- Contenu de chaque sprint
- Jalons et livrables clés

**Slides 14-16 : Enjeux légaux et éthiques**
- Principes du RGPD
- Gestion des données personnelles (photos, préférences)
- Biais potentiels du modèle (diversité, représentativité)
- Transparence et consentement

**Slides 17-19 : Risques et mitigation**
- Risques techniques (performance du modèle, scalabilité)
- Risques organisationnels (disponibilité des équipes)
- Risques financiers (dépassement de budget)
- Risques légaux (non-conformité RGPD)
- Plan d'action pour chaque risque critique

**Annexe : Backlog complet**
- Liste des user stories avec priorisation
- User stories du MVP clairement identifiées

### 4. `P12_04_ExportedEstimate_initial.xlsx`

**Estimation des coûts Azure pour la phase initiale**

Ce fichier contient l'estimation des coûts d'infrastructure cloud Azure pour :
- **Phase de conception** : Environnement de développement
- **Phase d'entraînement des modèles** :
  - Instances de calcul (GPU pour deep learning)
  - Stockage des données d'entraînement
  - Services de ML (Azure Machine Learning)
  - Services de vision par ordinateur (Computer Vision API)

**Export** : Calculateur de prix Azure (Azure Pricing Calculator)

### 5. `P12_05_ExportedEstimate_production.xlsx`

**Estimation des coûts Azure pour la production**

Ce fichier contient l'estimation des coûts d'infrastructure cloud Azure pour :
- **Production de l'application** :
  - Instances de calcul pour l'API de recommandation
  - Base de données (catalogue produits, historique utilisateurs)
  - Stockage blob (images utilisateurs)
  - CDN (Content Delivery Network)
  - Services d'inférence ML
  - Services de monitoring et logs
  - Bande passante

**Calcul** : Coûts mensuels et annuels basés sur une estimation du trafic

## 🛠️ Méthodologie

### Méthode Agile SCRUM

**Rôles** :
- **Product Owner** : Alicia (VP Product)
- **Scrum Master** : À définir
- **Development Team** : Data Scientists, ML Engineers, Développeurs, DevOps

**Cérémonies** :
- **Sprint Planning** : Planification du sprint (1-2 semaines)
- **Daily Scrum** : Point quotidien de 15 minutes
- **Sprint Review** : Démonstration en fin de sprint
- **Sprint Retrospective** : Amélioration continue

**Artefacts** :
- **Product Backlog** : Liste priorisée des user stories
- **Sprint Backlog** : User stories sélectionnées pour le sprint
- **Increment** : Produit potentiellement livrable à la fin du sprint

### Priorisation MoSCoW

- **Must have (M)** : Essentiel au MVP, non négociable
- **Should have (S)** : Important mais peut être reporté
- **Could have (C)** : Souhaitable si budget/temps disponible
- **Won't have (W)** : Exclu du périmètre actuel

### Estimation des charges

**Techniques utilisées** :
- Planning Poker pour l'estimation collaborative
- Story Points ou Jours/Homme selon les user stories
- Vélocité de l'équipe pour ajuster les prévisions

## 📈 Analyse de rentabilité

### Coûts

**Investissement initial** :
- Développement : Coût total des sprints
- Infrastructure Azure (phase initiale)
- Formation des équipes

**Coûts récurrents (annuels)** :
- Maintenance : 15% du coût de développement
- Infrastructure Azure (production)
- Support et évolutions

### Gains

**Gains estimés** :
- Augmentation du taux de conversion en ligne
- Augmentation du panier moyen
- Fidélisation client (réduction du churn)
- Cross-selling et up-selling

**Calcul du ROI** :
```
ROI = (Gains cumulés - Coûts cumulés) / Coûts cumulés × 100
```

**Break-even point** : Date à laquelle les gains cumulés égalent les coûts cumulés

### Visualisation

Graphique montrant :
- Courbe des coûts cumulés
- Courbe des gains cumulés
- Point de rentabilité (intersection des courbes)

## 🔒 Conformité RGPD

### Données personnelles traitées

- **Photos des utilisateurs** : Données sensibles (images de personnes)
- **Préférences vestimentaires** : Profilage comportemental
- **Historique de navigation** : Suivi des interactions
- **Données de compte** : Email, nom, coordonnées

### Registre de traitements (CNIL)

Pour chaque traitement IA :
- **Finalité** : Recommandation d'articles personnalisée
- **Catégories de données** : Images, préférences, comportement
- **Durée de conservation** : Définie et justifiée
- **Mesures de sécurité** : Chiffrement, pseudonymisation
- **Destinataires** : Équipe data, partenaires (si applicable)
- **Transferts hors UE** : Azure région Europe (conformité)

### Principes RGPD appliqués

- **Consentement explicite** : Opt-in pour l'utilisation des photos
- **Droit d'accès** : L'utilisateur peut consulter ses données
- **Droit à l'effacement** : Suppression des données sur demande
- **Portabilité** : Export des données personnelles
- **Minimisation** : Collecte uniquement des données nécessaires
- **Transparence** : Information claire sur l'utilisation des données

## ⚖️ Enjeux éthiques

### Biais du modèle

**Risques identifiés** :
- **Biais de représentation** : Sous-représentation de certaines morphologies, couleurs de peau, styles vestimentaires
- **Biais culturel** : Recommandations orientées vers des standards occidentaux
- **Biais de genre** : Stéréotypes vestimentaires genrés

**Plan d'action** :
- Diversification du jeu de données d'entraînement
- Audit régulier des recommandations
- Tests utilisateurs sur des populations variées
- Possibilité de feedback utilisateur pour améliorer le modèle

### Transparence et explicabilité

- Informer l'utilisateur sur le fonctionnement du système de recommandation
- Possibilité de comprendre pourquoi un article est recommandé
- Option de désactivation des recommandations personnalisées

## ⚠️ Analyse des risques

### Méthodologie

**Grille d'évaluation** :
- **Probabilité** : Faible / Moyenne / Élevée
- **Impact** : Faible / Moyen / Critique
- **Criticité** : Probabilité × Impact

**Checklist Spectre 7D** :
1. **Data** : Qualité, disponibilité, biais
2. **Development** : Compétences, technologies
3. **Deployment** : Infrastructure, scalabilité
4. **Dependability** : Fiabilité, performance
5. **Detectability** : Monitoring, alertes
6. **Diversity** : Inclusion, équité
7. **Documentation** : Traçabilité, conformité

### Principaux risques identifiés

**Risques techniques** :
- Performance insuffisante du modèle de recommandation
- Temps de latence trop élevé en production
- Difficultés de scalabilité avec la montée en charge

**Risques organisationnels** :
- Indisponibilité des ressources (Data Scientists)
- Turnover dans l'équipe projet
- Retards dans les livrables

**Risques financiers** :
- Dépassement du budget Azure
- Coûts de développement sous-estimés
- ROI plus long que prévu

**Risques légaux** :
- Non-conformité RGPD (sanctions financières)
- Plaintes d'utilisateurs sur la gestion des données
- Litiges liés aux biais du modèle

### Plan de mitigation

Pour chaque risque critique :
- **Action préventive** : Mesure pour réduire la probabilité
- **Action corrective** : Mesure si le risque se matérialise
- **Responsable** : Personne en charge du suivi
- **Échéance** : Date de mise en œuvre

**Exemples** :
- **Risque RGPD** → Audit de conformité avec le DPO avant chaque sprint
- **Risque de biais** → Diversification du dataset et tests réguliers
- **Risque de scalabilité** → Load testing et architecture cloud élastique

## 🎓 Compétences développées

- Cadrage de projet IA end-to-end
- Méthodologie agile SCRUM
- Estimation et chiffrage de projets techniques
- Analyse de rentabilité (ROI, break-even)
- Conformité RGPD et gestion des données personnelles
- Identification et mitigation des risques
- Enjeux éthiques de l'IA (biais, transparence)
- Gestion des parties prenantes (COMEX)
- Communication et présentation de projet

## 👤 Auteur

**Grégoire Mureau**  
Date de réalisation : 2025

---

*Ce projet a été réalisé dans le cadre du parcours AI Engineer*
