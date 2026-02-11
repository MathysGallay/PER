# PER2025-046 — Cartes de chaleur nocturnes par caméra thermique pour le suivi d'animaux sauvages

> **Objectif :** concevoir et valider une pipeline de vision embarquée légère qui surveille les animaux la nuit par imagerie thermique, produit des **cartes de chaleur d'activité**, et stocke les sorties de détection/suivi pour analyse ultérieure.  
> **Contexte :** suivi en zoo / bien-être animal et compréhension des comportements nocturnes.

---

## Table des matières

1. [Identification du projet](#1-identification-du-projet)  
2. [Présentation du sujet](#2-presentation-du-sujet)  
3. [Espace de solutions et choix](#3-espace-de-solutions-et-choix)  
4. [Positionnement vs travaux existants (État de l'art)](#4-positionnement-vs-travaux-existants-etat-de-lart)  
5. [Travail réalisé : produit et processus](#5-travail-realise-produit-et-processus)  
6. [Résultats actuels et plan d'évaluation](#6-resultats-actuels-et-plan-devaluation)  
7. [Limites et suites à donner](#7-limites-et-suites-a-donner)  
8. [Bibliographie / Références](#8-bibliographie--references)  
9. [Annexes](#9-annexes)

---

## 1. Identification du projet

### 1.1 Titre
**Cartes de chaleur nocturnes par caméra thermique pour le suivi d'animaux sauvages**

### 1.2 Équipe
- **Yanis Lamiri** — Matériel / Intégration embarquée sur Raspberry pi 5 + accélérateur Hailo
- **Yanis Abdellaoui** — Suivi / Métriques comportementales et analyse   / Intégration embarquée sur Jetson Orin Nanon
- **Mathys Gallay** — Vision thermique / Modèle de détection (fine-tuning YOLO)  

### 1.3 Enseignant référent
Jean Martinet


### 1.4 Établissement
Polytech Nice Sophia, Université Côte d’Azur — 930 Route des Colles, 06410 Biot, France


### 1.5 Dates
- Début : octobre 2025
- Fin : février 2026

### 1.6 Mots-clés
Imagerie thermique, Vision embarquée, Edge AI, YOLO, Suivi, Cartes de chaleur, Suivi en zoo

---

## 2. Présentation du sujet

### 2.1 Problème et motivation

Comprendre les **comportements nocturnes** est utile pour les zoos et le bien-être animal : les animaux peuvent montrer la nuit des comportements différents de ceux observés le jour (absence de visiteurs, stress réduit, conditions environnementales et météorologiques). Ces différences aident à interpréter la fatigue, peur, agressivité, stéréotypes, et plus généralement à améliorer l'aménagement des enclos et les pratiques de suivi.

La surveillance RGB classique devient difficile la nuit :
- nécessite un éclairage additionnel (qui peut perturber les animaux),
- souffre de bruit en basse luminosité,
- produit des résultats instables dans les scènes sombres.

L'imagerie thermique est une alternative naturelle car elle ne dépend **pas de la lumière visible** et capte les émissions infrarouges. Cependant, les capteurs thermiques low-cost imposent des contraintes (résolution, bruit, texture limitée) qui impactent la détection et suivi.

### 2.2 Utilisateurs cibles

- Soigneurs et vétérinaires (suivi du bien-être animal)
- Chercheurs (analyse comportementale)
- Personnel de zoo en charge des enclos / environnements

### 2.3 Périmètre (Ce que nous livrons)

Nous nous concentrons sur une **pipeline embarquée de bout en bout**, pragmatique, conçue comme preuve de concept :

- Acquisition de frames thermiques via une caméra thermique low-cost (Topdon TC001).
- Détection de mouvement et accumulation d'activité dans des **cartes de chaleur**.
- Détection d'animaux avec un détecteur **fine-tuné pour l'imagerie thermique** (YOLOv10s).
- Suivi optionnel des détections dans le temps.
- Stockage des résultats (détections, trajectoires, heatmaps, métadonnées) pour analyse ultérieure.

### 2.4 Hors périmètre (état actuel)

Pour garder le projet réaliste au regard des contraintes de temps et de matériel, nous ne visons pas (pour l'instant) :
- une classification comportementale riche (dormir/manger/jouer),
- un déploiement terrain à grande échelle (multiples enclos sur longue durée),
- une infrastructure cloud / MLOps complète.

---

## 3. Espace de solutions et choix

Cette section documente le **raisonnement d'ingénierie** : nous avons explorés différentes options techniques et retenu un design basé sur les contraintes (modalité thermique + calcul embarqué + interprétabilité).

### 3.1 Contraintes majeures

1. **Contexte nocturne** → fonctionner sans lumière visible.
2. **Contraintes embarquées** → calcul limité, puissance limitée.
3. **Interprétabilité** → le personnel a besoin de synthèses visuelles simples (heatmaps).
4. **Rareté des données** → peu de datasets thermiques annotés vs RGB.

### 3.2 Options de capteurs envisagées

- **Caméra RGB** : rejetée comme capteur principal la nuit (problèmes d'éclairage).
- **Caméra thermique** : retenue pour la robustesse nocturne, mais il faut gérer :
  - la faible résolution,
  - la faible texture,
  - les artefacts thermiques de fond (dépendance à ΔT).

### 3.3 Options de calcul embarqué

Nous avons évalué trois familles d'architectures embarquées :

- **CPU seul (Raspberry Pi)**  
  Avantages : simple, faible coût  
  Inconvénients : inférence deep learning souvent trop lente (goulot mémoire).

- **Accélérateur NPU (Raspberry Pi + Hailo-8L)**  
  Avantages : excellentes perfs par watt, adapté sur batterie  
  Inconvénients : quantification INT8 et outils spécifiques.

- **GPU embarqué (famille NVIDIA Jetson)**  
  Avantages : flexible, FP16/FP32, écosystème TensorRT  
  Inconvénients : consommation plus élevée et contraintes thermiques.

**Logique de décision :** pour un système autonome/sur batterie, les NPU sont attractifs ; pour l'expérimentation rapide et la flexibilité, Jetson est un bon baseline. Le benchmarking reste donc central pour quantifier le compromis vitesse/énergie.

### 3.4 Options d'architecture algorithmique

#### A) Heatmap-first (mouvement → accumulation) vs détection seule
- La détection seule peut rater des motifs de mouvement ou être instable au bruit thermique.
- Heatmap-first apporte :
  - un “a priori d'activité” robuste,
  - des sorties interprétables,
  - une façon de restreindre la zone d'intérêt pour détection/suivi.

**Choix :** heatmap-first est la colonne vertébrale du système.

#### B) Pré-traitement lourd (super-résolution / reconstruction) vs filtrage léger
L'état de l'art thermique inclut super-résolution et reconstruction profonde, mais c'est coûteux pour l'embarqué.
**Choix :** garder l'inférence embarquée légère ; si augmentation avancée, la réserver à l'entraînement offline.



**Choix :** privilégier un suivi léger, surtout pour les contraintes embarquées.

---

## 4. Positionnement vs travaux existants (État de l'art)

Nous avons structuré l'état de l'art autour de trois piliers : (1) matériel Edge AI embarqué, (2) contraintes de l'imagerie thermique, (3) suivi + analyse comportementale.

### 4.1 Edge AI embarqué pour détection temps réel

Le passage du cloud vers l'edge est motivé par la latence, la bande passante et l'autonomie. Les cartes embarquées CPU deviennent vite limitées par les transferts mémoire lors de l'inférence CNN. Les accélérateurs dédiés (NPU) et GPU embarqués améliorent le parallélisme et la localité, mais imposent des contraintes (quantification, design thermique/énergie).

### 4.2 Imagerie thermique : défis et solutions

L'imagerie thermique est robuste dans l'obscurité, mais les capteurs low-cost souffrent de :
- texture et contraste réduits,
- fonds bruités et artefacts thermiques,
- forte dépendance à la différence de température cible/fond (ΔT),
- difficulté à détecter des cibles petites/éloignées.

La littérature propose des réseaux de super-résolution et de reconstruction (gains de précision mais calcul lourd), ainsi que de l'augmentation et de l'adaptation de domaine (ex. CycleGAN) pour pallier la rareté des annotations.

### 4.3 Suivi et analyse comportementale

Le suivi multi-objets est nécessaire pour reconstruire les trajectoires et calculer des indicateurs comportementaux.
- Les trackers basés apparence gèrent mieux les occlusions mais sont coûteux.
- Les trackers basés mouvement/géométrie sont plus légers et adaptés à l'embarqué.

Pour le suivi en zoo, certains travaux ciblent la classification des comportements, mais une alternative est **l'analyse spatiale via heatmaps** et grilles d'occupation, fournissant des informations intuitives (zones de repos vs activité) sans modèle sémantique lourd.

### 4.4 Gap de recherche

La plupart des approches existantes optimisent soit :
- la précision de détection avec un calcul lourd,
- soit l'analyse comportementale sans fortes contraintes d'autonomie embarquée.

Notre positionnement : **une pipeline embarquée de bout en bout** combinant :
- accumulation de heatmap basée mouvement,
- détection d'objets adaptée au thermique (YOLO fine-tune),
- suivi léger,
- stockage local des résultats pour analyse ultérieure.

---

## 5. Travail réalisé : produit et processus

Cette section répond : **ce que nous avons construit** et **comment nous l'avons construit**.

### 5.1 Produit : pipeline de bout en bout

#### 5.1.1 Couche acquisition (caméra thermique)
- Capteur : **Topdon TC001**
- Sortie : frames thermiques streamées vers l'unité de traitement.
- Difficulté clé : faible texture des frames thermiques, nécessite normalisation/traitement soigné.

#### 5.1.2 Détection de mouvement + accumulation heatmap
Nous avons implémenté une segmentation de mouvement légère (CV classique) :
- soustraction de fond (ex. MOG2),
- filtrage morphologique pour réduire le bruit,
- accumulation temporelle pour construire une heatmap de densité d'activité sur une fenêtre choisie.

Pourquoi ce design :
- coût de calcul faible sur CPU embarqué,
- indicateur robuste même si la détection échoue par intermittence,
- directement interprétable pour le personnel.

#### 5.1.3 Détection : YOLOv10s fine-tune pour l'imagerie thermique
Nous avons utilisé un détecteur de la famille YOLO et adapté aux frames thermiques :
- préparation du dataset (frames thermiques + annotations),
- stratégie de fine-tuning pour réduire l'écart de domaine RGB→thermique,
- plan d'évaluation basé sur des métriques standard (précision/recall/mAP) et des checks qualitatifs.

Pourquoi ce design :
- la détection fournit la localisation nécessaire au suivi,
- le fine-tuning est indispensable pour éviter une mauvaise généralisation depuis des poids RGB.

#### 5.1.4 Suivi (optionnel / selon contraintes)
Nous planifions/implémentons un suivi léger :
- association des détections frame à frame,
- génération de trajectoires,
- calcul d'occupation spatiale pour enrichir la heatmap.

#### 5.1.5 Sorties et stockage
La pipeline stocke :
- détections (par frame),
- trajectoires (si activé),
- heatmaps (par période),
- métadonnées (timestamps, config device, version modèle),
afin que l'analyse puisse se faire plus tard sans retraiter la vidéo brute.

### 5.2 Processus : approche d'ingénierie

Nous avons suivi une approche itérative :
1. **Preuve de concept détection de mouvement + heatmap** (feedback rapide, risque faible).
2. **Intégration du modèle de détection** (fine-tuning + tests d'inférence sur échantillons thermiques).
3. **Benchmark des options matérielles** pour valider la faisabilité sous contraintes.
4. **Ajout du suivi et des métriques comportementales** une fois la détection stable.

Cet ordre réduit le risque :
- la heatmap produit déjà un résultat utile avant que la détection soit parfaite,
- détection et suivi peuvent être améliorés progressivement.

### 5.3 Méthodes explorées (y compris archive_v1) et raisons d'évolution

Nous documentons toutes les méthodes tentées, avec les raisons pour lesquelles certaines n'ont pas été retenues dans la pipeline finale.

#### A) Sous-ensemble Mini-COCO animaux (archive_v1)
- **Méthode :** filtrer les annotations COCO sur 10 classes animales et générer les labels YOLO ; télécharger un nombre limité d'images.
- **Pourquoi cela n'a pas bien fonctionné :** COCO est RGB, pas thermique. L'écart de domaine vers les frames thermiques réelles a dégradé le transfert, et le sous-ensemble était trop petit pour bien généraliser.

#### B) Simulation RGB → thermique (archive_v1)
- **Méthode :** downscale 256x192, grayscale, réduction de texture (blur), jitter luminosité/contraste, bruit thermique, colormap JET.
- **Pourquoi cela n'a pas bien fonctionné :** l'apparence thermique simulée ne reproduisait pas les artefacts du capteur ni le contraste dépendant de la température, donc le détecteur sur-apprenait des motifs synthétiques.

#### C) Augmentation de données thermiques (archive_v1)
- **Méthode :** flip, crop/resize, blur, ajout de bruit pour étendre le dataset synthétique.
- **Pourquoi cela n'a pas bien fonctionné :** l'augmentation augmente la variété mais ne comble pas le gap de domaine et reste différente du bruit thermique réel.

#### D) Pipeline YOLOv5 en notebooks (archive_v1)
- **Méthode :** entraînement de bout en bout via notebooks avec le dataset synthétique.
- **Pourquoi cela n'a pas bien fonctionné :** résultats peu reproductibles pour une pipeline stable et performances insuffisantes sur frames thermiques réelles.

#### E) Acquisition de dataset via Roboflow (actuel)
- **Méthode :** le script d'entraînement télécharge un dataset thermique curate depuis Roboflow au format YOLOv8.
- **Pourquoi cela marche mieux :** données thermiques réelles réduisent l'écart de domaine, et la pipeline garantit la reproductibilité.

#### F) Entraînement YOLO Ultralytics (actuel)
- **Méthode :** entraîner avec résolution thermique (256x192), epochs/patience ajustés, et sorties structurées.
- **Pourquoi cela marche mieux :** correspond à la résolution capteur et s'appuie sur une API d'entraînement stable.

#### G) Traitement motion heatmap-first (actuel)
- **Méthode :** soustraction de fond MOG2, nettoyage morphologique, accumulation du mouvement en heatmap.
- **Pourquoi cela marche mieux :** fournit un signal interprétable et peu coûteux, même quand la détection n'est pas parfaite.

#### H) Inférence embarquée avec Hailo (actuel)
- **Méthode :** pipeline GStreamer avec post-traitement Hailo, streaming live, et inférence optionnellement gâtée par le mouvement.
- **Pourquoi cela marche mieux :** inférence temps réel sous contraintes embarquées, avec une trajectoire vers un déploiement énergie-efficace.

---

## 6. Résultats actuels et plan d'évaluation

### 6.1 Validation preuve de concept (actuel)

Points valides à ce jour :
- génération de heatmap fonctionnelle en conditions intérieures contrôlées,
- pipeline de bout en bout stable,
- intégration du modèle de détection fonctionnelle (évaluation en cours).

### 6.2 Plan d'évaluation (ce que nous mesurons)

Nous évaluons selon trois axes :

1. **Métriques vision**
  - Précision / Recall / mAP (détection)
  - analyse qualitative des erreurs (faux positifs dus au bruit thermique, petits animaux manqués)

2. **Métriques système**
  - latence / FPS de bout en bout
  - utilisation CPU/NPU/GPU
  - consommation (si mesurable)

3. **Métriques comportementales proxy**
  - occupation spatiale
  - temps passé par zone
  - intensité d'activité dans le temps

---

## 7. Limites et suites à donner

### 7.1 Limites connues

- taille et diversité du dataset (le thermique est gourmand en données),
- sensibilité aux artefacts thermiques de fond (ΔT, surfaces chauffées),
- détection de petites cibles/distant reste difficile,
- les occlusions dégradent la persistance d'identité avec un suivi léger.

### 7.2 Prochaines étapes

- étendre le dataset thermique et la qualité des annotations,
- améliorer la stratégie d'entraînement (augmentations, adaptation de domaine si pertinent),
- finaliser le benchmark matériel et choisir la meilleure cible de déploiement,
- valider sur des configurations proches du zoo et des enregistrements plus longs.

---

## 8. Bibliographie / Références

> La bibliographie complète est suivie dans `references/` (BibTeX/Markdown) et citée dans la documentation.

Liste initiale de références (à compléter) :
- Benchmark Edge AI pour modèles de détection (contraintes embarquées)
- Détection faune thermique et dépendance à ΔT
- Super-résolution pour détection de petits objets
- CycleGAN pour l'adaptation de domaine
- Comparatifs de trackers MOT (DeepSORT vs SORT/ByteTrack)
- Études de suivi du bien-être en zoo et analyse comportementale via heatmaps

---

## 9. Annexes

Ce dépôt inclut (ou inclura) :

- `src/` — code source  
- `configs/` — fichiers de configuration (caméra, modèle, seuils)  
- `models/` — poids/exports des modèles (si autorisé)  
- `experiments/` — scripts, logs, sorties de benchmark  
- `assets/` — images (schémas de pipeline, exemples de heatmaps)  
- `references/` — fichiers de bibliographie  

Si des datasets/modèles externes ne sont pas commits, nous fournissons :
- instructions de téléchargement,
- versioning,
- checksums,
- et commandes de reproductibilité.
