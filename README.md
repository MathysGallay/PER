# PER - Projet d'Étude et Recherche

Projet de détection d'objets avec YOLO sur images thermiques.

## Architecture du Projet

```
PER/
├── README.md                    # Ce fichier
├── PLAN_AVANCEMENT.md          # Plan et suivi du projet
├── data.yaml                   # Configuration YOLO
├── .gitignore                  # Fichiers ignorés par Git
│
├── archive_v1/                 # Travail de la phase 1
│   ├── scripts/               # Scripts Python v1
│   ├── notebooks/             # Notebooks d'expérimentation v1
│   └── data/                  # Anciens datasets
│
├── datasets/                   # Tous les datasets
│   └── PER_ENT/               # Dataset principal actuel
│       ├── images/
│       │   ├── train/
│       │   └── val/
│       ├── labels/
│       │   ├── train/
│       │   └── val/
│       └── annotations/
│
├── benchmarks/                 # Résultats et évaluations
│   ├── metrics/               # Métriques de performance
│   ├── results/               # Résultats bruts
│   └── reports/               # Rapports et visualisations
│
├── src/                        # Code source
│   ├── preprocessing/         # Scripts de prétraitement
│   ├── training/              # Scripts d'entraînement
│   └── utils/                 # Utilitaires
│
├── tests/                      # Tests et évaluations
│   ├── unit/                  # Tests unitaires
│   ├── integration/           # Tests d'intégration
│   └── models/                # Tests de modèles
│
├── experiments/                # Expérimentations en cours
│   └── TEST_PER/              # Vos tests actuels
│
├── notebooks/                  # Notebooks de démonstration
│
├── models/                     # Modèles YOLO
│   └── yolov5/
│
└── runs/                       # Résultats d'entraînement YOLO
```

## Installation

```bash
# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
.\venv\Scripts\activate  # Windows

# Installer les dépendances
pip install torch torchvision opencv-python numpy tqdm
pip install ultralytics  # Pour YOLOv8
```

## Utilisation

### Ajouter des données au dataset PER_ENT

1. Placez vos images dans `datasets/PER_ENT/images/train/` ou `val/`
2. Placez les labels au format YOLO dans `datasets/PER_ENT/labels/train/` ou `val/`
3. Vérifiez que `data.yaml` est à jour

### Entraîner un modèle

```bash
# YOLOv5
python models/yolov5/train.py --data data.yaml --cfg yolov5s.yaml --weights yolov5s.pt

# YOLOv8
yolo train data=data.yaml model=yolov8s.pt epochs=100
```

### Évaluer un modèle

Les résultats sont automatiquement sauvegardés dans `runs/train/` et `runs/val/`.
Pour des benchmarks personnalisés, consultez `benchmarks/README.md`.

## Classes Détectées

1. bird (oiseau)
2. cat (chat)
3. dog (chien)
4. horse (cheval)
5. sheep (mouton)
6. cow (vache)
7. elephant (éléphant)
8. bear (ours)
9. zebra (zèbre)
10. giraffe (girafe)

## Historique

- **Phase 1** (archive_v1) : Expérimentation initiale avec conversion RGB→Thermique
- **Phase 2** (actuelle) : Dataset PER_ENT et benchmarks structurés

## Contact

Voir `PLAN_AVANCEMENT.md` pour plus de détails sur l'avancement du projet.
