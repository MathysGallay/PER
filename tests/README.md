# Tests

Ce dossier contient tous les tests et évaluations du projet.

## Structure

```
tests/
├── unit/           # Tests unitaires (fonctions individuelles)
│   └── test_utils.py
├── integration/    # Tests d'intégration (pipeline complet)
│   └── test_pipeline.py
└── models/         # Tests et évaluations de modèles
    └── test_inference.py
```

## Exécution des tests

### Tests unitaires
```bash
# Tous les tests unitaires
python tests/unit/test_utils.py

# Avec pytest (si installé)
pytest tests/unit/
```

### Tests d'intégration
```bash
# Pipeline complet
python tests/integration/test_pipeline.py
```

### Tests de modèles
```bash
# Test de vitesse d'inférence
python tests/models/test_inference.py
```

## Tests disponibles

### 1. Tests Unitaires (`tests/unit/test_utils.py`)
- ✅ Chargement d'images
- ✅ Redimensionnement
- ✅ Conversion formats de bbox (YOLO ↔ Pascal VOC)
- ✅ Calcul IoU
- ✅ Filtrage par confiance
- ✅ Mapping de classes
- ✅ Augmentations de données

### 2. Tests d'Intégration (`tests/integration/test_pipeline.py`)
- ✅ Vérification configuration complète
- ✅ Détection GPU/CUDA
- ✅ Chargement modèle
- ✅ Vérification dataset
- ✅ Test d'inférence
- ✅ Test export ONNX
- ✅ Pipeline entraînement (dry-run)
- ✅ Pipeline évaluation

### 3. Tests de Modèles (`tests/models/test_inference.py`)

**test_inference_speed()** : Mesure vitesse d'inférence
- Temps moyen/min/max par image
- FPS (frames per second)
- Écart-type
- Comparaison GPU vs CPU

**test_detection_accuracy()** : Mesure précision
- Précision, Recall, F1-Score
- Comparaison avec ground truth
- Métriques par classe

**compare_models()** : Comparaison de modèles
- Temps d'inférence relatif
- Nombre de détections
- Classes et confiances détectées

## Exemples d'utilisation

### Test rapide de performance
```python
from tests.models.test_inference import test_inference_speed

stats = test_inference_speed(
    model_path="runs/train/exp2/weights/best.pt",
    images_dir="datasets/PER_ENT/images/val/",
    iterations=50,
    device='cuda'
)
```

### Comparer deux modèles
```python
from tests.models.test_inference import compare_models

results = compare_models(
    models_dict={
        "YOLOv5n": "runs/train/exp2/weights/best.pt",
        "YOLOv8s": "runs/train/thermal_exp_improved/weights/best.pt"
    },
    test_image="datasets/PER_ENT/images/val/image1.jpg"
)
```

## Installation dépendances
```bash
pip install pytest pytest-cov
```

## Conventions

- Noms de fichiers: `test_*.py`
- Utiliser pytest pour les tests unitaires
- Documenter les résultats dans `benchmarks/`
