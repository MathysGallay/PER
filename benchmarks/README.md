# Benchmarks

Ce dossier contient tous les résultats de benchmarks et comparaisons de performance.

## Structure

```
benchmarks/
├── metrics/      # Fichiers de métriques (mAP, précision, recall, etc.)
├── results/      # Résultats bruts des évaluations
└── reports/      # Rapports et visualisations
```

## Types de benchmarks

### Comparaison de modèles
- YOLOv5 vs YOLOv8
- Différentes tailles de modèles (s, m, l, x)

### Comparaison de datasets
- RGB vs Thermal
- Différentes augmentations

### Métriques suivies
- mAP@0.5
- mAP@0.5:0.95
- Précision par classe
- Recall par classe
- Temps d'inférence
- FPS

## Utilisation

```python
# Exemple de sauvegarde de résultats
import json

results = {
    "model": "yolov5s",
    "dataset": "PER_ENT",
    "mAP50": 0.85,
    "mAP50_95": 0.62
}

with open("benchmarks/results/yolov5s_per_ent.json", "w") as f:
    json.dump(results, f, indent=4)
```
