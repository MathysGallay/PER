# Source Code

Ce dossier contient tout le code source organisé par fonctionnalité.

## Structure

### preprocessing/
Scripts de prétraitement des données :
- Conversion d'images
- Augmentation de données
- Formatage pour YOLO
- Génération d'images thermiques

### training/
Scripts d'entraînement :
- Pipelines d'entraînement YOLO
- Configuration d'hyperparamètres
- Callbacks personnalisés

### utils/
Utilitaires et fonctions communes :
- Fonctions de visualisation
- Métriques personnalisées
- Gestion de fichiers
- Helpers divers

## Conventions de Code

- Noms de fichiers : `snake_case.py`
- Noms de classes : `PascalCase`
- Noms de fonctions : `snake_case()`
- Documentation : Docstrings Google Style

## Exemple d'utilisation

```python
from src.preprocessing import thermal_converter
from src.utils import visualize

# Convertir une image en thermique
thermal_img = thermal_converter.convert(rgb_image)

# Visualiser
visualize.show_comparison(rgb_image, thermal_img)
```
