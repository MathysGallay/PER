# Archive v1

Ce dossier contient tout le travail effectué dans la première phase du projet.

## Contenu

### scripts/
Scripts Python utilisés pour :
- `image_thermal_converter.py` : Conversion d'images RGB en images thermiques simulées
- `download_mini_coco_animals.py` : Téléchargement du dataset mini COCO
- `thermal_augmentation.py` : Augmentation de données thermiques

### notebooks/
Notebooks Jupyter d'expérimentation :
- `per-entrainement (1).ipynb` : Premier notebook d'entraînement
- `pipeline_yolov5_restructured.ipynb` : Pipeline YOLOv5 restructuré
- `pipeline_yolov5.ipynb` : Pipeline YOLOv5 original

### data/
Datasets utilisés :
- `mini_coco/` : Subset du dataset COCO
- `thermal_coco/` : Images thermiques générées
- `thermal_yolo/` : Dataset formaté pour YOLO
- `annotations/` : Annotations COCO originales

## Utilisation

Ces fichiers sont conservés à titre de référence. Pour le développement actuel, utilisez les dossiers à la racine du projet :
- `src/` pour les nouveaux scripts
- `notebooks/` pour les nouveaux notebooks
- `datasets/PER_ENT/` pour le nouveau dataset
