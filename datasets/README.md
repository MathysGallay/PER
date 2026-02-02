# Datasets

Ce dossier contient tous les datasets utilisés dans le projet PER.

## Structure

### PER_ENT
Nouveau dataset principal pour l'entraînement
```
PER_ENT/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── annotations/
```

## Utilisation

Pour ajouter vos images au dataset PER_ENT :
1. Placez les images dans `PER_ENT/images/train/` ou `PER_ENT/images/val/`
2. Placez les labels YOLO correspondants dans `PER_ENT/labels/train/` ou `PER_ENT/labels/val/`
3. Mettez à jour `data.yaml` à la racine si nécessaire

## Anciens datasets

Les datasets précédents (mini_coco, thermal_coco, thermal_yolo) sont archivés dans `archive_v1/data/`
