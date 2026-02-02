"""
Tests unitaires pour les fonctions de preprocessing et utilitaires
"""
import pytest
import numpy as np
import cv2
from pathlib import Path


def test_image_loading():
    """Test du chargement d'image"""
    # Créer une image de test
    test_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    test_path = "test_temp.jpg"
    cv2.imwrite(test_path, test_img)
    
    # Charger l'image
    loaded_img = cv2.imread(test_path)
    
    # Vérifications
    assert loaded_img is not None, "L'image n'a pas été chargée"
    assert loaded_img.shape == test_img.shape, "Les dimensions ne correspondent pas"
    
    # Nettoyage
    Path(test_path).unlink()
    print("✓ Test chargement d'image réussi")


def test_image_resize():
    """Test du redimensionnement d'image"""
    # Image de test
    img = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
    target_size = (416, 416)
    
    # Redimensionner
    resized = cv2.resize(img, target_size)
    
    # Vérifications
    assert resized.shape[:2] == target_size, f"Taille incorrecte: {resized.shape[:2]}"
    assert resized.dtype == np.uint8, "Type de données incorrect"
    print("✓ Test redimensionnement réussi")


def test_bbox_format_conversion():
    """Test de conversion de format de bounding box"""
    # Format YOLO: [class, x_center, y_center, width, height] (normalisé)
    # Format Pascal VOC: [xmin, ymin, xmax, ymax] (pixels)
    
    img_width, img_height = 640, 480
    
    # YOLO bbox
    yolo_bbox = [0, 0.5, 0.5, 0.3, 0.4]  # [class, x_c, y_c, w, h]
    
    # Conversion YOLO -> Pascal VOC
    cls, x_c, y_c, w, h = yolo_bbox
    xmin = int((x_c - w/2) * img_width)
    ymin = int((y_c - h/2) * img_height)
    xmax = int((x_c + w/2) * img_width)
    ymax = int((y_c + h/2) * img_height)
    
    pascal_bbox = [xmin, ymin, xmax, ymax]
    
    # Vérifications
    assert xmin >= 0 and xmin < img_width, "xmin hors limites"
    assert ymin >= 0 and ymin < img_height, "ymin hors limites"
    assert xmax > xmin, "xmax doit être > xmin"
    assert ymax > ymin, "ymax doit être > ymin"
    
    print(f"✓ Conversion bbox réussie: YOLO {yolo_bbox[1:]} -> Pascal {pascal_bbox}")


def test_iou_calculation():
    """Test du calcul d'Intersection over Union"""
    # Deux bounding boxes qui se chevauchent
    box1 = [100, 100, 200, 200]  # [xmin, ymin, xmax, ymax]
    box2 = [150, 150, 250, 250]
    
    # Calcul IoU
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    iou = intersection / union if union > 0 else 0
    
    # Vérifications
    assert 0 <= iou <= 1, "IoU doit être entre 0 et 1"
    assert iou > 0, "Ces boxes devraient se chevaucher"
    
    print(f"✓ Calcul IoU réussi: {iou:.3f}")


def test_confidence_filtering():
    """Test du filtrage par confiance"""
    # Détections simulées avec scores de confiance
    detections = [
        {'class': 0, 'conf': 0.9, 'bbox': [100, 100, 200, 200]},
        {'class': 1, 'conf': 0.3, 'bbox': [300, 300, 400, 400]},
        {'class': 2, 'conf': 0.6, 'bbox': [500, 500, 600, 600]},
        {'class': 0, 'conf': 0.2, 'bbox': [700, 700, 800, 800]}
    ]
    
    threshold = 0.5
    
    # Filtrage
    filtered = [d for d in detections if d['conf'] >= threshold]
    
    # Vérifications
    assert len(filtered) == 2, f"Devrait avoir 2 détections, obtenu {len(filtered)}"
    assert all(d['conf'] >= threshold for d in filtered), "Certaines détections sous le seuil"
    
    print(f"✓ Filtrage confiance réussi: {len(detections)} -> {len(filtered)} détections")


def test_class_mapping():
    """Test du mapping de classes"""
    # Mapping COCO -> Classes du projet
    coco_to_project = {
        14: 0,  # bird
        15: 1,  # cat
        16: 2,  # dog
        17: 3,  # horse
        18: 4,  # sheep
        19: 5,  # cow
        20: 6,  # elephant
        21: 7,  # bear
        22: 8,  # zebra
        23: 9   # giraffe
    }
    
    class_names = ['bird', 'cat', 'dog', 'horse', 'sheep', 
                   'cow', 'elephant', 'bear', 'zebra', 'giraffe']
    
    # Vérifications
    assert len(class_names) == 10, "Devrait avoir 10 classes"
    assert len(coco_to_project) == 10, "Devrait avoir 10 mappings"
    assert class_names[coco_to_project[15]] == 'cat', "Mapping incorrect pour cat"
    
    print("✓ Mapping de classes correct")


def test_data_augmentation():
    """Test des augmentations de données"""
    # Image de test
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    # Test flip horizontal
    flipped = cv2.flip(img, 1)
    assert flipped.shape == img.shape, "Shape modifiée après flip"
    assert not np.array_equal(img, flipped), "L'image devrait être différente"
    
    # Test blur
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    assert blurred.shape == img.shape, "Shape modifiée après blur"
    
    # Test ajout de bruit
    noise = np.random.normal(0, 25, img.shape).astype(np.int16)
    noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    assert noisy.shape == img.shape, "Shape modifiée après bruit"
    
    print("✓ Tests d'augmentation réussis")


if __name__ == "__main__":
    print("=== Exécution des tests unitaires ===\n")
    
    test_image_loading()
    test_image_resize()
    test_bbox_format_conversion()
    test_iou_calculation()
    test_confidence_filtering()
    test_class_mapping()
    test_data_augmentation()
    
    print("\n✅ Tous les tests unitaires sont passés!")
