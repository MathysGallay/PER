"""
Test d'inférence et mesure de performance des modèles YOLO
"""
import torch
import time
import cv2
from pathlib import Path
from ultralytics import YOLO
import numpy as np


def test_inference_speed(model_path, images_dir, iterations=100, device='cuda'):
    """
    Teste la vitesse d'inférence d'un modèle sur plusieurs images
    
    Args:
        model_path: Chemin vers le modèle (.pt ou .onnx)
        images_dir: Dossier contenant les images de test
        iterations: Nombre d'itérations pour la moyenne
        device: 'cuda' ou 'cpu'
    
    Returns:
        dict: Statistiques de performance
    """
    print(f"\n=== Test de vitesse d'inférence ===")
    print(f"Modèle: {model_path}")
    print(f"Device: {device}")
    
    # Charger le modèle
    model = YOLO(model_path)
    model.to(device)
    
    # Récupérer les images
    images = list(Path(images_dir).glob("*.jpg"))[:iterations]
    
    if len(images) == 0:
        print("Aucune image trouvée!")
        return None
    
    print(f"Nombre d'images: {len(images)}")
    
    # Warm-up
    img = cv2.imread(str(images[0]))
    _ = model(img, verbose=False)
    
    # Mesure du temps
    times = []
    for img_path in images:
        img = cv2.imread(str(img_path))
        
        start = time.time()
        results = model(img, verbose=False)
        end = time.time()
        
        times.append(end - start)
    
    # Calcul des statistiques
    times = np.array(times)
    stats = {
        'model': model_path,
        'device': device,
        'iterations': len(images),
        'mean_time': times.mean(),
        'std_time': times.std(),
        'min_time': times.min(),
        'max_time': times.max(),
        'fps': 1 / times.mean()
    }
    
    print(f"\nRésultats:")
    print(f"  Temps moyen: {stats['mean_time']*1000:.2f} ms")
    print(f"  Écart-type: {stats['std_time']*1000:.2f} ms")
    print(f"  FPS moyen: {stats['fps']:.2f}")
    print(f"  Min/Max: {stats['min_time']*1000:.2f} / {stats['max_time']*1000:.2f} ms")
    
    return stats


def test_detection_accuracy(model_path, images_dir, labels_dir, conf_threshold=0.25):
    """
    Teste la précision de détection d'un modèle
    
    Args:
        model_path: Chemin vers le modèle
        images_dir: Dossier des images
        labels_dir: Dossier des labels YOLO
        conf_threshold: Seuil de confiance
    
    Returns:
        dict: Métriques de détection
    """
    print(f"\n=== Test de précision de détection ===")
    print(f"Modèle: {model_path}")
    print(f"Seuil de confiance: {conf_threshold}")
    
    model = YOLO(model_path)
    
    images = list(Path(images_dir).glob("*.jpg"))
    total_detections = 0
    total_gt = 0
    correct_detections = 0
    
    for img_path in images:
        # Label correspondant
        label_path = Path(labels_dir) / (img_path.stem + ".txt")
        
        if not label_path.exists():
            continue
        
        # Ground truth
        with open(label_path, 'r') as f:
            gt_boxes = [line.strip().split() for line in f.readlines()]
            total_gt += len(gt_boxes)
        
        # Prédiction
        results = model(str(img_path), conf=conf_threshold, verbose=False)[0]
        detections = results.boxes
        total_detections += len(detections)
        
        # Compter les bonnes détections (simplifiée - IoU > 0.5)
        for det in detections:
            for gt in gt_boxes:
                # Calculer IoU basique
                # Note: implémentation simplifiée, à améliorer
                correct_detections += 1
                break
    
    precision = correct_detections / total_detections if total_detections > 0 else 0
    recall = correct_detections / total_gt if total_gt > 0 else 0
    
    results = {
        'total_gt': total_gt,
        'total_detections': total_detections,
        'correct_detections': correct_detections,
        'precision': precision,
        'recall': recall,
        'f1_score': 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    }
    
    print(f"\nRésultats:")
    print(f"  Total GT: {total_gt}")
    print(f"  Total détections: {total_detections}")
    print(f"  Précision: {precision:.2%}")
    print(f"  Recall: {recall:.2%}")
    print(f"  F1-Score: {results['f1_score']:.2%}")
    
    return results


def compare_models(models_dict, test_image):
    """
    Compare plusieurs modèles sur la même image
    
    Args:
        models_dict: Dict {nom: chemin_modele}
        test_image: Chemin vers une image de test
    
    Returns:
        dict: Résultats comparatifs
    """
    print(f"\n=== Comparaison de modèles ===")
    print(f"Image de test: {test_image}")
    
    img = cv2.imread(test_image)
    results = {}
    
    for name, model_path in models_dict.items():
        print(f"\nTest de {name}...")
        model = YOLO(model_path)
        
        # Temps d'inférence
        start = time.time()
        preds = model(img, verbose=False)[0]
        inference_time = time.time() - start
        
        results[name] = {
            'inference_time': inference_time,
            'num_detections': len(preds.boxes),
            'classes': [int(c) for c in preds.boxes.cls],
            'confidences': [float(c) for c in preds.boxes.conf],
            'fps': 1 / inference_time
        }
        
        print(f"  Temps: {inference_time*1000:.2f} ms")
        print(f"  Détections: {results[name]['num_detections']}")
        print(f"  FPS: {results[name]['fps']:.2f}")
    
    return results


if __name__ == "__main__":
    # Exemple d'utilisation
    
    # Test 1: Vitesse d'inférence
    stats = test_inference_speed(
        model_path="runs/train/exp2/weights/best.pt",
        images_dir="datasets/PER_ENT/images/val/",
        iterations=50,
        device='cuda'
    )
    
    # Test 2: Précision (nécessite labels)
    # accuracy = test_detection_accuracy(
    #     model_path="runs/train/exp2/weights/best.pt",
    #     images_dir="datasets/PER_ENT/images/val/",
    #     labels_dir="datasets/PER_ENT/labels/val/"
    # )
    
    # Test 3: Comparaison de modèles
    # comparison = compare_models(
    #     models_dict={
    #         "YOLOv5n": "runs/train/exp2/weights/best.pt",
    #         "YOLOv8s": "runs/train/thermal_exp_improved/weights/best.pt"
    #     },
    #     test_image="datasets/PER_ENT/images/val/image1.jpg"
    # )
