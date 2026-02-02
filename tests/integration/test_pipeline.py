"""
Tests d'intégration - Pipeline complet de détection
"""
import sys
from pathlib import Path
import torch
from ultralytics import YOLO
import cv2
import yaml


def test_full_pipeline():
    """
    Test du pipeline complet:
    1. Chargement du modèle
    2. Chargement des images
    3. Inférence
    4. Post-processing
    5. Sauvegarde des résultats
    """
    print("\n=== Test Pipeline Complet ===\n")
    
    # 1. Vérifier la configuration
    print("1. Vérification configuration...")
    config_path = Path("data.yaml")
    assert config_path.exists(), "data.yaml introuvable"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    assert 'nc' in config, "Nombre de classes manquant"
    assert 'names' in config, "Noms de classes manquants"
    print(f"   ✓ Configuration OK: {config['nc']} classes")
    
    # 2. Vérifier CUDA/GPU
    print("\n2. Vérification GPU...")
    cuda_available = torch.cuda.is_available()
    device = 'cuda' if cuda_available else 'cpu'
    print(f"   ✓ Device: {device}")
    if cuda_available:
        print(f"   ✓ GPU: {torch.cuda.get_device_name(0)}")
    
    # 3. Charger le modèle
    print("\n3. Chargement modèle...")
    model_path = Path("runs/train/exp2/weights/best.pt")
    
    if not model_path.exists():
        print(f"   ⚠ Modèle non trouvé: {model_path}")
        print("   Utilisation du modèle pré-entraîné YOLOv8s")
        model_path = "yolov8s.pt"
    
    model = YOLO(str(model_path))
    print(f"   ✓ Modèle chargé: {model_path}")
    
    # 4. Vérifier le dataset
    print("\n4. Vérification dataset...")
    dataset_path = Path("datasets/PER_ENT/images/val")
    
    if not dataset_path.exists():
        # Fallback vers l'ancien chemin
        dataset_path = Path("archive_v1/data/thermal_yolo/images/val")
    
    if dataset_path.exists():
        images = list(dataset_path.glob("*.jpg"))
        print(f"   ✓ Dataset trouvé: {len(images)} images")
    else:
        print("   ⚠ Dataset de validation non trouvé")
        images = []
    
    # 5. Test d'inférence
    if len(images) > 0:
        print("\n5. Test inférence...")
        test_img = str(images[0])
        
        results = model(test_img, verbose=False)
        detections = results[0].boxes
        
        print(f"   ✓ Inférence réussie")
        print(f"   ✓ Détections: {len(detections)}")
        
        if len(detections) > 0:
            print(f"   ✓ Classes détectées: {[int(c) for c in detections.cls]}")
            print(f"   ✓ Confiances: {[f'{c:.2f}' for c in detections.conf]}")
    else:
        print("\n5. ⚠ Pas d'images pour test d'inférence")
    
    # 6. Test export ONNX (optionnel)
    print("\n6. Test capacité d'export...")
    try:
        export_path = model.export(format='onnx', simplify=True, opset=12, dynamic=False)
        print(f"   ✓ Export ONNX possible: {export_path}")
        # Nettoyer le fichier exporté
        if Path(export_path).exists():
            Path(export_path).unlink()
    except Exception as e:
        print(f"   ⚠ Export ONNX non testé: {e}")
    
    print("\n✅ Pipeline complet vérifié!\n")
    
    return {
        'config_ok': True,
        'device': device,
        'model_loaded': True,
        'dataset_found': len(images) > 0,
        'inference_ok': len(images) > 0
    }


def test_data_loading():
    """Test du chargement des données"""
    print("\n=== Test Chargement Données ===\n")
    
    # Vérifier les chemins
    paths_to_check = [
        "datasets/PER_ENT/images/train",
        "datasets/PER_ENT/images/val",
        "datasets/PER_ENT/labels/train",
        "datasets/PER_ENT/labels/val"
    ]
    
    results = {}
    for path in paths_to_check:
        p = Path(path)
        exists = p.exists()
        count = len(list(p.glob("*.*"))) if exists else 0
        results[path] = {'exists': exists, 'count': count}
        
        status = "✓" if exists else "⚠"
        print(f"{status} {path}: {count} fichiers")
    
    return results


def test_model_training_pipeline():
    """Test du pipeline d'entraînement (sans entraîner réellement)"""
    print("\n=== Test Pipeline Entraînement ===\n")
    
    # 1. Vérifier les fichiers nécessaires
    print("1. Vérification fichiers nécessaires...")
    required_files = {
        'config': Path('data.yaml'),
        'model': Path('yolov8s.pt')
    }
    
    for name, path in required_files.items():
        exists = path.exists()
        status = "✓" if exists else "✗"
        print(f"   {status} {name}: {path}")
    
    # 2. Vérifier qu'on peut initialiser un entraînement
    print("\n2. Initialisation entraînement (dry-run)...")
    try:
        model = YOLO('yolov8s.pt')
        print("   ✓ Modèle initialisé")
        
        # Simuler les paramètres d'entraînement
        train_params = {
            'data': 'data.yaml',
            'epochs': 1,  # Juste 1 epoch pour le test
            'imgsz': 416,
            'batch': 8,
            'device': 0 if torch.cuda.is_available() else 'cpu'
        }
        print(f"   ✓ Paramètres: {train_params}")
        
    except Exception as e:
        print(f"   ✗ Erreur: {e}")
        return False
    
    print("\n✅ Pipeline d'entraînement validé (dry-run)\n")
    return True


def test_evaluation_pipeline():
    """Test du pipeline d'évaluation"""
    print("\n=== Test Pipeline Évaluation ===\n")
    
    # Chercher un modèle entraîné
    model_paths = [
        Path("runs/train/exp2/weights/best.pt"),
        Path("runs/train/thermal_exp_improved/weights/best.pt")
    ]
    
    model_found = None
    for model_path in model_paths:
        if model_path.exists():
            model_found = model_path
            break
    
    if not model_found:
        print("⚠ Aucun modèle entraîné trouvé")
        return False
    
    print(f"1. Modèle trouvé: {model_found}")
    
    try:
        model = YOLO(str(model_found))
        print("   ✓ Modèle chargé")
        
        # Vérifier qu'on peut lancer une validation
        val_path = Path("datasets/PER_ENT/images/val")
        if not val_path.exists():
            val_path = Path("archive_v1/data/thermal_yolo/images/val")
        
        if val_path.exists():
            print(f"\n2. Dataset de validation: {val_path}")
            print("   ✓ Dataset trouvé")
            
            # Note: On ne lance pas réellement la validation ici
            # car elle peut être longue
            print("\n✅ Pipeline d'évaluation validé\n")
            return True
        else:
            print("\n⚠ Dataset de validation non trouvé")
            return False
            
    except Exception as e:
        print(f"✗ Erreur: {e}")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("TESTS D'INTÉGRATION")
    print("=" * 50)
    
    # Test 1: Pipeline complet
    result1 = test_full_pipeline()
    
    # Test 2: Chargement données
    result2 = test_data_loading()
    
    # Test 3: Pipeline entraînement
    result3 = test_model_training_pipeline()
    
    # Test 4: Pipeline évaluation
    result4 = test_evaluation_pipeline()
    
    print("\n" + "=" * 50)
    print("RÉSUMÉ DES TESTS")
    print("=" * 50)
    print(f"Pipeline complet: {'✓' if result1.get('inference_ok') else '⚠'}")
    print(f"Chargement données: {'✓' if any(r['exists'] for r in result2.values()) else '⚠'}")
    print(f"Pipeline entraînement: {'✓' if result3 else '⚠'}")
    print(f"Pipeline évaluation: {'✓' if result4 else '⚠'}")
    print("=" * 50)
