from marshal import version
import os
import random
from sympy import rf
from ultralytics import YOLO
from roboflow import Roboflow
import torch
import time

# --- CONFIGURATION ---

# Dimensions de la caméra Topdon
THERMAL_W = 256
THERMAL_H = 192

API_KEY = "kXOZ64haNNa94hFBQfxg"
TEST_MODE = True        # True pour ton test (5 imgs), False pour ton pote
DOWNLOAD_DATASET = True  # True pour télécharger le dataset depuis Roboflow, False pour utiliser le dataset local



                

# 1. Télécharger le dataset depuis Roboflow si nécessaire
def download_dataset(api_key):
    if DOWNLOAD_DATASET:
        print("--- Téléchargement du dataset 'location' depuis Roboflow ---")
        
        # Créer le dossier datasets s'il n'existe pas
        datasets_dir = os.path.join(os.getcwd(), "datasets")
        os.makedirs(datasets_dir, exist_ok=True)
        
        rf = Roboflow(api_key=api_key)
        project = rf.workspace("yanis-rqbs8").project("thermal-image-c4rau-9mvvo")
        version = project.version(2)
        
        # Télécharger dans le dossier datasets
        dataset = version.download("yolov8", location=datasets_dir)
        
        DATASET_PATH = dataset.location
        print(f"Dataset téléchargé dans : {DATASET_PATH}")
    else:
        print("--- Utilisation du dataset local ---")
        DATASET_PATH = os.path.join(os.getcwd(), "datasets", "Therm_animal-1")
    return DATASET_PATH

# 3. Entraînement
model = YOLO('yolov10s.pt')

if __name__ == '__main__':
    # Select device: use CPU in TEST_MODE or when no CUDA is available
    device_arg = 'cpu' if TEST_MODE else ('0' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'cpu')
    print(f"Selected device: {device_arg} | torch.cuda.is_available(): {torch.cuda.is_available()}")
    dataset_path = download_dataset(API_KEY)
    start = time.time()
    
    model.train(
        data=os.path.join(dataset_path, "data.yaml"),
        epochs= 1 if TEST_MODE else 120,
        imgsz=(THERMAL_W, THERMAL_H),  
        batch=16,
        device=device_arg,
        patience=20,
        save_period=1,
        project='experiments/PER_ENT',  # Résultats dans experiments
        name='v1_reduit'
    )
    end = time.time()
    print(f"Training completed for 120 epochs in {end - start:.2f} seconds.")