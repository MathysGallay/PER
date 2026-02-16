import time
import csv
import psutil
import subprocess
import os
import numpy as np
import torch
from datetime import datetime
from ultralytics import YOLO

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Recherche du modèle
path_local = os.path.join(SCRIPT_DIR, "data", "best.pt")
path_parent = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "data", "best.pt"))

if os.path.exists(path_local):
    MODEL_PATH = path_local
elif os.path.exists(path_parent):
    MODEL_PATH = path_parent
else:
    MODEL_PATH = "yolov8n.pt"

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results", "pc")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "charge.csv")

WARMUP_SEC = 5     
TEST_DURATION = 60 

def get_nvidia_metrics():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
            encoding='utf-8'
        )
        temp, power = out.strip().split(',')
        return float(temp), float(power)
    except:
        return 0.0, 0.0

def generate_synthetic_image(size=640):
    """Génère une image synthétique RGB pour les tests"""
    # Créer une image avec des patterns variés pour simuler une vraie scène
    img = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    
    # Ajouter quelques formes pour avoir du contenu détectable
    img[100:200, 100:300] = [255, 0, 0]  # Rectangle rouge
    img[300:450, 200:400] = [0, 255, 0]  # Rectangle vert
    img[400:500, 400:550] = [0, 0, 255]  # Rectangle bleu
    
    return img

def run_benchmark():
    print(f"--- MODE CHARGE V2 (PC NVIDIA) - IMAGE SYNTHETIQUE ---")
    
    # Nettoyage préventif du fichier CSV
    if os.path.exists(OUTPUT_FILE):
        try: os.remove(OUTPUT_FILE)
        except: pass
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"🔍 Modèle : {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
    except:
        print("⚠️ Erreur chargement modèle, téléchargement yolov8n.pt...")
        model = YOLO("yolov8n.pt")
    
    # IMPORTANT : Déplacer le modèle sur GPU explicitement
    model.model.to('cuda:0')
    model.model.eval()

    # Générer une image synthétique
    print("🎨 Génération d'une image synthétique (640x640)...")
    synthetic_image = generate_synthetic_image(640)
    
    # Convertir en tensor PyTorch (format YOLO)
    # Approche directe : créer le tensor sans passer par numpy intermediate
    img_rgb = synthetic_image[:, :, ::-1].copy()  # BGR to RGB
    img_float = img_rgb.astype(np.float32) / 255.0
    img_chw = img_float.transpose(2, 0, 1)  # HWC to CHW
    
    # Créer le tensor directement avec torch.tensor() au lieu de from_numpy()
    img_tensor = torch.tensor(img_chw, dtype=torch.float32, device='cuda:0').unsqueeze(0)
    
    print("⚡ Préchauffe du GPU...")

    headers = ["Timestamp", "Frame_ID", "Latence_ms", "FPS_Inst", "CPU_%", "RAM_%", "Temp_C", "Power_Soft_W", "Phase", "Power_Manual_W"]
    
    with open(OUTPUT_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        
        start_global = time.time()
        frame_count = 0
        phase = "WARMUP"
        
        print(f"▶️ Lancement du test ({TEST_DURATION}s)...")
        
        while True:
            elapsed = time.time() - start_global
            
            if elapsed < WARMUP_SEC: 
                phase = "WARMUP"
            elif elapsed < (WARMUP_SEC + TEST_DURATION): 
                phase = "TEST"
            else: 
                break

            # --- INFERENCE DIRECTE ---
            t0 = time.time()
            with torch.no_grad():
                # Inférence directe sur le modèle PyTorch
                results = model.model(img_tensor)
            torch.cuda.synchronize()  # Attendre la fin de l'inférence GPU
            t1 = time.time()

            latence_ms = (t1 - t0) * 1000
            fps_inst = 1.0 / (t1 - t0) if (t1 - t0) > 0 else 0
            frame_count += 1

            if frame_count % 10 == 0:
                cpu = psutil.cpu_percent()
                ram = psutil.virtual_memory().percent
                timestamp = datetime.now().strftime("%H:%M:%S")
                
                temp, power_gpu = get_nvidia_metrics()
                
                writer.writerow([timestamp, frame_count, round(latence_ms, 2), round(fps_inst, 2), cpu, ram, temp, power_gpu, phase, 0.0])

                print(f"\r[{phase}] Frame: {frame_count} | FPS: {fps_inst:.1f} | Latence: {latence_ms:.1f}ms | GPU: {temp}°C | Pwr: {power_gpu}W   ", end="")

    print(f"\n\n✅ Terminé ! Résultats : {OUTPUT_FILE}")
    print(f"📊 Total frames traités : {frame_count}")

if __name__ == "__main__":
    run_benchmark()
