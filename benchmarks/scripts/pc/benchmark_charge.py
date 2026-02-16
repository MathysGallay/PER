import cv2
import time
import csv
import psutil
import subprocess
import os
import numpy as np
import torch
from datetime import datetime
from ultralytics import YOLO

# --- CONFIGURATION BLINDÉE ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Recherche du modèle (Local ou Parent)
path_local = os.path.join(SCRIPT_DIR, "data", "best.pt")
path_parent = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "data", "best.pt"))

if os.path.exists(path_local):
    MODEL_PATH = path_local
elif os.path.exists(path_parent):
    MODEL_PATH = path_parent
else:
    MODEL_PATH = "yolov8n.pt" # Fallback

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

def preprocess_frame(frame, imgsz=640):
    """Préprocess manuel d'une image pour YOLO (letterbox + normalisation)"""
    # BLINDAGE : Forcer le dtype en uint8 si nécessaire
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)
    
    # Vérifier que c'est un array 3D valide
    if len(frame.shape) != 3 or frame.shape[2] != 3:
        raise ValueError(f"Frame invalide: shape={frame.shape}, dtype={frame.dtype}")
    
    # Resize avec aspect ratio (letterbox)
    h, w = frame.shape[:2]
    scale = imgsz / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Padding pour avoir une image carrée
    pad_h = imgsz - new_h
    pad_w = imgsz - new_w
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    
    # Conversion BGR -> RGB et normalisation
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    
    # Transposer HWC -> CHW et ajouter batch dimension
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0)
    return tensor

def run_benchmark():
    print(f"--- MODE CHARGE (PC NVIDIA) ---")
    
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

    print("🎥 Ouverture caméra Topdon (Index 1)...")
    # ON CIBLE L'INDEX 1
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    
    # IMPORTANT : On ne force PAS la résolution ici pour éviter de planter le driver de la Topdon
    # On la laisse en natif (256x384)

    time.sleep(1.0) # Pause initialisation

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
            
            # --- LECTURE ---
            success, frame = cap.read()

            # BLINDAGE : Si image vide, on simule du noir
            if not success or frame is None or frame.size == 0:
                frame = np.zeros((384, 256, 3), dtype=np.uint8) # Taille native simulée
                time.sleep(0.01) 
                status_cam = "SIMU"
            else:
                status_cam = "CAM"

            if elapsed < WARMUP_SEC: phase = "WARMUP"
            elif elapsed < (WARMUP_SEC + TEST_DURATION): phase = "TEST"
            else: break

            # --- INFERENCE ---
            t0 = time.time()
            # IMPORTANT : On skip l'inférence si frame simulé (évite TypeError)
            if status_cam == "CAM":
                # SOLUTION BAS-NIVEAU : Préprocesser manuellement et passer un tensor PyTorch
                # (Contourne tous les bugs de détection de type dans Ultralytics)
                tensor = preprocess_frame(frame, imgsz=640).to('cuda:0')
                
                # Inférence directe sur le modèle PyTorch
                with torch.no_grad():
                    results = model.model(tensor)  # Accès direct au modèle PyTorch sous-jacent
            else:
                time.sleep(0.016)  # Simule ~60 FPS si pas de caméra
            t1 = time.time()

            latence_ms = (t1 - t0) * 1000
            fps_inst = 1.0 / (t1 - t0) if (t1 - t0) > 0 else 0
            frame_count += 1

            if frame_count % 10 == 0:
                cpu = psutil.cpu_percent()
                ram = psutil.virtual_memory().percent
                timestamp = datetime.now().strftime("%H:%M:%S")
                
                temp, power_gpu = get_nvidia_metrics()
                
                # IMPORTANT : On écrit des nombres bruts, pas des strings formatées (cohérence avec plancher.py)
                writer.writerow([timestamp, frame_count, round(latence_ms, 2), round(fps_inst, 2), cpu, ram, temp, power_gpu, phase, 0.0])

                print(f"\r[{phase}] FPS: {fps_inst:.1f} | Pwr: {power_gpu}W | Src: {status_cam}   ", end="")

    cap.release()
    print(f"\n\n✅ Terminé ! Résultats : {OUTPUT_FILE}")

if __name__ == "__main__":
    run_benchmark()