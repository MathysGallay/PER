import cv2
import time
import csv
import psutil
import os
import sys
from datetime import datetime
from ultralytics import YOLO
from jtop import jtop

# --- CONFIGURATION TENSORRT ---
MODEL_PATH = "/home/jetson/benchmark/data/best.engine"
OUTPUT_FILE = "/home/jetson/benchmark/results/jetson_trt/charge.csv" # <--- Fichier de résultats séparé
WARMUP_SEC = 20     # <--- TensorRT met un peu plus de temps à "chauffer" la première fois
TEST_DURATION = 60 
IMG_SIZE = 256 

def get_temp_robust(jetson):
    t = jetson.stats.get('Temp GPU')
    if t is not None and t > 0: return t
    t = jetson.stats.get('GPU')
    if t is not None and t > 0: return t
    try: return jetson.temperature['GPU']['temp']
    except: pass
    try: return jetson.temperature['AO']['temp']
    except: return 0.0

def find_camera():
    for index in [0, 1]:
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret: return cap, index
            cap.release()
    return None, -1

def run_benchmark():
    print(f"--- MODE CHARGE GPU (TENSORRT / TOPDON) ---")
    
    if not os.path.exists('results'): os.makedirs('results')
    if not os.path.exists(MODEL_PATH):
        print(f"ERREUR : Le moteur {MODEL_PATH} est introuvable !")
        print("   -> Lance la commande : yolo export model=models/best.pt format=engine device=0 half=True imgsz=256")
        return

    print(f"Chargement du moteur TensorRT {MODEL_PATH}...")
    try:
        # task='detect' est parfois nécessaire avec les .engine pour éviter les ambiguïtés
        model = YOLO(MODEL_PATH, task='detect') 
    except Exception as e:
        print(f"Erreur chargement moteur : {e}")
        return

    print("📷 Recherche de la caméra...")
    cap, cam_idx = find_camera()
    if cap is None:
        print("ERREUR : Aucune caméra détectée.")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 192)

    headers = ["Timestamp", "Frame_ID", "Latence_ms", "FPS_Inst", "CPU_%", "RAM_%", "Temp_C", "Power_Soft_W", "Phase", "Power_Manual_W"]
    
    print("🔌 Connexion aux sondes jtop...")
    try:
        with jtop() as jetson:
            if not jetson.ok(): return

            with open(OUTPUT_FILE, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(headers)
                
                print(f"Échauffement ({WARMUP_SEC}s)...")
                start_global = time.time()
                frame_count = 0
                phase = "WARMUP"
                
                while True:
                    elapsed = time.time() - start_global
                    success, frame = cap.read()
                    if not success: break 

                    if elapsed < WARMUP_SEC: phase = "WARMUP"
                    elif elapsed < (WARMUP_SEC + TEST_DURATION): phase = "TEST"
                    else: break

                    # --- INFERENCE ---
                    t0 = time.time()
                    # Avec TensorRT, imgsz est fixe, device=0 est implicite mais on le garde
                    model.predict(frame, imgsz=IMG_SIZE, device=0, half=True, verbose=False)
                    t1 = time.time()

                    duration = t1 - t0
                    latence_ms = duration * 1000
                    fps_inst = 1.0 / duration if duration > 0 else 0
                    frame_count += 1

                    if frame_count % 10 == 0:
                        cpu = psutil.cpu_percent()
                        ram = psutil.virtual_memory().percent
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        temp = get_temp_robust(jetson)
                        power_w = jetson.stats.get('Power TOT', 0) / 1000.0
                        
                        writer.writerow([timestamp, frame_count, f"{latence_ms:.2f}", f"{fps_inst:.2f}", cpu, ram, temp, 0.0, phase, power_w])
                        sys.stdout.write(f"\r[{phase}] FPS: {fps_inst:.1f} | Latence: {latence_ms:.1f}ms | Pwr: {power_w:.1f}W   ")
                        sys.stdout.flush()

    except Exception as e:
        print(f"\nErreur : {e}")
    finally:
        cap.release()
        print(f"\n\nTerminé ! Résultats dans : {OUTPUT_FILE}")

if __name__ == "__main__":
    run_benchmark()