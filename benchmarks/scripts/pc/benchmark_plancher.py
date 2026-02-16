import psutil
import time
import csv
import subprocess
import os
from datetime import datetime

# --- CONFIGURATION BLINDÉE WINDOWS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results", "pc")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "plancher.csv")
DURATION_SEC = 60

def get_nvidia_metrics():
    """Récupère Température et Conso GPU via nvidia-smi"""
    try:
        # On demande la température et la conso instantanée
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
            encoding='utf-8'
        )
        temp, power = out.strip().split(',')
        return float(temp), float(power)
    except Exception as e:
        return 0.0, 0.0

def run_plancher():
    print(f"--- MODE PLANCHER (PC NVIDIA) ---")
    
    # Création dossier intelligent
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Destination : {OUTPUT_FILE}")
    print(f"Durée : {DURATION_SEC} secondes (Ne touchez à rien)")

    headers = ["Timestamp", "Frame_ID", "Latence_ms", "FPS_Inst", "CPU_%", "RAM_%", "Temp_C", "Power_Soft_W", "Phase", "Power_Manual_W"]

    with open(OUTPUT_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        
        start_time = time.time()
        frame_id = 0
        
        try:
            while (time.time() - start_time) < DURATION_SEC:
                cpu = psutil.cpu_percent(interval=None)
                ram = psutil.virtual_memory().percent
                timestamp = datetime.now().strftime("%H:%M:%S")        
                
                temp, power_gpu = get_nvidia_metrics()

                # On remplit avec des 0 pour les colonnes "FPS" et "Latence" car on est au repos
                writer.writerow([timestamp, frame_id, 0, 0, cpu, ram, temp, power_gpu, "IDLE", 0.0])
                
                remaining = int(DURATION_SEC - (time.time() - start_time))
                print(f"\rReste : {remaining}s | CPU: {cpu}% | GPU Temp: {temp}°C | Pwr: {power_gpu}W   ", end="")
                
                frame_id += 1
                time.sleep(1.0)
                
        except KeyboardInterrupt:
            print("\nArrêt manuel.")

    print(f"\n✅ Terminé ! Fichier : {OUTPUT_FILE}")

if __name__ == "__main__":
    run_plancher()