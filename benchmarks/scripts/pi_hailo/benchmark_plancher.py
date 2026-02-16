import time
import csv
import psutil
import os
import sys
from datetime import datetime

# --- CONFIGURATION ---
OUTPUT_FILE = "results/pi_hailo/plancher.csv"
DURATION_SEC = 60

def get_pi_temp():
    try:
        # Méthode spécifique Raspberry Pi
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return int(f.read()) / 1000.0
    except:
        return 0.0

def run_plancher():
    print(f"--- MODE PLANCHER (RASPBERRY PI) ---")
    
    # Création dossier intelligent
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    headers = ["Timestamp", "Frame_ID", "Latence_ms", "FPS_Inst", "CPU_%", "RAM_%", "Temp_C", "Power_Soft_W", "Phase", "Power_Manual_W"]
    
    print(f"Acquisition des données de repos ({DURATION_SEC}s)...")
    
    try:
        with open(OUTPUT_FILE, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            
            start_time = time.time()
            frame_count = 0
            
            while (time.time() - start_time) < DURATION_SEC:
                current_time = time.time()
                elapsed = current_time - start_time
                
                # Récupération stats
                cpu = psutil.cpu_percent(interval=0.1) # Petite pause pour avoir une lecture CPU valide
                ram = psutil.virtual_memory().percent
                temp = get_pi_temp()
                timestamp = datetime.now().strftime("%H:%M:%S")
                
                # Sur Pi, pas de sonde de puissance logicielle fiable par défaut -> 0.0
                writer.writerow([timestamp, frame_count, 0, 0, cpu, ram, temp, 0.0, "IDLE", 0.0])
                
                sys.stdout.write(f"\rReste : {int(DURATION_SEC - elapsed)}s | Temp: {temp:.1f}°C | CPU: {cpu}%   ")
                sys.stdout.flush()
                frame_count += 1
                
                time.sleep(0.9) # On vise environ 1 mesure par seconde

    except KeyboardInterrupt:
        print("\nArrêt manuel.")
    
    print(f"\n✅ Terminé ! Fichier : {OUTPUT_FILE}")

if __name__ == "__main__":
    run_plancher()