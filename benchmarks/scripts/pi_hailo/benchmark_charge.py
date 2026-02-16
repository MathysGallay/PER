import time
import csv
import psutil
import os
import sys
import subprocess
import threading
import re
from datetime import datetime

# --- CONFIGURATION ---
HEF_PATH = "data/yolov8s.hef" 
OUTPUT_FILE = "results/pi_hailo/charge.csv"
TEST_DURATION = 60 

# Variables globales
benchmark_stats = {"fps": 0.0, "latency": 0.0, "running": True}

def get_pi_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return int(f.read()) / 1000.0
    except:
        return 0.0

def run_hailo_process(final_hef_path):
    """Lance la commande Hailo et capture la sortie"""
    print(f"🚀 Démarrage du benchmark Hailo sur {final_hef_path}...")
    
    # ON RETIRE "--no-power" QUI FAISAIT PLANTER
    # On garde --time-to-run pour que ça dure bien 60s
    cmd = ["hailortcli", "benchmark", final_hef_path, "--time-to-run", str(TEST_DURATION)]
    
    try:
        # On capture stdout ET stderr pour voir les erreurs
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"\n❌ ERREUR COMMANDE HAILO (Code {result.returncode}) :")
            print(result.stderr) # Affiche pourquoi ça a planté
        
        output = result.stdout
        
        # Parsing (Extraction des FPS et Latence)
        # On cherche d'abord dans le résumé, sinon dans le flux
        fps_match = re.search(r"FPS\s*\(.*?\)\s*=\s*([\d\.]+)", output) # Format Summary
        if not fps_match:
            fps_match = re.search(r"FPS:\s*([\d\.]+)", output) # Format streaming
            
        lat_match = re.search(r"HW Latency:\s*([\d\.]+)", output) # Format streaming
        if not lat_match:
             lat_match = re.search(r"Latency\s*\(hw\)\s*=\s*([\d\.]+)", output) # Format Summary

        if fps_match: benchmark_stats["fps"] = float(fps_match.group(1))
        if lat_match: benchmark_stats["latency"] = float(lat_match.group(1))
        
    except Exception as e:
        print(f"❌ Exception Python : {e}")
    finally:
        benchmark_stats["running"] = False

def run_benchmark():
    global HEF_PATH
    
    print(f"--- MODE CHARGE (PI 5 + HAILO) ---")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    if not os.path.exists(HEF_PATH):
        files = [f for f in os.listdir("data") if f.endswith(".hef")]
        if files:
            HEF_PATH = os.path.join("data", files[0])
            print(f"⚠️ Fichier par défaut absent, utilisation de : {HEF_PATH}")
        else:
            print(f"❌ ERREUR : Aucun fichier .hef trouvé dans data/")
            return

    # Lancement du thread Hailo
    hailo_thread = threading.Thread(target=run_hailo_process, args=(HEF_PATH,))
    hailo_thread.start()

    headers = ["Timestamp", "Frame_ID", "Latence_ms", "FPS_Inst", "CPU_%", "RAM_%", "Temp_C", "Power_Soft_W", "Phase", "Power_Manual_W"]
    
    print("🔌 Enregistrement des sondes système...")
    
    with open(OUTPUT_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        
        start_global = time.time()
        frame_count = 0
        phase = "WARMUP"
        
        while benchmark_stats["running"]:
            elapsed = time.time() - start_global
            
            # Sécurité : Si Hailo plante, on ne boucle pas à l'infini
            if elapsed > (TEST_DURATION + 10) and benchmark_stats["fps"] == 0:
                print("\n⚠️ Timeout : Le benchmark semble bloqué ou fini.")
                break

            if elapsed < 5: phase = "WARMUP"
            else: phase = "TEST"

            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory().percent
            temp = get_pi_temp()
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            # On met 0 en FPS instantané car on aura la moyenne à la fin
            writer.writerow([timestamp, frame_count, 0, 0, cpu, ram, temp, 0.0, phase, 0.0])
            
            sys.stdout.write(f"\r[{phase}] Temp: {temp:.1f}°C | CPU: {cpu}% | Benchmark en cours...   ")
            sys.stdout.flush()
            
            frame_count += 1
            time.sleep(1)

    hailo_thread.join()
    print(f"\n\n🏁 RÉSULTATS HAILO OBTENUS :")
    print(f"   FPS Moyen : {benchmark_stats['fps']:.2f}")
    print(f"   Latence   : {benchmark_stats['latency']:.2f} ms")
    
    # Enregistrement final
    with open(OUTPUT_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        timestamp = datetime.now().strftime("%H:%M:%S")
        writer.writerow([timestamp, frame_count, benchmark_stats['latency'], benchmark_stats['fps'], 0, 0, 0, 0, "RESULT", 0])

    print(f"✅ Terminé ! Résultats dans : {OUTPUT_FILE}")

if __name__ == "__main__":
    run_benchmark()
