import time
import csv
import psutil
import os
from datetime import datetime
from jtop import jtop

OUTPUT_FILE = "/home/jetson/benchmark/results/jetson_pt/plancher.csv"
DURATION_SEC = 60  

def get_temp(jetson):
    """Fonction robuste pour récupérer la température"""
    # Essai 1 : Clé standard
    t = jetson.stats.get('Temp GPU')
    if t is not None and t > 0: return t
    
    # Essai 2 : Clé alternative (souvent 'GPU')
    t = jetson.stats.get('GPU')
    if t is not None and t > 0: return t
    
    # Essai 3 : Via le dictionnaire temperature direct
    try:
        return jetson.temperature['GPU']['temp']
    except:
        return 0.0

def run_plancher():
    print(f"--- MODE PLANCHER (JETSON) ---")
    print(f"Acquisition via sondes internes (jtop)...")
    print(f"Durée : {DURATION_SEC} secondes")

    if not os.path.exists('results'): os.makedirs('results')

    headers = ["Timestamp", "CPU_Load_%", "RAM_Load_%", "Temp_C", "Power_Soft_W", "Power_Manual_W"]

    with jtop() as jetson:
        if not jetson.ok():
            print("❌ Erreur: jtop ne fonctionne pas.")
            return

        with open(OUTPUT_FILE, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            
            start_time = time.time()
            try:
                while (time.time() - start_time) < DURATION_SEC:
                    cpu = psutil.cpu_percent(interval=None)
                    ram = psutil.virtual_memory().percent
                    timestamp = datetime.now().strftime("%H:%M:%S")        
                    
                    # Correction ici : appel de la nouvelle fonction
                    temp = get_temp(jetson)
                    
                    # Lecture puissance (mW -> W)
                    power_watts = jetson.stats.get('Power TOT', 0) / 1000.0 

                    writer.writerow([timestamp, cpu, ram, temp, 0.0, power_watts])
                    
                    remaining = int(DURATION_SEC - (time.time() - start_time))
                    print(f"\rReste : {remaining}s | Temp: {temp}°C | Conso: {power_watts:.2f} W", end="")
                    
                    time.sleep(1.0)
                    
            except KeyboardInterrupt:
                print("\nArrêt manuel.")

    print(f"\n✅ Terminé ! Fichier : {OUTPUT_FILE}")

if __name__ == "__main__":
    run_plancher()