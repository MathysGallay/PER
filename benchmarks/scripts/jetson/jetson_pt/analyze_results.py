import csv
import os


FILE_IDLE = "/home/jetson/benchmark/results/jetson_pt/plancher.csv"
FILE_LOAD = "/home/jetson/benchmark/results/jetson_pt/charge.csv"
REPORT_FILE = "/home/jetson/benchmark/results/jetson_pt/rapport_final.txt"
REPORT_CSV_FILE = "/home/jetson/benchmark/results/jetson_pt/rapport_final.csv"

def load_data(filepath):
    """Lit un fichier CSV et renvoie une liste de dictionnaires"""
    data = []
    if not os.path.exists(filepath):
        print(f"ERREUR : Le fichier {filepath} est introuvable.")
        return None
    
    with open(filepath, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

def get_avg(data, key, filter_phase=None):
    """Calcule la moyenne d'une colonne numérique"""
    values = []
    for row in data:
        if filter_phase and row.get("Phase") != filter_phase:
            continue    
        try:
            val = float(row[key])
            if val > 0:
                values.append(val)
        except (ValueError, KeyError):
            continue
    return sum(values) / len(values) if values else 0.0

def main():
    print("--- ANALYSEUR DE RÉSULTATS BENCHMARK ---")
    
    idle_data = load_data(FILE_IDLE)
    load_data_raw = load_data(FILE_LOAD)
    
    if not idle_data or not load_data_raw:
        return

    avg_temp_idle = get_avg(idle_data, "Temp_C")
    avg_pwr_soft_idle = get_avg(idle_data, "Power_Soft_W")
    
    avg_fps = get_avg(load_data_raw, "FPS_Inst", filter_phase="TEST")
    avg_lat = get_avg(load_data_raw, "Latence_ms", filter_phase="TEST")
    avg_temp_load = get_avg(load_data_raw, "Temp_C", filter_phase="TEST")
    avg_pwr_soft_load = get_avg(load_data_raw, "Power_Soft_W", filter_phase="TEST")

    final_pwr_idle = 0.0
    final_pwr_load = 0.0
    source_pwr = "LOGICIELLE (Interne)"

    if avg_pwr_soft_load > 1.0:
        final_pwr_idle = avg_pwr_soft_idle
        final_pwr_load = avg_pwr_soft_load
        print(f"Puissance détectée dans les logs (Mode PC).")
    else:
        source_pwr = "MATÉRIELLE (Sonde USB)"
        print("\nAucune puissance logicielle détectée (Mode RPi/Jetson).")
        
        final_pwr_idle = get_avg(idle_data, "Power_Manual_W")
        final_pwr_load = get_avg(load_data_raw, "Power_Manual_W", filter_phase="TEST")
        
        print(f"Valeurs lues depuis colonne Power_Manual_W des CSV")
        print(f"  - Plancher/Idle : {final_pwr_idle:.2f} W")
        print(f"  - Charge        : {final_pwr_load:.2f} W")

    delta_watts = final_pwr_load - final_pwr_idle
    efficiency = avg_fps / final_pwr_load if final_pwr_load > 0 else 0
    delta_temp = avg_temp_load - avg_temp_idle

    report = f"""
===================================================
        RAPPORT FINAL DE BENCHMARK IA
===================================================
Source des données : {source_pwr}

1. PERFORMANCES PURES
---------------------
FPS Moyen       : {avg_fps:.2f} fps
Latence Moyenne : {avg_lat:.2f} ms

2. THERMIQUE
------------
Temp. Idle      : {avg_temp_idle:.1f} °C
Temp. Charge    : {avg_temp_load:.1f} °C
Augmentation    : +{delta_temp:.1f} °C

3. ÉNERGIE & EFFICACITÉ
-----------------------
Conso Idle      : {final_pwr_idle:.2f} W
Conso Charge    : {final_pwr_load:.2f} W
DELTA IA        : {delta_watts:.2f} W  (Coût énergétique spécifique de l'IA)

EFFICACITÉ GLOBALE : {efficiency:.2f} FPS/Watt
===================================================
"""
    
    print(report)
    
    with open(REPORT_FILE, "w") as f:
        f.write(report)
    print(f"Rapport sauvegardé dans : {REPORT_FILE}")
    
    csv_headers = [
        "Source_Puissance",
        "FPS_Moyen", "Latence_ms",
        "Temp_Idle_C", "Temp_Charge_C", "Delta_Temp_C",
        "Conso_Idle_W", "Conso_Charge_W", "Delta_IA_W",
        "Efficacite_FPS_per_W"
    ]
    
    csv_data = [
        source_pwr,
        f"{avg_fps:.2f}", f"{avg_lat:.2f}",
        f"{avg_temp_idle:.1f}", f"{avg_temp_load:.1f}", f"{delta_temp:.1f}",
        f"{final_pwr_idle:.2f}", f"{final_pwr_load:.2f}", f"{delta_watts:.2f}",
        f"{efficiency:.2f}"
    ]
    
    with open(REPORT_CSV_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)
        writer.writerow(csv_data)
    
    print(f"Rapport CSV sauvegardé dans : {REPORT_CSV_FILE}")

if __name__ == "__main__":
    main()  