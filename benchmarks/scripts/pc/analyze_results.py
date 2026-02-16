import csv
import os

# --- CONFIGURATION BLINDÉE WINDOWS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"Script Directory: {SCRIPT_DIR}")
INPUT_DIR = os.path.join(SCRIPT_DIR, "results", "pc")

FILE_IDLE = os.path.join(INPUT_DIR, "plancher.csv")
FILE_LOAD = os.path.join(INPUT_DIR, "charge.csv")
REPORT_FILE = os.path.join(INPUT_DIR, "rapport_final.txt")
REPORT_CSV_FILE = os.path.join(INPUT_DIR, "rapport_final.csv")

def read_csv_data(filepath):
    """Lit un CSV et retourne les données sous forme de liste de dictionnaires"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

def calculate_mean(data, column):
    """Calcule la moyenne d'une colonne"""
    values = [float(row[column]) for row in data if row[column]]
    return sum(values) / len(values) if values else 0.0

def main():
    print(f"--- ANALYSEUR DE RÉSULTATS (PC) ---")
    
    if not os.path.exists(FILE_IDLE) or not os.path.exists(FILE_LOAD):
        print(f"❌ ERREUR : Fichiers manquants dans {INPUT_DIR}")
        return

    # Validation de la taille des fichiers (doivent avoir plus qu'un header)
    if os.path.getsize(FILE_IDLE) < 100 or os.path.getsize(FILE_LOAD) < 100:
        print(f"❌ ERREUR : Fichiers CSV incomplets ou vides")
        print(f"   - {FILE_IDLE}: {os.path.getsize(FILE_IDLE)} bytes")
        print(f"   - {FILE_LOAD}: {os.path.getsize(FILE_LOAD)} bytes")
        return

    # Chargement avec module CSV standard
    try:
        data_idle = read_csv_data(FILE_IDLE)
        data_load = read_csv_data(FILE_LOAD)
        
        print(f"✅ CSV chargés avec succès")
        print(f"   - Plancher: {len(data_idle)} lignes")
        print(f"   - Charge: {len(data_load)} lignes")
        
        # Validation des colonnes essentielles
        required_cols = ['Temp_C', 'Power_Soft_W', 'Phase']
        if data_load and not all(col in data_load[0] for col in required_cols):
            print(f"❌ ERREUR : Colonnes manquantes dans charge.csv")
            print(f"   Colonnes présentes: {list(data_load[0].keys())}")
            return
            
    except Exception as e:
        print(f"❌ Erreur lecture CSV : {e}")
        print(f"   Vérifiez que les benchmarks ont bien tourné jusqu'à la fin.")
        import traceback
        traceback.print_exc()
        return

    # CALCULS
    # Moyenne Température IDLE
    avg_temp_idle = calculate_mean(data_idle, 'Temp_C')
    avg_pwr_idle = calculate_mean(data_idle, 'Power_Soft_W')
    
    # Filtrage phase TEST pour la charge
    data_test = [row for row in data_load if row['Phase'] == 'TEST']
    
    if not data_test:
        print("⚠️ Pas de données de phase 'TEST' trouvées.")
        return

    avg_fps = calculate_mean(data_test, 'FPS_Inst')
    avg_lat = calculate_mean(data_test, 'Latence_ms')
    avg_temp_load = calculate_mean(data_test, 'Temp_C')
    avg_pwr_load = calculate_mean(data_test, 'Power_Soft_W')

    # Calcul Efficacité
    delta_watts = avg_pwr_load - avg_pwr_idle
    # Si delta_watts est trop faible (ou négatif à cause du bruit), on sécurise
    if delta_watts < 1.0: delta_watts = 1.0 
    
    efficiency = avg_fps / avg_pwr_load # Sur PC, on divise souvent par la conso totale GPU

    report = f"""
===================================================
        RAPPORT FINAL (PC NVIDIA)
===================================================
1. PERFORMANCES
---------------------
FPS Moyen       : {avg_fps:.2f} fps
Latence Moyenne : {avg_lat:.2f} ms

2. THERMIQUE (GPU)
------------------
Temp. Idle      : {avg_temp_idle:.1f} °C
Temp. Charge    : {avg_temp_load:.1f} °C
Delta           : +{avg_temp_load - avg_temp_idle:.1f} °C

3. ÉNERGIE (GPU Only)
---------------------
Conso Idle      : {avg_pwr_idle:.2f} W
Conso Charge    : {avg_pwr_load:.2f} W
DELTA IA        : {delta_watts:.2f} W

EFFICACITÉ (FPS/W) : {efficiency:.2f}
===================================================
"""
    print(report)
    
    with open(REPORT_FILE, "w", encoding='utf-8') as f: 
        f.write(report)
    
    # Génération du CSV final standardisé pour le graphique comparatif
    with open(REPORT_CSV_FILE, "w", encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Device", "FPS_Moyen", "Latence_ms", "Temp_Idle_C", "Temp_Charge_C", 
                        "Conso_Idle_W", "Conso_Charge_W", "Efficacite_FPS_per_W"])
        writer.writerow([
            "PC Portable (Nvidia)",
            round(avg_fps, 2),
            round(avg_lat, 2),
            round(avg_temp_idle, 1),
            round(avg_temp_load, 1),
            round(avg_pwr_idle, 2),
            round(avg_pwr_load, 2),
            round(efficiency, 2)
        ])
    
    print(f"✅ Rapports sauvegardés dans {INPUT_DIR}")

if __name__ == "__main__":
    main()