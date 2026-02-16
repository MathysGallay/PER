import pandas as pd
import os
import sys

# --- CONFIGURATION ---
FILE_IDLE = "results/pi_hailo/plancher.csv"
FILE_LOAD = "results/pi_hailo/charge.csv"
REPORT_FILE = "results/pi_hailo/rapport_final.txt"
REPORT_CSV_FILE = "results/pi_hailo/rapport_final.csv"

def main():
    print("\n--- ANALYSEUR RASPBERRY PI ---")
    
    if not os.path.exists(FILE_IDLE) or not os.path.exists(FILE_LOAD):
        print("❌ Fichiers manquants. Lancez les benchmarks d'abord.")
        return

    # Lecture des CSV
    df_idle = pd.read_csv(FILE_IDLE)
    df_load = pd.read_csv(FILE_LOAD)

    # 1. Calcul Température
    temp_idle = df_idle['Temp_C'].mean()
    # On prend la température max atteinte pendant la charge
    temp_load = df_load[df_load['Phase'] == 'TEST']['Temp_C'].mean()

    # 2. Récupération FPS (Spécial Hailo : c'est souvent dans la dernière ligne 'RESULT')
    last_row = df_load.tail(1)
    if last_row['Phase'].values[0] == "RESULT":
        fps_mean = float(last_row['FPS_Inst'].values[0])
        latence_mean = float(last_row['Latence_ms'].values[0])
    else:
        # Fallback si pas de ligne result
        fps_mean = 0
        latence_mean = 0

    # 3. Gestion Puissance (Manuelle obligatoire sur Pi)
    print("\n⚠️  Pas de sonde de puissance interne sur Raspberry Pi.")
    try:
        watts_idle = float(input("👉 Entrez la consommation au REPOS (Watt) lue sur le wattmètre : "))
        watts_load = float(input("👉 Entrez la consommation en CHARGE (Watt) lue sur le wattmètre : "))
    except ValueError:
        print("Entrée invalide, utilisation de valeurs par défaut (0W).")
        watts_idle = 0.0
        watts_load = 0.0

    delta_watts = watts_load - watts_idle
    efficiency = fps_mean / delta_watts if delta_watts > 0 else 0

    # 4. Génération Rapport
    report = f"""
===================================================
        RAPPORT FINAL : RPI 5 + HAILO 8L
===================================================
1. PERFORMANCES
---------------------
FPS Moyen       : {fps_mean:.2f} fps
Latence         : {latence_mean:.2f} ms

2. THERMIQUE
------------
Temp. Idle      : {temp_idle:.1f} °C
Temp. Charge    : {temp_load:.1f} °C
Delta           : +{temp_load - temp_idle:.1f} °C

3. EFFICACITÉ
-----------------------
Conso Idle      : {watts_idle:.2f} W
Conso Charge    : {watts_load:.2f} W
DELTA IA        : {delta_watts:.2f} W
EFFICACITÉ      : {efficiency:.2f} FPS/Watt
===================================================
"""
    print(report)
    
    # Sauvegarde TXT
    with open(REPORT_FILE, "w") as f:
        f.write(report)
        
    # Sauvegarde CSV (Pour le graphique final)
    df_final = pd.DataFrame([{
        'Device': 'Raspberry Pi 5 + Hailo',
        'FPS_Moyen': fps_mean,
        'Latence_ms': latence_mean,
        'Temp_Idle_C': temp_idle,
        'Temp_Charge_C': temp_load,
        'Conso_Idle_W': watts_idle,
        'Conso_Charge_W': watts_load,
        'Efficacite_FPS_per_W': efficiency
    }])
    df_final.to_csv(REPORT_CSV_FILE, index=False)
    print(f"✅ Rapport prêt pour agrégation : {REPORT_CSV_FILE}")

if __name__ == "__main__":
    main()