import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# --- CONFIGURATION ---
BASE_DIR = "results"
OUTPUT_DIR = "results/analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# On cible uniquement les deux champions pour le comparatif
DEVICES_MAP = {
    "jetson_trt": "Jetson Nano (TensorRT)",
    "pi_hailo": "RPi 5 + Hailo-8L"
}

def get_data():
    """Récupère et filtre les données pour le duel Jetson vs RPi"""
    rows = []
    # On ne scanne que les dossiers pertinents
    for folder, label in DEVICES_MAP.items():
        path = os.path.join(BASE_DIR, folder, "rapport_final.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            rows.append({
                "Device": label,
                "FPS": df["FPS_Moyen"].values[0],
                "Efficiency_FPS_per_Watt": df["Efficacite_FPS_per_W"].values[0]
            })
    
    if not rows:
        print("⚠️ Attention : Pas de CSV trouvés, utilisation de données simulées.")
        return pd.DataFrame([
            {"Device": "Jetson Nano (TensorRT)", "FPS": 214.2, "Efficiency_FPS_per_Watt": 33.6},
            {"Device": "RPi 5 + Hailo-8L", "FPS": 44.4, "Efficiency_FPS_per_Watt": 32.4}
        ])
        
    return pd.DataFrame(rows)

def plot_duel(df):
    # Configuration du style
    plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    # Couleurs : Jetson (Vert Nvidia), RPi/Hailo (Framboise/Orange)
    colors = ['#76b900', '#C51A4A'] 
    
    # --- GRAPHIQUE 1 : VITESSE (FPS) ---
    df_fps = df.sort_values('FPS', ascending=True)
    
    bars1 = ax1.barh(df_fps['Device'], df_fps['FPS'], color=colors, edgecolor='black')
    ax1.set_title("Vitesse d'Inférence (FPS)", fontsize=16, fontweight='bold', color='#333')
    ax1.set_xlabel("Frames Per Second", fontsize=12)
    
    for bar in bars1:
        width = bar.get_width()
        ax1.text(width + 5, bar.get_y() + bar.get_height()/2, 
                 f'{width:.1f} FPS', va='center', fontweight='bold', fontsize=12)
    
    # Message "Victoire"
    try:
        jetson_fps = df[df['Device'] == "Jetson Nano (TensorRT)"]['FPS'].values[0]
        rpi_fps = df[df['Device'] == "RPi 5 + Hailo-8L"]['FPS'].values[0]
        gain = jetson_fps / rpi_fps
        ax1.text(0.5, 0.9, f"Jetson {gain:.1f}x plus rapide !", 
                 transform=ax1.transAxes, ha='center', color='darkgreen', 
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='green'))
    except: pass

    # --- GRAPHIQUE 2 : EFFICACITÉ (FPS/W) ---
    df_eff = df.sort_values('Efficiency_FPS_per_Watt', ascending=True)
    
    bars2 = ax2.barh(df_eff['Device'], df_eff['Efficiency_FPS_per_Watt'], color=colors, edgecolor='black')
    ax2.set_title("Efficacité Énergétique", fontsize=16, fontweight='bold', color='#333')
    ax2.set_xlabel("FPS par Watt (Higher is better)", fontsize=12)
    
    for bar in bars2:
        width = bar.get_width()
        ax2.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                 f'{width:.1f} FPS/W', va='center', fontweight='bold', fontsize=12)

    ax2.text(0.5, 0.9, "Excellente efficacité du NPU Hailo", 
             transform=ax2.transAxes, ha='center', color='#C51A4A', 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='#C51A4A'))

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "poster_duel_jetson_vs_rpi.png")
    plt.savefig(output_path, dpi=300)
    print(f"✅ Graphique généré : {output_path}")
    plt.show()

if __name__ == "__main__":
    # 1. Récupération des données
    data = get_data()
    
    # 2. EXPORT CSV (Pour Canva/Excel)
    csv_filename = "donnees_duel_jetson_rpi.csv"
    csv_path = os.path.join(OUTPUT_DIR, csv_filename)
    data.to_csv(csv_path, index=False, sep=';', decimal=',') # Format compatible Excel français
    print(f"\n💾 DONNÉES SAUVEGARDÉES : {csv_path}")
    print(f"   (Tu peux ouvrir ce fichier dans Excel pour faire tes propres graphiques)\n")

    # 3. Génération du graphique Python
    plot_duel(data)