import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io

# 1. CHARGEMENT DES DONNÉES (Intégrées pour la portabilité)
csv_data = """Device,FPS_Moyen,Latence_ms,Temp_Charge_C,Conso_Charge_W,Efficacite_FPS_per_W
Raspberry Pi 4,1.88,531.4,81.6,5.57,0.34
Jetson Orin Nano,33.84,29.57,58.0,7.02,4.82
PC Portable (Nvidia),112.19,10.75,45.7,13.38,8.39"""

df = pd.read_csv(io.StringIO(csv_data))

# Configuration du style
plt.style.use('seaborn-v0_8-whitegrid')
colors = {'Raspberry Pi 4': '#d62728', 'Jetson Orin Nano': '#2ca02c', 'PC Portable (Nvidia)': '#1f77b4'}

# --- GRAPHIQUE 1 : Le "Sweet Spot" (FPS vs Efficacité) ---
# Ce graphique montre qui est le plus intelligent énergétiquement
def plot_fps_efficiency():
    fig, ax1 = plt.subplots(figsize=(10, 6))

    devices = df['Device']
    x = np.arange(len(devices))
    width = 0.4

    # Axe Gauche : FPS (Barres)
    bars = ax1.bar(x, df['FPS_Moyen'], width, label='Vitesse (FPS)', color='skyblue', alpha=0.7)
    ax1.set_ylabel('Vitesse (FPS)', color='tab:blue', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_title('COMPROMIS : Puissance Brute vs Efficacité', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(devices, fontweight='bold')

    # Ajout des valeurs FPS
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', color='tab:blue')

    # Axe Droit : Efficacité (Ligne)
    ax2 = ax1.twinx()
    line = ax2.plot(x, df['Efficacite_FPS_per_W'], color='crimson', marker='o', linewidth=3, markersize=10, label='Efficacité (FPS/W)')
    ax2.set_ylabel('Efficacité (FPS / Watt)', color='crimson', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='crimson')

    # Ajout des valeurs Efficacité
    for i, txt in enumerate(df['Efficacite_FPS_per_W']):
        ax2.annotate(f'{txt:.2f}', (x[i], df['Efficacite_FPS_per_W'][i]), 
                     xytext=(0, 10), textcoords='offset points', ha='center', color='crimson', fontweight='bold')

    # Zone de viabilité (30 FPS)
    ax1.axhline(y=30, color='gray', linestyle='--', alpha=0.5)
    ax1.text(2.6, 31, 'Seuil Temps Réel (30 FPS)', color='gray', fontsize=9)

    plt.tight_layout()
    plt.savefig('results/final/1_performance_vs_efficacite.png')
    print("✅ Graphique 1 généré : Performance vs Efficacité")

# --- GRAPHIQUE 2 : La Carte des Risques (Latence vs Température) ---
# Ce scatter plot montre les dangers (Chauffe + Lag)
def plot_risk_map():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Échelle Logarithmique pour la latence car la Pi explose le score
    ax.set_yscale('log')
    
    for i, row in df.iterrows():
        # La taille du point dépend de la consommation
        size = row['Conso_Charge_W'] * 50  
        ax.scatter(row['Temp_Charge_C'], row['Latence_ms'], s=size, label=row['Device'], color=colors[row['Device']], alpha=0.7, edgecolors='black')
        
        # Annotation
        ax.annotate(row['Device'], (row['Temp_Charge_C'], row['Latence_ms']), 
                    xytext=(0, 10), textcoords='offset points', ha='center', fontweight='bold')

    ax.set_xlabel('Température en Charge (°C)', fontsize=12)
    ax.set_ylabel('Latence (ms) - Échelle Log', fontsize=12)
    ax.set_title('CARTE DES RISQUES : Surchauffe et Latence', fontsize=14)
    
    # Zones de danger
    ax.axvline(x=70, color='red', linestyle='--', alpha=0.3)
    ax.text(71, 15, 'Zone Surchauffe (>70°C)', color='red', rotation=90)
    
    ax.axhline(y=100, color='orange', linestyle='--', alpha=0.3)
    ax.text(35, 110, 'Seuil Lag perceptible (>100ms)', color='orange')

    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig('results/final/2_analyse_risques.png')
    print("✅ Graphique 2 généré : Carte des Risques")

# --- GRAPHIQUE 3 : Le Coût Réel (Watts par Frame) ---
# Inverse de l'efficacité : combien me coûte 1 image en énergie ?
def plot_energy_cost():
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Calcul : Joules par Frame = Watts / FPS
    energy_per_frame = df['Conso_Charge_W'] / df['FPS_Moyen']
    
    bars = ax.bar(df['Device'], energy_per_frame, color=['#d62728', '#2ca02c', '#1f77b4'], alpha=0.8)
    
    ax.set_ylabel('Énergie par Image (Joules)', fontsize=12)
    ax.set_title('COÛT ÉNERGÉTIQUE : Consommation pour traiter 1 image', fontsize=14)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f} J', ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig('results/final/3_cout_energetique.png')
    print("✅ Graphique 3 généré : Coût Énergétique")

if __name__ == "__main__":
    import os
    if not os.path.exists('results/final'):
        os.makedirs('results/final')
        
    plot_fps_efficiency()
    plot_risk_map()
    plot_energy_cost()
    plt.show()