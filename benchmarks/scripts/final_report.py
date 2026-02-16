import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Configuration des dossiers
BASE_DIR = "results"
OUTPUT_DIR = "results/analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# On liste les dossiers tels qu'ils sont nommés sur ton PC
DEVICES = {
    "jetson_pt": "Jetson Nano (PyTorch)",
    "jetson_trt": "Jetson Nano (TensorRT)",
    "pc": "PC Portable (RTX 4060)",
    "pi_hailo": "RPi 5 + Hailo-8L"
}

def aggregate_all_results():
    """
    Agrège tous les résultats des différents devices en un seul DataFrame.
    Gère les différences de colonnes entre les CSV (Device vs Source_Puissance).
    """
    all_rows = []
    for folder, label in DEVICES.items():
        path = os.path.join(BASE_DIR, folder, "rapport_final.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            
            # Récupération de toutes les colonnes importantes
            row = {
                "Device": label,
                "FPS": df["FPS_Moyen"].values[0],
                "Latence_ms": df["Latence_ms"].values[0],
                "Temp_Idle_C": df["Temp_Idle_C"].values[0] if "Temp_Idle_C" in df.columns else np.nan,
                "Temp_Charge_C": df["Temp_Charge_C"].values[0],
                "Conso_Idle_W": df["Conso_Idle_W"].values[0],
                "Conso_Charge_W": df["Conso_Charge_W"].values[0],
                "Efficacite_FPS_per_W": df["Efficacite_FPS_per_W"].values[0],
            }
            
            # Colonnes optionnelles (présentes uniquement pour Jetson)
            if "Delta_Temp_C" in df.columns:
                row["Delta_Temp_C"] = df["Delta_Temp_C"].values[0]
            else:
                row["Delta_Temp_C"] = row["Temp_Charge_C"] - row["Temp_Idle_C"] if not np.isnan(row["Temp_Idle_C"]) else np.nan
            
            if "Delta_IA_W" in df.columns:
                row["Delta_IA_W"] = df["Delta_IA_W"].values[0]
            else:
                row["Delta_IA_W"] = row["Conso_Charge_W"] - row["Conso_Idle_W"]
            
            all_rows.append(row)
    
    return pd.DataFrame(all_rows).sort_values(by="FPS", ascending=False)

def generate_fps_plot(df):
    """Graphique : Vitesse d'inférence (FPS)"""
    plt.figure(figsize=(10, 6))
    df_sorted = df.sort_values(by="FPS")
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df_sorted)))
    
    bars = plt.barh(df_sorted['Device'], df_sorted['FPS'], color=colors)
    plt.title("Vitesse d'Inférence (FPS)", fontweight='bold', fontsize=14)
    plt.xlabel("FPS (Plus est mieux)", fontsize=12)
    
    for i, (bar, v) in enumerate(zip(bars, df_sorted['FPS'])):
        plt.text(v + 2, i, f"{v:.1f}", va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "01_fps_comparison.png"), dpi=300)
    print("✅ Graphique '01_fps_comparison.png' généré.")
    plt.close()

def generate_efficiency_plot(df):
    """Graphique : Efficacité énergétique (FPS/W)"""
    plt.figure(figsize=(10, 6))
    df_sorted = df.sort_values(by="Efficacite_FPS_per_W")
    colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(df_sorted)))
    
    bars = plt.barh(df_sorted['Device'], df_sorted['Efficacite_FPS_per_W'], color=colors)
    plt.title("Efficacité Énergétique (FPS/Watt)", fontweight='bold', fontsize=14)
    plt.xlabel("FPS/W (Plus est mieux)", fontsize=12)
    
    for i, (bar, v) in enumerate(zip(bars, df_sorted['Efficacite_FPS_per_W'])):
        plt.text(v + 0.5, i, f"{v:.1f}", va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "02_efficiency_comparison.png"), dpi=300)
    print("✅ Graphique '02_efficiency_comparison.png' généré.")
    plt.close()

def generate_latency_plot(df):
    """Graphique : Latence d'inférence (ms)"""
    plt.figure(figsize=(10, 6))
    df_sorted = df.sort_values(by="Latence_ms", ascending=True)
    colors = plt.cm.coolwarm_r(np.linspace(0.3, 0.9, len(df_sorted)))
    
    bars = plt.barh(df_sorted['Device'], df_sorted['Latence_ms'], color=colors)
    plt.title("Latence d'Inférence", fontweight='bold', fontsize=14)
    plt.xlabel("Latence en ms (Moins est mieux)", fontsize=12)
    
    for i, (bar, v) in enumerate(zip(bars, df_sorted['Latence_ms'])):
        plt.text(v + 0.5, i, f"{v:.2f} ms", va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "03_latency_comparison.png"), dpi=300)
    print("✅ Graphique '03_latency_comparison.png' généré.")
    plt.close()

def generate_power_consumption_plot(df):
    """Graphique : Consommation énergétique (Idle vs Charge)"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(df))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, df['Conso_Idle_W'], width, label='Idle', color='lightblue', edgecolor='black')
    bars2 = ax.bar(x + width/2, df['Conso_Charge_W'], width, label='Charge (IA)', color='coral', edgecolor='black')
    
    ax.set_xlabel('Device', fontweight='bold', fontsize=12)
    ax.set_ylabel('Consommation (W)', fontweight='bold', fontsize=12)
    ax.set_title('Consommation Énergétique : Idle vs Charge', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(df['Device'], rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Annotations
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}W', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}W', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "04_power_consumption.png"), dpi=300)
    print("✅ Graphique '04_power_consumption.png' généré.")
    plt.close()

def generate_temperature_plot(df):
    """Graphique : Températures (Idle vs Charge)"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(df))
    width = 0.35
    
    # Filtrer les NaN pour Temp_Idle_C
    temp_idle = df['Temp_Idle_C'].fillna(0)
    
    bars1 = ax.bar(x - width/2, temp_idle, width, label='Idle', color='lightgreen', edgecolor='black')
    bars2 = ax.bar(x + width/2, df['Temp_Charge_C'], width, label='Charge (IA)', color='orangered', edgecolor='black')
    
    ax.set_xlabel('Device', fontweight='bold', fontsize=12)
    ax.set_ylabel('Température (°C)', fontweight='bold', fontsize=12)
    ax.set_title('Températures : Idle vs Charge', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(df['Device'], rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Annotations
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        if not np.isnan(df['Temp_Idle_C'].iloc[i]):
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}°C', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}°C', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "05_temperature.png"), dpi=300)
    print("✅ Graphique '05_temperature.png' généré.")
    plt.close()

def generate_delta_ia_plot(df):
    """Graphique : Surconsommation due à l'IA (Delta_IA_W)"""
    plt.figure(figsize=(10, 6))
    df_sorted = df.sort_values(by="Delta_IA_W", ascending=True)
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(df_sorted)))
    
    bars = plt.barh(df_sorted['Device'], df_sorted['Delta_IA_W'], color=colors, edgecolor='black')
    plt.title("Surconsommation Énergétique due à l'IA", fontweight='bold', fontsize=14)
    plt.xlabel("Delta Puissance (W) - Moins est mieux", fontsize=12)
    
    for i, (bar, v) in enumerate(zip(bars, df_sorted['Delta_IA_W'])):
        plt.text(v + 0.05, i, f"{v:.2f} W", va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "06_delta_ia_power.png"), dpi=300)
    print("✅ Graphique '06_delta_ia_power.png' généré.")
    plt.close()

def generate_radar_chart(df):
    """Graphique : Radar Chart multi-critères (normalisé)"""
    # Sélection des critères à comparer
    categories = ['FPS', 'Efficacité\n(FPS/W)', 'Latence\n(inverse)', 'Conso\n(inverse)', 'Temp\n(inverse)']
    
    # Normalisation des données (0-1) avec inverse pour latence, conso, temp
    def normalize(values, inverse=False):
        min_val, max_val = values.min(), values.max()
        if max_val == min_val:
            return np.ones_like(values)
        normalized = (values - min_val) / (max_val - min_val)
        return 1 - normalized if inverse else normalized
    
    # Préparer les données normalisées
    data_norm = {
        'FPS': normalize(df['FPS'].values),
        'Efficacite': normalize(df['Efficacite_FPS_per_W'].values),
        'Latence': normalize(df['Latence_ms'].values, inverse=True),
        'Conso': normalize(df['Conso_Charge_W'].values, inverse=True),
        'Temp': normalize(df['Temp_Charge_C'].values, inverse=True),
    }
    
    # Configuration du radar
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Fermer le cercle
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    colors = plt.cm.Set2(np.linspace(0, 1, len(df)))
    
    for idx, device in enumerate(df['Device']):
        values = [
            data_norm['FPS'][idx],
            data_norm['Efficacite'][idx],
            data_norm['Latence'][idx],
            data_norm['Conso'][idx],
            data_norm['Temp'][idx]
        ]
        values += values[:1]  # Fermer le polygone
        
        ax.plot(angles, values, 'o-', linewidth=2, label=device, color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.set_title("Comparaison Multi-Critères (normalisée)", fontweight='bold', fontsize=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "07_radar_comparison.png"), dpi=300, bbox_inches='tight')
    print("✅ Graphique '07_radar_comparison.png' généré.")
    plt.close()

def generate_performance_ratio_plot(df):
    """Graphique : Ratio Performance/Consommation"""
    plt.figure(figsize=(10, 6))
    
    # Calculer le ratio FPS/Consommation
    df['Perf_Ratio'] = df['FPS'] / df['Conso_Charge_W']
    df_sorted = df.sort_values(by="Perf_Ratio")
    
    colors = plt.cm.summer(np.linspace(0.3, 0.9, len(df_sorted)))
    bars = plt.barh(df_sorted['Device'], df_sorted['Perf_Ratio'], color=colors, edgecolor='black')
    
    plt.title("Ratio Performance/Consommation", fontweight='bold', fontsize=14)
    plt.xlabel("FPS par Watt consommé (Plus est mieux)", fontsize=12)
    
    for i, (bar, v) in enumerate(zip(bars, df_sorted['Perf_Ratio'])):
        plt.text(v + 0.5, i, f"{v:.2f}", va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "08_performance_ratio.png"), dpi=300)
    print("✅ Graphique '08_performance_ratio.png' généré.")
    plt.close()

def generate_all_plots(df):
    """Génère tous les graphiques d'analyse"""
    print("\n📊 Génération des graphiques...")
    generate_fps_plot(df)
    generate_efficiency_plot(df)
    generate_latency_plot(df)
    generate_power_consumption_plot(df)
    generate_temperature_plot(df)
    generate_delta_ia_plot(df)
    generate_radar_chart(df)
    generate_performance_ratio_plot(df)
    print(f"\n✅ Tous les graphiques ont été générés dans '{OUTPUT_DIR}/'")

if __name__ == "__main__":
    print("🔍 Agrégation des résultats...")
    df_res = aggregate_all_results()
    
    print("\n" + "="*60)
    print("--- SYNTHÈSE GLOBALE DES BENCHMARKS ---")
    print("="*60)
    print(df_res.to_string(index=False))
    print("="*60)
    
    # Sauvegarde du CSV complet
    output_csv = os.path.join(OUTPUT_DIR, "synthese_globale_projet.csv")
    df_res.to_csv(output_csv, index=False)
    print(f"\n💾 CSV complet sauvegardé : {output_csv}")
    
    # Génération de tous les graphiques
    generate_all_plots(df_res)
    
    print("\n✨ Analyse terminée avec succès !")