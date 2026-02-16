#!/bin/bash

# --- CONFIGURATION ---
ENV_PATH="$HOME/per_env"  # Chemin vers ton environnement virtuel

# Activation de l'environnement virtuel
if [ -f "$ENV_PATH/bin/activate" ]; then
    echo "Activation de l'environnement : per_env"
    source "$ENV_PATH/bin/activate"
else
    echo "ERREUR : Environnement introuvable dans $ENV_PATH"
    exit 1
fi

echo "========================================================"
echo "   BENCHMARK GLOBAL : PYTORCH vs TENSORRT"
echo "========================================================"

# ---------------------------------------------------------
# PARTIE 1 : PYTORCH (BASELINE + CHARGE)
# ---------------------------------------------------------
echo ""
echo "Entrée dans le dossier : jetson_pt"
cd jetson_pt || { echo "Dossier jetson_pt introuvable"; exit 1; }

echo "[1/4] Mesure PLANCHER (Repos commun)..."
echo "      Ne touchez à rien pendant 60s."
python3 benchmark_plancher.py

echo ""
echo "[2/4] Mesure CHARGE (PyTorch Classique)..."
python3 benchmark_charge.py

echo ""
echo "Analyse des résultats PyTorch..."
python3 analyze_results.py

# On remonte à la racine
cd ..

# ---------------------------------------------------------
# TRANSFERT DE DONNÉES (ASTUCE)
# ---------------------------------------------------------
echo ""
echo "Transfert des données de repos vers TensorRT..."
# On crée le dossier results s'il n'existe pas encore
mkdir -p jetson_trt/results
# On copie le fichier plancher.csv pour éviter de refaire le test
cp /home/jetson/benchmark/results/jetson_pt/plancher.csv /home/jetson/benchmark/results/jetson_trt/
echo "Fichier plancher.csv copié."

# ---------------------------------------------------------
# PAUSE THERMIQUE
# ---------------------------------------------------------
echo ""
echo "PAUSE REFROIDISSEMENT (30s)..."
sleep 30

# ---------------------------------------------------------
# PARTIE 2 : TENSORRT (CHARGE UNIQUEMENT)
# ---------------------------------------------------------
echo ""
echo "Entrée dans le dossier : jetson_trt"
cd jetson_trt || { echo "Dossier jetson_trt introuvable"; exit 1; }

echo "[3/4] Mesure CHARGE (TensorRT .engine)..."
python3 benchmark_charge.py

echo ""
echo "Analyse des résultats TensorRT..."
python3 analyze_results.py

# On remonte à la racine
cd ..

echo "========================================================"
echo "BENCHMARK TERMINÉ !"
echo "   - Rapport PyTorch  : jetson_pt/results/rapport_final.csv"
echo "   - Rapport TensorRT : jetson_trt/results/rapport_final.csv"
echo "========================================================"