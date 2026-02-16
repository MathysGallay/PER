#!/bin/bash

# --- CONFIGURATION ---
# Si tu utilises un environnement virtuel sur la Pi, mets son chemin ici.
# Sinon, laisse vide ou commenté.
ENV_PATH="$HOME/my_hailo_env"

echo "========================================================"
echo "   BENCHMARK RASPBERRY PI 5 + HAILO 8L"
echo "========================================================"

# Activation de l'environnement virtuel (Optionnel)
if [ -d "$ENV_PATH" ] && [ -f "$ENV_PATH/bin/activate" ]; then
    echo "Activation de l'environnement : $ENV_PATH"
    source "$ENV_PATH/bin/activate"
else
    echo "Aucun venv détecté (ou chemin incorrect), utilisation du python système."
fi

# Vérification rapide de la structure
if [ ! -d "data" ]; then
    echo "ERREUR : Le dossier 'data' est manquant."
    echo "   Veuillez créer 'data' et y mettre votre fichier .hef"
    exit 1
fi

# ---------------------------------------------------------
# ÉTAPE 1 : MESURE AU REPOS (IDLE)
# ---------------------------------------------------------
echo ""
echo "[1/3] Mesure PLANCHER (Repos)..."
echo "      Assurez-vous que rien ne tourne sur la Pi."
echo "      Durée : 60 secondes."

python3 benchmark_plancher.py

# ---------------------------------------------------------
# PAUSE THERMIQUE
# ---------------------------------------------------------
echo ""
echo " PAUSE REFROIDISSEMENT (10s)..."
sleep 10

# ---------------------------------------------------------
# ÉTAPE 2 : MESURE EN CHARGE (HAILO)
# ---------------------------------------------------------
echo ""
echo "[2/3] Mesure CHARGE (Inférence Hailo)..."
echo "      Le benchmark Hailo va tourner en arrière-plan."
echo "      ⏳ Durée : 60 secondes."

python3 benchmark_charge.py

# ---------------------------------------------------------
# ÉTAPE 3 : ANALYSE ET RAPPORT
# ---------------------------------------------------------
echo ""
echo "[3/3] Analyse des résultats..."
echo " ATTENTION : Préparez-vous à saisir les Watts manuellement."
echo ""

python3 analyze_results.py

echo ""
echo "========================================================"
echo "BENCHMARK TERMINÉ !"
echo "   - Rapport TXT : results/pi_hailo/rapport_final.txt"
echo "   - Données CSV : results/pi_hailo/rapport_final.csv"
echo "========================================================"