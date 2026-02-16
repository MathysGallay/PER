@echo off
cd /d "%~dp0"
title BENCHMARK PC NVIDIA (GPU FORCE)
color 0A

echo ========================================================
echo   BENCHMARK PC (NVIDIA-SMI)
echo ========================================================

:: 1. VERIFICATION ENVIRONNEMENT
:: Si on est dans Conda, on ne touche pas aux venv
if defined CONDA_DEFAULT_ENV (
    echo [INFO] Environnement Conda detecte : %CONDA_DEFAULT_ENV%
    echo        On utilise l'environnement actuel.
) else (
    :: Gestion VENV standard si pas de Conda
    if not exist venv (
        echo [INFO] Pas de Conda detecte. Creation d'un venv...
        python -m venv venv
    )
    if exist venv\Scripts\activate.bat (
        echo [INFO] Activation du venv...
        call venv\Scripts\activate.bat
    )
)

:: 2. INSTALLATION FORCEE DU GPU (Le point critique)
echo.
echo [INSTALL] Verification des librairies GPU...

:: On installe d'abord les petits outils
pip install -r requirements.txt >nul 2>&1

:: On verifie si torch est la. Si non, on FORCE la version CUDA.
python -c "import torch; exit(0) if torch.cuda.is_available() else exit(1)" >nul 2>&1
if %errorlevel% neq 0 (
    echo [ATTENTION] PyTorch GPU manquant ou version CPU detectee !
    echo [INSTALL] Installation forcee de PyTorch CUDA 12.1...
    echo           (Cela peut prendre quelques minutes, ~2.5 Go)
    
    :: La commande magique pour Windows qui force le GPU
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
    :: Ensuite on installe Ultralytics (qui utilisera le torch qu'on vient d'installer)
    pip install ultralytics
) else (
    echo [OK] PyTorch GPU est deja pret.
    :: On s'assure juste qu'ultralytics est là
    pip install ultralytics >nul 2>&1
)

:: 3. ULTIME VERIFICATION AVANT DE LANCER
echo.
python -c "import torch; print(f'GPU ACTIF : {torch.cuda.is_available()} ({torch.cuda.get_device_name(0)})') if torch.cuda.is_available() else print('ERREUR : TOUJOURS PAS DE GPU !')"

echo.
echo ========================================================
echo [ETAPE 1/3] Mesure PLANCHER (Idle GPU)...
echo ========================================================
python benchmark_plancher.py

echo.
echo ========================================================
echo [PAUSE] Refroidissement (30s)...
echo ========================================================
timeout /t 30 /nobreak

echo.
echo ========================================================
echo [ETAPE 2/3] Mesure CHARGE (YOLO CUDA - V2 Synthétique)...
echo ========================================================
python benchmark_charge_v2.py

echo.
echo ========================================================
echo [ETAPE 3/3] Analyse et Rapport...
echo ========================================================
python analyze_results.py

echo.
echo ========================================================
echo   TEST TERMINE - Resultats dans results/pc/
echo ========================================================

pause