import os
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

# --- CONFIGURATION ---
MODEL_PATH = "SIA_Final/v1_reduit4/weights/best.pt"
OUTPUT_DIR = "results_images/tests_specials"
CONF_THRESHOLD = 0.25

# Liste des dossiers de tests spéciaux
TESTS = {
    "animaux_exotiques": "test_specials/animaux_exotiques",  # images de singes, chiens, chats, etc.
    "humains_postures": "test_specials/humains_postures",      # humains dans des postures inhabituelles
    "images_bruitees": "test_specials/images_bruitees",        # images bruitées/floues
    "conditions_extremes": "test_specials/conditions_extremes",# images en conditions extrêmes
    "aucun_sujet": "test_specials/aucun_sujet",                # images sans humain ni animal
    "objets_ressemblants": "test_specials/objets_ressemblants" # mannequins, statues, etc.
}

def test_special_images(model_path, test_dirs, output_dir, conf_threshold=0.25):
    model = YOLO(model_path)
    os.makedirs(output_dir, exist_ok=True)
    for test_name, folder in test_dirs.items():
        folder_path = Path(folder)
        if not folder_path.exists():
            print(f"Dossier non trouvé: {folder_path}")
            continue
        out_dir = Path(output_dir) / test_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for img_file in folder_path.glob("*.jpg"):
            img = cv2.imread(str(img_file))
            results = model.predict(source=str(img_file), conf=conf_threshold, save=False, verbose=False)
            result = results[0]
            # Annoter l'image
            annotated = result.plot()
            out_path = out_dir / img_file.name
            cv2.imwrite(str(out_path), annotated)
            print(f"[✓] {test_name}: {img_file.name} → {out_path}")

if __name__ == "__main__":
    test_special_images(MODEL_PATH, TESTS, OUTPUT_DIR, CONF_THRESHOLD)
    print("\nTous les tests spéciaux sont terminés.")
