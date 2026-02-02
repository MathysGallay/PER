import cv2
import numpy as np
import os
from glob import glob
from tqdm import tqdm
import random

# ---------------------------------------
# PARAMÈTRES CAMÉRA TC001
# ---------------------------------------
THERMAL_W = 256
THERMAL_H = 192

# ---------------------------------------
# INPUT / OUTPUT
# ---------------------------------------
INPUT_DIR = "mini_coco/images"
OUTPUT_DIR = "thermal_coco/images"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------
# FONCTION : Ajouter du bruit thermique
# ---------------------------------------
def add_thermal_noise(img):
    noise = np.random.normal(0, 7, img.shape).astype(np.int16)
    img = img.astype(np.int16) + noise
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)

# ---------------------------------------
# FONCTION : Conversion RGB → Thermique Like
# ---------------------------------------
def rgb_to_thermal(path):
    img = cv2.imread(path)
    
    # 1. Downscale dès le départ (fidèle à la TC001)
    img = cv2.resize(img, (THERMAL_W, THERMAL_H), interpolation=cv2.INTER_LINEAR)

    # 2. Grayscale (caméra thermique = 1 canal)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Réduction de texture
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)

    # 4. Variation thermique globale
    alpha = random.uniform(0.8, 1.4)  # contraste
    beta = random.randint(-20, 20)    # brightness
    gray = np.clip(alpha * gray + beta, 0, 255).astype(np.uint8)

    # 5. Bruit thermique réaliste
    gray = add_thermal_noise(gray)

    # 6. Inverser pour que les objets clairs (animaux) soient chauds (rouge/jaune)
    gray_inverted = 255 - gray

    # 7. Appliquer colormap thermique (JET : bleu=froid, rouge=chaud)
    thermal_color = cv2.applyColorMap(gray_inverted, cv2.COLORMAP_JET)

    return gray_inverted, thermal_color

# ---------------------------------------
# BOUCLE PRINCIPALE
# ---------------------------------------
images = glob(f"{INPUT_DIR}/*.jpg")

for img_path in tqdm(images, desc="Conversion RGB → Thermal"):
    gray, thermal_color = rgb_to_thermal(img_path)

    base = os.path.basename(img_path)
    save_path = f"{OUTPUT_DIR}/{base}"

    # Enregistrer en 3 canaux avec colormap thermique
    cv2.imwrite(save_path, thermal_color)

print("Conversion thermique terminée !")
print(f"Images enregistrées dans {OUTPUT_DIR}")
