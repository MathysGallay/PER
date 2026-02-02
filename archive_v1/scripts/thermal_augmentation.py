import cv2
import os
import random
import numpy as np
from glob import glob
from tqdm import tqdm

INPUT_DIR = "thermal_coco/images"
OUTPUT_DIR = "thermal_coco/augmented"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def augment(img):
    # Flip horizontal
    if random.random() < 0.5:
        img = cv2.flip(img, 1)

    # Flip vertical
    if random.random() < 0.2:
        img = cv2.flip(img, 0)

    # Crop & resize
    if random.random() < 0.4:
        h, w = img.shape[:2]
        crop_x = random.randint(0, w // 8)
        crop_y = random.randint(0, h // 8)
        img = img[crop_y:h-crop_y, crop_x:w-crop_x]
        img = cv2.resize(img, (w, h))

    # Random blur
    if random.random() < 0.3:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    # Noise
    if random.random() < 0.6:
        noise = np.random.normal(0, 8, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img


images = glob(f"{INPUT_DIR}/*.jpg")

for path in tqdm(images, desc="Augmentation thermique"):
    # Lire en couleur car les images thermiques sont déjà en RGB avec colormap
    img_color = cv2.imread(path)

    base = os.path.basename(path)

    # 5 augmentations par image
    for i in range(5):
        aug = augment(img_color)
        cv2.imwrite(f"{OUTPUT_DIR}/{base.replace('.jpg', f'_aug{i}.jpg')}", aug)

print("Augmentation thermique terminée !")
