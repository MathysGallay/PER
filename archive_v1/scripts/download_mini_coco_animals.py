import os
import requests
import zipfile
import json
from tqdm import tqdm

# ---------------------------------------------
# 1. Répertoires
# ---------------------------------------------
os.makedirs("mini_coco/images", exist_ok=True)
os.makedirs("mini_coco/labels", exist_ok=True)

# ---------------------------------------------
# 2. Télécharger annotations COCO
# ---------------------------------------------
ANNOT_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

if not os.path.exists("annotations_trainval2017.zip"):
    print("Téléchargement des annotations COCO...")
    r = requests.get(ANNOT_URL, stream=True)
    with open("annotations_trainval2017.zip", "wb") as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)

print("Extraction annotations...")
with zipfile.ZipFile("annotations_trainval2017.zip") as z:
    z.extractall(".")

# ---------------------------------------------
# 3. Classes animales
# ---------------------------------------------
ANIMAL_CLASSES = {
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
}

# ---------------------------------------------
# 4. Charger COCO JSON
# ---------------------------------------------
with open("annotations/instances_train2017.json") as f:
    coco = json.load(f)

img_id_to_labels = {}
selected_images = set()

# ---------------------------------------------
# 5. Filtrer uniquement les animaux
# ---------------------------------------------
for ann in coco["annotations"]:
    if ann["category_id"] in ANIMAL_CLASSES:
        img_id = ann["image_id"]
        selected_images.add(img_id)
        img_id_to_labels.setdefault(img_id, []).append(ann)

print(f"Nombre d'images trouvées : {len(selected_images)}")

MAX_IMAGES = 500
selected_images = set(list(selected_images)[:MAX_IMAGES])
# ---------------------------------------------
# 6. Télécharger uniquement les images nécessaires
# ---------------------------------------------
IMG_BASE_URL = "http://images.cocodataset.org/train2017/"
downloaded = 0
for img in tqdm(coco["images"], desc="Téléchargement images"):
    if img["id"] in selected_images:
        file_name = img["file_name"]
        url = IMG_BASE_URL + file_name
        save_path = f"mini_coco/images/{file_name}"

        if not os.path.exists(save_path):
            try:
                r = requests.get(url, timeout=10)
                with open(save_path, "wb") as f:
                    f.write(r.content)
                downloaded += 1
            except Exception as e:
                print(f"Erreur pour {file_name}: {e}")

print(f"Images téléchargées : {downloaded}")

# ---------------------------------------------
# 7. Créer les labels YOLO
# ---------------------------------------------
for img in coco["images"]:
    if img["id"] not in selected_images:
        continue

    w, h = img["width"], img["height"]
    file = img["file_name"].replace(".jpg", ".txt")
    label_path = f"mini_coco/labels/{file}"

    with open(label_path, "w") as f:
        for ann in img_id_to_labels[img["id"]]:
            cat = ann["category_id"]
            if cat not in ANIMAL_CLASSES:
                continue

            # bbox COCO → YOLO
            x, y, bw, bh = ann["bbox"]
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            bw /= w
            bh /= h

            # ID YOLO = index dans liste
            cls_id = list(ANIMAL_CLASSES.keys()).index(cat)

            f.write(f"{cls_id} {cx} {cy} {bw} {bh}\n")

print("Mini-COCO (animaux uniquement) généré")
