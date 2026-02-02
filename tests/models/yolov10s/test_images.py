from ultralytics import YOLO
import cv2
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import random

def test_model_on_images(model_path, images_path, output_dir='results_images', conf_threshold=0.25, n_humans=3, n_animals=3, show_results=True):
    """
    Teste un modèle YOLO sur des images
    
    Args:
        model_path: Chemin vers le modèle (.pt)
        images_path: Chemin vers le dossier contenant les images ou chemin vers une image
        output_dir: Dossier de sortie pour les résultats
        conf_threshold: Seuil de confiance minimum pour les détections
        n_humans: Nombre d'images avec humains à sélectionner
        n_animals: Nombre d'images avec animaux à sélectionner
        show_results: Afficher les résultats avec matplotlib
    """
    # Charger le modèle
    print(f"Chargement du modèle: {model_path}")
    model = YOLO(model_path)
    
    # Créer le dossier de sortie
    os.makedirs(output_dir, exist_ok=True)
    
    # Récupérer toutes les images du dossier (sans sélection par labels)
    if os.path.isfile(images_path):
        image_files = [images_path]
    else:
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in extensions:
            image_files.extend([str(p) for p in Path(images_path).glob(ext)])
        random.shuffle(image_files)
    print(f"\nNombre d'images à traiter: {len(image_files)}")
    if not image_files:
        print("Aucune image trouvée!")
        return
    
    # Traiter chaque image
    annotated_images = []
    
    for idx, img_path in enumerate(image_files):
        print(f"\nTraitement de l'image {idx+1}/{len(image_files)}: {os.path.basename(img_path)}")
        
        # Faire la prédiction
        results = model.predict(
            source=img_path,
            conf=conf_threshold,
            save=False,
            verbose=False
        )
        
        # Récupérer les résultats
        result = results[0]
        
        # Afficher les détections
        if len(result.boxes) > 0:
            print(f"  {len(result.boxes)} détection(s) trouvée(s)")
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls]
                coords = box.xyxy[0].cpu().numpy()
                print(f"    - {class_name}: {conf:.2f} [x1:{coords[0]:.0f}, y1:{coords[1]:.0f}, x2:{coords[2]:.0f}, y2:{coords[3]:.0f}]")
        else:
            print("  Aucune détection")
        
        # Sauvegarder l'image annotée
        annotated_img = result.plot()
        output_path = os.path.join(output_dir, f"result_{os.path.basename(img_path)}")
        cv2.imwrite(output_path, annotated_img)
        
        # Stocker pour affichage
        if show_results:
            annotated_images.append((os.path.basename(img_path), cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)))
    
    print(f"\n✓ Résultats sauvegardés dans: {output_dir}")
    print(f"✓ {len(image_files)} images traitées")
    
    # Afficher et enregistrer les résultats en une seule image
    if show_results and annotated_images:
        n_images = len(annotated_images)
        cols = min(2, n_images)
        rows = (n_images + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(12, 6*rows))
        if n_images == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_images > 1 else [axes]
        for idx, (name, img) in enumerate(annotated_images):
            axes[idx].imshow(img)
            axes[idx].set_title(name)
            axes[idx].axis('off')
        for idx in range(n_images, len(axes)):
            axes[idx].axis('off')
        plt.tight_layout()
        # Enregistrement de la grille d'images
        grid_path = os.path.join(output_dir, "grille_resultats.png")
        fig.savefig(grid_path)
        plt.show()
        print(f"\n✓ Résultats affichés et grille enregistrée sous: {grid_path}")


if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "SIA_Final/v1_reduit4/weights/best.pt"  # Chemin vers votre modèle
    IMAGES_PATH = "images_test"  # Dossier contenant les images de test
    OUTPUT_DIR = "results_images"  # Dossier de sortie
    CONF_THRESHOLD = 0.25  # Seuil de confiance
    N_HUMANS = 3  # Nombre d'images avec humains
    N_ANIMALS = 3  # Nombre d'images avec animaux
    
    # Tester le modèle
    test_model_on_images(
        model_path=MODEL_PATH,
        images_path=IMAGES_PATH,
        output_dir=OUTPUT_DIR,
        conf_threshold=CONF_THRESHOLD,
        n_humans=N_HUMANS,
        n_animals=N_ANIMALS,
        show_results=True
    )
