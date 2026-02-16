import cv2
import time

print("🔍 SCAN DES CAMÉRAS (Windows DirectShow)...")
print("---------------------------------------------")

# On teste les index de 0 à 3
for index in range(4):
    print(f"Testing Index {index}...", end=" ")
    
    # L'astuce : cv2.CAP_DSHOW est vital pour les caméras thermiques/indus sur Windows
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            h, w, _ = frame.shape
            print(f"✅ TROUVÉ !")
            print(f"   -> Résolution : {w}x{h}")
            print(f"   -> Pour vérifier : Une fenêtre va s'ouvrir. Appuie sur 'q' pour fermer.")
            
            # On affiche l'image pour que tu sois sûr que c'est la Topdon
            while True:
                ret, frame = cap.read()
                if not ret: break
                cv2.imshow(f"Camera Index {index}", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cv2.destroyAllWindows()
        else:
            print("❌ Ouverte mais image vide (Webcam bloquée ?)")
    else:
        print("❌ Pas de caméra.")
    
    cap.release()

print("---------------------------------------------")
print("Scan terminé.")