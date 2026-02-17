import time
import asyncio
import json
import csv
import numpy as np
import cv2
import threading
import datetime
import os
import signal
import sys

# Ajout du répertoire parent pour l'import de l'interpréteur
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from interpretation.trt_llm_interpreter import TrtLlmInterpreter
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# --- IMPORT SPÉCIFIQUE JETSON (YOLO) ---
from ultralytics import YOLO

# --- CONFIGURATION ---
# On pointe directement vers ton fichier moteur TensorRT
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.engine")
CSV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detections.csv")
COOLDOWN_TIME = 5.0
MIN_AREA = 20
CAMERA_INDEX = 0

# Variables Globales
last_frame_heatmap = None
last_frame_inference = None
frame_lock = threading.Lock()
csv_lock = threading.Lock()

accumulator = None
last_movement_time = 0
is_inferencing = True  # On commence actif par défaut
last_interpretation = ""  # Texte d'interprétation courant

# Initialisation de l'interpréteur
interpreter = TrtLlmInterpreter()

# Répertoire de base = dossier du script (src/inference/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Initialisation FastAPI
app = FastAPI()
# Création des dossiers si inexistants pour éviter les erreurs
os.makedirs(os.path.join(BASE_DIR, "static"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "templates"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "heatmap"), exist_ok=True)

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# --- CHARGEMENT DU MODÈLE YOLO ---
print(f"Chargement du modèle {MODEL_PATH}...")
# Remplacez votre bloc de chargement actuel par celui-ci :
try:
    # On charge directement en mode "predict" pour éviter que YOLO cherche à charger 
    # des fonctionnalités d'entraînement incompatibles avec le .engine
    model = YOLO(MODEL_PATH, task='detect') 
    print(f"Modèle {MODEL_PATH} chargé avec succès.")
except Exception as e:
    print(f"Erreur chargement modèle: {e}")

# --- ROUTES FASTAPI (IDENTIQUES À LA VERSION PI) ---

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/history", response_class=HTMLResponse)
async def history(request: Request):
    detections = []
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'r') as f:
            reader = csv.DictReader(f)
            detections = list(reader)
            detections.reverse()
    return templates.TemplateResponse("history.html", {"request": request, "detections": detections})

@app.get("/status")
async def status_stream(request: Request):
    async def event_generator():
        while True:
            if await request.is_disconnected():
                break
            state = "INFERENCE ACTIVE" if is_inferencing else "IDLE (Monitoring)"
            data = json.dumps({
                "status": state,
                "active": is_inferencing,
                "interpretation": last_interpretation
            })
            yield f"data: {data}\n\n"
            await asyncio.sleep(0.5)
    return StreamingResponse(event_generator(), media_type="text/event-stream")

def generate_heatmap_feed():
    global last_frame_heatmap
    while True:
        with frame_lock:
            if last_frame_heatmap is None:
                frame = np.zeros((384, 512, 3), dtype=np.uint8)
            else:
                frame = last_frame_heatmap
        
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.05)

def generate_inference_feed():
    global last_frame_inference, is_inferencing
    while True:
        with frame_lock:
            if last_frame_inference is None or not is_inferencing:
                frame = np.zeros((256, 256, 3), dtype=np.uint8)
                cv2.putText(frame, "PAUSED (No Motion)", (50, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            else:
                frame = last_frame_inference
        
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.05)

@app.get("/heatmap_feed")
def heatmap_feed():
    return StreamingResponse(generate_heatmap_feed(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/inference_feed")
def inference_feed():
    return StreamingResponse(generate_inference_feed(), media_type="multipart/x-mixed-replace; boundary=frame")

def run_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info")

# --- FONCTIONS MÉTIER ---

def log_detection_to_csv(results):
    """
    Adapte les résultats de YOLO (Ultralytics) pour le format CSV existant.
    """
    global CSV_FILE
    
    # YOLO retourne une liste de résultats (une par frame)
    result = results[0] 
    boxes = result.boxes
    
    if len(boxes) == 0:
        return

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    
    with csv_lock:
        file_exists = os.path.exists(CSV_FILE)
        with open(CSV_FILE, 'a') as f:
            if not file_exists:
                f.write("Timestamp,Label,Confidence,BBox\n")
            
            for box in boxes:
                # Extraction des données YOLO
                coords = box.xyxy[0].tolist() # [x1, y1, x2, y2]
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = result.names[cls_id]
                
                bbox_str = f"[{coords[0]:.2f},{coords[1]:.2f},{coords[2]:.2f},{coords[3]:.2f}]"
                f.write(f"{timestamp},{label},{conf:.2f},{bbox_str}\n")

def process_heatmap_frame(frame_rgb):
    """
    Même logique que sur la Pi : Background Subtraction pour détecter le mouvement.
    """
    global accumulator
    
    # Initialisation statique du background subtractor
    if not hasattr(process_heatmap_frame, "fgbg"):
        process_heatmap_frame.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
    
    # Appliquer le masque
    fgmask = process_heatmap_frame.fgbg.apply(frame_rgb)
    
    # Nettoyage morphologique
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    fgmask = cv2.dilate(fgmask, kernel, iterations=2)
    
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    clean_mask = np.zeros_like(fgmask)
    motion_detected = False
    
    for cnt in contours:
        if cv2.contourArea(cnt) > MIN_AREA:
            cv2.drawContours(clean_mask, [cnt], -1, 255, thickness=cv2.FILLED)
            motion_detected = True
            
    if accumulator is None:
        accumulator = np.zeros((frame_rgb.shape[0], frame_rgb.shape[1]), dtype=np.float32)
        
    motion_binary = (clean_mask > 0).astype(np.float32)
    accumulator += motion_binary

    # Normalisation pour affichage
    max_val = np.max(accumulator)
    if max_val > 0:
        heatmap_norm = (accumulator / max_val) * 255
    else:
        heatmap_norm = accumulator
    
    heatmap_norm = np.clip(heatmap_norm, 0, 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)

    # Redimensionnement si nécessaire (sécurité)
    if frame_rgb.shape[:2] != heatmap_color.shape[:2]:
        heatmap_color = cv2.resize(heatmap_color, (frame_rgb.shape[1], frame_rgb.shape[0]))
        
    overlay = cv2.addWeighted(frame_rgb, 0.6, heatmap_color, 0.4, 0)
    
    return overlay, motion_detected

def save_artifacts():
    print("\nSauvegarde des artefacts Heatmap...")
    if accumulator is not None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Sauvegarde numpy
        npy_path = os.path.join(BASE_DIR, "heatmap", f"heatmap_final_{timestamp}.npy")
        np.save(npy_path, accumulator)
        print(f"Matrice sauvegardée: {npy_path}")
        
        # Sauvegarde image PNG
        max_val = np.max(accumulator)
        if max_val > 0:
            heatmap_norm = (accumulator / max_val) * 255
        else:
            heatmap_norm = accumulator
        heatmap_norm = np.clip(heatmap_norm, 0, 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
        
        img_path = os.path.join(BASE_DIR, "heatmap", f"heatmap_final_{timestamp}.png")
        cv2.imwrite(img_path, heatmap_color)
        print(f"Image sauvegardée: {img_path}")
    else:
        print("Pas de données heatmap à sauvegarder.")

# --- BOUCLE PRINCIPALE (REMPLACE GSTREAMER) ---

def main_camera_loop():
    global last_frame_heatmap, last_frame_inference, is_inferencing, last_movement_time, last_interpretation
    
    print(f"Ouverture caméra index {CAMERA_INDEX}...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    
    # Configurer la résolution (pour correspondre à peu près au setup Pi)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("Erreur: Impossible d'ouvrir la caméra.")
        return

    print("Boucle principale démarrée.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erreur lecture frame caméra")
            time.sleep(0.1)
            continue

        # 1. Traitement Heatmap & Mouvement
        # On redimensionne pour le traitement heatmap (plus rapide sur petite image)
        frame_resized = cv2.resize(frame, (256, 192))
        heatmap_overlay, motion_detected = process_heatmap_frame(frame_resized)
        
        # Mise à jour stream Heatmap
        with frame_lock:
            last_frame_heatmap = heatmap_overlay

        # 2. Logique de "Valve" (Mouvement -> Inférence)
        current_time = time.time()
        
        if motion_detected:
            last_movement_time = current_time
            if not is_inferencing:
                print("MOUVEMENT DÉTECTÉ -> Activation Inférence")
                is_inferencing = True
        
        elif is_inferencing and (current_time - last_movement_time > COOLDOWN_TIME):
            print(f"COOLDOWN EXPIRÉ ({COOLDOWN_TIME}s) -> Arrêt Inférence")
            is_inferencing = False

        # 3. Inférence (Si actif)
        if is_inferencing:
            # YOLO sur Jetson est rapide, on peut lui envoyer l'image originale ou resize
            # On utilise stream=True pour ne pas accumuler en mémoire
            results = model.predict(frame, imgsz=256, verbose=False, stream=False)
            
            # Logging CSV
            log_detection_to_csv(results)
            
            # Interprétation des détections
            result = results[0]
            detections = []
            for box in result.boxes:
                cls_id = int(box.cls[0])
                detections.append({
                    "label": result.names[cls_id],
                    "conf": float(box.conf[0])
                })
            last_interpretation = interpreter.interpret(detections)
            
            # Dessin des boîtes (Annotated frame)
            annotated_frame = results[0].plot()
            
            # Mise à jour stream Inférence
            with frame_lock:
                last_frame_inference = annotated_frame
        
        # Petit sleep pour ne pas saturer le CPU si la cam est lente
        time.sleep(0.01)

    cap.release()

def main():
    # Lancement du serveur Web en background
    server_thread = threading.Thread(target=run_fastapi)
    server_thread.daemon = True
    server_thread.start()
    print(f"Serveur Web démarré sur http://127.0.0.1:5000")

    # Lancement de la boucle caméra/IA (Main Thread)
    try:
        main_camera_loop()
    except KeyboardInterrupt:
        print("\nArrêt demandé...")
    finally:
        save_artifacts()
        print("Nettoyage terminé.")

if __name__ == "__main__":
    main()