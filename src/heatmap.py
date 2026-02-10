import cv2
import numpy as np
from flask import Flask, Response, redirect, url_for
import time

app = Flask(__name__)

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 192)

time.sleep(2)

if not cap.isOpened():
    print("Erreur: Impossible d'ouvrir le flux vidéo (V4L2)")
    exit(1)

fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

accumulator = None

MIN_AREA = 20 

def generate_frames():
    global accumulator
    
    while True:
        
        ret, frame = cap.read()
        if not ret:
            print("Erreur lecture frame")
            time.sleep(0.1)
            continue
        
        if accumulator is None:
            accumulator = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)

        fgmask = fgbg.apply(frame)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fgmask = cv2.erode(fgmask, kernel, iterations=1)
        fgmask = cv2.dilate(fgmask, kernel, iterations=2)

        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        clean_mask = np.zeros_like(fgmask)
        
        for cnt in contours:
            if cv2.contourArea(cnt) > MIN_AREA:
                cv2.drawContours(clean_mask, [cnt], -1, 255, thickness=cv2.FILLED)

        motion_binary = (clean_mask > 0).astype(np.float32)
        
        accumulator += motion_binary

        max_val = np.max(accumulator)
        if max_val > 0:
            heatmap_norm = (accumulator / max_val) * 255
        else:
            heatmap_norm = accumulator

        heatmap_norm = np.clip(heatmap_norm, 0, 255).astype(np.uint8)
        
        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(frame, 0.6, heatmap_color, 0.4, 0)

        ret, buffer = cv2.imencode('.jpg', overlay)
        if not ret: continue    
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/reset')
def reset_heatmap():
    global accumulator
    if accumulator is not None:
        accumulator.fill(0)
    return redirect(url_for('index'))

@app.route('/')
def index():
    return '''
    <html>
        <head>
            <title>Heatmap Filtrée</title>
            <style>
                body { background: #111; text-align: center; color: white; font-family: sans-serif; }
                .btn {
                    display: inline-block; padding: 10px 20px; margin-top: 20px;
                    background-color: #d9534f; color: white; text-decoration: none;
                    border-radius: 5px; font-size: 18px; border: none; cursor: pointer;
                }
            </style>
        </head>
        <body>
            <h1>Heatmap</h1>
            <img src="/video_feed" style="width: 512px; height: 384px; image-rendering: pixelated; border: 2px solid red;">
            <br>
            <a href="/reset"><button class="btn">RESET</button></a>
        </body>
    </html>
    '''

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("Serveur Flask démarré sur http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)