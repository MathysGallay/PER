import cv2
import numpy as np
from flask import Flask, Response

app = Flask(__name__)

cap = cv2.VideoCapture('heatmap_test_video.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

accumulator = None
alpha = 0.01

def generate_frames():
    global accumulator
    
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            accumulator = None
            continue

        frame = cv2.resize(frame, (256, 192))
        
        if accumulator is None:
            accumulator = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)

        fgmask = fgbg.apply(frame)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fgmask = cv2.erode(fgmask, kernel, iterations=1)
        fgmask = cv2.dilate(fgmask, kernel, iterations=2)

        accumulator += fgmask.astype(np.float32) * 0.5
        accumulator *= (1.0 - alpha) 
        
        heatmap_norm = np.clip(accumulator, 0, 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(frame, 0.6, heatmap_color, 0.4, 0)

        ret, buffer = cv2.imencode('.jpg', overlay)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return '''
    <html>
        <head><title>Heatmap Stream</title></head>
        <body>
            <h1>Heatmap Video Stream</h1>
            <img src="/video_feed" width="256" height="192">
        </body>
    </html>
    '''

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("Serveur Flask démarré sur http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)