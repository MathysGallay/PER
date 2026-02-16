import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import hailo
import time
import sys

from flask import Flask, Response
import threading
import io

app = Flask(__name__)
frame_lock = threading.Lock()
last_frame = None

class UserData:
    def __init__(self):
        self.frame_count = 0
        self.start_time = time.time()
        self.last_time = time.time()
        self.fps_interval_count = 0

    def increment(self):
        self.frame_count += 1
        self.fps_interval_count += 1
        
        current_time = time.time()
        if current_time - self.last_time >= 1.0:
            fps = self.fps_interval_count / (current_time - self.last_time)
            print(f"FPS: {fps:.2f}")
            self.last_time = current_time
            self.fps_interval_count = 0

    def get_count(self):
        return self.frame_count

def app_callback(pad, info, user_data):
    user_data.increment()
    return Gst.PadProbeReturn.OK

def on_new_sample(sink):
    global last_frame
    sample = sink.emit("pull-sample")
    if not sample:
        return Gst.FlowReturn.ERROR
    
    buffer = sample.get_buffer()
    success, map_info = buffer.map(Gst.MapFlags.READ)
    if not success:
        return Gst.FlowReturn.ERROR
    
    try:
        with frame_lock:
            last_frame = map_info.data
    finally:
        buffer.unmap(map_info)
        
    return Gst.FlowReturn.OK

def generate_frames():
    global last_frame
    while True:
        with frame_lock:
            if last_frame is None:
                frame_data = None
            else:
                frame_data = last_frame
        
        if frame_data:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
        
        time.sleep(0.04)
@app.route('/')
def index():
    return '''
    <html>
        <head>
            <title>Detection Thermique</title>
            <style>body { background: #111; text-align: center; color: white; }</style>
        </head>
        <body>
            <h1>Flux Thermique HAILO</h1>
            <img src="/video_feed" style="width: 640px; height: 640px; border: 2px solid red;">
        </body>
    </html>
    '''

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def run_flask():
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)

if __name__ == "__main__":
    print("Pre-requis: pip install flask")
    print("Lancement du stream sur http://0.0.0.0:5000")
    
    Gst.init(None)
    
    pipeline_str = (
        "v4l2src device=/dev/video0 ! "
        "video/x-raw, width=256, height=192, format=YUY2 ! "
        "videocrop left=32 right=32 ! " 
        "videoscale ! "
        "video/x-raw, width=256, height=256 ! "
        "videoconvert ! "
        "video/x-raw, format=RGB ! "
        "queue max-size-buffers=3 leaky=downstream ! "
        "hailonet hef-path=/home/yanis/PER/yolov5_custom_hailo8l.hef batch-size=1 ! "
        "queue max-size-buffers=3 leaky=downstream ! "
        "hailofilter name=filter so-path=/usr/local/hailo/resources/so/libyolo_hailortpp_postprocess.so "
        "function-name=filter qos=false ! "
        "queue max-size-buffers=3 leaky=downstream ! "
        "hailooverlay ! "
        "videoconvert ! "
        "jpegenc quality=85 ! "
        "appsink name=flask_sink emit-signals=True max-buffers=1 drop=True sync=False"
    )
    
    try:
        pipeline = Gst.parse_launch(pipeline_str)
    except Exception as e:
        print(f"Erreur pipeline: {e}")
        sys.exit(1)
    
    # Callback pour FPS
    hailo_filter = pipeline.get_by_name("filter")
    if hailo_filter:
        src_pad = hailo_filter.get_static_pad("src")
        user_data = UserData()
        src_pad.add_probe(Gst.PadProbeType.BUFFER, app_callback, user_data)
        
    # Callback pour Flask
    sink = pipeline.get_by_name("flask_sink")
    if sink:
        sink.connect("new-sample", on_new_sample)
    
    # Démarrer Flask dans un thread
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()
    
    pipeline.set_state(Gst.State.PLAYING)
    loop = GLib.MainLoop()
    try:
        loop.run()
    except KeyboardInterrupt:
        print("\nArrêt...")
        pipeline.set_state(Gst.State.NULL)