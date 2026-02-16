import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import hailo
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
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

HEF_PATH = "/home/yanis/PER/yolov5_custom_hailo8l.hef"
POSTPROC_SO = "/usr/local/hailo/resources/so/libyolo_hailortpp_postprocess.so"
CSV_FILE = "detections.csv"
COOLDOWN_TIME = 5.0
MIN_AREA = 20

last_frame_heatmap = None
last_frame_inference = None
frame_lock = threading.Lock()

accumulator = None
valve_element = None
last_movement_time = 0
is_inferencing = True

csv_lock = threading.Lock()

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

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
            data = json.dumps({"status": state, "active": is_inferencing})
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

def log_detection_to_csv(detections):
    global CSV_FILE
    if not detections:
        return

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    
    with csv_lock:
        file_exists = os.path.exists(CSV_FILE)
        with open(CSV_FILE, 'a') as f:
            if not file_exists:
                f.write("Timestamp,Label,Confidence,BBox\n")
            
            for detection in detections:
                label = detection.get_label()
                confidence = detection.get_confidence()
                bbox = detection.get_bbox()
                bbox_str = f"[{bbox.xmin():.2f},{bbox.ymin():.2f},{bbox.xmax():.2f},{bbox.ymax():.2f}]"
                f.write(f"{timestamp},{label},{confidence:.2f},{bbox_str}\n")


def process_heatmap_frame(frame_rgb):
    global accumulator, fgbg
    
    if not hasattr(process_heatmap_frame, "fgbg"):
        process_heatmap_frame.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
    
    fgmask = process_heatmap_frame.fgbg.apply(frame_rgb)
    
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

    max_val = np.max(accumulator)
    if max_val > 0:
        heatmap_norm = (accumulator / max_val) * 255
    else:
        heatmap_norm = accumulator
    
    heatmap_norm = np.clip(heatmap_norm, 0, 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)

    if frame_rgb.shape[:2] != heatmap_color.shape[:2]:
        heatmap_color = cv2.resize(heatmap_color, (frame_rgb.shape[1], frame_rgb.shape[0]))
        
    overlay = cv2.addWeighted(frame_rgb, 0.6, heatmap_color, 0.4, 0)
    
    return overlay, motion_detected



def probe_callback(pad, info, user_data):
    buffer = info.get_buffer()
    if not buffer: return Gst.PadProbeReturn.OK
    
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    
    log_detection_to_csv(detections)
    
    return Gst.PadProbeReturn.OK

def on_inference_sample(sink):
    global last_frame_inference
    
    sample = sink.emit("pull-sample")
    if not sample: return Gst.FlowReturn.ERROR
    
    buffer = sample.get_buffer()
    success, map_info = buffer.map(Gst.MapFlags.READ)
    if not success: return Gst.FlowReturn.ERROR
    
    try:
        caps = sample.get_caps()
        h = caps.get_structure(0).get_value('height')
        w = caps.get_structure(0).get_value('width')
        frame = np.ndarray((h, w, 3), buffer=map_info.data, dtype=np.uint8)
        
        with frame_lock:
            last_frame_inference = frame
            
    finally:
        buffer.unmap(map_info)
        
    return Gst.FlowReturn.OK

def save_artifacts():
    print("\nSaving final heatmap artifacts...")
    if accumulator is not None:
        if not os.path.exists("heatmap"):
            os.makedirs("heatmap")
            
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        npy_path = os.path.join("heatmap", f"heatmap_final_{timestamp}.npy")
        np.save(npy_path, accumulator)
        print(f"Saved Matrix: {npy_path}")
        
        max_val = np.max(accumulator)
        if max_val > 0:
            heatmap_norm = (accumulator / max_val) * 255
        else:
            heatmap_norm = accumulator
        heatmap_norm = np.clip(heatmap_norm, 0, 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
        
        img_path = os.path.join("heatmap", f"heatmap_final_{timestamp}.png")
        cv2.imwrite(img_path, heatmap_color)
        print(f"Saved Image: {img_path}")
    else:
        print("No accumulator data to save.")

def main():
    global valve_element
    
    Gst.init(None)
    
    pipeline_str = (
        "v4l2src device=/dev/video0 io-mode=2 ! "
        "video/x-raw, width=256, height=192, format=YUY2 ! "
        "tee name=t ! "
        "queue max-size-buffers=5 leaky=downstream ! "
        "videoconvert ! video/x-raw, format=RGB ! "
        "appsink name=sink_heatmap emit-signals=True max-buffers=1 drop=True sync=False "
        "t. ! "
        "valve name=gated_valve drop=false ! " 
        "queue max-size-buffers=5 leaky=downstream ! "
        "videocrop left=32 right=32 ! " 
        "videoscale ! "
        "video/x-raw, width=256, height=256 ! "
        "videoconvert ! "

        "hailonet hef-path=/home/yanis/PER/yolov5_custom_hailo8l.hef batch-size=1 ! "
        "queue max-size-buffers=5 leaky=downstream ! "
        "hailofilter name=filter so-path=/usr/local/hailo/resources/so/libyolo_hailortpp_postprocess.so "
        "function-name=filter qos=false ! "
        "queue max-size-buffers=5 leaky=downstream ! "
        "hailooverlay ! "
        "videoconvert ! video/x-raw, format=RGB ! "
        "appsink name=sink_inference emit-signals=True max-buffers=1 drop=True sync=False"
    )
    
    try:
        pipeline = Gst.parse_launch(pipeline_str)
    except Exception as e:
        print(f"Pipeline Build Error: {e}")
        return


    valve_element = pipeline.get_by_name("gated_valve")
    if not valve_element:
        print("Error: Could not find valve element")
        return

    sink_heatmap = pipeline.get_by_name("sink_heatmap")
    sink_heatmap.set_property("emit-signals", False)

    sink_heatmap.set_property("drop", True)
    sink_heatmap.set_property("max-buffers", 1)
    sink_heatmap.set_property("sync", False)
    
    hailo_filter = pipeline.get_by_name("filter") 
    if hailo_filter:
        pad = hailo_filter.get_static_pad("src")
        pad.add_probe(Gst.PadProbeType.BUFFER, probe_callback, None)
        
    sink_inference = pipeline.get_by_name("sink_inference")
    sink_inference.connect("new-sample", on_inference_sample)
    
    server_thread = threading.Thread(target=run_fastapi)
    server_thread.daemon = True
    server_thread.start()

    def poll_heatmap_loop():
        print("Heatmap polling thread started")
        sink = sink_heatmap
        while True:
            try:
                sample = sink.emit("pull-sample")

                if sample:
                    buffer = sample.get_buffer()
                    if buffer:
                        success, map_info = buffer.map(Gst.MapFlags.READ)
                        if success:
                            try:
                                caps = sample.get_caps()
                                h = caps.get_structure(0).get_value('height')
                                w = caps.get_structure(0).get_value('width')
                                frame = np.ndarray((h, w, 3), buffer=map_info.data, dtype=np.uint8)
                                
                                display_frame, motion_detected = process_heatmap_frame(frame)
                                
                                with frame_lock:
                                    global last_frame_heatmap
                                    last_frame_heatmap = display_frame
                                    
                                current_time = time.time()
                                global is_inferencing, last_movement_time, valve_element
                                
                                if motion_detected:
                                    last_movement_time = current_time
                                    if not is_inferencing:
                                        print("MOTION DETECTED -> Opening Valve (Starting Inference)")
                                        is_inferencing = True
                                        if valve_element:
                                            valve_element.set_property("drop", False)
                                
                                elif is_inferencing and (current_time - last_movement_time > COOLDOWN_TIME):
                                        print(f"COOLDOWN EXPIRED ({COOLDOWN_TIME}s) -> Closing Valve (Stopping Inference)")
                                        is_inferencing = False
                                        if valve_element:
                                            valve_element.set_property("drop", True)
                            finally:
                                buffer.unmap(map_info)
                else:
                    time.sleep(0.01)
            except Exception as e:
                print(f"Polling Error: {e}")
                time.sleep(0.1)

    poll_thread = threading.Thread(target=poll_heatmap_loop)
    poll_thread.daemon = True
    poll_thread.start()

    print(f"Starting pipeline...")
    print(f"Mode: On-Demand Inference (Cooldwon: {COOLDOWN_TIME}s)")
    print(f"CSV Logging: {CSV_FILE}")
    print(f"Dashboard: http://127.0.0.1:5000")
    
    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        print("ERROR: Unable to set the pipeline to the playing state.")
        return
    print(f"Pipeline state set to PLAYING: {ret}")
    loop = GLib.MainLoop()
    
    try:
        loop.run()
    except KeyboardInterrupt:
        print("\nStopping...")
        loop.quit() 
    finally:
        pipeline.set_state(Gst.State.NULL)
        save_artifacts()

if __name__ == "__main__":
    main()