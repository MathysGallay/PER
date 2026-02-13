import os
import sys
import argparse
import numpy as np
import cv2
from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams, InputVStreamParams, OutputVStreamParams, FormatType)

# YOLOv5 Anchors (Standard for 640, might need adjustment for 256 if trained differently)
# However, for 256, strides are usually 8, 16, 32.
ANCHORS = [
    [[10, 13], [16, 30], [33, 23]],       # P3/8
    [[30, 61], [62, 45], [59, 119]],      # P4/16
    [[116, 90], [156, 198], [373, 326]]   # P5/32
]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def process_output(output, anchor_grid, stride, num_classes):
    # output: (Batch, Grid_Y, Grid_X, Anchors * (5 + Classes))
    # Reshape to (Batch, Anchors, Grid_Y, Grid_X, 5 + Classes)
    b, h, w, c = output.shape
    num_anchors = len(anchor_grid)
    output = output.reshape(b, h, w, num_anchors, 5 + num_classes)
    # Permute to (Batch, Anchors, Grid_Y, Grid_X, ...)
    # Actually, hailo output might be (Batch, H, W, Channels)
    
    # Process
    # We will just iterate or use numpy vectorization
    # However, output is UINT8 (quantized) or FLOAT32?
    # Usually parse-hef showed UINT8, but VStream can dequantize to FLOAT32.
    # We will request FLOAT32 from VStream.
    
    # Sigmoid on cx, cy, obj_conf, class_conf
    # y = sigmoid(x)
    
    # But wait, if we use VStream format FLOAT32, HailoRT might handles dequantization.
    
    # Boxes:
    # bx = (sigmoid(tx) * 2 - 0.5 + cx) * stride
    # by = (sigmoid(ty) * 2 - 0.5 + cy) * stride
    # bw = (sigmoid(tw) * 2) ** 2 * pw
    # bh = (sigmoid(th) * 2) ** 2 * ph
    
    # Let's implementation vectorization later if needed, for now Loop or simple numpy
    
    preds = []
    
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    grid_x = grid_x.reshape(1, h, w, 1) # broadcasting
    grid_y = grid_y.reshape(1, h, w, 1)
    
    # (B, H, W, A, C)
    # We permute to make Anchors dim 3 compatible with our logic if needed, 
    # but here weights are usually (A, C) or similar. 
    # Let's match output shape.
    
    # Assume output is (1, H, W, A, 5+C)
    # Wait, output shape from parse-hef was 8x8x45. 45 = 3 * 15.
    # So channel dim is last.
    
    output = output[0] # Take batch 0 -> (H, W, A, 5+C)

    # Sigmoid
    # Note: YOLOv5 usually has sigmoid embedded in the model for some exports, but usually raw output is logits.
    # Let's assume logits.
    
    xy = sigmoid(output[..., 0:2]) * 2 - 0.5
    wh = (sigmoid(output[..., 2:4]) * 2) ** 2
    
    # Coordinates
    # Grid is (H, W, 1) broadcasted to (H, W, A)
    # Anchor grid is (A, 2) broadcasted
    
    anchors = np.array(anchor_grid).reshape(1, 1, num_anchors, 2)
    
    # Add grid
    # xy is offset from grid center
    
    # We need grid broadcasted:
    # grid_x: (1, H, W, 1) -> (H, W, 1)
    # output[..., 0] is x
    
    pred_x = (xy[..., 0] + grid_x[0]) * stride
    pred_y = (xy[..., 1] + grid_y[0]) * stride
    pred_w = wh[..., 0] * anchors[..., 0]
    pred_h = wh[..., 1] * anchors[..., 1]
    
    # Objectness
    conf = sigmoid(output[..., 4:5])
    
    # Classes
    cls_conf = sigmoid(output[..., 5:])
    
    # (H, W, A, C)
    
    # Filter
    conf_thresh = 0.25
    candidates = conf > conf_thresh
    
    # Get indices
    # This is complex to vectorise cleanly without checking shape carefully.
    
    # Let's simply flatten everything
    # box: x,y,w,h
    
    # Flatten
    # (N, 5+C)
    
    # x center, y center, w, h
    # Convert to x1, y1, x2, y2
    
    bx = pred_x.flatten()
    by = pred_y.flatten()
    bw = pred_w.flatten()
    bh = pred_h.flatten()
    bconf = conf.flatten()
    
    # We need to multiply obj_conf * cls_conf for final score
    # But first filter by obj_conf
    
    mask = bconf > conf_thresh
    
    if not np.any(mask):
        return []
        
    bx = bx[mask]
    by = by[mask]
    bw = bw[mask]
    bh = bh[mask]
    bconf = bconf[mask]
    
    # Get class scores
    cls_scores = cls_conf.reshape(-1, num_classes)[mask]
    
    # Max class
    class_ids = np.argmax(cls_scores, axis=1)
    class_scores = cls_scores[np.arange(len(cls_scores)), class_ids]
    
    final_scores = bconf * class_scores
    
    # x1, y1, x2, y2
    x1 = bx - bw / 2
    y1 = by - bh / 2
    x2 = bx + bw / 2
    y2 = by + bh / 2
    
    boxes = np.stack([x1, y1, x2, y2], axis=1)
    
    return boxes, final_scores, class_ids

def nms(boxes, scores, iou_thresh):
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 0.25, iou_thresh)
    if len(indices) > 0:
        return indices.flatten()
    return []

def preprocess(image, target_size=(256, 256)):
    h, w = image.shape[:2]
    scale = min(target_size[0]/h, target_size[1]/w)
    nw, nh = int(w * scale), int(h * scale)
    
    resized = cv2.resize(image, (nw, nh))
    
    canvas = np.full((target_size[1], target_size[0], 3), 114, dtype=np.uint8)
    
    dx = (target_size[0] - nw) // 2
    dy = (target_size[1] - nh) // 2
    
    canvas[dy:dy+nh, dx:dx+nw] = resized
    
    return canvas, scale, dx, dy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="output_images")
    args = parser.parse_args()
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        
    # Valid Extensions
    valid_exts = ['.jpg', '.jpeg', '.png', '.bmp']
    images = []
    if os.path.isdir(args.image):
        for root, _, files in os.walk(args.image):
            for f in files:
                if any(f.lower().endswith(ext) for ext in valid_exts):
                    images.append(os.path.join(root, f))
    else:
        images.append(args.image)
        
    print(f"Found {len(images)} images.")
    
    # Initialize Hailo
    hef = HEF(args.model)
    
    target = VDevice()
    infer_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
    
    network_groups = target.configure(hef, infer_params)
    network_group = network_groups[0]
    
    input_params = InputVStreamParams.make(network_group, format_type=FormatType.AUTO)
    output_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
    
    network_group_params = network_group.create_params()
    
    input_vstream_info = hef.get_input_vstream_infos()[0]
    
    with network_group.activate(network_group_params):
        with InferVStreams(network_group, input_params, output_params) as infer_pipeline:
            for img_path in images:
                print(f"Processing {img_path}...")
                
                # Load and preprocess
                orig_img = cv2.imread(img_path)
                if orig_img is None:
                    print(f"Failed to load {img_path}")
                    continue
                    
                input_img, scale, dx, dy = preprocess(orig_img)
                input_data = np.expand_dims(input_img, axis=0) # Batch dim
                
                # Infer
                # Input dict
                input_name = input_vstream_info.name
                infer_results = infer_pipeline.infer({input_name: input_data})
                
                # Parse outputs
                # We need to map output tensor names to strides (8, 16, 32)
                # Helper: map based on shape
                
                all_boxes = []
                all_scores = []
                all_classes = []
                
                for name, output_data in infer_results.items():
                    # Check shape
                    # shape is (Batch, H, W, C)
                    shape = output_data.shape
                    grid_h = shape[1]
                    
                    stride = 256 // grid_h
                    
                    # Match stride to anchors
                    if stride == 8:
                        anchors = ANCHORS[0]
                    elif stride == 16:
                        anchors = ANCHORS[1]
                    elif stride == 32:
                        anchors = ANCHORS[2]
                    else:
                        print(f"Unknown stride {stride} for shape {shape}")
                        continue
                        
                    # Process
                    boxes, scores, classes = process_output(output_data, anchors, stride, 10)
                    
                    if len(boxes) > 0:
                        all_boxes.append(boxes)
                        all_scores.append(scores)
                        all_classes.append(classes)
                        
                if not all_boxes:
                    # No detections
                    pass
                else:
                    all_boxes = np.concatenate(all_boxes)
                    all_scores = np.concatenate(all_scores)
                    all_classes = np.concatenate(all_classes)
                    
                    # NMS
                    indices = nms(all_boxes, all_scores, 0.45)
                    
                    # Draw
                    filename = os.path.basename(img_path)
                    
                    # Rescale boxes to original image
                    # box is on 256x256 canvas
                    # (x - dx) / scale
                    
                    if len(indices) > 0:
                        for idx in indices:
                            box = all_boxes[idx]
                            score = all_scores[idx]
                            cls = all_classes[idx]
                            
                            x1, y1, x2, y2 = box
                            
                            x1 = (x1 - dx) / scale
                            y1 = (y1 - dy) / scale
                            x2 = (x2 - dx) / scale
                            y2 = (y2 - dy) / scale
                            
                            # Clip
                            h, w = orig_img.shape[:2]
                            x1 = max(0, min(w, x1))
                            y1 = max(0, min(h, y1))
                            x2 = max(0, min(w, x2))
                            y2 = max(0, min(h, y2))
                            
                            # Draw
                            cv2.rectangle(orig_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            label = f"Class {cls}: {score:.2f}"
                            cv2.putText(orig_img, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    out_file = os.path.join(args.output, f"pred_{filename}")
                    cv2.imwrite(out_file, orig_img)
                    print(f"Saved {out_file}")

if __name__ == "__main__":
    main()