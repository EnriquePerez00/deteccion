import numpy as np
import cv2
from PIL import Image, ImageDraw
import os
from src.logic.golden_crop import GoldenCropExtractor
from ultralytics import YOLO
from src.logic.sahi_slicer import run_sahi_inference

models_dir = "/Users/I764690/Code_personal/test_heavy_image_recognition/lego_recognition_system/models"
yolo_path = os.path.join(models_dir, "yolo_model", "universal_detector_20260303_1650.pt")
yolo = YOLO(yolo_path)

img_path = "/Users/I764690/Code_personal/test_heavy_image_recognition/lego_recognition_system/render_local/images_mix/images/w0_img_0000.jpg"
image = Image.open(img_path).convert("RGB")
img_w, img_h = image.size

dets = run_sahi_inference(yolo, image, conf_threshold=0.15)
detections_input = [{'box': d['box'], 'polygon': d['polygon']} for d in dets]

draw = ImageDraw.Draw(image)

for det in detections_input:
    poly = det.get('polygon')
    box = det.get('box')

    if poly is not None and len(poly) > 0:
        centroid_x = np.mean([p[0] for p in poly])
        centroid_y = np.mean([p[1] for p in poly])
        min_x, max_x = np.min(poly[:, 0]), np.max(poly[:, 0])
        min_y, max_y = np.min(poly[:, 1]), np.max(poly[:, 1])
    elif box is not None:
        centroid_x = (box[0] + box[2]) / 2
        centroid_y = (box[1] + box[3]) / 2
        min_x, min_y, max_x, max_y = box
    else:
        continue
        
    piece_w = max_x - min_x
    piece_h = max_y - min_y
    
    target_occupancy = 0.8
    source_crop_size = max(piece_w, piece_h) / target_occupancy
    source_crop_size = max(source_crop_size, 60)
    
    x1 = int(centroid_x - source_crop_size / 2)
    y1 = int(centroid_y - source_crop_size / 2)
    x2 = int(x1 + source_crop_size)
    y2 = int(y1 + source_crop_size)
    
    # Draw RED box for the extracted crop
    draw.rectangle([x1, y1, x2, y2], outline="red", width=5)
    
    # Draw GREEN box for the detection
    draw.rectangle([min_x, min_y, max_x, max_y], outline="green", width=3)

image.save("/tmp/test_crop_render.jpg")
print("Done saving /tmp/test_crop_render.jpg")
