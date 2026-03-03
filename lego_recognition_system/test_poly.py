import numpy as np
from PIL import Image
import os
from src.logic.golden_crop import GoldenCropExtractor
from ultralytics import YOLO
from src.logic.sahi_slicer import run_sahi_inference

models_dir = "/Users/I764690/Code_personal/test_heavy_image_recognition/lego_recognition_system/models"
yolo = YOLO(os.path.join(models_dir, "yolo_model", "universal_detector_20260303_1650.pt"))
extractor = GoldenCropExtractor(target_size=384)
img_path = "/Users/I764690/Code_personal/test_heavy_image_recognition/lego_recognition_system/render_local/images_mix/images/w0_img_0000.jpg"
image = Image.open(img_path).convert("RGB")

dets = run_sahi_inference(yolo, image, conf_threshold=0.15)
detections_input = [{'box': d['box'], 'polygon': d['polygon']} for d in dets]

for i, det in enumerate(detections_input):
    poly = det.get('polygon')
    box = det.get('box')
    print(f"Piece {i}")
    if poly is not None and len(poly) > 0:
        poly_arr = np.array(poly)
        print(f"  poly shape: {poly_arr.shape}")
        min_x, max_x = np.min(poly_arr[:, 0]), np.max(poly_arr[:, 0])
        min_y, max_y = np.min(poly_arr[:, 1]), np.max(poly_arr[:, 1])
        print(f"  poly_w: {max_x - min_x}, poly_h: {max_y - min_y}")
    elif box is not None:
        print(f"  box_w: {box[2] - box[0]}, box_h: {box[3] - box[1]}")
