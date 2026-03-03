import numpy as np
import cv2
from PIL import Image
import os

from src.logic.golden_crop import GoldenCropExtractor
from ultralytics import YOLO

def test():
    models_dir = "/Users/I764690/Code_personal/test_heavy_image_recognition/lego_recognition_system/models"
    yolo = YOLO(os.path.join(models_dir, "yolo_model", "universal_detector_20260303_1650.pt"))
    extractor = GoldenCropExtractor(target_size=384)
    
    img_path = "/Users/I764690/Code_personal/test_heavy_image_recognition/lego_recognition_system/render_local/images_mix/images/w0_img_0000.jpg"
    image = Image.open(img_path).convert("RGB")
    
    from src.logic.sahi_slicer import run_sahi_inference
    dets = run_sahi_inference(yolo, image, conf_threshold=0.15)
    
    print(f"Detections: {len(dets)}")
    if not dets: return
    
    detections_input = [{'box': d['box'], 'polygon': d['polygon']} for d in dets]
    
    crops = extractor.process_detections(np.array(image), detections_input)
    
    for i, c in enumerate(crops):
        c.save(f"/tmp/test_crop_{i}.jpg")
        print(f"Saved crop {i}, size: {c.size}")
        print(f"Det box {i}: {dets[i]['box']}")
        
if __name__ == '__main__':
    test()
