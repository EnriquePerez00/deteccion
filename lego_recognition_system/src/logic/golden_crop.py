import os
import glob
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO

class GoldenCropExtractor:
    def __init__(self, model_path=None, target_size=384):
        self.target_size = target_size
        
        if model_path is None:
            # Auto-detect latest .pt model in project root
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            pt_files = glob.glob(os.path.join(project_root, "*.pt"))
            # Correctly check models/yolo_model
            pt_files += glob.glob(os.path.join(project_root, "models", "yolo_model", "*.pt"))
            
            if not pt_files:
                raise FileNotFoundError("❌ No YOLO .pt models found in project root or models/yolo_model/.")
            
            # Pick latest modified
            self.model_path = max(pt_files, key=os.path.getmtime)
            print(f"🎯 GoldenCrop: Using auto-detected model: {os.path.basename(self.model_path)}")
        else:
            self.model_path = model_path
            
        self.model = YOLO(self.model_path)

    def extract_crops(self, image_input):
        """
        Infers masks using YOLO and extracts 384x384 'Golden Crops' centered at piece centroids.
        
        Args:
            image_input: Path to image or PIL Image or numpy array.
            
        Returns:
            List of PIL Images (standardized 384x384 crops).
        """
        results = self.model(image_input, verbose=False)
        
        # Original Image
        if isinstance(image_input, str):
            img = cv2.imread(image_input)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, Image.Image):
            img = np.array(image_input)
        else:
            img = image_input

        if not results or len(results) == 0:
            return []
            
        result = results[0]
        
        # Convert results to a unified detection format for process_detections
        detections = []
        for i in range(len(result.boxes)):
            det = {
                'box': result.boxes[i].xyxy[0].cpu().numpy(),
                'polygon': result.masks[i].xy[0] if result.masks is not None else None
            }
            detections.append(det)
            
        return self.process_detections(img, detections)

    def process_detections(self, image, detections):
        """
        Extracts 384x384 'Golden Crops' from provided image and detection list.
        
        Args:
            image: numpy array (RGB).
            detections: List of dicts with 'box' (xyxy) and optional 'polygon' (xy).
            
        Returns:
            List of PIL Images.
        """
        crops = []
        h, w = image.shape[:2]

        for det in detections:
            poly = det.get('polygon')
            box = det.get('box')

            # 1. Calculate Centroid and piece dimensions
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
            
            # 2. Rescaling Logic (Fase A)
            scale = 1.0
            if piece_w > self.target_size or piece_h > self.target_size:
                scale = self.target_size / max(piece_w, piece_h)
            
            source_crop_size = self.target_size / scale
            
            x1 = int(centroid_x - source_crop_size / 2)
            y1 = int(centroid_y - source_crop_size / 2)
            x2 = int(x1 + source_crop_size)
            y2 = int(y1 + source_crop_size)
            
            # 3. Padding/Clipping
            pad_left = max(0, -x1)
            pad_top = max(0, -y1)
            pad_right = max(0, x2 - w)
            pad_bottom = max(0, y2 - h)
            
            vx1, vy1 = max(0, x1), max(0, y1)
            vx2, vy2 = min(w, x2), min(h, y2)
            
            raw_crop = image[vy1:vy2, vx1:vx2]
            
            if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
                final_source_crop = cv2.copyMakeBorder(
                    raw_crop, pad_top, pad_bottom, pad_left, pad_right, 
                    cv2.BORDER_CONSTANT, value=(0, 0, 0)
                )
            else:
                final_source_crop = raw_crop
                
            # 4. Final Resize to 384x384
            if final_source_crop.size == 0: continue
            final_384 = cv2.resize(final_source_crop, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA)
            
            crops.append(Image.fromarray(final_384))
            
        return crops

# Standalone Test
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        extractor = GoldenCropExtractor()
        results = extractor.extract_crops(sys.argv[1])
        print(f"Extracted {len(results)} Golden Crops from {sys.argv[1]}")
        for i, crop in enumerate(results):
            crop.save(f"test_golden_crop_{i}.jpg")
