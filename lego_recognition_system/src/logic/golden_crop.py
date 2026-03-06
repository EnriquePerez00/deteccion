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

    def extract_crops(self, image_input, use_grabcut=True):
        """
        Infers masks using YOLO and extracts 384x384 'Golden Crops' centered at piece centroids.
        
        Args:
            image_input: Path to image or PIL Image or numpy array.
            use_grabcut: Whether to apply GrabCut background removal.
            
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
            
        return self.process_detections(img, detections, use_grabcut=use_grabcut)

    def process_detections(self, image, detections, use_grabcut=True):
        """
        Extracts 384x384 'Golden Crops' from provided image and detection list.
        
        Args:
            image: numpy array (RGB).
            detections: List of dicts with 'box' (xyxy) and optional 'polygon' (xy).
            use_grabcut: Whether to apply GrabCut background removal.
            
        Returns:
            List of tuples of PIL Images: (raw_crop, masked_crop).
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
            
            # --- NEW PHASE: GrabCut Background Masking ---
            source_image_for_crop = image # Default if no grabcut
            fallback_triggered = False
            
            if use_grabcut:
                # Create a localized box for GrabCut with an adaptive margin
                # We want at least 20px of 'carpet' context for GrabCut to learn from
                margin_x = max(20, int(piece_w * 0.10))
                margin_y = max(20, int(piece_h * 0.10))
                
                gc_x1 = max(0, int(min_x - margin_x))
                gc_y1 = max(0, int(min_y - margin_y))
                gc_x2 = min(w, int(max_x + margin_x))
                gc_y2 = min(h, int(max_y + margin_y))
                
                gc_w = gc_x2 - gc_x1
                gc_h = gc_y2 - gc_y1
                
                # Extract the ROI to run GrabCut (smaller image = much faster)
                roi = image[gc_y1:gc_y2, gc_x1:gc_x2].copy()
                
                # GrabCut requires a mask and two arrays for its internal models
                # We initialize the ENTIRE extended ROI (including the margins) as Probable Background (GC_PR_BGD)
                mask = np.full(roi.shape[:2], cv2.GC_PR_BGD, dtype=np.uint8)
                bgdModel = np.zeros((1, 65), np.float64)
                fgdModel = np.zeros((1, 65), np.float64)
                
                # Calculate the original YOLO bounding box boundaries within the ROI
                inner_x1 = max(0, int(min_x - gc_x1))
                inner_y1 = max(0, int(min_y - gc_y1))
                inner_x2 = min(roi.shape[1], int(max_x - gc_x1))
                inner_y2 = min(roi.shape[0], int(max_y - gc_y1))
                
                # Set the inner bounding box to Probable Foreground
                mask[inner_y1:inner_y2, inner_x1:inner_x2] = cv2.GC_PR_FGD
                
                # Run GrabCut (5 iterations is usually enough)
                try:
                    cv2.grabCut(roi, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
                    
                    # mask is modified to contain: 0=bg, 1=fg, 2=pr_bg, 3=pr_fg
                    mask2 = np.where((mask==cv2.GC_FGD)|(mask==cv2.GC_PR_FGD), 1, 0).astype('uint8')
                    
                    # --- DOUBLE GUARD CHECK: Tiny Pieces & Spill Prevention ---
                    mask_pixels = np.sum(mask2)
                    roi_pixels = mask2.shape[0] * mask2.shape[1]
                    
                    # If mask is < 0.5% (failed) OR > 95% (spilled to background)
                    if mask_pixels < (roi_pixels * 0.005) or mask_pixels > (roi_pixels * 0.95):
                        fallback_triggered = True
                    else:
                        # Apply the mask to the ROI (black out background)
                        roi_masked = roi * mask2[:, :, np.newaxis]
                        
                        # Create the full source image with the background blacked out
                        source_image_for_crop = np.zeros_like(image)
                        source_image_for_crop[gc_y1:gc_y2, gc_x1:gc_x2] = roi_masked
                    
                except Exception as e:
                    print(f"⚠️ GrabCut failed on a crop: {e}")
                    fallback_triggered = True
            else:
                # If not using grabcut, we still want to ensure the background is black 
                # outside the bounding box for consistency if the original wasn't perfect.
                # Actually, for reference renders it usually is. 
                # But let's be safe and apply black-out of everything outside the box 
                # without running the expensive GrabCut algorithm.
                source_image_for_crop = np.zeros_like(image)
                iy1, iy2 = max(0, int(min_y)), min(h, int(max_y))
                ix1, ix2 = max(0, int(min_x)), min(w, int(max_x))
                source_image_for_crop[iy1:iy2, ix1:ix2] = image[iy1:iy2, ix1:ix2]

            if fallback_triggered:
                # --- INTELLIGENT FALLBACK ---
                # Instead of the original raw image (with carpet), we use the YOLO BBox 
                # and put it on a 100% BLACK background.
                # This makes the resulting vector much cleaner and more compatible with references.
                source_image_for_crop = np.zeros_like(image)
                
                # Extract the raw YOLO box and paste it into a black canvas
                iy1, iy2 = max(0, int(min_y)), min(h, int(max_y))
                ix1, ix2 = max(0, int(min_x)), min(w, int(max_x))
                source_image_for_crop[iy1:iy2, ix1:ix2] = image[iy1:iy2, ix1:ix2]
            # ---------------------------------------------

            # 3. Rescaling Logic 
            is_fallback = det.get('is_fallback', False)
            
            if is_fallback:
                # Bounding Box Compression for Full-Frame Fallback
                # We take the bounding box, make it a square by using its longest side,
                # and add a small 5% margin to avoid clipping the edges. This prevents
                # taking only a part of the image and compresses the full detected box to 384x384.
                margin_factor = 1.05
                source_crop_size = max(piece_w, piece_h) * margin_factor
            else:
                # Dynamic Zoom for standard YOLO detections
                # We want the piece to occupy ~80% of the target_size (384px) to isolate it
                target_occupancy = 0.8
                source_crop_size = max(piece_w, piece_h) / target_occupancy
            
            # Security: Don't zoom in TOO much on tiny noise (min 60px window)
            source_crop_size = max(source_crop_size, 60)
            
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
            
            raw_crop = source_image_for_crop[vy1:vy2, vx1:vx2]
            
            if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
                final_source_crop = cv2.copyMakeBorder(
                    raw_crop, pad_top, pad_bottom, pad_left, pad_right, 
                    cv2.BORDER_CONSTANT, value=(0, 0, 0)
                )
            else:
                final_source_crop = raw_crop
                
            # --- CAPTURE RAW CROP FOR UI COMPARISON ---
            # We want the original YOLO bounding box behavior for the raw crop
            vx1_raw, vy1_raw = max(0, int(min_x)), max(0, int(min_y))
            vx2_raw, vy2_raw = min(w, int(max_x)), min(h, int(max_y))
            
            # Use source_crop_size for scaling to make them identical in scale
            x1_raw = int(centroid_x - source_crop_size / 2)
            y1_raw = int(centroid_y - source_crop_size / 2)
            x2_raw = int(x1_raw + source_crop_size)
            y2_raw = int(y1_raw + source_crop_size)
            
            pad_left_raw = max(0, -x1_raw)
            pad_top_raw = max(0, -y1_raw)
            pad_right_raw = max(0, x2_raw - w)
            pad_bottom_raw = max(0, y2_raw - h)
            
            vx1_r, vy1_r = max(0, x1_raw), max(0, y1_raw)
            vx2_r, vy2_r = min(w, x2_raw), min(h, y2_raw)
            
            raw_orig_crop = image[vy1_r:vy2_r, vx1_r:vx2_r]
            
            if pad_left_raw > 0 or pad_top_raw > 0 or pad_right_raw > 0 or pad_bottom_raw > 0:
                final_raw_crop = cv2.copyMakeBorder(
                    raw_orig_crop, pad_top_raw, pad_bottom_raw, pad_left_raw, pad_right_raw, 
                    cv2.BORDER_CONSTANT, value=(0, 0, 0)
                )
            else:
                final_raw_crop = raw_orig_crop
            
            if final_raw_crop.size > 0:
                final_raw_384 = cv2.resize(final_raw_crop, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA)
                raw_pil = Image.fromarray(final_raw_384)
            else:
                raw_pil = None
            # ------------------------------------------

            if final_source_crop.size == 0: continue
            final_384 = cv2.resize(final_source_crop, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA)
            masked_pil = Image.fromarray(final_384)
            
            # Return tuple: (raw_crop, grabcut_crop)
            crops.append((raw_pil, masked_pil))
            
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
