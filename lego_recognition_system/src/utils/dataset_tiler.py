import os
import cv2
import numpy as np
import glob
from pathlib import Path
from tqdm import tqdm
from PIL import Image

def slice_image_and_labels(image_path, label_path, output_dir, slice_size=640, overlap_ratio=0.25):
    """
    Slices a single image and its YOLO segmentation labels into overlapping tiles.
    """
    base_name = Path(image_path).stem
    img = cv2.imread(image_path)
    if img is None:
        return
    h_orig, w_orig = img.shape[:2]
    
    # Read labels
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        lines = []

    base_name = Path(image_path).stem
    stride = int(slice_size * (1.0 - overlap_ratio))
    
    img_dir = Path(output_dir) / "images"
    lbl_dir = Path(output_dir) / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    y = 0
    tile_count = 0
    while y < h_orig:
        x = 0
        while x < w_orig:
            x2 = min(x + slice_size, w_orig)
            y2 = min(y + slice_size, h_orig)
            
            # Adjusted origin if tile is at the edge to maintain full size if possible
            if x2 - x < slice_size and w_orig >= slice_size:
                x = w_orig - slice_size
                x2 = w_orig
            if y2 - y < slice_size and h_orig >= slice_size:
                y = h_orig - slice_size
                y2 = h_orig
            
            tile_w = x2 - x
            tile_h = y2 - y
            
            tile_img = img[y:y2, x:x2]
            tile_labels = []
            
            for line in lines:
                parts = line.strip().split()
                if not parts: continue
                cls_id = parts[0]
                coords = np.array([float(p) for p in parts[1:]]).reshape(-1, 2)
                
                # Convert global normalized to global pixel
                coords_px = coords * np.array([w_orig, h_orig])
                
                # Filter points inside or near the tile
                # We subtract tile origin to get local pixel coords
                local_coords_px = coords_px - np.array([x, y])
                
                # Clip to tile boundaries
                # (Simple approach: just clamp points to tile area)
                clipped_coords = np.clip(local_coords_px, [0, 0], [tile_w, tile_h])
                
                # Normalize to tile size
                norm_coords = clipped_coords / np.array([tile_w, tile_h])
                
                # Check if the object is significantly inside the tile
                # One way is to check if the bounding box of the points in the tile has area
                if len(norm_coords) > 2:
                    min_c = np.min(norm_coords, axis=0)
                    max_c = np.max(norm_coords, axis=0)
                    # If the box is very small in the tile, it might be just an edge
                    if (max_c[0] - min_c[0]) > 0.01 and (max_c[1] - min_c[1]) > 0.01:
                        # Success! Convert back to string
                        coord_str = " ".join([f"{c:.6f}" for c in norm_coords.flatten()])
                        tile_labels.append(f"{cls_id} {coord_str}")

            if tile_labels:
                tile_fn = f"{base_name}_tile_{tile_count}.jpg"
                cv2.imwrite(str(img_dir / tile_fn), tile_img)
                with open(lbl_dir / f"{base_name}_tile_{tile_count}.txt", 'w') as f:
                    f.write("\n".join(tile_labels))
                tile_count += 1

            if x2 >= w_orig: break
            x += stride
        if y2 >= h_orig: break
        y += stride

def process_dataset(input_dir, output_dir, slice_size=640, overlap_ratio=0.25):
    img_paths = glob.glob(os.path.join(input_dir, "images", "*.jpg"))
    print(f"Slicing {len(img_paths)} images from {input_dir} into {output_dir}...")
    
    for img_p in tqdm(img_paths):
        # Correct path replacement: only replace the LAST occurrence of 'images' with 'labels'
        img_p_obj = Path(img_p)
        lbl_p = str(img_p_obj.parent.parent / "labels" / (img_p_obj.stem + ".txt"))
        slice_image_and_labels(img_p, lbl_p, output_dir, slice_size, overlap_ratio)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="render_local/images_mix")
    parser.add_argument("--output", default="render_local/images_mix_tiled")
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.25)
    args = parser.parse_args()
    
    process_dataset(args.input, args.output, args.size, args.overlap)
