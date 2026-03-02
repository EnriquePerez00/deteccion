import os
import json
import cv2
import numpy as np
from PIL import Image
from src.logic.feature_extractor import FeatureExtractor
from src.logic.vector_index import VectorIndex
from src.logic.lego_colors import get_color_onehot, get_num_colors, get_color_name
from pathlib import Path

def build_index(dataset_dir, output_folder, unified=True, force=False):
    """
    Builds or updates vector indices incrementally.
    """
    print(f"🛠️ Building/Updating Reference Indices in {output_folder}...")
    os.makedirs(output_folder, exist_ok=True)
    
    extractor = FeatureExtractor(model_name='dinov2_vits14')
    unified_index = VectorIndex() if unified else None
    
    # --- INCREMENTAL LOGIC ---
    master_path = os.path.join(output_folder, "lego.index")
    if unified and os.path.exists(master_path):
        print(f"🔄 Found existing index. Loading to append new pieces...")
        unified_index.load(master_path)
    
    indexed_ids = unified_index.get_indexed_ids() if unified_index else set()
    # -------------------------
    
    piece_data = {} # ldraw_id -> {'embeddings': [], 'metadata': []}
    
    images_dir = os.path.join(dataset_dir, "images")
    labels_dir = os.path.join(dataset_dir, "labels")
    
    data_yaml_path = os.path.join(dataset_dir, "data.yaml")
    names_map = []
    if os.path.exists(data_yaml_path):
        import yaml
        with open(data_yaml_path, 'r') as f:
            y_data = yaml.safe_load(f)
            names_map = y_data.get('names', [])

    if not os.path.exists(labels_dir):
        print(f"❌ Error: {labels_dir} not found.")
        return

    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    processed_count = 0
    
    # --- METADATA MAPPING (Strategy C / Universal) ---
    image_meta = {}
    meta_path = os.path.join(dataset_dir, "image_meta.jsonl")
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    # Store full entry dict to preserve color_ids alongside ids
                    image_meta[entry['img']] = {
                        'ids': entry.get('ids', []),
                        'color_ids': entry.get('color_ids', [])
                    }
                except: continue
        print(f"   📖 Loaded real-ID mapping for {len(image_meta)} images from image_meta.jsonl")
    # ------------------------------------------------
    
    for label_file in label_files:
        img_file = label_file.replace('.txt', '.jpg')
        img_path = os.path.join(images_dir, img_file)
        if not os.path.exists(img_path): continue
            
        img = cv2.imread(img_path)
        if img is None: continue
        h, w, _ = img.shape
        
        with open(os.path.join(labels_dir, label_file), 'r') as f:
            lines = f.readlines()
            
        # Resolve Identity: Metadata first, then names_map, then class_id
        real_ids = image_meta.get(img_file, {}).get('ids', image_meta.get(img_file, []))
        real_colors = image_meta.get(img_file, {}).get('color_ids', [])
        # Handle legacy format where image_meta value is just a list of ids
        if isinstance(real_ids, list) and not isinstance(image_meta.get(img_file, {}), dict):
            real_ids = image_meta.get(img_file, [])
            real_colors = []
        
        for inst_idx, line in enumerate(lines):
            parts = line.split()
            if not parts: continue
            class_id = int(parts[0])
            
            # Use Segmentation or fallback to BBox
            if len(parts) >= 5: 
                if len(parts) > 5: # Segmentation
                    coords = list(map(float, parts[1:]))
                    pixel_points = [[int(coords[i]*w), int(coords[i+1]*h)] for i in range(0, len(coords), 2)]
                    pts = np.array(pixel_points, np.int32).reshape((-1, 1, 2))
                    x_b, y_b, w_b, h_b = cv2.boundingRect(pts)
                else: # Classic YOLO BBox: class x_c y_c w h
                    x_c, y_c, w_n, h_n = map(float, parts[1:])
                    w_b, h_b = int(w_n * w), int(h_n * h)
                    x_b, y_b = int((x_c * w) - w_b/2), int((y_c * h) - h_b/2)
                
                # Safety Crop
                x_b, y_b = max(0, x_b), max(0, y_b)
                w_b, h_b = min(w - x_b, w_b), min(h - y_b, h_b)
                if w_b < 5 or h_b < 5: continue
                
                # Extract piece
                crop = img[y_b:y_b+h_b, x_b:x_b+w_b]
                
                # Letterboxing
                max_dim = max(h_b, w_b)
                canvas = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
                canvas[(max_dim-h_b)//2:(max_dim-h_b)//2+h_b, (max_dim-w_b)//2:(max_dim-w_b)//2+w_b] = crop
                
                # Get Embedding
                crop_pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
                embedding = extractor.get_embedding(crop_pil)
                
                # Resolve Identity: Metadata first, then names_map, then class_id
                if inst_idx < len(real_ids):
                    ldraw_id = real_ids[inst_idx]
                else:
                    ldraw_id = names_map[class_id] if class_id < len(names_map) else str(class_id)
                
                # Resolve Color
                color_id = real_colors[inst_idx] if inst_idx < len(real_colors) else -1
                
                if ldraw_id not in piece_data:
                    piece_data[ldraw_id] = {'embeddings': [], 'metadata': []}
                piece_data[ldraw_id]['embeddings'].append(embedding)
                piece_data[ldraw_id]['metadata'].append({
                    'ldraw_id': ldraw_id,
                    'color_id': color_id,
                    'color_name': get_color_name(color_id) if color_id >= 0 else 'Unknown'
                })
                
                processed_count += 1
            
        if processed_count % 200 == 0 and processed_count > 0:
            print(f"   Processed {processed_count} instances...")

    # Multiview Embedding Aggregation (KMeans Clustering)
    from sklearn.cluster import KMeans
    print("🧠 Clustering embeddings for Multiview representations...")
    
    total_indexed = 0
    for ldraw_id, data in piece_data.items():
        embeddings = np.array(data['embeddings'])
        num_clusters = min(5, len(embeddings))
        
        if num_clusters > 1:
            try:
                kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
                kmeans.fit(embeddings)
                final_embeddings = kmeans.cluster_centers_
            except Exception as e:
                print(f"⚠️ Clustering failed for {ldraw_id}: {e}")
                final_embeddings = embeddings
        else:
            final_embeddings = embeddings
            
        data['final_embeddings'] = [emb / (np.linalg.norm(emb) or 1.0) for emb in final_embeddings]

    # Save results
    if unified:
        added_count = 0
        for ldraw_id, data in piece_data.items():
            if not force and ldraw_id in indexed_ids:
                print(f"   ⏭️ Piece {ldraw_id} already in index. Skipping append.")
                continue
            
            for emb in data['final_embeddings']:
                unified_index.add(emb, {'ldraw_id': ldraw_id})
                added_count += 1
        
        if added_count > 0:
            out_path = os.path.join(output_folder, "lego.index")
            unified_index.save(out_path)
            print(f"✅ Unified index updated: {out_path} ({unified_index.index.ntotal} total instances)")
        else:
            print("   ℹ️ No new pieces to add to index.")
    else:
        for ldraw_id, data in piece_data.items():
            out_path = os.path.join(output_folder, f"{ldraw_id}.index")
            temp_index = VectorIndex()
            for emb in data['final_embeddings']:
                temp_index.add(emb, {'ldraw_id': ldraw_id})
            temp_index.save(out_path)
            print(f"✅ Part index saved: {ldraw_id}.index")

    print(f"🎉 Batch indexing finished. {processed_count} total embeddings processed.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    build_index(args.dataset, args.output)
