import os
import json
import cv2
import numpy as np
from PIL import Image
from src.logic.feature_extractor import FeatureExtractor
from src.logic.vector_index import VectorIndex
from src.logic.golden_crop import GoldenCropExtractor
from src.logic.lego_colors import get_color_onehot, get_num_colors, get_color_name
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

def determine_optimal_k(embeddings, max_k=12, similarity_threshold=0.96):
    """
    Determines optimal number of K-Means clusters for a set of embeddings.
    It increments K until the clusters are 'tight' enough, OR the distance
    between new cluster centers becomes too small (meaning we are over-segmenting).
    """
    if len(embeddings) <= 2:
        return len(embeddings), embeddings
        
    best_k = 1
    best_centers = []
    
    # We always want at least 2 views (usually front/back or top/bottom) 
    # unless there's only 1 image.
    start_k = 2 if len(embeddings) >= 2 else 1
    
    for k in range(start_k, max_k + 1):
        if k > len(embeddings):
            break
            
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(embeddings)
            centers = kmeans.cluster_centers_
            
            # Check if the centers are too similar to each other
            if k > 1:
                # L2 normalize centers for cosine similarity check
                norms = np.linalg.norm(centers, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1e-10, norms)
                norm_centers = centers / norms
                
                sim_matrix = np.dot(norm_centers, norm_centers.T)
                # Find the maximum off-diagonal similarity
                np.fill_diagonal(sim_matrix, -1)
                max_sim = np.max(sim_matrix)
                
                # If two centers are very similar, we have reached the limit of useful distinct views
                if max_sim > similarity_threshold:
                    # We overshot. The previous K was good.
                    if not best_centers: # case for k=2 failing immediately
                        best_k = k
                        best_centers = centers
                    break
            
            best_k = k
            best_centers = centers
        except:
            break
            
    return best_k, best_centers

def build_index(dataset_dir, output_folder, unified=True, force=False, extractor_weights=None):
    """
    Builds or updates vector indices incrementally.
    
    Args:
        extractor_weights: Optional path to fine-tuned DINOv2 weights (.pth).
    """
    print(f"🛠️ Building/Updating Reference Indices in {output_folder}...")
    os.makedirs(output_folder, exist_ok=True)
    
    extractor = FeatureExtractor(model_name='dinov2_vits14', weights_path=extractor_weights)
    golden_extractor = GoldenCropExtractor()
    # Perfect-Fit Mode: We bypass GoldenCropExtractor (YOLO) since renders are already standardized
    unified_index = VectorIndex() if unified else None
    
    # --- INCREMENTAL LOGIC ---
    master_path = os.path.join(output_folder, "lego.index")
    if unified and os.path.exists(master_path):
        print(f"🔄 Found existing index. Loading to append new pieces...")
        unified_index.load(master_path)
    
    indexed_pairs = unified_index.get_indexed_ids() if unified_index else set()
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
    
    # --- QUICK SKIP OPTIMIZATION ---
    if not force and unified and image_meta:
        # Get the first piece identity from metadata
        first_img = next(iter(image_meta.values()))
        if first_img['ids']:
            ld_id = first_img['ids'][0]
            if ld_id in indexed_pairs:
                print(f"   ⏭️  Piece {ld_id} already in index. Skipping extraction.")
                return
    # -------------------------------
    
    for label_file in label_files:
        img_file = label_file.replace('.txt', '.jpg')
        img_path = os.path.join(images_dir, img_file)
        if not os.path.exists(img_path): continue
            
        # Resolve Identity: Metadata first, then names_map, then class_id
        real_ids = image_meta.get(img_file, {}).get('ids', [])
        real_colors = image_meta.get(img_file, {}).get('color_ids', [])
        
        # Parse YOLO label to get the ground-truth bounding box
        # Format: class_id x_center y_center width height (normalized)
        full_label_path = os.path.join(labels_dir, label_file)
        try:
            with open(full_label_path, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            continue
            
        # For ref_pieza, there is usually only 1 piece per image
        if not lines: continue
        
        orig_img = cv2.imread(img_path)
        if orig_img is None: continue
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        h, w = orig_img.shape[:2]

        for inst_idx, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) < 5: continue
            
            # Use real_ids if available, else fallback
            if inst_idx < len(real_ids):
                ldraw_id = real_ids[inst_idx]
                color_id = real_colors[inst_idx] if inst_idx < len(real_colors) else -1
            else:
                continue # Skip if no trusted metadata for this instance
                
            if ldraw_id == "unknown": continue
                
            # Convert normalized YOLO coordinates to pixel Bounding Box
            # image is 384x384 in reference renders
            cx_norm, cy_norm, w_norm, h_norm = map(float, parts[1:5])
            cx, cy = cx_norm * w, cy_norm * h
            box_w, box_h = w_norm * w, h_norm * h
            min_x = cx - (box_w / 2)
            min_y = cy - (box_h / 2)
            max_x = cx + (box_w / 2)
            max_y = cy + (box_h / 2)
            
            # Format detection for GoldenCropExtractor
            # We use perfect centering based on ground-truth centroid
            detection = [{'box': np.array([min_x, min_y, max_x, max_y])}]
            
            # Use GoldenCropExtractor logic to perfectly center and rescale to 80% occupancy
            # ensuring vector generation math perfectly mirrors recognition math.
            # Reference renders are already on black background, so we skip GrabCut (much faster).
            crops = golden_extractor.process_detections(orig_img, detection, use_grabcut=False)
            
            if not crops: continue
            raw_pil, crop_pil = crops[0] # Unpack the tuple

            # Get Embedding from the perfectly scaled and centered crop
            embedding = extractor.get_embedding(crop_pil)
            
            if ldraw_id not in piece_data:
                piece_data[ldraw_id] = {'embeddings': [], 'metadata': []}
            piece_data[ldraw_id]['embeddings'].append(embedding)
            piece_data[ldraw_id]['metadata'].append({
                'ldraw_id': ldraw_id,
                'color_id': color_id,
                'color_name': 'Reference'
            })
            
            processed_count += 1
            print("VECTOR_PROCESSED: 1")
            
        if processed_count % 200 == 0 and processed_count > 0:
            print(f"   Processed {processed_count} instances...")

    # Multiview Embedding Aggregation (Adaptive K-Means Clustering)
    print("🧠 Clustering embeddings for Adaptive Multiview representations...")
    
    total_indexed = 0
    for ldraw_id, data in piece_data.items():
        embeddings = np.array(data['embeddings'])
        
        # Adaptive K based on visual variance
        k, final_embeddings = determine_optimal_k(embeddings, max_k=12, similarity_threshold=0.96)
        print(f"   ↳ {ldraw_id}: selected K={k} clusters (from {len(embeddings)} images)")
        
        data['final_embeddings'] = [emb / (np.linalg.norm(emb) or 1.0) for emb in final_embeddings]

    # Save results
    if unified:
        added_count = 0
        for ldraw_id, data in piece_data.items():
            # Check for each instance in data['metadata'] to be precise about color
            # Since clustering might have merged multiple colors (though ref_pieza dirs are single-color-ish)
            # we check the first metadata entry's color for simplicity in build_reference_index
            color_id = data['metadata'][0].get('color_id', -1) if data['metadata'] else -1
            
            if not force and (ldraw_id, color_id) in indexed_pairs:
                print(f"   ⏭️ Piece {ldraw_id} (color {color_id}) already in index. Skipping append.")
                continue
            
            for emb in data['final_embeddings']:
                unified_index.add(emb, {'ldraw_id': ldraw_id, 'color_id': color_id})
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
