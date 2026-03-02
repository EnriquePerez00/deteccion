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

def build_index(dataset_dir, output_folder, unified=True, force=False):
    """
    Builds or updates vector indices incrementally.
    """
    print(f"🛠️ Building/Updating Reference Indices in {output_folder}...")
    os.makedirs(output_folder, exist_ok=True)
    
    extractor = FeatureExtractor(model_name='dinov2_vits14')
    # Perfect-Fit Mode: We bypass GoldenCropExtractor (YOLO) since renders are already standardized
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
            
        # Perfect-Fit Strategy: Direct extraction without YOLO
        # The image is already 384x384 and perfectly centered
        crop_pil = Image.open(img_path).convert('RGB')
        
        # Resolve Identity: Metadata first, then names_map, then class_id
        real_ids = image_meta.get(img_file, {}).get('ids', [])
        real_colors = image_meta.get(img_file, {}).get('color_ids', [])

        # For ref_pieza, there is only 1 piece per image
        for inst_idx in range(len(real_ids)):
            # Get Embedding
            embedding = extractor.get_embedding(crop_pil)
            
            # Resolve Identity
            ldraw_id = real_ids[inst_idx]
            
            # Resolve Color
            color_id = real_colors[inst_idx]
            
            if ldraw_id != "unknown":
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
