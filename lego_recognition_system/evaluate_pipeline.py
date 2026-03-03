import os
import sys
import json
import numpy as np
from PIL import Image
from src.logic.golden_crop import GoldenCropExtractor
from src.logic.feature_extractor import FeatureExtractor
from src.logic.vector_index import VectorIndex

# Set environment variable to avoid crash
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def main():
    dataset_dir = "/Users/I764690/Code_personal/test_heavy_image_recognition/lego_recognition_system/render_local/images_mix"
    images_dir = os.path.join(dataset_dir, "images")
    labels_dir = os.path.join(dataset_dir, "labels")
    models_dir = "/Users/I764690/Code_personal/test_heavy_image_recognition/lego_recognition_system/models"
    
    # Load Models
    yolo_model_path = os.path.join(models_dir, "yolo_model", "universal_detector_20260303_1650.pt")
    extractor = GoldenCropExtractor(model_path=yolo_model_path)
    feature_extractor = FeatureExtractor(model_name='dinov2_vits14')
    v_index = VectorIndex()
    v_index.load(os.path.join(models_dir, "piezas_vectores", "lego.index"))
    
    # Get 10 sample images
    all_images = sorted([f for f in os.listdir(images_dir) if f.endswith(".jpg")])
    sample_images = all_images[:10]
    
    results_summary = []
    
    print(f"🚀 Starting evaluation on {len(sample_images)} images...\n")
    
    for img_name in sample_images:
        img_path = os.path.join(images_dir, img_name)
        label_path = os.path.join(labels_dir, img_name.replace(".jpg", ".txt"))
        
        # Load Ground Truth
        gt_ids = []
        if os.path.exists(label_path):
            # In SAHI/Mix mode, we usually have a meta file mapping filename to IDs
            # but for this simulation, we'll try to find the meta from the image_meta.jsonl
            pass
            
        # For simplicity in this simulation, we'll look at YOLO detections vs Index matches
        image = Image.open(img_path).convert("RGB")
        crops = extractor.extract_crops(image)
        
        num_detections = len(crops)
        match_stats = []
        
        for i, crop in enumerate(crops):
            emb = feature_extractor.get_embedding(crop)
            matches = v_index.search(emb, k=1)
            if matches:
                best = matches[0]
                match_stats.append({
                    "piece_idx": i,
                    "id": best['metadata'].get('ldraw_id'),
                    "similarity": best['similarity']
                })
        
        results_summary.append({
            "image": img_name,
            "detections": num_detections,
            "matches": match_stats
        })
        
        print(f"📸 {img_name}: {num_detections} pieces detected.")
        for m in match_stats:
            print(f"   - Piece {m['piece_idx']}: Predicted {m['id']} (Sim: {m['similarity']:.4f})")

    print("\n--- 📊 Evaluation Insights ---")
    total_det = sum(r['detections'] for r in results_summary)
    avg_sim = np.mean([m['similarity'] for r in results_summary for m in r['matches']]) if total_det > 0 else 0
    
    print(f"Total Detections: {total_det}")
    print(f"Average Similarity: {avg_sim:.4f}")
    
    print("\n💡 enseñanzas o propuestas de mejora:")
    print("1. YOLO Localization: Si el número de detecciones coincide con el esperado (~8-10 por imagen mix), YOLO está detectando bien.")
    print("2. Vector Similarity: Una similitud media de >0.85 suele indicar una identificación sólida. Valores <0.60 sugieren que la pieza no está bien representada en la base de datos de vectores o la iluminación es muy distinta.")
    print("3. Mejora: Implementar un filtro de confianza antes de la búsqueda vectorial para evitar falsos positivos de YOLO.")

if __name__ == "__main__":
    main()
