import os
import sys
import numpy as np
from PIL import Image
from src.logic.golden_crop import GoldenCropExtractor
from src.logic.feature_extractor import FeatureExtractor
from src.logic.vector_index import VectorIndex

def main():
    img_path = "/tmp/query.jpg"
    if not os.path.exists(img_path):
        print(f"❌ Error: {img_path} not found.")
        return

    models_dir = "/Users/I764690/Code_personal/test_heavy_image_recognition/lego_recognition_system/models"
    
    print("⏳ Loading YOLO and GoldenCropExtractor...")
    yolo_model_path = os.path.join(models_dir, "yolo_model", "universal_detector_20260303_1650.pt")
    extractor = GoldenCropExtractor(model_path=yolo_model_path)
    
    print("⏳ Loading DINOv2 FeatureExtractor...")
    feature_extractor = FeatureExtractor(model_name='dinov2_vits14')
    
    print("⏳ Loading VectorIndex...")
    v_index = VectorIndex()
    index_path = os.path.join(models_dir, "piezas_vectores", "lego.index")
    v_index.load(index_path)
    
    print(f"🔍 Processing image: {img_path}")
    image = Image.open(img_path).convert("RGB")
    
    # 1. Extract Crops (YOLO Detection + Centering)
    crops = extractor.extract_crops(image)
    
    if not crops:
        print("⚠️ No pieces detected by YOLO.")
        return
    
    print(f"✅ Detected {len(crops)} pieces.")
    
    # 2. Process the first detection (as requested)
    target_crop = crops[0]
    
    # 3. Get Embedding
    embedding = feature_extractor.get_embedding(target_crop)
    
    # 4. Search Index
    results = v_index.search(embedding, k=3)
    
    print("\n--- 🏆 Best Vector Match ---")
    if results:
        best = results[0]
        meta = best['metadata']
        print(f"LDraw ID: {meta.get('ldraw_id')}")
        print(f"Color: {meta.get('color_name')} (ID: {meta.get('color_id')})")
        print(f"Similarity: {best['similarity']:.4f}")
        
        print("\n--- 🥈 Top 3 Matches ---")
        for i, res in enumerate(results):
            m = res['metadata']
            print(f"{i+1}. ID {m.get('ldraw_id')} | Similarity: {res['similarity']:.4f}")
    else:
        print("⚠️ No matches found in index.")

if __name__ == "__main__":
    main()
