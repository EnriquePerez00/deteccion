import streamlit as st
import numpy as np
import os
import ssl
import urllib.request
import io
import certifi
from PIL import Image, ImageDraw

# Fix macOS SSL certificate verification issue (common with Python from python.org)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from src.logic.feature_extractor import FeatureExtractor
from src.logic.vector_index import VectorIndex

@st.cache_resource(show_spinner="Loading AI Models...")
def load_models(models_dir):
    """Loads and caches all models found in the models directory."""
    if YOLO is None:
        st.error("The `ultralytics` package is required for YOLO inference.")
        return None, None, None
        
    # Search for YOLO model
    yolo_model = None
    yolo_dir = os.path.join(models_dir, "yolo_model")
    yolo_path = os.path.join(yolo_dir, "detector_universal.pt")
    
    if not os.path.exists(yolo_path) and os.path.exists(yolo_dir):
        pt_files = [f for f in os.listdir(yolo_dir) if f.endswith(".pt")]
        if pt_files:
            yolo_path = os.path.join(yolo_dir, sorted(pt_files)[-1])
            
    if os.path.exists(yolo_path):
        yolo_model = YOLO(yolo_path)
        st.sidebar.info(f"🧠 Detector: `{os.path.basename(yolo_path)}`")
    else:
        st.warning(f"⚠️ No hay modelos YOLO (.pt) en {yolo_dir}")
    
    feature_extractor = FeatureExtractor(model_name='mobilenet_v3_small')
    
    # Load ALL available vector indices
    vector_index = VectorIndex()
    piezas_dir = os.path.join(models_dir, "piezas_vectores")
    index_files = []
    
    if os.path.exists(piezas_dir):
        index_files = [f for f in os.listdir(piezas_dir) if f.endswith(".pkl")]
        
    for idx_file in index_files:
        try:
            vector_index.load(os.path.join(piezas_dir, idx_file))
        except Exception as e:
            st.error(f"Error loading index {idx_file}: {e}")
            
    if not index_files:
        st.sidebar.warning("⚠️ No hay índices de vectores (.pkl)")
    else:
        st.sidebar.success(f"📚 {len(index_files)} Índices cargados ({len(vector_index.embeddings)} vectores)")
        
    return yolo_model, feature_extractor, vector_index


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_lego_image(ldraw_id):
    """Fetches a thumbnail image of the LEGO part from Rebrickable CDN with SSL bypass."""
    # Clean up the ID since the Vector DB might return '3001.dat' instead of '3001'
    clean_id = str(ldraw_id).replace('.dat', '').strip()
    
    # Color 14 is Light Bluish Gray (standard render color)
    url = f"https://cdn.rebrickable.com/media/parts/ldraw/14/{clean_id}.png"
    
    try:
        # Create a secure context using certifi specifically to fix macOS errors
        context = ssl.create_default_context(cafile=certifi.where())
        
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, context=context, timeout=3) as response:
            image_data = response.read()
            return Image.open(io.BytesIO(image_data))
    except Exception as e:
        # Silently fail and return None if missing or network error
        return None

def render_recognition_ui(uploaded_file, models_dir, conf_threshold):
    """Renders the main recognition pipeline visualization."""
    
    try:
        # Load the image
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return
        
    yolo, extractor, v_index = load_models(models_dir)
    if yolo is None or extractor is None:
        return
        
    st.subheader("1. YOLO Detection (Phase 1)")
    
    with st.spinner("Running YOLO detector..."):
        # Run inference
        results = yolo.predict(image, conf=conf_threshold)
        
    if not results or len(results[0].boxes) == 0:
        st.info("No LEGO pieces detected in this image.")
        st.image(image, use_container_width=True)
        return
        
    result = results[0]
    
    # Check if we have OBB (Oriented Bounding Boxes) from yolo11-obb model
    use_obb = hasattr(result, 'obb') and result.obb is not None
    
    if use_obb:
        # Extract 8-coordinate polygons
        polygons = result.obb.xyxyxyxy.cpu().numpy()
        confidences = result.obb.conf.cpu().numpy()
        
        # We also need an AABB (x1,y1,x2,y2) for Phase 2 image cropping
        boxes = []
        for poly in polygons:
            min_x, max_x = np.min(poly[:, 0]), np.max(poly[:, 0])
            min_y, max_y = np.min(poly[:, 1]), np.max(poly[:, 1])
            boxes.append([min_x, min_y, max_x, max_y])
    else:
        # Fallback to standard rectangular boxes
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
    
    # Draw boxes for the UI
    drawn_image = image.copy()
    draw = ImageDraw.Draw(drawn_image)
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        conf = confidences[i]
        
        if use_obb:
            # Draw oriented polygon
            poly = polygons[i]
            points = [(int(pt[0]), int(pt[1])) for pt in poly]
            draw.polygon(points, outline="red", width=3)
            # Label on the top-left-most point of the polygon
            draw.text((x1, max(0, y1 - 15)), f"Part {i+1} ({conf:.2f})", fill="red")
        else:
            # Draw standard bounding box
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, max(0, y1 - 15)), f"Part {i+1} ({conf:.2f})", fill="red")
        
    st.image(drawn_image, caption=f"Detected {len(boxes)} potential LEGO parts {'(Oriented)' if use_obb else '(Rectangular)'}", use_container_width=True)
    
    st.markdown("---")
    st.subheader("2. Vector Search Classification (Phase 2)")
    
    if v_index is None or not v_index.embeddings:
        st.warning("⚠️ No Vector Index loaded. Generating embeddings without classification.")
        
    # Process each detected bounding box
    identified_parts = {}
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        # Expand box slightly for better context
        pad = 5
        cx1, cy1 = max(0, x1-pad), max(0, y1-pad)
        cx2, cy2 = min(image.width, x2+pad), min(image.height, y2+pad)
        
        cropped_img = image.crop((cx1, cy1, cx2, cy2))
        
        # OBB ROBUST ALIGNMENT & MASKING: Erase background noise AND physically rotate the image to be straight
        if use_obb:
            poly = polygons[i]
            
            import cv2
            # 1. Convert crop to numpy array for OpenCV processing
            cv_img = np.array(cropped_img)
            
            # 2. Shift the global OBB coordinates to relative crop coordinates
            cropped_poly = [(pt[0] - cx1, pt[1] - cy1) for pt in poly]
            
            # 3. Create a pure black mask
            mask = np.zeros(cv_img.shape[:2], dtype=np.uint8)
            pts = np.array(cropped_poly, np.int32).reshape((-1, 1, 2))
            
            # 4. Draw the piece shape in solid white on the mask
            cv2.fillPoly(mask, [pts], (255))
            
            # 5. Composite: Only let the real image show where the mask is white (Blackout the rest)
            masked_img = cv2.bitwise_and(cv_img, cv_img, mask=mask)
            
            # 6. MATHEMATICAL ALIGNMENT (Affine Transformation)
            # Find the minimum area bounding rectangle to get the precise angle of the piece
            rect = cv2.minAreaRect(pts)
            (center, (width, height), angle) = rect
            
            # OpenCV's minAreaRect can return angles close to 90 or -90 depending on version
            # We want the piece to lay flat horizontally or vertically, so we adjust
            if width < height:
                angle += 90
            
            # Calculate rotation matrix for affine warp
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Rotate both the masked image and the mask itself
            rotated_full = cv2.warpAffine(masked_img, M, (cv_img.shape[1], cv_img.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
            rotated_mask = cv2.warpAffine(mask, M, (cv_img.shape[1], cv_img.shape[0]))
            
            # 7. Final Straight Crop: Now that it's straight, we crop the excess black space
            x_rot, y_rot, w_rot, h_rot = cv2.boundingRect(rotated_mask)
            pad_final = 2
            x_rot = max(0, x_rot - pad_final)
            y_rot = max(0, y_rot - pad_final)
            w_rot = min(rotated_full.shape[1] - x_rot, w_rot + pad_final * 2)
            h_rot = min(rotated_full.shape[0] - y_rot, h_rot + pad_final * 2)
            
            straightened_crop = rotated_full[y_rot:y_rot+h_rot, x_rot:x_rot+w_rot]
            
            # 8. Convert back to PIL for MobileNet feature extractor
            if straightened_crop.size > 0: # Safety check
                cropped_img = Image.fromarray(straightened_crop)
            else:
                 cropped_img = Image.fromarray(masked_img) # Fallback if math fails
        
        with st.expander(f"🧩 Part {i+1} - Confidence: {confidences[i]:.0%}", expanded=True):
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                st.image(cropped_img, caption="Mundo Real (YOLO)", use_container_width=True)
                
            # Extract embedding
            embedding = extractor.get_embedding(cropped_img)
            
            if v_index is not None and v_index.embeddings:
                matches = v_index.search(embedding, k=3)
                
                # We extract the Top Match data for the center column
                top_match_id = matches[0]['metadata'].get('ldraw_id', 'Unknown')
                top_match_sim = matches[0]['similarity']
                
                with col2:
                    if top_match_sim > 0.4:
                        stock_img = fetch_lego_image(top_match_id)
                        if stock_img:
                            st.image(stock_img, caption=f"Stock ({top_match_id}.dat)", use_container_width=True)
                        else:
                            st.info("Sin Imagen 3D")
                    else:
                        st.info("Baja Confianza")

                with col3:
                    st.markdown("**Top Matches (Vector DB):**")
                    for match_idx, match in enumerate(matches):
                        ldraw_id = match['metadata'].get('ldraw_id', 'Unknown')
                        sim = match['similarity']
                        
                        # Count the best match for the final summary (threshold > 0.4 to be safe)
                        if match_idx == 0 and sim > 0.4:
                            identified_parts[ldraw_id] = identified_parts.get(ldraw_id, 0) + 1
                            
                        # Use a progress bar to show similarity
                        st.markdown(f"**{match_idx + 1}. ID: `{ldraw_id}`** (Sim: {sim:.2f})")
                        st.progress(max(0.0, min(1.0, float(sim))), text=None)
            else:
                with col3:
                    st.info(f"Generated Vector (Size: {embedding.shape[0]})")
                    st.code(str(embedding[:5]) + " ...")

    # Final Summary Metrics
    st.markdown("---")
    st.subheader("📊 Recognition Summary")
    
    col_a, col_b = st.columns(2)
    col_a.metric("Total Bounding Boxes (YOLO)", len(boxes))
    col_b.metric("Total Pieces Identified (Vector DB)", sum(identified_parts.values()))
    
    if identified_parts:
        st.write("**Piece Breakdown:**")
        cols = st.columns(4)
        for idx, (p_id, count) in enumerate(sorted(identified_parts.items(), key=lambda item: item[1], reverse=True)):
            with cols[idx % 4]:
                st.info(f"🧱 **{p_id}**: {count} units")
