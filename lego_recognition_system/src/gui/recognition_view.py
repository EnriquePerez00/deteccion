import streamlit as st
import numpy as np
import os
import ssl
import urllib.request
import io
import certifi
from PIL import Image, ImageDraw, ImageFont

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
from src.logic.golden_crop import GoldenCropExtractor


@st.cache_resource(show_spinner="⏳ Cargando modelos de IA...")
def load_models(models_dir):
    """Loads and caches YOLO model + vector index. Stores status in session_state."""
    status = {
        "yolo_name": None,
        "yolo_ok": False,
        "index_count": 0,
        "index_ok": False,
        "errors": []
    }

    if YOLO is None:
        status["errors"].append("El paquete `ultralytics` no está instalado.")
        st.session_state["model_status"] = status
        return None, None, None

    # ── YOLO MODEL ──────────────────────────────────────────────────────────────
    yolo_model = None
    yolo_dir = os.path.join(models_dir, "yolo_model")
    yolo_path = None

    if os.path.exists(yolo_dir):
        pt_files = sorted([f for f in os.listdir(yolo_dir) if f.endswith(".pt")], reverse=True)
        if pt_files:
            yolo_path = os.path.join(yolo_dir, pt_files[0])

    if yolo_path and os.path.exists(yolo_path):
        try:
            yolo_model = YOLO(yolo_path)
            status["yolo_name"] = os.path.basename(yolo_path)
            status["yolo_ok"] = True
        except Exception as e:
            status["errors"].append(f"Error cargando YOLO: {e}")
    else:
        status["errors"].append(f"No se encontró modelo YOLO en {yolo_dir}")

    # ── FEATURE EXTRACTOR ───────────────────────────────────────────────────────
    feature_extractor = FeatureExtractor(model_name='dinov2_vits14')

    # ── VECTOR INDEX ─────────────────────────────────────────────────────────────
    vector_index = VectorIndex()
    piezas_dir = os.path.join(models_dir, "piezas_vectores")

    if os.path.exists(piezas_dir):
        # Prefer lego.index (Strategy C default)
        target_index = os.path.join(piezas_dir, "lego.index")
        if not os.path.exists(target_index):
            # Fallback to any .index file if lego.index is missing
            index_files = [f for f in os.listdir(piezas_dir) if f.endswith(".index")]
            if index_files:
                target_index = os.path.join(piezas_dir, index_files[0])
            else:
                target_index = None

        if target_index and os.path.exists(target_index):
            try:
                if vector_index.load(target_index):
                    status["index_count"] = vector_index.index.ntotal
                    status["index_ok"] = True
            except Exception as e:
                status["errors"].append(f"Error cargando índice {os.path.basename(target_index)}: {e}")
        else:
            status["errors"].append(f"No se encontró ningún archivo .index en {piezas_dir}")
    else:
        status["errors"].append(f"Directorio de vectores no existe: {piezas_dir}")

    # ── GOLDEN CROP EXTRACTOR ──────────────────────────────────────────────────
    golden_extractor = GoldenCropExtractor(target_size=384)

    # Store status for sidebar (but also return it for cache-hit support)
    st.session_state["model_status"] = status
    return yolo_model, feature_extractor, vector_index, status, golden_extractor

def render_sidebar_model_status():
    """Reads model_status from session_state and renders it in the sidebar."""
    s = st.session_state.get("model_status")
    
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        # Botón de refresco manual
        if st.button("🔄 Refrescar Modelos", use_container_width=True, help="Limpia el cache y vuelve a escanear la carpeta models/"):
            st.cache_resource.clear()
            st.cache_data.clear()
            st.rerun()
            
        st.markdown("---")
        st.markdown("**Estado del Sistema:**")
        
        if not s:
            st.caption("⏳ Cargando...")
            return

        if s["yolo_ok"]:
            st.success(f"🧠 YOLO: `{s['yolo_name']}`")
        else:
            st.error("❌ Detector YOLO no encontrado")

        if s["index_ok"]:
            st.success(f"📚 {s['index_count']} vectores cargados")
        else:
            st.error("❌ Índice vectorial no encontrado")

        for err in s.get("errors", []):
            st.warning(f"⚠️ {err}")
            
        st.markdown("---")


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_lego_image(ldraw_id):
    """Fetches a thumbnail image of the LEGO part from Rebrickable CDN."""
    clean_id = str(ldraw_id).replace('.dat', '').strip()
    url = f"https://cdn.rebrickable.com/media/parts/ldraw/14/{clean_id}.png"
    try:
        context = ssl.create_default_context(cafile=certifi.where())
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, context=context, timeout=5) as response:
            image_data = response.read()
            return Image.open(io.BytesIO(image_data))
    except Exception:
        return None


def _draw_annotations(image, boxes, polygons, confidences, use_masks):
    """Draw bounding boxes / segmentation polygons on a copy of the image with semi-transparent overlay."""
    # Convert to RGBA for alpha blending
    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Base Image for the outline (sharp)
    drawn = image.convert("RGBA")
    draw_edge = ImageDraw.Draw(drawn)

    # Dynamic styling based on resolution
    base_dim = min(image.size)
    line_width = max(3, int(base_dim / 300))
    font_size = max(14, int(base_dim / 80))
    
    try:
        # Try different font paths common on macOS
        font_paths = [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc"
        ]
        font = None
        for fp in font_paths:
            if os.path.exists(fp):
                font = ImageFont.truetype(fp, font_size)
                break
        if font is None:
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    colors_hex = ["#00FFFF", "#FF6B6B", "#FFE66D", "#4ECDC4",
                  "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD"]
    
    # Convert hex to RGBA
    def hex_to_rgba(h, alpha=100):
        h = h.lstrip('#')
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4)) + (alpha,)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        conf = confidences[i]
        color_hex = colors_hex[i % len(colors_hex)]
        color_rgba_fill = hex_to_rgba(color_hex, 60) # 60/255 transparency
        color_rgba_edge = hex_to_rgba(color_hex, 255)
        
        label = f"#{i+1} {conf:.0%}"

        if use_masks and polygons is not None:
            poly = polygons[i]
            if len(poly) > 2:
                points = [(int(pt[0]), int(pt[1])) for pt in poly]
                # 1. Fill on overlay
                draw.polygon(points, fill=color_rgba_fill, outline=None)
                # 2. Border on main image
                draw_edge.line(points + [points[0]], fill=color_rgba_edge, width=line_width)
        else:
            # Bounding Box mode
            draw.rectangle([x1, y1, x2, y2], fill=color_rgba_fill, outline=None)
            draw_edge.rectangle([x1, y1, x2, y2], outline=color_rgba_edge, width=line_width)

        # Labels (Draw directly on main image for max crispness)
        # Use a background box for the label to ensure readability
        text_bbox = draw_edge.textbbox((x1, max(0, y1 - font_size - 4)), label, font=font)
        draw_edge.rectangle([text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2], fill=color_rgba_edge)
        draw_edge.text((x1, max(0, y1 - font_size - 4)), label, fill="black", font=font)

    # Composite overlay onto the original image
    result = Image.alpha_composite(drawn, overlay).convert("RGB")
    return result


def render_recognition_ui(uploaded_file, models_dir, conf_threshold, similarity_threshold):
    """Renders the full recognition pipeline visualization."""

    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"Error cargando imagen: {e}")
        return

    yolo, extractor, v_index, _, golden_extractor = load_models(models_dir)
    if yolo is None or extractor is None:
        st.error("❌ Los modelos no pudieron cargarse. Revisa los errores en el sidebar.")
        return

    # ── PHASE 1: YOLO DETECTION (SAHI — Slicing Aided Hyper Inference) ─────────
    st.subheader("Fase 1 — Detección YOLO (SAHI — Tiled 24MP)")

    from src.logic.sahi_slicer import run_sahi_inference, generate_slices

    orig_w, orig_h = image.size

    # Progress UI
    sahi_status = st.empty()
    sahi_progress = st.progress(0)

    def _sahi_progress(current, total):
        sahi_progress.progress(current / total)
        sahi_status.text(f"🔪 Procesando tile {current}/{total}...")

    with st.spinner("Ejecutando SAHI (tiled inference)..."):
        final_detections = run_sahi_inference(
            yolo, image,
            conf_threshold=conf_threshold,
            slice_size=1824,
            overlap_ratio=0.30,
            iou_threshold=0.5,
            center_bonus=0.05,
            progress_callback=_sahi_progress,
        )

    sahi_progress.progress(1.0)
    sahi_status.text(f"✅ SAHI completo: {len(final_detections)} detecciones tras NMS global.")

    num_detections = len(final_detections)

    # ── Debug info ───────────────────────────────────────────────────────────
    with st.expander("🔍 Debug — SAHI Info", expanded=(num_detections == 0)):
        n_tiles = len(generate_slices(image, 1824, 0.30))
        st.markdown(f"**Imagen:** `{orig_w}×{orig_h}` | **Tiles:** `{n_tiles}` (1824px, 30% overlap)")
        if final_detections:
            confs_arr = [d['conf'] for d in final_detections]
            st.bar_chart({"conf": confs_arr})

    if num_detections == 0:
        st.warning(f"⚠️ No se detectaron piezas LEGO con confianza ≥ {conf_threshold:.0%}.")
        st.image(image, caption="Imagen original (sin detecciones)", use_container_width=True)
        return

    # Build scaled_boxes, scaled_polygons, confidences from SAHI results
    # (coordinates are already in original image space)
    scaled_boxes = [d['box'] for d in final_detections]
    confidences = np.array([d['conf'] for d in final_detections])

    has_any_polygon = any(d['polygon'] is not None for d in final_detections)
    use_masks = has_any_polygon
    if has_any_polygon:
        scaled_polygons = [d['polygon'] if d['polygon'] is not None else np.array([]) for d in final_detections]
    else:
        scaled_polygons = None

    # Draw annotations on top of high-res image
    annotated = _draw_annotations(image, scaled_boxes, scaled_polygons, confidences, use_masks)

    # Original and annotated image side by side
    col_orig, col_annotated = st.columns(2)
    with col_orig:
        st.image(image, caption="📸 Imagen Original", use_container_width=True)
    with col_annotated:
        st.image(
            annotated,
            caption=f"🔍 Detección YOLO ({num_detections} piezas)",
            use_container_width=True
        )

    # ── PHASE 2: VECTOR SEARCH CLASSIFICATION ────────────────────────────────────
    st.markdown("---")
    st.subheader("Fase 2 — Clasificación con recortes de alta resolución")

    no_index = v_index is None or v_index.index.ntotal == 0
    if no_index:
        st.warning("⚠️ No hay índice de vectores cargado. Mostrando crops sin clasificar.")

    identified_parts = {}

    # Prepare detections for Golden Crop processing
    # Note: SAHI gives us detections in original image coordinates
    detections_input = []
    for d in final_detections:
        detections_input.append({
            'box': d['box'],
            'polygon': d['polygon']
        })

    # Execute Golden Crop extraction (Standardized 384x384)
    # We pass the full original image and our SAHI detections
    golden_crops = golden_extractor.process_detections(np.array(image), detections_input)

    for i, cropped_img in enumerate(golden_crops):
        conf_val = final_detections[i]['conf']

        with st.expander(f"🧩 PASO {i+1}: Identificando pieza (YOLO Conf: {conf_val:.0%})", expanded=True):
            col_crop, col_ref, col_details = st.columns([1.2, 1.2, 2.1])

            with col_crop:
                st.markdown("**📸 Recorte Real (YOLO)**")
                # Add a border-like container using streamlit components if possible, 
                # but standard st.image is fine for a clean look
                st.image(cropped_img, use_container_width=True)
                st.caption(f"Dim: {cropped_img.width}x{cropped_img.height}")

            # Extract features and search
            with st.spinner(f"Extrayendo vectores para pieza #{i+1}..."):
                embedding = extractor.get_embedding(cropped_img)

            if not no_index:
                # Safety check for cached instances
                import inspect
                search_sig = inspect.signature(v_index.search)
                if 'deduplicate' in search_sig.parameters:
                    matches = v_index.search(embedding, k=3, deduplicate=True)
                else:
                    matches = v_index.search(embedding, k=3)

                if matches:
                    top_match = matches[0]
                    top_id = top_match['metadata'].get('ldraw_id', 'Unknown')
                    top_sim = top_match['similarity']

                    with col_ref:
                        st.markdown("**🎨 Referencia 3D (Top #1)**")
                        if top_sim >= similarity_threshold:
                            stock_img = fetch_lego_image(top_id)
                            if stock_img:
                                st.image(stock_img, use_container_width=True)
                                st.success(f"ID: **{top_id}**")
                            else:
                                st.warning(f"ID: **{top_id}**")
                                st.info("Imagen 3D no encontrada en Rebrickable")
                        else:
                            st.error("⚠️ Pieza Dudosa / Desconocida")
                            st.caption(f"La similitud ({top_sim:.1%}) es menor al umbral ({similarity_threshold:.1%})")

                    with col_details:
                        st.markdown("**🏆 Top 3 Candidatos**")
                        for rank, match in enumerate(matches):
                            ldraw_id = match['metadata'].get('ldraw_id', 'Unknown')
                            sim = match['similarity']

                            medal = ["🥇", "🥈", "🥉"][rank]
                            # Dynamic color based on similarity
                            if sim > 0.85: color = "#2ecc71" # Green
                            elif sim > 0.6: color = "#f1c40f" # Yellow
                            else: color = "#e67e22" # Orange
                            
                            st.markdown(
                                f'<div style="background:#2d3436; padding:10px; border-left: 5px solid {color}; border-radius:5px; margin-bottom:8px; color: white;">'
                                f'<span style="font-size:20px;">{medal}</span> <span style="font-weight:bold; font-size:16px; margin-left:10px;">ID: {ldraw_id}</span>'
                                f'<div style="float:right; color:{color}; font-weight:bold; font-size:16px;">{sim:.1%}</div>'
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                            st.progress(max(0.0, min(1.0, float(sim))))

                        # Count top match for summary (threshold check)
                        if top_sim >= similarity_threshold:
                            identified_parts[top_id] = identified_parts.get(top_id, 0) + 1
                else:
                    with col_details:
                        st.info("Sin resultados en el índice.")
            else:
                with col_details:
                    st.info(f"Vector generado (dim: {embedding.shape[0]})")
                    st.code(str(embedding[:5]) + " ...")
                    st.warning("⚠️ Carga un índice de vectores para clasificar.")

    # ── SUMMARY ──────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Resumen de reconocimiento")

    col_a, col_b = st.columns(2)
    col_a.metric("Piezas detectadas (YOLO)", num_detections)
    col_b.metric("Piezas clasificadas (Vector DB)", sum(identified_parts.values()))

    if identified_parts:
        st.write("**Desglose de piezas identificadas:**")
        cols = st.columns(min(4, len(identified_parts)))
        for idx, (p_id, count) in enumerate(
            sorted(identified_parts.items(), key=lambda x: x[1], reverse=True)
        ):
            with cols[idx % len(cols)]:
                st.info(f"🧱 **{p_id}**: {count} ud.")
