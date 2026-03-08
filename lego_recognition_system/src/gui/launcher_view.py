import streamlit as st
import os
import sys
import json
from datetime import datetime
import shutil
from pathlib import Path
import re
from src.logic.part_resolver import resolve_set, resolve_piece, update_universal_inventory
from src.logic.lego_colors import LEGO_COLORS
from src.logic.model_registry import get_training_status, filter_pending
from src.logic.analysis_helper import get_stable_faces_for_piece

def get_best_python_executable(project_root):
    """
    Finds the most appropriate python executable.
    Priority: 
    1. .venv/bin/python3 (Local project environment)
    2. sys.executable (Fallback to current streamlit runtime)
    """
    venv_python = os.path.join(project_root, ".venv", "bin", "python3")
    if os.path.exists(venv_python):
        return venv_python
    return sys.executable


def natural_sort_key(s):
    """Sort strings containing numbers in natural order (e.g., 1, 2, 11 instead of 1, 11, 2)."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', str(s))]

def render_launcher_ui(project_root):
    # --- Sanitize Session State (Ensures radio labels match) ---
    render_mode_label_map = {
        "images_mix": "images_mix (mezcla para YOLO)",
        "ref_pieza": "ref_pieza (referencia 360°)",
        "both": "both (ambos)",
    }
    if 'render_mode_select' in st.session_state:
        val = st.session_state['render_mode_select']
        if val in render_mode_label_map:
            st.session_state['render_mode_select'] = render_mode_label_map[val]

    # --- Initialize Session State Defaults ---
    if 'training_mode' not in st.session_state:
        st.session_state['training_mode'] = "Referencia Set"
    if 'set_id_input' not in st.session_state:
        st.session_state['set_id_input'] = "75078-1"
    if 'piece_id_input' not in st.session_state:
        st.session_state['piece_id_input'] = "2877, 3001"
    if 'minifig_id_input' not in st.session_state:
        st.session_state['minifig_id_input'] = "sw0001"
    if 'num_parts_slider' not in st.session_state:
        st.session_state['num_parts_slider'] = 5
    if 'render_mode_select' not in st.session_state:
        st.session_state['render_mode_select'] = "images_mix (mezcla para YOLO)"

    st.header("🚀 LEGO Training Launchpad v2.0")
    st.markdown("Pipeline dual: **ref_pieza** (360° referencia) + **images_mix** (mezcla N/K).")
    
    # Environment Check
    best_python = get_best_python_executable(project_root)
    if ".venv" not in best_python:
        st.warning("⚠️ **Entorno virtual (.venv) no detectado.** Se recomienda crear uno e instalar las dependencias de `requirements.txt` para evitar fallos en la indexación.")

    # --- 1. Selección de Objetivo ---
    st.subheader("1. Selección de Objetivo")
    
    mode = st.radio(
        "Modo de entrada:", 
        ["Referencia Set", "Listado de piezas (separado por ,)", "Listado de Minifigs (separado por ,)"], 
        horizontal=True, key="training_mode"
    )
    
    col1, _ = st.columns(2)
    with col1:
        if mode == "Referencia Set":
            target_id = st.text_input("Referencia Set (Batch ID):", key="set_id_input")
            num_parts = st.slider("Máximo de piezas random:", 1, 50, key="num_parts_slider")
        elif mode == "Listado de piezas (separado por ,)":
            target_id = st.text_input("Listado de piezas (separado por ,):", key="piece_id_input")
            num_parts = len([x.strip() for x in target_id.split(",") if x.strip()])
        else:
            target_id = st.text_input("Listado de Minifigs (separado por ,):", key="minifig_id_input")
            num_parts = len([x.strip() for x in target_id.split(",") if x.strip()])

    # --- Color selector per piece (SUPPRESSED: Color-neutral mode) ---
    piece_color_map = {}  # pid -> {color_id, color_name}
    # Color selection logic removed to focus strictly on shape.

    st.divider()

    # --- 2. Modo de Renderizado ---
    st.subheader("2. Modo de Renderizado")
    
    render_mode = st.radio(
        "Tipo de renderizado:",
        ["images_mix (mezcla para YOLO)", "ref_pieza (referencia 360°)", "both (ambos)"],
        horizontal=True,
        key="render_mode_select"
    )
    render_mode_key = render_mode.split(" ")[0]  # Extract 'images_mix', 'ref_pieza', or 'both'
    
    # Resolve pieces for visualization
    detail_parts = []
    if mode == "Referencia Set" and target_id:
        try:
            resolved = resolve_set(target_id, max_parts=num_parts)
            detail_parts = resolved
        except: pass
    elif mode == "Listado de piezas (separado por ,)":
        raw_ids = [x.strip() for x in target_id.split(",") if x.strip()]
        detail_parts = []
        for i, rid in enumerate(raw_ids):
            # Smart resolve (Rebrickable -> LDraw mapping)
            resolved = resolve_piece(rid)
            detail_parts.append({
                "ldraw_id": resolved.get("ldraw_id", rid),
                "part_num": resolved.get("part_num", rid),
                "name": resolved.get("name", rid),
                "color_id": 7, # Default Light Gray
                "color_name": "Light Gray",
            })
    else:
        detail_parts = [{"ldraw_id": x.strip()} for x in target_id.split(",") if x.strip()]

    # Sort pieces by reference (ldraw_id) using natural alphanumeric order
    if detail_parts:
        detail_parts.sort(key=lambda x: natural_sort_key(x.get("ldraw_id", "")))

    # --- Formula Display ---
    if detail_parts:
        # Separate regular pieces from minifigs for the formula
        regular = [p for p in detail_parts 
                   if not (p.get('ldraw_id', '').startswith('sw') or p.get('ldraw_id', '').startswith('fig'))]
        minifigs = [p for p in detail_parts 
                    if p.get('ldraw_id', '').startswith('sw') or p.get('ldraw_id', '').startswith('fig')]
        
        X = len(regular)
        U = len(set(p.get('ldraw_id') for p in regular))  # Distinct shapes
        
        col_formula, col_summary = st.columns(2)
        
        from src.utils.render_tiers import calculate_mix_params
        
        # Calculate parameters based on backend Tier logic
        N_tier, K_tier, M_tier = calculate_mix_params(U)
        
        with col_formula:
            if render_mode_key in ('images_mix', 'both') and U > 0:
                st.markdown(f"""
                **🎯 Objetivo de Entrenamiento:**
                - Imágenes Combinadas (**YOLO26**): **{N_tier}** imgs
                - Variedad total (**formas únicas**): **{U}** tipos
                - Variedad por imagen: **{K_tier}** tipos de pieza
                - Densidad: **{M_tier}** piezas físicas/imagen
                """)
                
                # Update N, K for later use in config/resume logic
                N = N_tier
                K = K_tier
                # Density M_tier is also available
            else:
                N = 0
                K = 0
        
        with col_summary:
            # Dynamically calculate ref_count for ref_pieza mode based on stable geometric faces
            ref_count = 0
            piece_ref_counts = {} # to display in table
            
            if render_mode_key in ('ref_pieza', 'both'):
                # We need to know exact amounts to show, so check analysis per piece
                for i, part in enumerate(detail_parts):
                    p_id = part.get("ldraw_id", str(part))
                    c_id = int(part.get("color_id", 15))
                    c_name = part.get("color_name", "White")
                    is_mf = p_id.startswith('sw') or p_id.startswith('fig')
                    if not is_mf:
                        faces = get_stable_faces_for_piece(p_id, c_id, c_name)
                        count = faces * 24 if faces > 0 else 300 # Fallback 300 if no stable
                        piece_ref_counts[i] = count
                        ref_count += count
            
            st.markdown(f"""
            **📊 Resumen de Imágenes:**
            """)
            if render_mode_key in ('ref_pieza', 'both'):
                st.markdown(f"- 🔍 **Referencia (Vectores)**: **{ref_count}** imgs")
            if render_mode_key in ('images_mix', 'both') and X > 0:
                st.markdown(f"- 🎭 **Mezcla (YOLO)**: **{N}** imgs")
            
            total_plan = 0
            if render_mode_key in ('ref_pieza', 'both'):
                total_plan += ref_count
            if render_mode_key in ('images_mix', 'both'):
                total_plan += N
            st.markdown(f"--- \n**Total**: ~**{total_plan}** imágenes")

            # Resolution Selector for Mix
            if render_mode_key in ('images_mix', 'both'):
                st.markdown("---")
                mix_res_choice = st.radio(
                    "🎯 Resolución images_mix (YOLO):",
                    ["2048x2048 (Recomendado Entrenamiento)", "4096x4096 (Máximo Detalle 50x50)", "1920x1024 (16:9)", "640x640 (Mini)"],
                    index=0,
                    help="Usa resoluciones cuadradas para abarcar los 50x50cm sin distorsiones en los bordes.",
                    key="mix_res_choice_ui"
                )
                if "4096" in mix_res_choice:
                    st.session_state['mix_res_val'] = 4096
                    st.session_state['mix_res_square'] = True
                elif "2048" in mix_res_choice:
                    st.session_state['mix_res_val'] = 2048
                    st.session_state['mix_res_square'] = True
                elif "1920" in mix_res_choice:
                    st.session_state['mix_res_val'] = 1920
                    st.session_state['mix_res_square'] = False
                else:
                    st.session_state['mix_res_val'] = 640
                    st.session_state['mix_res_square'] = True

            # Resolution Selector for Reference Pipeline
            if render_mode_key in ('ref_pieza', 'both'):
                st.markdown("---")
                ref_res_choice = st.radio(
                    "🎯 Resolución ref_pieza (Vectores 360):",
                    ["512x512 (Óptimo nativo FAISS/DINOv2)", "1024x1024 (Alta resolución)"],
                    index=0,
                    help="Resolución de las imágenes aisladas generadas para calcular los embeddings de referencia.",
                    key="ref_res_choice_ui"
                )
                if "1024" in ref_res_choice:
                    st.session_state['ref_res_val'] = 1024
                else:
                    st.session_state['ref_res_val'] = 512
        
        # Detailed table
        table_data = []
        for i, part in enumerate(detail_parts):
            p_id = part.get("ldraw_id", str(part))
            orig_id = part.get("part_num", p_id)
            is_mf = p_id.startswith('sw') or p_id.startswith('fig')
            
            # Show both IDs if they differ to avoid confusion (Rebrickable vs LDraw)
            display_id = f"{p_id} ({orig_id})" if p_id != orig_id else p_id
            
            row = {
                "Pieza (LDraw)": display_id,
                "Tipo": "Minifig" if is_mf else "Regular",
                "num imagenes": f"{piece_ref_counts.get(i, '—')}" if render_mode_key in ('ref_pieza', 'both') else "",
                "images_mix": "Excluida" if is_mf else ("Incluida" if render_mode_key in ('images_mix', 'both') else ""),
            }
            table_data.append(row)
        
        st.table(table_data)
    else:
        mix_ratio = 0.75
        st.info("Ingresa una referencia o ID de pieza para ver el plan de renderizado.")

    st.markdown("""
    > 📷 **Zona**: 20×20cm @ 70cm (iPhone 16, 24MP)  
    > 🎨 **Motor**: CYCLES (SSS + Raytracing)  
    > 🖥️ **Resolución**: 1280px (Unificada)
    """)

    launch_date = datetime.now().strftime("%Y%m%d_%H%M")
    clean_id = target_id.replace(", ", "_").replace(",", "_")
    full_ref = f"{clean_id}_{launch_date}"
    st.info(f"Referencia de sesión: **{full_ref}**")

    # --- 3. Renderizado Local (Mac M4) ---
    st.divider()
    st.subheader("3. Renderizado Local (M4 Pro)")
    
    force_render = st.checkbox("🧹 Forzar re-renderizado (borrar caché local)", value=False, help="Elimina los renders previos de las piezas seleccionadas para forzar una nueva generación completa.")
    gen_zip = st.checkbox("📦 Generar paquete ZIP para entrenamiento (Lightning/Kaggle)", value=False, help="Crea un archivo .zip con el dataset y el código fuente. Desactívalo para ahorrar espacio en pruebas locales.")

    if st.button("🍎 Iniciar Renderizado Local", type="primary"):
        # 1. Prepare config_train.json
        config_path = os.path.join(project_root, "config_train.json")
        
        if mode == "Referencia Set":
            all_requested_ids = [
                {
                    "part_id": p['ldraw_id'],
                    "color_id": p.get('color_id', 15),
                    "color_name": p.get('color_name', 'White'),
                }
                for p in detail_parts
            ]
        elif mode == "Listado de piezas (separado por ,)":
            all_requested_ids = [
                {
                    "part_id": p['ldraw_id'],
                    "color_id": p.get('color_id', 15),
                    "color_name": p.get('color_name', 'White'),
                }
                for p in detail_parts
            ]
        else:  # Minifigs
            all_requested_ids = [x.strip() for x in target_id.split(",") if x.strip()]

        if not all_requested_ids:
            st.warning("⚠️ No hay piezas seleccionadas para renderizar.")
            return

        # Cache management
        pending_ids = []
        skipped_ids = []
        render_base = os.path.join(project_root, "render_local")

        if force_render:
            import shutil
            # 1. Clear mix cache if applicable
            if render_mode_key in ("images_mix", "both"):
                mix_dir = os.path.join(render_base, "images_mix")
                if os.path.exists(mix_dir):
                    try:
                        shutil.rmtree(mix_dir)
                        st.write(f"🗑️ Caché borrada: `images_mix`")
                    except: pass

        # Cache check for images_mix is handled by Blender or the task manager logic if needed, 
        # but we remove the specific Streamlit 'Resume' info blocks here.

        for i, p_id in enumerate(all_requested_ids):
            p_id_str = p_id.get("part_id", str(p_id)) if isinstance(p_id, dict) else str(p_id)
            dir_key = p_id_str

            if force_render:
                # 2. Clear per-piece cache if applicable
                if render_mode_key in ("ref_pieza", "both"):
                    import shutil
                    # Use simplified directory key: {p_id}
                    for check_dir in [
                        os.path.join(render_base, "ref_pieza", dir_key),
                        os.path.join(render_base, p_id_str),  # legacy
                    ]:
                        if os.path.exists(check_dir):
                            try:
                                shutil.rmtree(check_dir)
                                st.write(f"🗑️ Caché borrada: `ref_pieza/{os.path.basename(check_dir)}`")
                            except: pass
                
                pending_ids.append(p_id)
                continue

            # Check both new structure and legacy
            found_dir = None
            if render_mode_key in ("ref_pieza", "both"):
                for check_dir in [
                    os.path.join(render_base, "ref_pieza", dir_key, "images"),
                    os.path.join(render_base, "ref_pieza", p_id_str, "images"),
                    os.path.join(render_base, p_id_str, "images"),
                ]:
                    if os.path.exists(check_dir):
                        found_dir = check_dir
                        break
            
            if found_dir:
                existing_imgs = [f for f in os.listdir(found_dir) if f.lower().endswith('.jpg')]
                required = piece_ref_counts.get(i, 300)
                
                if len(existing_imgs) >= required:
                    skipped_ids.append(p_id_str)
                else:
                    pending_ids.append(p_id)
            else:
                # If it's ref_pieza mode and dir not found, it's pending.
                # If it's pure images_mix mode, pieces are always "pending" to be included in random drops,
                # but we don't count them as "skipped" based on ref folders.
                pending_ids.append(p_id)
        
        # In images_mix only mode, we don't skip pieces based on ref folders
        if render_mode_key == "images_mix":
            skipped_ids = []
            pending_ids = all_requested_ids
        
        if skipped_ids:
            st.info(f"⏭️ **Caché detectada:** Saltando {len(skipped_ids)} piezas completas: `{', '.join(skipped_ids[:10])}`")
            
        if not pending_ids:
            st.success("✅ Todas las piezas ya están en caché local.")
            skip_render = True
        else:
            skip_render = False

        if not skip_render:
            render_settings = {
                "engine": "CYCLES",
                "render_mode": render_mode_key,
                "mix_ratio": mix_ratio if render_mode_key in ('images_mix', 'both') else 0.75,
                "res": st.session_state.get('mix_res_val', 1920),
                "ref_res": st.session_state.get('ref_res_val', 640),
                "square": st.session_state.get('mix_res_square', False)
            }
            


            
            with open(config_path, "w") as f:
                json.dump({
                    "session_reference": full_ref,
                    "target_parts": pending_ids,
                    "render_settings": render_settings
                }, f, indent=4)
            
            # UI Progress - Unified Pipeline
            st.markdown(f"### ⚙️ Pipeline End-to-End ({render_mode_key})")
            
            # Progress Metrics
            m1, m2, m3, m4, m5 = st.columns(5)
            total_images_ui = m1.metric("Total Imágenes", "Calculando...")
            progress_pct_ui = m2.metric("Progreso Global", "0%")
            generated_ui = m3.metric("Imágenes generadas", "0")
            vector_ui = m4.metric("Vectores creados", f"0/{ref_count}")
            status_ui = m5.metric("Fase Actual", "Iniciando...")
            
            progress_bar = st.progress(0)

            
            # --- PHASE 1: RENDERING (0% to 80%) ---
            status_ui.metric("Fase Actual", "1/2: Renderizado 3D")
            import subprocess
            
            # Resolve the correct python (favouring .venv even if Streamlit is system-wide)
            best_python = get_best_python_executable(project_root)
            if best_python != sys.executable:
                st.caption(f"🔧 Utilizando entorno: `.venv` ({os.path.basename(best_python)})")
            
            render_script = os.path.join(project_root, "run_local_render.py")
            cmd_render = [best_python, render_script]
            if not gen_zip:
                cmd_render.append("--no-zip")
            
            # Force unbuffered Python output
            env = dict(os.environ, PYTHONUNBUFFERED='1')
            process_render = subprocess.Popen(cmd_render, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
            
            total_found = 0
            for line in process_render.stdout:
                if "Total Render Plan:" in line or "Formula:" in line:
                    try:
                        if "Total Render Plan:" in line:
                            total_found = int(line.split("Total Render Plan:")[1].strip().split(" ")[0])
                        elif "=" in line and "images" in line.lower():
                            parts_line = line.split("=")
                            if len(parts_line) > 2:
                                total_found = int(parts_line[-1].strip().split(" ")[0])
                        total_images_ui.metric("Total Imágenes", f"{total_found}")
                    except: pass
                    
                if "Progress:" in line:
                    try:
                        parts_progress = line.split("Progress:")[1].strip().split(" ")[0]
                        done, total = map(int, parts_progress.split("/"))
                        
                        # Rendering is 80% of the total pipeline
                        pct = (done / total) * 0.80
                        progress_bar.progress(min(1.0, pct))
                        progress_pct_ui.metric("Progreso Global", f"{int(pct*100)}%")

                        
                        # Update Generated Images metric
                        generated_ui.metric("Imágenes generadas", f"{done}")
                            
                    except: pass

            
            process_render.wait()
            render_success = (process_render.returncode == 0)
            
            # --- PHASE 2: VECTOR INDEXING (80% to 100%) ---
            if render_success:
                status_ui.metric("Fase Actual", "2/2: Vectores FAISS")

                progress_bar.progress(0.85)
                progress_pct_ui.metric("Progreso Global", "85%")
                
                index_script = os.path.join(project_root, "run_incremental_indexing.py")
                cmd_index = [best_python, index_script]
                env = dict(os.environ, PYTHONUNBUFFERED='1')
                process_index = subprocess.Popen(cmd_index, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
                
                vectors_processed = 0
                for line in process_index.stdout:
                    l_strip = line.strip()
                    if l_strip:
                        if l_strip.startswith("VECTOR_PROCESSED:"):
                            vectors_processed += 1
                            vector_ui.metric("Vectores creados", f"{vectors_processed}/{ref_count}")
                        elif "PROGRESS:" in l_strip:
                            try:
                                # Example line: "PROGRESS: 5/10 | Processing ..."
                                prog_part = l_strip.split("PROGRESS:")[1].split("|")[0].strip()
                                done, total = map(int, prog_part.split("/"))
                                # Map 0..100% of indexing to 80%..95% of total UI progress
                                sub_pct = (done / total)
                                pct = 0.80 + (sub_pct * 0.15)
                                progress_bar.progress(min(0.99, pct))
                                progress_pct_ui.metric("Progreso Global", f"{int(pct*100)}%")
                            except: pass
                        elif "Unified index updated" in l_strip or "Part index saved" in l_strip:
                            progress_bar.progress(0.98)
                            progress_pct_ui.metric("Progreso Global", "98%")

                
                process_index.wait()
                index_success = (process_index.returncode == 0)
            else:
                index_success = False
            
            pipeline_success = render_success and index_success
            
        else:
            pipeline_success = True

        if pipeline_success:

            status_ui.success("Pipeline finalizado con éxito")
            if not skip_render:
                progress_bar.progress(1.0)
                progress_pct_ui.metric("Progreso Global", "100%")
                status_ui.metric("Fase Actual", "Completado ✅")

            st.success("✅ Renderizado e Indexación Vectorial completados localmente.")
            render_dir = os.path.join(project_root, "render_local")
            all_images = list(Path(render_dir).rglob("images/*.jpg"))

            import glob as _glob
            zips = sorted(_glob.glob(os.path.join(project_root, "lightning_dataset_*.zip")))
            if zips:
                latest_zip = zips[-1]
                zip_size = os.path.getsize(latest_zip) / (1024 * 1024)
                st.success(f"📦 ZIP listo: **{os.path.basename(latest_zip)}** ({zip_size:.1f} MB)")
                st.session_state['latest_lightning_zip'] = latest_zip
        else:
            if not render_success:
                st.error(f"❌ Fallo en Renderizado (código {process_render.returncode})")
            elif not index_success:
                st.error(f"❌ Fallo en Indexación Vectorial (código {process_index.returncode})")
            else:
                st.error("❌ Fallo desconocido en el pipeline.")

    # --- 4. Notebooks (Lightning + Kaggle) ---
    st.divider()
    st.subheader("4. Generar Notebooks (Lightning AI + Kaggle)")
    
    import glob as _glob
    existing_zips = sorted(_glob.glob(os.path.join(project_root, "lightning_dataset_*.zip")))
    
    if existing_zips:
        latest_zip = existing_zips[-1]
        zip_size = os.path.getsize(latest_zip) / (1024 * 1024)
        st.info(f"📦 Último dataset: **{os.path.basename(latest_zip)}** ({zip_size:.1f} MB)")
        
        manifest_path = os.path.join(project_root, "render_local", "dataset_manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            st.markdown(f"**{manifest.get('total_images', '?')}** imágenes | "
                       f"**{manifest.get('total_pieces', '?')}** piezas | "
                       f"Modo: **{manifest.get('render_mode', 'legacy')}**")
    else:
        st.warning("⚠️ No hay dataset generado. Usa el botón de renderizado local primero.")
    
    col_nb1, col_nb2 = st.columns(2)
    
    with col_nb1:
        if st.button("⚡ Generar Notebook Lightning AI"):
            try:
                from src.utils.notebook_generator_v6 import generate_lightning_v6
                gen_timestamp = datetime.now().strftime('%Y%m%d_%H%M')
                nb_path = generate_lightning_v6(
                    output_dir=os.path.join(project_root, "notebooks"),
                    timestamp=gen_timestamp
                )
                st.success(f"✅ Lightning: **{os.path.basename(nb_path)}**")
            except Exception as e:
                st.error(f"❌ Error: {e}")
    
    with col_nb2:
        if st.button("🏔️ Generar Notebook Kaggle"):
            try:
                from src.utils.notebook_generator_v6 import generate_kaggle_v6
                gen_timestamp = datetime.now().strftime('%Y%m%d_%H%M')
                nb_path = generate_kaggle_v6(
                    output_dir=os.path.join(project_root, "notebooks"),
                    timestamp=gen_timestamp
                )
                st.success(f"✅ Kaggle: **{os.path.basename(nb_path)}**")
            except Exception as e:
                st.error(f"❌ Error: {e}")
    


    # Clean state tracking
    if 'prev_target_id' not in st.session_state or st.session_state['prev_target_id'] != target_id:
        st.session_state['prev_target_id'] = target_id
