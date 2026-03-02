import streamlit as st
import os
import sys
import json
from datetime import datetime
import shutil
from pathlib import Path
from src.logic.part_resolver import resolve_set, resolve_piece, update_universal_inventory
from src.logic.lego_colors import LEGO_COLORS
from src.logic.model_registry import get_training_status, filter_pending

def render_launcher_ui(project_root):
    st.header("🚀 LEGO Training Launchpad v2.0")
    st.markdown("Pipeline dual: **ref_pieza** (360° referencia) + **images_mix** (mezcla N/K).")

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
            target_id = st.text_input("Referencia Set (Batch ID):", value="75078-1", key="set_id_input")
            num_parts = st.slider("Máximo de piezas random:", 1, 50, 5, key="num_parts_slider")
        elif mode == "Listado de piezas (separado por ,)":
            target_id = st.text_input("Listado de piezas (separado por ,):", value="2877, 3001", key="piece_id_input")
            num_parts = len([x.strip() for x in target_id.split(",") if x.strip()])
        else:
            target_id = st.text_input("Listado de Minifigs (separado por ,):", value="sw0001", key="minifig_id_input")
            num_parts = len([x.strip() for x in target_id.split(",") if x.strip()])

    # --- Color selector per piece (only for manual list mode) ---
    piece_color_map = {}  # pid -> {color_id, color_name}
    if mode == "Listado de piezas (separado por ,)" and target_id:
        raw_piece_ids = [x.strip() for x in target_id.split(",") if x.strip()]
        if raw_piece_ids:
            st.markdown("**🎨 Color por pieza:**")
            color_options = {v["name"]: k for k, v in LEGO_COLORS.items()}
            color_names_list = list(color_options.keys())
            default_color_name = "White"
            default_color_idx = color_names_list.index(default_color_name) if default_color_name in color_names_list else 0
            cols_color = st.columns(min(len(raw_piece_ids), 4))
            for ci, pid in enumerate(raw_piece_ids):
                col_c = cols_color[ci % len(cols_color)]
                with col_c:
                    selected_color_name = st.selectbox(
                        f"Pieza `{pid}`:",
                        color_names_list,
                        index=default_color_idx,
                        key=f"color_select_{pid}_{ci}"
                    )
                    selected_color_id = color_options[selected_color_name]
                    piece_color_map[pid] = {"color_id": selected_color_id, "color_name": selected_color_name}

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
        detail_parts = [
            {
                "ldraw_id": x.strip(),
                "color_id": piece_color_map.get(x.strip(), {}).get("color_id", 15),
                "color_name": piece_color_map.get(x.strip(), {}).get("color_name", "White"),
            }
            for x in target_id.split(",") if x.strip()
        ]
    else:
        detail_parts = [{"ldraw_id": x.strip()} for x in target_id.split(",") if x.strip()]

    # --- Formula Display ---
    if detail_parts:
        # Separate regular pieces from minifigs for the formula
        regular = [p for p in detail_parts 
                   if not (p.get('ldraw_id', '').startswith('sw') or p.get('ldraw_id', '').startswith('fig'))]
        minifigs = [p for p in detail_parts 
                    if p.get('ldraw_id', '').startswith('sw') or p.get('ldraw_id', '').startswith('fig')]
        
        X = len(regular)
        
        col_formula, col_summary = st.columns(2)
        
        with col_formula:
            if render_mode_key in ('images_mix', 'both') and X > 0:
                mix_ratio = st.slider("Mix ratio (K/X):", 0.3, 0.9, 0.75, 0.05, key="mix_ratio")
                K = max(1, int(X * mix_ratio))
                N = min(1000, max(500, (X * 1500) // K))
                
                st.markdown(f"""
                **📐 Fórmula images_mix:**
                - `X` = **{X}** piezas regulares
                - `K` = **{K}** piezas/tipo por imagen ({mix_ratio*100:.0f}% de X)
                - `N` = min(1000, max(500, ({X} × 1500) / {K})) = **{N}** imágenes
                - **30** piezas físicas por imagen
                """)
            else:
                mix_ratio = 0.75
                K = 0
                N = 0
        
        with col_summary:
            ref_count = len(detail_parts) * 300
            st.markdown(f"""
            **📊 Resumen del plan:**
            """)
            if render_mode_key in ('ref_pieza', 'both'):
                st.markdown(f"- 📸 **ref_pieza**: {len(detail_parts)} piezas × 300 = **{ref_count}** imgs")
            if render_mode_key in ('images_mix', 'both') and X > 0:
                st.markdown(f"- 🎲 **images_mix**: **{N}** imgs (excl. {len(minifigs)} minifigs)")
            total_plan = 0
            if render_mode_key in ('ref_pieza', 'both'):
                total_plan += ref_count
            if render_mode_key in ('images_mix', 'both'):
                total_plan += N
            st.markdown(f"- **Total**: ~**{total_plan}** imágenes sintéticas")
        
        # Detailed table
        table_data = []
        for part in detail_parts:
            p_id = part.get("ldraw_id", str(part))
            is_mf = p_id.startswith('sw') or p_id.startswith('fig')
            row = {
                "Pieza": p_id,
                "Color": part.get("color_name", "-"),
                "Tipo": "Minifig" if is_mf else "Regular",
                "ref_pieza": "300 imgs" if render_mode_key in ('ref_pieza', 'both') else "",
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
            raw_ids = [x.strip() for x in target_id.split(",") if x.strip()]
            all_requested_ids = [
                {
                    "part_id": pid,
                    "color_id": piece_color_map.get(pid, {}).get("color_id", 15),
                    "color_name": piece_color_map.get(pid, {}).get("color_name", "White"),
                }
                for pid in raw_ids
            ]
        else:  # Minifigs
            all_requested_ids = [x.strip() for x in target_id.split(",") if x.strip()]

        if not all_requested_ids:
            st.warning("⚠️ No hay piezas seleccionadas para renderizar.")
            return

        # Cache check
        pending_ids = []
        skipped_ids = []
        render_base = os.path.join(project_root, "render_local")
        
        for p_id in all_requested_ids:
            if isinstance(p_id, dict):
                p_id_str = p_id.get("part_id", str(p_id))
            else:
                p_id_str = str(p_id)

            if force_render:
                # Clear cache for this piece
                import shutil
                for check_dir in [
                    os.path.join(render_base, "ref_pieza", p_id_str),
                    os.path.join(render_base, p_id_str),
                    os.path.join(render_base, "images_mix"), # Clear mix too if forcing
                ]:
                    if os.path.exists(check_dir):
                        try:
                            shutil.rmtree(check_dir)
                            st.write(f"🗑️ Caché borrada: `{os.path.basename(check_dir)}`")
                        except: pass
                
                if isinstance(p_id, dict):
                    pending_ids.append(p_id)
                else:
                    pending_ids.append(p_id_str)
                continue

            # Check both new structure and legacy
            for check_dir in [
                os.path.join(render_base, "ref_pieza", p_id_str, "images"),
                os.path.join(render_base, p_id_str, "images"),
            ]:
                if os.path.exists(check_dir):
                    existing_imgs = [f for f in os.listdir(check_dir) if f.lower().endswith('.jpg')]
                    if len(existing_imgs) > 0:
                        skipped_ids.append(p_id_str)
                        break
            else:
                if isinstance(p_id, dict):
                    pending_ids.append(p_id)
                else:
                    pending_ids.append(p_id_str)
        
        if skipped_ids:
            st.info(f"⏭️ **Caché detectada:** Saltando {len(skipped_ids)} piezas: `{', '.join(skipped_ids[:10])}`")
            
        if not pending_ids:
            st.success("✅ Todas las piezas ya están en caché local.")
            skip_render = True
        else:
            skip_render = False

        if not skip_render:
            render_settings = {
                "engine": "CYCLES",
                "render_mode": render_mode_key,
                "mix_ratio": mix_ratio if render_mode_key in ('images_mix', 'both') else 0.75
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
            m1, m2, m3 = st.columns(3)
            total_images_ui = m1.metric("Total Imágenes", "Calculando...")
            progress_pct_ui = m2.metric("Progreso Global", "0%")
            status_ui = m3.metric("Fase Actual", "Iniciando...")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # --- PHASE 1: RENDERING (0% to 80%) ---
            status_ui.metric("Fase Actual", "1/2: Renderizado 3D")
            import subprocess
            render_script = os.path.join(project_root, "run_local_render.py")
            cmd_render = [sys.executable, render_script]
            if not gen_zip:
                cmd_render.append("--no-zip")
            
            process_render = subprocess.Popen(cmd_render, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            
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
                        status_text.text(f"🚀 Renderizando imagen {done} de {total}...")
                    except: pass
                else:
                    l_strip = line.strip()
                    if l_strip: status_text.text(f"Render: {l_strip}")
            
            process_render.wait()
            render_success = (process_render.returncode == 0)
            
            # --- PHASE 2: VECTOR INDEXING (80% to 100%) ---
            if render_success:
                status_ui.metric("Fase Actual", "2/2: Vectores FAISS")
                status_text.text("🧠 Iniciando extracción DINOv2 y Clustering...")
                progress_bar.progress(0.85)
                progress_pct_ui.metric("Progreso Global", "85%")
                
                index_script = os.path.join(project_root, "run_incremental_indexing.py")
                cmd_index = [sys.executable, index_script]
                process_index = subprocess.Popen(cmd_index, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                
                for line in process_index.stdout:
                    l_strip = line.strip()
                    if l_strip:
                        status_text.text(f"Vectores: {l_strip}")
                        if "PROGRESS:" in l_strip:
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
                        elif "Unified index saved" in l_strip:
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
            if not skip_render:
                progress_bar.progress(1.0)
                progress_pct_ui.metric("Progreso Global", "100%")
                status_ui.metric("Fase Actual", "Completado ✅")
                status_text.text("🎉 Pipeline End-to-End finalizado con éxito.")
            st.success("✅ Renderizado e Indexación Vectorial completados localmente.")
            render_dir = os.path.join(project_root, "render_local")
            all_images = list(Path(render_dir).rglob("images/*.jpg"))
            if all_images:
                import random
                st.markdown("### 🖼️ Muestras Aleatorias")
                
                if st.button("🔄 Refrescar Muestras"):
                    st.rerun()

                cols = st.columns(4)
                num_to_show = min(4, len(all_images))
                samples = random.sample(all_images, num_to_show)
                for i, img_p in enumerate(samples):
                    piece_id = img_p.parent.parent.name
                    cols[i].image(str(img_p), caption=f"Pieza: {piece_id}", use_container_width=True)
            
            import glob as _glob
            zips = sorted(_glob.glob(os.path.join(project_root, "lightning_dataset_*.zip")))
            if zips:
                latest_zip = zips[-1]
                zip_size = os.path.getsize(latest_zip) / (1024 * 1024)
                st.success(f"📦 ZIP listo: **{os.path.basename(latest_zip)}** ({zip_size:.1f} MB)")
                st.session_state['latest_lightning_zip'] = latest_zip
        else:
            st.error(f"❌ Fallo {process.returncode}")

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
