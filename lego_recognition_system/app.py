import streamlit as st
import os
import sys

# Fix for macOS: "OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized."
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Ensure src is in the python path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.gui.recognition_view import render_recognition_ui, render_sidebar_model_status
from src.gui.launcher_view import render_launcher_ui

st.set_page_config(
    page_title="LEGO Vision AI & Trainer",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.sidebar.title("🧩 LEGO Vision AI")
    st.sidebar.markdown("---")

    app_mode = st.sidebar.selectbox(
        "Selecciona aplicación:",
        ["🚀 Entrenamiento (Launchpad)", "🔍 Reconocimiento (Test)"]
    )

    st.sidebar.markdown("---")

    if app_mode == "🔍 Reconocimiento (Test)":
        st.sidebar.info(
            "**Arquitectura 2 etapas:**\n"
            "1. **YOLOv11-Seg** detecta bounding boxes.\n"
            "2. **Búsqueda vectorial** clasifica cada pieza."
        )

        models_dir = os.path.join(PROJECT_ROOT, "models")

        # Cargar modelos proactivamente
        # Cargar modelos y actualizar estado (incluso si hay cache hit)
        from src.gui.recognition_view import load_models
        _, _, _, status, _ = load_models(models_dir)
        st.session_state["model_status"] = status
        
        # Mostrar el estado en la barra lateral
        render_sidebar_model_status()

        st.sidebar.markdown("---")
        conf_threshold = st.sidebar.slider(
            "Umbral de confianza YOLO",
            min_value=0.05,
            max_value=1.0,
            value=0.15,
            step=0.05,
            help="Valores bajos (0.10–0.20) son recomendados para el modelo universal."
        )

        st.sidebar.markdown("---")
        st.sidebar.subheader("🎯 Fase 2 — Vectores")
        similarity_threshold = st.sidebar.slider(
            "Umbral de Similitud (Fase 2)",
            min_value=0.0,
            max_value=1.0,
            value=0.45,
            step=0.05,
            help="Si la similitud vectorial es menor a esto, la pieza se marcará como desconocida o dudosa."
        )

        st.title("🔍 Reconocimiento Interactivo de Piezas")
        st.markdown("Sube una imagen para ejecutar el pipeline de reconocimiento de 2 etapas.")

        uploaded_file = st.file_uploader(
            "Elige una imagen (JPG/PNG)",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            render_recognition_ui(uploaded_file, models_dir, conf_threshold, similarity_threshold)
        else:
            test_img_path = os.path.join(PROJECT_ROOT, "temp_query.jpg")
            if os.path.exists(test_img_path):
                st.info("No hay imagen subida. Usando muestra 'temp_query.jpg'.")
                with open(test_img_path, "rb") as f:
                    render_recognition_ui(f, models_dir, conf_threshold, similarity_threshold)
            else:
                st.markdown(
                    """
                    ### 📷 Cómo usar este módulo

                    1. Usa el **slider de confianza** (panel izquierdo) — recomendado: `0.10`–`0.20`
                    2. Sube una imagen con piezas LEGO
                    3. El sistema detectará las piezas y mostrará las 3 más similares del índice

                    > 💡 **Tip:** El modelo fue entrenado con imágenes sintéticas de Blender.
                    > Las mejores detecciones se logran con fondos neutros y buena iluminación.
                    """
                )

    elif app_mode == "🚀 Entrenamiento (Launchpad)":
        render_launcher_ui(PROJECT_ROOT)

if __name__ == "__main__":
    main()
