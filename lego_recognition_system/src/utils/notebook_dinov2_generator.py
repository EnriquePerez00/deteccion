"""
Notebook DINOv2 Generator — Lightning AI + Kaggle
==================================================
Generates independent DINOv2 fine-tuning notebooks with dedicated ZIP packages.
"""
import json
import os
import datetime
import zipfile
import shutil
from src.utils.notebook_dinov2_templates import (
    DINOV2_MARKDOWN_HEADER,
    DINOV2_C0_SETUP_LIGHTNING,
    DINOV2_C0_SETUP_KAGGLE,
    DINOV2_C1_LOAD_DATASET,
    DINOV2_C2_FINETUNE,
    DINOV2_C3_EXPORT,
)


def _make_code_cell(source_lines):
    cleaned = [line.rstrip('\n\r') + '\n' for line in source_lines]
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": cleaned
    }


def _make_markdown_cell(source_lines):
    return {"cell_type": "markdown", "metadata": {}, "source": source_lines}


def _build_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.12"}
        },
        "nbformat": 4, "nbformat_minor": 2
    }


def _create_dataset_zip(zip_path, project_root):
    """
    Create a ZIP with all data needed for DINOv2 fine-tuning.

    Contents:
        ref_pieza/          - Rendered images (only images/ subdirs)
        hard_negatives.json - Hard negative map
        src/logic/triplet_dataset.py
        src/logic/finetune_dinov2.py
    """
    print(f"📦 Creating DINOv2 dataset ZIP: {os.path.basename(zip_path)}...")

    render_dir = os.path.join(project_root, "render_local", "ref_pieza")
    hn_path = os.path.join(project_root, "models", "hard_negatives.json")

    if not os.path.isdir(render_dir):
        raise FileNotFoundError(f"❌ render_local/ref_pieza not found at {render_dir}")
    if not os.path.exists(hn_path):
        raise FileNotFoundError(f"❌ hard_negatives.json not found at {hn_path}")

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # 1. Add ref_pieza images
        total_imgs = 0
        for piece_dir in sorted(os.listdir(render_dir)):
            img_dir = os.path.join(render_dir, piece_dir, "images")
            if not os.path.isdir(img_dir):
                continue
            for img_file in os.listdir(img_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    fp = os.path.join(img_dir, img_file)
                    arcname = f"ref_pieza/{piece_dir}/images/{img_file}"
                    zf.write(fp, arcname=arcname)
                    total_imgs += 1
        print(f"   + {total_imgs} images from ref_pieza/")

        # 2. Add hard_negatives.json
        zf.write(hn_path, arcname="hard_negatives.json")
        print("   + hard_negatives.json")

        # 3. Add scripts
        scripts = [
            ("src/logic/triplet_dataset.py", os.path.join(project_root, "src", "logic", "triplet_dataset.py")),
            ("src/logic/finetune_dinov2.py", os.path.join(project_root, "src", "logic", "finetune_dinov2.py")),
        ]
        for arcname, filepath in scripts:
            if os.path.exists(filepath):
                zf.write(filepath, arcname=arcname)
                print(f"   + {arcname}")

    sz = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"✅ ZIP ready: {os.path.basename(zip_path)} ({sz:.1f} MB)")


def generate_lightning_dinov2(output_dir=None, timestamp=None):
    """
    Generate a Lightning AI notebook + ZIP for DINOv2 fine-tuning.

    Returns:
        Tuple of (notebook_path, zip_path)
    """
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    if output_dir is None:
        output_dir = os.path.join(project_root, 'notebooks')

    lightning_dir = os.path.join(output_dir, 'lightning')
    os.makedirs(lightning_dir, exist_ok=True)

    # Generate notebook
    cells = [
        _make_markdown_cell(DINOV2_MARKDOWN_HEADER),
        _make_code_cell(DINOV2_C0_SETUP_LIGHTNING),
        _make_code_cell(DINOV2_C1_LOAD_DATASET),
        _make_code_cell(DINOV2_C2_FINETUNE),
        _make_code_cell(DINOV2_C3_EXPORT),
    ]
    notebook = _build_notebook(cells)

    nb_filename = f"lightning_dinov2_{timestamp}.ipynb"
    nb_path = os.path.join(lightning_dir, nb_filename)
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=4, ensure_ascii=False)
    print(f"✅ Notebook: {nb_filename}")

    # Generate ZIP
    zip_filename = f"lightning_dinov2_{timestamp}.zip"
    zip_path = os.path.join(lightning_dir, zip_filename)
    _create_dataset_zip(zip_path, project_root)

    return nb_path, zip_path


def generate_kaggle_dinov2(output_dir=None, timestamp=None):
    """
    Generate a Kaggle notebook + ZIP for DINOv2 fine-tuning.

    Returns:
        Tuple of (notebook_path, zip_path)
    """
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    if output_dir is None:
        output_dir = os.path.join(project_root, 'notebooks')

    kaggle_dir = os.path.join(output_dir, 'kaggle')
    os.makedirs(kaggle_dir, exist_ok=True)

    # Generate notebook
    cells = [
        _make_markdown_cell(DINOV2_MARKDOWN_HEADER),
        _make_code_cell(DINOV2_C0_SETUP_KAGGLE),
        _make_code_cell(DINOV2_C1_LOAD_DATASET),
        _make_code_cell(DINOV2_C2_FINETUNE),
        _make_code_cell(DINOV2_C3_EXPORT),
    ]
    notebook = _build_notebook(cells)

    nb_filename = f"kaggle_dinov2_{timestamp}.ipynb"
    nb_path = os.path.join(kaggle_dir, nb_filename)
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=4, ensure_ascii=False)
    print(f"✅ Notebook: {nb_filename}")

    # Generate ZIP
    zip_filename = f"kaggle_dinov2_{timestamp}.zip"
    zip_path = os.path.join(kaggle_dir, zip_filename)
    _create_dataset_zip(zip_path, project_root)

    return nb_path, zip_path
