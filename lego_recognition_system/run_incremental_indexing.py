#!/usr/bin/env python3
"""
🧠 Incremental Vector Indexer
==============================
Scans the entire render_local/ directory and builds/updates
the unified FAISS index in models/piezas_vectores/.

Supports the new dual-mode directory structure:
  - render_local/ref_pieza/{piece_id}/images/
  - render_local/images_mix/images/
  - render_local/{piece_id}/images/ (legacy)
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.logic.build_reference_index import build_index

RENDER_LOCAL = os.path.join(PROJECT_ROOT, "render_local")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models", "piezas_vectores")


def run_full_indexing():
    """Scan all piece directories and build/update the unified index."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    processed = 0
    
    render_path = Path(RENDER_LOCAL)
    if not render_path.exists():
        print(f"❌ render_local/ not found at {RENDER_LOCAL}")
        return
    
    print(f"🔍 Scanning {RENDER_LOCAL} for piece directories...")
    
    # 1. Collect all directories to process
    dirs_to_process = []
    
    # Process ref_pieza subdirectories
    ref_base = render_path / "ref_pieza"
    if ref_base.exists():
        for piece_dir in sorted(ref_base.iterdir()):
            if piece_dir.is_dir() and (piece_dir / "images").exists():
                img_count = len(list((piece_dir / "images").glob("*.jpg")))
                if img_count > 0:
                    dirs_to_process.append(piece_dir)
    
    # Process images_mix directory
    mix_dir = render_path / "images_mix"
    if mix_dir.exists() and (mix_dir / "images").exists():
        img_count = len(list((mix_dir / "images").glob("*.jpg")))
        if img_count > 0:
            dirs_to_process.append(mix_dir)
    
    # Process legacy per-piece directories
    for piece_dir in sorted(render_path.iterdir()):
        if not piece_dir.is_dir(): continue
        if piece_dir.name in ('ref_pieza', 'images_mix', '.DS_Store'): continue
        if piece_dir.name.startswith('.'): continue
        
        if (piece_dir / "images").exists():
            img_count = len(list((piece_dir / "images").glob("*.jpg")))
            if img_count > 0:
                dirs_to_process.append(piece_dir)

    total_dirs = len(dirs_to_process)
    print(f"📦 Total directories to index: {total_dirs}")

    # 2. Process each directory
    for i, piece_dir in enumerate(dirs_to_process):
        print(f"PROGRESS: {i+1}/{total_dirs} | Processing {piece_dir.name}...")
        build_index(str(piece_dir), OUTPUT_DIR, unified=True)
        processed += 1
    
    print(f"\n✅ Indexación completada: {processed} directorios procesados")
    
    # Check final index
    index_path = os.path.join(OUTPUT_DIR, "lego.index")
    if os.path.exists(index_path):
        size_mb = os.path.getsize(index_path) / (1024 * 1024)
        print(f"📊 Índice FAISS: {index_path} ({size_mb:.1f} MB)")
    else:
        print("⚠️ No se generó el archivo de índice.")


if __name__ == "__main__":
    print(f"🚀 Iniciando indexación incremental completa...")
    run_full_indexing()
