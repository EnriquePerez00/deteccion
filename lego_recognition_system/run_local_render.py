#!/usr/bin/env python3
"""
🚀 Local Render Engine v2.0 (Mac Pro M4 Optimized)
==================================================
Dual-mode rendering pipeline:
  - images_mix: Scattered pieces in 20×20cm zone, N = (X × 1500) / K formula
  - ref_pieza: Single piece centered, 300 images with 10° rotation across 360°

Optimized for Apple Silicon (METAL rendering) with CYCLES engine.
Zone: 20×20cm at 70cm height (iPhone 16 @ 24MP reference).
"""

import os
import sys
import json
import time
import shutil
import zipfile
import argparse
import subprocess
import concurrent.futures
import multiprocessing
import math
from datetime import datetime
from pathlib import Path

# --- CONFIGURATION ---
PROJECT_ROOT = Path(os.path.abspath(os.path.dirname(__file__)))
RENDER_LOCAL_DIR = PROJECT_ROOT / "render_local"

# Paths
LDRAW_PATH = PROJECT_ROOT / "assets" / "ldraw"
ADDON_PATH = PROJECT_ROOT / "src" / "blender_scripts"
SCENE_SETUP_PY = PROJECT_ROOT / "src" / "blender_scripts" / "scene_setup.py"
# Default path for Blender on macOS
BLENDER_PATH = "/Applications/Blender.app/Contents/MacOS/Blender"

# If blender is not in Applications, try to find it in PATH
if not os.path.exists(BLENDER_PATH):
    try:
        BLENDER_PATH = subprocess.check_output(["which", "blender"]).decode().strip()
    except:
        pass


def calculate_mix_params(num_unique_pieces, mix_ratio=0.75):
    """
    Calculate the optimal number of images for images_mix mode.
    
    Formula: N = (X × 1500) / K
    Where:
        X = number of unique pieces (excluding minifigs)
        K = mix count per image (50-60% of X)
        1500 = minimum appearances per class for robust YOLO training
    
    Returns: (N_images, K_per_image, pieces_per_image)
    """
    X = num_unique_pieces
    
    # Variety Logic: If set is small (< 10 types), use all types in every image (K = X)
    if X < 10:
        K = X
    else:
        K = max(1, int(X * mix_ratio))  # 75% of X by default
    
    # N Calculation with Minimum Enforcement (N >= 500) and Temporary Max Cap (1000)
    N = (X * 1500) // K if K > 0 else 1500
    N = min(1000, max(N, 500))
    
    pieces_per_image = 30  # Fixed to 30 as requested
    return N, K, pieces_per_image


def setup_structure(piece_id=None, mode='images_mix'):
    """Initializes the local render workspace or piece-specific subfolders."""
    if mode == 'images_mix':
        base = RENDER_LOCAL_DIR / "images_mix"
    elif mode == 'ref_pieza':
        base = RENDER_LOCAL_DIR / "ref_pieza" / str(piece_id) if piece_id else RENDER_LOCAL_DIR / "ref_pieza"
    else:
        base = RENDER_LOCAL_DIR / str(piece_id) if piece_id else RENDER_LOCAL_DIR
    
    subdirs = {
        'images': base / "images",
        'labels': base / "labels",
        'logs': base / "logs",
        'configs': base / "configs"
    }
    
    for d in subdirs.values():
        d.mkdir(parents=True, exist_ok=True)
    
    return subdirs, base


def run_render_worker(worker_id, piece_id, chunks_for_worker, render_mode='images_mix'):
    """Execution function for a single Blender process."""
    if not chunks_for_worker: return
    
    mode_dirs, base_dir = setup_structure(piece_id, mode=render_mode)
    
    worker_cfg = {
        'worker_id': str(worker_id),
        'set_id': piece_id,
        'render_mode': render_mode,
        'pieces_config': chunks_for_worker,
        'output_base': str(base_dir),
        'assets_dir': str(PROJECT_ROOT / 'assets'),
        'ldraw_path': str(LDRAW_PATH),
        'addon_path': str(ADDON_PATH)
    }
    
    # Add mode-specific config
    if render_mode == 'ref_pieza':
        worker_cfg['ref_num_images'] = chunks_for_worker[0].get('imgs', 300)
    elif render_mode == 'images_mix':
        worker_cfg['ref_num_images'] = chunks_for_worker[0].get('imgs', 250)
        worker_cfg['parts_per_image'] = chunks_for_worker[0].get('parts_per_image', 20)
    
    cfg_path = mode_dirs['configs'] / f'render_cfg_{worker_id}.json'
    with open(cfg_path, 'w') as f:
        json.dump(worker_cfg, f, indent=4)
    
    log_file = mode_dirs['logs'] / f'worker_{worker_id}.log'
    
    env = os.environ.copy()
    env['PYTHONPATH'] = str(PROJECT_ROOT)
    env['UNIVERSAL_DETECTOR'] = '1'
    
    print(f"  ↳ [{piece_id} | {render_mode} | Worker {worker_id}] Starting render subprocess...")
    
    cmd = [
        'caffeinate', '-i', BLENDER_PATH, '--background', '--python', str(SCENE_SETUP_PY),
        '--', str(cfg_path)
    ]
    
    with open(log_file, 'w') as f_out:
        f_out.write(f"--- WORKER {worker_id} START ---\n")
        f_out.write(f"TIME: {datetime.now().isoformat()}\n")
        f_out.write(f"CMD: {' '.join(cmd)}\n")
        f_out.write(f"ENV: PYTHONPATH={env.get('PYTHONPATH')}\n")
        f_out.write("-" * 40 + "\n\n")
        f_out.flush()
        
        try:
            result = subprocess.run(cmd, stdout=f_out, stderr=subprocess.STDOUT, env=env, check=False)
            f_out.write(f"\n" + "-" * 40 + "\n")
            f_out.write(f"COMPLETED with return code: {result.returncode}\n")
        except Exception as e:
            f_out.write(f"\nFATAL SUBPROCESS ERROR: {str(e)}\n")
    
    return worker_id


def filter_minifig_parts(parts):
    """Separate regular pieces from minifigure pieces."""
    regular = []
    minifigs = []
    for p in parts:
        if isinstance(p, dict):
            pid = p.get('part_id', p.get('ldraw_id', ''))
            cat = p.get('category', '')
        else:
            pid = str(p)
            cat = ''
        
        is_minifig = (
            pid.startswith('sw') or 
            pid.startswith('fig') or 
            'Minifig' in cat or
            'minifig' in cat.lower()
        )
        
        if is_minifig:
            minifigs.append(p)
        else:
            regular.append(p)
    
    return regular, minifigs


def main(target_parts, render_settings=None):
    start_time = time.time()
    
    render_mode = 'images_mix'  # Default
    if render_settings:
        render_mode = render_settings.get('render_mode', 'images_mix')
    
    # 1. Resolve parts and split minifigs
    from src.logic.resolve_minifig import MinifigResolver
    resolver = MinifigResolver(ldraw_path=LDRAW_PATH)
    
    regular_parts, minifig_parts = filter_minifig_parts(target_parts)
    
    print(f"📊 Parts breakdown: {len(regular_parts)} regular + {len(minifig_parts)} minifigs")
    
    # ═══════════════════════════════════════════════════════
    # MODE: ref_pieza — 300 images per piece, centered, 10° rotation
    # ═══════════════════════════════════════════════════════
    if render_mode == 'ref_pieza':
        print("\n📸 MODE: ref_pieza — Generating reference images per piece")
        all_parts = regular_parts + minifig_parts  # All parts get refs
        
        ref_imgs_per_piece = 300
        if render_settings and render_settings.get('ref_num_images'):
            ref_imgs_per_piece = render_settings['ref_num_images']
        
        total_imgs = len(all_parts) * ref_imgs_per_piece
        print(f"📊 Total Render Plan: {total_imgs} images ({len(all_parts)} pieces × {ref_imgs_per_piece} imgs)")
        
        num_cores = multiprocessing.cpu_count()
        max_workers = max(1, num_cores - 2)
        print(f"🚀 M4 Parallelism: {max_workers} workers")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for p_entry in all_parts:
                if isinstance(p_entry, dict):
                    p_id = p_entry.get('part_id', p_entry.get('ldraw_id', str(p_entry)))
                    color_id = p_entry.get('color_id', 15)
                    color_name = p_entry.get('color_name', 'White')
                else:
                    p_id = str(p_entry)
                    color_id = 15
                    color_name = 'White'
                
                piece_cfg = [{
                    'part': {'ldraw_id': p_id, 'color_id': int(color_id), 'color_name': color_name, 'name': p_id},
                    'tier': 'REF',
                    'imgs': ref_imgs_per_piece,
                    'engine': 'CYCLES',
                    'res': 384,  # Optimized for DINOv2 (Perfect-Fit)
                    'ref_num_images': ref_imgs_per_piece,
                    'is_minifig': p_entry.get('is_minifig', False),
                }]
                
                futures.append(executor.submit(
                    run_render_worker, "0", p_id, piece_cfg, render_mode='ref_pieza'
                ))
            
            # Monitor
            while any(f.running() for f in futures):
                curr_imgs = 0
                ref_base = RENDER_LOCAL_DIR / "ref_pieza"
                if ref_base.exists():
                    for p_dir in ref_base.iterdir():
                        if p_dir.is_dir():
                            img_dir = p_dir / "images"
                            if img_dir.exists():
                                curr_imgs += len(list(img_dir.glob("*.jpg")))
                
                pct = (curr_imgs / total_imgs) * 100 if total_imgs > 0 else 0
                print(f"📈 Progress: {min(total_imgs, curr_imgs)}/{total_imgs} ({min(100.0, pct):.1f}%)", flush=True)
                time.sleep(3)
        
    # ═══════════════════════════════════════════════════════
    # MODE: images_mix — Scatter N images with K pieces each
    # ═══════════════════════════════════════════════════════
    elif render_mode == 'images_mix':
        print("\n🎲 MODE: images_mix — Generating mixed scatter images")
        
        # Only regular parts in the mix (exclude minifigs)
        if not regular_parts:
            print("📭 No regular parts for images_mix. Skipping.")
        else:
            X = len(regular_parts)
            mix_ratio = 0.75  # 75% default
            if render_settings and render_settings.get('mix_ratio'):
                mix_ratio = render_settings['mix_ratio']
            
            N, K, pieces_per_image = calculate_mix_params(X, mix_ratio)
            
            if render_settings and render_settings.get('num_images'):
                N = render_settings['num_images']
            if render_settings and render_settings.get('parts_per_image'):
                pieces_per_image = render_settings['parts_per_image']
            
            print(f"📊 Formula: N = ({X} × 1500) / {K} = {N} images")
            print(f"   Mix: {K}/{X} piece types ({mix_ratio*100:.0f}%), {pieces_per_image} pcs/image")
            
            total_imgs = N
            
            num_cores = multiprocessing.cpu_count()
            max_workers = max(1, num_cores - 2)
            print(f"🚀 M4 Parallelism: {max_workers} workers")
            
            # Build piece configs for each image batch
            # Each worker gets a chunk of N/max_workers images to render
            chunk_size = max(1, N // max_workers)
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                
                for w_idx in range(max_workers):
                    start_img = w_idx * chunk_size
                    end_img = min(N, (w_idx + 1) * chunk_size)
                    if w_idx == max_workers - 1:
                        end_img = N  # Last worker gets remainder
                    
                    n_imgs = end_img - start_img
                    if n_imgs <= 0:
                        continue
                    
                    # For each image, select random 75% of available piece types
                    # The actual piece selection happens in scene_setup.py
                    # We pass all regular parts and let Blender choose per-image
                    pieces_config = []
                    for p_entry in regular_parts:
                        if isinstance(p_entry, dict):
                            p_id = p_entry.get('part_id', p_entry.get('ldraw_id', str(p_entry)))
                            color_id = p_entry.get('color_id', 15)
                            color_name = p_entry.get('color_name', 'White')
                        else:
                            p_id = str(p_entry)
                            color_id = 15
                            color_name = 'White'
                        
                        pieces_config.append({
                            'part': {'ldraw_id': p_id, 'color_id': int(color_id), 'color_name': color_name, 'name': p_id},
                            'tier': 'MIX',
                            'imgs': n_imgs,
                            'engine': 'CYCLES',
                            'res': 5656, # Calibrated for iPhone 16 24MP (4:3 aspect handled in scene_setup)
                            'parts_per_image': pieces_per_image,
                        })
                    
                    futures.append(executor.submit(
                        run_render_worker, str(w_idx), "mix", pieces_config, render_mode='images_mix'
                    ))
                
                # Monitor
                while any(f.running() for f in futures):
                    mix_dir = RENDER_LOCAL_DIR / "images_mix" / "images"
                    curr_imgs = len(list(mix_dir.glob("*.jpg"))) if mix_dir.exists() else 0
                    pct = (curr_imgs / total_imgs) * 100 if total_imgs > 0 else 0
                    print(f"📈 Progress: {min(total_imgs, curr_imgs)}/{total_imgs} ({min(100.0, pct):.1f}%)", flush=True)
                    time.sleep(3)
    
    # ═══════════════════════════════════════════════════════
    # MODE: both — Run ref_pieza first, then images_mix
    # ═══════════════════════════════════════════════════════
    elif render_mode == 'both':
        print("\n🔄 MODE: both — Running ref_pieza then images_mix")
        # Recursive calls with each mode
        ref_settings = dict(render_settings or {})
        ref_settings['render_mode'] = 'ref_pieza'
        main(target_parts, render_settings=ref_settings)
        
        mix_settings = dict(render_settings or {})
        mix_settings['render_mode'] = 'images_mix'
        main(target_parts, render_settings=mix_settings)
        return  # Skip post-processing (already done in each sub-call)
    
    # 4. Post-Processing
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n✅ All renders completed in {duration/60:.1f} min.")
    
    # 4a. Similarity Filter (remove near-duplicate images, cosine > 0.98)
    print("🔍 Running similarity filter on rendered images...")
    total_deleted = 0
    try:
        from src.logic.feature_extractor import FeatureExtractor
        from PIL import Image
        import numpy as np
        
        extractor = FeatureExtractor()
        THRESHOLD = 0.98
        
        # Determine directories to filter based on mode
        filter_dirs = []
        if render_mode == 'ref_pieza':
            ref_base = RENDER_LOCAL_DIR / "ref_pieza"
            if ref_base.exists():
                filter_dirs = [d for d in ref_base.iterdir() if d.is_dir() and not d.name.startswith('.')]
        elif render_mode == 'images_mix':
            mix_dir = RENDER_LOCAL_DIR / "images_mix"
            if mix_dir.exists():
                filter_dirs = [mix_dir]
        
        for piece_dir in filter_dirs:
            images_dir = piece_dir / "images"
            labels_dir = piece_dir / "labels"
            if not images_dir.exists(): continue
            
            img_paths = sorted(list(images_dir.glob("*.jpg")))
            if len(img_paths) < 2: continue
            
            last_embedding = None
            deleted_in_piece = 0
            
            for img_path in img_paths:
                try:
                    img = Image.open(img_path).convert('RGB')
                    embedding = extractor.get_embedding(img)
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm
                    
                    if last_embedding is not None:
                        similarity = float(np.dot(embedding, last_embedding))
                        if similarity > THRESHOLD:
                            img_path.unlink()
                            label_path = labels_dir / img_path.name.replace('.jpg', '.txt')
                            if label_path.exists():
                                label_path.unlink()
                            deleted_in_piece += 1
                            continue
                    
                    last_embedding = embedding
                except Exception as e:
                    print(f"  ⚠️ Filter error on {img_path.name}: {e}")
            
            if deleted_in_piece > 0:
                print(f"  🗑️ {piece_dir.name}: removed {deleted_in_piece} near-duplicates")
                total_deleted += deleted_in_piece
    except ImportError:
        print("  ⚠️ FeatureExtractor not available. Skipping similarity filter.")
    
    print(f"✅ Filter complete. Removed {total_deleted} duplicates total.")
    
    # 4b. Generate data.yaml and manifest
    print("📝 Generating data.yaml files...")
    piece_manifest = []
    
    # Process ref_pieza directories
    ref_base = RENDER_LOCAL_DIR / "ref_pieza"
    if ref_base.exists():
        for piece_dir in sorted(ref_base.iterdir()):
            if not piece_dir.is_dir() or piece_dir.name.startswith('.'): continue
            images_dir = piece_dir / "images"
            labels_dir = piece_dir / "labels"
            if not images_dir.exists(): continue
            
            img_count = len(list(images_dir.glob("*.jpg")))
            lbl_count = len(list(labels_dir.glob("*.txt"))) if labels_dir.exists() else 0
            if img_count == 0: continue
            
            data_yaml_path = piece_dir / "data.yaml"
            with open(data_yaml_path, 'w') as f:
                f.write(f"path: .\n")
                f.write(f"train: images\n")
                f.write(f"val: images\n")
                f.write(f"nc: 1\n")
                f.write(f"names: ['lego']\n")
            
            piece_manifest.append({
                "piece_id": piece_dir.name,
                "mode": "ref_pieza",
                "images": img_count,
                "labels": lbl_count,
                "data_yaml": f"ref_pieza/{piece_dir.name}/data.yaml",
            })
            print(f"  ✅ ref_pieza/{piece_dir.name}: {img_count} imgs → data.yaml")
    
    # Process images_mix directory
    mix_dir = RENDER_LOCAL_DIR / "images_mix"
    if mix_dir.exists():
        images_dir = mix_dir / "images"
        labels_dir = mix_dir / "labels"
        if images_dir.exists():
            img_count = len(list(images_dir.glob("*.jpg")))
            lbl_count = len(list(labels_dir.glob("*.txt"))) if labels_dir.exists() else 0
            
            if img_count > 0:
                data_yaml_path = mix_dir / "data.yaml"
                with open(data_yaml_path, 'w') as f:
                    f.write(f"path: .\n")
                    f.write(f"train: images\n")
                    f.write(f"val: images\n")
                    f.write(f"nc: 1\n")
                    f.write(f"names: ['lego']\n")
                
                piece_manifest.append({
                    "piece_id": "images_mix",
                    "mode": "images_mix",
                    "images": img_count,
                    "labels": lbl_count,
                    "data_yaml": "images_mix/data.yaml",
                })
                print(f"  ✅ images_mix: {img_count} imgs → data.yaml")
    
    # Also process legacy per-piece directories (backward compatibility)
    for piece_dir in sorted(RENDER_LOCAL_DIR.iterdir()):
        if not piece_dir.is_dir(): continue
        if piece_dir.name in ('images_mix', 'ref_pieza', '.DS_Store'): continue
        if piece_dir.name.startswith('.'): continue
        
        images_dir = piece_dir / "images"
        labels_dir = piece_dir / "labels"
        if not images_dir.exists(): continue
        
        img_count = len(list(images_dir.glob("*.jpg")))
        lbl_count = len(list(labels_dir.glob("*.txt"))) if labels_dir.exists() else 0
        if img_count == 0: continue
        
        data_yaml_path = piece_dir / "data.yaml"
        if not data_yaml_path.exists():
            with open(data_yaml_path, 'w') as f:
                f.write(f"path: .\n")
                f.write(f"train: images\n")
                f.write(f"val: images\n")
                f.write(f"nc: 1\n")
                f.write(f"names: ['lego']\n")
        
        piece_manifest.append({
            "piece_id": piece_dir.name,
            "mode": "legacy",
            "images": img_count,
            "labels": lbl_count,
            "data_yaml": f"{piece_dir.name}/data.yaml",
        })
    
    # 4c. Global manifest
    manifest_path = RENDER_LOCAL_DIR / "dataset_manifest.json"
    total_images = sum(p['images'] for p in piece_manifest)
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "duration_minutes": round(duration / 60, 1),
        "render_mode": render_mode,
        "total_pieces": len(piece_manifest),
        "total_images": total_images,
        "duplicates_removed": total_deleted,
        "pieces": piece_manifest,
    }
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"📋 Manifest: {manifest_path}")
    
    # 5. ZIP for Lightning AI / Kaggle
    if not render_settings.get('skip_zip', False):
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        zip_name = f"lightning_dataset_{ts}.zip"
        zip_path = PROJECT_ROOT / zip_name
        
        print(f"📦 Creating training package: {zip_name}...")
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # A. Dataset (render_local/ contents)
            for file in RENDER_LOCAL_DIR.rglob("*"):
                if file.is_file() and not file.name.startswith('.'):
                    arcname = Path("render_local") / file.relative_to(RENDER_LOCAL_DIR)
                    zipf.write(file, arcname=arcname)
            
            # B. Source code needed for training
            src_dir = PROJECT_ROOT / "src"
            for py_file in src_dir.rglob("*.py"):
                if '__pycache__' in str(py_file): continue
                arcname = Path("src") / py_file.relative_to(src_dir)
                zipf.write(py_file, arcname=arcname)
            
            # C. Config
            config_path = PROJECT_ROOT / "config_train.json"
            if config_path.exists():
                zipf.write(config_path, arcname="config_train.json")
            
            # D. Credentials for Drive sync (if available)
            for cred_file in ["credentials.json", "token_1973.pickle"]:
                cred_path = PROJECT_ROOT / cred_file
                if cred_path.exists():
                    zipf.write(cred_path, arcname=cred_file)
            
            # E. Existing models (Incremental training weights ONLY)
            models_dir = PROJECT_ROOT / "models"
            if models_dir.exists():
                # 1. Skip vector indices (only needed for inference/indexing locally)
                # 2. Only include the LATEST .pt from yolo_model
                yolo_dir = models_dir / "yolo_model"
                if yolo_dir.exists():
                    pt_files = list(yolo_dir.glob("*.pt"))
                    if pt_files:
                        latest_pt = max(pt_files, key=os.path.getmtime)
                        arcname = Path("models") / "yolo_model" / latest_pt.name
                        zipf.write(latest_pt, arcname=arcname)
                        print(f"  🧠 Including latest weights: {latest_pt.name}")
                        
            # F. Notebooks (include generated notebooks)
            notebooks_dir = PROJECT_ROOT / "notebooks"
            if notebooks_dir.exists():
                for nb_file in notebooks_dir.rglob("*.ipynb"):
                    arcname = Path("notebooks") / nb_file.relative_to(notebooks_dir)
                    zipf.write(nb_file, arcname=arcname)
                    print(f"  📎 Including notebook: {nb_file.name}")
        
        zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
        print(f"🌟 Training package ready: {zip_name} ({zip_size_mb:.1f} MB)")
    else:
        print("⏭️ Skipping ZIP generation as requested.")
    print(f"📊 Contents: {total_images} images across {len(piece_manifest)} pieces")
    print(f"⏱️ Total pipeline time: {duration/60:.1f} min")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-zip", action="store_true", help="Skip ZIP generation")
    args = parser.parse_args()

    # Get pieces from config_train.json if exists, else defaults
    parts = []
    render_settings = {'skip_zip': args.no_zip}
    config_path = PROJECT_ROOT / "config_train.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            data = json.load(f)
            parts = data.get('target_parts', [])
            render_settings = data.get('render_settings', {})
            
    if not parts:
        parts = ["3022", "32054", "3795", "4073"]
        
    main(parts, render_settings=render_settings)
