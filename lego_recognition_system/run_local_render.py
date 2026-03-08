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
import threading
from datetime import datetime
from pathlib import Path
from src.logic.lego_tiering_logic import get_part_tier

# Dynamic worker manager (psutil-based resource monitoring)
try:
    from src.utils.blender_worker_manager import BlenderWorkerManager
    DYNAMIC_WORKERS = True
except ImportError:
    DYNAMIC_WORKERS = False

# Automatic Indexing Hook
from run_incremental_indexing import run_full_indexing

# --- CONFIGURATION ---
PROJECT_ROOT = Path(os.path.abspath(os.path.dirname(__file__)))
RENDER_LOCAL_DIR = PROJECT_ROOT / "render_local"
MANAGER_LOG = str(RENDER_LOCAL_DIR / "logs" / "worker_manager.log")

# Paths
ASSETS_DIR = PROJECT_ROOT / "assets"
LDRAW_PATH = ASSETS_DIR / "ldraw"
ADDON_PATH = PROJECT_ROOT / "src" / "blender_scripts"
SCENE_SETUP_PY = ADDON_PATH / "scene_setup.py"
SINGLE_PIECE_SETUP_PY = ADDON_PATH / "single_piece_setup.py"
# Default path for Blender on macOS
BLENDER_PATH = "/Applications/Blender.app/Contents/MacOS/Blender"

# If blender is not in Applications, try to find it in PATH
if not os.path.exists(BLENDER_PATH):
    try:
        BLENDER_PATH = subprocess.check_output(["which", "blender"]).decode().strip()
    except:
        pass


from src.utils.render_tiers import calculate_mix_params


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
    if chunks_for_worker:
        is_square = chunks_for_worker[0].get('square', False)
        
        if render_mode == 'ref_pieza':
            res_x = chunks_for_worker[0].get('ref_res', 640)
            worker_cfg['resolution_x'] = res_x
            worker_cfg['resolution_y'] = res_x
            worker_cfg['ref_num_images'] = chunks_for_worker[0].get('imgs', 300)
            # Inject numbering metadata at root level for Blender script simplicity
            if 'numbering_meta' in chunks_for_worker[0]:
                worker_cfg['numbering_meta'] = chunks_for_worker[0]['numbering_meta']
        else:
            res_x = chunks_for_worker[0].get('res', 1920)
            worker_cfg['resolution_x'] = res_x
            worker_cfg['resolution_y'] = res_x if is_square else int(res_x * 0.75)
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
    
    active_script = SINGLE_PIECE_SETUP_PY if render_mode == 'ref_pieza' else SCENE_SETUP_PY

    cmd = [
        'caffeinate', '-i', BLENDER_PATH, '--background', '--python', str(active_script),
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


def start_progress_reporter(total_imgs, render_mode, piece_dir_keys=None):
    """
    Starts a background thread that polls the disk for new images
    and prints progress lines for the UI to parse.
    piece_dir_keys: optional list of directory names to filter (e.g. ['4073_6', '4073_47'])
    """
    stop_event = threading.Event()

    def report_loop():
        start_time = time.time()
        while not stop_event.is_set():
            curr_imgs = 0
            if render_mode == 'ref_pieza':
                ref_base = RENDER_LOCAL_DIR / "ref_pieza"
                if ref_base.exists():
                    for p_dir in ref_base.iterdir():
                        if not p_dir.is_dir(): continue
                        # If we have specific dir keys, only count those
                        if piece_dir_keys and p_dir.name not in piece_dir_keys:
                            continue
                        img_dir = p_dir / "images"
                        if img_dir.exists():
                            curr_imgs += len(list(img_dir.glob("*.jpg")))
            elif render_mode == 'images_mix':
                mix_dir = RENDER_LOCAL_DIR / "images_mix" / "images"
                if mix_dir.exists():
                    curr_imgs = len(list(mix_dir.glob("*.jpg")))
            
            pct = (curr_imgs / total_imgs) * 100 if total_imgs > 0 else 0
            print(f"📈 Progress: {min(total_imgs, curr_imgs)}/{total_imgs} ({min(100.0, pct):.1f}%)", flush=True)
            
            # Wait 5 seconds or until stopped
            if stop_event.wait(timeout=5):
                break
    
    reporter_thread = threading.Thread(target=report_loop, daemon=True)
    reporter_thread.start()
    return stop_event, reporter_thread


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
    num_cores = multiprocessing.cpu_count()
    render_mode = render_settings.get('render_mode', 'images_mix') if render_settings else 'images_mix'
    
    # 1. Resolve parts and split minifigs
    from src.logic.resolve_minifig import MinifigResolver
    from src.logic.part_resolver import resolve_piece
    
    print(f"🔍 Enriching metadata for {len(target_parts)} pieces...")
    enriched_parts = []
    for p in target_parts:
        if isinstance(p, dict):
            pid = p.get('part_id', p.get('ldraw_id'))
            if pid:
                meta = resolve_piece(str(pid))
                meta.update(p) # Keep original overrides
                enriched_parts.append(meta)
            else:
                enriched_parts.append(p)
        else:
            enriched_parts.append(resolve_piece(str(p)))
    target_parts = enriched_parts

    resolver = MinifigResolver(ldraw_path=LDRAW_PATH)
    
    regular_parts, minifig_parts = filter_minifig_parts(target_parts)
    
    print(f"📊 Parts breakdown: {len(regular_parts)} regular + {len(minifig_parts)} minifigs")
    
    # ═══════════════════════════════════════════════════════
    # MODE: ref_pieza — Geometric Stable Face Analysis + 24 Rotations
    # ═══════════════════════════════════════════════════════
    if render_mode == 'ref_pieza':
        print("\n📸 MODE: ref_pieza — Geometric Stability Analysis Pass")
        all_parts = regular_parts + minifig_parts
        
        # 1. 🧪 GEOMETRIC ANALYSIS PHASE (Simplified via Part Resolver)
        print(f"🔍 Using pre-calculated geometric stability and symmetry for {len(all_parts)} pieces...")
        
        part_orientations = {}
        for p_entry in all_parts:
            if isinstance(p_entry, dict):
                p_id = p_entry.get('part_id', p_entry.get('ldraw_id', str(p_entry)))
                dir_key = f"{p_id}"
                part_orientations[dir_key] = p_entry.get('geometric_poses', [])

        # 2. 🚀 RENDERING JOB GENERATION
        ref_jobs = []
        total_imgs = 0
        
        for i, p_entry in enumerate(all_parts):
            if isinstance(p_entry, dict):
                p_id = p_entry.get('part_id', p_entry.get('ldraw_id', str(p_entry)))
                color_id = p_entry.get('color_id', 15)
                color_name = p_entry.get('color_name', 'White')
                p_cat = p_entry.get('category', '')
            else:
                p_id = str(p_entry)
                color_id = 15
                color_name = 'White'
                p_cat = ''
            
            dir_key = f"{p_id}"
            oris = p_entry.get('geometric_poses', [])
            
            # Determine Tier for this piece
            p_ldraw_path = str(LDRAW_PATH / f"{p_id}.dat")
            tier = get_part_tier(p_id, p_cat, p_ldraw_path)
            
            # Tier-based Rotation Settings
            tier_rot_config = {
                "TIER1": {"angles": 12, "elevations": 1},
                "TIER2": {"angles": 36, "elevations": 1},
                "TIER3": {"angles": 10, "elevations": 10},
                "TIER4": {"angles": 20, "elevations": 10},
            }
            config = tier_rot_config.get(tier, tier_rot_config["TIER2"])
            total_angles_per_face = config["angles"] * config["elevations"]

            if not oris:
                # Absolute fallback (no matrix)
                oris = [{"matrix": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], "symmetry": 1}]

            # Rotations per stable face based on Tier and Symmetry
            for f_idx, pose_data in enumerate(oris):
                symmetry = pose_data.get('symmetry', 1)
                optimized_imgs = max(1, int(total_angles_per_face / symmetry))
                
                piece_cfg = [{
                    'part': {'ldraw_id': p_id, 'color_id': color_id, 'name': p_id, 'color_name': color_name},
                    'tier': tier,
                    'imgs': optimized_imgs,
                    'engine': 'CYCLES',
                    'res': render_settings.get('res', 1920) if render_settings else 1920,
                    'ref_res': render_settings.get('ref_res', 640) if render_settings else 640,
                    'square': True,
                    'stable_face_idx': f_idx,
                    'stable_matrix': pose_data.get('matrix'),
                    'symmetry_order': symmetry,
                    'ref_num_images': optimized_imgs,
                    'offset_idx': total_imgs,
                    'tier_config': config
                }]
                
                # Inject numbering metadata into each part's config for ref_pieza mode
                # Inject numbering metadata into each part's config for ref_pieza mode
                for pc in piece_cfg:
                    pc['numbering_meta'] = {
                        'offset_idx': total_imgs,
                        'ref_num_images': optimized_imgs
                    }
                
                ref_jobs.append((f"{i}_{f_idx}", dir_key, piece_cfg, 'ref_pieza'))
                total_imgs += optimized_imgs
            
            print(f"✅ {p_id} ({tier}): {len(oris)} stable faces. Optimized renders: {sum(max(1, int(total_angles_per_face / p.get('symmetry', 1))) for p in oris)}")

        print(f"📊 Total Render Plan: {total_imgs} images distributed across {len(ref_jobs)} specialized jobs.")
        
        # Collect unique directory keys for progress filtering
        active_dir_keys = list(set(job[1] for job in ref_jobs))
        (RENDER_LOCAL_DIR / "logs").mkdir(parents=True, exist_ok=True)
        stop_reporter, reporter_thread = start_progress_reporter(total_imgs, 'ref_pieza', piece_dir_keys=active_dir_keys)
        
        try:
            # Default to cores (multi-process scale)
            max_workers = num_cores
            # Cap to a safe limit to prevent RAM panic causing aggressive throttle down
            # M4 Pro has 12 cores, 8 workers is a conservative but stable sweet spot for UI responsiveness.
            max_workers = min(max_workers, 8)
            
            # Support manual override or 'low_impact' mode
            if render_settings:
                if render_settings.get('low_impact'):
                    max_workers = max(2, num_cores // 2)
                if render_settings.get('max_workers'):
                    max_workers = render_settings['max_workers']
            
            if DYNAMIC_WORKERS:
                manager = BlenderWorkerManager(max_workers=max_workers, log_path=MANAGER_LOG, verbose=True, low_priority=True)
                manager.run_with_dynamic_workers(jobs=ref_jobs, worker_fn=run_render_worker)
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(run_render_worker, *job) for job in ref_jobs]
                    concurrent.futures.wait(futures)
        finally:
            stop_reporter.set()
            reporter_thread.join(timeout=2)
        
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
            
            # M4 Pro Dynamic Scaling for Mix (initial cap — manager will adjust at runtime):
            mix_res = render_settings.get('res', 1920) if render_settings else 1920
            
            low_impact = render_settings.get('low_impact', False) if render_settings else False
            
            if mix_res <= 640:
                # CPU-bounded, but still keep headroom for macOS
                max_workers = num_cores - (6 if low_impact else 4)
                print(f"🚀 M4 Parallelism (Low-Res Mix): {max_workers} clusters active (extra headroom reserved).")
            else:
                # GPU-bounded, needs more breathing room for WindowServer/UI
                max_workers = max(1, num_cores // (4 if low_impact else 3))
                print(f"🚀 M4 Standard Parallelism (High-Res Mix): {max_workers} clusters active.")
            
            # Cap the baseline worker count to prevent CPU/RAM panic oscillations
            # Upgraded for M4 Pro: 8 workers for stable parallel mix jobs while leaving margin.
            max_workers = min(max_workers, 8)
            
            if render_settings and render_settings.get('max_workers'):
                max_workers = render_settings['max_workers']
            
            max_workers = max(1, max_workers)

            # Build piece configs for each image batch
            # Each worker gets a chunk of N/max_workers images to render
            chunk_size = max(1, N // max_workers)

            mix_jobs = []
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

                    tier = p_entry.get('tier', 'TIER2') if isinstance(p_entry, dict) else 'TIER2'
                    
                    # YOLO Frequency Multiplier
                    freq_map = {"TIER1": 0.5, "TIER2": 1.0, "TIER3": 2.5, "TIER4": 4.0}
                    multiplier = freq_map.get(tier, 1.0)

                    pieces_config.append({
                        'part': {'ldraw_id': p_id, 'color_id': int(color_id), 'color_name': color_name, 'name': p_id},
                        'tier': tier,
                        'imgs': n_imgs,
                        'engine': 'CYCLES',
                        'res': mix_res,
                        'square': render_settings.get('square', False) if render_settings else True,
                        'parts_per_image': pieces_per_image,
                        'different_pieces': K,  # Tiered variety (D)
                        'frequency_multiplier': multiplier
                    })

                mix_jobs.append((str(w_idx), "mix", pieces_config, 'images_mix'))

            # Progress tracking
            mix_img_dir = RENDER_LOCAL_DIR / "images_mix" / "images"
            _last_mix_progress = [time.time()]

            def _mix_progress(done: int, total_jobs: int) -> None:
                now = time.time()
                if now - _last_mix_progress[0] >= 3:
                    curr_imgs = len(list(mix_img_dir.glob("*.jpg"))) if mix_img_dir.exists() else 0
                    pct = (curr_imgs / total_imgs) * 100 if total_imgs > 0 else 0
                    print(f"Progress: {min(total_imgs, curr_imgs)}/{total_imgs} ({min(100.0, pct):.1f}%)", flush=True)
                    _last_mix_progress[0] = now

            # Launch via dynamic manager with real-time reporter
            (RENDER_LOCAL_DIR / "logs").mkdir(parents=True, exist_ok=True)
            
            stop_reporter, reporter_thread = start_progress_reporter(total_imgs, 'images_mix')
            
            try:
                if DYNAMIC_WORKERS:
                    manager = BlenderWorkerManager(
                        max_workers=max_workers,
                        log_path=MANAGER_LOG,
                        verbose=True,
                        low_priority=True,
                    )
                    manager.run_with_dynamic_workers(
                        jobs=mix_jobs,
                        worker_fn=run_render_worker
                    )
                else:
                    print("⚠️  psutil not available — using static worker pool")
                    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = [executor.submit(run_render_worker, *job) for job in mix_jobs]
                        concurrent.futures.wait(futures)
            finally:
                stop_reporter.set()
                reporter_thread.join(timeout=2)
    
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
        if render_mode == 'images_mix':
            mix_dir = RENDER_LOCAL_DIR / "images_mix"
            if mix_dir.exists():
                filter_dirs = [mix_dir]
        else:
            print("  ⏭️ Skipping similarity filter for ref_pieza to preserve identical views/colors.")
        
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
    
    # 4d. Tiling Strategy for Training
    tiling_cfg = render_settings.get('tiling', {})
    if tiling_cfg.get('enabled', False):
        print("\n🧩 Tiling Strategy: Fragmenting images for better YOLO accuracy...")
        tiling_size = tiling_cfg.get('size', 640)
        tiling_overlap = tiling_cfg.get('overlap', 0.25)
        
        tiled_dir = RENDER_LOCAL_DIR / "images_mix_tiled"
        try:
            from src.utils.dataset_tiler import process_dataset
            process_dataset(
                input_dir=str(mix_dir.parent), 
                output_dir=str(tiled_dir),
                slice_size=tiling_size,
                overlap_ratio=tiling_overlap
            )
            
            # Create data.yaml for tiled set
            with open(tiled_dir / "data.yaml", 'w') as f:
                f.write(f"path: .\ntrain: images\nval: images\nnc: 1\nnames: ['lego']\n")
                
            piece_manifest.append({
                "piece_id": "images_mix_tiled",
                "mode": "tiled",
                "images": len(list((tiled_dir / "images").glob("*.jpg"))),
                "data_yaml": "images_mix_tiled/data.yaml",
            })
        except Exception as e:
            print(f"  ❌ Tiling failed: {e}")

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
                    # EXCLUSION: Skip loose pieces (ref_pieza) for training ZIP
                    if "ref_pieza" in file.parts or "ref_pieza" in str(file):
                        continue
                    
                    # OPTIONAL: If tiling is enabled, we could skip original images_mix to save space,
                    # but for now we include both for comparison.
                        
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

    # 6. Automatic Vector Indexing
    if render_mode in ('ref_pieza', 'both') and not render_settings.get('skip_indexing', False):
        print("\n🧠 PHASE 2: Automatic Vector Indexing...")
        try:
            run_full_indexing()
            print("✅ Vectors updated successfully.")
        except Exception as e:
            print(f"❌ Indexing failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-zip", action="store_true", help="Skip ZIP generation")
    args = parser.parse_args()

    # Get pieces from config_train.json if exists, else defaults
    parts = []
    render_settings = {}
    config_path = PROJECT_ROOT / "config_train.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            data = json.load(f)
            parts = data.get('target_parts', [])
            render_settings = data.get('render_settings', {})
    
    # Preserve CLI flag (not overwritten by config)
    render_settings['skip_zip'] = args.no_zip
            
    if not parts:
        parts = ["3022", "32054", "3795", "4073"]
        
    main(parts, render_settings=render_settings)
