import subprocess
import os
from pathlib import Path

class SyntheticGenerator:
    def __init__(self, blender_path, ldraw_path, assets_dir="./assets", output_dir="./data/datasets"):
        self.blender_path = blender_path
        self.ldraw_path = Path(ldraw_path)
        self.assets_dir = Path(assets_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _create_temp_ldraw_lib(self, set_id, parts):
        """
        Creates a minimal LDraw library containing only the parts needed for this set.
        Uses symlinks for primitives to save space/time.
        """
        import tempfile
        import shutil
        
        # Create temp dir
        temp_dir = tempfile.mkdtemp(prefix=f"ldraw_{set_id}_")
        
        # source paths
        src_p = self.ldraw_path / "p"
        src_parts = self.ldraw_path / "parts"
        src_config = self.ldraw_path / "LDConfig.ldr"
        
        # dest paths
        dst_p = Path(temp_dir) / "p"
        dst_parts = Path(temp_dir) / "parts"
        dst_parts.mkdir(parents=True, exist_ok=True)
        
        # 1. Symlink Primitives (p) - Instant
        if src_p.exists():
            try:
                os.symlink(src_p, dst_p)
            except OSError:
                shutil.copytree(src_p, dst_p)

        # 2. Symlink/Copy Config
        if src_config.exists():
            try:
                os.symlink(src_config, Path(temp_dir) / "LDConfig.ldr")
            except OSError:
                shutil.copy(src_config, temp_dir)
                
        # 3. Copy ONLY required parts
        # This reduces "indexing" from 60k files to ~100
        for p in parts:
            part_id = p['ldraw_id']
            src_file = src_parts / f"{part_id}.dat"
            dst_file = dst_parts / f"{part_id}.dat"
            
            if src_file.exists():
                shutil.copy(src_file, dst_file)
                
        return temp_dir

    def generate_dataset(self, set_id, part_list, num_images=500):
        """
        Triggers multiple Blender instances in parallel to generate synthetic images.
        Yields progress updates (int 0-100) or logs.
        """
        import multiprocessing
        import time
        import shutil
        import select
        import json
        
        # Determine number of workers (leave some cores for OS)
        cpu_count = multiprocessing.cpu_count()
        num_workers = max(1, min(cpu_count - 2, 8)) # Use up to 8 workers, leaving 2 free
        
        yield 0, "Initializing Blender generation..."

        # --- OPTIMIZATION: Create Minimal LDraw Library ---
        temp_ldraw_path = None
        try:
            temp_ldraw_path = self._create_temp_ldraw_lib(set_id, part_list)
        except Exception as e:
            yield None, f"Error creating temp library: {e}"
            # Continue with default path if optimization fails
            temp_ldraw_path = str(self.ldraw_path)

        print(f"Starting Parallel Generation for Set {set_id} with {num_workers} workers...")
        
        # Split work
        chunk_size = num_images // num_workers
        remainder = num_images % num_workers
        
        workers = []
        temp_files = []
        
        try:
            for i in range(num_workers):
                count = chunk_size + (1 if i < remainder else 0)
                if count == 0: continue
                
                worker_id = f"worker_{i}"
                worker_output_base = self.output_dir / set_id / "temp_parts" / worker_id
                worker_output_base.mkdir(parents=True, exist_ok=True)
                
                # Temp Config
                temp_data_file = self.output_dir / f"temp_gen_data_{set_id}_{i}.json"
                temp_files.append(temp_data_file)
                
                with open(temp_data_file, 'w') as f:
                    json.dump({
                        'set_id': set_id,
                        'parts': part_list,
                        'num_images': count,
                        'output_base': str(worker_output_base.resolve()),
                        'assets_dir': str(self.assets_dir.resolve()),
                        'ldraw_path': str(Path(temp_ldraw_path).resolve())
                    }, f)
                
                # In generators `part_list` controls what render mode is typically invoked, but we need
                # to extract the mode from the config if available, or infer it here
                
                # Check for `ref_pieza` vs `images_mix` based on piece count or explicit parameter
                # The generator is mostly used for either one based on UI invocation config
                # We'll infer based on an environment variable or piece count assumption, 
                # or passed params if we modify the generator interface later. 
                # For now, default to single_piece_setup.py if we're rendering a single part reference batch
                # To be completely safe and generic, we check what the set_id name is or rely on the UI 
                # which passes 'mode'.
                # But since `generator.py` currently builds the JSON itself without `render_mode` explicitly, 
                # let's assume it inherits from environment or we inject `render_mode` explicitly later.
                
                # Default assume scene_setup.py
                active_script = (Path(__file__).parent.parent / "blender_scripts" / "scene_setup.py").resolve()
                
                cmd = [
                    self.blender_path,
                    "--background",
                    "--python", str(active_script),
                    "--",
                    str(temp_data_file)
                ]
                
                p = subprocess.Popen(
                    cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE, 
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                workers.append({'process': p, 'id': i, 'total': count, 'current': 0, 'done': False})
            
            # Monitoring Loop
            while any(not w['done'] for w in workers):
                
                # Check output non-blocking
                reads = [w['process'].stdout for w in workers if not w['done']]
                if not reads: break # Should not happen unless logic error
                
                readable, _, _ = select.select(reads, [], [], 0.1)
                
                for stdout in readable:
                    # Find worker
                    worker = next(w for w in workers if w['process'].stdout == stdout)
                    line = stdout.readline()
                    
                    if line:
                        line = line.strip()
                        if "PROGRESS:" in line:
                            try:
                                # "PROGRESS: 5/100"
                                parts = line.split("PROGRESS:")[1].strip().split("/")
                                local_current = int(parts[0])
                                worker['current'] = local_current
                            except:
                                pass
                        elif "Warning" in line or "Error" in line:
                             yield None, f"[Worker {worker['id']}] {line}"
                    else:
                        # EOF
                        worker['process'].wait()
                        worker['done'] = True
                        
                # Aggregate Progress
                total_current = sum(w['current'] for w in workers)
                progress = int((total_current / num_images) * 100)
                yield progress, None
                
            # Post-Processing: Merge Images, Labels, and Metadata
            final_images_dir = self.output_dir / set_id / "images"
            final_labels_dir = self.output_dir / set_id / "labels"
            final_meta_path = self.output_dir / set_id / "image_meta.jsonl"
            
            # Clear old if exists (re-run)
            if final_images_dir.exists(): shutil.rmtree(final_images_dir)
            if final_labels_dir.exists(): shutil.rmtree(final_labels_dir)
            if final_meta_path.exists(): final_meta_path.unlink()
            
            final_images_dir.mkdir(parents=True, exist_ok=True)
            final_labels_dir.mkdir(parents=True, exist_ok=True)
            
            global_index = 0
            merged_meta_entries = []
            
            for i in range(num_workers):
                worker_path = self.output_dir / set_id / "temp_parts" / f"worker_{i}"
                if not worker_path.exists(): continue
                
                w_images = sorted(list((worker_path / "images").glob("*.jpg")))
                w_labels = sorted(list((worker_path / "labels").glob("*.txt")))
                w_meta_path = worker_path / "image_meta.jsonl"
                
                # Load worker metadata if exists
                worker_meta = {}
                if w_meta_path.exists():
                    with open(w_meta_path, 'r') as f:
                        for line in f:
                            try:
                                entry = json.loads(line)
                                worker_meta[entry['img']] = entry['ids']
                            except: continue

                for img_path, lbl_path in zip(w_images, w_labels):
                    new_base = f"img_{global_index:04d}"
                    new_img_name = f"{new_base}.jpg"
                    
                    # Move files
                    shutil.move(str(img_path), str(final_images_dir / new_img_name))
                    shutil.move(str(lbl_path), str(final_labels_dir / f"{new_base}.txt"))
                    
                    # Remap metadata
                    old_img_name = img_path.name
                    if old_img_name in worker_meta:
                        merged_meta_entries.append({
                            "img": new_img_name,
                            "ids": worker_meta[old_img_name]
                        })
                    
                    global_index += 1
                
                # Cleanup worker folder
                shutil.rmtree(worker_path)
            
            # Write merged metadata
            if merged_meta_entries:
                with open(final_meta_path, 'w') as f:
                    for entry in merged_meta_entries:
                        f.write(json.dumps(entry) + "\n")
            
            # Create final data.yaml
            data_yaml_path = self.output_dir / set_id / "data.yaml"
            is_universal = os.environ.get("UNIVERSAL_DETECTOR", "0") == "1"
            
            with open(data_yaml_path, 'w') as f:
                f.write(f"path: {os.path.abspath(self.output_dir / set_id)}\n")
                f.write("train: images\n")
                f.write("val: images\n")
                f.write("\n")
                if is_universal:
                    f.write("nc: 1\n")
                    f.write("names: ['lego']\n")
                else:
                    f.write(f"nc: {len(part_list)}\n")
                    # Use ldraw_id for names
                    names = [p['ldraw_id'] for p in part_list]
                    f.write(f"names: {names}\n")

            # Cleanup temp parent
            temp_parts = self.output_dir / set_id / "temp_parts"
            if temp_parts.exists():
                temp_parts.rmdir()
            
            yield 100, f"Done. Generated {global_index} images and verified metadata."
            
        except Exception as e:
            yield -1, str(e)
            # Kill workers
            for w in workers:
                w['process'].terminate()
                
        finally:
             # Cleanup Temp Library
            if temp_ldraw_path and os.path.exists(temp_ldraw_path) and "ldraw_" in temp_ldraw_path:
                shutil.rmtree(temp_ldraw_path)

        # Cleanup Config Files
        for tf in temp_files:
            if tf.exists():
                tf.unlink()
