import json
import os
import subprocess

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RENDER_LOCAL_DIR = os.path.join(PROJECT_ROOT, "render_local")
BLENDER_PATH = "/Applications/Blender.app/Contents/MacOS/Blender"
SCENE_SETUP_PY = os.path.join(PROJECT_ROOT, "src", "blender_scripts", "scene_setup.py")

def get_stable_faces_for_piece(p_id, color_id=15, color_name="White"):
    """
    Returns the number of stable faces, reading from cache if available,
    or forcing an analysis run.
    """
    dir_key = f"{str(p_id)}_{color_id}"
    out_base = os.path.join(RENDER_LOCAL_DIR, "ref_pieza", dir_key)
    meta_dir = os.path.join(out_base, "meta")
    res_file = os.path.join(meta_dir, "analysis_cfg.result")
    
    if os.path.exists(res_file):
        try:
            with open(res_file, 'r') as f:
                data = json.load(f)
                return len(data.get('orientations', []))
        except: pass
        
    # Run analysis
    config_path = os.path.join(meta_dir, "analysis_cfg.json")
    os.makedirs(meta_dir, exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump({
            'parts': [{'ldraw_id': str(p_id), 'color_id': color_id, 'color_name': color_name}],
            'render_mode': 'ref_pieza',
            'ref_num_images': 1,
            'is_analyze_only': True,
            'output_base': str(out_base),
            'assets_dir': os.path.join(PROJECT_ROOT, "assets"),
            'ldraw_path': os.path.join(PROJECT_ROOT, "assets", "ldraw"),
            'addon_path': os.path.join(PROJECT_ROOT, "src", "blender_scripts")
        }, f)
        
    cmd = [BLENDER_PATH, "--background", "--python", str(SCENE_SETUP_PY), "--", str(config_path)]
    try:
        # Increase timeout slightly and capture output for potential log check
        res = subprocess.run(cmd, check=False, timeout=90, capture_output=True, text=True)
        if os.path.exists(res_file):
            with open(res_file, 'r') as f:
                data = json.load(f)
                return len(data.get('orientations', []))
        else:
            print(f"⚠️ Blender analysis failed for {p_id}. Output:\n{res.stdout[-500:]}")
    except Exception as e:
        print(f"Error analytical blender: {e}")
        
    return 0

