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
    Fast version using direct geometric logic.
    """
    try:
        from src.logic.geometric_pose_analysis import get_pose_universe
        from src.blender_scripts.ldraw_resolver import LDrawResolver
        
        ldraw_path_base = os.path.join(PROJECT_ROOT, "assets", "ldraw")
        resolver = LDrawResolver(ldraw_path_base=ldraw_path_base)
        part_path = os.path.join(ldraw_path_base, "parts", f"{p_id}.dat")
        
        if not os.path.exists(part_path):
            return 0
            
        poses = get_pose_universe(part_path, resolver)
        return len(poses)
    except Exception as e:
        print(f"⚠️ Direct analysis failed for {p_id}. Error: {e}")
        return 0
