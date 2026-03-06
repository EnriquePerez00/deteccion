
import os
import sys
import json
import hashlib
import zipfile
import shutil
from pathlib import Path

# Determine script directory for relative path resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# File tracking
SYNC_STATE_FILE = os.path.join(SCRIPT_DIR, '.sync_state.json')
CONFIG_FILE = os.path.join(SCRIPT_DIR, 'config_train.json')

# Directories to sync
SYNC_DIRS = [os.path.join(SCRIPT_DIR, d) for d in ['src', 'models/piezas_vectores', 'models/yolo_model']]
SYNC_FILES = [os.path.join(SCRIPT_DIR, f) for f in [
    'requirements.txt', 'config_train.json', 'generate_notebooks.py',
    'download_assets.py', 'assets/backgrounds/background.jpg', 'cycles_kernels.zip',
    'optix_cache.zip', 'credentials.json', 'token_1973.pickle'
]]

def calculate_file_hash(filepath):
    """Calculates MD5 hash of a file."""
    # Ensure filepath is absolute if it's one of the SYNC_FILES
    if not os.path.isabs(filepath):
        filepath = os.path.join(SCRIPT_DIR, filepath)
    
    if not os.path.exists(filepath):
        return None
        
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def get_current_state():
    """Scans project and returns a dict of {absolute_path: hash}."""
    state = {}
    
    # Files
    for filepath in SYNC_FILES:
        if os.path.exists(filepath):
            h = calculate_file_hash(filepath)
            if h: state[filepath] = h
    
    # Add generated notebooks (Recursive search in notebooks/ directory)
    nb_base = os.path.join(SCRIPT_DIR, 'notebooks')
    if os.path.exists(nb_base):
        for root, _, files in os.walk(nb_base):
            for file in files:
                if file.endswith('.ipynb'):
                    is_tracked = any(file.startswith(p) for p in ['Colab_', 'Kaggle_', 'LightningAI_', 'train_'])
                    if is_tracked:
                        filepath = os.path.join(root, file)
                        h = calculate_file_hash(filepath)
                        if h: state[filepath] = h
    
    # Also check SCRIPT_DIR for legacy/root notebooks
    for file in os.listdir(SCRIPT_DIR):
        if file.endswith('.ipynb'):
            if any(file.startswith(p) for p in ['Colab_', 'Kaggle_', 'LightningAI_', 'train_']):
                filepath = os.path.join(SCRIPT_DIR, file)
                if filepath not in state: # Don't duplicate if already found in notebooks/
                    h = calculate_file_hash(filepath)
                    if h: state[filepath] = h
            
    # Add generated inventories
    for file in os.listdir(SCRIPT_DIR):
        if file.startswith('inventory_') and file.endswith('.json'):
            filepath = os.path.join(SCRIPT_DIR, file)
            h = calculate_file_hash(filepath)
            if h: state[filepath] = h
            
    # Directories
    for dirname in SYNC_DIRS:
        if os.path.exists(dirname):
            for root, _, files in os.walk(dirname):
                # EXCLUSIONS
                if '__pycache__' in root or '.ipynb_checkpoints' in root or '.git' in root:
                    continue
                # Exclude large binary/runtime deps (downloaded by notebook at runtime)
                if 'data/datasets' in root or 'ImportLDraw' in root:
                    continue
                # Exclude tests and outputs
                if 'tests' in root or 'test_output' in root:
                    continue
                
                for file in files:
                    # Exclude security and temporary files
                    if file == '.DS_Store' or file.endswith('.pyc'):
                        continue
                    if 'credentials' in file and file.endswith('.json'):
                        continue
                    if 'token' in file and file.endswith('.pickle'):
                        continue
                    if file in ['.sync_state.json', 'kaggle_project.zip', 'temp_query.jpg']:
                        continue
                    
                    filepath = os.path.join(root, file)
                    h = calculate_file_hash(filepath)
                    if h: state[filepath] = h
            
    # Add large dataset ZIPs from root
    import glob
    for zip_file in glob.glob(os.path.join(SCRIPT_DIR, 'lightning_dataset_*.zip')):
        h = calculate_file_hash(zip_file)
        if h: state[zip_file] = h
    return state

def load_previous_state():
    """Loads the last sync state from JSON."""
    if os.path.exists(SYNC_STATE_FILE):
        with open(SYNC_STATE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_state(state):
    """Saves the current state to JSON."""
    with open(SYNC_STATE_FILE, 'w') as f:
        # Filter state to only include what we actually want to persist
        filtered_state = {k: v for k, v in state.items() if os.path.exists(k)}
        json.dump(filtered_state, f, indent=4)

def _get_latest_notebook():
    """Returns the absolute path to the most recently generated notebook."""
    notebooks = []
    
    # 1. Check notebooks/ directory recursively
    nb_base = os.path.join(SCRIPT_DIR, 'notebooks')
    if os.path.exists(nb_base):
        for root, _, files in os.walk(nb_base):
            for file in files:
                if file.endswith('.ipynb'):
                    if any(file.startswith(p) for p in ['Colab_', 'Kaggle_', 'LightningAI_', 'train_']):
                        if file != 'master_unified_pipeline.ipynb':
                            notebooks.append(os.path.join(root, file))
                            
    # 2. Check root SCRIPT_DIR for legacy
    for file in os.listdir(SCRIPT_DIR):
        if file.endswith('.ipynb'):
            # Track all environment-specific notebooks
            if any(file.startswith(p) for p in ['Colab_', 'Kaggle_', 'LightningAI_', 'train_']):
                if file != 'master_unified_pipeline.ipynb':
                    fp = os.path.join(SCRIPT_DIR, file)
                    if fp not in notebooks:
                        notebooks.append(fp)
    
    if not notebooks:
        return None
    # Sort by modification time
    notebooks.sort(key=os.path.getmtime, reverse=True)
    return notebooks[0]

def pack_project(output_zip="project.zip", env=None):
    """
    Zips the project structure. 
    If env is specified ('colab', 'kaggle', 'lightning'), it ONLY includes 
    notebooks matching that environment prefix.
    """
    env_prefixes = {'colab': 'Colab_', 'kaggle': 'Kaggle_', 'lightning': 'LightningAI_'}
    target_prefix = env_prefixes.get(env)
    
    print(f"📦 Packing project for {env or 'all'} into {output_zip}...")
    
    current_state = get_current_state()
    
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for filepath in current_state.keys():
             basename = os.path.basename(filepath)
             
             # FILTER: If it's a notebook and we have a target env, filter it
             if filepath.endswith('.ipynb'):
                 if target_prefix and not basename.startswith(target_prefix):
                     continue
                 # Also exclude master/legacy notebooks in packages
                 if basename == 'master_unified_pipeline.ipynb':
                     continue
                     
             arcname = os.path.relpath(filepath, SCRIPT_DIR)
             
             # EXCLUSION: Skip loose pieces (ref_pieza) in project packages
             if "ref_pieza" in arcname:
                 continue
                 
             # Prevent Kaggle/Lightning AI from recursively auto-unzipping our inner archives
             if arcname in ['cycles_kernels.zip', 'optix_cache.zip']:
                 arcname = arcname.replace('.zip', '.bin')
             zipf.write(filepath, arcname=arcname)
                          
    print(f"✅ Package Ready: {output_zip}")

def sync_to_drive():
    """Identifies changed files and uploads them to Drive. Restricts notebooks to Colab only."""
    print("🔄 Checking for changes to sync...")
    
    current_state = get_current_state()
    previous_state = load_previous_state()
    
    changed_files = []
    
    for file, hash_val in current_state.items():
        # FILTER: On Drive, we ONLY want Colab notebooks
        if file.endswith('.ipynb'):
            if not os.path.basename(file).startswith('Colab_'):
                continue
                
        if file not in previous_state or previous_state[file] != hash_val:
            changed_files.append(file)
            
    if not changed_files:
        print("✅ No content changes detected locally.")
        # We proceed to verify we have all essential files synced
        print("� Verifying all tracked files against Drive...")
        
    # ACCOUNTS CONFIGURATION
    ACCOUNTS = [
        {"email": "enriqueperezbcn1973@gmail.com", "creds": os.path.join(SCRIPT_DIR, "credentials.json"), "token": os.path.join(SCRIPT_DIR, "token_1973.pickle")},
        {"email": "enriqueperezbcn19732@gmail.com", "creds": os.path.join(SCRIPT_DIR, "credentials_2.json"), "token": os.path.join(SCRIPT_DIR, "token_19732.pickle")}
    ]
    
    try:
        from src.utils.drive_manager import DriveManager
        
        for account in ACCOUNTS:
            print(f"\n--- ☁️ Syncing to Account: {account['email']} ---")
            
            # Check if we have tokens or credentials
            if not os.path.exists(account['creds']) and not os.path.exists(account['token']):
                print(f"⚠️  Credentials not found for {account['email']}. Skipping.")
                continue
                
            dm = DriveManager(credentials_path=account['creds'], token_path=account['token'])
            
            try:
                root_id = dm.ensure_folder("Lego_Training_75078")
                
                # Always sync changed files, or verify all tracked files if none changed
                files_to_sync = changed_files if changed_files else list(current_state.keys())
                
                for file in files_to_sync:
                    # CRITICAL: Ensure we are inside SCRIPT_DIR to avoid "Users" or ".." folders
                    if not file.startswith(SCRIPT_DIR):
                        continue
                        
                    rel_file_path = os.path.relpath(file, SCRIPT_DIR)
                    if rel_file_path.startswith('..'):
                        continue # Safety check
                    
                    # EXCLUSION: Skip loose pieces (ref_pieza) for Drive sync
                    if "ref_pieza" in rel_file_path:
                        continue
                        
                    print(f"Checking {rel_file_path}...")
                    
                    parent_id = root_id
                    path_parts = os.path.dirname(rel_file_path).split(os.sep)
                    
                    for part in path_parts:
                        if not part or part == '.': continue
                        parent_id = dm.ensure_folder(part, parent_id=parent_id)
                    
                    # Upload with date check (logic added to DriveManager)
                    dm.upload_file(file, parent_id, remote_name=os.path.basename(file), overwrite=True, check_date=True)
                    
            except Exception as e:
                print(f"❌ Failed to sync to {account['email']}: {e}")

        # Update State only after attempting both? 
        # If we update state, we assume it's "synced".
        if changed_files:
            save_state(current_state)
        print("✅ Sync Cycle Complete!")
        
    except ImportError:
        print("❌ Error: src.utils.drive_manager not found!")
    except Exception as e:
        print(f"❌ Critical Sync Failure: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--kaggle', action='store_true', help='Pack for Kaggle')
    parser.add_argument('--drive', action='store_true', help='Sync to Google Drive')
    parser.add_argument('--all', action='store_true', help='Sync to both Drive and Kaggle')
    
    args = parser.parse_args()
    
    # If no arguments provided, or --all is used
    if args.all or (not args.kaggle and not args.drive):
        print("🚀 Starting full synchronization (Drive + Kaggle + Lightning)...")
        sync_to_drive()
        pack_project("kaggle_project.zip", env='kaggle')
        pack_project("lightning_project.zip", env='lightning')
    else:
        if args.kaggle:
            pack_project("kaggle_project.zip", env='kaggle')
        if args.drive:
            sync_to_drive()
