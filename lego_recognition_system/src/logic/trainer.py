from ultralytics import YOLO
import os
import datetime
import glob

class ModelTrainer:
    def __init__(self, model_dir="./models/yolo"):
        self.model_dir = os.path.abspath(model_dir)
        os.makedirs(self.model_dir, exist_ok=True)
        
    def train_model(self, set_id, dataset_path, epochs=10, progress_callback=None, base_model='yolo26n.pt'):
        """
        Train a YOLO model on the generated dataset.
        dataset_path: Path to the data.yaml file or dataset directory.
        progress_callback: Optional function(progress_percent, status_string)
        base_model: Path or name of the starting weights (e.g. 'yolo26n.pt' or 'best.pt')
        """
        print(f"Initializing YOLO with base: {base_model} for Set {set_id}...")
        model = YOLO(base_model) 
        
        # Callback for UI Progress
        if progress_callback:
            def on_train_epoch_end(trainer):
                current_epoch = trainer.epoch + 1
                total_epochs = trainer.epochs
                progress = int((current_epoch / total_epochs) * 100)
                status = f"Training Epoch {current_epoch}/{total_epochs}"
                progress_callback(progress, status)
                
            model.add_callback("on_train_epoch_end", on_train_epoch_end)

        project_name = f"yolo26_set_{set_id}"
        print(f"Starting training on device='mps' for {epochs} epochs...")
        
        if os.path.isdir(dataset_path):
             data_file = os.path.join(dataset_path, "data.yaml")
        else:
             data_file = dataset_path
             
        results = model.train(
            data=data_file,
            epochs=epochs,
            imgsz=640,    
            device='cpu', # Forced CPU for stability (MPS has shape mismatch issues here)
            project=self.model_dir,
            name=project_name,
            exist_ok=True,
            plots=True,
            workers=8,
            batch=8,
            cache=True,
            amp=True,
            
            # 🖼️ Safe settings for MPS stability
            mosaic=0.0, 
            erasing=0.1,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=90.0,
            scale=0.1,
            optimizer='auto'
        )
        
        # Move to a timestamped version in models_dir
        # Standardize: detector_universal_YYYYMMDD_HHMM.pt
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
        # We put it in models/yolo_model usually, but here trainer has its own model_dir
        # Let's use the internal logic: best.pt is in project_name/weights/
        best_pt_path = os.path.join(self.model_dir, project_name, "weights", "best.pt")
        
        target_name = f"yolo_model_{timestamp}.pt"
        target_path = os.path.join(self.model_dir, target_name)
        
        if os.path.exists(best_pt_path):
            import shutil
            shutil.copy2(best_pt_path, target_path)
            print(f"✅ Training complete. Versioned model saved: {target_path}")
        else:
            print(f"⚠️ Training complete but best.pt not found at {best_pt_path}")
            
        return results

    def get_model_path(self, set_id):
        # Look for the latest timestamped model
        pattern = os.path.join(self.model_dir, "yolo_model_*.pt")
        models = glob.glob(pattern)
        if models:
            # Sort by name (which includes timestamp) to get the latest
            return sorted(models)[-1]
        
        # Fallback to the ultralytics default if no versioned one exists
        fallback = os.path.join(self.model_dir, f"yolo26_set_{set_id}", "weights", "best.pt")
        return fallback

