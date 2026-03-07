import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import ssl
import warnings

# Suppress xFormers warning (not applicable for Mac/MPS)
warnings.filterwarnings("ignore", message=".*xFormers is not available.*")

# Inject proper certificates globally for PyTorch Hub downloads (MacOS Fix)
import certifi
import ssl
os.environ['SSL_CERT_FILE'] = certifi.where()
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

class FeatureExtractor:
    def __init__(self, model_name='dinov2_vits14', weights_path=None):
        # Optimize for M4: Priority CUDA -> MPS (Metal) -> CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        # Load a pre-trained model for feature extraction
        if model_name == 'dinov2_vits14':
            # DINOv2 ViT-Small (384 dimensions), highly optimized for T4
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            self.feature_dim = 384
        elif model_name == 'dinov2_vitl14':
            # DINOv2 ViT-Large (1024 dimensions), high descriptor quality
            print("🚀 Loading HEAVY Descriptor Model: DINOv2 ViT-L (1024 dims)...")
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
            self.feature_dim = 1024
        elif model_name == 'mobilenet_v3_small':
            base_model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
            self.model = base_model.features
            self.avgpool = base_model.avgpool
            self.feature_dim = 576
        elif model_name == 'resnet50':
            base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.model = nn.Sequential(*list(base_model.children())[:-1])
            self.feature_dim = 2048
        
        # Load custom fine-tuned weights if provided
        if weights_path and os.path.exists(weights_path):
            try:
                checkpoint = torch.load(weights_path, map_location='cpu')
                state_dict = checkpoint.get('state_dict', checkpoint)
                self.model.load_state_dict(state_dict, strict=True)
                config = checkpoint.get('config', {})
                print(f"✅ Loaded fine-tuned weights: {os.path.basename(weights_path)} "
                      f"(epoch {checkpoint.get('epoch', '?')}, loss {checkpoint.get('loss', '?'):.4f})")
            except Exception as e:
                print(f"⚠️ Failed to load custom weights ({e}). Using base model.")

        self.model.eval()
        self.model.to(self.device)
        
        # Standard ImageNet transforms, ensuring 224x224 (multiple of 14 for ViT)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def get_embedding(self, image):
        """Image can be a PIL Image or a path."""
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        img_tTask = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model(img_tTask)
            if hasattr(self, 'avgpool'):
                features = self.avgpool(features)
            features = torch.flatten(features, 1)
            
        return features.cpu().numpy().flatten()

    def get_batch_embeddings(self, images):

        """Batch processing for speed."""
        processed_images = []
        for img in images:
            if isinstance(img, str):
                img = Image.open(img).convert('RGB')
            processed_images.append(self.transform(img))
            
        batch_tTask = torch.stack(processed_images).to(self.device)
        
        with torch.no_grad():
            features = self.model(batch_tTask)
            if hasattr(self, 'avgpool'):
                features = self.avgpool(features)
            features = torch.flatten(features, 1)
            
        return features.cpu().numpy()
