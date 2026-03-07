"""
triplet_dataset.py
==================
PyTorch Dataset that generates (anchor, positive, negative) triplets
for DINOv2 fine-tuning on LEGO pieces.

- Anchor: image of piece X
- Positive: different image of SAME piece X (different angle)
- Negative: image of a hard negative piece (visually similar but different)
"""
import os
import json
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class TripletDataset(Dataset):
    """
    Generates triplets from rendered LEGO images.

    Args:
        render_dir: Path to render_local/ref_pieza/
        hard_negatives_path: Path to hard_negatives.json
        transform: torchvision transform to apply to images
        samples_per_piece: Number of triplets to generate per piece per epoch
    """

    def __init__(
        self,
        render_dir: str,
        hard_negatives_path: str,
        transform=None,
        samples_per_piece: int = 20,
    ):
        self.render_dir = render_dir
        self.samples_per_piece = samples_per_piece

        # Default transform (DINOv2 standard)
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Load hard negatives
        with open(hard_negatives_path, "r") as f:
            data = json.load(f)
        self.hard_negatives = data.get("hard_negatives", {})

        # Build piece → image paths mapping
        self.piece_images = {}  # piece_id → [img_path, ...]
        self._scan_images()

        # Filter to pieces that have both images AND hard negatives
        valid_pieces = []
        for pid in self.piece_images:
            if len(self.piece_images[pid]) >= 2 and pid in self.hard_negatives:
                # Check that at least one hard negative has images
                neg_ids = [n["piece_id"] for n in self.hard_negatives[pid]]
                neg_with_imgs = [n for n in neg_ids if n in self.piece_images and self.piece_images[n]]
                if neg_with_imgs:
                    valid_pieces.append(pid)

        self.valid_pieces = valid_pieces
        self.total_samples = len(valid_pieces) * samples_per_piece

        print(f"📦 TripletDataset: {len(valid_pieces)} pieces, "
              f"{self.total_samples} triplets/epoch, "
              f"{sum(len(v) for v in self.piece_images.values())} total images")

    def _scan_images(self):
        """Scan render_dir for piece images."""
        if not os.path.isdir(self.render_dir):
            return

        for piece_dir in os.listdir(self.render_dir):
            img_dir = os.path.join(self.render_dir, piece_dir, "images")
            if not os.path.isdir(img_dir):
                continue

            images = [
                os.path.join(img_dir, f)
                for f in os.listdir(img_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            if images:
                self.piece_images[piece_dir] = images

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # Determine which piece this sample belongs to
        piece_idx = idx // self.samples_per_piece
        piece_id = self.valid_pieces[piece_idx % len(self.valid_pieces)]

        # Anchor: random image of this piece
        piece_imgs = self.piece_images[piece_id]
        anchor_path = random.choice(piece_imgs)

        # Positive: different image of SAME piece
        positive_candidates = [p for p in piece_imgs if p != anchor_path]
        if not positive_candidates:
            positive_candidates = piece_imgs  # Fallback if only 1 image
        positive_path = random.choice(positive_candidates)

        # Negative: image from hard negative piece
        neg_piece_ids = [n["piece_id"] for n in self.hard_negatives[piece_id]]
        neg_with_imgs = [n for n in neg_piece_ids if n in self.piece_images and self.piece_images[n]]

        if neg_with_imgs:
            # Weighted selection: harder negatives (higher similarity) more likely
            weights = []
            for n in neg_with_imgs:
                # Find similarity score
                sim = 0.5
                for entry in self.hard_negatives[piece_id]:
                    if entry["piece_id"] == n:
                        sim = entry["similarity"]
                        break
                weights.append(sim)

            neg_piece = random.choices(neg_with_imgs, weights=weights, k=1)[0]
        else:
            # Fallback: random different piece
            other_pieces = [p for p in self.valid_pieces if p != piece_id]
            neg_piece = random.choice(other_pieces)

        negative_path = random.choice(self.piece_images[neg_piece])

        # Load and transform images
        anchor = self._load_image(anchor_path)
        positive = self._load_image(positive_path)
        negative = self._load_image(negative_path)

        return anchor, positive, negative

    def _load_image(self, path):
        """Load and transform a single image."""
        img = Image.open(path).convert("RGB")
        return self.transform(img)
