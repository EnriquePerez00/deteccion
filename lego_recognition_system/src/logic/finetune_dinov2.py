"""
finetune_dinov2.py
==================
Fine-tunes DINOv2-ViT-S14 using Triplet Margin Loss for LEGO piece specialization.

Unfreezes the last N encoder layers and trains with (anchor, positive, negative) triplets.
Saves versioned weights as models/dinov2_lego/dinov2_lego_{timestamp}.pth.
"""
import os
import sys
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def finetune(
    render_dir: str,
    hard_negatives_path: str,
    output_dir: str,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-5,
    margin: float = 0.3,
    unfreeze_layers: int = 4,
    samples_per_piece: int = 20,
):
    """
    Fine-tune DINOv2-ViT-S14 with Triplet Loss.

    Args:
        render_dir: Path to render_local/ref_pieza/
        hard_negatives_path: Path to hard_negatives.json
        output_dir: Directory to save fine-tuned weights
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        margin: Triplet loss margin
        unfreeze_layers: Number of encoder layers to unfreeze (from the end)
        samples_per_piece: Triplets per piece per epoch
    """
    from src.logic.triplet_dataset import TripletDataset

    # --- Device ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"🖥️ Device: {device}")

    # --- Load DINOv2 Base Model ---
    print("🧠 Loading DINOv2-ViT-S14 base model...")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

    # --- Freeze all parameters first ---
    for param in model.parameters():
        param.requires_grad = False

    # --- Unfreeze last N encoder layers ---
    total_layers = len(model.blocks)
    unfreeze_from = max(0, total_layers - unfreeze_layers)
    unfrozen_count = 0
    for i in range(unfreeze_from, total_layers):
        for param in model.blocks[i].parameters():
            param.requires_grad = True
            unfrozen_count += 1

    # Also unfreeze the final layer norm
    if hasattr(model, 'norm'):
        for param in model.norm.parameters():
            param.requires_grad = True
            unfrozen_count += 1

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Parameters: {trainable_params:,} trainable / {total_params:,} total "
          f"({100*trainable_params/total_params:.1f}%)")
    print(f"   Unfroze last {unfreeze_layers} of {total_layers} blocks")

    model = model.to(device)
    model.train()

    # --- Dataset ---
    dataset = TripletDataset(
        render_dir=render_dir,
        hard_negatives_path=hard_negatives_path,
        samples_per_piece=samples_per_piece,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # --- Loss and Optimizer ---
    criterion = nn.TripletMarginLoss(margin=margin, p=2)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # --- Training Loop ---
    print(f"\n🚀 Starting fine-tuning: {epochs} epochs, batch_size={batch_size}, "
          f"lr={lr}, margin={margin}")
    print(f"   Dataset: {len(dataset)} triplets/epoch\n")

    best_loss = float("inf")
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (anchors, positives, negatives) in enumerate(dataloader):
            anchors = anchors.to(device)
            positives = positives.to(device)
            negatives = negatives.to(device)

            # Forward pass
            anchor_emb = model(anchors)
            positive_emb = model(positives)
            negative_emb = model(negatives)

            # L2 Normalize embeddings
            anchor_emb = nn.functional.normalize(anchor_emb, p=2, dim=1)
            positive_emb = nn.functional.normalize(positive_emb, p=2, dim=1)
            negative_emb = nn.functional.normalize(negative_emb, p=2, dim=1)

            loss = criterion(anchor_emb, positive_emb, negative_emb)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(num_batches, 1)

        # Log
        lr_current = scheduler.get_last_lr()[0]
        print(f"[EPOCH {epoch+1}/{epochs}] Loss: {avg_loss:.4f} | LR: {lr_current:.2e}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            ts = datetime.datetime.now().strftime('%Y%m%d_%H%M')
            best_path = os.path.join(output_dir, f"dinov2_lego_{ts}.pth")

            # Save only trainable state_dict (smaller file)
            save_dict = {
                "state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "loss": avg_loss,
                "config": {
                    "base_model": "dinov2_vits14",
                    "unfreeze_layers": unfreeze_layers,
                    "margin": margin,
                    "feature_dim": 384,
                },
            }
            torch.save(save_dict, best_path)
            print(f"   💾 Best model saved: {best_path}")

    print(f"\n✅ Fine-tuning complete. Best loss: {best_loss:.4f}")
    return best_path


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    finetune(
        render_dir=os.path.join(project_root, "render_local", "ref_pieza"),
        hard_negatives_path=os.path.join(project_root, "models", "hard_negatives.json"),
        output_dir=os.path.join(project_root, "models", "dinov2_lego"),
        epochs=50,
        batch_size=32,
    )
