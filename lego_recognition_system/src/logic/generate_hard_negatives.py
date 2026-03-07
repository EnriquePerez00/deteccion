"""
generate_hard_negatives.py
==========================
Generates a hard_negatives.json file for DINOv2 Triplet fine-tuning.

Strategy (2-layer):
  Layer 1: Rebrickable API → Group pieces by part_cat_id (geometric family).
  Layer 2: FAISS visual similarity → Rank within category by cosine similarity.

Result: For each piece, the top-5 most visually similar *different* pieces.
"""
import os
import json
import numpy as np
import logging
import ssl
import certifi

# Fix SSL certificate issues for macOS
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

logger = logging.getLogger("LegoVision")

REBRICKABLE_API_BASE = "https://rebrickable.com/api/v3/lego"


def _get_api_key():
    """Load Rebrickable API key from config.json."""
    try:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                return json.load(f).get("rebrickable_api_key")
    except:
        pass
    return None


def _fetch_part_category(part_num, api_key=None):
    """Fetch part_cat_id from Rebrickable API."""
    import urllib.request
    # Strip color suffix (e.g., "3022_72" → "3022")
    clean_num = part_num.split("_")[0]
    url = f"{REBRICKABLE_API_BASE}/parts/{clean_num}/"
    headers = {"User-Agent": "Mozilla/5.0"}
    if api_key:
        headers["Authorization"] = f"key {api_key}"
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        return {
            "part_cat_id": data.get("part_cat_id"),
            "part_cat_name": data.get("part_categories", {}).get("name", ""),
            "name": data.get("name", ""),
        }
    except Exception as e:
        logger.warning(f"⚠️ Could not fetch category for {clean_num}: {e}")
        return None


def generate_hard_negatives(
    index_path: str,
    render_dir: str,
    output_path: str,
    top_k: int = 5,
):
    """
    Generate hard_negatives.json from existing FAISS index and Rebrickable categories.

    Args:
        index_path: Path to lego.index (FAISS index)
        render_dir: Path to render_local/ref_pieza/
        output_path: Path to save hard_negatives.json
        top_k: Number of hard negatives per piece
    """
    from src.logic.vector_index import VectorIndex

    # 1. Load current FAISS index
    print("📂 Loading FAISS index...")
    v_index = VectorIndex()
    if not v_index.load(index_path):
        raise FileNotFoundError(f"Could not load index from {index_path}")

    # 2. Discover rendered pieces
    rendered_pieces = set()
    if os.path.isdir(render_dir):
        for d in os.listdir(render_dir):
            img_dir = os.path.join(render_dir, d, "images")
            if os.path.isdir(img_dir) and os.listdir(img_dir):
                rendered_pieces.add(d)
    print(f"🧩 Found {len(rendered_pieces)} rendered pieces")

    # 3. Fetch categories from Rebrickable (Layer 1)
    api_key = _get_api_key()
    piece_categories = {}
    category_groups = {}  # cat_id → [piece_ids]

    print("🌐 Fetching part categories from Rebrickable...")
    for piece_id in sorted(rendered_pieces):
        clean_id = piece_id.split("_")[0]
        # Avoid duplicate lookups for same base part
        if clean_id in piece_categories:
            cat_id = piece_categories[clean_id]
        else:
            info = _fetch_part_category(piece_id, api_key)
            cat_id = info["part_cat_id"] if info else None
            piece_categories[clean_id] = cat_id
            if info:
                print(f"   {piece_id} → cat {cat_id} ({info.get('name', '')})")

        if cat_id is not None:
            category_groups.setdefault(cat_id, []).append(piece_id)

    # Pieces with no category → fallback to global pool
    uncategorized = [p for p in rendered_pieces if piece_categories.get(p.split("_")[0]) is None]
    if uncategorized:
        category_groups[-1] = uncategorized
        print(f"   ⚠️ {len(uncategorized)} pieces without category (will use global pool)")

    # 4. Compute centroids per piece from metadata
    piece_centroids = {}
    piece_vectors = {}

    for i, meta in enumerate(v_index.metadata):
        ldraw_id = meta.get("ldraw_id", "Unknown")
        color_id = meta.get("color_id", -1)
        uid = f"{ldraw_id}_{color_id}" if color_id >= 0 else ldraw_id

        if uid not in piece_vectors:
            piece_vectors[uid] = []

        # Reconstruct vector from FAISS
        cpu_index = v_index.index
        try:
            vec = cpu_index.reconstruct(i)
            piece_vectors[uid].append(vec)
        except:
            pass

    for uid, vecs in piece_vectors.items():
        if vecs:
            centroid = np.mean(vecs, axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            piece_centroids[uid] = centroid

    print(f"📊 Computed centroids for {len(piece_centroids)} piece variants")

    # 5. Rank by visual similarity within category groups (Layer 2)
    hard_negatives = {}

    for cat_id, members in category_groups.items():
        # Filter to members that have centroids
        members_with_centroids = [m for m in members if m in piece_centroids]

        for piece_id in members_with_centroids:
            anchor = piece_centroids[piece_id]

            # Candidates: same category + global pool (for diversity)
            candidates = [m for m in members_with_centroids if m != piece_id]

            # Also add pieces from other categories that might be confusing
            # (cross-category confusion is important too)
            all_with_centroids = [p for p in piece_centroids if p != piece_id and p not in candidates]
            candidates.extend(all_with_centroids[:20])  # Add up to 20 from other categories

            if not candidates:
                continue

            # Compute cosine similarities
            similarities = []
            for cand in candidates:
                sim = float(np.dot(anchor, piece_centroids[cand]))
                similarities.append((cand, sim))

            # Sort by similarity descending (most confusing first)
            similarities.sort(key=lambda x: x[1], reverse=True)

            # Take top-K
            hard_negatives[piece_id] = [
                {"piece_id": s[0], "similarity": round(s[1], 4)}
                for s in similarities[:top_k]
            ]

    # 6. Save result
    output = {
        "generated_at": __import__("datetime").datetime.now().isoformat(),
        "total_pieces": len(hard_negatives),
        "top_k": top_k,
        "category_groups": {str(k): v for k, v in category_groups.items()},
        "hard_negatives": hard_negatives,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"✅ Hard negatives saved to {output_path}")
    print(f"   {len(hard_negatives)} pieces with {top_k} hard negatives each")

    return output


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    generate_hard_negatives(
        index_path=os.path.join(project_root, "models", "piezas_vectores", "lego.index"),
        render_dir=os.path.join(project_root, "render_local", "ref_pieza"),
        output_path=os.path.join(project_root, "models", "hard_negatives.json"),
    )
