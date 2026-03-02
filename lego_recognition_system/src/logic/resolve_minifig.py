
import os
import json
import logging
import urllib.request
import ssl
from pathlib import Path

# Import color lookup for human-readable names
try:
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent))
    from lego_colors import LEGO_COLORS as _LEGO_COLORS, get_color_name as _get_color_name
except ImportError:
    _LEGO_COLORS = {}
    def _get_color_name(cid): return f"Color {cid}"

# Create an unverified context for macOS SSL issues
ssl_ctx = ssl.create_default_context()
ssl_ctx.check_hostname = False
ssl_ctx.verify_mode = ssl.CERT_NONE

logger = logging.getLogger("LegoVision")

REBRICKABLE_API_BASE = "https://rebrickable.com/api/v3/lego"

class MinifigResolver:
    def __init__(self, ldraw_path=None, api_key=None):
        self.ldraw_path = Path(ldraw_path) if ldraw_path else None
        self.api_key = api_key
        self.mapping = {} # Rebrickable ID -> LDraw ID
        
        if self.ldraw_path:
            self._load_parts_lst()

    def _load_parts_lst(self):
        """Parses LDraw parts.lst for cross-referencing."""
        parts_lst_path = self.ldraw_path / "parts.lst"
        if not parts_lst_path.exists():
            logger.warning(f"⚠️ {parts_lst_path} not found. Mapping might be limited.")
            return

        logger.info(f"📜 Parsing LDraw parts.lst at {parts_lst_path}")
        try:
            with open(parts_lst_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    # Skip comments and metadata
                    if not line or line.startswith('0'): continue
                    
                    # parts.lst format: <ldraw_filename> <description>
                    parts = line.split(maxsplit=1)
                    if len(parts) < 2: continue
                    
                    filename = parts[0].lower()
                    description = parts[1]
                    
                    # Extract IDs from filename (remove .dat, .png, etc.)
                    base_id = filename.split('.')[0]
                    self.mapping[base_id] = base_id # Default mapping
                    
                    # Look for external IDs in description (Rebrickable/BrickLink)
                    # Common patterns: "= 1234", "BrickLink ID 1234", "BL 1234"
                    import re
                    # Look for "= [ID]" or "BrickLink [ID]"
                    match = re.search(r'=\s*(\d+[a-z]*\d*)', description, re.IGNORECASE)
                    if not match:
                        match = re.search(r'BrickLink\s+ID\s+(\d+[a-z]*\d*)', description, re.IGNORECASE)
                    
                    if match:
                        ext_id = match.group(1).lower()
                        self.mapping[ext_id] = base_id
                        # logger.debug(f"Mapped {ext_id} -> {base_id}")
                        
        except Exception as e:
            logger.error(f"❌ Error parsing parts.lst: {e}")

    def get_minifig_parts(self, minifig_id):
        """Fetches components for a minifig from Rebrickable."""
        url = f"{REBRICKABLE_API_BASE}/minifigs/{minifig_id}/parts/"
        
        headers = {"User-Agent": "Mozilla/5.0"}
        if self.api_key:
            headers["Authorization"] = f"key {self.api_key}"
            
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10, context=ssl_ctx) as resp:
                data = json.loads(resp.read())
            
            results = data.get("results", [])
            parts = []
            
            for item in results:
                p = item.get("part", {})
                rb_id = p.get("part_num", "").lower()
                
                # Resolving logic:
                # 1. Try direct mapping from parts.lst
                # 2. Try rb_id as ldraw_id
                # 3. Try to strip 'pb' (printed) or 'c01' (assembly) suffixes
                ldraw_id = self.mapping.get(rb_id)
                
                if not ldraw_id:
                    # Fallback: trial and error with common suffixes
                    clean_id = rb_id.split('p')[0].split('c')[0]
                    ldraw_id = self.mapping.get(clean_id, clean_id)
                
                color_id_val = item.get("color", {}).get("id", 0)
                parts.append({
                    "rb_id": rb_id,
                    "ldraw_id": ldraw_id,
                    "name": p.get("name", ""),
                    "color_id": color_id_val,
                    "color_name": _get_color_name(color_id_val),
                    "type": p.get("category", {}).get("name", "Unknown")
                })
            
            logger.info(f"✅ Resolved {len(parts)} parts for minifig {minifig_id}")
            return parts
        except Exception as e:
            logger.error(f"❌ Failed to fetch minifig parts for {minifig_id}: {e}")
            return []

if __name__ == "__main__":
    # Test
    resolver = MinifigResolver()
    # parts = resolver.get_minifig_parts("sw0001")
    # print(json.dumps(parts, indent=4))
