import os
import re
import logging

logger = logging.getLogger("LegoVision")

def count_ldraw_lines(ldraw_path):
    """
    Parses an LDraw .dat file and counts structural lines (types 2, 3, 4, 5).
    Type 1 (sub-files) are not followed to keep it simple and represent 
    component-based complexity.
    """
    if not ldraw_path or not os.path.exists(ldraw_path):
        return 0
    
    count = 0
    try:
        with open(ldraw_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # LDraw line types:
                # 0: Comment/Meta
                # 1: Sub-file reference
                # 2: Line
                # 3: Triangle
                # 4: Quadrilateral
                # 5: Optional Line
                if line[0] in ('2', '3', '4', '5'):
                    count += 1
    except Exception as e:
        logger.warning(f"Error parsing LDraw file {ldraw_path}: {e}")
        
    return count

def get_part_tier(part_id, category="", ldraw_path=None):
    """
    Classifies a LEGO part into one of 4 Tiers based on geometric descriptors.
    
    Tier 1: Simple/High Symmetry (Bricks, Plates, Tiles < 50 lines)
    Tier 2: Medium/Bilateral (Slopes, Wedges, Beams)
    Tier 3: Complex/Asymmetric (Gears, Pins, Connectors > 150 lines)
    Tier 4: Organic/Chaos (Minifigs, Plants, Transparent)
    """
    line_count = count_ldraw_lines(ldraw_path) if ldraw_path else 0
    cat = category.lower()
    pid = str(part_id).lower()

    # Tier 4: Organic / Chaos / Transparent
    if any(k in cat for k in ['minifig', 'plant', 'animal', 'cloth', 'hair']) or \
       any(k in pid for k in ['sw', 'fig']) or \
       'transparent' in cat:
        return "TIER4"

    # Tier 1: Bricks, Plates, Tiles (Simple shapes)
    if any(k in cat for k in ['brick', 'plate', 'tile']):
        if line_count > 0 and line_count < 60:
            return "TIER1"
        elif line_count == 0: # No LDraw context, assume Tier 1 if it's a standard simple category
            return "TIER1"

    # Tier 3: Complex / High Detail / Mechanical
    if any(k in cat for k in ['gear', 'technic pin', 'technic connector', 'large figure', 'electric', 'pneumatic']):
        return "TIER3"
    
    # Special complex bricks
    if 'technic brick' in cat and line_count > 100:
        return "TIER3"

    # Tier 2: Medium / Slopes / Beams
    if any(k in cat for k in ['slope', 'wedge', 'technic beam', 'technic axle', 'technic steering']):
        return "TIER2"

    # Fallbacks based on line count if category is vague
    if line_count > 0:
        if line_count < 60: return "TIER1"
        if line_count < 150: return "TIER2"
        return "TIER3"

    # Default fallback
    return "TIER2"
