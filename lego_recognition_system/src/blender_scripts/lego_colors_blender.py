import bpy

# Full mapping of LDraw color IDs to RGBA (Linear values for Blender)
# This data is extracted and converted from src/logic/lego_colors.py
# and common LDraw color tables.

def srgb_to_linear(srgb):
    if srgb <= 0.04045:
        return srgb / 12.92
    else:
        return ((srgb + 0.055) / 1.055) ** 2.4

def hex_to_rgba(hex_str):
    hex_str = hex_str.lstrip('#')
    r_s = int(hex_str[0:2], 16) / 255.0
    g_s = int(hex_str[2:4], 16) / 255.0
    b_s = int(hex_str[4:6], 16) / 255.0
    return (srgb_to_linear(r_s), srgb_to_linear(g_s), srgb_to_linear(b_s), 1.0)

LEGO_COLORS_HEX = {
    0:  "#05131D", # Black
    1:  "#0055BF", # Blue
    2:  "#237841", # Green
    3:  "#008F9B", # Dark Turquoise
    4:  "#C91A09", # Red
    5:  "#C870A0", # Dark Pink
    6:  "#583927", # Brown
    7:  "#9BA19D", # Light Gray
    8:  "#6D6E5C", # Dark Gray
    9:  "#B4D2E3", # Light Blue
    10: "#4B9F4A", # Bright Green
    11: "#55A5AF", # Light Turquoise
    12: "#F2705E", # Salmon
    13: "#FC97AC", # Pink
    14: "#F2CD37", # Yellow
    15: "#FFFFFF", # White
    17: "#C2DAB8", # Light Green
    18: "#FBE696", # Light Yellow
    19: "#E4CD9E", # Tan
    22: "#81007B", # Purple
    25: "#FE8A18", # Orange
    26: "#923978", # Magenta
    27: "#BBE90B", # Lime
    28: "#958A73", # Dark Tan
    29: "#E4ADC8", # Bright Pink
    33: "#0020A0", # Trans Blue
    34: "#237841", # Trans Green
    36: "#C91A09", # Trans Red
    46: "#F5CD2F", # Trans Yellow
    47: "#FCFCFC", # Trans Clear
    70: "#582A12", # Reddish Brown
    71: "#A0A5A9", # Stone Gray (Light Bluish Gray)
    72: "#6C6E68", # Dark Stone Gray (Dark Bluish Gray)
    78: "#AC8247", # Pearl Light Gold
    84: "#CC702A", # Medium Dark Flesh
    85: "#6C6E68", # Dark Bluish Gray
    272: "#0A3463", # Dark Blue
    288: "#184632", # Dark Green
    320: "#720E0F", # Dark Red
    484: "#A95500", # Dark Orange
}

_BLENDER_RGBA_CACHE = {}

def get_blender_rgba(color_id):
    """Returns a fixed neutral gray (clay render style) regardless of the color_id."""
    return (0.5, 0.5, 0.5, 1.0)

