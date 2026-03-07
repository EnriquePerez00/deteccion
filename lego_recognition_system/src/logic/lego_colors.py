"""Mapping of official LDraw color IDs to their names and hex values.

Used by the UI to present human-readable color options and by the
render pipeline to apply the correct material color.
"""

LEGO_COLORS = {
    0:  {"name": "Black",            "hex": "#05131D"},
    1:  {"name": "Blue",             "hex": "#0055BF"},
    2:  {"name": "Green",            "hex": "#237841"},
    3:  {"name": "Dark Turquoise",   "hex": "#008F9B"},
    4:  {"name": "Red",              "hex": "#C91A09"},
    5:  {"name": "Dark Pink",        "hex": "#C870A0"},
    6:  {"name": "Brown",            "hex": "#583927"},
    7:  {"name": "Light Gray",       "hex": "#9BA19D"},
    8:  {"name": "Dark Gray",        "hex": "#6D6E5C"},
    9:  {"name": "Light Blue",       "hex": "#B4D2E3"},
    10: {"name": "Bright Green",     "hex": "#4B9F4A"},
    11: {"name": "Light Turquoise",  "hex": "#55A5AF"},
    12: {"name": "Salmon",           "hex": "#F2705E"},
    13: {"name": "Pink",             "hex": "#FC97AC"},
    14: {"name": "Yellow",           "hex": "#F2CD37"},
    15: {"name": "White",            "hex": "#FFFFFF"},
    17: {"name": "Light Green",      "hex": "#C2DAB8"},
    18: {"name": "Light Yellow",     "hex": "#FBE696"},
    19: {"name": "Tan",              "hex": "#E4CD9E"},
    22: {"name": "Purple",           "hex": "#81007B"},
    25: {"name": "Orange",           "hex": "#FE8A18"},
    26: {"name": "Magenta",          "hex": "#923978"},
    27: {"name": "Lime",             "hex": "#BBE90B"},
    28: {"name": "Dark Tan",         "hex": "#958A73"},
    29: {"name": "Bright Pink",      "hex": "#E4ADC8"},
    33: {"name": "Trans Blue",       "hex": "#0020A0"},
    34: {"name": "Trans Green",      "hex": "#237841"},
    36: {"name": "Trans Red",        "hex": "#C91A09"},
    46: {"name": "Trans Yellow",     "hex": "#F5CD2F"},
    47: {"name": "Trans Clear",      "hex": "#FCFCFC"},
    70: {"name": "Reddish Brown",    "hex": "#582A12"},
    71: {"name": "Stone Gray",       "hex": "#A0A5A9"},
    72: {"name": "Dark Stone Gray",  "hex": "#6C6E68"},
    78: {"name": "Pearl Light Gold", "hex": "#AC8247"},
    84: {"name": "Medium Dark Flesh","hex": "#CC702A"},
    85: {"name": "Dark Bluish Gray", "hex": "#6C6E68"},
    272: {"name": "Dark Blue",       "hex": "#0A3463"},
    288: {"name": "Dark Green",      "hex": "#184632"},
    320: {"name": "Dark Red",        "hex": "#720E0F"},
    484: {"name": "Dark Orange",     "hex": "#A95500"},
}
# --- HELPER FUNCTIONS ---
def get_color_name(color_id: int) -> str:
    """Returns the human-readable name of an LDraw color ID."""
    if color_id in LEGO_COLORS:
        return LEGO_COLORS[color_id]["name"]
    return f"Unknown Color ({color_id})"

def get_num_colors() -> int:
    """Returns total number of tracked LEGO colors for one-hot vector sizing."""
    return len(LEGO_COLORS)

def get_color_onehot(color_id: int):
    """DEPRECATED: Returns a zero vector as color is no longer used."""
    import numpy as np
    return np.zeros(1, dtype=np.float32)

__all__ = ["LEGO_COLORS", "get_color_name", "get_num_colors", "get_color_onehot"]
