"""Shared figure style: ggsci "npg" palette (Nature Publishing Group; ggsci::pal_npg) and serif fonts
matching the original MSEA-Net figures.

Palette order as in ggsci: red, cyan, teal, navy, salmon, grey-blue, light teal, dark red, brown, beige.
"""
from matplotlib.colors import LinearSegmentedColormap

NPG = ["#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F",
       "#8491B4", "#91D1C2", "#DC0000", "#7E6148", "#B09C85"]
RED, CYAN, TEAL, NAVY, SALMON, GREYBLUE, LIGHTTEAL, DARKRED, BROWN, BEIGE = NPG

# one colour per architecture, used in every figure (MSEA-Net is always NPG red)
MODEL_COLOR = {"msea_net": RED, "efficientnetv2s_plain": CYAN, "densenet121": TEAL, "resnet50": NAVY,
               "inception_v3": SALMON, "mobilenetv3": GREYBLUE, "vgg16": BEIGE}
COMPARATOR = GREYBLUE  # muted colour for comparators in highlight-style bar charts

# one colour per Kvasir-v2 class (ROC/PR curves and UMAP); two reds avoided (brown instead of dark red)
CLASS_COLOR = [RED, CYAN, TEAL, NAVY, SALMON, GREYBLUE, LIGHTTEAL, BROWN]

# continuous maps built only from NPG colours
NPG_HEAT = LinearSegmentedColormap.from_list("npg_heat", [NAVY, CYAN, LIGHTTEAL, SALMON, RED, DARKRED])
NPG_SEQ_NAVY = LinearSegmentedColormap.from_list("npg_navy", ["#FFFFFF", "#D3DAE8", "#8491B4", NAVY, "#1F2D4A"])

INK, INK2, GRID, AXIS = "#0b0b0b", "#4a4a4a", "#e3e3e3", "#bdbdbd"


def use_style(size=7.0):
    import matplotlib
    matplotlib.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
        "font.size": size, "axes.titlesize": size + 0.5, "axes.titleweight": "bold",
        "axes.labelsize": size, "xtick.labelsize": size - 0.5, "ytick.labelsize": size - 0.5,
        "legend.fontsize": size - 0.5, "savefig.dpi": 300,
    })
