import sys
from matplotlib.cm import Paired, Set3, bwr, tab20, tab20b, tab20c
from matplotlib.colors import LinearSegmentedColormap, ListedColormap


# =============================================================================+
# Figure
# =============================================================================+
FIGURE_SIZE = (16, 16)

# =============================================================================+
# Fonts
# =============================================================================+
FONT_LARGEST = {'fontsize': 24, 'weight': 'bold', 'color': '#220530'}
FONT_LARGER = {'fontsize': 20, 'weight': 'bold', 'color': '#220530'}
FONT_STANDARD = {'fontsize': 16, 'weight': 'bold', 'color': '#220530'}
FONT_SMALLER = {'fontsize': 12, 'weight': 'bold', 'color': '#220530'}
FONT_SMALLEST = {'fontsize': 8, 'weight': 'bold', 'color': '#220530'}

# =============================================================================+
# Color maps
# =============================================================================+
C_BAD = 'wheat'

# Continuous colormap
CMAP_CONTINUOUS_BWR = bwr
CMAP_CONTINUOUS_BWR.set_bad(C_BAD)

reds = [0.26, 0.26, 0.26, 0.39, 0.69, 1, 1, 1, 1, 1, 1]
greens_half = [0.26, 0.16, 0.09, 0.26, 0.69]
colordict = {
    'red':
    tuple([(0.1 * i, r, r) for i, r in enumerate(reds)]),
    'green':
    tuple([
        (0.1 * i, r, r)
        for i, r in enumerate(greens_half + [1] + list(reversed(greens_half)))
    ]),
    'blue':
    tuple([(0.1 * i, r, r) for i, r in enumerate(reversed(reds))])
}
CMAP_CONTINUOUS_ASSOCIATION = LinearSegmentedColormap('association', colordict)
CMAP_CONTINUOUS_ASSOCIATION.set_bad(C_BAD)

# Categorical colormap
CMAP_CATEGORICAL_PAIRED = Paired
CMAP_CATEGORICAL_PAIRED.set_bad(C_BAD)

CMAP_CATEGORICAL_SET3 = Set3
CMAP_CATEGORICAL_SET3.set_bad(C_BAD)

CMAP_CATEGORICAL_TAB20 = tab20
CMAP_CATEGORICAL_TAB20.set_bad(C_BAD)

CMAP_CATEGORICAL_TAB20B = tab20b
CMAP_CATEGORICAL_TAB20B.set_bad(C_BAD)

CMAP_CATEGORICAL_TAB20C = tab20c
CMAP_CATEGORICAL_TAB20C.set_bad(C_BAD)

# Binary colormap
CMAP_BINARY_WB = ListedColormap(['#E8E8E8', '#080808'])
CMAP_BINARY_WB.set_bad(C_BAD)

CMAP_BINARY_RE = ListedColormap(['#F80088', '#20D9BA'])
CMAP_BINARY_RE.set_bad(C_BAD)
