#
# Copyright 2026 Wageningen University & Research (WUR)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Repository plotting conventions."""

import matplotlib as mpl
import numpy as np

# Fixed order, never cycled past its length; checked for colour-vision separation.
CYCLE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7", "#e34948"]

COLORS = {
    "env": "#52514e",
    "plume": "#eb6834",
    "les": "#2a78d6",
    "obs": "#1a1a19",
    "buoy": "#2a78d6",
    "drag": "#eb6834",
    "total": "#52514e",
    "ent": "#2a78d6",
    "det": "#eb6834",
}

LINESTYLES = {
    "env": "-",
    "plume": "-",
    "model": "-",
    "reference": "--",     # LES or observations
    "saturated": ":",
    "envelope": (0, (5, 2, 1, 2)),  # upper percentile of the reference data
    "zero": (0, (1, 3)),
}

LINEWIDTHS = {"profile": 2.0, "reference": 2.0, "guide": 1.0, "envelope": 1.0}

# Opacity of the band showing the spread of the reference data about its mean.
BAND_ALPHA = 0.15

# A sweep over a magnitude (H, area) is one hue, light to dark; 2D fields share one map.
SEQUENTIAL_CMAP = "Blues"
FIELD_CMAP = "viridis"

MARKERS = {"condensed": "o", "reached_top": "x", "lcl": "o", "case": "o"}


def run_color(i):
    """Colour of the i-th run in a sensitivity series; identity, so never rank-ordered."""
    return CYCLE[i % len(CYCLE)]


def sweep_colors(n):
    """n colours for the steps of a sweep, light to dark, kept clear of the white surface."""
    cmap = mpl.colormaps[SEQUENTIAL_CMAP]
    return [cmap(x) for x in np.linspace(0.45, 1.0, n)]


def apply():
    """Set the repository's matplotlib defaults."""
    mpl.rcParams.update({
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.prop_cycle": mpl.cycler(color=CYCLE),
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": "#c3c2b7",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.6,
        "lines.linewidth": LINEWIDTHS["profile"],
        "legend.frameon": False,
    })
