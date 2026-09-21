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

"""Repository-wide plotting conventions. Nothing below is decided in a script."""

import matplotlib as mpl

# Fixed order, never cycled past its length; checked for colour-vision separation.
CYCLE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7", "#e34948"]

COLORS = {
    "env": "#52514e",      # environment, neutral so the plume carries the identity
    "plume": "#eb6834",
    "les": "#2a78d6",      # reference data the model is validated against
    "obs": "#1a1a19",
    "buoy": "#2a78d6",     # w-budget: buoyancy production
    "drag": "#eb6834",     # w-budget: entrainment drag
    "total": "#52514e",    # w-budget: sum
    "ent": "#2a78d6",
    "det": "#eb6834",
}

LINESTYLES = {
    "env": "-",
    "plume": "-",
    "model": "-",
    "reference": "--",     # LES or observations, against the model's solid line
    "saturated": ":",
    "zero": (0, (1, 3)),
}

LINEWIDTHS = {"profile": 2.0, "reference": 2.0, "guide": 1.0}


def run_color(i):
    """Colour of the i-th run in a sensitivity series; identity, so never rank-ordered."""
    return CYCLE[i % len(CYCLE)]


def apply():
    """Set the repository's matplotlib defaults."""
    mpl.rcParams.update({
        "figure.dpi": 120,
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
