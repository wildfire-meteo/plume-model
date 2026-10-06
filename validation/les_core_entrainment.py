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

"""How does the LES plume core entrain, and what base state does it start from?

The core is the centreline of the 95th-percentile w, with theta'_core on it and w_95 as its
w. Its theta' budget, dtheta'/dz = -eps theta' - dtheta_e/dz, is inverted for an implied
eps_core (les.py), which is compared with the buoyant scaling B_core / w_core**2, the Morton
floor 1/sqrt(A_0) and the default model's eps = fac_ent/sqrt(A_0) + a B/w**2 on the LES core.
The ratio a = eps_core w_core**2 / B_core is the coefficient the core implies for the buoyant
term alone, and eta = d(w_core**2)/dz / 2 B_core its net efficiency.

The core w budget, d(w_core**2)/dz = 2 a_w B_core - 2 b_w eps_core w_core**2, is regressed
for a_w at fixed b_w and shown against z / h_abl, and the base check compares PlumeBase at the default model's a_w, b_w
with the LES core and mean plume at h0.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import config
from plume_model import PlumeBase
from plume_model.entrainment import A_BUOYANT, FAC_ENT_BUOYANT
from plume_model.plume import A_W, B_W, H0_PLUME

sys.path.insert(0, str(Path(__file__).parent))
import les
from les_dry_line_fire import at_height, fli_values, guide_line

OUTPUT = Path(__file__).parent / "output" / "les_dry_line_fire" / "core_entrainment"

Z_TABLE = (50.0, 100.0, 250.0, 500.0, 800.0)
# Heights relative to the boundary-layer top, for the core above it.
Z_REL_TABLE = (0.8, 1.0, 1.2, 1.4)
# The w budget is regressed between Z_BUDGET_MIN and this fraction of the boundary-layer top.
Z_BUDGET_MIN = 30.0
H_ABL_BUDGET = 0.9
B_W_FIXED = (0.2, B_W, 1.0)


def buoyant_scaling(case):
    """B_core / w_core**2 [1/m], where the core is positively buoyant."""
    scaling = case.buoy_core / case.w_core**2
    scaling[case.buoy_core <= 0] = np.nan
    return scaling


def implied_a(case):
    return case.core_entrainment / buoyant_scaling(case)


def core_model_entrainment(case):
    """The default model's eps [1/m], evaluated on the LES core."""
    return (FAC_ENT_BUOYANT / np.sqrt(case.area_fire)
            + A_BUOYANT * np.maximum(case.buoy_core, 0.0) / case.w_core**2)


def print_tables(cases):
    morton = np.array([1 / np.sqrt(c.area_fire) for c in cases])
    print("Core = centreline of the 95th-percentile w; w_core = w_95. Morton floor 1/sqrt(A_0) "
          "= %.2f-%.2f /km" % (morton.min() * 1e3, morton.max() * 1e3))
    tables = [
        ("eps_core [1/km], implied by the core theta' budget",
         lambda c: c.core_entrainment, 1e3),
        ("B_core / w_core**2 [1/km]", buoyant_scaling, 1e3),
        ("default model eps = %g/sqrt(A_0) + %g B/w**2 on the LES core [1/km]"
         % (FAC_ENT_BUOYANT, A_BUOYANT), core_model_entrainment, 1e3),
        ("a = eps_core w_core**2 / B_core", implied_a, 1.0),
        ("eta_core = d(w_core**2)/dz / 2 B_core", lambda c: c.core_efficiency, 1.0),
        ("w_core_centreline / w_95", lambda c: c.w_core_centreline / c.w_core, 1.0),
    ]
    for title, values, scale in tables:
        print("\n" + title)
        print("%-16s" % "z [m]" + "".join("%9.0f" % z for z in Z_TABLE)
              + "   |   z / h_abl" + "".join("%7.1f" % z for z in Z_REL_TABLE))
        rows = []
        for c in cases:
            row = [at_height(c.z, values(c), z) * scale for z in Z_TABLE]
            row += [at_height(c.z, values(c), z * c.h_abl) * scale for z in Z_REL_TABLE]
            rows.append(row)
        rows = np.array(rows)
        for c, row in zip(cases, rows):
            print("%-16s" % c.name + "".join("%9.2f" % v for v in row[:len(Z_TABLE)]) + "   |   "
                  + " " * 9 + "".join("%7.2f" % v for v in row[len(Z_TABLE):]))
        med = np.nanmedian(rows, axis=0)
        print("%-16s" % "median" + "".join("%9.2f" % v for v in med[:len(Z_TABLE)]) + "   |   "
              + " " * 9 + "".join("%7.2f" % v for v in med[len(Z_TABLE):]))


def w_budget_terms(case):
    """d(w_core**2)/dz, 2 B_core and -2 eps_core w_core**2 on the LES grid."""
    w2 = les.running_mean(case.z, case.w_core, les.SMOOTH_DEPTH) ** 2
    return np.gradient(w2, case.z), 2 * case.buoy_core, -2 * case.core_entrainment * w2


def fit_w_budget(cases):
    """[(a_w, b_w, R2)]: least-squares a_w at each fixed b_w, then both free."""
    lhs, buoyancy, drag = [], [], []
    for c in cases:
        dw2dz, buoy_term, drag_term = w_budget_terms(c)
        use = ((c.z > Z_BUDGET_MIN) & (c.z < H_ABL_BUDGET * c.h_abl) & (c.buoy_core > 0)
               & np.isfinite(dw2dz) & np.isfinite(drag_term))
        lhs.append(dw2dz[use])
        buoyancy.append(buoy_term[use])
        drag.append(drag_term[use])
    lhs, buoyancy, drag = (np.concatenate(v) for v in (lhs, buoyancy, drag))

    def r2(pred):
        return 1 - np.sum((lhs - pred) ** 2) / np.sum((lhs - lhs.mean()) ** 2)

    fits = []
    for b_w in B_W_FIXED:
        a_w = np.sum((lhs - b_w * drag) * buoyancy) / np.sum(buoyancy**2)
        fits.append((a_w, b_w, r2(a_w * buoyancy + b_w * drag)))
    (a_w, b_w), *_ = np.linalg.lstsq(np.column_stack([buoyancy, drag]), lhs, rcond=None)
    fits.append((a_w, b_w, r2(a_w * buoyancy + b_w * drag)))
    print("\nCore w budget, d(w**2)/dz = 2 a_w B - 2 b_w eps_core w**2, %g m < z < %g h_abl"
          " (%d levels); the last row has both free" % (Z_BUDGET_MIN, H_ABL_BUDGET, len(lhs)))
    for a_w, b_w, fit in fits:
        print("a_w = %4.2f, b_w = %4.2f, R2 = %.2f" % (a_w, b_w, fit))
    return fits


def plot_w_budget(cases, fits, path):
    """Median over cases on z / h_abl of the LES d(w_core**2)/dz and each fitted budget."""
    z_rel = np.arange(0.02, 1.6, 0.02)
    lhs, buoy_term, drag_term = [], [], []
    for c in cases:
        terms = w_budget_terms(c)
        for out, term in zip((lhs, buoy_term, drag_term), terms):
            out.append([at_height(c.z, term, z * c.h_abl) for z in z_rel])
    lhs, buoy_term, drag_term = (np.array(v) for v in (lhs, buoy_term, drag_term))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    les_style = dict(color=config.COLORS["les"], linestyle=config.LINESTYLES["reference"],
                     linewidth=config.LINEWIDTHS["reference"])
    axes[0].plot(np.nanmedian(lhs, axis=0), z_rel, label="LES", **les_style)
    for i, (a_w, b_w, fit) in enumerate(fits):
        model = a_w * buoy_term + b_w * drag_term
        label = "a_w = %.2f, b_w = %.2f (R$^2$ %.2f)" % (a_w, b_w, fit)
        axes[0].plot(np.nanmedian(model, axis=0), z_rel, color=config.run_color(i + 1),
                     label=label)
        axes[1].plot(np.nanmedian(model - lhs, axis=0), z_rel, color=config.run_color(i + 1))
    for ax in axes:
        guide_line(ax, 1.0)
        guide_line(ax, 0.0, vertical=True)
    axes[0].set_xlabel(r"$d(w_{core}^2)/dz$ [m/s$^2$]")
    axes[1].set_xlabel(r"$2 a_w B - 2 b_w \epsilon_{core} w^2 - d(w_{core}^2)/dz$ [m/s$^2$]")
    axes[0].set_ylabel("z / h_abl")
    axes[0].legend(fontsize=8)
    fig.suptitle("LES core w budget with $\\epsilon_{core}$, median over cases; fitted below "
                 "%g h_abl" % H_ABL_BUDGET)
    fig.savefig(path)
    plt.close(fig)


def print_base_check(cases):
    print("\nBase check at h0 = %g m: PlumeBase(a_w = %g, b_w = %g) against the LES"
          % (H0_PLUME, A_W, B_W))
    print("%-16s %8s | %7s %7s %7s | %7s %7s %7s"
          % ("case", "H", "dtheta", "core", "mean", "w0", "w_core", "w"))
    print("%-16s %8s | %7s %7s %7s | %7s %7s %7s"
          % ("", "[kW/m2]", "model", "LES", "LES", "model", "LES", "LES"))
    for c in cases:
        base = PlumeBase(c.H, 0.0, c.area_fire, c.environment(), a_w=A_W, b_w=B_W)
        print("%-16s %8.1f | %7.2f %7.2f %7.2f | %7.2f %7.2f %7.2f"
              % (c.name, c.H / 1e3, base.dtheta, at_height(c.z, c.theta_prime_core, H0_PLUME),
                 at_height(c.z, c.theta_prime, H0_PLUME), base.w0,
                 at_height(c.z, c.w_core, H0_PLUME), at_height(c.z, c.w, H0_PLUME)))


def plot_profiles(cases, path):
    """Rows: FLI. Columns: eps_core against the scalings, implied a, eta_core, theta'_core."""
    flis = fli_values(cases)
    fig, axes = plt.subplots(len(flis), 4, figsize=(16, 4.2 * len(flis)), sharey=True,
                             squeeze=False)
    scaling_style = dict(linestyle=config.LINESTYLES["scaling"],
                         linewidth=config.LINEWIDTHS["guide"])
    for row, fli in zip(axes, flis):
        sel = [c for c in cases if c.fli == fli]
        colors = config.sweep_colors(len(sel))
        for c, color in zip(sel, colors):
            zk = c.z / 1e3
            row[0].plot(c.core_entrainment * 1e3, zk, color=color, label="U = %g m/s" % c.U)
            row[0].plot(core_model_entrainment(c) * 1e3, zk, color=color, **scaling_style)
            row[1].plot(implied_a(c), zk, color=color)
            row[2].plot(c.core_efficiency, zk, color=color)
            row[3].plot(c.theta_prime_core, zk, color=color)
            for ax in row:
                ax.axhline(c.h_abl / 1e3, color=color, linestyle=config.LINESTYLES["zero"],
                           linewidth=config.LINEWIDTHS["guide"])
        guide_line(row[0], 1e3 / np.sqrt(sel[0].area_fire), vertical=True)
        guide_line(row[1], 0.0, vertical=True)
        guide_line(row[2], A_W - B_W * A_BUOYANT, vertical=True)
        guide_line(row[3], 0.0, vertical=True)
        row[0].set_ylabel("FLI = %g MW/m\nz [km]" % (fli / 1e6))
        row[0].set_xscale("symlog", linthresh=1.0)
        row[0].set_xlim(-10, 100)
        row[1].set_xlim(-1, 4)
        row[2].set_xlim(-1, 1.5)
        row[3].set_xlim(-3, 15)
        row[0].legend(loc="upper right", fontsize=8)
    labels = [r"$\epsilon_{core}$ (solid), default model $\epsilon$ (dashed) [1/km]",
              r"$a = \epsilon_{core} w_{core}^2 / B_{core}$",
              r"$\eta_{core} = d(w_{core}^2)/dz \,/\, 2B_{core}$",
              r"$\theta'_{core}$ [K]"]
    for ax, lab in zip(axes[-1], labels):
        ax.set_xlabel(lab)
    axes[0][0].set_ylim(0, 2.5)
    fig.suptitle("LES core (centreline of the 95th-percentile w, $w_{core} = w_{95}$); "
                 "grey dotted: Morton floor $1/\\sqrt{A_0}$, default model $a_w - b_w a$, 0;\n"
                 "coloured dotted: boundary-layer top")
    fig.savefig(path)
    plt.close(fig)


def main():
    config.apply()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cases = [les.LESCase(name) for name in les.case_names()]
    print_tables(cases)
    plot_w_budget(cases, fit_w_budget(cases), OUTPUT / "w_budget.png")
    print_base_check(cases)
    plot_profiles(cases, OUTPUT / "core_entrainment.png")
    print("\nFigures written to %s" % OUTPUT)


if __name__ == "__main__":
    main()
