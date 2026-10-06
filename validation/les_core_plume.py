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

"""The default model against the LES plume core, next to the reference implementation's.

The default model has PlumeBase, d(w**2)/dz = 2 (a_w B - b_w eps w**2), Morton and dynamic
detrainment, and BuoyantEntrainment:

    eps = fac_ent / sqrt(A_0) + a max(B, 0) / w**2.

It is also run with each coefficient moved in SENSITIVITY, and at the (a_w, b_w) pairs in
W_EQUATION_PAIRS. Its targets are the LES core (les.py): theta' on the centreline of the 95th-percentile w, w_95, the
displacement and u of that centreline, and the plume top. Mass flux and area are carried,
but have no LES core counterpart.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import config
from plume_model import BuoyantEntrainment, MortonEntrainment
from plume_model.entrainment import A_BUOYANT, FAC_ENT_BUOYANT
from plume_model.plume import A_W, B_W

sys.path.insert(0, str(Path(__file__).parent))
import les
from crosscheck_js import REFERENCE_A_W, REFERENCE_B_W
from les_dry_line_fire import at_height, guide_line, plot_sweep_plume_top, run_model

OUTPUT = Path(__file__).parent / "output" / "les_dry_line_fire" / "core_plume"

# The buoyant term makes eps*dz ~ 0.3 at h0 for dz = 10 m.
DZ = 2.5
W_EQUATION_PAIRS = [(0.5, 0.5), (0.38, 0.2)]

# Each coefficient of the default model moved by about +-20%, one at a time.
SENSITIVITY = [("a", 0.25), ("a", 0.35), ("fac_ent", 0.3), ("fac_ent", 0.5),
               ("a_w", 0.4), ("a_w", 0.6), ("b_w", 0.4), ("b_w", 0.6)]

Z_TABLE = (20.0, 50.0, 100.0, 250.0, 500.0, 800.0)
FLI_PROFILES = 10e6
PROFILE_WINDS = [3.0, 9.0]


def run_core(case, a=A_BUOYANT, fac_ent=FAC_ENT_BUOYANT, a_w=A_W, b_w=B_W, dz=DZ,
             ventilated=False):
    return run_model(case, BuoyantEntrainment(a, fac_ent=fac_ent), dz, a_w=a_w, b_w=b_w,
                     ventilated=ventilated)


def run_original(case, dz=DZ):
    return run_model(case, MortonEntrainment(), dz, a_w=REFERENCE_A_W, b_w=REFERENCE_B_W)


def core_pairs(case, plume):
    """(name, model profile, LES core profile) for the core targets."""
    return [
        ("theta'", plume.thetav - plume.thetav_env, case.theta_prime_core),
        ("w", plume.w, case.w_core),
        ("x", plume.x, case.x_core),
        ("u", plume.u, case.u_core),
    ]


def print_summary(cases, runs):
    """runs: (label, plumes) pairs."""
    low = np.array([c.U <= 3 for c in cases])
    high = np.array([c.U >= 9 for c in cases])
    print("Against the LES core, dz = %g m. Default model: fac_ent = %g, a = %g, a_w = %g, b_w = %g"
          % (DZ, FAC_ENT_BUOYANT, A_BUOYANT, A_W, B_W))
    print("\nz_top: model / LES, median (min - max), and median by wind")
    print("%-24s %20s %8s %8s %8s" % ("", "all", "U <= 3", "U >= 9", "rms [m]"))
    for label, plumes in runs:
        r = np.array([p.z_top / c.z_top for c, p in zip(cases, plumes)])
        err = np.array([p.z_top - c.z_top for c, p in zip(cases, plumes)])
        print("%-24s %6.2f (%4.2f - %4.2f) %8.2f %8.2f %8.0f"
              % (label, np.median(r), r.min(), r.max(), np.median(r[low]), np.median(r[high]),
                 np.sqrt(np.mean(err**2))))

    print("\nmodel / LES core, median over cases with both plumes at that height")
    for k, (name, _, _) in enumerate(core_pairs(cases[0], runs[0][1][0])):
        print("\n%-24s" % name + "".join("%7.0f" % z for z in Z_TABLE))
        for label, plumes in runs:
            row = []
            for z_at in Z_TABLE:
                ratios = []
                for c, p in zip(cases, plumes):
                    _, model, ref = core_pairs(c, p)[k]
                    ratios.append(at_height(p.z, model, z_at) / at_height(c.z, ref, z_at))
                row.append(np.nanmedian(ratios))
            print("%-24s" % label + "".join("%7.2f" % v for v in row))


def plot_core_profiles(cases, runs, U, path):
    """FLI_PROFILES at wind U: theta', w, x, u, radius against the LES core."""
    case_k = [k for k, c in enumerate(cases) if c.fli == FLI_PROFILES and c.U == U][0]
    case = cases[case_k]
    fig, axes = plt.subplots(1, 5, figsize=(17, 4.6), sharey=True)
    les_style = dict(color=config.COLORS["les"], linestyle=config.LINESTYLES["reference"],
                     linewidth=config.LINEWIDTHS["reference"])
    les_mean_style = dict(color=config.COLORS["les"], linestyle=config.LINESTYLES["zero"],
                          linewidth=config.LINEWIDTHS["guide"])
    zk = case.z / 1e3
    for ax, core, mean in zip(axes, [case.theta_prime_core, case.w_core, case.x_core / 1e3,
                                     case.u_core],
                              [case.theta_prime, case.w, case.x / 1e3, None]):
        ax.plot(core, zk, label="LES core", **les_style)
        if mean is not None:
            ax.plot(mean, zk, label="LES mean plume", **les_mean_style)
    axes[3].plot(case.u_env, zk, color=config.COLORS["env"], linestyle=config.LINESTYLES["env"],
                 linewidth=config.LINEWIDTHS["guide"], label="environment")
    axes[4].plot(np.sqrt(case.area / np.pi), zk, label="LES mean plume", **les_mean_style)
    for i, (label, plumes) in enumerate(runs):
        p = plumes[case_k]
        zm = p.z / 1e3
        for ax, profile in zip(axes, [p.thetav - p.thetav_env, p.w, p.x / 1e3, p.u,
                                      np.sqrt(p.area / np.pi)]):
            ax.plot(profile, zm, color=config.run_color(i + 1), label=label)
    for ax in axes:
        guide_line(ax, case.h_abl / 1e3)
        ax.axhline(case.z_top / 1e3, color=config.COLORS["les"],
                   linestyle=config.LINESTYLES["zero"], linewidth=config.LINEWIDTHS["guide"])
    guide_line(axes[0], 0.0, vertical=True)
    axes[0].set_xlim(-3, 30)
    axes[4].set_xscale("log")
    labels = [r"$\theta'$ [K]", "w [m/s]", "x [km]", "u [m/s]",
              r"radius $\sqrt{A/\pi}$ [m] (no LES core)"]
    for ax, lab in zip(axes, labels):
        ax.set_xlabel(lab)
    axes[0].set_ylabel("z [km]")
    axes[0].set_ylim(0, 3)
    axes[0].legend(fontsize=7)
    axes[3].legend(fontsize=7, handles=axes[3].get_legend_handles_labels()[0][:2])
    fig.suptitle("%s  (grey dotted: boundary-layer top; blue dotted: LES plume top)" % case.name)
    fig.savefig(path)
    plt.close(fig)


def main():
    config.apply()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cases = [les.LESCase(name) for name in les.case_names()]

    runs = [("reference", [run_original(c) for c in cases]),
            ("default model", [run_core(c) for c in cases]),
            ("ventilated base", [run_core(c, ventilated=True) for c in cases])]
    print_summary(cases, runs)

    sensitivity = [("default model", runs[1][1])]
    sensitivity += [("%s = %g" % (name, value), [run_core(c, **{name: value}) for c in cases])
                    for name, value in SENSITIVITY]
    sensitivity.append(("default model, dz = 10 m", [run_core(c, dz=10.0) for c in cases]))
    print("\n\nSensitivity of the default model, one coefficient at a time")
    print_summary(cases, sensitivity)

    plot_sweep_plume_top(cases, runs, "Reference and default model", OUTPUT / "plume_top.png")
    for U in PROFILE_WINDS:
        plot_core_profiles(cases, runs, U, OUTPUT / ("profiles_U%g.png" % U))

    pairs = [("a_w = %g, b_w = %g" % (a_w, b_w), [run_core(c, a_w=a_w, b_w=b_w) for c in cases])
             for a_w, b_w in W_EQUATION_PAIRS]
    print("\n\nDefault model at other (a_w, b_w)")
    print_summary(cases, pairs)
    plot_sweep_plume_top(cases, pairs, "Default model", OUTPUT / "plume_top_w_equation.png")
    for U in PROFILE_WINDS:
        plot_core_profiles(cases, pairs, U, OUTPUT / ("profiles_w_equation_U%g.png" % U))
    print("\nFigures written to %s" % OUTPUT)


if __name__ == "__main__":
    main()
