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

"""Does the LES plume top follow from its core rather than from its mean?

The model's w equation is integrated upward through prescribed LES buoyancy, from the LES w
at Z_START, once with the mean plume (theta', top-hat w) and once with the core (theta' on
the centreline of the 95th-percentile w, and w_95). With the default model's entrainment
eps = eps_M + a max(B, 0)/w**2, two forms of the equation bound it:

    plume:           d(w**2)/dz = 2 (a_w - b_w a) B       - 2 b_w eps_M w**2   where B > 0
                     d(w**2)/dz = 2 a_w B                 - 2 b_w eps_M w**2   where B <= 0
    non-entraining:  d(w**2)/dz = 2 a_w B

Above the highest LES level with core or mean data, the air keeps its last theta, so it rises
unmixed through the environment. The top is where w**2 reaches zero.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import config
from plume_model.entrainment import A_BUOYANT, FAC_ENT_BUOYANT
from plume_model.plume import A_W, B_W
from plume_model.thermo import g

sys.path.insert(0, str(Path(__file__).parent))
import les
from les_dry_line_fire import fli_values, guide_line

OUTPUT = Path(__file__).parent / "output" / "les_dry_line_fire" / "core_buoyancy"

Z_START = 100.0
DZ = 5.0
Z_MAX = 6000.0


def buoyancy_profile(case, theta_prime, z):
    """B on z from an LES theta' profile, continued above its data at the last theta."""
    has = np.isfinite(theta_prime)
    theta_env = np.interp(z, case.z, case.theta_env)
    theta = np.interp(z, case.z[has], (case.theta_env + theta_prime)[has])
    above = z > case.z[has][-1]
    theta[above] = case.theta_env[has][-1] + theta_prime[has][-1]
    return g * (theta - theta_env) / theta_env


def rise(case, theta_prime, w_profile, a_w, b_w, a, eps_m):
    """w on a uniform grid from Z_START through the prescribed buoyancy, and its top."""
    z = np.arange(Z_START, Z_MAX, DZ)
    B = buoyancy_profile(case, theta_prime, z)
    has = np.isfinite(w_profile)
    w2 = np.full_like(z, np.nan)
    w2[0] = np.interp(Z_START, case.z[has], w_profile[has]) ** 2
    for k in range(1, len(z)):
        forcing = (a_w - b_w * a) * B[k] if B[k] > 0 else a_w * B[k]
        w2[k] = w2[k - 1] + 2 * (forcing - b_w * eps_m * w2[k - 1]) * DZ
        if w2[k] <= 0:
            return z[:k], np.sqrt(w2[:k]), z[k - 1]
    return z, np.sqrt(w2), z[-1]


def runs(case):
    """(label, which LES profiles, a_w, b_w, a, eps_M) for the four integrations."""
    eps_m = FAC_ENT_BUOYANT / np.sqrt(case.area_fire)
    return [
        ("core, plume w equation", "core", A_W, B_W, A_BUOYANT, eps_m),
        ("core, non-entraining", "core", A_W, 0.0, 0.0, 0.0),
        ("mean, plume w equation", "mean", A_W, B_W, A_BUOYANT, eps_m),
        ("mean, non-entraining", "mean", A_W, 0.0, 0.0, 0.0),
    ]


def profiles(case, which):
    if which == "core":
        return case.theta_prime_core, case.w_core
    return case.theta_prime, case.w


def main():
    config.apply()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cases = [les.LESCase(name) for name in les.case_names()]

    results = {}
    for case in cases:
        for label, which, a_w, b_w, a, eps_m in runs(case):
            theta_prime, w = profiles(case, which)
            results[(case.name, label)] = rise(case, theta_prime, w, a_w, b_w, a, eps_m)

    labels = [r[0] for r in runs(cases[0])]
    print("Plume top [m] from the w equation through prescribed LES buoyancy, from z = %g m"
          % Z_START)
    print("a_w = %g, b_w = %g, a = %g, fac_ent = %g\n"
          % (A_W, B_W, A_BUOYANT, FAC_ENT_BUOYANT))
    print("%-16s %13s " % ("case", "LES z_top") + "".join("%13s" % l.split(",")[0] + "" for l in labels))
    print("%-16s %13s " % ("", "") + "".join("%13s" % l.split(", ")[1][:12] for l in labels))
    errors = {l: [] for l in labels}
    for case in cases:
        tops = [results[(case.name, l)][2] for l in labels]
        for l, t in zip(labels, tops):
            errors[l].append(t / case.z_top)
        print("%-16s %6.0f +-%4.0f " % (case.name, case.z_top, case.z_top_std)
              + "".join("%13.0f" % t for t in tops))
    print("\nmodel top / LES top: median (min - max)")
    for l in labels:
        e = np.array(errors[l])
        print("%-26s %5.2f (%4.2f - %4.2f)" % (l, np.median(e), e.min(), e.max()))

    styles = {"core": config.LINESTYLES["envelope"], "mean": config.LINESTYLES["reference"]}
    colors = dict(zip(labels, [config.run_color(i) for i in range(len(labels))]))
    for U in (3.0, 9.0):
        sel = [c for c in cases if c.U == U]
        fig, axes = plt.subplots(len(sel), 2, figsize=(10, 3.9 * len(sel)), sharey=True,
                                 squeeze=False)
        for row, case in zip(axes, sel):
            zk = case.z / 1e3
            for which in ("mean", "core"):
                theta_prime, w = profiles(case, which)
                les_style = dict(color=config.COLORS["les"], linestyle=styles[which],
                                 linewidth=config.LINEWIDTHS["reference"])
                row[0].plot(w, zk, label="LES %s" % which, **les_style)
                row[1].plot(theta_prime, zk, label="LES %s" % which, **les_style)
            for label in labels:
                z, w, _ = results[(case.name, label)]
                row[0].plot(w, z / 1e3, color=colors[label], label=label)
            for ax in row:
                guide_line(ax, case.h_abl / 1e3)
                ax.axhline(case.z_top / 1e3, color=config.COLORS["les"],
                           linestyle=config.LINESTYLES["zero"],
                           linewidth=config.LINEWIDTHS["guide"])
            guide_line(row[1], 0.0, vertical=True)
            row[0].set_ylabel("%s\nz [km]" % case.name)
            row[1].set_xlim(-4, 12)
        axes[-1][0].set_xlabel("w [m/s]")
        axes[-1][1].set_xlabel(r"$\theta'$ [K]")
        axes[0][0].set_ylim(0, 4)
        axes[0][0].legend(fontsize=7)
        fig.suptitle("w equation through prescribed LES buoyancy, U = %g m/s\n"
                     "(grey dotted: boundary-layer top; blue dotted: LES plume top)" % U)
        fig.savefig(OUTPUT / ("w_through_les_buoyancy_U%g.png" % U))
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for i, fli in enumerate(fli_values(cases)):
        for label, marker in zip(labels, ["o", "s", "^", "v"]):
            sel = [c for c in cases if c.fli == fli]
            ax.plot([c.z_top / 1e3 for c in sel], [results[(c.name, label)][2] / 1e3 for c in sel],
                    color=colors[label], marker=marker, linestyle="none", alpha=0.8)
    lims = [0.5, 5.5]
    ax.plot(lims, lims, color=config.COLORS["env"], linestyle=config.LINESTYLES["zero"],
            linewidth=config.LINEWIDTHS["guide"])
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.set_xlabel("LES z_top [km]")
    ax.set_ylabel("top of the w equation through LES buoyancy [km]")
    ax.legend(handles=[Line2D([], [], color=colors[l], marker=m, linestyle="none", label=l)
                       for l, m in zip(labels, ["o", "s", "^", "v"])], fontsize=8)
    fig.savefig(OUTPUT / "plume_top.png")
    plt.close(fig)
    print("\nFigures written to %s" % OUTPUT)


if __name__ == "__main__":
    main()
