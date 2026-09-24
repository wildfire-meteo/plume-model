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

"""The plume model against a dry LES sweep of a line fire in fire intensity and wind speed.

Each case runs the model from its time-mean LES environment, with the mean fire heat flux
as H, LE = 0 and the fire area as A_0, and compares plume top, top-hat w, radius, theta',
volume flux, net entrainment and trajectory with the time-mean LES convective plume (les.py).
Each plume top is on its own definition: the LES highest detected level (w > 1 m/s and
tracer), the model's first level with w < 1e-6.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import config
from plume_model import MortonEntrainment, Plume, PlumeBase

sys.path.insert(0, str(Path(__file__).parent))
import les

OUTPUT = Path(__file__).parent / "output" / "les_dry_line_fire"

FLI_PROFILES = 10e6


def run_model(case):
    env = case.environment()
    base = PlumeBase(case.H, 0.0, case.area_fire, env)
    return Plume(env, base, MortonEntrainment()).ascend()


def guide_line(ax, value, vertical=False):
    style = dict(color=config.COLORS["env"], linestyle=config.LINESTYLES["zero"],
                 linewidth=config.LINEWIDTHS["guide"])
    if vertical:
        ax.axvline(value, **style)
    else:
        ax.axhline(value, **style)


def fli_values(cases):
    return sorted(set(c.fli for c in cases))


def print_summary(cases, plumes):
    print("Time mean over t >= %.0f s, LES method %s" % (les.T_START, les.METHOD))
    print("\n%-16s %8s %6s | %13s %7s | %7s %7s | %8s %8s"
          % ("case", "H", "U", "LES z_top", "max w", "model", "max w", "dz_top", "dz_top"))
    print("%-16s %8s %6s | %13s %7s | %7s %7s | %8s %8s"
          % ("", "[kW/m2]", "[m/s]", "[m]", "[m/s]", "z_top", "[m/s]", "[m]", "[%]"))
    for case, plume in zip(cases, plumes):
        dz = plume.z_top - case.z_top
        print("%-16s %8.1f %6.1f | %7.0f +-%4.0f %7.1f | %7.0f %7.1f | %+8.0f %+8.0f"
              % (case.name, case.H / 1e3, case.U, case.z_top, case.z_top_std,
                 np.nanmax(case.w), plume.z_top, np.nanmax(plume.w), dz,
                 100 * dz / case.z_top))
    dz = np.array([p.z_top - c.z_top for c, p in zip(cases, plumes)])
    print("\nz_top error over all cases: mean %+.0f m, rms %.0f m" % (dz.mean(),
                                                                     np.sqrt(np.mean(dz**2))))


def plot_environment(cases, path):
    winds = sorted(set(c.U for c in cases))
    colors = dict(zip(winds, config.sweep_colors(len(winds))))

    fig, axes = plt.subplots(1, 3, figsize=(10, 4.5), sharey=True)
    for case in cases:
        zk = case.z / 1e3
        color = colors[case.U]
        axes[0].plot(case.theta_env, zk, color=color)
        axes[0].plot(np.interp(case.h_abl, case.z, case.theta_env), case.h_abl / 1e3,
                     marker=config.MARKERS["case"], color=color, linestyle="none")
        axes[1].plot(case.u_env, zk, color=color)
        axes[2].plot(case.v_env, zk, color=color)

    axes[0].set_xlabel(r"$\theta$ [K]")
    axes[1].set_xlabel("u [m/s]")
    axes[2].set_xlabel("v [m/s]")
    axes[0].set_ylabel("z [km]")
    axes[0].set_ylim(0, 4)
    axes[0].set_xlim(300, 312)
    handles = [Line2D([], [], color=colors[U], label="U = %g m/s" % U) for U in winds]
    handles.append(Line2D([], [], color=config.COLORS["env"], marker=config.MARKERS["case"],
                          linestyle="none", label="boundary-layer top"))
    axes[0].legend(handles=handles, loc="upper left")
    fig.suptitle("Time-mean LES environment, all %d cases" % len(cases))
    fig.savefig(path)
    plt.close(fig)


def plot_plume_top(cases, plumes, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for i, fli in enumerate(fli_values(cases)):
        pairs = [(c, p) for c, p in zip(cases, plumes) if c.fli == fli]
        U = [c.U for c, _ in pairs]
        color = config.run_color(i)
        label = "FLI = %g MW/m" % (fli / 1e6)

        axes[0].errorbar(U, [c.z_top / 1e3 for c, _ in pairs],
                         yerr=[c.z_top_std / 1e3 for c, _ in pairs], color=color,
                         linestyle=config.LINESTYLES["reference"], marker=config.MARKERS["case"],
                         capsize=3)
        axes[0].plot(U, [p.z_top / 1e3 for _, p in pairs], color=color,
                     linestyle=config.LINESTYLES["model"], label=label)

        axes[1].errorbar([c.z_top / 1e3 for c, _ in pairs], [p.z_top / 1e3 for _, p in pairs],
                         xerr=[c.z_top_std / 1e3 for c, _ in pairs], color=color,
                         marker=config.MARKERS["case"], linestyle="none", capsize=3,
                         label=label)

    lims = [1.0, 3.3]
    axes[1].plot(lims, lims, color=config.COLORS["env"], linestyle=config.LINESTYLES["zero"],
                 linewidth=config.LINEWIDTHS["guide"])
    axes[1].set_xlim(lims)
    axes[1].set_ylim(lims)
    axes[1].set_aspect("equal")

    axes[0].set_xlabel("U [m/s]")
    axes[0].set_ylabel("plume top z_top [km]")
    axes[1].set_xlabel("LES z_top [km]")
    axes[1].set_ylabel("model z_top [km]")

    style_handles = [
        Line2D([], [], color=config.COLORS["env"], linestyle=config.LINESTYLES["model"],
               label="model"),
        Line2D([], [], color=config.COLORS["env"], linestyle=config.LINESTYLES["reference"],
               marker=config.MARKERS["case"], label="LES, $\\pm$1 std over time"),
    ]
    axes[1].legend(handles=axes[0].get_legend_handles_labels()[0] + style_handles,
                   loc="lower right")
    fig.suptitle("Plume top against wind speed")
    fig.savefig(path)
    plt.close(fig)


def w_ratio(case, plume, les_w):
    """Model w over an LES w statistic, on the model grid, where both plumes are present."""
    has_les = np.isfinite(les_w)
    z_les_top = case.z[has_les][-1]
    les_on_model = np.interp(plume.z, case.z[has_les], les_w[has_les])
    both = (plume.z <= min(plume.z_top, z_les_top)) & (plume.z >= case.z[has_les][0])
    return plume.z[both], plume.w[both] / les_on_model[both]


def plot_w_statistic(cases, plumes, path):
    """Model w against two LES statistics of w: the top-hat mean and the 95th percentile."""
    statistics = [("w", "top-hat mean", config.LINESTYLES["reference"]),
                  ("w_95", "95th percentile", config.LINESTYLES["envelope"])]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for i, fli in enumerate(fli_values(cases)):
        color = config.run_color(i)
        pairs = [(c, p) for c, p in zip(cases, plumes) if c.fli == fli]
        for ax, (name, _, _) in zip(axes[:2], statistics):
            for c, p in pairs:
                z, ratio = w_ratio(c, p, getattr(c, name))
                ax.plot(ratio, z / 1e3, color=color, linewidth=config.LINEWIDTHS["guide"])
        for name, label, linestyle in statistics:
            mean_ratio = [np.mean(w_ratio(c, p, getattr(c, name))[1]) for c, p in pairs]
            axes[2].plot([c.U for c, _ in pairs], mean_ratio, color=color, linestyle=linestyle,
                         marker=config.MARKERS["case"])

    for ax, (_, label, _) in zip(axes[:2], statistics):
        guide_line(ax, 1.0, vertical=True)
        ax.set_xlabel("model w / LES %s w" % label)
        ax.set_xscale("log")
        ax.set_xlim(0.3, 6)
        ax.set_ylabel("z [km]")
    guide_line(axes[2], 1.0)
    axes[2].set_yscale("log")
    axes[2].set_ylim(0.5, 6)
    axes[2].set_xlabel("U [m/s]")
    axes[2].set_ylabel("height-mean of model w / LES w")

    handles = [Line2D([], [], color=config.run_color(i), label="FLI = %g MW/m" % (fli / 1e6))
               for i, fli in enumerate(fli_values(cases))]
    handles += [Line2D([], [], color=config.COLORS["env"], linestyle=linestyle,
                       marker=config.MARKERS["case"], label="LES " + label)
                for _, label, linestyle in statistics]
    axes[2].legend(handles=handles)
    fig.suptitle("Model w against the LES plume's top-hat mean and 95th-percentile w, "
                 "up to the lower of the two plume tops")
    fig.savefig(path)
    plt.close(fig)


def plot_profiles(runs, title, path):
    """runs: (LESCase, Plume, label) triples, drawn in one colour each."""
    colors = config.sweep_colors(len(runs))
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5), sharey=True)
    ax = axes.ravel()
    ref = dict(linestyle=config.LINESTYLES["reference"],
               linewidth=config.LINEWIDTHS["reference"])
    envelope = dict(linestyle=config.LINESTYLES["envelope"],
                    linewidth=config.LINEWIDTHS["envelope"])

    for (case, plume, label), color in zip(runs, colors):
        zk_les = case.z / 1e3
        zk = plume.z / 1e3
        les_profiles = [
            (case.w, case.band("w")),
            (np.sqrt(case.area / np.pi), case.band("radius")),
            (case.theta_prime, case.band("theta_prime")),
            (case.volume_flux, None),
            (case.net_entrainment * 1e3, None),
            (case.x / 1e3, case.band("x") / 1e3),
        ]
        model_profiles = [
            plume.w,
            np.sqrt(plume.area / np.pi),
            plume.thetav - plume.thetav_env,
            plume.area * plume.w,
            (plume.eps - plume.delta - plume.delta_dyn) * 1e3,
            plume.x / 1e3,
        ]
        for a, (mean, band), model in zip(ax, les_profiles, model_profiles):
            if band is not None:
                a.fill_betweenx(zk_les, band[0], band[1], color=color, alpha=config.BAND_ALPHA,
                                linewidth=0)
            a.plot(mean, zk_les, color=color, **ref)
            a.plot(model, zk, color=color, label=label)
        ax[0].plot(case.w_95, zk_les, color=color, **envelope)
        ax[2].plot(case.theta_prime_95, zk_les, color=color, **envelope)

    for a in ax:
        guide_line(a, runs[0][0].h_abl / 1e3)
    guide_line(ax[2], 0.0, vertical=True)
    guide_line(ax[4], 0.0, vertical=True)

    labels = ["top-hat w [m/s]", r"radius $\sqrt{A/\pi}$ [m]", r"$\theta'$ [K]",
              r"volume flux $Q = A w$ [m$^3$/s]", r"net entrainment $d\ln Q/dz$ [1/km]",
              "x [km]"]
    for a, lab in zip(ax, labels):
        a.set_xlabel(lab)
    ax[3].set_xscale("log")
    ax[2].set_xlim(-3, 20)
    ax[4].set_xlim(-6, 6)
    ax[0].set_ylabel("z [km]")
    ax[3].set_ylabel("z [km]")
    ax[0].set_ylim(0, 3.5)

    grey = config.COLORS["env"]
    style_handles = [
        Line2D([], [], color=grey, linestyle=config.LINESTYLES["model"], label="model"),
        Line2D([], [], color=grey, label="LES time mean", **ref),
        Patch(color=grey, alpha=config.BAND_ALPHA * 2, label="LES 25-75% over time"),
        Line2D([], [], color=grey, label="LES 95th percentile", **envelope),
    ]
    ax[0].legend()
    ax[1].legend(handles=style_handles)
    fig.suptitle(title + "  (dotted: mean boundary-layer top)")
    fig.savefig(path)
    plt.close(fig)


def main():
    config.apply()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    cases = [les.LESCase(name) for name in les.case_names()]
    plumes = [run_model(case) for case in cases]
    print_summary(cases, plumes)

    plot_environment(cases, OUTPUT / "environment.png")
    plot_plume_top(cases, plumes, OUTPUT / "plume_top.png")
    plot_w_statistic(cases, plumes, OUTPUT / "w_statistic.png")

    along_U = [(c, p, "U = %g m/s" % c.U) for c, p in zip(cases, plumes)
               if c.fli == FLI_PROFILES]
    plot_profiles(along_U, "Varying U at FLI = %g MW/m" % (FLI_PROFILES / 1e6),
                  OUTPUT / "profiles_vs_U.png")

    winds_all_fli = sorted(set.intersection(*[set(c.U for c in cases if c.fli == fli)
                                              for fli in fli_values(cases)]))
    for U in winds_all_fli:
        along_fli = [(c, p, "FLI = %g MW/m" % (c.fli / 1e6)) for c, p in zip(cases, plumes)
                     if c.U == U]
        plot_profiles(along_fli, "Varying FLI at U = %g m/s" % U,
                      OUTPUT / ("profiles_vs_FLI_U%g.png" % U))

    print("\nFigures written to %s" % OUTPUT)


if __name__ == "__main__":
    main()
