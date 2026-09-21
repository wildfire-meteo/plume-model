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

"""Plume rise through a well-mixed layer capped by a sharp inversion.

Sweeps the sensible heat flux H over 1-100 kW/m2 and the fire area over 0.1-10 km2, at
LE = 0, and plots the environment, the plume-top regime diagram, profiles along both sweep
directions, and the w**2 budget.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import config
from plume_model import MortonEntrainment, Plume, PlumeBase
from plume_model.thermo import qsat

sys.path.insert(0, str(Path(__file__).parent))
import environments

OUTPUT = Path(__file__).parent / "output" / "mixed_layer_inversion"

H_VALUES = np.logspace(3, 5, 11)
AREA_VALUES = np.logspace(5, 7, 11)
LE = 0.0

# Every other sweep step, for the profile figures.
PROFILE_STEPS = [0, 2, 4, 6, 8, 10]
I_H_FIXED = 5      # H = 10 kW/m2
J_AREA_FIXED = 5   # area = 1 km2


def run_sweep(env):
    plumes = []
    for H in H_VALUES:
        row = []
        for area in AREA_VALUES:
            base = PlumeBase(H, LE, area, env)
            row.append(Plume(env, base, MortonEntrainment()).ascend())
        plumes.append(row)
    return plumes


def print_summary(env, plumes):
    print("Inversion at h = %.0f m; humidity jump set by RH above: dq = %.2f g/kg"
          % (env.h, env.dq_inversion * 1e3))
    print("\nPlume top z_top [m]; * condensed, ^ reached the domain top without stopping")
    print("H [kW/m2] \\ area [km2] " + "".join("%8.2f" % (a / 1e6) for a in AREA_VALUES))
    for i, H in enumerate(H_VALUES):
        cells = []
        for plume in plumes[i]:
            flag = "^" if not plume.stopped else ("*" if plume.k_lcl >= 0 else " ")
            cells.append("%7.0f%s" % (plume.z_top, flag))
        print("%22.2f  " % (H / 1e3) + "".join(cells))


def guide_line(ax, z, vertical=False):
    style = dict(color=config.COLORS["env"], linestyle=config.LINESTYLES["zero"],
                 linewidth=config.LINEWIDTHS["guide"])
    if vertical:
        ax.axvline(z, **style)
    else:
        ax.axhline(z, **style)


def plot_environment(env, path):
    zk = env.z / 1e3
    rh = env.qt / qsat(env.T, env.p)

    fig, axes = plt.subplots(1, 3, figsize=(10, 4.5), sharey=True)

    ax = axes[0]
    ax.plot(env.theta, zk, color=config.COLORS["env"], label=r"$\theta$")
    ax.plot(env.thetav, zk, color=config.COLORS["plume"], label=r"$\theta_v$")
    ax.set_xlabel("K")
    ax.set_ylabel("z [km]")
    ax.legend()

    ax = axes[1]
    ax.plot(env.qt * 1e3, zk, color=config.COLORS["env"], label=r"$q_t$")
    ax.plot(qsat(env.T, env.p) * 1e3, zk, color=config.COLORS["env"],
            linestyle=config.LINESTYLES["reference"], label=r"$q_{sat}$")
    ax.set_xlabel("g/kg")
    ax.legend()

    ax = axes[2]
    ax.plot(100 * rh, zk, color=config.COLORS["env"])
    ax.set_xlabel("RH [%]")
    ax.set_xlim(0, 100)

    for ax in axes:
        guide_line(ax, env.h / 1e3)
    fig.suptitle("Environment: mixed layer to h = %.1f km, then dT/dz fixed at fixed RH"
                 % (env.h / 1e3))
    fig.savefig(path)
    plt.close(fig)


def plot_regime(env, plumes, path):
    z_top = np.array([[p.z_top for p in row] for row in plumes]) / 1e3
    w_max = np.array([[np.nanmax(p.w) for p in row] for row in plumes])
    condensed = np.array([[p.k_lcl >= 0 for p in row] for row in plumes])
    reached_top = np.array([[not p.stopped for p in row] for row in plumes])

    A_km2 = AREA_VALUES / 1e6
    H_kw = H_VALUES / 1e3
    AA, HH = np.meshgrid(A_km2, H_kw)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, field, label in [(axes[0], z_top, "plume top z_top [km]"),
                             (axes[1], w_max, "max w [m/s]")]:
        mesh = ax.pcolormesh(AA, HH, field, cmap=config.FIELD_CMAP, shading="nearest")
        fig.colorbar(mesh, ax=ax, label=label)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("fire area [km$^2$]")
        ax.grid(False)
        ax.scatter(AA[condensed & ~reached_top], HH[condensed & ~reached_top],
                   marker=config.MARKERS["condensed"], facecolors="none",
                   edgecolors="white", s=30, label="condensed")
        ax.scatter(AA[reached_top], HH[reached_top], marker=config.MARKERS["reached_top"],
                   color="white", s=30, label="reached domain top")
    axes[0].set_ylabel("H [kW/m$^2$]")
    axes[1].legend(loc="lower right", frameon=True)
    fig.suptitle("LE = 0, inversion at h = %.1f km" % (env.h / 1e3))
    fig.savefig(path)
    plt.close(fig)


def plot_profiles(env, plumes_and_labels, title, path):
    colors = config.sweep_colors(len(plumes_and_labels))
    fig, axes = plt.subplots(2, 4, figsize=(15, 8), sharey=True)
    ax = axes.ravel()

    for (plume, label), color in zip(plumes_and_labels, colors):
        zk = plume.z / 1e3
        style = dict(color=color)
        ax[0].plot(plume.w, zk, label=label, **style)
        if plume.k_lcl >= 0:
            ax[0].plot(plume.w[plume.k_lcl], zk[plume.k_lcl], marker=config.MARKERS["lcl"],
                       color=color, linestyle="none")
        ax[1].plot(np.sqrt(plume.area / np.pi), zk, **style)
        ax[2].plot(plume.thetav - plume.thetav_env, zk, **style)
        ax[3].plot(plume.qt * 1e3, zk, **style)
        ax[4].plot(plume.mass_flux, zk, **style)
        ax[5].plot(plume.eps * 1e3, zk, **style)
        ax[6].plot((plume.delta + plume.delta_dyn) * 1e3, zk, **style)
        ax[7].plot(plume.x / 1e3, zk, **style)

    zk_env = env.z / 1e3
    ax[3].plot(env.qt * 1e3, zk_env, color=config.COLORS["env"],
               linewidth=config.LINEWIDTHS["guide"], label="environment")
    guide_line(ax[2], 0.0, vertical=True)
    for a in ax:
        guide_line(a, env.h / 1e3)

    labels = ["w [m/s]", "radius $\\sqrt{A/\\pi}$ [m]", r"$\theta_{v,p}-\theta_{v,e}$ [K]",
              "$q_t$ [g/kg]", "mass flux M [kg/s]", r"$\epsilon$ [1/km]",
              r"$\delta + \delta_{dyn}$ [1/km]", "x [km]"]
    for a, lab in zip(ax, labels):
        a.set_xlabel(lab)
    ax[4].set_xscale("log")
    ax[6].set_xscale("log")
    ax[0].set_ylabel("z [km]")
    ax[4].set_ylabel("z [km]")
    ax[0].legend()
    ax[3].legend()
    fig.suptitle(title + "  (o: lifting condensation level of the plume)")
    fig.savefig(path)
    plt.close(fig)


def plot_w_budget(env, plumes_and_labels, path):
    fig, axes = plt.subplots(1, len(plumes_and_labels), figsize=(4 * len(plumes_and_labels), 4.5),
                             sharey=True)
    for ax, (plume, label) in zip(axes, plumes_and_labels):
        zk = plume.z / 1e3
        ax.plot(plume.dw2dz_buoy, zk, color=config.COLORS["buoy"], label=r"$2 a_w B$")
        ax.plot(plume.dw2dz_drag, zk, color=config.COLORS["drag"],
                label=r"$-2 b_w \epsilon w^2$")
        ax.plot(plume.dw2dz_buoy + plume.dw2dz_drag, zk, color=config.COLORS["total"],
                linewidth=config.LINEWIDTHS["guide"], label=r"$d(w^2)/dz$")
        guide_line(ax, env.h / 1e3)
        guide_line(ax, 0.0, vertical=True)
        ax.set_title(label)
        ax.set_xlabel(r"m s$^{-2}$")
    axes[0].set_ylabel("z [km]")
    axes[0].legend()
    fig.suptitle(r"$w^2$ budget, area = %.0f km$^2$" % (AREA_VALUES[J_AREA_FIXED] / 1e6))
    fig.savefig(path)
    plt.close(fig)


def main():
    config.apply()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    env = environments.mixed_layer_inversion()
    plumes = run_sweep(env)
    print_summary(env, plumes)

    plot_environment(env, OUTPUT / "environment.png")
    plot_regime(env, plumes, OUTPUT / "regime.png")

    along_H = [(plumes[i][J_AREA_FIXED], "H = %.3g kW/m$^2$" % (H_VALUES[i] / 1e3))
               for i in PROFILE_STEPS]
    plot_profiles(env, along_H, "Varying H at area = %.0f km$^2$"
                  % (AREA_VALUES[J_AREA_FIXED] / 1e6), OUTPUT / "profiles_vs_H.png")

    along_area = [(plumes[I_H_FIXED][j], "A = %.3g km$^2$" % (AREA_VALUES[j] / 1e6))
                  for j in PROFILE_STEPS]
    plot_profiles(env, along_area, "Varying area at H = %.0f kW/m$^2$"
                  % (H_VALUES[I_H_FIXED] / 1e3), OUTPUT / "profiles_vs_area.png")

    budget_runs = [(plumes[i][J_AREA_FIXED], "H = %.0f kW/m$^2$" % (H_VALUES[i] / 1e3))
                   for i in (0, 5, 10)]
    plot_w_budget(env, budget_runs, OUTPUT / "w_budget.png")

    print("\nFigures written to %s" % OUTPUT)


if __name__ == "__main__":
    main()
