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

"""Dry LES of a line fire, in fire intensity and wind speed, as the plume model sees it.

The data is not in the repository; it is read from DATA, one NetCDF file per case with
vertical profiles every 30 s, plus sim_metadata.json with the fire and domain parameters.

Each LESCase is reduced to time-mean top-hat profiles over t >= T_START:

    A = <A_t>,   Q = <Q_t>,   w = Q / A,   theta' = <A_t theta'_t> / <A_t>

with Q_t the volume flux through the detected plume area A_t and theta'_t its conditional
mean. Undetected levels count as A_t = Q_t = 0, so these are fluxes and areas of the
time-mean plume, and the plume top is the time mean of each snapshot's highest detected level.
"""

import json
from pathlib import Path

import numpy as np
import xarray as xr

from plume_model import environment_from_theta
from plume_model.thermo import Rd, cp, exner, g, p0

DATA = Path(__file__).parent / "cases" / "dry_les_hu"

# Convective plume: tracer above 0.0025 and w above 1 m/s.
METHOD = "M2C_Tracer_0.0025_W_1_continuous_RootArea_1020m_Increase"
T_START = 8220.0

# Stand-in until the LES pressure profile is provided.
P_SURFACE = 1e5


def case_names():
    """The cases with a profile file, ordered by fire intensity, then wind speed."""
    meta = json.loads((DATA / "sim_metadata.json").read_text())
    names = [p.name.split("_all_methods")[0]
             for p in (DATA / "Vertical_Plume_Profs").glob("*_all_methods*.nc")]
    return sorted(names, key=lambda n: (meta[n]["FLI"], meta[n]["U"]))


def pressure(case, z, p_sfc=P_SURFACE):
    """Hydrostatic pressure at heights z from p_sfc at z[0], through the dry time-mean theta."""
    theta = np.interp(z, case.z, case.theta_env)
    pi = np.empty_like(z)
    pi[0] = exner(p_sfc)
    for k in range(1, len(z)):
        # Trapezoidal dpi/dz = -g / (cp theta); theta_v = theta in the dry LES.
        pi[k] = pi[k - 1] - 0.5 * g * (z[k] - z[k - 1]) / cp * (1 / theta[k] + 1 / theta[k - 1])
    return p0 * pi ** (cp / Rd)


def time_mean(values):
    """Mean over time of the finite values, NaN where no snapshot detected the plume."""
    out = np.full(values.shape[1], np.nan)
    has_data = np.isfinite(values).any(axis=0)
    out[has_data] = np.nanmean(values[:, has_data], axis=0)
    return out


def time_percentiles(values, q=(25, 75)):
    """Percentiles over time, NaN where no snapshot detected the plume."""
    out = np.full((len(q), values.shape[1]), np.nan)
    has_data = np.isfinite(values).any(axis=0)
    out[:, has_data] = np.nanpercentile(values[:, has_data], q, axis=0)
    return out


class LESCase:
    """One LES case: its fire, its time-mean environment and its time-mean plume."""

    def __init__(self, name, method=METHOD, t_start=T_START):
        meta = json.loads((DATA / "sim_metadata.json").read_text())[name]
        path = next((DATA / "Vertical_Plume_Profs").glob(name + "_all_methods*.nc"))
        ds = xr.open_dataset(path, decode_times=False).sel({"method": method})
        ds = ds.sel(time=slice(t_start, None))

        self.name = name
        self.fli = meta["FLI"] * 1e6
        self.U = meta["U"]
        self.H = meta["mean_fire_heat_flux"]
        self.area_fire = meta["fire_area"]
        self.theta_0 = meta["theta_0"]
        self.times = ds.time.values.astype(float)

        t_abl = np.array(meta["timestamps_ambient_ABL_structure_variables"], dtype=float)
        h_abl = np.array(meta["ambient_ABL_top_height"])
        self.h_abl = float(np.mean(h_abl[t_abl >= t_start]))

        self.z = ds.z.values.astype(float)
        self.theta_env = ds.theta_mean_profile_ref.mean("time").values
        self.u_env = ds.u_mean_profile_ref.mean("time").values
        self.v_env = ds.v_mean_profile_ref.mean("time").values

        area_t = np.nan_to_num(ds.plume_area.values)
        flux_t = np.nan_to_num(ds.volume_flux.values)
        detected = area_t > 0
        self.area_t = area_t
        self.detected = detected

        self.area = area_t.mean(axis=0)
        self.volume_flux = flux_t.mean(axis=0)
        inside = self.area > 0
        self.w = np.full_like(self.z, np.nan)
        self.w[inside] = self.volume_flux[inside] / self.area[inside]
        theta_prime_t = np.where(detected, ds.theta_prime_mean.values, 0.0)
        self.theta_prime = np.full_like(self.z, np.nan)
        self.theta_prime[inside] = ((area_t * theta_prime_t).sum(axis=0)[inside]
                                    / area_t.sum(axis=0)[inside])

        # Time-mean of the conditional upper percentiles, where the plume was detected.
        self.w_95 = time_mean(np.where(detected, ds.w_95th.values, np.nan))
        self.theta_prime_95 = time_mean(np.where(detected, ds.theta_95th.values, np.nan))

        # Every snapshot's centreline starts at x = 0 at its lowest detected level.
        x_t = np.where(detected, ds.weighted_w_centerline_x.values, np.nan)
        k_low = np.argmax(detected, axis=1)
        x_t = x_t - x_t[np.arange(len(x_t)), k_low][:, None]
        self.x = time_mean(x_t)

        self.w_t = np.where(detected, ds.w_mean.values, np.nan)
        self.theta_prime_t = np.where(detected, ds.theta_prime_mean.values, np.nan)
        self.radius_t = np.where(detected, np.sqrt(area_t / np.pi), np.nan)
        self.x_t = x_t

        k_top_t = np.array([np.nonzero(d)[0][-1] if d.any() else -1 for d in detected])
        z_top_t = np.where(k_top_t >= 0, self.z[k_top_t], np.nan)
        self.z_top = float(np.nanmean(z_top_t))
        self.z_top_std = float(np.nanstd(z_top_t))

        # Net fractional entrainment of the time-mean plume, d ln Q / dz on the LES grid.
        self.net_entrainment = np.full_like(self.z, np.nan)
        self.net_entrainment[inside] = np.gradient(np.log(self.volume_flux[inside]),
                                                   self.z[inside])

    def band(self, name, q=(25, 75)):
        """Spread over time of a per-snapshot profile: w, theta_prime, radius, x."""
        return time_percentiles(getattr(self, name + "_t"), q)

    def environment(self):
        """The time-mean environment, dry, extended from the lowest LES level down to z = 0."""
        z = np.concatenate([[0.0], self.z])
        p = pressure(self, z)
        theta = np.concatenate([self.theta_env[:1], self.theta_env])
        u = np.concatenate([self.u_env[:1], self.u_env])
        v = np.concatenate([self.v_env[:1], self.v_env])
        return environment_from_theta(z, theta, np.zeros_like(z), p, u, v)
