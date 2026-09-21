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

"""Idealised environments, built in the form the model takes: (z, T, Td, p, u, v)."""

import numpy as np

from plume_model import Environment
from plume_model.thermo import Rd, dewpoint, exner, g, qsat


def hydrostatic_pressure(z, theta, qt, p_sfc=1e5):
    """Integrate dp/dz = -rho g upward, with rho from the local theta_v."""
    p = np.empty_like(z)
    p[0] = p_sfc
    for k in range(1, len(z)):
        thetav = theta[k - 1] * (1 + 0.61 * qt[k - 1])
        rho = p[k - 1] / (Rd * exner(p[k - 1]) * thetav)
        p[k] = p[k - 1] - rho * g * (z[k] - z[k - 1])
    return p


def build(z, theta, rh, u=None, v=None, p_sfc=1e5, n_iter=3):
    """Environment from (theta, RH); pressure and humidity are solved for together."""
    qt = np.zeros_like(z)
    for _ in range(n_iter):
        p = hydrostatic_pressure(z, theta, qt, p_sfc)
        T = theta * exner(p)
        qt = rh * qsat(T, p)
    return Environment(z, T, dewpoint(qt, p), p, u, v)


def dry_neutral(z_top=6000.0, n=61, theta_sfc=300.0, rh=0.3):
    z = np.linspace(0.0, z_top, n)
    return build(z, np.full_like(z, theta_sfc), np.full_like(z, rh))


def stable(z_top=6000.0, n=61, theta_sfc=300.0, gamma=4e-3, rh=0.2):
    z = np.linspace(0.0, z_top, n)
    return build(z, theta_sfc + gamma * z, np.full_like(z, rh))


def inversion(z_top=6000.0, n=61, theta_sfc=298.0, z_inv=1000.0,
              dtheta_inv=6.0, gamma_ft=5e-3, rh_bl=0.7, rh_ft=0.2):
    """Well-mixed layer under a step inversion, moist below and dry above."""
    z = np.linspace(0.0, z_top, n)
    theta = np.where(z < z_inv, theta_sfc,
                     theta_sfc + dtheta_inv + gamma_ft * (z - z_inv))
    rh = np.where(z < z_inv, rh_bl, rh_ft)
    return build(z, theta, rh)


def sheared(z_top=6000.0, n=61, theta_sfc=300.0, gamma=3e-3, rh=0.4,
            u_sfc=2.0, du_dz=4e-3, v_sfc=-1.0):
    z = np.linspace(0.0, z_top, n)
    return build(z, theta_sfc + gamma * z, np.full_like(z, rh),
                 u_sfc + du_dz * z, np.full_like(z, v_sfc))


CATALOGUE = {
    "dry_neutral": dry_neutral,
    "stable": stable,
    "inversion": inversion,
    "sheared": sheared,
}
