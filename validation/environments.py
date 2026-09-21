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
from plume_model.thermo import Rd, cp, dewpoint, exner, g, p0, qsat, virtual_temp


def pressure_from_exner(pi):
    return p0 * pi ** (cp / Rd)


def build(z, theta, rh, u=None, v=None, p_sfc=1e5, n_iter=5):
    """Environment from (theta, RH) profiles, in hydrostatic balance."""
    n = len(z)
    pi = np.empty(n)
    T = np.empty(n)
    qt = np.empty(n)

    pi[0] = exner(p_sfc)
    T[0] = theta[0] * pi[0]
    qt[0] = rh[0] * qsat(T[0], p_sfc)

    for k in range(1, n):
        dz = z[k] - z[k - 1]
        thetav_prev = virtual_temp(theta[k - 1], qt[k - 1])
        pi[k] = pi[k - 1] - g * dz / (cp * thetav_prev)
        # Trapezoidal dpi/dz = -g / (cp thetav), iterated since qt depends on p through RH.
        for _ in range(n_iter):
            T[k] = theta[k] * pi[k]
            qt[k] = rh[k] * qsat(T[k], pressure_from_exner(pi[k]))
            thetav = virtual_temp(theta[k], qt[k])
            pi[k] = pi[k - 1] - 0.5 * g * dz / cp * (1 / thetav_prev + 1 / thetav)
        T[k] = theta[k] * pi[k]
        qt[k] = rh[k] * qsat(T[k], pressure_from_exner(pi[k]))

    p = pressure_from_exner(pi)
    return Environment(z, T, dewpoint(qt, p), p, u, v)


def mixed_layer_inversion(h=1500.0, thetal_ml=300.0, qt_ml=8e-3, dthetal=2.0,
                          lapse_rate=6.5e-3, rh_ft=0.3, u=5.0, v=0.0,
                          p_sfc=1e5, z_top=8000.0, dz=50.0, n_iter=5):
    """Well-mixed layer in (thetal, qt) up to h, capped by a sharp inversion.

    Above h the temperature falls at a fixed lapse rate from the value the thetal jump
    gives at h, T = (thetal_ml + dthetal) pi(h) - lapse_rate (z - h), at a fixed relative
    humidity. The humidity jump at h is therefore not prescribed but set by rh_ft. The
    inversion sits between the level at h and the next one, so h must be a multiple of dz.
    """
    k_h = int(round(h / dz))
    if abs(k_h * dz - h) > 1e-9:
        raise ValueError("h = %g m is not a multiple of dz = %g m" % (h, dz))

    z = np.arange(0.0, z_top + 0.5 * dz, dz)
    n = len(z)
    pi = np.empty(n)
    T = np.empty(n)
    qt = np.empty(n)

    # Unsaturated mixed layer, so thetal = theta and thetav is constant: pi is exactly linear.
    thetav_ml = virtual_temp(thetal_ml, qt_ml)
    for k in range(k_h + 1):
        pi[k] = exner(p_sfc) - g * z[k] / (cp * thetav_ml)
        T[k] = thetal_ml * pi[k]
        qt[k] = qt_ml

    T_base = (thetal_ml + dthetal) * pi[k_h]

    for k in range(k_h + 1, n):
        thetav_prev = virtual_temp(T[k - 1] / pi[k - 1], qt[k - 1])
        T[k] = T_base - lapse_rate * (z[k] - h)
        pi[k] = pi[k - 1] - g * dz / (cp * thetav_prev)
        # Trapezoidal dpi/dz = -g / (cp thetav), iterated since qt depends on p through RH.
        for _ in range(n_iter):
            qt[k] = rh_ft * qsat(T[k], pressure_from_exner(pi[k]))
            thetav = virtual_temp(T[k] / pi[k], qt[k])
            pi[k] = pi[k - 1] - 0.5 * g * dz / cp * (1 / thetav_prev + 1 / thetav)
        qt[k] = rh_ft * qsat(T[k], pressure_from_exner(pi[k]))

    p = pressure_from_exner(pi)
    if np.any(qt[:k_h + 1] >= qsat(T[:k_h + 1], p[:k_h + 1])):
        raise ValueError("the mixed layer is saturated; the model's environment must not be")

    env = Environment(z, T, dewpoint(qt, p), p, np.full(n, u), np.full(n, v))
    env.h = h
    env.dq_inversion = rh_ft * qsat(T_base, p[k_h]) - qt_ml
    return env


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
