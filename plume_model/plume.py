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

"""Steady bulk plume, integrated upward in height at the environmental pressure."""

import numpy as np

from .entrainment import MortonEntrainment
from .thermo import Rd, dewpoint, exner, g, qsat, sat_adjust, virtual_temp

A_W = 1.0
B_W = 0.2
H0_PLUME = 20.0
DZ_PLUME = 50.0

W_EPS = 1e-6


class Plume:
    """A Plume is a trajectory through an environment (env), from an initial state (base), with
    a prescribed entrainment model (entrainment), on a grid with spacing dz until z_max.

    The main function is ascend, which integrates upwards a uniform grid. All profiles
    are calculated on this grid, including the environment on the plume's own grid and the
    terms of the w**2 budget.
    """

    def __init__(self, env, base, entrainment=None, dz=DZ_PLUME, z_max=None,
                 a_w=A_W, b_w=B_W, h0=H0_PLUME, full_ascent=False):
        self.env = env
        self.base = base
        self.entrainment_model = MortonEntrainment() if entrainment is None else entrainment
        self.dz = dz
        self.z_max = env.z_top if z_max is None else z_max
        self.a_w = a_w
        self.b_w = b_w
        self.h0 = h0
        self.full_ascent = full_ascent

        self.k_top = -1
        self.k_lcl = -1
        self.z_top = np.nan
        self.stopped = False
        self.ascended = False

    def ascend(self):
        """Integrate the plume from the surface up to where it stops, or to z_max."""
        env = self.env
        dz = self.dz
        n = int(np.floor(self.z_max / dz))

        # Allocate all profiles
        self._allocate(n)
        z = self.z

        # The environment is interpolated to the plume grid, then its variables are derived.
        T_e = np.interp(z, env.z, env.T)
        Td_e = np.interp(z, env.z, env.Td)
        p_e = np.interp(z, env.z, env.p)
        u_e = np.interp(z, env.z, env.u)
        v_e = np.interp(z, env.z, env.v)

        exner_e = exner(p_e)
        theta_e = T_e / exner_e
        qt_e = qsat(Td_e, p_e)
        thetav_e = virtual_temp(theta_e, qt_e)
        rho_e = p_e / (Rd * exner_e * thetav_e)

        self.T_env = T_e
        self.Td_env = Td_e
        self.p = p_e
        self.theta_env = theta_e
        self.qt_env = qt_e
        self.thetav_env = thetav_e
        self.rho_env = rho_e
        self.u_env = u_e
        self.v_env = v_e

        # If the plume's initialisation gives no updraft, stop immediately
        if self.base.w0 < W_EPS:
            self._finish(0)
            return self

        # Execute the first model level separately
        self._start(exner_e, theta_e, qt_e, thetav_e, rho_e, p_e)

        # Iterate until plume top, or until w < W_EPS, or until mass flux <= 0
        i = 1
        while i < n:

            # Main dynamical vertical upwards integration
            self._step(i, dz, exner_e, theta_e, qt_e, thetav_e, rho_e, p_e, u_e, v_e)

            if self.w[i] < W_EPS or self.mass_flux[i] <= 0:
                if self.k_top == -1:
                    self.k_top = i - 1
                    self.stopped = True
                if not self.full_ascent:
                    break
            i += 1

        if self.k_top == -1:
            self.k_top = i - 1

        self._finish(i)
        return self

    def _allocate(self, n):
        self.z = np.arange(n, dtype=float) * self.dz
        for name in ("thetal", "qt", "thetav", "T", "Tv", "area", "w", "mass_flux",
                     "entrainment", "detrainment", "eps", "delta", "delta_dyn",
                     "buoy", "u", "v", "x", "y", "dw2dz_buoy", "dw2dz_drag"):
            setattr(self, name, np.zeros(n))
        self.saturated = np.zeros(n, dtype=bool)

    def _start(self, exner_e, theta_e, qt_e, thetav_e, rho_e, p_e):
        env, base = self.env, self.base

        # The fire perturbation is a dry heat excess (ql = 0 here), so thetal == theta.
        self.thetal[0] = theta_e[0] + base.dtheta
        self.qt[0] = qt_e[0] + base.dq

        T, ql, qi = sat_adjust(self.thetal[0], self.qt[0], p_e[0])
        self.T[0] = T
        self.Tv[0] = virtual_temp(T, self.qt[0], ql, qi)
        self.thetav[0] = self.Tv[0] / exner_e[0]
        self.buoy[0] = g / thetav_e[0] * (self.thetav[0] - thetav_e[0])
        self.area[0] = base.area
        self.w[0] = base.w0
        self.mass_flux[0] = rho_e[0] * self.area[0] * self.w[0]

        # The plume leaves the surface with the environmental momentum at h0.
        self.u[0] = np.interp(self.h0, env.z, env.u)
        self.v[0] = np.interp(self.h0, env.z, env.v)

        self.eps[0], self.delta[0] = self.entrainment_model.rates(
            self.z[0], self.w[0], self.area[0], base.area, self.mass_flux[0], self.buoy[0])
        self.entrainment[0] = self.eps[0] * self.mass_flux[0]
        self.detrainment[0] = 0.0
        self.dw2dz_buoy[0] = 2 * self.a_w * self.buoy[0]
        self.dw2dz_drag[0] = -2 * self.b_w * self.eps[0] * self.w[0] ** 2

    def _step(self, i, dz, exner_e, theta_e, qt_e, thetav_e, rho_e, p_e, u_e, v_e):
        j = i - 1
        M = self.mass_flux
        E = self.entrainment

        # Constant-fraction detrainment only; the dynamic part follows once w[i] is known.
        M[i] = M[j] + (E[j] - self.delta[j] * M[j]) * dz
        
        # theta_e stands in for thetal_e, exact only while the environment is unsaturated.
        self.thetal[i] = self.thetal[j] - E[j] * (self.thetal[j] - theta_e[j]) / M[j] * dz
        self.qt[i] = self.qt[j] - E[j] * (self.qt[j] - qt_e[j]) / M[j] * dz

        alive = self.k_top == -1

        if alive:
            self.u[i] = self.u[j] - E[j] * (self.u[j] - u_e[j]) / M[j] * dz
            self.v[i] = self.v[j] - E[j] * (self.v[j] - v_e[j]) / M[j] * dz
            self.x[i] = self.x[j] + self.u[j] / self.w[j] * dz
            self.y[i] = self.y[j] + self.v[j] / self.w[j] * dz
        else:
            self.u[i] = self.u[j]
            self.v[i] = self.v[j]
            self.x[i] = self.x[j]
            self.y[i] = self.y[j]

        T, ql, qi = sat_adjust(self.thetal[i], self.qt[i], p_e[i])

        # Pseudoadiabatic: the condensate is removed rather than carried up with the plume.
        self.qt[i] -= ql
        self.thetal[i] = T / exner_e[i]

        self.T[i] = T
        self.Tv[i] = virtual_temp(T, self.qt[i], 0.0, 0.0)
        self.thetav[i] = self.Tv[i] / exner_e[i]
        self.saturated[i] = ql > 0 or qi > 0

        self.buoy[i] = g / thetav_e[i] * (self.thetav[i] - thetav_e[i])

        self.dw2dz_buoy[i] = 2 * self.a_w * self.buoy[i]
        self.dw2dz_drag[i] = -2 * self.b_w * self.eps[j] * self.w[j] ** 2
        w2 = self.w[j] ** 2 + (self.dw2dz_buoy[i] + self.dw2dz_drag[i]) * dz
        self.w[i] = np.sqrt(max(0.0, w2)) if alive else 0.0

        decelerating = alive and self.w[i] >= W_EPS and self.w[i] < self.w[j]
        if decelerating:
            self.delta_dyn[i] = self.entrainment_model.dynamic_detrainment(
                self.w[i], self.w[j], dz)
            M[i] *= np.exp(-self.delta_dyn[i] * dz)

        self.eps[i], self.delta[i] = self.entrainment_model.rates(
            self.z[i], self.w[i], self.area[j], self.base.area, M[i], self.buoy[i])
        E[i] = self.eps[i] * M[i]
        self.detrainment[i] = (self.delta[i] + self.delta_dyn[i]) * M[i]

        self.area[i] = M[i] / (rho_e[i] * self.w[i]) if self.w[i] > W_EPS else np.nan

    def _truncate(self, n_keep):
        """Cut every profile down to the part that was actually integrated."""
        for name, value in list(vars(self).items()):
            if isinstance(value, np.ndarray):
                setattr(self, name, value[:n_keep])

    def _finish(self, n_keep):
        self._truncate(n_keep)
        # Above the LCL the plume is saturated, so its dewpoint is capped at T.
        self.Td = np.minimum(dewpoint(self.qt, self.p), self.T)
        self.k_lcl = int(np.argmax(self.saturated)) if self.saturated.any() else -1
        self.z_top = float(self.z[self.k_top]) if self.k_top >= 0 else np.nan
        self.ascended = True


def find_lcl(T_sfc, Td_sfc, p_sfc, tol=5.0):
    """LCL by bisection on pressure, where the dry adiabat meets the isohume."""
    theta_sfc = T_sfc / exner(p_sfc)
    q_sfc = qsat(Td_sfc, p_sfc)

    p_lo, p_hi = 500e2, p_sfc
    while (p_hi - p_lo) > tol:
        p_mid = 0.5 * (p_lo + p_hi)
        if theta_sfc * exner(p_mid) - dewpoint(q_sfc, p_mid) > 0:
            p_hi = p_mid
        else:
            p_lo = p_mid

    p_lcl = 0.5 * (p_lo + p_hi)
    return p_lcl, theta_sfc * exner(p_lcl)
