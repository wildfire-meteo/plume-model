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

import numpy as np

from .plume import A_W, B_W, H0_PLUME
from .thermo import Lv, cp, g


class PlumeBase:
    """The plume's base state: the excesses a surface fire imposes at the plume bottom.

    The heat flux sets the temperature excess through free-convective scaling, combining
    w0**3 = C H with H = rho cp dtheta w0, so that

        w0**2 = K dtheta,    K = 3 g a_w h0 / (2 thetav (1 + b_w)),

    and the moisture flux then sets the moisture excess at that w0.
    """

    def __init__(self, H, LE, area, env, a_w=A_W, b_w=B_W, h0=H0_PLUME):
        self.H = H
        self.LE = LE
        self.area = area
        self.fire_area = area

        rho = env.rho[0]
        thetav = env.thetav[0]
        K = 3 * g * a_w * h0 / (2 * thetav * (1 + b_w))

        self.dtheta = 0.0 if H <= 0 else (H / (rho * cp * np.sqrt(K))) ** (2 / 3)
        self.w0 = 0.0 if self.dtheta <= 0 else np.sqrt(K * self.dtheta)
        self.dq = 0.0 if self.w0 <= 0 else LE / (rho * Lv * self.w0)


class VentilatedPlumeBase:
    """The plume's base state when horizontal wind contributes to volume flux out of a control
    volume above the fire of height h0 and along-wind depth L. Heat flux per unit fire
    area is ventilated by a volume flux per unit fire area equal wu + s * h0/L, so conservation
    of heat reads:

        H = rho cp dtheta (w0 + s h0 / L),    w0**2 = K dtheta    (K as in PlumeBase),

    so w0**2 (w0 + s h0 / L) = K H / (rho cp). Limits:
     - s h0 / L >> w0:  w0 ~ H**(1/2) * s**(-1/2)
     - s h0 / L << w0:  w0 ~ H**(1/3)
    
    Assumptions:
    - The fire is a rectangle, so L = sqrt(area / aspect_ratio).
    - The base area A_f (w0 + s h0 / L) / w0 includes the air vented downwind, so that the plume
      carries the fire's heat flux.
    - s is the *environmental* wind speed, but it could be further constrained.
    """

    def __init__(self, H, LE, area, env, aspect_ratio=1.0, a_w=A_W, b_w=B_W, h0=H0_PLUME):
        self.H = H
        self.LE = LE
        self.fire_area = area
        self.fire_depth = np.sqrt(area / aspect_ratio)

        rho = env.rho[0]
        thetav = env.thetav[0]
        K = 3 * g * a_w * h0 / (2 * thetav * (1 + b_w))
        self.wind_speed = np.hypot(np.interp(h0, env.z, env.u), np.interp(h0, env.z, env.v))
        self.u_vent = self.wind_speed * h0 / self.fire_depth

        if H <= 0:
            self.w0 = 0.0
            self.dtheta = 0.0
            self.dq = 0.0
            self.area = area
            return

        roots = np.roots([1.0, self.u_vent, 0.0, -K * H / (rho * cp)])
        self.w0 = float(max(r.real for r in roots if abs(r.imag) < 1e-9 * abs(r) + 1e-12))
        self.dtheta = self.w0 ** 2 / K
        ventilation = self.w0 + self.u_vent
        self.dq = LE / (rho * Lv * ventilation)
        self.area = area * ventilation / self.w0

