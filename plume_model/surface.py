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

        rho = env.rho[0]
        thetav = env.thetav[0]
        K = 3 * g * a_w * h0 / (2 * thetav * (1 + b_w))

        self.dtheta = 0.0 if H <= 0 else (H / (rho * cp * np.sqrt(K))) ** (2 / 3)
        self.w0 = 0.0 if self.dtheta <= 0 else np.sqrt(K * self.dtheta)
        self.dq = 0.0 if self.w0 <= 0 else LE / (rho * Lv * self.w0)
