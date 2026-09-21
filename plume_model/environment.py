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

from .thermo import Rd, dewpoint, exner, qsat, virtual_temp


class Environment:
    """The environment the plume rises through, on its own ascending height grid."""

    def __init__(self, z, T, Td, p, u=None, v=None):
        self.z = np.asarray(z, dtype=float)
        self.T = np.asarray(T, dtype=float)
        self.Td = np.asarray(Td, dtype=float)
        self.p = np.asarray(p, dtype=float)
        self.u = np.zeros_like(self.z) if u is None else np.asarray(u, dtype=float)
        self.v = np.zeros_like(self.z) if v is None else np.asarray(v, dtype=float)

        self.exner = exner(self.p)
        self.theta = self.T / self.exner
        self.qt = qsat(self.Td, self.p)
        self.thetav = virtual_temp(self.theta, self.qt)
        self.rho = self.p / (Rd * self.exner * self.thetav)
        self.z_top = float(self.z[-1])


def environment_from_theta(z, theta, qt, p, u=None, v=None):
    """Build an Environment from (theta, qt), the form LES and models usually give."""
    pi = exner(np.asarray(p, dtype=float))
    T = np.asarray(theta, dtype=float) * pi
    return Environment(z, T, dewpoint(np.asarray(qt, dtype=float), p), p, u, v)
