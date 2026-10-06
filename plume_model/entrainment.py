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

"""Entrainment models.

An entrainment model is any object with two methods:

    rates(z, w, area, area_0, mass_flux, buoy)  -> (eps, delta)
    dynamic_detrainment(w, w_prev, dz)          -> delta_dyn

The integrator calls them at every level, so a model may use as much or as little of the
plume state as its formulation needs.
"""

import numpy as np

FAC_ENT = 1.0  # Non-dimensional scaling of entrainment, from Eyken (2026)
BETA = 0.5     # Ratio of fractional detrainment to fractional entrainment
C_DET = 2.0    # Dynamic detrainment -C_DET/w dw/dz where w decreases
A_BUOYANT = 0.3        # Buoyancy-driven entrainment a B/w**2, from the LES plume core
FAC_ENT_BUOYANT = 0.4  # Morton part of BuoyantEntrainment, from the LES plume core


class MortonEntrainment:
    """eps = fac_ent / sqrt(A_0), constant with height; delta = beta * eps."""

    def __init__(self, fac_ent=FAC_ENT, beta=BETA, c_det=C_DET):
        self.fac_ent = fac_ent
        self.beta = beta
        self.c_det = c_det

    def rates(self, z, w, area, area_0, mass_flux, buoy):
        eps = self.fac_ent / np.sqrt(area_0)
        return eps, self.beta * eps

    def dynamic_detrainment(self, w, w_prev, dz):
        """Integrated exactly over the step, so M scales by (w/w_prev)**c_det."""
        return -self.c_det * np.log(w / w_prev) / dz


class BuoyantEntrainment:
    """Morton plus buoyancy-driven entrainment:

        eps   = fac_ent / sqrt(A_0) + a max(B, 0) / w**2,
        delta = beta fac_ent / sqrt(A_0).

    B / w**2 is the inverse of the length over which buoyancy accelerates the plume, 
    which is equal to fractional entrainment if plume area is constant. a sets the
    efficiency of that process. This term is largest near plume base where w is small.
    """

    def __init__(self, a=A_BUOYANT, fac_ent=FAC_ENT_BUOYANT, beta=BETA, c_det=C_DET):
        self.a = a
        self.morton = MortonEntrainment(fac_ent, beta, c_det)

    def rates(self, z, w, area, area_0, mass_flux, buoy):
        eps_morton, delta = self.morton.rates(z, w, area, area_0, mass_flux, buoy)
        return eps_morton + self.a * max(buoy, 0.0) / max(w, 1e-6) ** 2, delta

    def dynamic_detrainment(self, w, w_prev, dz):
        return self.morton.dynamic_detrainment(w, w_prev, dz)
