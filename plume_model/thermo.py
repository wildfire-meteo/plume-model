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

Rd = 287.04
Rv = 461.5
cp = 1005.0
Lv = 2.5e6
p0 = 1e5
eps = Rd / Rv
e0 = 611.2
a = 17.67
b = 243.5
T0 = 273.15
g = 9.81

# Smallest positive double, guarding log(0) in the dewpoint inversion.
TINY = 5e-324


def esat(T):
    """Saturation vapour pressure over liquid, Bolton (1980)."""
    Tc = T - T0
    return e0 * np.exp(a * Tc / (Tc + b))


def qsat(T, p):
    """Saturation specific humidity."""
    es = esat(T)
    return eps * es / (p - (1.0 - eps) * es)


def esat_from_q(q, p):
    """Vapour pressure from specific humidity, the inverse of qsat."""
    return q * p / (eps + (1.0 - eps) * q)


def dewpoint(q, p):
    """Dewpoint from specific humidity, inverting the Bolton formula."""
    es = esat_from_q(q, p)
    lnr = np.log(np.maximum(TINY, es / e0))
    Tc = b * lnr / (a - lnr)
    return Tc + T0


def exner(p):
    """Exner function."""
    return (p / p0) ** (Rd / cp)


def dqsatdT(T, p):
    """Clausius-Clapeyron slope of qsat at constant pressure."""
    es = esat(T)
    Tc = T - T0
    des_dT = es * a * b / (Tc + b) ** 2
    den = p - (1.0 - eps) * es
    return eps * p * des_dT / den**2


def virtual_temp(T, qt, ql=0.0, qi=0.0):
    """Virtual (density) temperature; condensate loads through ql + qi."""
    return T * (1 + (Rv / Rd - 1) * qt - Rv / Rd * (ql + qi))


def thetal(theta, ql, T):
    """Liquid water potential temperature, exponential form."""
    return theta * np.exp(-Lv * ql / (cp * T))


def sat_adjust(thetal, qt, p):
    """Saturation adjustment: solve (T, ql, qi) from (thetal, qt, p) by Newton iteration."""
    pi = exner(p)
    tl = thetal * pi
    qs = qsat(tl, p)

    if qt - qs <= 0.0:
        return tl, 0.0, 0.0

    tnr = tl
    tnr_old = 1e9
    niter = 0
    while abs(tnr - tnr_old) / tnr_old > 1e-5 and niter < 10:
        niter += 1
        tnr_old = tnr
        qs = qsat(tnr, p)
        ql = qt - qs
        exp_A = np.exp(-Lv * ql / (cp * tnr))
        f = tnr * exp_A - tl
        f_prime = exp_A * (1.0 + Lv / cp * (dqsatdT(tnr, p) + ql / tnr))
        tnr -= f / f_prime

    ql = max(0.0, qt - qsat(tnr, p))
    return tnr, ql, 0.0
