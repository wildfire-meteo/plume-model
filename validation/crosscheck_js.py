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

"""Equivalence check of this package against the reference JavaScript implementation.

Local and optional: it needs `node` and a checkout of wildfire-meteo-dmt, located via
$WILDFIRE_METEO_DMT or ../wildfire-meteo-dmt. It skips, and exits 0, when either is absent.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import environments as envs
from plume_model import MortonEntrainment, Plume, PlumeBase

HERE = Path(__file__).parent
DRIVER = HERE / "js" / "run_parcel.mjs"

# Name in the JavaScript output, name on the Plume. Compared element by element.
ARRAYS = {"z": "z", "p": "p", "T": "T", "Tv": "Tv", "Td": "Td", "thetal": "thetal",
          "thetav": "thetav", "qt": "qt", "area": "area", "w": "w", "buoy": "buoy",
          "u": "u", "v": "v", "x": "x", "y": "y", "mass_flux": "mass_flux",
          "entrainment": "entrainment", "detrainment": "detrainment", "type": "saturated"}
SCALARS = ["k_top", "k_lcl", "stopped"]

# Environment, sensible heat flux, latent heat flux, fire area, extra options.
CASES = [
    ("dry_neutral", 50e3, 0.0, 1e6, {}),
    ("stable", 10e3, 0.0, 1e6, {}),
    ("inversion", 100e3, 20e3, 1e5, {}),
    ("sheared", 150e3, 10e3, 1e4, {}),
    ("sheared", 40e3, 5e3, 1e6, {"beta": 0.2, "c_det": 4.0, "fac_ent": 2.0}),
    ("dry_neutral", 50e3, 0.0, 1e6, {"fac_ent": 0.0, "full_ascent": True}),
    ("inversion", 0.0, 0.0, 1e6, {}),
]


def reference_repo():
    p = Path(os.environ.get("WILDFIRE_METEO_DMT", HERE.parent.parent / "wildfire-meteo-dmt"))
    return p if (p / "web" / "parcel.js").is_file() else None


def build_cases():
    cases = []
    for name, H, LE, area, opts in CASES:
        env = envs.CATALOGUE[name]()
        base = PlumeBase(H, LE, area, env)
        # The non-entraining mode needs a nominal w0 to seed an ascent with no fire.
        w0 = max(base.w0, 1e-3) if opts.get("full_ascent") else base.w0
        base.w0 = w0
        cases.append({"name": "%s_H%g_LE%g_A%g" % (name, H / 1e3, LE / 1e3, area),
                      "env": env, "base": base, "opts": opts})
    return cases


def js_payload(cases):
    payload = []
    for c in cases:
        env, base = c["env"], c["base"]
        opts = dict(c["opts"])
        opts.setdefault("z_max", env.z_top)
        payload.append({"z_env": env.z.tolist(), "T_env": env.T.tolist(),
                        "Td_env": env.Td.tolist(), "p_env": env.p.tolist(),
                        "u_env": env.u.tolist(), "v_env": env.v.tolist(),
                        "dtheta": base.dtheta, "dq": base.dq, "w0": base.w0,
                        "area": base.area, "opts": opts})
    return payload


def run_python(case):
    opts = case["opts"]
    ent = MortonEntrainment(**{k: v for k, v in opts.items()
                               if k in ("fac_ent", "beta", "c_det")})
    plume = Plume(case["env"], case["base"], entrainment=ent,
                  z_max=opts.get("z_max", case["env"].z_top),
                  full_ascent=opts.get("full_ascent", False))
    return plume.ascend()


def as_array(values):
    """JSON has no NaN; JSON.stringify writes null, which comes back as None."""
    return np.array([np.nan if x is None else x for x in values], dtype=float)


def compare(js, plume):
    problems = []
    worst = 0.0

    for key in SCALARS:
        a, b = js[key], getattr(plume, key)
        differs = bool(a) != bool(b) if key == "stopped" else a != b
        if differs:
            problems.append("%s: js=%s py=%s" % (key, a, b))

    for js_key, py_key in ARRAYS.items():
        # The reference's empty return omits some keys; absent means an empty profile.
        a = as_array(js.get(js_key, []))
        b = np.asarray(getattr(plume, py_key), dtype=float)
        if a.shape != b.shape:
            problems.append("%s: shape js=%s py=%s" % (js_key, a.shape, b.shape))
            continue
        if a.size == 0:
            continue
        if not np.array_equal(np.isfinite(a), np.isfinite(b)):
            problems.append("%s: NaN pattern differs" % js_key)
        finite = np.isfinite(a) & np.isfinite(b)
        rel = np.abs(a[finite] - b[finite]) / np.maximum(np.abs(a[finite]), 1e-30)
        if rel.size:
            worst = max(worst, float(rel.max()))
            if rel.max() > 1e-10:
                k = int(np.argmax(rel))
                problems.append("%s: max rel %.3e at index %d (js=%.12g py=%.12g)"
                                % (js_key, rel.max(), k, a[finite][k], b[finite][k]))
    return problems, worst


def main():
    repo = reference_repo()
    if repo is None:
        print("SKIP: wildfire-meteo-dmt not found (set $WILDFIRE_METEO_DMT)")
        return 0
    if shutil.which("node") is None:
        print("SKIP: node not on PATH")
        return 0

    cases = build_cases()
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(js_payload(cases), f)
        cases_path = f.name
    try:
        result = subprocess.run(
            ["node", str(DRIVER), str(repo / "web" / "parcel.js"), cases_path],
            capture_output=True, text=True)
    finally:
        os.unlink(cases_path)

    if result.returncode != 0:
        print("FAIL: node driver errored\n" + result.stderr)
        return 1

    print("reference: %s" % repo)
    n_failed = 0
    worst_all = 0.0
    for case, js in zip(cases, json.loads(result.stdout)):
        problems, worst = compare(js, run_python(case))
        worst_all = max(worst_all, worst)
        print("%s %-34s n=%4d max rel diff %.2e"
              % ("OK  " if not problems else "FAIL", case["name"], len(js["z"]), worst))
        for p in problems:
            print("       " + p)
        n_failed += bool(problems)

    print("\n%d/%d cases agree; worst relative difference %.3e"
          % (len(cases) - n_failed, len(cases), worst_all))
    return 1 if n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
