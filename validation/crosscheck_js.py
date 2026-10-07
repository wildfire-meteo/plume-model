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

Both sides run their default model: BuoyantEntrainment, the default w equation, and
VentilatedPlumeBase with aspect ratio 1. The JavaScript base state is built from (H, LE, area)
by fire_surface.js, as the web tool does, and compared with VentilatedPlumeBase.

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
from plume_model import BuoyantEntrainment, Plume, VentilatedPlumeBase

HERE = Path(__file__).parent
DRIVER = HERE / "js" / "run_parcel.mjs"

# The reference implementation's former w equation, kept for the LES scripts' reference runs.
REFERENCE_A_W = 1.0
REFERENCE_B_W = 0.2

# Name in the JavaScript output, name on the Plume. Compared element by element.
ARRAYS = {"z": "z", "p": "p", "T": "T", "Tv": "Tv", "Td": "Td", "thetal": "thetal",
          "thetav": "thetav", "qt": "qt", "area": "area", "w": "w", "buoy": "buoy",
          "u": "u", "v": "v", "x": "x", "y": "y", "mass_flux": "mass_flux",
          "entrainment": "entrainment", "detrainment": "detrainment", "type": "saturated"}
SCALARS = ["k_top", "k_lcl", "stopped"]
BASE = ["dtheta", "dq", "w0", "area", "u_vent"]

WINDY = {"u_sfc": 12.0, "v_sfc": -5.0}

# Environment, its options, sensible heat flux, latent heat flux, fire area, JavaScript options.
CASES = [
    ("dry_neutral", {}, 50e3, 0.0, 1e6, {}),
    ("stable", {}, 10e3, 0.0, 1e6, {}),
    ("inversion", {}, 100e3, 20e3, 1e5, {}),
    ("sheared", {}, 150e3, 10e3, 1e4, {}),
    ("sheared", {}, 300e3, 0.0, 1e5, {}),
    ("sheared", {}, 40e3, 5e3, 1e6, {"a_e": 1.0, "b_e": 0.6, "beta": 0.2, "c_det": 4.0}),
    ("sheared", WINDY, 100e3, 10e3, 1e5, {}),
    ("sheared", WINDY, 30e3, 0.0, 1e4, {}),
    ("sheared", WINDY, 300e3, 0.0, 1e6, {"a_e": 0.1}),
    ("sheared", WINDY, 50e3, 0.0, 1e6, {"non_entraining": True}),
    ("inversion", {}, 0.0, 0.0, 1e6, {}),
]


def reference_repo():
    p = Path(os.environ.get("WILDFIRE_METEO_DMT", HERE.parent.parent / "wildfire-meteo-dmt"))
    return p if (p / "web" / "parcel.js").is_file() else None


def build_cases():
    cases = []
    for name, env_opts, H, LE, area, opts in CASES:
        env = envs.CATALOGUE[name](**env_opts)
        label = "%s%s_H%g_LE%g_A%g" % (name, "_windy" if env_opts else "", H / 1e3, LE / 1e3, area)
        if opts.get("non_entraining"):
            label += "_classic"
        cases.append({"name": label, "env": env, "H": H, "LE": LE, "area": area, "opts": opts})
    return cases


def js_payload(cases):
    payload = []
    for c in cases:
        env = c["env"]
        payload.append({"z_env": env.z.tolist(), "T_env": env.T.tolist(),
                        "Td_env": env.Td.tolist(), "p_env": env.p.tolist(),
                        "u_env": env.u.tolist(), "v_env": env.v.tolist(),
                        "rho": float(env.rho[0]), "thetav": float(env.thetav[0]),
                        "H": c["H"], "LE": c["LE"], "fire_area": c["area"],
                        "z_max": float(env.z_top), "opts": c["opts"]})
    return payload


def run_python(case):
    """The Python equivalent of the web tool's entraining and non-entraining modes."""
    opts = case["opts"]
    env = case["env"]
    base = VentilatedPlumeBase(case["H"], case["LE"], case["area"], env)
    if opts.get("non_entraining"):
        # The web tool seeds the non-entraining ascent with a nominal w0.
        base.w0 = max(base.w0, 1e-3)
        ent = BuoyantEntrainment(a=0.0, fac_ent=0.0)
        plume = Plume(env, base, entrainment=ent, a_w=1.0, full_ascent=True)
    else:
        names = {"a_e": "fac_ent", "b_e": "a", "beta": "beta", "c_det": "c_det"}
        ent = BuoyantEntrainment(**{names[k]: v for k, v in opts.items() if k in names})
        plume = Plume(env, base, entrainment=ent)
    return base, plume.ascend()


def as_array(values):
    """JSON has no NaN; JSON.stringify writes null, which comes back as None."""
    return np.array([np.nan if x is None else x for x in values], dtype=float)


def relative_difference(a, b):
    return np.abs(a - b) / np.maximum(np.abs(a), 1e-30)


def compare_base(js_base, base):
    problems = []
    worst = 0.0
    for key in BASE:
        rel = float(relative_difference(js_base[key], getattr(base, key)))
        worst = max(worst, rel)
        if rel > 1e-10:
            problems.append("base %s: rel %.3e (js=%.12g py=%.12g)"
                            % (key, rel, js_base[key], getattr(base, key)))
    return problems, worst


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
        rel = relative_difference(a[finite], b[finite])
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
            ["node", str(DRIVER), str(repo / "web"), cases_path],
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
        base, plume = run_python(case)
        base_problems, base_worst = compare_base(js["base"], base)
        problems, worst = compare(js, plume)
        problems = base_problems + problems
        worst = max(worst, base_worst)
        worst_all = max(worst_all, worst)
        print("%s %-42s n=%4d u_vent %5.2f max rel diff %.2e"
              % ("OK  " if not problems else "FAIL", case["name"], len(js["z"]),
                 js["base"]["u_vent"], worst))
        for p in problems:
            print("       " + p)
        n_failed += bool(problems)

    print("\n%d/%d cases agree; worst relative difference %.3e"
          % (len(cases) - n_failed, len(cases), worst_all))
    return 1 if n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
