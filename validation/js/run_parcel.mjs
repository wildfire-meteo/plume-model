//
// Copyright 2026 Wageningen University & Research (WUR)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

// Driver builds the base state and runs calc_parcel_ascent as skewt.js does, on cases
// given as JSON on argv.
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";

const [, , web_path, cases_path] = process.argv;
const P = await import(pathToFileURL(web_path + "/parcel.js").href);
const F = await import(pathToFileURL(web_path + "/fire_surface.js").href);
const cases = JSON.parse(readFileSync(cases_path, "utf8"));

const out = cases.map(c => {
    const wind_speed = Math.hypot(P.interp([P.H0_PLUME], c.z_env, c.u_env)[0],
                                  P.interp([P.H0_PLUME], c.z_env, c.v_env)[0]);
    const u_vent = wind_speed * P.H0_PLUME / Math.sqrt(c.fire_area);
    const dtheta = F.dtheta_from_H(c.H, c.rho, c.thetav, u_vent);
    const dq     = F.dq_from_LE(c.LE, dtheta, c.rho, c.thetav, u_vent);
    const w0     = F.w0_from_dtheta(dtheta, c.thetav);
    const area   = w0 > 0 ? c.fire_area * (w0 + u_vent) / w0 : c.fire_area;

    const classic = !!c.opts.non_entraining;
    const { non_entraining, ...opts } = c.opts;
    const no_ent  = classic ? { a_e: 0, b_e: 0, a_w: 1 } : {};
    const w0_run  = classic ? Math.max(w0, 1e-3) : w0;

    const r = P.calc_parcel_ascent(
        c.z_env, c.T_env, c.Td_env, c.p_env, c.u_env, c.v_env,
        dtheta, dq, w0_run, area, c.fire_area,
        { ...no_ent, ...opts, z_max: c.z_max, full_ascent: classic },
    );
    r.base = { dtheta, dq, w0: w0_run, area, u_vent };
    return r;
});

process.stdout.write(JSON.stringify(out));
