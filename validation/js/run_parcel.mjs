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

// Driver runs the reference calc_parcel_ascent on cases given as JSON on argv.
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";

const [, , parcel_path, cases_path] = process.argv;
const { calc_parcel_ascent } = await import(pathToFileURL(parcel_path).href);
const cases = JSON.parse(readFileSync(cases_path, "utf8"));

const out = cases.map(c => calc_parcel_ascent(
    c.z_env, c.T_env, c.Td_env, c.p_env, c.u_env, c.v_env,
    c.dtheta, c.dq, c.w0, c.area,
    c.opts,
));

process.stdout.write(JSON.stringify(out));
