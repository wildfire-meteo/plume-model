# plume-model

A simple entraining plume model of the dynamics of wildfire plumes for education and research.

We intend this to be the training ground for the model running in [`wildfire-meteo-dmt`](https://github.com/wildfire-meteo/wildfire-meteo-dmt), without the web interface and the `open_meteo`.

## Install

```bash
python -m venv /path/to/venv
source /path/to/venv/bin/activate
pip install -e .
```

This gives an editable installation, where  `plume_model` and `config` contain importable modules, and edits in those modules propagate into the scripts that call them directly. If you add a new top-level package you have to run `pip install -e .` again.

## How to run

Build an environment, set the plume's base state from the fire's surface fluxes, and ascend.

```python
import numpy as np
from plume_model import Environment, PlumeBase, Plume, MortonEntrainment

# Define on any ascending height grid z, T, Td, p, u, v in SI units.
env = Environment(z, T, Td, p, u, v)

# The fire's surface fluxes set the plume's excesses at its base.
base = PlumeBase(H=50e3, LE=5e3, area=1e6, env=env)

plume = Plume(env, base, MortonEntrainment(fac_ent=1.0, beta=0.5, c_det=2.0))
plume.ascend()

print(plume.z_top, plume.w.max(), plume.area[plume.k_top])
```

After `ascend()`, every profile is an attribute of `plume`: the plume (`w`, `area`, `thetav`,
`thetal`, `T`, `Tv`, `qt`, `buoy`, `mass_flux`, `u`, `v`, `x`, `y`), the environment on
the plume's own grid (`theta_env`, `thetav_env`, `T_env`, `Td_env`, `p_env`, `rho_env`, 
`qt_env`, `u_env`, `v_env`), the entrainment and detrainment rates (`eps`, `delta`, `delta_dyn`, 
`entrainment`, `detrainment`) and the two terms of the `w²` budget (`dw2dz_buoy`, `dw2dz_drag`). 
`k_top`, `z_top`, `k_lcl` and `stopped` describe where the ascent ended and whether it condensed.

If the environment comes as potential temperature and total water, as LES output usually
does, build it with `environment_from_theta(z, theta, qt, p, u, v)` instead.

Classic non-entraining parcel theory is the same integrator with entrainment switched off:

```python
plume = Plume(env, base, MortonEntrainment(fac_ent=0.0), full_ascent=True).ascend()
```

### Validation runs

Validation cases are in `validation/` :

```bash
python validation/crosscheck_js.py
```

Plot styles come from `config`.

```python
import config
config.apply()
```

### Cross-check against (wildfire-meteo-dmt)[https://github.com/wildfire-meteo/wildfire-meteo-dmt]

`validation/crosscheck_js.py` integrates a set of environments in both this package and the
original JavaScript, and compares every returned array. It is local and optional: it needs
`node` and a checkout of `wildfire-meteo-dmt`, found via `$WILDFIRE_METEO_DMT` or at
`../wildfire-meteo-dmt`, and skips with a message when either is absent.

```bash
WILDFIRE_METEO_DMT=/path/to/wildfire-meteo-dmt python validation/crosscheck_js.py
```
