# pycav

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23414448.svg)](https://doi.org/10.5281/zenodo.23414448)

Gaussian beam and cavity propagation and design.

pycav propagates Gaussian beams through optical systems (spaces, lenses, curved
mirrors at an angle of incidence, interfaces, bulk material, Kerr lenses, and
elements with user-defined ABCD matrices) using separate sagittal and
tangential q-parameters. It computes the eigenmodes of
linear and ring cavities, scans beam sizes over one or two parameters, and
provides an interactive matplotlib cavity designer.

## Installation

pycav requires Python 3.13 or newer.

```bash
pip install git+https://github.com/puls-lab/pycav
```

or, in a uv project:

```bash
uv add git+https://github.com/puls-lab/pycav
```

The example notebooks use interactive matplotlib widgets. To install the
extra dependencies for them, use `pycav[examples]`, e.g.
`pip install "pycav[examples] @ git+https://github.com/puls-lab/pycav"`.

## Usage

```python
import numpy as np
import pycav

# standing-wave cavity at 1 µm wavelength
cav = pycav.Cavity(1e-6, ring=False)
cav.add_mirror(r=0, aoi=0, idn='SESAM')
cav.add_space(d=1, idn='S_TD')
cav.add_mirror(r=1, aoi=np.deg2rad(5), idn='TD')
cav.add_space(d=0.2, idn='TD_R1')
cav.add_mirror(r=2, aoi=-np.deg2rad(5), idn='R1')
cav.add_space(d=0.3, idn='R1_OC')
cav.add_mirror(r=0, aoi=0, idn='OC')

# beam size along the cavity
d, w = cav.get_beam()

# beam size on the disk while varying the SESAM-disk distance
lengths = np.linspace(0, 3, 1000)
w_disk = cav.vary(('S_TD', 'd', lengths), 'TD')
```

More examples are in the [`examples`](examples) directory.

## License

Copyright (C) 2017-2026 Dr. Martin Saraceno

pycav is free software, licensed under the
[GNU General Public License v3.0 or later](LICENSE).

Additional terms under section 7(b) of the GPL apply: copies, modified
versions and works based on pycav that are distributed to others must preserve
the copyright notices and the author attribution. See [NOTICE](NOTICE) for the
exact terms.

## Citing pycav

If you use pycav in work that leads to a scientific publication, please cite
it. Citation metadata is in [CITATION.cff](CITATION.cff); on GitHub, use the
"Cite this repository" button.

Each release is archived on Zenodo. To cite pycav in general, use the DOI
[10.5281/zenodo.23414448](https://doi.org/10.5281/zenodo.23414448), which
always refers to the latest version; the Zenodo page also lists a DOI for
each individual version.
