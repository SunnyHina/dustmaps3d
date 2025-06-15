# dustmaps3d

An all-sky 3D dust map based on Gaia and LAMOST.

## Installation

```bash
pip install dustmaps3d

from dustmaps3d import dustmaps3d
EBV, dust, sigma, max_d = dustmaps3d(l, b, d)

