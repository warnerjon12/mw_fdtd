# mw-fdtd

Moving window FDTD implementation (work in progress). Window can follow arbitrary paths through a geometry (limited
to 10 degree rotations at a time).

## Installation

Requires Python 3.11. In the top level directory, install the python dependencies with,

```bash
pip install -r requirements.txt
```

## Usage

```python
import mw_fdtd
```

This script demonstrates a simple moving window example and shows a side by side comparison with a non-moving grid,

```bash
python scripts/mw_comparison.py
```

A TM FDTD grid can be created with,
```python
from mw_fdtd import FDTD_TM_2D
grid = FDTD_TM_2D(imax, kmax, nmax, fmax)
```

The runner calls a function at each time step that can be used for saving fields,
```python
n_step = 10
def func_(n, ex, ez, hy):
    # at every 10th time step, save the fields
    if (n % n_step) == 0:
        ez_fg[n // n_step] = ez

# the mw_border parameters defines how 
# many grid cells from the right edge field energy is allowed to 
# extend before triggering a grid shift
grid.run(func_, mw_border=60)
```

The moving window moves only in the positive x direction. To follow arbitrary trajectories, the grid can be rotated around the y-axis. The y-axis points into the screen, so a positive rotation is clockwise.
This rotates the permittivity gradient, and subsequent sources applied to the grid.
```python
mw_grid.rotate_grid(10)
```
