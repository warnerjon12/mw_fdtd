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
# many grid cells from the right edge field 
# energy is allowed to extend before triggering a grid shift
grid.run(func_, mw_border=60)
```

The moving window moves only in the positive x direction. To follow arbitrary trajectories, the grid can be rotated around the y-axis. The y-axis points into the screen, so a positive rotation is clockwise.
This rotates the permittivity gradient, and subsequent sources applied to the grid.
```python
mw_grid.rotate_grid(10)
```

To rotate the trajectory of the moving window, the fields are captured and re-introduced on another grid (with the geometry already rotated),
```python
mw_grid.set_capture(n_start, n_end, x0, rotation=10)
```
The `n_start` and `n_end` arguments are the time steps to begin and end the capture. The `x0` argument
is the x coordinate of the vertical line to capture the fields on. This line is then rotated to match
the change in trajectory.

The fields can be mapped onto the next moving grid with,
```python
mw_grid_s = FDTD_TM_2D(imax_mw, kmax_mw, nmax, fmax)
# rotate the permittivity geometry 
mw_grid_s.rotate_grid(10)
# secondary grid begins where the first moving window 
# left off from, move to the grid center location of the
# first moving window
mw_grid_s.translate_grid_center(mw_grid.grid_center)
# setup the TFSF line boundary (left edge only) using the 
# data collected from the first moving window,
mw_grid_s.add_tfsf_line_source(
    mw_grid.capture["ez_data"], 
    mw_grid.capture["hy_data"], 
    x0=imax_mw // 2, 
)
```