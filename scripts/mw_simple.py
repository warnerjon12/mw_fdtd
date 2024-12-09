import numpy as np 
import matplotlib.pyplot as plt 
from PIL import Image
import io

import matplotlib as mpl
from IPython.display import Image as ipyimage
from pathlib import Path

from mw_fdtd import FDTD_TM_2D
from mw_fdtd.plotting import plot_field
dir_ = Path(__file__).parent

np.set_printoptions(suppress=True, precision=4)


imax = 401
kmax = 401

imax_mw = 350
kmax_mw = 350

nmax = 800
fmax = 3e9
f0 = 2e9

grid = FDTD_TM_2D(imax, kmax, nmax, fmax)


grid.add_pml_all_sides()

er_gradient = np.ones(kmax)
er_gradient[int(kmax * .75):] = 1.1

grid.set_er_profile(np.arange(kmax), er_gradient)
grid.rotate_grid(-45, init=True)
grid.add_soft_source(f0, 150, 150, 240, 20, axis="z")

n_step = 20
n_save = nmax // n_step

ez_fg = np.zeros((n_save,) + grid.ez_shape, dtype=np.float32)

fig, (ax1, ax_cb) = plt.subplots(1,2, width_ratios=[1, 0.2])
ax1.set_aspect("equal")
ax1.set_xlim([0, grid.ez_shape[0]])
ax1.set_ylim([0, grid.ez_shape[1]])
fig.canvas.draw()

ax1._active_background = ax1.figure.canvas.copy_from_bbox(ax1.bbox)
images = []

norm = mpl.colors.Normalize(vmin=-100, vmax=0)
map = mpl.cm.ScalarMappable(norm=norm, cmap="terrain_r")
fig.colorbar(map, ax=ax_cb, label="|Ez| [dB]", ticks=np.arange(-100, 20, 20))
ax_cb.set_axis_off()

def func_full(n, ex, ez, hy):

    if (n % n_step) == 0:
        ax1.figure.canvas.restore_region(ax1._active_background)
        plot_field(ax1, ez)
        ax1.figure.canvas.blit(ax1.bbox)

        buf = io.BytesIO()

        # ax1.set_title("n={}".format(n * n_step))
        fig.savefig(buf, format="png")
        plt.close("all")

        images.append(Image.open(buf))


grid.run(func_full, mw_border=30)

gifname = f"full_grid_solve.gif"
images[0].save(
    gifname,
    format="GIF",
    append_images=images,
    save_all=True,
    duration=50,
    loop=0,
    optimize=False,
)

ipyimage(filename=gifname)