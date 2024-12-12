import numpy as np 
import matplotlib.pyplot as plt 
from PIL import Image
import io
from scipy.signal import windows
import sys

import matplotlib as mpl
from IPython.display import Image as ipyimage
from pathlib import Path

from mw_fdtd import FDTD_TM_2D
from mw_fdtd.plotting import get_window_fields
from mw_fdtd.utils import conv, const

dir_ = Path(__file__).parent

np.set_printoptions(suppress=True, precision=4)


imax = 1000
kmax = 500

imax_mw = 350
kmax_mw = 350

nmax = 1600
fmax = 3e9
f0 = 2e9


rotation = 20
capture_n=800

grid = FDTD_TM_2D(imax, kmax, nmax, fmax)

grid.add_soft_source(f0, 150, 150, 240, 20, axis="z")
grid.add_pml_all_sides(50)
er_gradient = np.ones(kmax)
er_gradient[int(kmax * .75):] = 1.1
grid.set_er_profile(np.arange(kmax), er_gradient)

mw_grid = FDTD_TM_2D(imax_mw, kmax_mw, nmax, fmax)

mw_grid.set_er_profile(np.arange(kmax), er_gradient)
mw_grid.rotate_grid(20, init=True)
mw_grid.add_soft_source(f0, 150, 150, 240, 20, axis="z")
mw_grid.add_pml_all_sides(50)
mw_grid.set_capture(capture_n, nmax, imax_mw - 50, rotation=-20)

n_step = 10
n_save = nmax // n_step

ez_fg = np.zeros((n_save,) + grid.ez_shape, dtype=grid.dtype)
mw_outline = []
ez_mw = np.zeros((n_save,) + mw_grid.ez_shape, dtype=grid.dtype)
ez_loc = np.zeros((n_save, 3) + mw_grid.ez_shape, dtype=grid.dtype)


def func_full(n, ex, ez, hy):

    if (n % n_step) == 0:
        ez_fg[n // n_step] = ez
        grid.get_grid_outline()

def func_mw(n, ex, ez, hy):

    if (n % n_step) == 0:
        ez_mw[n // n_step] = ez
        mw_outline.append(mw_grid.get_grid_outline())
        ez_loc[n // n_step] = np.array(mw_grid.ez_loc)

grid.run(func_full)
mw_grid.run(func_mw, mw_border=60)

mw_grid_s = FDTD_TM_2D(imax_mw, kmax_mw, nmax, fmax)

mw_grid_s.set_er_profile(np.arange(kmax), er_gradient)
mw_grid_s.translate_grid_center(mw_grid.grid_center)
mw_grid_s.rotate_grid(-20, init=True)
mw_grid_s.add_tfsf_line_source(mw_grid.capture["ez_data"], mw_grid.capture["hy_data"], x0=imax_mw // 2)
mw_grid_s.add_pml_all_sides(50)

ez_mw_s = np.zeros((n_save,) + mw_grid_s.ez_shape, dtype=np.float32)
mw_s_outline = []
ez_s_loc = np.zeros((n_save, 3) + mw_grid_s.ez_shape, dtype=grid.dtype)

def func_mw_s(n, ex, ez, hy):

    if (n % n_step) == 0:
        ez_mw_s[n // n_step] = ez
        mw_s_outline.append(mw_grid_s.get_grid_outline())
        ez_s_loc[n // n_step] = np.array(mw_grid_s.ez_loc)

mw_grid_s.run(func_mw_s, mw_border=60)

# plots
#%%
fig, axes = plt.subplot_mosaic(
    [["upper", "upper", "upper", "right"], ["lower_left", "lower_center", "lower_right", "right"]], 
    figsize=(15, 10), 
    height_ratios=[1, 0.5],
    width_ratios=[1, 1, 1, 0.2]
)


ax1 = axes["upper"]
ax2 = axes["lower_center"]
ax3 = axes["lower_left"]
ax4 = axes["lower_right"]
ax_cb = axes["right"]

ax1.set_xlim([0, imax])
ax1.set_ylim([0, kmax])

ax2.set_xlim([0, imax_mw])
ax2.set_ylim([0, kmax_mw])

ax3.set_xlim([0, imax_mw])
ax3.set_ylim([0, kmax_mw])

ax4.set_xlim([0, imax_mw])
ax4.set_ylim([0, kmax_mw])

src0_loc = grid.sources[0]["loc"]
ax1.plot(src0_loc[0], src0_loc[2], linewidth=3)

norm = mpl.colors.Normalize(vmin=-100, vmax=0)
map = mpl.cm.ScalarMappable(norm=norm, cmap="terrain_r")
fig.colorbar(map, ax=ax_cb, label="|Ez| [dB]", ticks=np.arange(-100, 20, 20))
ax_cb.set_axis_off()

ax1.set_aspect("equal")
ax2.set_aspect("equal")
ax3.set_aspect("equal")
ax4.set_aspect("equal")
fig.tight_layout()
images = []
pcm = []

outline_ln = []

ez_fg_db = conv.db20_v(np.where(np.abs(ez_fg) < 1e-16, 1e-16, ez_fg))
ez_mw_db = conv.db20_v(np.where(np.abs(ez_mw) < 1e-16, 1e-16, ez_mw))
ez_mw_s_db = conv.db20_v(np.where(np.abs(ez_mw_s) < 1e-16, 1e-16, ez_mw_s))

z_loc, x_loc = np.meshgrid(np.arange(ez_fg_db.shape[2]), np.arange(ez_fg_db.shape[1]))
zw_loc, xw_loc = np.meshgrid(np.arange(ez_mw_db.shape[2]), np.arange(ez_mw_db.shape[1]))

for n in range(n_save):
    sys.stdout.write(f"\rGenerating Frame: {n}/{n_save}\t\t\t\t")
    [p.remove() for p in pcm]
    [ln.remove() for ln in outline_ln]
    pcm.clear()
    outline_ln.clear()

    n_s = n - (capture_n // n_step)

    pcm1 = ax1.pcolormesh(x_loc, z_loc, ez_fg_db[n], vmin=-100, vmax=0, shading='nearest', cmap="terrain_r")

    pcm2 = ax2.pcolormesh(xw_loc, zw_loc, ez_mw_db[n], vmin=-100, vmax=0, shading='nearest', cmap="terrain_r")
    if n_s >= 0:
        pcm4 = ax4.pcolormesh(xw_loc, zw_loc, ez_mw_s_db[n_s], vmin=-100, vmax=0, shading='nearest', cmap="terrain_r")
        outline_ln += [ax1.plot(w[0], w[2], color="r", linewidth=2)[0] for w in mw_s_outline[n_s]]
        pcm.append(pcm4)

        mw_field = get_window_fields(ez_fg_db[n], ez_s_loc[n_s])
        
    else:
        outline_ln += [ax1.plot(w[0], w[2], color="r", linewidth=2)[0] for w in mw_s_outline[0]]
        mw_field = get_window_fields(ez_fg_db[n], ez_loc[n])

    outline_ln += [ax1.plot(w[0], w[2], color="g", linewidth=2)[0] for w in mw_outline[n]]

    pcm3 = ax3.pcolormesh(xw_loc, zw_loc, mw_field, vmin=-100, vmax=0, shading='nearest', cmap="terrain_r")
    
    pcm.append(pcm1)
    pcm.append(pcm2)
    pcm.append(pcm3)
    

    buf = io.BytesIO()

    ax1.set_title("n={}".format(n * n_step))
    fig.savefig(buf, format="png")
    plt.close("all")

    images.append(Image.open(buf))


gifname = f"full_grid_comparison.gif"
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

# %%
