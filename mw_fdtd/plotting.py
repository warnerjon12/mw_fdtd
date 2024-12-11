import matplotlib.pyplot as plt
from .utils import conv
import numpy as np    

from scipy import ndimage

def plot_field(ax: plt.Axes, field: np.ndarray, vmin=-100, vmax=0, cmap="terrain_r"):
        
    field = np.where(np.abs(field) < 1e-16, 1e-16, field)
    field_db = conv.db20_v(field)
    
    z_loc, x_loc = np.meshgrid(np.arange(field_db.shape[1]), np.arange(field_db.shape[0]))

    pcm = ax.pcolormesh(x_loc, z_loc, field_db, vmin=vmin, vmax=vmax, shading='nearest', cmap=cmap)

    return pcm

def get_window_fields(field_val, field_loc, order=4):
    """
    Interpolates field_val at the field_loc points. Useful for comparing the moving window fields with a larger
    static grid. 
    """

    return ndimage.map_coordinates(
        field_val, np.array([field_loc[0], field_loc[2]]), mode="constant", cval=0, order=order
    )