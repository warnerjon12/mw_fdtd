import numpy as np  
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.spatial.transform import Rotation
from scipy.interpolate import CubicSpline
from scipy.signal import windows
from scipy.interpolate import interp1d

from mw_fdtd.utils import const, conv

u0 = const.u0
e0 = const.e0
c0 = const.c0
eta0 = const.eta0

class FDTD_TM_2D(object):

    def __init__(
        self, 
        imax: int,
        kmax: int,
        nmax: int,
        fmax: float,
        dt: float = None,
        max_er: float = 1.5,
        cells_per_wavelength: int = 15,
        dtype = np.float32
    ):
        """
        2D transverse magnetic solver. Ex vector points to the right, Ez points up, Hy points into the screen. 
        Grid origin is the bottom left. 

        Parameters:
        -----------
        imax: int
            number of grid cells in x direction
        kmax: int
            number of grid cells in z direction
        nmax: int
            number of time steps
        fmax: float
            maximum frequency [Hz] of any of the sources applied in the grid. 
        dt: float, optional
            time step in seconds. Default is 90% of the Courant stability factor.
        max_er: float, optional
            maximum relative permittivity in the grid
        cells_per_wavelength: int, optional
        
        """
        self.imax = imax
        self.kmax = kmax
        self.nmax = nmax
        self.dtype = dtype

        # compute minimum lambda from the maximum frequency
        vp = c0 / np.sqrt(max_er)
        lam_min = vp / fmax

        # spatial step size, use smallest wavelength
        self.delta = lam_min / cells_per_wavelength

        # compute maximum time step that ensures convergence
        if dt is None:
            S = 0.90 * (1 / np.sqrt(2))
            dt = S * (self.delta / const.c0)
        self.dt = dt

        self.sources = []
        self.tfsf = None
        self.er_profile = None

        self.grid_rotation = 0
        self.shift_x = 0

        # locations of field components in grid cell units
        # origin is the bottom left of grid.
        loc_mid_z = np.arange(0.5, kmax - 0.5, 1)
        loc_end_z = np.arange(0, kmax, 1)
        loc_mid_x = np.arange(0.5, imax - 0.5, 1)
        loc_end_x = np.arange(0, imax, 1)

        loc_ez_z, loc_ez_x = np.meshgrid(loc_mid_z, loc_end_x)
        loc_ex_z, loc_ex_x = np.meshgrid(loc_end_z, loc_mid_x)
        loc_hy_z, loc_hy_x = np.meshgrid(loc_mid_z, loc_mid_x)

        self.ex_loc = np.array([loc_ex_x, np.zeros(loc_ex_z.shape), loc_ex_z])
        self.ez_loc = np.array([loc_ez_x, np.zeros(loc_ez_z.shape), loc_ez_z])
        self.hy_loc = np.array([loc_hy_x, np.zeros(loc_hy_z.shape), loc_hy_z])

        self.ex_loc_static = self.ex_loc.copy()
        self.ez_loc_static = self.ez_loc.copy()
        self.hy_loc_static = self.hy_loc.copy()

        self.ex_shape = self.ex_loc.shape[1:]
        self.ez_shape = self.ez_loc.shape[1:]
        self.hy_shape = self.hy_loc.shape[1:]

        # location of the physical center of the grid
        self.grid_center = np.array([int((self.imax - 1) / 2), 0, int((self.kmax - 1) / 2)], dtype=np.float64)

        # initialize permittivity
        self.epsilon_ex = np.ones(self.ex_shape) * e0
        self.epsilon_ez = np.ones(self.ez_shape) * e0
        self.epsilon_hy = np.ones(self.hy_shape) * e0

        # initialize conductivity
        self.sigmax_ez = np.zeros(self.ez_shape, dtype=self.dtype)
        self.sigmaz_ex = np.zeros(self.ex_shape, dtype=self.dtype)
        self.sigmax_hy = np.zeros(self.hy_shape, dtype=self.dtype)
        self.sigmaz_hy = np.zeros(self.hy_shape, dtype=self.dtype)

        self.tfsf = dict(
            ex_btm=None,
            ex_top=None,
            ez_left=None,
            ez_right=None,
            hy_left=None,
            hy_right=None,
            hy_top=None,
            hy_btm=None,
            sf_wz=0,
            sf_wx=0
        )

        self.capture = None

    def add_soft_source(
        self, 
        f0: float, 
        i0: int, 
        k0: int, 
        width_n: int,
        length: int, 
        axis: str = "x", 
    ):
        """
        Adds a Gaussian modulated sine wave line source (E-field).

        Parameters:
        ----------
        f0: float
            frequency of source
        i0: float
            location of the center of the source in x axis
        k0: float
            location of the center source in z axis
        width_n: int
            time length of pulse in time steps.
        length: int
            physical length of source in grid cell units.
        axis: str
            axis the line source is aligned with. Default is x.
        """
        # width of half pulse in time
        t_half = (self.dt * (width_n // 8))
        # center of the pulse in time
        t0 = (self.dt * (width_n // 2))

        t = np.linspace(0, self.dt * self.nmax, self.nmax)
        # gaussian modulated sine wave source
        source_gms = (np.sin(2*np.pi*f0 * (t - t0)) * np.exp(-((t - t0) / t_half)**2)).astype(self.dtype)

        # vector for source location, origin is at source center
        axis_idx = dict(x=0, z=2)
        source_loc = np.zeros((3, length))
        source_loc[axis_idx[axis]] = np.arange(-(length // 2), (length // 2) + 1, 1)[:length]

        # rotate the source locations, the source is rotated the opposite direction as the geometry
        rot = Rotation.from_euler("xyz", (0, -self.grid_rotation, 0), degrees=True)
        rot_m = rot.as_matrix()
        source_rloc = np.einsum("ij,j...->i...", rot_m, source_loc)

        # rotate the source polarization
        rot_rad = np.deg2rad(-self.grid_rotation)
        if axis == "x":
            source_x = source_gms * np.cos(rot_rad)
            source_z = source_gms * np.sin(rot_rad)
        else: # axis = "z"
            source_z = source_gms * np.cos(rot_rad)
            source_x = -source_gms * np.sin(rot_rad)

        # translate to the source location 
        source_rloc[0] += i0
        source_rloc[2] += k0

        self.sources.append(
            dict(x=source_x, z=source_z, loc=source_rloc)
        )


    def set_er_profile(self, loc: np.ndarray, er: np.ndarray, axis="z"):
        """
        Sets a 1D permittivity profile on the grid.

        Parameters:
        ----------
        loc: np.ndarray
            location of each er point in grid cells. Must extend at least to the grid edge, but can be larger than
            grid to support a moving window.
        er: np.ndarray
            relative permittivity values assigned to each grid cell.
        axis: str
            direction of the profile. Default is a vertical profile along "z" axis.
        """
        axis_max = dict(x=self.imax, z=self.kmax)
        if len(loc) < axis_max[axis]:
            raise ValueError(f"er profile must extend at least to grid edge ({axis_max} grid cells)")

        # use coordinates to pull out the refractivity at the rotated points
        axis_idx = dict(x=0, z=2)[axis]
        self.er_profile = CubicSpline(loc, er), axis_idx

        profile, axis_idx = self.er_profile

        self.epsilon_ex = profile(self.ex_loc[axis_idx]) * e0
        self.epsilon_ez = profile(self.ez_loc[axis_idx]) * e0
        self.epsilon_hy = profile(self.hy_loc[axis_idx]) * e0


    def add_pml(self, d_pml=15, coeff=0.8, direction="x", side="upper"):
        """
        Adds a perfectly matched layer (PML) boundary condition.

        Parameters:
        ----------
        d_pml: int
            width of PML in grid cells, defaults to 15
        coeff: float
            coefficient to scale sigma maximum by, default is 0.8.
        direction: str 
            Direction of absorbed fields ["x", "z"]. "x" will create a vertical PML, "z" will create a horizontal PML.
        side: str
            "upper" (default) or "lower". Determines which side of the grid to place PML. 
        """
        m_pml = 3 # sigma profile order
        sigma_max =  coeff * (m_pml + 1) / (eta0 * self.delta)

        pml_k = (1, 0) if direction == "x" else (0, 1)

        pml_origin = np.array([self.imax - d_pml, self.kmax - d_pml])[..., None, None]

        # get projected distance from the origin along the k vector
        d_pml_ex = np.einsum("i, i...->...", np.array(pml_k), self.ex_loc_static[::2] - pml_origin)
        d_pml_ez = np.einsum("i, i...->...", np.array(pml_k), self.ez_loc_static[::2] - pml_origin)
        d_pml_hy = np.einsum("i, i...->...", np.array(pml_k), self.hy_loc_static[::2] - pml_origin)

        if side == "lower":
            d_pml_ex = np.flip(d_pml_ex, axis=(0, 1))
            d_pml_ez = np.flip(d_pml_ez, axis=(0, 1))
            d_pml_hy = np.flip(d_pml_hy, axis=(0, 1))

        # now define the values of sigma and sigma_m from the profiles
        x_pml, z_pml = pml_k
        
        d_pml_ez = np.clip(d_pml_ez, 0, d_pml)
        d_pml_ex = np.clip(d_pml_ex, 0, d_pml)
        d_pml_hy = np.clip(d_pml_hy, 0, d_pml)

        self.sigmax_ez  += x_pml * sigma_max * ((d_pml_ez) / (d_pml))**m_pml
        self.sigmaz_ex  += z_pml * sigma_max * ((d_pml_ex) / (d_pml))**m_pml
        self.sigmax_hy += x_pml * sigma_max * ((d_pml_hy) / (d_pml))**m_pml
        self.sigmaz_hy += z_pml * sigma_max * ((d_pml_hy) / (d_pml))**m_pml

    def add_pml_all_sides(self, d_pml: int = 15, coeff: float = 0.8):
        """
        Places a PML on all four edges of the grid. See add_pml() for parameter options.
        """
        self.add_pml(d_pml, coeff, direction="x", side="lower")
        self.add_pml(d_pml, coeff, direction="z", side="upper")
        self.add_pml(d_pml, coeff, direction="z", side="lower")
        self.add_pml(d_pml, coeff, direction="x", side="upper")

    def rotate_grid(self, rotation: float):
        """
        Rotates the er profile around the y axis. FDTD time stepping will start with this rotation already applied.
        Invalidates all previously defined sources-- add sources after calling this method.
        """
        
        # ex_loc has shape (3, imax, kmax), broadcast grid center location across the imax and kmax dimensions
        grid_center = self.grid_center[..., None, None]
        
        # rotate the grid locations around the grid center 
        rot = Rotation.from_euler("xyz", (0, rotation, 0), degrees=True).as_matrix()
        self.ex_loc = np.einsum("ij,j...->i...", rot, self.ex_loc - grid_center) + grid_center
        self.ez_loc = np.einsum("ij,j...->i...", rot, self.ez_loc - grid_center) + grid_center
        self.hy_loc = np.einsum("ij,j...->i...", rot, self.hy_loc - grid_center) + grid_center

        # map the epsilon profile to the new rotated locations
        if self.er_profile is not None:
            profile, axis_idx = self.er_profile

            self.epsilon_ex = profile(self.ex_loc[axis_idx]) * e0
            self.epsilon_ez = profile(self.ez_loc[axis_idx]) * e0
            self.epsilon_hy = profile(self.hy_loc[axis_idx]) * e0

        # invalidate existing sources
        self.sources = []

        self.grid_rotation = rotation
    
    def shift_mw(self, shift_x: int):
        """
        Shifts the FDTD grid to the right by shift_x. Must be an integer.
        """

        ang_rad = np.deg2rad(-self.grid_rotation)
        step_x = np.cos(ang_rad) * shift_x
        step_z = np.sin(ang_rad) * shift_x

        self.shift_x += shift_x
        
        shift_v = np.array([step_x, 0, step_z])
        self.grid_center += shift_v
        
        # increment the x and z location of each field
        self.ex_loc += shift_v[..., None, None]
        self.ez_loc += shift_v[..., None, None]
        self.hy_loc += shift_v[..., None, None]

        # shift the permittivity values 
        self.epsilon_ex[:-1] = self.epsilon_ex[1:]
        self.epsilon_ez[:-1] = self.epsilon_ez[1:]
        self.epsilon_hy[:-1] = self.epsilon_hy[1:]

        # use the er profile to get the values at the right edge of the grid where a new column of grid cells
        # was introduced
        if self.er_profile is not None:
            profile, axis_idx = self.er_profile
            self.epsilon_ex[-1] = profile(self.ex_loc[axis_idx, -1]) * e0
            self.epsilon_ez[-1] = profile(self.ez_loc[axis_idx, -1]) * e0
            self.epsilon_hy[-1] = profile(self.hy_loc[axis_idx, -1]) * e0

    def get_grid_outline(self):
        """
        Returns the grid cell locations of the grid boundaries.
        """
        top = self.ex_loc[:, :, -1].copy()
        btm = self.ex_loc[:, :, 0].copy()
        left = self.ez_loc[:, 0, :].copy()
        right = self.ez_loc[:, -1, :].copy()

        return [top, right, btm, left]


    def compute_fdtd_coeff(self):
        sigmax_m_hy = self.sigmax_hy * (u0 / self.epsilon_hy)
        sigmaz_m_hy = self.sigmaz_hy * (u0 / self.epsilon_hy)

        dt = self.dt
        delta = self.delta

        # coefficients in front of the previous time values of E/H. These are the same as the 1D case
        Ca_x = (2 * self.epsilon_ez - (self.sigmax_ez * dt)) / (2 * self.epsilon_ez + (self.sigmax_ez * dt))
        Ca_z = (2 * self.epsilon_ex - (self.sigmaz_ex * dt)) / (2 * self.epsilon_ex + (self.sigmaz_ex * dt))
        Da_x = (2 * u0 - (sigmax_m_hy * dt)) / (2 * u0 + (sigmax_m_hy * dt))
        Da_z = (2 * u0 - (sigmaz_m_hy * dt)) / (2 * u0 + (sigmaz_m_hy * dt))
        # coefficients in front of the difference terms in amperes and faradays equations
        Cb_x = (2 * dt) / ((2 * self.epsilon_ez + (self.sigmax_ez * dt)) * delta)  #(dt / (e0 * delta))
        Cb_z = (2 * dt) / ((2 * self.epsilon_ex + (self.sigmaz_ex * dt)) * delta)  #(dt / (e0 * delta))
        Db_x = (2 * dt) / ((2 * u0 + (sigmax_m_hy * dt)) * delta) 
        Db_z = (2 * dt) / ((2 * u0 + (sigmaz_m_hy * dt)) * delta) 

        return Ca_x, Ca_z, Cb_x, Cb_z, Da_x, Da_z, Db_x, Db_z

    def tfsf_1d_propagtor(
        self, 
        phi: float,
        f0: float, 
        width_n: int,
        sf_wx = 30,
        sf_wz = 30,
        mx=2,
    ):
        """
        Create a total field/ scattered field boundary using a 1D plane wave source.

        References:
        -----------
        T. Tan and M. Potter, 
        "1-D Multipoint Auxiliary Source Propagator for the Total-Field/Scattered-Field FDTD Formulation," 
        in IEEE Antennas and Wireless Propagation Letters, vol. 6, pp. 144-148, 2007, doi: 10.1109/LAWP.2007.891959.

        Parameters:
        -----------
        phi: float
            angle of incident plane wave
        f0: float
            frequency of guassian modulated sine wave source [Hz]
        width_n: int
            width of source in time steps
        sf_wx: int
            width of the scattered field region in the x direction, grid cell units
        sf_wx: int
            width of the scattered field region in the z direction, grid cell units
        """

        t_half = (self.dt * (width_n // 8))
        # center of the pulse in time
        t0 = (self.dt * (width_n // 2))

        mz = int(np.around(np.clip(mx * np.tan(np.deg2rad(phi)), 1, 40)))
        phi_rad = np.arctan(mz/mx)

        px = np.cos(phi_rad)
        pz = np.sin(phi_rad)
        dr = px * self.delta / mx # py * delta / my

        rmax = mx * self.imax + mz * self.kmax
        # propagation vector k can be at arbitrary angles, assume it is 0 degrees for the
        # 1D propagator (z-direction). It is a TEM wave so there is no ez component
        e_inc = np.zeros((self.nmax, int((rmax)/ 2) -1), dtype=self.dtype)
        h_inc = np.zeros((self.nmax, rmax), dtype=self.dtype)

        Sx = self.dt * const.c0 / self.delta
        Sz = self.dt * const.c0 / self.delta

        # advanced by dr
        source_dt = (dr / const.c0)

        for n in range(1, self.nmax-1):
            t = n * self.dt

            h_inc[n, 0] = (np.sin(2*np.pi*f0 * self.dt * (n)) / const.eta0) * np.exp(-((t - t0) / t_half)**2)
            h_inc[n, 1] = (np.sin(2*np.pi*f0 * self.dt * (n) - 2*np.pi*f0 *source_dt) / const.eta0) * np.exp(-((t - t0 - source_dt) / t_half)**2)

            h_inc[n+1] = -h_inc[n -1] + 2*(1 - Sx**2 - Sz**2) * h_inc[n] 
            h_inc[n+1, mx:-mx] += Sx**2 * (h_inc[n, :-(mx*2)] + h_inc[n, mx*2:])
            h_inc[n+1, mz:-mz] += Sz**2 * (h_inc[n, :-(mz*2)] + h_inc[n, mz*2:])

            # normal update for hinc
            e_inc[n] = e_inc[n-1] - (self.dt / (e0 * dr * 2)) * np.diff(h_inc[n, ::2])#

        # get the maximum e index of the fields at each time step
        max_e_arg = np.argmax(e_inc, axis=1)
        # the time where the e field reaches the end of the grid (max grid index)
        inc_n_end = np.argmax(max_e_arg)

        # clip the inc fields after the pulse reaches the end
        e_inc[inc_n_end:] = 0
        h_inc[inc_n_end:] = 0

        # create projection vectors, see figure 5.11 in taflove
        # propagation origin is bottom left corner
        tf_imax = self.imax - (sf_wx * 2)
        tf_kmax = self.kmax - (sf_wz * 2)

        # bottom and top tfsf edge distance vector for ex components,
        # ordered x, then z
        r_ex_btm = np.arange(0.5, tf_imax-1), np.zeros(tf_imax - 1)
        r_ex_top = np.arange(0.5, tf_imax-1), np.ones(tf_imax - 1) * (tf_kmax -1)

        # left and right tfsf edge distance vector for ez components
        r_ez_left = np.zeros(tf_kmax - 1), np.arange(0.5, tf_kmax -1)
        r_ez_right = np.ones(tf_kmax - 1) * (tf_imax - 1), np.arange(0.5, tf_kmax-1)

        # we also need the hy components just outside the total field region, these are
        # offset by a half grid cell from the ex and ez components
        r_hy_btm = r_ex_btm[0], r_ex_btm[1] - 0.5
        r_hy_top = r_ex_btm[0], r_ex_top[1] + 0.5

        r_hy_left = r_ez_left[0] - 0.5, r_ez_left[1]
        r_hy_right = r_ez_right[0] + 0.5, r_ez_right[1]

        # k vector. ordered x, then z
        k_inc = np.cos(phi_rad), np.sin(phi_rad)

        # projection distance vectors for each component, in grid cells.
        # the distances are relative to the total field corner, we want to move the 
        # distance so it's relative to the full grid, add the total field corner distance
        tf_d = np.array([sf_wx, sf_wz])[:, None]

        d_ex_btm = np.einsum("i, ij->j", np.array(k_inc), np.array(r_ex_btm) + tf_d)
        d_ex_top = np.einsum("i, ij->j", np.array(k_inc), np.array(r_ex_top) + tf_d)
        d_ez_left = np.einsum("i, ij->j", np.array(k_inc), np.array(r_ez_left) + tf_d)
        d_ez_right = np.einsum("i, ij->j", np.array(k_inc), np.array(r_ez_right) + tf_d)

        d_hy_btm = np.einsum("i, ij->j", np.array(k_inc), np.array(r_hy_btm) + tf_d)
        d_hy_top = np.einsum("i, ij->j", np.array(k_inc), np.array(r_hy_top) + tf_d)
        d_hy_left = np.einsum("i, ij->j", np.array(k_inc), np.array(r_hy_left) + tf_d)
        d_hy_right = np.einsum("i, ij->j", np.array(k_inc), np.array(r_hy_right) + tf_d)

        # set each incident field component based on the incident angle
        # if phi_inc is zero, only ez and hy should be nonzero, 
        # if phi_inc is 90, only ex and hy should be nonzero
        # interpolate the incident grid using distance vectors
        re_inc = np.arange(dr, len(e_inc[0]) * 2 * dr, 2 * dr)
        f = interp1d(re_inc, e_inc, axis=-1)

        self.tfsf["ex_btm"] = f(d_ex_btm * self.delta) * np.sin(phi_rad)
        self.tfsf["ex_top"] = f(d_ex_top * self.delta) * np.sin(phi_rad)
        self.tfsf["ez_left"] = -f(d_ez_left * self.delta) * np.cos(phi_rad)
        self.tfsf["ez_right"] = -f(d_ez_right * self.delta) * np.cos(phi_rad)

        # hy does not change based on the incident angle
        rh_inc = np.arange(0, rmax  * dr, dr)
        f = interp1d(rh_inc, h_inc, axis=-1)
        self.tfsf["hy_left"] = f(d_hy_left * self.delta)
        self.tfsf["hy_right"] = f(d_hy_right * self.delta)
        self.tfsf["hy_top"] = f(d_hy_top * self.delta)
        self.tfsf["hy_btm"] = f(d_hy_btm * self.delta)
        self.tfsf["sf_wx"] = sf_wx
        self.tfsf["sf_wz"] = sf_wz

    def add_tfsf_line_source(self, ez, hy, x0, window=None):
        """
        Add a TFSF boundary on left edge of grid. Optionally apply a scipy.signal.window function
        to attenuate the egdes near the PML.
        """
        if window is None:
            window = np.ones_like(hy)
        else:
            window = np.broadcast_to(window(len(hy[0]))[None], hy.shape).copy()

        self.tfsf = dict()
        self.tfsf["hy_left"] = window * hy.copy()
        self.tfsf["ez_left"] = window * ez.copy()
        self.tfsf["sf_wx"] = x0
        self.tfsf["sf_wz"] = 0

    def set_capture(self, n_start, n_stop, x0, rotation=0):
        """
        Setup a TFSF capture on a verical line in the grid.

        Parameters:
        -----------
        n_start: int
            beginning time step of capture
        n_stop: int
            ending time step of capture
        x0: int
            x locatino of vertical line
        rotation: float
            rotation in degrees to rotate capture line by (around y axis)
        """
        rot_s = Rotation.from_euler("xyz", (0, rotation, 0), degrees=True).as_matrix()

        x0_c = (self.imax - 1) // 2
        center_m = np.array([int((self.imax - 1) // 2), 0, int((self.kmax - 1) // 2)])[..., None]
        ez_capture_loc = np.einsum("ij,j...->i...", rot_s, (self.ez_loc_static)[:, x0_c] - center_m) + center_m
        hy_capture_loc = np.einsum("ij,j...->i...", rot_s, (self.hy_loc_static)[:, x0_c-1] - center_m) + center_m

        ez_capture_loc += np.array([x0 - x0_c, 0, 0])[..., None]
        hy_capture_loc += np.array([x0 - x0_c, 0, 0])[..., None]

        # convert to indices, first location of hy is offset by half a grid cell in z and x axis
        hy_capture_loc -= np.array([0.5, 0, 0.5])[..., None]
        # convert to indices, first location of ez is offset by half a grid cell in z axis
        ez_capture_loc -= np.array([0, 0, 0.5])[..., None]

        # only x and z locations are used for interpolation, remove y
        hy_capture_loc = hy_capture_loc[0::2]
        ez_capture_loc = ez_capture_loc[0::2]

        n_duration = n_stop - n_start
        self.capture = dict(
            n_start=n_start, 
            n_stop=n_stop, 
            hy_loc=hy_capture_loc, 
            ez_loc=ez_capture_loc, 
            rotation=rotation, 
            hy_data=np.zeros((n_duration,) + hy_capture_loc.shape[1:]),
            ez_data=np.zeros((n_duration,) + ez_capture_loc.shape[1:])
        )

    def translate_grid_center(self, position):
        """
        Moves the center of the grid to position (xyz vector)
        """
        # figure out how far to move the grid by taking the difference of the current location and the desired one
        translate = position - self.grid_center

        # move the grid center
        self.grid_center += translate
        
        # move each of the field locations
        self.ex_loc += translate[..., None, None]
        self.ez_loc += translate[..., None, None]
        self.hy_loc += translate[..., None, None]

        # interpolate the er profile at the new grid cell locations
        if self.er_profile is not None:
            profile, axis_idx = self.er_profile
            self.epsilon_ex = profile(self.ex_loc[axis_idx]) * e0
            self.epsilon_ez = profile(self.ez_loc[axis_idx]) * e0
            self.epsilon_hy = profile(self.hy_loc[axis_idx]) * e0

    def run(self, iter_func = None, mw_border=None):
        K_AXIS = 1
        I_AXIS = 0

        imax = self.imax
        kmax = self.kmax

        ez = np.zeros((imax, kmax - 1), dtype=self.dtype)
        ex = np.zeros((imax - 1, kmax), dtype=self.dtype)
        hyx = np.zeros((imax - 1, kmax -1), dtype=self.dtype)
        hyz = np.zeros((imax - 1, kmax -1), dtype=self.dtype)

        # tfsf boundary fields
        ex_top = self.tfsf.get("ex_top", None)
        ex_btm = self.tfsf.get("ex_btm", None)
        ez_left = self.tfsf.get("ez_left", None)
        ez_right = self.tfsf.get("ez_right", None)
        hy_top = self.tfsf.get("hy_top", None)
        hy_btm = self.tfsf.get("hy_btm", None)
        hy_left = self.tfsf.get("hy_left", None)
        hy_right = self.tfsf.get("hy_right", None)

        # width of the total field portion of the grid
        tfsf_tf_x = self.imax - (2 * self.tfsf["sf_wx"])
        tfsf_tf_z = self.kmax - (2 * self.tfsf["sf_wz"])

        tfsf_x0 = self.tfsf["sf_wx"]
        tfsf_x1 = tfsf_x0 + tfsf_tf_x -1

        tfsf_z0 = self.tfsf["sf_wz"]
        tfsf_z1 = tfsf_z0 + tfsf_tf_z -1

        # correct for the delay caused by the TFSF source, advances the grid to catch up to the actual location
        # of the fields.
        if self.tfsf["sf_wx"] != 0 and mw_border is not None:
            self.shift_mw(self.imax - mw_border - tfsf_x0)

        Ca_x, Ca_z, Cb_x, Cb_z, Da_x, Da_z, Db_x, Db_z = self.compute_fdtd_coeff()

        # number of tfsf grid cells in the x direction that are remaining as the source slides out of the grid.
        # only used for bottom and top boundaries
        h_tfsf_x_clip = tfsf_tf_x - (tfsf_x1 - tfsf_x0) - 1

        start_capture, end_capture = self.nmax, self.nmax

        # set up variables if there is a capture specified
        if self.capture is not None:
            start_capture, end_capture = self.capture["n_start"], self.capture["n_stop"]
            rot_rad_s = np.deg2rad(-self.capture["rotation"])
            mkwargs = dict(mode="constant", cval=0, order=3)
        
        for n in range(self.nmax-1):
            
            if n > start_capture and n < end_capture:
                n_c = n - start_capture
                ez_inc_left_z = ndimage.map_coordinates(ez, self.capture["ez_loc"], **mkwargs)
                ez_inc_left_x = ndimage.map_coordinates(ex, self.capture["ez_loc"], **mkwargs)

                self.capture["ez_data"][n_c] = (-ez_inc_left_x * np.sin(rot_rad_s) + ez_inc_left_z * np.cos(rot_rad_s))
                self.capture["hy_data"][n_c] = ndimage.map_coordinates(hyx + hyz, self.capture["hy_loc"], **mkwargs)
                # cancel the moving window once a capture starts
                mw_border = None

            elif mw_border is not None:
                energy_grid = np.sqrt(np.abs(ex[-mw_border:,  1:])**2 + np.abs(ez[-mw_border, :])**2)
                energy_at_right = np.any(energy_grid > 1e-6)
        
                # shift window one cell to the right, dt ensures that energy can't make it from
                # one cell to the next in a single time step, so moving one cell at a time should
                # keep the pulse in the window
                if energy_at_right:
                    self.shift_mw(1)
                    Ca_x, Ca_z, Cb_x, Cb_z, Da_x, Da_z, Db_x, Db_z = self.compute_fdtd_coeff()

                    ez[:-1] = ez[1:]
                    ez[-1] = 0
                    hyx[:-1] = hyx[1:]
                    hyx[-1] = 0
                    hyz[:-1] = hyz[1:]
                    hyz[-1] = 0
                    ex[:-1] = ex[1:]
                    ex[-1] = 0

                    # shift tfsf boundaries
                    tfsf_x0 -= 1
                    tfsf_x1 -= 1

                    if tfsf_x0 < 0:
                        ez_left = None
                        hy_left = None
                        tfsf_x0 = 0
                    if tfsf_x1 < 0:
                        ez_right = None
                        hy_right = None
                        ex_btm = None
                        ex_top = None
                        hy_top = None
                        hy_btm = None
                        hy_left = None
                        hy_right = None
                        tfsf_x1 = 0

                    # number of points to drop from the top and bottom boundary as it slides out of view
                    h_tfsf_x_clip = tfsf_tf_x - (tfsf_x1 - tfsf_x0) - 1


            # # update the total/scattered field grid
            # compute h_y for the next half time step (n+0.5)
            # first compute the spatial difference between the ex components along the z direction at the previous 
            # half time step.
            hyz = (Da_z * hyz) - Db_z * (np.diff(ex, axis=K_AXIS)) #h_yz[n+1]
            hyx = (Da_x * hyx) + Db_x * (np.diff(ez, axis=I_AXIS)) #h_yx[n+1]

            if ex_btm is not None:
                # the hy component just below the tfsf boundary is in the scattered region, but uses a ex value in the total region.
                # subtract the incident field value.
                hyz[tfsf_x0:tfsf_x1, tfsf_z0 -1] += Db_z[tfsf_x0:tfsf_x1, tfsf_z0] * ex_btm[n, h_tfsf_x_clip:]

            if ex_top is not None:
                # the hy component just above the tfsf boundary in the scattered region uses a ex value in the total region, 
                # subtract incident field to remove it from the update equation
                hyz[tfsf_x0:tfsf_x1, tfsf_z1] -= Db_z[tfsf_x0:tfsf_x1, tfsf_z1] * ex_top[n, h_tfsf_x_clip:]

            if ez_left is not None and n < len(ez_left) -1:            
                # the hy component to the left of the total field in the scattered region uses a ez value in the total region,
                # subtract incident field
                hyx[tfsf_x0 -1, tfsf_z0:tfsf_z1] -= Db_x[tfsf_x0, tfsf_z0:tfsf_z1] * ez_left[n]

            if ez_right is not None:
                # the hy component to the right of the total field in the scattered region uses a ez value in the total region,
                # subtract incident field
                hyx[tfsf_x1, tfsf_z0:tfsf_z1] += Db_x[tfsf_x1, tfsf_z0:tfsf_z1] * ez_right[n]
            
            hy = hyx + hyz

            # compute ex for the next full time step (n+1)
            # spatial difference of the H field along z direction at the previous (n+0.5) time step
            # leave the edge cells along z direction (k) at zero (PEC)
            ex[:, 1:-1] = Ca_z[:, 1: -1] * ex[:, 1: -1] - Cb_z[:, 1: -1] * np.diff(hy, axis=K_AXIS)

            if hy_btm is not None:
                # the ex component in the total region on the bottom boundary of the total region uses a scattered field value of hy below it.
                # Add the inc value to scattered hy value
                ex[tfsf_x0:tfsf_x1, tfsf_z0] += Cb_z[tfsf_x0:tfsf_x1, tfsf_z0 - 1] * hy_btm[n+1, h_tfsf_x_clip:]

            if hy_top is not None:
                # the ex component on the top boundary of the tfsf region uses a hy component in the scattered region above it. 
                # Add the incident field value to the scattered value to get the total field.
                ex[tfsf_x0:tfsf_x1, tfsf_z1] -= Cb_z[tfsf_x0:tfsf_x1, tfsf_z1] * hy_top[n+1, h_tfsf_x_clip:]

            # compute ez for the next full time step (n+1).
            # spatial difference of the H field along x direction at the previous (n+0.5) time step.
            # leave the edge cells along x direction (i) at zero (PEC)
            ez[1:-1] = Ca_x[1:-1] * ez[1:-1] + Cb_x[1:-1] * np.diff(hy, axis=I_AXIS)

            if hy_left is not None and n < len(hy_left) -1:
                # the ez component on the left boundary of the tfsf region uses a hy component in the scattered region.
                # add the incident field value 
                ez[tfsf_x0, tfsf_z0:tfsf_z1] -= Cb_x[tfsf_x0 -1, tfsf_z0:tfsf_z1] * hy_left[n+1]

            if hy_right is not None:
                # the ez component on the right boundary of the tfsf region uses a hy component in the scattered region.
                # add the incident field value
                ez[tfsf_x1, tfsf_z0:tfsf_z1] += Cb_x[tfsf_x1, tfsf_z0:tfsf_z1] * hy_right[n+1]

            # soft sources
            for s in self.sources:
                x_s = s["loc"][0].astype(int) - self.shift_x
                z_s = s["loc"][2].astype(int)
                if (np.min(x_s) < 0):
                    continue

                ex[x_s, z_s] -= (self.dt / self.epsilon_ex[x_s, z_s]) * s["x"][n]
                ez[x_s, z_s] += (self.dt / self.epsilon_ez[x_s, z_s]) * s["z"][n]

            if iter_func is not None:
                iter_func(n, ex, ez, hy)

