import numpy as np

def dtft(xn: np.ndarray, omega: np.ndarray):
    """
    Compute the DTFT of the discrete time signal x[n] over omega (radians per sample).
    """
    n = np.arange(len(xn))
    omega_mesh, n_mesh = np.meshgrid(omega, n)

    # broadcast input sequence across all omega
    x_b = np.broadcast_to(xn[..., None], (len(n), len(omega)))
    # sum across all non-zero n
    Xw = np.sum(x_b * np.exp(-1j* omega_mesh * n_mesh), axis=0)

    return Xw

def dtft_f(xn: np.ndarray, f: np.ndarray, fs: float):
    """
    Compute the DTFT of the discrete time signal x[n] over a frequency range. (cycles per second)
    """
    # convert the continuous time frequency into a discrete frequency range. The discrete frequencies
    # are bounded by -0.5 to 0.5 if there is no aliasing.
    # To convert the continuous time frequency (cycles / sec), into the discrete frequency (cycles / sample). 
    # divide it by the sampling rate fs (samples / sec):
    # (cycles / sec) / (samples / sec) = (cycles / sample)
    fn = f / fs
    return dtft(xn, 2 * np.pi * fn)
        
class const():
    """
    Physical Constants.
    
    eta0  : Wave impedance of free space
    c0    : Speed of light [m/s]
    e0    : Permittivity of free space [F/m]
    u0    : Permeability of free space [H/m]
    k     : Boltzmann's constant [J/K]
    r_e   : Radius of Earth [m]
    eta0  : Plane wave impedance in free space
    """
    eta0 = 376.73
    c0 = 299792458
    c0_in = 11802859050.705801
    e0 = 8.854187817e-12
    k =  1.380649e-23
    u0 = 4 * np.pi * 1e-7
    r_e = 6371
    eta0 = np.sqrt(u0/e0)



class conv():
    """ 
    Unit conversion functions. 
    Follows the same convention as matlab's units_ratio: to_from ()
    i.e. To convert from feet to meters: <meters> = conv.m_ft(<feet>)

    Length:
    m : meter
    km: kilometer
    ft : feet
    in : inch
    nmi : nautical mile
    mi  : mile
    
    Voltage/ Power:
    v :   Volts
    w :   Watts
    dbm  : dB (10log10) ref to 1mW
    db10 : dB (10log10)
    db20 : dB (20log10)

    Impedance: 
    db   : S11 in dB
    z    : Impedance (complex)
    vswr : Voltage standing wave ratio
    gamma: reflection coeff

    Electric Fields:
    e   : plane wave peak electric field (V/m)
    wm2 : average power density (W/m^2)

    Temperature:
    c    : Celcius
    f    : Farenheit
    k    : Kelvin
    """
    # voltage/power:
    v_w        = lambda W, R=50  : np.sqrt(2*R*W)
    w_dbm      = lambda dbm : (10**(dbm/10))*1e-3
    dbm_w      = lambda W   : 10*np.log10(W/1e-3)
    db20_v     = lambda x   : 20*np.log10(np.abs(x))
    db10_v     = lambda x   : 10*np.log10(np.abs(x))
    v_db10     = lambda x   : 10**(np.abs(x)/10)
    v_db20     = lambda x   : 10**(np.abs(x)/20)
    v_dbm      = lambda dbm, R=50 : np.sqrt(2*R*((10**(dbm/10))*1e-3))
    dbm_v      = lambda v,   R=50 : 10*np.log10(((np.abs(v)**2)/(2*R))/1e-3)
    # impedance
    db_vswr    = lambda vswr : 20*np.log10((vswr - 1) / (vswr + 1))
    vswr_db    = lambda db: (1 + np.abs(10**(db/20))) / (1 - np.abs(10**(db/20)))
    gamma_vswr = lambda vswr: (vswr - 1) / (vswr + 1)
    vswr_gamma = lambda gamma: (1 + np.abs(gamma)) / (1 - np.abs(gamma))
    gamma_z    = lambda z, refz=50: (z-refz)/(z+refz)
    z_gamma    = lambda gamma, refz=50: refz*((1+gamma)/(1-gamma))
    # temperature
    c_f        = lambda f: (f - 32) * (5/9)
    f_c        = lambda c: (c * (9/5)) + 32
    k_c        = lambda c: c + 273.15
    c_k        = lambda k: k - 273.15
    # electric field
    wm2_e      = lambda efield : (1/(2*const.eta0)) * (np.abs(efield)**2)
    e_wm2      = lambda wm2 : np.sqrt(2*wm2*const.eta0)


_ratios =   dict(
    ft_m   = 3.28084,
    ft_km  = 3280.84,
    ft_nmi = 6076.12,
    ft_mi = 5280,
    m_mi = 1609.34,
    mi_nmi = 1.15078,
    km_nmi = 1.85200,
    km_mi  = 1.60934,
    in_m   = 39.3701,
    mil_mm = 39.3701
)

# create forward and inverse functions for the simple length conversions in _ratios
for k,v in _ratios.items():
    ksp = k.split('_')
    # use nested lambda to enforce scope
    setattr(conv, k, (lambda x : lambda v : x*v)(v))
    setattr(conv, ksp[1]+'_'+ksp[0], (lambda x : lambda v : x*v)(1/v))
    