################################################################################
#                                                                              #
#               Multi-Layer Snow Storage RC Thermal Model                      #
#                                                                              #
#  Authors : Siim Erik Pugal, siim.pugal@taltech.ee                            #
#            Hossein Alimohammadi,  hossein.alimohammadi@taltech.ee            #
#  Date    : March 2026                                                        #
#  License : MIT                                                               #
#                                                                              #
# ---------------------------------------------------------------------------- #
#                                                                              #
#  DESCRIPTION                                                                 #
#                                                                              #
#  This script simulates the thermal evolution and melt of a seasonal snow     #
#  storage pile using a 3-layer RC (resistance-capacitance) network model.     #
#  The snow pile is divided into three vertical layers (surface, middle,       #
#  bottom), each with its own temperature and liquid water content (LWC).      #
#  An insulating woodchip layer sits above the snow, and ground heat exchange  #
#  is handled at the base.  Meteorological forcing is read from an hourly      #
#  CSV file and interpolated to the sub-hourly simulation time step.           #
#                                                                              #
# ---------------------------------------------------------------------------- #
#                                                                              #
#  PHYSICS OVERVIEW                                                            #
#                                                                              #
#  1. RC THERMAL NETWORK                                                       #
#     The three snow layers (T1 = top, T2 = middle, T3 = bottom) are           #
#     connected by conductive resistances R_12 and R_23.  The top layer        #
#     also exchanges heat with the atmosphere via a combined surface           #
#     resistance R_a2s = R_eff + R_ins, where:                                 #
#       R_eff = 1 / (h_conv + h_rad)   [air-side convection + LW radiation]    #
#       R_ins = Hi / k_eff             [insulation layer, thickness Hi]        #
#     The ODE system dT/dt for the three layers is integrated with a 4th-      #
#     order Runge-Kutta (RK4) scheme at dt = 600 s (10 min).                   #
#                                                                              #
#  2. ADVANCED INSULATION MODEL (USE_ADVANCED_INSULATION = True)               #
#     The woodchip insulation layer has properties that evolve in time:        #
#       - k_eff : effective thermal conductivity, depends on moisture          #
#                 fraction f = W / W_sat and material age via exponential      #
#                 aging factors.                                               #
#       - alpha_eff : effective solar absorptivity, similarly moisture- and    #
#                 age-dependent.                                               #
#     Moisture W [kg/m^2] inside the insulation evolves through a balance      #
#     of rainfall infiltration, wind-driven evaporation, and drainage.         #
#     Heat fluxes reaching the snow surface include:                           #
#       q_solar   = alpha_eff * I_solar                                        #
#       q_rain    = penetration fraction * rho_w * c_w * Prain * (Tair-T0)     #
#       q_evap    = -Lv * E    (latent heat loss from evaporation)             #
#     If USE_ADVANCED_INSULATION = False, constant alpha and eta are used.     #
#                                                                              #
#  3. GROUND HEAT FLUX - ROBIN BOUNDARY CONDITION                              #
#     Ground flux at the bottom of the snow pile uses a combined resistance:   #
#       R_total = L_soil/k_soil + 1/h_ground                                   #
#       q_ground = (T_soil - T3) / R_total                                     #
#     T_soil is read directly from the CSV (column: Soil_Temp_320cm).          #
#                                                                              #
#  4. MELTING                                                                  #
#     Each layer melts when its temperature exceeds 0 degC.  The temperature   #
#     is clamped to 0 degC and the excess energy is converted to melt water:   #
#       dM = dE / (rho_i * Lf)    [m w.e.]                                     #
#     Surface melt (Layer 1) uses the net surface flux; interior layers use    #
#     the thermal excess above 0 degC.                                         #
#                                                                              #
#  5. REFREEZING                                                               #
#     Liquid water can refreeze in sub-freezing layers. Following the cold-    #
#     content approach of Bartelt & Lehning (2002), the maximum refreezing     #
#     per layer is limited by the cold content of the ice matrix:              #
#       dtheta_w_max = -(dT * (theta_i*rho_i*cs + LWC*rho_w*cw)) /             #
#                       (rho_w * (Lf - dT * (cs - cw)))                        #
#     Temperature is updated to account for latent heat release.               #
#                                                                              #
#  6. PERCOLATION - BUCKET METHOD                                              #
#     Excess liquid water above the irreducible water content theta_e          #
#     percolates downward layer by layer.  Water leaving the bottom layer      #
#     becomes runoff.                                                          #
#                                                                              #
# ---------------------------------------------------------------------------- #
#                                                                              #
#  SMR COMPARISON MODELS                                                       #
#                                                                              #
#  After the main RC simulation, three reference SMR (Snow Melt Rate) models   #
#  are computed over the same hourly forcing and compared:                     #
#                                                                              #
#  A. EMPIRICAL MODEL 1 (Skogsberg 2005)                                       #
#       emp1 [mm/h] = -0.09 + 0.00014*I + 0.0575*T + 0.0012*T*U                #
#                     - 0.18*T*d_ins                                           #
#                                                                              #
#  B. EMPIRICAL MODEL 2 (Skogsberg & Nordell 2001)                             #
#       emp2 [mm/h] = -0.97 - 0.097*(d_ins*100) + 0.164*U + 0.00175*I          #
#                     + 0.102*T + 0.192*w                                      #
#     where w is absolute humidity [g/m^3].                                    #
#                                                                              #
#  C. TRANSIENT 1D HEAT EQUATION (transient1D_smr)                             #
#     An implicit finite-difference scheme (Crank-Nicolson style) solves the   #
#     1D heat equation through the insulation layer at sub-step dt = 10 s.     #
#     Robin BCs are applied at both boundaries:                                #
#       Outer: wind-driven h_out = f(U)                                        #
#       Inner: fixed contact HTC h_i = 99.75 W/(m^2 K) at T_snow = 0 degC      #
#     The inner surface temperature Tsi drives melt via q = Tsi * h_i.         #
#     This solver is JIT-compiled with Numba for performance.                  #
#                                                                              #
#  An optional SNOWPACK .met file can be loaded for additional validation.     #
#                                                                              #
# ---------------------------------------------------------------------------- #
#                                                                              #
#  INPUT DATA                                                                  #
#                                                                              #
#  File    : DATA_2024.csv                                                     #
#  Columns :                                                                   #
#    Time             : timestamp string                                       #
#    Temp_C           : air temperature [degC]                                 #
#    Air_Vel_m/s_10m  : wind speed at 10 m [m/s]                               #
#    Prec_m/h         : precipitation rate [m/h]                               #
#    Glo_Sol_Ir_W/m2  : global solar irradiance [W/m^2]                        #
#    RH_%             : relative humidity [%]                                  #
#    Soil_Temp_320cm  : soil temperature at 3.2 m depth [degC]                 #
#                                                                              #
#  OUTPUT                                                                      #
#    12 PNG figures saved to ./figures/                                        #
#    Console energy balance and SMR comparison statistics                      #
#                                                                              #
# ---------------------------------------------------------------------------- #
#                                                                              #
#  DEPENDENCIES                                                                #
#    numpy, matplotlib, numba, pandas                                          #
#    Standard library: csv, os, time, datetime                                 #
#                                                                              #
#  USAGE                                                                       #
#    Ensure DATA_2024.csv is in the working directory, then run:               #
#       python3 main.py                                                        #
#                                                                              #
################################################################################

import os
import csv
import numpy as np
import pandas as pd
import time as py_time
from numba import njit
from datetime import datetime
import matplotlib.pyplot as plt


# ==============================================================================
#  SIMULATION FEATURE FLAGS
# ==============================================================================

USE_ADVANCED_INSULATION = True   # moisture- and age-dependent insulation model
USE_REFREEZING          = True   # refreeze liquid water in sub-zero layers
USE_PERCOLATION         = True   # bucket-method percolation between layers


# ==============================================================================
#  PHYSICAL CONSTANTS
# ==============================================================================

sigma   = 5.670374419e-8   # Stefan-Boltzmann constant      [W/m^2 K^4]
Lf      = 3.34e5           # Latent heat of fusion           [J/kg]
rho_i   = 917.0            # Ice density                     [kg/m^3]
rho_w   = 1000.0           # Water density                   [kg/m^3]
c_w     = 4180.0           # Water specific heat             [J/(kg K)]
Tfreeze = 273.15           # Freezing temperature            [K]


# ==============================================================================
#  RC MODEL PARAMETERS
#  These are the primary inputs to the 3-layer RC thermal network.
#  Modify these values to tune the model to a specific site.
# ==============================================================================

# --- Snow pile geometry ---
Hs   = 4.5                 # Total snow pile thickness       [m]
Ns   = 3                   # Number of snow layers           [-]
dz_s = Hs / Ns             # Thickness per layer             [m]

# --- Snow thermal properties ---
rho_s  = 400.0             # Snow bulk density               [kg/m^3]
c_s    = 2100.0            # Snow specific heat              [J/(kg K)]
k_snow = 0.45              # Snow thermal conductivity       [W/(m K)]

# --- Insulation layer ---
Hi       = 0.20            # Insulation thickness            [m]
k_i_base = 0.07            # Base thermal conductivity       [W/(m K)]

# --- Ground boundary condition (Robin BC) ---
h_ground = 2.5             # Ground interface HTC            [W/(m^2 K)]
                           # Low  = well-insulated ground    
                           # High = conductive/high water table

# --- Surface heat transfer (air side, convection + longwave radiation) ---
h_conv  = 8.0              # Convective HTC                  [W/(m^2 K)]
epsilon = 0.95             # Longwave emissivity             [-]
T_mean  = 273.15 + 3.0     # Nominal mean temperature for h_rad  [K]

# --- Snowpack liquid water & initial conditions ---
theta_e = 0.04             # Irreducible water content       [-]
T1_init = 273.15 - 2.0     # Initial temperature, layer 1    [K]
T2_init = 273.15 - 4.0     # Initial temperature, layer 2    [K]
T3_init = 273.15 - 6.0     # Initial temperature, layer 3    [K]

# --- Fallback insulation constants (USE_ADVANCED_INSULATION = False only) ---
alpha_const    = 0.80      # Solar absorptivity (constant)   [-]
eta_rain_const = 1.0       # Rain heat fraction reaching snow[-]

# --- Ground insulation layer (used to compute R_3g) ---
Hg_ins = 0.3               # Ground insulation thickness     [m]
kg_ins = 0.04              # Ground insulation conductivity  [W/(m K)]


# ==============================================================================
#  DERIVED RC QUANTITIES  (computed from RC parameters above)
# ==============================================================================

h_rad   = 4.0 * epsilon * sigma * T_mean**3
h_eff   = h_conv + h_rad
R_eff   = 1.0 / h_eff           # Air-side resistance             [m^2 K/W]

R_layer = dz_s / k_snow         # Resistance of one snow layer    [m^2 K/W]
R_12    = R_layer               # Resistance between layers 1-2   [m^2 K/W]
R_23    = R_layer               # Resistance between layers 2-3   [m^2 K/W]
R_g_ins = Hg_ins / kg_ins
R_3g    = R_layer + R_g_ins     # Resistance from layer 3 to ground[m^2 K/W]

Cs_layer = rho_s * c_s * dz_s   # Thermal capacity per layer      [J/(m^2 K)]

# --- Mutable state arrays (reset at start of main()) ---
T             = np.array([T1_init, T2_init, T3_init], dtype=float)
LWC           = np.array([0.0, 0.0, 0.0], dtype=float)
ice_fractions = np.array([0.4, 0.4, 0.4])
heights       = np.array([dz_s, dz_s, dz_s])


# ==============================================================================
#  TRANSIENT 1D SOLVER PARAMETERS
#  These govern the transient1D_smr reference solver only.
#  They are kept separate from the RC parameters above to avoid confusion.
# ==============================================================================

# --- Insulation material (woodchips, wet) ---
rho_dry    = 200.0         # Dry bulk density                [kg/m^3]
moist_cont = 60.0          # Moisture content                [%]
c_dry      = 1.5e3         # Dry specific heat               [J/(kg K)]

# Derived wet properties
rho_wet   = rho_dry + moist_cont / 100.0 * rho_w
c_wet     = (1.0 - moist_cont / 100.0) * c_dry + moist_cont / 100.0 * c_w
D_ins     = k_i_base / (c_wet * rho_wet)   # Thermal diffusivity [m^2/s]

# --- Inner contact HTC for transient1D solver (insulation-snow interface) ---
_h_i_tdma     = 99.75      # Inner HTC for transient1D solver[W/(m^2 K)]
_alpha_solair = 0.80       # Outer surface solar absorptivity[-]


# ==============================================================================
#  ADVANCED INSULATION MODEL STATE (only used if USE_ADVANCED_INSULATION=True)
# ==============================================================================

if USE_ADVANCED_INSULATION:
    InsPar = {
        "Hi":     Hi,        # total insulation thickness      [m]
        "k_dry":  0.05,      # dry conductivity                [W/(m K)]
        "k_sat":  0.12,      # saturated conductivity          [W/(m K)]
        "n_k":    1.5,       # conductivity moisture exponent  [-]

        "W_sat":   100.0,    # saturation moisture content     [%]
        "W_field":  40.0,    # field moisture content          [%]

        "alpha_dry": 0.05,   # solar absorptivity, dry         [-]
        "alpha_wet": 0.08,   # solar absorptivity, wet         [-]
        "n_alpha":   1.0,    # absorptivity moisture exponent  [-]

        "delta_k_age":     0.5,   # conductivity aging factor [-]
        "tau_k_years":     2.0,   # conductivity aging timescale [yr]
        "delta_alpha_age": 0.05,  # absorptivity aging factor [-]
        "tau_alpha_years": 2.0,   # absorptivity aging timescale [yr]

        "zeta0":   0.25,     # initial porosity                [-]
        "gamma_H": 0.5,      # porosity-conductivity exponent  [-]
        "gamma_W": 2.0,      # moisture-conductivity exponent  [-]
        "beta_w":  3.0,      # moisture-absorptivity exponent  [-]

        "K_E":    1e-5,      # evaporation coefficient         [s^-1]
        "K_D":    5e-6,      # drainage coefficient            [s^-1]

        "Lv":      2.5e6,    # latent heat of vaporization     [J/kg]
        "rho_w":   rho_w,    # water density                   [kg/m^3]
        "c_w":     c_w,      # water specific heat             [J/(kg K)]
        "Tfreeze": Tfreeze,  # freezing temperature            [K]

        "rho_air": 1.2,      # air density                     [kg/m^3]
        "C_E":     1.3e-3,   # wind-evaporation coefficient    [-]
        "P0":      101325.0  # reference pressure              [Pa]
    }
    InsState = {
        "W":        60.0,    # initial moisture content        [%]
        "age_days":  0.0     # initial age                     [days]
    }
else:
    InsPar   = None
    InsState = {"W": 0.0, "age_days": 0.0}


# SNOWPACK offset - removes woodchip mass from SNOWPACK SWE output:
#   Layer_Thick * Vol_Frac_W * rho_water + Layer_Thick * Vol_Frac_S * rho_ins
WOODCHIP_OFFSET = 72.53    # [kg/m^2]


# ==============================================================================
#  HELPER: WATER VAPOUR SATURATION PRESSURE
# ==============================================================================

def Psat_WV(T_K):
    """
    Saturation vapour pressure using the IAPWS-95 correlation.

    Parameters
    ----------
    T_K : float or array
        Absolute temperature [K].

    Returns
    -------
    Psat : float or array
        Saturation vapour pressure [hPa].
    """
    Tc = 647.096   # Critical temperature [K]
    Pc = 220640    # Critical pressure [hPa]
    C1, C2, C3 = -7.85951783,  1.84408259, -11.7866497
    C4, C5, C6 = 22.6807411,  -15.9618719,   1.80122502
    teta = 1.0 - T_K / Tc
    x = Tc / T_K * (C1*teta + C2*teta**1.5 + C3*teta**3
                    + C4*teta**3.5 + C5*teta**4 + C6*teta**7.5)
    return np.exp(x) * Pc


# ==============================================================================
#  CSV DATA LOADING
# ==============================================================================

def read_csv_data(filename):
    """
    Read hourly meteorological data from a CSV file.

    Expected columns: Time, Temp_C, Air_Vel_m/s_10m, Prec_m/h,
    Glo_Sol_Ir_W/m2, RH_%, Soil_Temp_320cm.

    Parameters
    ----------
    filename : str
        Path to the CSV file.

    Returns
    -------
    data : dict
        Keys: 'time', 'temp', 'wind', 'precip', 'solar', 'rh',
        'soil_temp'.  All numeric fields are Python lists of float.
    """
    data = {
        'time': [], 'temp': [], 'wind': [], 'precip': [],
        'solar': [], 'rh': [], 'soil_temp': []
    }

    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                data['time'].append(row['Time'])
                data['temp'].append(float(row['Temp_C']))
                data['wind'].append(float(row['Air_Vel_m/s_10m']))
                data['precip'].append(float(row['Prec_m/h']))
                data['solar'].append(float(row['Glo_Sol_Ir_W/m2']))
                data['rh'].append(float(row['RH_%']))
                data['soil_temp'].append(float(row['Soil_Temp_320cm']))
            except (ValueError, KeyError) as e:
                print(f"Warning: Skipping row due to error: {e}")
                continue

    return data


def interpolate_data(data_vec, t_query, dt_data=3600.0):
    """
    Linearly interpolate a uniformly-spaced data array at time t_query.

    Parameters
    ----------
    data_vec : list or array
        Data values at uniform intervals of dt_data [s].
    t_query  : float
        Query time [s].
    dt_data  : float
        Interval between data points [s] (default 3600).

    Returns
    -------
    float
        Interpolated value.
    """
    idx      = t_query / dt_data
    idx_low  = int(np.floor(idx))
    idx_high = int(np.ceil(idx))

    if idx_low < 0:
        return data_vec[0]
    if idx_high >= len(data_vec):
        return data_vec[-1]
    if idx_low == idx_high:
        return data_vec[idx_low]

    frac = idx - idx_low
    return data_vec[idx_low] * (1 - frac) + data_vec[idx_high] * frac


# ==============================================================================
#  TRIDIAGONAL SOLVER (TDMA / Thomas Algorithm)
# ==============================================================================

@njit
def solve_tdma(a, b, c, d, n):
    """
    Solve a tridiagonal system A*x = d using the Thomas algorithm.

    Parameters
    ----------
    a, b, c : arrays of length n
        Sub-, main-, and super-diagonal coefficients.
    d       : array of length n
        Right-hand side.
    n       : int
        System size.

    Returns
    -------
    x : array of length n
        Solution vector.
    """
    c_prime = np.zeros(n)
    d_prime = np.zeros(n)
    x       = np.zeros(n)

    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]

    for i in range(1, n):
        denom      = b[i] - a[i] * c_prime[i - 1]
        c_prime[i] = c[i] / denom
        d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) / denom

    x[-1] = d_prime[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]

    return x


# ==============================================================================
#  TRANSIENT 1D INSULATION SOLVER (SMR REFERENCE MODEL)
# ==============================================================================

@njit
def transient1D_smr(t_o, h_o, d_ins, lam_i, D, dx=0.005, dt=10.0,
                    h_i=99.75):
    """
    Solve transient 1D conduction through the insulation layer.

    Uses an implicit finite-difference scheme with Robin boundary
    conditions at both surfaces, JIT-compiled via Numba.

    Outer BC (environment side):
        q = h_o * (T_env - T_outer)   where h_o is wind-dependent.
    Inner BC (snow contact, fixed 0 degC):
        q = h_i * (T_inner - 0)

    Parameters
    ----------
    t_o   : 1-D float64 array, length n_hours
        Sol-air temperature [degC] at each hour.
    h_o   : 1-D float64 array, length n_hours
        External HTC [W/(m^2 K)] at each hour.
    d_ins : float
        Insulation thickness [m].
    lam_i : float
        Insulation thermal conductivity [W/(m K)].
    D     : float
        Insulation thermal diffusivity [m^2/s].
    dx    : float
        Spatial step [m] (default 0.005 m = 5 mm).
    dt    : float
        Time sub-step [s] (default 10 s).
    h_i   : float
        Inner surface HTC [W/(m^2 K)] (default 99.75).

    Returns
    -------
    T_nh : 2-D float64 array, shape (nodes, n_hours)
        Temperature profile at end of each hour [degC].
        T_nh[0,:]  = outer surface temperature.
        T_nh[-1,:] = inner surface temperature (Tsi).
    """
    t_i     = 0.0
    n_el    = d_ins / dx
    nodes   = int(n_el + 1)
    n_hours = len(t_o)
    nh      = int(3600 / dt)

    T_n  = np.zeros(nodes)
    T_nh = np.zeros((nodes, n_hours))

    a = np.zeros(nodes)
    b = np.zeros(nodes)
    c = np.zeros(nodes)
    d = np.zeros(nodes)

    for h in range(n_hours):
        dFo    = D   * dt / dx**2
        dBio_i = h_i * dx / lam_i
        dBio_o = h_o[h] * dx / lam_i

        for k in range(nh):
            # Outer node - Robin BC, environment side
            b[0] = 1.0 + 2.0*dFo + 2.0*dFo*dBio_o
            c[0] = -2.0 * dFo
            d[0] = T_n[0] + 2.0*dFo*dBio_o*t_o[h]
            a[0] = 0.0

            # Interior nodes - standard implicit scheme
            for j in range(1, nodes - 1):
                a[j] = -dFo
                b[j] = 1.0 + 2.0*dFo
                c[j] = -dFo
                d[j] = T_n[j]

            # Inner node - Robin BC, snow contact at 0 degC
            a[-1] = -2.0 * dFo
            b[-1] = 1.0 + 2.0*dFo + 2.0*dFo*dBio_i
            c[-1] = 0.0
            d[-1] = T_n[-1] + 2.0*dFo*dBio_i*t_i

            T_n = solve_tdma(a, b, c, d, nodes)

        T_nh[:, h] = T_n

    return T_nh


# ==============================================================================
#  SNOW PHYSICS SUBROUTINES
# ==============================================================================

def refreezing_layer(T_layer, LWC_layer, ice_frac):
    """
    Refreeze liquid water in a single snow layer.

    Follows the cold-content approach of Bartelt & Lehning (2002)
    as implemented in COSIPY.  Refreezing is limited by the available
    liquid water and the cold content of the ice-water mixture.

    Parameters
    ----------
    T_layer   : float  Layer temperature [K].
    LWC_layer : float  Volumetric liquid water content [-].
    ice_frac  : float  Volumetric ice fraction [-].

    Returns
    -------
    new_T        : float  Updated layer temperature [K].
    new_LWC      : float  Updated liquid water content [-].
    new_ice_frac : float  Updated ice fraction [-].
    refrozen_mass: float  Mass of refrozen water [kg/m^2].
    """
    if (T_layer >= Tfreeze) or (LWC_layer <= 0.0):
        return T_layer, LWC_layer, ice_frac, 0.0

    dT_max = T_layer - Tfreeze
    dtheta_w_max = (
        -(dT_max * (ice_frac * rho_i * c_s + LWC_layer * rho_w * c_w))
        / (rho_w * (Lf - dT_max * (c_s - c_w)))
    )
    dtheta_w  = min(LWC_layer, dtheta_w_max)
    dtheta_i  = (rho_w / rho_i) * dtheta_w
    dT        = (dtheta_w * rho_w * Lf) / (
                    ice_frac * rho_i * c_s + LWC_layer * rho_w * c_w)

    new_T         = T_layer + dT
    new_LWC       = LWC_layer - dtheta_w
    new_ice_frac  = ice_frac + dtheta_i
    refrozen_mass = dtheta_w * dz_s * rho_w

    return new_T, new_LWC, new_ice_frac, refrozen_mass


def percolate_water(LWC_array, heights, theta_e):
    """
    Percolate liquid water downward through layers (bucket method).

    Excess water above theta_e drains from each layer to the layer
    below; excess leaving the bottom layer becomes runoff.

    Parameters
    ----------
    LWC_array : array  Volumetric LWC per layer [-].
    heights   : array  Layer thicknesses [m].
    theta_e   : float  Irreducible (field capacity) water content [-].

    Returns
    -------
    new_LWC : array  Updated LWC array [-].
    runoff  : float  Runoff mass from bottom layer [kg/m^2].
    """
    n_layers = len(LWC_array)
    new_LWC  = LWC_array.copy()

    for i in range(n_layers - 1):
        if new_LWC[i] > theta_e:
            excess        = new_LWC[i] - theta_e
            new_LWC[i]   = theta_e
            excess_mass   = excess * heights[i]
            new_LWC[i+1] += excess_mass / heights[i+1]

    runoff = 0.0
    if new_LWC[n_layers-1] > theta_e:
        excess                  = new_LWC[n_layers-1] - theta_e
        new_LWC[n_layers-1]    = theta_e
        runoff                  = excess * heights[n_layers-1] * rho_w

    return new_LWC, runoff


def ground_flux_robin_bc(T3, T_soil_K, h_ground,
                         k_soil=1.5, L_soil=1.0):
    """
    Compute ground-to-snow heat flux using a Robin boundary condition.

    Combined resistance includes conduction through soil and a
    contact interface resistance at the snow-ground boundary.

    Parameters
    ----------
    T3       : float  Bottom snow layer temperature [K].
    T_soil_K : float  Soil temperature from CSV data [K].
    h_ground : float  Ground interface HTC [W/(m^2 K)].
    k_soil   : float  Soil conductivity [W/(m K)] (default 1.5).
    L_soil   : float  Effective soil thickness [m] (default 1.0).

    Returns
    -------
    q_ground : float  Heat flux from ground to snow [W/m^2].
               Positive = warming the snow.
    """
    R_cond      = L_soil / k_soil
    R_interface = 1.0 / h_ground
    R_total     = R_cond + R_interface
    return (T_soil_K - T3) / R_total


# ==============================================================================
#  INSULATION STEP (ADVANCED MODEL)
# ==============================================================================

def insulation_step(state_in, forc, p, dt):
    """
    Advance the advanced insulation state by one time step.

    Updates moisture W, effective conductivity k_eff, and effective
    solar absorptivity alpha_eff.  Returns the thermal resistance
    R_ins = Hi / k_eff (single-layer approximation) along with the
    heat fluxes that reach the snow surface.

    Parameters
    ----------
    state_in : dict   Keys: 'W' (moisture [%]), 'age_days' (age [d]).
    forc     : dict   Keys: 'Isolar', 'Prain', 'T_rain', 'RH', 'Ta',
                      'U10', 'h_out'.
    p        : dict   InsPar dictionary of material constants.
    dt       : float  Time step [s].

    Returns
    -------
    R_ins      : float  Insulation thermal resistance [m^2 K/W].
    q_solar    : float  Solar flux reaching snow [W/m^2].
    q_rain_snow: float  Rain heat flux reaching snow [W/m^2].
    q_evap     : float  Evaporative flux (negative = cooling) [W/m^2].
    state_out  : dict   Updated insulation state.
    """
    W        = state_in["W"]
    age_days = state_in["age_days"]

    f       = np.clip(W / p["W_sat"], 0.0, 1.0)
    age_yr  = age_days / 365.0

    k_moist      = p["k_dry"] + (p["k_sat"] - p["k_dry"]) * (f**p["n_k"])
    k_age_factor = 1.0 + p["delta_k_age"] * (
                       1.0 - np.exp(-age_yr / p["tau_k_years"]))
    k_eff        = k_moist * k_age_factor

    alpha_moist = (p["alpha_dry"]
                   + (p["alpha_wet"] - p["alpha_dry"]) * (f**p["n_alpha"]))
    alpha_age   = alpha_moist + p["delta_alpha_age"] * (
                      1.0 - np.exp(-age_yr / p["tau_alpha_years"]))
    alpha_eff   = np.clip(alpha_age, 0.0, 1.0)

    q_solar    = alpha_eff * forc["Isolar"]
    eta_rain   = max(0.0, 1.0 - f)
    P_in_mass  = eta_rain * p["rho_w"] * forc["Prain"]

    zeta_rain   = (p["zeta0"]
                   * np.exp(-p["gamma_H"] * p["Hi"])
                   * np.exp(-p["gamma_W"] * f))
    q_rain_snow = (zeta_rain * p["rho_w"] * p["c_w"]
                   * forc["Prain"] * (forc["T_rain"] - p["Tfreeze"]))

    Tc_s    = p["Tfreeze"] - 273.15
    Tc_a    = forc["Ta"]  - 273.15
    e_sat_s = 611.0 * np.exp(17.27 * Tc_s / (Tc_s + 237.3))
    e_sat_a = 611.0 * np.exp(17.27 * Tc_a / (Tc_a + 237.3))
    VPD     = max(0.0, e_sat_s - forc["RH"] * e_sat_a)

    E0        = p["rho_air"] * p["C_E"] * forc["U10"] * VPD / p["P0"]
    f_breath  = np.exp(-p["beta_w"] * f)
    E         = E0 * f_breath
    q_evap    = -p["Lv"] * E

    D         = p["K_D"] * max(0.0, W - p["W_field"])
    W_new     = np.clip(W + dt * (P_in_mass - E - D), 0.0, p["W_sat"])

    state_out              = dict(state_in)
    state_out["W"]         = W_new
    state_out["age_days"]  = age_days + dt / 86400.0
    state_out["k_eff"]     = k_eff
    state_out["alpha_eff"] = alpha_eff
    state_out["f_sat"]     = f

    R_ins = Hi / k_eff

    return R_ins, q_solar, q_rain_snow, q_evap, state_out


# ==============================================================================
#  NUMBA-JIT HELPER FUNCTIONS FOR RK4 INTEGRATOR
# ==============================================================================

@njit
def compute_h_out(wind_speed):
    """
    External HTC from wind speed, following Lunde (1980) correlation.

    Parameters
    ----------
    wind_speed : float  Wind speed [m/s].

    Returns
    -------
    h_out : float  External HTC [W/(m^2 K)].
    """
    if wind_speed <= 5.0:
        return 6.0 + 4.0 * wind_speed
    else:
        return 7.41 * (wind_speed**0.78)


@njit
def _interp_numba(data_vec, t_query, dt_data):
    """
    Numba-compatible linear interpolation of a 1D array.

    Parameters
    ----------
    data_vec : 1-D float64 array   Values at uniform dt_data intervals.
    t_query  : float               Query time [s].
    dt_data  : float               Data interval [s].

    Returns
    -------
    float  Interpolated value.
    """
    idx      = t_query / dt_data
    idx_low  = int(np.floor(idx))
    idx_high = idx_low + 1
    n        = len(data_vec)
    if idx_low < 0:
        return data_vec[0]
    if idx_high >= n:
        return data_vec[n - 1]
    if idx_low == idx_high:
        return data_vec[idx_low]
    frac = idx - idx_low
    return data_vec[idx_low] * (1.0 - frac) + data_vec[idx_high] * frac


@njit
def _dTdt_numba(t, Tv,
                R_a2s, q_solar, q_rain, q_evap,
                temp_data, soil_temp_data, dt_data,
                R_12, R_23, Cs_layer,
                h_ground, Tfreeze):
    """
    Numba-compatible ODE right-hand side for the 3-layer RC snow model.

    Parameters
    ----------
    t              : float   Current time [s].
    Tv             : array   [T1, T2, T3] layer temperatures [K].
    R_a2s          : float   Air-to-snow resistance [m^2 K/W].
    q_solar        : float   Solar flux at snow surface [W/m^2].
    q_rain         : float   Rain heat flux [W/m^2].
    q_evap         : float   Evaporative flux [W/m^2].
    temp_data      : array   Hourly air temperatures [degC].
    soil_temp_data : array   Hourly soil temperatures [degC].
    dt_data        : float   Data time step [s].
    R_12, R_23     : float   Inter-layer resistances [m^2 K/W].
    Cs_layer       : float   Layer heat capacity [J/(m^2 K)].
    h_ground       : float   Ground interface HTC [W/(m^2 K)].
    Tfreeze        : float   0 degC in Kelvin.

    Returns
    -------
    out : array   [dT1/dt, dT2/dt, dT3/dt] [K/s].
    """
    T1, T2, T3 = Tv[0], Tv[1], Tv[2]

    Ta       = _interp_numba(temp_data, t, dt_data) + 273.15
    q_a      = (Ta - T1) / R_a2s
    q_surf   = q_a + q_solar + q_rain + q_evap
    q_12     = (T2 - T1) / R_12
    dT1      = (q_surf + q_12) / Cs_layer

    q_21     = (T1 - T2) / R_12
    q_23     = (T3 - T2) / R_23
    dT2      = (q_21 + q_23) / Cs_layer

    T_soil_K    = _interp_numba(soil_temp_data, t, dt_data) + 273.15
    R_cond      = 1.0 / 1.5
    R_interface = 1.0 / h_ground
    R_total     = R_cond + R_interface
    q_3g        = (T_soil_K - T3) / R_total
    q_32        = (T2 - T3) / R_23
    dT3         = (q_32 + q_3g) / Cs_layer

    out    = np.empty(3)
    out[0] = dT1
    out[1] = dT2
    out[2] = dT3
    return out


def rk4_step(t, T, dt,
             R_a2s, q_solar, q_rain, q_evap,
             temp_data, soil_temp_data, dt_data,
             R_12, R_23, Cs_layer,
             h_ground, Tfreeze):
    """
    Advance snow temperatures by one RK4 step.

    Parameters
    ----------
    t, T, dt       : float, array, float
                     Current time [s], temperatures [K], step size [s].
    R_a2s          : float   Combined air + insulation resistance [m^2 K/W].
    q_solar        : float   Solar flux [W/m^2].
    q_rain         : float   Rain heat flux [W/m^2].
    q_evap         : float   Evaporative flux [W/m^2].
    temp_data      : array   Hourly air temperatures [degC].
    soil_temp_data : array   Hourly soil temperatures [degC].
    dt_data        : float   Data time step [s].
    R_12, R_23     : float   Inter-layer resistances [m^2 K/W].
    Cs_layer       : float   Layer capacity [J/(m^2 K)].
    h_ground       : float   Ground HTC [W/(m^2 K)].
    Tfreeze        : float   Freezing temperature [K].

    Returns
    -------
    T_new : array   Updated temperatures [K].
    """
    args = (R_a2s, q_solar, q_rain, q_evap,
            temp_data, soil_temp_data, dt_data,
            R_12, R_23, Cs_layer, h_ground, Tfreeze)

    k1 = _dTdt_numba(t,          T,              *args)
    k2 = _dTdt_numba(t + dt/2.0, T + dt*k1/2.0, *args)
    k3 = _dTdt_numba(t + dt/2.0, T + dt*k2/2.0, *args)
    k4 = _dTdt_numba(t + dt,     T + dt*k3,      *args)

    return T + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)


# ==============================================================================
#  SNOWPACK .MET FILE READER
# ==============================================================================

def read_snowpack_met(met_path):
    """
    Parse a SNOWPACK .met file into a DataFrame.

    Parameters
    ----------
    met_path : str  Path to the .met file.

    Returns
    -------
    df        : pd.DataFrame  Data indexed by datetime.
    col_units : dict          Column -> unit string mapping.
    """
    with open(met_path) as f:
        lines = f.readlines()

    header_block   = [l for l in lines
                      if l.startswith(',,') or l.startswith('ID,Date,')]
    col_names_line = header_block[1]
    col_units_line = header_block[2]

    col_names     = [c.strip() for c in col_names_line.split(',')]
    col_units_raw = [c.strip() for c in col_units_line.split(',')]
    col_units     = {col_names[i]: col_units_raw[i]
                     for i in range(min(len(col_names), len(col_units_raw)))}

    data_rows = [l.strip().split(',') for l in lines if l[:4].isdigit()]

    seen = {}
    deduped = []
    for name in col_names[:len(data_rows[0])]:
        if name in seen:
            seen[name] += 1
            deduped.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 0
            deduped.append(name)

    df = pd.DataFrame(data_rows, columns=deduped)
    df['Date'] = pd.to_datetime(df['Date'].str.strip(),
                                format='%d.%m.%Y %H:%M:%S')
    df = df.set_index('Date').sort_index()

    for c in df.columns:
        if c not in ('ID',):
            df[c] = pd.to_numeric(df[c], errors='coerce').replace(
                        -999.0, np.nan)

    return df, col_units


def load_snowpack_smr(met_path):
    """
    Extract hourly cumulative runoff, SWE, and snow depth from a
    SNOWPACK .met file.

    Parameters
    ----------
    met_path : str  Path to the SNOWPACK .met file.

    Returns
    -------
    sp_cs    : np.ndarray    Cumulative runoff [kg/m^2], full length.
    sp_swe   : pd.Series     SWE (snow only) [kg/m^2].
    sp_depth : pd.Series     Snow depth [m] (or None if unavailable).
    sp_index : DatetimeIndex Timestamps of the .met file.
    """
    sp, _ = read_snowpack_met(met_path)

    sp_runoff_col = 'Snowpack runoff (virtual lysimeter -- snow only)'
    if sp_runoff_col in sp.columns:
        sp_cs = np.cumsum(sp[sp_runoff_col].fillna(0).values)
    else:
        print("  WARNING: SNOWPACK runoff column not found in .met file.")
        sp_cs = np.zeros(len(sp))

    sp_swe = None
    if 'SWE (of snowpack)' in sp.columns:
        sp_swe = sp['SWE (of snowpack)'] - WOODCHIP_OFFSET

    sp_depth = None
    if 'Modelled snow depth (vertical)' in sp.columns:
        sp_depth = sp['Modelled snow depth (vertical)'] / 100.0  # cm->m

    return sp_cs, sp_swe, sp_depth, sp.index


# ==============================================================================
#  MAIN SIMULATION
# ==============================================================================

def main():
    global T, LWC, ice_fractions, InsState

    # Reset mutable state for a clean run
    T             = np.array([T1_init, T2_init, T3_init], dtype=float)
    LWC           = np.array([0.0, 0.0, 0.0], dtype=float)
    ice_fractions = np.array([0.4, 0.4, 0.4])

    print("=" * 60)
    print("Snow Storage RC Model")
    print("=" * 60)

    # ------------------------------------------------------------------
    #  Load meteorological data
    # ------------------------------------------------------------------
    print("\nLoading meteorological data from DATA_2024.csv...")
    try:
        met_data = read_csv_data('DATA_2024.csv')
        print(f"  Loaded {len(met_data['temp'])} hourly data points")
        print(f"  Period: {met_data['time'][0]} to {met_data['time'][-1]}")
        t_range = (min(met_data['soil_temp']), max(met_data['soil_temp']))
        print(f"  Soil temp range: {t_range[0]:.1f} to {t_range[1]:.1f} C")
    except FileNotFoundError:
        print("ERROR: DATA_2024.csv not found!")
        print("Ensure DATA_2024.csv is in the same directory as this script.")
        return

    # ------------------------------------------------------------------
    #  Optional SNOWPACK .met validation file
    # ------------------------------------------------------------------
    SNOWPACK_MET       = os.path.join('output', 'snow_storage_snow_storage.met')
    snowpack_available = os.path.isfile(SNOWPACK_MET)
    if snowpack_available:
        print(f"\nSNOWPACK .met found: {SNOWPACK_MET}")
    else:
        print(f"\nSNOWPACK .met not found at '{SNOWPACK_MET}' - skipped.")

    # ------------------------------------------------------------------
    #  Time integration settings
    # ------------------------------------------------------------------
    t0      = 0.0
    dt      = 600.0    # Simulation time step: 10 min          [s]
    dt_data = 3600.0   # Input data time step: 1 hour          [s]

    n_hours = len(met_data['temp'])
    tf      = n_hours * dt_data
    t_vec   = np.arange(t0, tf, dt)
    Nt      = len(t_vec)

    print(f"\nSimulation settings:")
    print(f"  Duration:    {n_hours} h  ({n_hours/24:.1f} days)")
    print(f"  Time step:   {dt} s  ({dt/60:.1f} min)")
    print(f"  Total steps: {Nt}")
    print(f"  Refreezing:  {USE_REFREEZING}")
    print(f"  Percolation: {USE_PERCOLATION}")
    print(f"  Adv. insul.: {USE_ADVANCED_INSULATION}")

    # ------------------------------------------------------------------
    #  Allocate history arrays
    # ------------------------------------------------------------------
    T_hist   = np.zeros((Nt, 3))
    LWC_hist = np.zeros((Nt, 3))
    T_hist[0, :]   = T
    LWC_hist[0, :] = LWC

    qnet_surf_hist = np.zeros(Nt)
    qa_hist        = np.zeros(Nt)
    qsolar_hist    = np.zeros(Nt)
    qrain_hist     = np.zeros(Nt)
    qevap_hist     = np.zeros(Nt)
    qground_hist   = np.zeros(Nt)

    Ta_hist        = np.zeros(Nt)
    Isolar_hist    = np.zeros(Nt)
    Prain_hist     = np.zeros(Nt)
    Tsoil_hist     = np.zeros(Nt)

    melt_rate_hist = np.zeros(Nt)
    refrozen_hist  = np.zeros(Nt)
    runoff_hist    = np.zeros(Nt)
    E_melt         = 0.0

    W_hist     = np.zeros(Nt)
    k_eff_hist = np.zeros(Nt)
    alpha_hist = np.zeros(Nt)
    Rins_hist  = np.zeros(Nt)
    fsat_hist  = np.zeros(Nt)

    # Pre-convert met lists to numpy arrays (required by Numba JIT)
    temp_arr      = np.array(met_data['temp'],      dtype=np.float64)
    soil_temp_arr = np.array(met_data['soil_temp'], dtype=np.float64)

    progress_points = {int(Nt * p): f"{int(p*100)}%"
                       for p in [0.25, 0.5, 0.75]}

    # ------------------------------------------------------------------
    #  Main time loop
    # ------------------------------------------------------------------
    print("\nRunning simulation...")
    t1 = py_time.time()

    for k in range(Nt - 1):
        t     = t_vec[k]
        t_mid = t + dt / 2.0

        if k in progress_points:
            print(f"  Progress: {progress_points[k]}")

        # Interpolate hourly forcing to sub-hourly time step
        Ta_C     = interpolate_data(met_data['temp'],   t_mid, dt_data)
        Isolar   = interpolate_data(met_data['solar'],  t_mid, dt_data)
        Prain_mh = interpolate_data(met_data['precip'], t_mid, dt_data)
        wind     = interpolate_data(met_data['wind'],   t_mid, dt_data)
        RH_pct   = interpolate_data(met_data['rh'],     t_mid, dt_data)
        Tsoil_C  = interpolate_data(met_data['soil_temp'], t_mid, dt_data)

        Ta_K    = Ta_C + 273.15
        Prain   = Prain_mh / 3600.0   # [m/h] -> [m/s]
        RH_frac = RH_pct / 100.0
        Tsoil_K = Tsoil_C + 273.15
        h_out   = compute_h_out(wind)

        forc = {
            "Isolar": Isolar,
            "Prain":  Prain,
            "T_rain": Ta_K,
            "RH":     RH_frac,
            "Ta":     Ta_K,
            "U10":    wind,
            "h_out":  h_out
        }

        # Update insulation
        if USE_ADVANCED_INSULATION:
            R_ins, q_solar_ins, q_rain_snow, q_evap, InsState = \
                insulation_step(InsState, forc, InsPar, dt)
            W_hist[k]     = InsState["W"]
            k_eff_hist[k] = InsState["k_eff"]
            alpha_hist[k] = InsState["alpha_eff"]
            Rins_hist[k]  = R_ins
            fsat_hist[k]  = InsState["f_sat"]
        else:
            R_ins       = Hi / k_i_base
            q_solar_ins = alpha_const * forc["Isolar"]
            q_rain_snow = (eta_rain_const * rho_w * c_w
                           * forc["Prain"] * (forc["T_rain"] - Tfreeze))
            q_evap      = 0.0

        R_a2s = R_eff + R_ins

        # RK4 integration (Numba-JIT)
        T_new = rk4_step(
            t, T, dt,
            R_a2s, q_solar_ins, q_rain_snow, q_evap,
            temp_arr, soil_temp_arr, dt_data,
            R_12, R_23, Cs_layer,
            h_ground, Tfreeze
        )

        # Flux diagnostics at mid-step
        q_a_mid      = (Ta_K - T[0]) / R_a2s
        q_surf_mid   = q_a_mid + q_solar_ins + q_rain_snow + q_evap
        q_ground_mid = ground_flux_robin_bc(T[2], Tsoil_K, h_ground)

        # Refreezing
        total_refrozen = 0.0
        if USE_REFREEZING:
            for i in range(3):
                T_new[i], LWC[i], ice_fractions[i], refrozen = \
                    refreezing_layer(T_new[i], LWC[i], ice_fractions[i])
                total_refrozen += refrozen
        refrozen_hist[k] = total_refrozen

        # Melting - all three layers
        total_melt_water = 0.0

        if T_new[0] > Tfreeze:
            T_new[0] = Tfreeze
            if q_surf_mid > 0.0:
                dE_melt_1 = q_surf_mid * dt
                dM_melt_1 = dE_melt_1 / (rho_i * Lf)
            else:
                dE_melt_1 = dM_melt_1 = 0.0
            E_melt           += dE_melt_1
            total_melt_water += dM_melt_1
            LWC[0]           += dM_melt_1 / dz_s

        if T_new[1] > Tfreeze:
            dE_melt_2 = Cs_layer * (T_new[1] - Tfreeze)
            dM_melt_2 = dE_melt_2 / (rho_i * Lf)
            T_new[1]  = Tfreeze
            E_melt           += dE_melt_2
            total_melt_water += dM_melt_2
            LWC[1]           += dM_melt_2 / dz_s

        if T_new[2] > Tfreeze:
            dE_melt_3 = Cs_layer * (T_new[2] - Tfreeze)
            dM_melt_3 = dE_melt_3 / (rho_i * Lf)
            T_new[2]  = Tfreeze
            E_melt           += dE_melt_3
            total_melt_water += dM_melt_3
            LWC[2]           += dM_melt_3 / dz_s

        melt_rate_hist[k] = total_melt_water / dt   # [m w.e./s]

        # Percolation
        runoff = 0.0
        if USE_PERCOLATION:
            LWC, runoff = percolate_water(LWC, heights, theta_e)
        runoff_hist[k] = runoff

        # Advance state
        T = T_new
        T_hist[k+1, :]   = T
        LWC_hist[k+1, :] = LWC

        qnet_surf_hist[k] = q_surf_mid
        qa_hist[k]        = q_a_mid
        qsolar_hist[k]    = q_solar_ins
        qrain_hist[k]     = q_rain_snow
        qevap_hist[k]     = q_evap
        qground_hist[k]   = q_ground_mid

        Ta_hist[k]     = Ta_K
        Isolar_hist[k] = Isolar
        Prain_hist[k]  = Prain
        Tsoil_hist[k]  = Tsoil_K

    # Fill final time step
    Ta_hist[-1]      = interpolate_data(
        met_data['temp'],   t_vec[-1], dt_data) + 273.15
    Isolar_hist[-1]  = interpolate_data(
        met_data['solar'],  t_vec[-1], dt_data)
    Prain_hist[-1]   = interpolate_data(
        met_data['precip'], t_vec[-1], dt_data) / 3600.0
    Tsoil_hist[-1]   = interpolate_data(
        met_data['soil_temp'], t_vec[-1], dt_data) + 273.15
    qground_hist[-1] = ground_flux_robin_bc(
        T[-1], Tsoil_hist[-1], h_ground)

    if USE_ADVANCED_INSULATION:
        W_hist[-1]     = InsState["W"]
        k_eff_hist[-1] = InsState["k_eff"]
        alpha_hist[-1] = InsState["alpha_eff"]
        Rins_hist[-1]  = R_ins
        fsat_hist[-1]  = InsState["f_sat"]

    elapsed_time = py_time.time() - t1
    print("  Progress: 100%")
    print(f"\nSimulation complete!  Elapsed: {elapsed_time:.4e} s")

    # ------------------------------------------------------------------
    #  Energy balance diagnostics
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Energy Balance Diagnostics")
    print("=" * 60)

    E_a     = np.trapezoid(qa_hist,      t_vec)
    E_solar = np.trapezoid(qsolar_hist,  t_vec)
    E_rain  = np.trapezoid(qrain_hist,   t_vec)
    E_evap  = np.trapezoid(qevap_hist,   t_vec)
    E_g     = np.trapezoid(qground_hist, t_vec)
    E_snow_change= Cs_layer * np.sum(T_hist[-1, :] - T_hist[0, :])
    E_refrozen   = np.sum(refrozen_hist) * Lf
    M_melt       = E_melt / (rho_i * Lf)
    M_runoff     = np.sum(runoff_hist)

    print(f"\nEnergy gains [MJ/m^2]:")
    print(f"  Solar radiation: {E_solar/1e6:>10.3f}")
    print(f"  Air convection:  {E_a/1e6:>10.3f}")
    print(f"  Refreezing:      {E_refrozen/1e6:>10.3f}")
    print(f"  Ground heat:     {E_g/1e6:>10.3f}")
    print(f"  Rain heat:       {E_rain/1e6:>10.3f}")
    print(f"  " + "-" * 28)
    E_total_in = E_solar + E_a + E_refrozen + E_g +E_rain
    print(f"  Total in:        {E_total_in/1e6:>10.3f}")

    print(f"\nEnergy losses [MJ/m^2]:")
    print(f"  Melting:         {E_melt/1e6:>10.3f}")
    print(f"  Snow warming:    {E_snow_change/1e6:>10.3f}")
    print(f"  Evaporation:     {-E_evap/1e6:>10.3f}")
    print(f"  " + "-" * 28)
    E_total_out = E_melt + E_snow_change - E_evap
    print(f"  Total out:       {E_total_out/1e6:>10.3f}")

    print(f"\nMass balance:")
    print(f"  Total melt:      {M_melt:>10.3f} m w.e.")
    print(f"  Total runoff:    {M_runoff:>10.3f} kg/m^2")
    print(f"  Melt rate (avg): {M_melt/(n_hours/24)*1000:>10.3f} mm/day")

    # ------------------------------------------------------------------
    #  Empirical & 1D SMR comparison
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SMR Comparison (Empirical + Transient 1D + RC Model)")
    print("=" * 60)

    n_h        = len(met_data['temp'])
    air_temp_h = np.array(met_data['temp'],   dtype=float)
    air_vel_h  = np.array(met_data['wind'],   dtype=float)
    glob_sol_h = np.array(met_data['solar'],  dtype=float)
    rh_h       = np.array(met_data['rh'],     dtype=float)

    _d_ins = Hi

    # Empirical model 1 - Skogsberg (2005), p. 39
    emp1_h = (-0.09
              + 0.00014 * glob_sol_h
              + 0.0575  * air_temp_h
              + 0.0012  * air_temp_h * air_vel_h
              - 0.18    * air_temp_h * _d_ins)
    emp1_wc_h = np.where((emp1_h < 0) | (air_temp_h < 0), 0.0, emp1_h)

    # Empirical model 2 - Skogsberg (2005), p. 39
    Psat_h = Psat_WV(air_temp_h + 273.15) / 10.0
    Pw_h   = Psat_h * rh_h / 100.0
    w_h    = 2.16679 * Pw_h * 1000.0 / (273.15 + air_temp_h)
    emp2_h = (-0.97
              - 0.097  * (_d_ins * 100)
              + 0.164  * air_vel_h
              + 0.00175 * glob_sol_h
              + 0.102  * air_temp_h
              + 0.192  * w_h)
    emp2_wc_h = np.where((emp2_h < 0) | (air_temp_h < 0), 0.0, emp2_h)

    # Transient 1D solver
    h_o_h = np.where(air_vel_h <= 5.0,
                     6.0 + 4.0 * air_vel_h,
                     7.41 * air_vel_h**0.78)
    t_sol_air_h = _alpha_solair * glob_sol_h / h_o_h + air_temp_h

    _c_wet_tdma   = (1.0 - moist_cont/100.0)*c_dry + moist_cont/100.0*c_w
    _rho_wet_tdma = rho_dry + moist_cont/100.0*rho_w
    _D_ins_tdma   = k_i_base / (_c_wet_tdma * _rho_wet_tdma)

    print("\nRunning transient 1D insulation solver...")
    T_nh = transient1D_smr(
        t_sol_air_h.astype(np.float64),
        h_o_h.astype(np.float64),
        d_ins = Hi,
        lam_i = k_i_base,
        D     = _D_ins_tdma,
        dx    = 0.005,
        dt    = 10.0,
        h_i   = _h_i_tdma
    )
    Tsi_h       = T_nh[-1, :]
    tdma_melt_h = Tsi_h * _h_i_tdma / (Lf * rho_s) * 3600.0 * 1000.0
    tdma_wc_h   = np.where(
        (tdma_melt_h < 0) | (air_temp_h < 0), 0.0, tdma_melt_h)
    print("  Transient 1D solver complete.")

    emp1_cs = np.cumsum(emp1_wc_h)
    emp2_cs = np.cumsum(emp2_wc_h)
    tdma_cs = np.cumsum(tdma_wc_h)

    steps_per_hour = int(dt_data / dt)
    melt_mm_step   = melt_rate_hist[:-1] * dt * 1000.0
    n_full_hours   = min(n_h, len(melt_mm_step) // steps_per_hour)
    melt_mm_hourly = np.array(
        [np.sum(melt_mm_step[i*steps_per_hour:(i+1)*steps_per_hour])
         for i in range(n_full_hours)]
    )
    snowsim_cs = np.cumsum(melt_mm_hourly)

    emp1_cs_trim = emp1_cs[:n_full_hours]
    emp2_cs_trim = emp2_cs[:n_full_hours]
    tdma_cs_trim = tdma_cs[:n_full_hours]
    emp1_h_trim  = emp1_wc_h[:n_full_hours]
    emp2_h_trim  = emp2_wc_h[:n_full_hours]
    tdma_h_trim  = tdma_wc_h[:n_full_hours]
    days_hourly  = np.arange(n_full_hours) / 24.0

    # Optional SNOWPACK loading
    if snowpack_available:
        print(f"\nLoading SNOWPACK .met data...")
        sp_cs, sp_swe, sp_depth, sp_index = load_snowpack_smr(SNOWPACK_MET)
        print(f"  SNOWPACK cumulative runoff: {sp_cs[-1]:.1f} kg/m^2")
    else:
        sp_cs = sp_swe = sp_depth = sp_index = None

    print(f"\nTotal cumulative SMR over {n_full_hours} h "
          f"({n_full_hours/24:.1f} days):")
    print(f"  RC model:              {snowsim_cs[-1]:>10.1f} mm w.e.")
    print(f"  1D Heat Eq.:           {tdma_cs_trim[-1]:>10.1f} mm w.e.")
    print(f"  Emp. model 1 (Skogs.): {emp1_cs_trim[-1]:>10.1f} mm w.e.")
    print(f"  Emp. model 2 (+ RH):   {emp2_cs_trim[-1]:>10.1f} mm w.e.")
    if sp_cs is not None:
        print(f"  SNOWPACK:              {sp_cs[-1]:>10.1f} kg/m^2")

    # Bias relative to RC model (fractional)
    bias1  = 1.0 - emp1_cs_trim[-1]/snowsim_cs[-1]
    bias2  = 1.0 - emp2_cs_trim[-1]/snowsim_cs[-1]
    bias_t = 1.0 - tdma_cs_trim[-1]/snowsim_cs[-1]

    dif1 = emp1_cs_trim[-1] - snowsim_cs[-1]
    dif2 = emp2_cs_trim[-1] - snowsim_cs[-1]
    dif_t = tdma_cs_trim[-1] - snowsim_cs[-1]

    print(f"\nSnowsim bias vs Emp1: {dif1:+.1f} mm  ({100*bias1:+.1f}%)")
    print(f"Snowsim bias vs Emp2: {dif2:+.1f} mm  ({100*bias2:+.1f}%)")
    print(f"Snowsim bias vs TDMA: {dif_t:+.1f} mm  ({100*bias_t:+.1f}%)")
    if sp_cs is not None:
        sp_cs_trim = sp_cs[:n_full_hours]
        bias_sp = 1.0 - sp_cs_trim[-1]/snowsim_cs[-1]
        dif_sp = sp_cs_trim[-1] - snowsim_cs[-1]
        print(f"Snowsim bias vs SNOWPACK: {dif_sp:+.1f} mm  ({100*bias_sp:+.1f}%)")
    else:
        bias_sp = None

    # Hourly RMSE
    rmse1  = np.sqrt(np.mean((melt_mm_hourly - emp1_h_trim)**2))
    rmse2  = np.sqrt(np.mean((melt_mm_hourly - emp2_h_trim)**2))
    rmse_t = np.sqrt(np.mean((melt_mm_hourly - tdma_h_trim)**2))
    print(f"\nHourly RMSE vs Emp1:  {rmse1:.3f} mm/h")
    print(f"Hourly RMSE vs Emp2:  {rmse2:.3f} mm/h")
    print(f"Hourly RMSE vs TDMA:  {rmse_t:.3f} mm/h")

    # ------------------------------------------------------------------
    #  Figures - saved individually to ./figures/
    # ------------------------------------------------------------------
    print("\nGenerating figures...")
    days    = t_vec / 86400.0
    fig_dir = "figures"
    os.makedirs(fig_dir, exist_ok=True)

    def _save(fig, name):
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, name), dpi=150,
                    bbox_inches='tight')
        plt.close(fig)

    # 1. Snow layer temperatures
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(days, Ta_hist    - 273.15, 'k--', lw=1, label='Air',   alpha=0.7)
    ax.plot(days, Tsoil_hist - 273.15, 'g--', lw=1, label='Soil',  alpha=0.7)
    ax.plot(days, T_hist[:,0] - 273.15, lw=1.5, label='T1 (surface)')
    ax.plot(days, T_hist[:,1] - 273.15, lw=1.5, label='T2 (middle)')
    ax.plot(days, T_hist[:,2] - 273.15, lw=1.5, label='T3 (bottom)')
    ax.axhline(0, color='gray', ls=':', alpha=0.5)
    ax.set(ylabel='Temperature [degC]', xlabel='Time [days]',
           title='Snow Layer Temperatures')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    _save(fig, '01_temperatures.png')

    # 2. Solar radiation
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(days, 0, Isolar_hist, alpha=0.5)
    ax.set(ylabel='Solar [W/m^2]', xlabel='Time [days]',
           title='Solar Radiation Input')
    ax.grid(True, alpha=0.3)
    _save(fig, '02_solar_radiation.png')

    # 3. Liquid water content
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(days, LWC_hist[:,0], lw=1.5, label='Layer 1 (surface)')
    ax.plot(days, LWC_hist[:,1], lw=1.5, label='Layer 2 (middle)')
    ax.plot(days, LWC_hist[:,2], lw=1.5, label='Layer 3 (bottom)')
    ax.axhline(theta_e, color='r', ls='--', lw=1, label='Field capacity')
    ax.set(ylabel='LWC [-]', xlabel='Time [days]',
           title='Liquid Water Content')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    _save(fig, '03_liquid_water_content.png')

    # 4. Precipitation
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(days, 0, Prain_hist * 1000.0 * 3600.0,
                    alpha=0.5, color='blue')
    ax.set(ylabel='Precip [mm/h]', xlabel='Time [days]',
           title='Precipitation')
    ax.grid(True, alpha=0.3)
    _save(fig, '04_precipitation.png')

    # 5. Cumulative melt and runoff
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(days, np.cumsum(melt_rate_hist * dt),
            lw=2, label='Cumulative melt')
    ax.plot(days, np.cumsum(runoff_hist) / rho_w,
            lw=2, label='Cumulative runoff')
    ax.set(ylabel='Water [m w.e.]', xlabel='Time [days]',
           title='Melt and Runoff')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    _save(fig, '05_melt_and_runoff.png')

    # 6. Heat fluxes
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(days, qsolar_hist, lw=1, alpha=0.7, label='Solar')
    ax.plot(days, qrain_hist,  lw=1, alpha=0.7, label='Rain')
    ax.plot(days, qevap_hist,  lw=1, alpha=0.7, label='Evaporation')
    ax.plot(days, qa_hist,     lw=1, alpha=0.7, label='Air convection')
    ax.plot(days, qground_hist,lw=1, alpha=0.7, label='Ground flux')
    ax.set(ylabel='Heat flux [W/m^2]', xlabel='Time [days]',
           title='Surface and Ground Heat Fluxes')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    _save(fig, '06_heat_fluxes.png')

    # 7. Ground interface
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(days, Tsoil_hist - 273.15, 'g-', lw=1.5,
            label='Soil temp (320 cm)')
    ax.plot(days, T_hist[:,2] - 273.15, 'b-', lw=1.5,
            label='T3 (bottom snow)')
    ax.plot(days, qground_hist, 'r-', lw=1, label='Ground heat flux')
    ax.axhline(0, color='gray', ls=':', alpha=0.5)
    ax.set(ylabel='Temperature [degC] / Flux [W/m^2]',
           xlabel='Time [days]', title='Ground Interface')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    _save(fig, '07_ground_interface.png')

    # 8 & 9. Insulation properties (advanced model only)
    if USE_ADVANCED_INSULATION:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax2 = ax.twinx()
        ln1 = ax.plot(days, k_eff_hist, 'b-', lw=1.5, label='k_eff')
        ln2 = ax2.plot(days, alpha_hist, 'r-', lw=1.5, label='alpha_eff')
        ax.set_ylabel('k_eff [W/(m K)]', color='b')
        ax2.set_ylabel('alpha_eff [-]', color='r')
        ax.tick_params(axis='y', labelcolor='b')
        ax2.tick_params(axis='y', labelcolor='r')
        lns  = ln1 + ln2
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, fontsize=8)
        ax.set(xlabel='Time [days]', title='Insulation Properties')
        ax.grid(True, alpha=0.3)
        _save(fig, '08_insulation_properties.png')

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(days, W_hist, lw=1.5, color='steelblue')
        ax.set(ylabel='Moisture [kg/m^2]', xlabel='Time [days]',
               title='Insulation Moisture Content')
        ax.grid(True, alpha=0.3)
        _save(fig, '09_insulation_moisture.png')

    # 10. SWE and depth
    fig, ax = plt.subplots(figsize=(10, 5))
    melt_cumul_m = np.cumsum(melt_rate_hist[:-1]) * dt
    SWE_py = np.maximum(0.0, Hs * rho_s - melt_cumul_m * rho_i)
    ax.plot(days[:-1], SWE_py, 'steelblue', lw=2, label='Python SWE')
    if sp_swe is not None:
        from datetime import datetime as _dt
        t0_sim  = _dt(2024, 4, 1)
        sp_days = np.array([(ts - t0_sim).total_seconds() / 86400.0
                            for ts in sp_index])
        ax.plot(sp_days, sp_swe.values, 'navy', lw=2, ls='--',
                label='SNOWPACK SWE (snow only)')
        if sp_depth is not None:
            ax2 = ax.twinx()
            ax2.plot(sp_days, sp_depth.values, 'cornflowerblue',
                     lw=1, alpha=0.5, label='SP depth [m]')
            ax2.set_ylabel('Depth [m]', color='cornflowerblue')
            ax2.tick_params(axis='y', labelcolor='cornflowerblue')
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, fontsize=7)
        else:
            ax.legend(fontsize=8)
    else:
        ax.legend(fontsize=8)
    ax.set(ylabel='SWE [kg/m^2]', xlabel='Time [days]',
           title='Snow Water Equivalent and Depth')
    ax.grid(True, alpha=0.3)
    _save(fig, '10_swe_and_depth.png')

    # 11. Cumulative SMR comparison
    _sp_bias_str = (f'  |  Bias vs SP: {bias_sp:+.1f} mm'
                    if bias_sp is not None else '')
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(days_hourly, snowsim_cs,   color='steelblue',
            lw=2,   label='RC model')
    ax.plot(days_hourly, tdma_cs_trim, color='purple',
            lw=1.5, ls='-.', label='1D Heat Eq.')
    ax.plot(days_hourly, emp1_cs_trim, color='darkorange',
            lw=1.5, ls='--', label='Emp. 1 (T+solar+wind)')
    ax.plot(days_hourly, emp2_cs_trim, color='forestgreen',
            lw=1.5, ls=':',  label='Emp. 2 (+humidity)')
    if sp_cs is not None:
        ax.plot(days_hourly, sp_cs[:n_full_hours],
                color='red', lw=1.5, ls=(0, (5, 2, 1, 2)),
                label='SNOWPACK')
    ax.set(ylabel='Cumulative melt [mm w.e.]', xlabel='Time [days]',
           title=(f'Cumulative SMR Comparison\n'
                  f'Bias vs Emp1: {bias1:+.1f} mm  |  '
                  f'Bias vs TDMA: {bias_t:+.1f} mm' + _sp_bias_str))
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    _save(fig, '11_cumulative_smr.png')

    # 12. Hourly SMR comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(days_hourly, melt_mm_hourly, color='steelblue',
            lw=1, alpha=0.8, label='RC model')
    ax.plot(days_hourly, tdma_h_trim,    color='purple',
            lw=1, alpha=0.8, ls='-.', label='TDMA 1D')
    ax.plot(days_hourly, emp1_h_trim,    color='darkorange',
            lw=1, alpha=0.7, ls='--', label='Emp. 1')
    ax.plot(days_hourly, emp2_h_trim,    color='forestgreen',
            lw=1, alpha=0.7, ls=':',  label='Emp. 2')
    ax.set(ylabel='Hourly melt rate [mm/h]', xlabel='Time [days]',
           title=(f'Hourly SMR Comparison\n'
                  f'RMSE vs Emp1: {rmse1:.3f} mm/h  |  '
                  f'RMSE vs TDMA: {rmse_t:.3f} mm/h'))
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    _save(fig, '12_hourly_smr.png')

    print(f"  Saved 12 figures to '{fig_dir}/'")
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
