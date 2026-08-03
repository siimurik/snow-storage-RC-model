################################################################################
#                                                                              #
#               Snow Storage RC Thermal Model  -  Nordell Validation           #
#                                                                              #
#  Authors : Siim Erik Pugal, siim.pugal@taltech.ee                            #
#            Hossein Alimohammadi,  hossein.alimohammadi@taltech.ee            #
#  Date    : June 2026                                                         #
#  License : MIT                                                               #
#                                                                              #
# ---------------------------------------------------------------------------- #
#                                                                              #
#  DESCRIPTION                                                                 #
#                                                                              #
#  Validation script for Nordell & Sundin (1998) snow storage melt volume.     #
#  The snow pile is divided into three vertical layers (top, middle, bottom)   #
#  covered by an insulating woodchip layer whose thermal conductivity and      #
#  absorptivity depend on moisture content.  A surface energy balance (SEB)    #
#  is solved at the cover layer, and the snow temperatures are advanced with   #
#  a 4th-order Runge-Kutta (RK4) scheme.                                       #
#                                                                              #
# ---------------------------------------------------------------------------- #
#                                                                              #
#  PHYSICS OVERVIEW                                                            #
#                                                                              #
#  1. SURFACE ENERGY BALANCE (SEB)                                             #
#     The cover-layer temperature Tc is found by solving the residual:         #
#       R_area * (qSW + qLW + qH + qE + qRAIN) - qins = 0                      #
#     via bisection on each time step.                                         #
#       qLW  : sky long-wave minus cover emission                              #
#       qH   : sensible heat (bulk aerodynamic, CH_eff * U * dT)               #
#       qE   : latent heat (wind-driven evaporation from cover)                #
#       qRAIN: rain sensible heat (above-freezing rain only)                   #
#       qins : conduction from cover to top snow layer                         #
#                                                                              #
#  2. RC THERMAL NETWORK                                                       #
#     Three snow layers connected by R12 / R23.  The bottom layer exchanges    #
#     heat with the soil via R3g (combined snow + ground insulation layer).    #
#     ODE:  dT/dt = dTdt_snow(T, qtop, R12, R23, R3g, Tsoil, Cs)               #
#                                                                              #
#  3. MELTING                                                                  #
#     Each layer clamps to 0 °C; excess energy converts to melt water          #
#     (dM = dE / (rho_i * Lf)).  Melt water is added to the layer LWC.         #
#                                                                              #
#  4. REFREEZING (Bartelt & Lehning 2002 / cold-content approach)              #
#     Sub-zero layers refreeze available liquid water up to the cold-content   #
#     limit; latent heat release warms the layer toward 0 °C.                  #
#                                                                              #
#  5. PERCOLATION - BUCKET METHOD                                              #
#     Excess LWC above irreducible saturation theta_e percolates downward.     #
#     Water leaving the bottom layer becomes runoff.                           #
#                                                                              #
#  6. VOLUME TRACKING                                                          #
#     Cumulative ice melt is converted to volume loss                          #
#       V(t) = V0 - Aref * (rho_i / rho_s) * cumulative_melt(t)                #
#     and compared against digitised curves from Nordell & Sundin (1998).      #
#                                                                              #
# ---------------------------------------------------------------------------- #
#                                                                              #
#  VALIDATION CASES                                                            #
#                                                                              #
#  A. NORDELL 1998   - V0 = 30 000 m^3, rho_s = 650 kg/m^3, Hs_eff = 3.3 m     #
#  B. SKOGSBERG 2005 - V0 = 27 000 m^3, rho_s = 730 kg/m^3, Aref = 8 400 m^2   #
#                                                                              #
# ---------------------------------------------------------------------------- #
#                                                                              #
#  INPUT DATA                                                                  #
#                                                                              #
#  Primary forcing  : DATA_2024.csv (hourly met data; columns below)           #
#  Reference forcing: Nordell_Fig4_Forcing_AllVars_refYear.csv                 #
#    Columns (either file):                                                    #
#      Time / Date_refYear  : timestamp                                        #
#      Temp_C               : air temperature               [°C]               #
#      Air_Vel_m/s_10m      : wind speed at 10 m            [m/s]              #
#      Prec_m/h             : precipitation rate            [m/h]              #
#      Glo_Sol_Ir_W/m2      : global solar irradiance       [W/m^2]            #
#      RH_%                 : relative humidity             [%]                #
#      Soil_Temp_C          : soil temperature              [°C]               #
#                                                                              #
#  OUTPUT                                                                      #
#    Nordell_validation_volume.png / .pdf  - volume comparison plot            #
#    Nordell_digitized_snow_volume.csv     - digitised paper data              #
#    Console: run summary, RMSE per insulation thickness                       #
#                                                                              #
# ---------------------------------------------------------------------------- #
#                                                                              #
#  DEPENDENCIES                                                                #
#    numpy, pandas, matplotlib, numba, scipy (optional - PCHIP interpolation)  #
#    Standard library: copy, math                                              #
#                                                                              #
#  USAGE                                                                       #
#    Place DATA_2024.csv and Nordell_Fig4_Forcing_AllVars_refYear.csv in the   #
#    working directory, then run:                                              #
#       python3 validat.py                                                     #
#                                                                              #
################################################################################
import os
import copy
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit

try:
    from scipy.interpolate import PchipInterpolator

    def interp_pchip(x_src, y_src, x_new):
        """Monotone cubic (PCHIP) interpolation; falls back to linear if SciPy absent."""
        f = PchipInterpolator(x_src, y_src, extrapolate=True)
        return f(x_new)

except ImportError:
    print("SciPy not found - falling back to linear interpolation.")

    def interp_pchip(x_src, y_src, x_new):
        return np.interp(x_new, x_src, y_src)


# ==============================================================================
#  PHYSICAL CONSTANTS  (module-level; shared by all routines)
# ==============================================================================

SIGMA   = 5.670374419e-8   # Stefan-Boltzmann constant          [W/m^2 K^4]
LF      = 3.34e5           # Latent heat of fusion              [J/kg]
LV      = 2.5e6            # Latent heat of vaporisation        [J/kg]
RHO_I   = 917.0            # Ice density                        [kg/m^3]
RHO_W   = 1000.0           # Liquid water density               [kg/m^3]
RHO_AIR = 1.225            # Air density at sea level           [kg/m^3]
CP_AIR  = 1005.0           # Air specific heat                  [J/(kg K)]
C_W     = 4180.0           # Water specific heat                [J/(kg K)]
C_S     = 2100.0           # Ice/snow specific heat             [J/(kg K)]
P0      = 101325.0         # Reference pressure                 [Pa]
TFREEZE = 273.15           # Freezing temperature               [K]
THETA_E = 0.04             # Irreducible (field-capacity) LWC   [-]

# ==============================================================================
#  INSULATION MODEL TOGGLE
#  True  = aging + porosity insulation model
#  False = simple moisture-only model
# ==============================================================================
USE_AGING_MODEL = False

# ==============================================================================
#  GENERAL HELPER UTILITIES
# ==============================================================================

def dict_to_ns(d):
    """Recursively convert a nested dict into a SimpleNamespace."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_ns(v) for k, v in d.items()})
    if isinstance(d, list):
        return [dict_to_ns(x) for x in d]
    return d


def ensure_attr(obj, name, default):
    """Set an attribute on *obj* only if it does not already exist."""
    if not hasattr(obj, name):
        setattr(obj, name, default)


def rmse(a, b):
    """Root-mean-square error, ignoring non-finite pairs."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    return np.sqrt(np.mean((a[mask] - b[mask]) ** 2))


# ==============================================================================
#  NUMBA-JIT HELPER FUNCTIONS
# ==============================================================================

@njit
def e_sat_scalar(T_K):
    """
    Saturation vapour pressure at temperature T_K [K], scalar only.

    Uses Magnus-type formula:
      - Above freezing : coefficients for liquid water
      - Below freezing : coefficients for ice surface

    Returns
    -------
    e_sat : float  [Pa]
    """
    dT = T_K - TFREEZE
    if T_K >= TFREEZE:
        return 611.2 * math.exp(17.27 * dT / (dT + 237.3))
    else:
        return 611.2 * math.exp(22.46 * dT / (dT + 272.62))


@njit
def dTdt_snow(T, qtop, R12, R23, R3g, Tsoil_K, Cs):
    """
    ODE right-hand side for 3-layer snow temperatures [K/s].

    Parameters
    ----------
    T        : float64[3]  Layer temperatures [K].
    qtop     : float       Heat flux into the top layer [W/m^2].
    R12, R23 : float       Inter-layer thermal resistances [m^2 K/W].
    R3g      : float       Bottom-layer to soil resistance [m^2 K/W].
    Tsoil_K  : float       Soil temperature [K].
    Cs       : float64[3]  Per-layer volumetric heat capacity [J/(m^2 K)],
                            computed each time step from the current ice
                            fraction, LWC, and layer height so densification
                            and wetting are reflected in the thermal inertia.

    Returns
    -------
    dT : float64[3]   [dT1/dt, dT2/dt, dT3/dt]
    """
    T1, T2, T3 = T[0], T[1], T[2]
    q12 = (T1 - T2) / R12
    q23 = (T2 - T3) / R23
    q3g = (T3 - Tsoil_K) / R3g   # positive = heat leaving snow downward

    dT = np.empty(3)
    dT[0] = (qtop - q12) / Cs[0]
    dT[1] = (q12  - q23) / Cs[1]
    dT[2] = (q23  - q3g) / Cs[2]
    return dT


@njit
def rk4_snow(T, qtop, R12, R23, R3g, Tsoil_K, Cs, dt):
    """
    Advance snow temperatures by one RK4 step.

    Parameters
    ----------
    T        : float64[3]  Current layer temperatures [K].
    qtop     : float       Surface heat flux [W/m^2] (held constant over step).
    R12, R23 : float       Inter-layer resistances [m^2 K/W].
    R3g      : float       Bottom-to-soil resistance [m^2 K/W].
    Tsoil_K  : float       Soil temperature [K].
    Cs       : float64[3]  Per-layer heat capacity [J/(m^2 K)] (see dTdt_snow).
    dt       : float       Time step [s].

    Returns
    -------
    T_new : float64[3]   Updated layer temperatures [K].
    """
    k1 = dTdt_snow(T,                    qtop, R12, R23, R3g, Tsoil_K, Cs)
    k2 = dTdt_snow(T + (dt / 2.0) * k1, qtop, R12, R23, R3g, Tsoil_K, Cs)
    k3 = dTdt_snow(T + (dt / 2.0) * k2, qtop, R12, R23, R3g, Tsoil_K, Cs)
    k4 = dTdt_snow(T + dt * k3,          qtop, R12, R23, R3g, Tsoil_K, Cs)
    return T + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


@njit
def refreezing_layer(T_layer, LWC_layer, ice_frac, dz_s):
    """
    Refreeze liquid water in one snow layer (Bartelt & Lehning 2002).

    Refreezing is limited by both the available LWC and the cold content
    of the ice-water mixture.  Temperature is updated to account for the
    latent heat released.

    Parameters
    ----------
    T_layer   : float  Layer temperature [K].
    LWC_layer : float  Volumetric liquid water content [-].
    ice_frac  : float  Volumetric ice fraction [-].
    dz_s      : float  Layer thickness [m].

    Returns
    -------
    new_T         : float  Updated temperature [K].
    new_LWC       : float  Updated LWC [-].
    new_ice_frac  : float  Updated ice fraction [-].
    refrozen_mass : float  Mass of refrozen water [kg/m^2].
    """
    if (T_layer >= TFREEZE) or (LWC_layer <= 0.0):
        return T_layer, LWC_layer, ice_frac, 0.0

    dT_max = T_layer - TFREEZE
    denom  = RHO_W * (LF - dT_max * (C_S - C_W))
    dtheta_w_max = (
        -(dT_max * (ice_frac * RHO_I * C_S + LWC_layer * RHO_W * C_W)) / denom
    )
    dtheta_w     = min(LWC_layer, dtheta_w_max)
    dtheta_i     = (RHO_W / RHO_I) * dtheta_w
    dT           = (dtheta_w * RHO_W * LF) / (
                       ice_frac * RHO_I * C_S + LWC_layer * RHO_W * C_W)

    new_T         = T_layer   + dT
    new_LWC       = LWC_layer - dtheta_w
    new_ice_frac  = ice_frac  + dtheta_i
    refrozen_mass = dtheta_w  * dz_s * RHO_W
    return new_T, new_LWC, new_ice_frac, refrozen_mass


@njit
def percolate_water(LWC_array, heights):
    """
    Bucket-method liquid water percolation through layers.

    Excess above THETA_E drains layer-by-layer downward.  Water leaving
    the bottom layer accumulates as runoff.

    Parameters
    ----------
    LWC_array : float64[n]  Volumetric LWC per layer [-].
    heights   : float64[n]  Layer thicknesses [m].

    Returns
    -------
    new_LWC : float64[n]  Updated LWC [-].
    runoff  : float        Runoff mass from bottom [kg/m^2].
    """
    n       = len(LWC_array)
    new_LWC = LWC_array.copy()

    for i in range(n - 1):
        if new_LWC[i] > THETA_E:
            excess         = new_LWC[i] - THETA_E
            new_LWC[i]     = THETA_E
            new_LWC[i + 1] += excess * heights[i] / heights[i + 1]

    runoff = 0.0
    if new_LWC[n - 1] > THETA_E:
        excess          = new_LWC[n - 1] - THETA_E
        new_LWC[n - 1] = THETA_E
        runoff          = excess * heights[n - 1] * RHO_W

    return new_LWC, runoff


@njit
def densification_boone(T_layers, LWC_layers, ice_fractions, heights, dz_initial, dt):
    """
    Overburden densification for 3-layer snow model (Essery et al. 2013 - Boone method).
    Numba-optimized and mass-conservative.

    Parameters
    ----------
    T_layers      : float64[3]  Layer temperatures [K].
    LWC_layers    : float64[3]  Volumetric liquid water content per layer [-].
    ice_fractions : float64[3]  Volumetric ice fraction per layer [-].
    heights       : float64[3]  Current layer thicknesses [m].
    dz_initial    : float       Initial (reference) layer thickness [m].
    dt            : float       Time step [s].

    Returns
    -------
    new_heights       : float64[3]  Updated layer thicknesses [m].
    new_ice_fractions : float64[3]  Updated ice fractions [-].
    new_LWC           : float64[3]  Updated LWC [-].
    """
    c1 = 2.8e-6; c2 = 0.042; c3 = 0.046; c4 = 0.081; c5 = 0.018
    eta0 = 3.7e7; rho0 = 150.0

    new_heights       = heights.copy()
    new_ice_fractions = ice_fractions.copy()
    new_LWC           = LWC_layers.copy()

    M_s = 0.0
    for i in range(3):
        rho_layer = ice_fractions[i] * RHO_I + LWC_layers[i] * RHO_W

        if rho_layer < 800.0 and heights[i] > 0.01:
            T_C = T_layers[i] - TFREEZE

            # Viscosity
            eta = eta0 * math.exp(c4 * (-T_C) + c5 * rho_layer)

            # Fractional density change rate
            dRho = ((M_s * 9.81) / eta +
                    c1 * math.exp(-c2 * (-T_C) - c3 * max(0.0, rho_layer - rho0))) * dt

            # Limit to 10 % per step for stability
            dRho = min(dRho, 0.1)

            # Mass conservation: volumetric fractions scale up as bulk volume shrinks
            new_ice_fractions[i] = (1.0 + dRho) * ice_fractions[i]
            new_LWC[i]           = (1.0 + dRho) * LWC_layers[i]

            # Compaction: reduce layer thickness
            new_heights[i] = max((1.0 - dRho) * heights[i], dz_initial * 0.1)

        # Accumulate overburden mass for the layer beneath
        M_s += rho_layer * heights[i]

    return new_heights, new_ice_fractions, new_LWC


# ==============================================================================
#  SURFACE ENERGY BALANCE (SEB) ROUTINES
# ==============================================================================

def cover_fluxes(Tc, Ta, U, RH, Train, Pr, Ts1, Rins, f, qSW, p):
    """
    Compute individual heat fluxes at the insulating cover layer.

    Parameters
    ----------
    Tc   : float  Cover temperature [K].
    Ta   : float  Air temperature [K].
    U    : float  Wind speed [m/s].
    RH   : float  Relative humidity [0-1].
    Train: float  Rain temperature [K].
    Pr   : float  Precipitation rate [m/s].
    Ts1  : float  Top snow layer temperature [K].
    Rins : float  Insulation resistance [m^2 K/W].
    f    : float  Moisture saturation fraction [0-1].
    qSW  : float  Absorbed shortwave at cover [W/m^2].
    p    : namespace  Model parameters.

    Returns
    -------
    qLW, qH, qE, qRAIN, qins : float  Individual flux components [W/m^2].
    """
    # Long-wave: sky emission minus cover emission
    e_a_Pa  = RH * e_sat_scalar(Ta)
    e_a_hPa = e_a_Pa / 100.0
    eps_sky  = min(max(1.24 * (e_a_hPa / Ta) ** (1.0 / 7.0), 0.6), 1.0)
    qLW      = eps_sky * SIGMA * Ta**4 - p.eps_c * SIGMA * Tc**4

    # Sensible heat (bulk aerodynamic)
    qH = RHO_AIR * CP_AIR * p.CH * (1.0 + p.CV) * U * (Ta - Tc)

    # Latent heat (wind-driven evaporation, modified by cover wetness)
    e_c_Pa  = e_sat_scalar(Tc)
    q_a     = 0.622 * e_a_Pa / (P0 - 0.378 * e_a_Pa)
    q_star  = 0.622 * e_c_Pa / (P0 - 0.378 * e_c_Pa)
    E0      = p.f_shelter * RHO_AIR * p.CE * U * (q_star - q_a)
    E       = E0 * math.exp(-p.beta_w * f)
    L       = LV + LF if Tc < TFREEZE else LV
    qE      = -L * E

    # Rain sensible heat (above-freezing rain only)
    qRAIN = RHO_W * C_W * Pr * (Train - TFREEZE) if (Ta > TFREEZE and Pr > 0) else 0.0

    # Conductive flux from cover to top snow layer
    qins = (Tc - Ts1) / Rins

    return qLW, qH, qE, qRAIN, qins


def cover_seb_residual(Tc, Ta, U, RH, Train, Pr, Ts1, Rins, f, qSW, p, rA):
    """
    SEB residual: total flux in (times area ratio) minus flux into snow.

    Zero when the cover is in instantaneous energy balance.
    """
    qLW, qH, qE, qRAIN, qins = cover_fluxes(
        Tc, Ta, U, RH, Train, Pr, Ts1, Rins, f, qSW, p
    )
    return rA * (qSW + qLW + qH + qE + qRAIN) - qins


def bisect_root(fun, lo, hi, tol=1e-6, max_iter=100):
    """
    Simple bisection root-finder for continuous scalar functions.

    Parameters
    ----------
    fun      : callable  f(x) -> float; must change sign on [lo, hi].
    lo, hi   : float     Initial bracket.
    tol      : float     Convergence tolerance on |f| and bracket width.
    max_iter : int       Maximum iterations.

    Returns
    -------
    root : float
    """
    flo = fun(lo)
    fhi = fun(hi)

    if not (np.isfinite(flo) and np.isfinite(fhi)):
        raise ValueError("bisect_root: non-finite bracket values.")
    if np.sign(flo) == np.sign(fhi):
        raise ValueError("bisect_root: bracket does not contain a sign change.")

    for _ in range(max_iter):
        mid  = 0.5 * (lo + hi)
        fmid = fun(mid)
        if not np.isfinite(fmid):
            break
        if abs(fmid) < tol or abs(hi - lo) < tol:
            return mid
        if np.sign(flo) != np.sign(fmid):
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid

    return 0.5 * (lo + hi)


def solve_cover_temperature(Ta, U, RH, Train, Pr, Ts1, Rins, f, qSW, p, rA):
    """
    Find cover temperature Tc by solving the SEB residual via bisection.

    Falls back to a coarse grid search when bisection bracketing fails.

    Parameters
    ----------
    (same as cover_fluxes plus rA and p.solve sub-namespace)

    Returns
    -------
    Tc : float  Cover temperature [K].
    """
    Tc_min = TFREEZE + p.solve.Tc_min_C
    Tc_max = TFREEZE + p.solve.Tc_max_C

    fun = lambda Tc: cover_seb_residual(Tc, Ta, U, RH, Train, Pr, Ts1, Rins, f, qSW, p, rA)

    lo = max(Ta - p.solve.bracket_dT_lo, Tc_min)
    hi = min(Ta + p.solve.bracket_dT_hi, Tc_max)
    Flo, Fhi = fun(lo), fun(hi)

    for ntry in range(1, p.solve.max_expand + 1):
        if np.isfinite(Flo) and np.isfinite(Fhi) and np.sign(Flo) != np.sign(Fhi):
            break
        factor = p.solve.expand_factor ** ntry
        lo  = max(Ta - p.solve.bracket_dT_lo * factor, Tc_min)
        hi  = min(Ta + p.solve.bracket_dT_hi * factor, Tc_max)
        Flo, Fhi = fun(lo), fun(hi)

    if np.isfinite(Flo) and np.isfinite(Fhi) and np.sign(Flo) != np.sign(Fhi):
        return bisect_root(fun, lo, hi, tol=1e-6, max_iter=100)

    # Fallback: coarse grid scan
    Tc_grid = np.linspace(lo, hi, 25)
    Fg      = np.array([fun(x) for x in Tc_grid])
    return Tc_grid[np.argmin(np.abs(Fg))]


# ==============================================================================
#  DATA LOADING AND FORCING CONSTRUCTION
# ==============================================================================

# Candidate column names for adaptive CSV parsing
_COL_ALIASES = {
    "T"  : ["Temp_C", "Ta_C"],
    "U"  : ["Air_Vel_m/s_10m", "Air_Vel_m_s", "U_ms"],
    "RH" : ["RH_%", "RH_", "RH_pct"],
    "G"  : ["Glo_Sol_Ir_W/m2", "Glo_Sol_Ir_W_m2", "G_W_m2"],
    "Pr" : ["Prec_m/h", "Prec_m_h", "Pr_mm_day"],
    "Tg" : ["Soil_Temp_C"],
}


def detect_columns(met_columns, aliases=_COL_ALIASES):
    """
    Map generic field keys ('T', 'U', …) to actual CSV column names.

    Parameters
    ----------
    met_columns : sequence of str  Columns present in the DataFrame.
    aliases     : dict             {key: [candidate, …]}.

    Returns
    -------
    mapping : dict  {key: actual_column_name or None}
    """
    return {
        key: next((c for c in cands if c in met_columns), None)
        for key, cands in aliases.items()
    }


def clean_met(df):
    """Strip whitespace and BOM characters from DataFrame column names."""
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    return df


def build_forc_arrays(met, col_map, t_out, dt):
    """
    Interpolate raw met DataFrame columns onto the uniform time grid *t_out*.

    Parameters
    ----------
    met     : pd.DataFrame   Meteorological data with 'tsec' column.
    col_map : dict           {key: column_name or None}.
    t_out   : np.ndarray     Target time grid [s].
    dt      : float          Target time step [s].

    Returns
    -------
    forc : SimpleNamespace  Forcing arrays on t_out.
    """
    ts = met["tsec"].to_numpy(dtype=float)

    forc    = SimpleNamespace()
    forc.dt = dt
    forc.t  = t_out
    forc.Nt = len(t_out)
    forc.days = t_out / 86400.0

    forc.Ta = np.interp(t_out, ts, met[col_map["T"]].to_numpy(dtype=float) + TFREEZE)
    forc.U  = np.interp(t_out, ts, met[col_map["U"]].to_numpy(dtype=float))

    rh_div  = 100.0 if col_map["RH"] in ("RH_%", "RH_", "RH_pct") else 1.0
    forc.RH = np.interp(t_out, ts, met[col_map["RH"]].to_numpy(dtype=float) / rh_div)

    forc.G  = np.interp(t_out, ts, met[col_map["G"]].to_numpy(dtype=float))

    Pr_raw  = met[col_map["Pr"]].to_numpy(dtype=float)
    Pr_mps  = (Pr_raw * 1e-3 / 86400.0 if col_map["Pr"] == "Pr_mm_day"
               else Pr_raw / 3600.0)
    forc.Pr = np.interp(t_out, ts, Pr_mps)

    if col_map["Tg"]:
        forc.Tg = np.interp(t_out, ts, met[col_map["Tg"]].to_numpy(dtype=float) + TFREEZE)
    else:
        forc.Tg = None

    return forc


def load_primary_forcing(data_file):
    """
    Load the primary hourly meteorological file and build forcing arrays.

    Parameters
    ----------
    data_file : str  Path to the CSV file.

    Returns
    -------
    forc : SimpleNamespace  Interpolated forcing on a 10-min grid.
    base : SimpleNamespace  Physical parameters and geometry.
    p0   : SimpleNamespace  Model tuning parameters.
    """
    met = clean_met(pd.read_csv(data_file))
    met["Time_dt"] = pd.to_datetime(met["Time"], format="%Y-%m-%dT%H:%M", errors="coerce")
    if met["Time_dt"].isna().any():
        met["Time_dt"] = pd.to_datetime(met["Time"], errors="coerce")
    if met["Time_dt"].isna().any():
        raise ValueError(f"Could not parse all Time values in {data_file}.")

    t0 = met["Time_dt"].iloc[0]
    met["tsec"] = (met["Time_dt"] - t0).dt.total_seconds()

    col_map  = detect_columns(met.columns)
    missing  = [k for k in ("T", "U", "RH", "G", "Pr") if not col_map[k]]
    if missing:
        raise KeyError(f"Missing columns in {data_file}: {missing}")

    print(f"\nLoading primary data : {data_file}")
    print(f"  Columns detected   : T='{col_map['T']}', U='{col_map['U']}', "
          f"RH='{col_map['RH']}', G='{col_map['G']}', Pr='{col_map['Pr']}'")
    if col_map["Tg"]:
        print(f"  Soil temperature   : '{col_map['Tg']}'  (dynamic BC enabled)")
    else:
        print("  Soil temperature   : not found - using fixed ground temperature")

    dt    = 600.0
    t_out = np.arange(0.0, met["tsec"].iloc[-1] + dt, dt, dtype=float)
    forc  = build_forc_arrays(met, col_map, t_out, dt)

    # -----------------------------------------------------------------
    #  Physical constants and geometry (base)
    # -----------------------------------------------------------------
    base = SimpleNamespace()

    base.par = SimpleNamespace(
        sigma=SIGMA, rho_air=RHO_AIR, cp_air=CP_AIR, P0=P0,
        Lv=LV, Lf=LF, rho_i=RHO_I, rho_w=RHO_W,
        c_w=C_W, c_s=C_S, Tf=TFREEZE,
    )

    Hs = 2.0;  Ns = 3;  dz = Hs / Ns
    base.snow = SimpleNamespace(
        Hs=Hs, Ns=Ns, dz=dz, rho=400, c=C_S, k=0.25,
        Cs=400 * C_S * dz,
        R12=dz / 0.25, R23=dz / 0.25,
        T0=np.array([TFREEZE - 2, TFREEZE - 4, TFREEZE - 6], dtype=float),
    )

    base.ground = SimpleNamespace(
        Hg=0.3, kg=0.04,
        h_ground=2.5,
        # Robin BC: conduction through bottom snow layer
        #         + conduction through ground insulation layer
        #         + contact interface resistance  1/h_ground
        R3g=dz / 0.25 + 0.3 / 0.04 + 1.0 / 2.5,
        Tg=TFREEZE + 2.0,      # static fallback
    )

    base.meta = SimpleNamespace(enable_volume=False, V0=np.nan,
                                rho_s=base.snow.rho, Hs_eff=Hs, Aref=np.nan)

    # -----------------------------------------------------------------
    #  Default model tuning parameters (p0)
    # -----------------------------------------------------------------
    p0 = SimpleNamespace(
        CH         = 0.002855,
        CE         = 0.001408,
        f_shelter  = 0.1622,
        beta_w     = 4.714,
        eps_c      = 0.8318,
        a_snow     = 0.4387,
        tau_dry    = 0.111,
        tau_wet    = 0.03651,
        k_dry      = 0.2596,
        k_sat      = 0.2697,
        n_k        = 1.896,
        CV         = 0.6043,
        Hi=0.6, 
        W_sat=30.0, W_field=10.0, KD=5e-6, 
        alb_dry=0.65, alb_wet=0.50,
        U10=1.0,
        # ======== Advanced insulation aging & porosity ========
        # Conductivity aging
        delta_k_age = 0.5,        # conductivity increase per year of age (relative)
        tau_k_years = 2.0,        # timescale for conductivity aging (years)
        
        # Absorptivity aging  
        delta_alpha_age = 0.05,   # absorptivity increase per year of age (relative)
        tau_alpha_years = 2.0,    # timescale for absorptivity aging (years)
        
        # Porosity effects on conductivity
        zeta0 = 0.25,             # initial porosity (dimensionless)
        gamma_H = 0.5,            # exponent for porosity effect on conductivity
        gamma_W = 2.0,            # exponent for moisture effect on conductivity
        # ================================================================
    )

    required = ["CH", "CE", "eps_c", "f_shelter", "Hi", "k_dry", "k_sat",
                "n_k", "W_sat", "W_field", "KD", "beta_w", "alb_dry",
                "alb_wet", "tau_dry", "tau_wet", "a_snow", "U10", "CV"]
    missing = [name for name in required if not hasattr(p0, name)]
    if missing:
        raise ValueError(f"p0 is missing fields: {', '.join(missing)}")

    p0.solve = SimpleNamespace(
        Tc_min_C=-50, Tc_max_C=50,
        bracket_dT_lo=40, bracket_dT_hi=20,
        max_expand=6, expand_factor=1.6,
    )

    return forc, base, p0


def load_reference_forcing(filename, dt):
    """
    Load the reference-year forcing CSV and build forcing arrays.

    Handles 'Date_refYear' (daily rows) and 'Time' (hourly rows) layouts.

    Parameters
    ----------
    filename : str    Path to the reference CSV file.
    dt       : float  Target time step [s].

    Returns
    -------
    forc : SimpleNamespace  Forcing arrays on a uniform dt grid.
    """
    met = clean_met(pd.read_csv(filename))
    print(f"\nLoading reference forcing: {filename}")
    print(f"  Detected columns: {met.columns.tolist()}")

    # -- Parse time column --
    if "Date_refYear" in met.columns:
        s = met["Date_refYear"].astype(str).str.strip().str.replace("\ufeff", "", regex=False)
        t_parsed = pd.to_datetime(s, format="%d-%b-%y", errors="coerce")
        if t_parsed.isna().any():
            t_parsed = t_parsed.fillna(pd.to_datetime(s, errors="coerce", dayfirst=True))
        if t_parsed.isna().any():
            raise ValueError("Could not parse all dates in 'Date_refYear'.")
        met["Time_dt"] = t_parsed

        met["row_in_day"] = met.groupby("Time_dt").cumcount()
        nmax = max(met["row_in_day"].max() + 1, 1)
        met["tsec"] = (
            (met["Time_dt"] - met["Time_dt"].iloc[0]).dt.total_seconds()
            + met["row_in_day"] * (86400.0 / nmax)
        )
    elif "Time" in met.columns:
        s = met["Time"].astype(str).str.strip().str.replace("\ufeff", "", regex=False)
        met["Time_dt"] = pd.to_datetime(s, errors="coerce")
        if met["Time_dt"].isna().any():
            raise ValueError("Could not parse all values in 'Time'.")
        met["tsec"] = (met["Time_dt"] - met["Time_dt"].iloc[0]).dt.total_seconds()
    else:
        raise KeyError(f"No usable time column found. Available: {met.columns.tolist()}")

    col_map = detect_columns(met.columns)
    missing = [k for k in ("T", "U", "RH", "G", "Pr") if not col_map[k]]
    if missing:
        raise KeyError(f"Missing columns in {filename}: {missing}")

    print(f"  Using columns - T='{col_map['T']}', U='{col_map['U']}', "
          f"RH='{col_map['RH']}', G='{col_map['G']}', Pr='{col_map['Pr']}'")
    if col_map["Tg"]:
        print(f"  Soil temperature: '{col_map['Tg']}'")
    else:
        print("  Soil temperature: not found - using fixed ground temperature")

    t_out = np.arange(0.0, float(met["tsec"].iloc[-1]) + dt, dt, dtype=float)
    return build_forc_arrays(met, col_map, t_out, dt)


# ==============================================================================
#  Advanced Insulation Aging & Porosity Functions
# ==============================================================================

def update_insulation_properties(W, age_days, zeta, Hi, p):
    """
    Update effective insulation properties accounting for moisture,
    aging, and porosity effects.
    
    Parameters
    ----------
    W        : float  Cover moisture content [% of saturation]
    age_days : float  Insulation age [days]
    zeta     : float  Current porosity [-]
    Hi       : float  Insulation thickness [m]
    p        : SimpleNamespace  Model parameters
    
    Returns
    -------
    k_eff      : float  Effective conductivity [W/(mK)]
    alpha_eff  : float  Effective absorptivity [-]
    tau_eff    : float  Effective transmissivity [-]
    new_zeta   : float  Updated porosity [-]
    """
    # Moisture saturation fraction
    f = min(1.0, max(0.0, W / p.W_sat))
    
    # Age in years
    age_yr = age_days / 365.0
    
    # ---- Conductivity with porosity and aging effects ----
    # Base moisture-dependent conductivity
    k_moist = p.k_dry + (p.k_sat - p.k_dry) * (f ** p.n_k)
    
    # Porosity effect: zeta * exp(-gamma_H * Hi) * exp(-gamma_W * f)
    # This accounts for compaction with depth and moisture
    porosity_factor = zeta * np.exp(-p.gamma_H * Hi) * np.exp(-p.gamma_W * f)
    
    # Aging factor: 1 + delta_k_age * (1 - exp(-age_yr / tau_k_years))
    k_age_factor = 1.0 + p.delta_k_age * (1.0 - np.exp(-age_yr / p.tau_k_years))
    
    # Combined conductivity
    k_eff = k_moist * (1.0 + porosity_factor) * k_age_factor
    
    # ---- Optical properties with aging ----
    # Base moisture-dependent albedo
    alpha_moist = p.alb_dry + (p.alb_wet - p.alb_dry) * (f ** p.beta_w)
    
    # Aging effect on absorptivity
    alpha_age = alpha_moist + p.delta_alpha_age * (1.0 - np.exp(-age_yr / p.tau_alpha_years))
    alpha_eff = min(max(alpha_age, 0.0), 1.0)
    
    # Transmissivity
    tau_moist = p.tau_dry + (p.tau_wet - p.tau_dry) * (f ** p.beta_w)
    tau_eff = min(max(tau_moist, 0.0), 1.0)
    
    # Update porosity (compaction/settlement)
    # Simple exponential settlement with time and moisture
    new_zeta = p.zeta0 * np.exp(-p.gamma_H * Hi) * np.exp(-p.gamma_W * f)
    
    return k_eff, alpha_eff, tau_eff, new_zeta

# ==============================================================================
#  CORE SNOW MODEL
# ==============================================================================

def run_snow_model(p, forc, base):
    """
    Integrate the 3-layer RC snow model forward in time.

    At each step:
      1. Cover SEB is solved for Tc (bisection).
      2. Snow temperatures are advanced with RK4 (Numba-JIT).
      3. Refreezing is applied to sub-zero layers.
      4. Melting is applied to layers exceeding 0 °C.
      5. Liquid water percolates downward (bucket method).
      6. Cover moisture W is updated.

    Parameters
    ----------
    p    : SimpleNamespace  Tuning parameters (Hi, k_dry, …).
    forc : SimpleNamespace  Forcing arrays (Ta, U, RH, G, Pr, Tg, dt, …).
    base : SimpleNamespace  Geometry, constants, and initial conditions.

    Returns
    -------
    out : SimpleNamespace  Simulation outputs (histories, totals, volume).
    """
    dt, Nt = forc.dt, forc.Nt
    sn, gr  = base.snow, base.ground

    # --- State variables ---
    T         = np.array(sn.T0, dtype=float).copy()
    W         = 5.0                                   # cover moisture [% sat]
    LWC_snow  = np.zeros(3, dtype=float)
    # Initial ice fraction derived from bulk density (sn.rho) so that the
    # per-case density overrides (configure_case) still set the correct
    # initial thermal mass now that Cs is computed dynamically below.
    ice_fracs = np.full(3, sn.rho / RHO_I, dtype=float)
    heights   = np.full(3, sn.dz, dtype=float)

    # ======== Advanced Insulation Aging & Porosity Functions ========
    age_days = 0.0           # insulation age [days]
    zeta     = p.zeta0       # current porosity [-]
    # ============================================================

    # ---- Densification tracking ----
    heights_hist  = np.full((3, Nt), np.nan)    # layer thickness evolution [m]
    ice_frac_hist = np.full((3, Nt), np.nan)    # ice fraction evolution    [-]
    qground_hist = np.full(Nt, np.nan)

    # ---- Dynamic heat capacity tracking (ported from v7) ----
    # Per-layer volumetric heat capacity [J/(m^2 K)], recomputed each step
    # from the current ice fraction, LWC, and layer height so densification
    # and wetting are reflected in the thermal inertia (previously this was
    # a single value frozen at the t=0 bulk density, ignoring densification).
    Cs_hist = np.full((3, Nt), np.nan)

    # --- History arrays ---
    T_hist     = np.full((3, Nt), np.nan)
    Tc_hist    = np.full(Nt, np.nan)
    W_hist     = np.full(Nt, np.nan)
    fsat_hist  = np.full(Nt, np.nan)
    melt_top   = np.zeros(Nt)
    melt_mid   = np.zeros(Nt)
    melt_bot   = np.zeros(Nt)

    qSW_hist           = np.full(Nt, np.nan)
    qLW_hist           = np.full(Nt, np.nan)
    qH_hist            = np.full(Nt, np.nan)
    qE_hist            = np.full(Nt, np.nan)
    qRAIN_hist         = np.full(Nt, np.nan)
    qins_hist          = np.full(Nt, np.nan)
    qNET_hist          = np.full(Nt, np.nan)
    qSWcov_hist        = np.full(Nt, np.nan)
    qSWsnow_hist       = np.full(Nt, np.nan)
    qSW_into_snow_hist = np.full(Nt, np.nan)
    qtop_hist          = np.full(Nt, np.nan)

    # Flux cap for numerical stability (no-insulation edge case)
    max_surface_flux = np.max(forc.G) * getattr(base.meta, "rA", 1.0) * 1.5

    # Store initial state for densification histories
    heights_hist[:, 0]  = heights
    ice_frac_hist[:, 0] = ice_fracs
    Cs_hist[:, 0]       = (ice_fracs * RHO_I * C_S + LWC_snow * RHO_W * C_W) * heights

    for k in range(Nt - 1):
        # -- Clip forcing to physical bounds --
        Ta   = forc.Ta[k]
        U    = max(0.1,  p.U10 * forc.U[k])
        RH   = min(max(forc.RH[k], 0.0), 1.0)
        G    = max(0.0,  forc.G[k])
        Pr   = max(0.0,  forc.Pr[k])
        Tg_k = forc.Tg[k] if (forc.Tg is not None) else gr.Tg

        W_hist[k] = W

        # -- Insulation properties --
        f = min(1.0, max(0.0, W / p.W_sat))
        fsat_hist[k] = f

        if USE_AGING_MODEL and p.Hi > 0:
            # aging + porosity model (only when insulation exists)
            k_eff, alpha_eff, tau_eff, zeta = update_insulation_properties(
                W, age_days, zeta, p.Hi, p
            )
            Rins = max(p.Hi / k_eff, 1e-4)
            alpha_cov = alpha_eff
            tau_cov   = tau_eff
        else:
            # simple moisture-dependent model
            # (also used for no-insulation case, regardless of USE_AGING_MODEL)
            k_eff = p.k_dry + (p.k_sat - p.k_dry) * (f ** p.n_k)
            Rins = max(p.Hi / k_eff, 1e-4) if p.Hi > 0 else 1e-4
            alpha_cov = p.alb_dry * (1 - f) + p.alb_wet * f
            tau_cov   = p.tau_dry * (1 - f) + p.tau_wet * f

        # -- Shortwave partitioning (same for both models) --
        qSW_cov       = (1 - alpha_cov) * (1 - tau_cov) * G
        qSW_snow      = (1 - alpha_cov) * tau_cov * G
        qSW_into_snow = p.a_snow * qSW_snow

        rA = getattr(base.meta, "rA", 1.0)

        # -- Solve SEB for cover temperature Tc --
        Tc = solve_cover_temperature(
            Ta, U, RH, Ta, Pr, T[0], Rins, f, qSW_cov, p, rA
        )

        qLW, qH, qE, qRAIN, qins = cover_fluxes(
            Tc, Ta, U, RH, Ta, Pr, T[0], Rins, f, qSW_cov, p
        )

        # Effective fluxes scaled by area ratio rA
        qSWcov_eff       = rA * qSW_cov
        qSWinto_eff      = rA * qSW_into_snow
        qLW_eff          = rA * qLW
        qH_eff           = rA * qH
        qE_eff           = rA * qE
        qRAIN_eff        = rA * qRAIN

        qtop = min(max(qins + qSWinto_eff, -max_surface_flux), max_surface_flux)

        # Record flux histories
        Tc_hist[k]              = Tc
        qSW_hist[k]             = qSWcov_eff + qSWinto_eff
        qLW_hist[k]             = qLW_eff
        qH_hist[k]              = qH_eff
        qE_hist[k]              = qE_eff
        qRAIN_hist[k]           = qRAIN_eff
        qins_hist[k]            = qins
        qtop_hist[k]            = qtop
        qNET_hist[k]            = qSWcov_eff + qLW_eff + qH_eff + qE_eff + qRAIN_eff - qins
        qSWcov_hist[k]          = qSWcov_eff
        qSWsnow_hist[k]         = rA * qSW_snow
        qSW_into_snow_hist[k]   = qSWinto_eff

        # -- Dynamic thermal resistances based on current layer heights --
        R12_dyn = (heights[0] + heights[1]) / (2.0 * sn.k)
        R23_dyn = (heights[1] + heights[2]) / (2.0 * sn.k)
        R3g_dyn = (heights[2] / (2.0 * sn.k)) + (gr.Hg / gr.kg) + (1.0 / gr.h_ground)
        qground = (Tg_k - T[2]) / R3g_dyn
        qground_hist[k] = qground

        # -- Dynamic per-layer volumetric heat capacity [J/(m^2 K)] (v7) --
        # Recomputed each step from current ice fraction, LWC, and layer
        # height so densification/melt are reflected in the thermal inertia,
        # rather than freezing Cs at the t=0 bulk density.
        Cs_layers = (ice_fracs * RHO_I * C_S + LWC_snow * RHO_W * C_W) * heights
        Cs_hist[:, k] = Cs_layers

        # -- RK4 temperature integration (Numba-JIT) --
        Tnew = rk4_snow(T, qtop, R12_dyn, R23_dyn, R3g_dyn, Tg_k, Cs_layers, dt)

        # -- Refreezing --
        for i in range(3):
            Tnew[i], LWC_snow[i], ice_fracs[i], _ = refreezing_layer(
                Tnew[i], LWC_snow[i], ice_fracs[i], sn.dz
            )

        # -- Melting --
        if Tnew[0] > TFREEZE:
            dE            = Cs_layers[0] * (Tnew[0] - TFREEZE)
            melt_top[k]   = dE / (RHO_I * LF)
            LWC_snow[0]  += melt_top[k] / heights[0]
            Tnew[0]       = TFREEZE

        if Tnew[1] > TFREEZE:
            dE            = Cs_layers[1] * (Tnew[1] - TFREEZE)
            melt_mid[k]   = dE / (RHO_I * LF)
            LWC_snow[1]  += melt_mid[k] / heights[1]
            Tnew[1]       = TFREEZE

        if Tnew[2] > TFREEZE:
            dE            = Cs_layers[2] * (Tnew[2] - TFREEZE)
            melt_bot[k]   = dE / (RHO_I * LF)
            LWC_snow[2]  += melt_bot[k] / heights[2]
            Tnew[2]       = TFREEZE

        # -- Boone overburden densification --
        heights, ice_fracs, LWC_snow = densification_boone(
            Tnew, LWC_snow, ice_fracs, heights, sn.dz, dt
        )

        # -- Percolation --
        LWC_snow, _ = percolate_water(LWC_snow, heights)

        # -- Update cover moisture --
        L_used = LV + LF if Tc < TFREEZE else LV
        E_rate = -qE / L_used
        eta_r  = max(0.0, 1.0 - f)
        m_in   = eta_r * RHO_W * Pr
        D      = p.KD * max(0.0, W - p.W_field)
        W      = min(max(W + (m_in - E_rate - D) * dt, 0.0), p.W_sat)

        # ======== Update insulation age ========
        # Update insulation age (only used when USE_AGING_MODEL is True)
        if USE_AGING_MODEL and p.Hi > 0:
            age_days += dt / 86400.0
        # ========================================================

        T = Tnew
        T_hist[:, k] = T
        heights_hist[:, k + 1]  = heights
        ice_frac_hist[:, k + 1] = ice_fracs
        Cs_hist[:, k + 1]       = (ice_fracs * RHO_I * C_S + LWC_snow * RHO_W * C_W) * heights

    # -- Forward-fill last timestep for all history arrays (loop skips Nt-1) --
    Tc_hist[-1] = Tc_hist[-2]
    #W_hist[-1] = W  # W is already the final value from the last iteration
    fsat_hist[-1] = fsat_hist[-2]
    qSW_hist[-1] = qSW_hist[-2]
    qLW_hist[-1] = qLW_hist[-2]
    qH_hist[-1] = qH_hist[-2]
    qE_hist[-1] = qE_hist[-2]
    qRAIN_hist[-1] = qRAIN_hist[-2]
    qins_hist[-1] = qins_hist[-2]
    qtop_hist[-1] = qtop_hist[-2]
    qNET_hist[-1] = qNET_hist[-2]
    qSWcov_hist[-1] = qSWcov_hist[-2]
    qSWsnow_hist[-1] = qSWsnow_hist[-2]
    qSW_into_snow_hist[-1] = qSW_into_snow_hist[-2]
    qground_hist[-1] = qground_hist[-2]
    T_hist[:, -1] = T_hist[:, -2]

    # -- Compile outputs --
    idx = slice(0, Nt - 1)

    out = SimpleNamespace(
        melt_total  = float(np.sum(melt_top[idx] + melt_mid[idx] + melt_bot[idx])),
        qins_mean   = float(np.nanmean(qins_hist[idx])),
        qE_meanAbs  = float(np.nanmean(np.abs(qE_hist[idx]))),
        f_mean      = float(np.nanmean(fsat_hist[idx])),
        T_hist      = T_hist, Tc_hist=Tc_hist, W_hist=W_hist, fsat_hist=fsat_hist,
        heights_hist  = heights_hist,   # layer thickness evolution [m]
        ice_frac_hist = ice_frac_hist,  # ice fraction evolution    [-]
        Cs_hist       = Cs_hist,        # per-layer dynamic Cs [J/(m^2 K)]
        qSW_hist    = qSW_hist, qLW_hist=qLW_hist, qH_hist=qH_hist,
        qE_hist     = qE_hist, qRAIN_hist=qRAIN_hist, qins_hist=qins_hist,
        qNET_hist   = qNET_hist, qSWcov_hist=qSWcov_hist, qSWsnow_hist=qSWsnow_hist,
        qSW_into_snow_hist=qSW_into_snow_hist, qtop_hist=qtop_hist,
        melt_top=melt_top, melt_mid=melt_mid, melt_bot=melt_bot,
        qground_hist = qground_hist,
    )
    out.W_hist[Nt - 1] = W

    # -- Optional volume tracking --
    if getattr(base.meta, "enable_volume", False):
        V0     = base.meta.V0
        rho_s  = base.meta.rho_s
        Aref   = base.meta.Ab

        cum_melt  = np.cumsum(melt_top[:Nt-1] + melt_mid[:Nt-1] + melt_bot[:Nt-1])
        cum_melt  = np.append(cum_melt, cum_melt[-1]) if len(cum_melt) else np.zeros(Nt)
        V_hist    = np.maximum(V0 - Aref * (RHO_I / rho_s) * cum_melt, 0.0)

        hits = np.where(V_hist <= 1e-6 * V0)[0]
        out.V_hist      = V_hist
        out.Vend        = float(V_hist[-1])
        out.Vfrac       = V_hist / V0
        out.meltout_day = float(forc.days[hits[0]]) if len(hits) else np.nan

    return out


# ==============================================================================
#  DIGITISED REFERENCE DATA  (Nordell & Sundin 1998, Fig. 4)
# ==============================================================================

# Each array is [day_of_season, snow_volume_m3].
# H00 = no insulation, H01 = 100 mm, H02 = 200 mm.

H02 = np.array([
    [0.1767183699576691,  30065.456703052547],
    [10.506568814751855,  29794.729799106208],
    [25.21389676735203,   29427.36431042174 ],
    [46.583585116858416,  29281.04932712782 ],
    [63.74628636834176,   29030.205606847394],
    [76.69732955167103,   28474.865130997572],
    [87.90454388607883,   28264.797717536763],
    [100.66796976914468,  27177.40849527433 ],
    [115.17678149117566,  25811.40221681718 ],
    [129.5151028034237,   24646.563260084673],
    [147.34886676373307,  23124.458209545613],
    [156.43546065864888,  22128.9103594543  ],
    [165.1779597539114,   21402.385140423674],
    [173.219813329518,    20680.65713396006 ],
    [180.21225561568437,  20032.780952541263],
    [185.28726466305895,  19798.03314284753 ],
])

H01 = np.array([
    [0.702202509699628,   30061.85879362729 ],
    [12.605391393632328,  29647.026149016576],
    [23.80793475790909,   29236.99071697288 ],
    [35.19031047223088,   29025.724000370315],
    [45.172952137284454,  28890.707715096076],
    [49.020274535217425,  28597.69902120032 ],
    [54.80371405246635,   28691.434029911048],
    [60.04609952953652,   28122.20688610408 ],
    [65.46676036660803,   27685.092451543947],
    [70.36193706393757,   27251.57592640908 ],
    [74.21081645191427,   27025.22323870762 ],
    [80.16474637894613,   26917.790925693706],
    [85.76056859593162,   26479.477187991823],
    [93.44587145153554,   25493.523763034536],
    [103.91507250523907,  23688.509413477645],
    [111.94758414058359,  22566.84536984826 ],
    [119.97386781575335,  21178.55730144169 ],
    [125.55723411238944,  20206.995514185448],
    [132.54189144833737,  19225.83930179517 ],
    [139.17934000454474,  18380.393708076994],
    [146.51432010032067,  17396.83888940321 ],
    [152.1070283372188,   16825.213139312742],
    [154.54371775557783,  16141.862833385236],
    [157.32917294372115,  15389.457914979932],
    [160.47896380208553,  15234.55844603978 ],
    [163.44580833031756,  14747.5782492699  ],
    [168.6881938073877,   14178.351105462934],
    [173.40353815467222,  13546.065864886928],
    [178.81641404152532,  12775.671399355328],
    [183.5379863489846,   12410.010183556507],
    [184.58739763842485,  12336.158358511693],
])

H00 = np.array([
    [0.350322759827975,   29997.601393716497],
    [2.802582078623786,   29980.81114973195 ],
    [4.729357257677645,   29967.618815172664],
    [5.764755636724768,   29293.862934379184],
    [8.032511635344516,   28878.335956370618],
    [10.297153653876908,  28329.496965973453],
    [12.741628022454318,  27979.426691017434],
    [14.318080441680205,  27968.632962741656],
    [15.195444331293817,  28029.29245322718 ],
    [16.066580260732717,  27823.32791893553 ],
    [17.10664960991087,   27349.540056724938],
    [18.15138992922007,   27075.720213097236],
    [20.250212508100557,  26928.016563007604],
    [21.820436967151725,  26650.59880995464 ],
    [24.095977915989874,  26568.351862917552],
    [26.72184162465599,   26483.706309596953],
    [37.58184717932318,   26409.349514808244],
    [38.80641984867739,   26334.298386621675],
    [39.49928041811495,   25996.22114308318 ],
    [41.05393497672932,   25052.243328087265],
    [41.57007717620918,   24648.709381496228],
    [42.09244733586378,   24511.799459682377],
    [42.96825423543372,   24505.802943973613],
    [46.29632045379947,   24483.016184280295],
    [47.51622215302267,   24207.99703751084 ],
    [47.86343093276328,   24072.28641883874 ],
    [48.90194329189775,   23531.84255043386 ],
    [49.25226605172573,   23529.44394415035 ],
    [49.77463621138034,   23392.5340223365  ],
    [50.65044311095026,   23386.537506627734],
    [51.70296838047787,   23445.997693971505],
    [53.27942079970374,   23435.203965695728],
    [53.97383835918498,   23163.78272835153 ],
    [58.988124794856034,  20329.45067708027 ],
    [62.24924044134356,   17440.455651032247],
    [64.13086290913068,   15494.239136838387],
    [65.52903996835524,   15351.332699315768],
    [65.8715777779648,    15015.654062060785],
    [66.55665339718396,   14344.296787550818],
    [67.41377641622974,   13538.428197510504],
    [68.10040902549255,   12933.726929194832],
    [68.78392765466802,   12195.713648490568],
    [69.64260766375747,   11456.501064644538],
    [70.50751563302165,   10983.912505575714],
    [71.71963238202646,   10375.613327834777],
    [74.15787879042914,    9758.919028101573],
    [75.90326462939429,    9480.301971906854],
    [77.29521372844411,    9070.771509607057],
    [78.16479266783932,    8798.150969121107],
    [79.2188749274106,     8924.267162659176],
    [80.09156784689317,    8784.958634561819],
    [82.00277312551025,    8105.206238059571],
    [83.38849426438533,    7429.051750982591],
    [85.2981425529587,     6682.643348286056],
    [87.20156288135736,    5669.610920812338],
    [89.10654019979971,    4723.234499532911],
    [91.01307450828571,    3843.514084447779],
    [92.22207727720314,    3101.9028943182566],
    [94.64942475530007,    2018.6165512249718],
    [96.21186426413283,    1407.9187672005355],
    [97.94635117279223,     662.7096676457513],
    [99.86222742154034,     182.9252897263941],
    [103.35767006960167,   -174.34080408015143],
    [108.09014130736665,    -73.40997651890939],
    [112.29245743525865,   -168.84925811528228],
    [184.64812025012836,    -64.25739991078444],
])
H00[:, 1] = np.maximum(H00[:, 1], 0.0)   # clamp negatives to zero


# ==============================================================================
#  VALIDATION CASE SETUP
# ==============================================================================

def configure_case(which_case, base, p0):
    """
    Apply case-specific geometry and parameter overrides.

    Parameters
    ----------
    which_case : str   'nordell1998' or 'skogsberg2005'.
    base       : SimpleNamespace  Modified in-place.
    p0         : SimpleNamespace  Modified in-place.

    Returns
    -------
    p0_CH_noIns   : float  CH for no-insulation run.
    p0_beta_noIns : float  beta_w for no-insulation run.
    """
    if which_case.lower() == "nordell1998":
        base.meta.V0      = 30_000.0
        base.meta.rho_s   = 650.0
        base.meta.Hs_eff  = 3.3
        base.meta.Ab      = base.meta.V0 / base.meta.Hs_eff
        R = math.sqrt(base.meta.Ab / math.pi)
        s = math.sqrt(R**2 + base.meta.Hs_eff**2)
        base.meta.rA      = s / R

        base.ground.Tg    = TFREEZE + 6.0   # 6 °C soil temperature

        base.snow.rho = base.meta.rho_s
        # NOTE: base.snow.Cs is kept for backward compatibility / reference
        # only. run_snow_model() no longer reads it directly -- Cs is now
        # computed dynamically each step from ice_fracs/heights/LWC, with the
        # initial ice fraction derived from base.snow.rho (set just above),
        # so this density override still propagates into the initial
        # thermal mass.
        base.snow.Cs  = base.snow.rho * base.snow.c * base.snow.dz

        p0.k_dry    = 0.2596482158;   p0.k_sat    = 0.2696602322
        p0.tau_dry  = 0.1109989073;   p0.tau_wet  = 0.03650875679
        p0.n_k      = 1.89630393;     p0.CV       = 0.6042931863
        p0.CE       = 0.001407639551; p0.f_shelter = 0.162172101
        p0.eps_c    = 0.8317670781;   p0.CH       = 0.002855113604
        p0.beta_w   = 4.713703342;    p0.a_snow   = 0.4387427801

        p0_CH_noIns   = 0.0027
        p0_beta_noIns = 0.0

    elif which_case.lower() == "skogsberg2005":
        base.meta.V0      = 27_000.0
        base.meta.rho_s   = 730.0
        base.meta.Hs_eff  = 3.3
        base.meta.Ab      = 140.0 * 60.0
        R = math.sqrt(base.meta.Ab / math.pi)
        s = math.sqrt(R**2 + base.meta.Hs_eff**2)
        base.meta.rA      = s / R

        base.ground.Tg    = TFREEZE + 6.0
        p0.Hi     = 0.20
        p0.k_dry  = 0.35;  p0.k_sat = 0.35

        base.snow.rho = base.meta.rho_s
        # NOTE: see comment in the nordell1998 branch above -- base.snow.Cs
        # is reference-only; the initial ice fraction (sn.rho/RHO_I) is what
        # actually drives the dynamic Cs used in run_snow_model().
        base.snow.Cs  = base.snow.rho * base.snow.c * base.snow.dz

        p0_CH_noIns   = getattr(p0, "CH",     1.8e-3)
        p0_beta_noIns = getattr(p0, "beta_w", 0.0)

    else:
        raise ValueError(f"Unknown case '{which_case}'. "
                         "Choose 'nordell1998' or 'skogsberg2005'.")

    return p0_CH_noIns, p0_beta_noIns


# ==============================================================================
#  PLOTTING
# ==============================================================================

def plot_volume_validation(t_model, model_outputs, paper_volumes,
                           Hi_list, t0_date, save_prefix="Nordell_validation_volume"):
    """
    Plot simulated vs. digitised Nordell & Sundin (1998) snow volumes.

    Parameters
    ----------
    t_model       : np.ndarray   Days-of-season for the model time grid.
    model_outputs : list         run_snow_model outputs for each Hi.
    paper_volumes : list[array]  Digitised paper volumes for each Hi.
    Hi_list       : list[float]  Insulation thicknesses [m].
    t0_date       : pd.Timestamp Start date of the season.
    save_prefix   : str          Base filename for saved figures.
    """
    Hi_labels = {0: "no insulation", 0.10: "100 mm", 0.20: "200 mm"}
    colors     = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")

    for i, Hi in enumerate(Hi_list):
        lbl  = Hi_labels.get(Hi, f"{Hi*1000:.0f} mm")
        col  = colors[i % len(colors)]
        V_m  = model_outputs[i].V_hist
        V_p  = paper_volumes[i]

        ax.plot(t_model, V_p, color=col, linewidth=2,
                label=f"Nordell & Sundin (1998) - {lbl}")
        ax.plot(t_model, V_m, color=col, linewidth=2, linestyle="--",
                label=f"Model - {lbl}")

    # Mark site transition (outdoor → indoor storage)
    d_transition    = pd.Timestamp(year=t0_date.year, month=5, day=1)
    x_transition    = (d_transition - t0_date).days
    ax.axvline(x_transition, linestyle="--", linewidth=1.5,
               color="0.4", label="Site transition")

    # Calendar x-axis
    xt        = np.arange(t_model.min(), t_model.max() + 1e-9, 15)
    xt_labels = [(t0_date + pd.Timedelta(days=float(x))).strftime("%d-%b") for x in xt]
    ax.set_xticks(xt)
    ax.set_xticklabels(xt_labels, rotation=45)
    ax.set_xlim([t_model.min(), t_model.max()])

    ax.set_xlabel("Day of season")
    ax.set_ylabel("Snow volume  V(t)  (m³)")
    ax.legend(loc="lower left", fontsize=9, frameon=False)
    ax.grid(True, alpha=0.4)
    ax.tick_params(direction="out")
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    fig.tight_layout()
    fig.savefig(f"{save_prefix}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{save_prefix}.pdf",           bbox_inches="tight")
    print(f"\n  Figure saved: {save_prefix}.png / .pdf")
    return fig


# ==============================================================================
#  MAIN
# ==============================================================================

def main():
    print("=" * 60)
    print("  Snow Storage RC Model  -  Nordell 1998 Validation")
    print("=" * 60)

    # ------------------------------------------------------------------
    #  1. Load primary forcing and build default parameters
    # ------------------------------------------------------------------
    DATA_FILE = "../DATA_2024_40cm.csv"

    try:
        forc, base, p0 = load_primary_forcing(DATA_FILE)
    except FileNotFoundError:
        print(f"\nERROR: '{DATA_FILE}' not found.")
        print("Please check the file paths and re-run.")
        return

    # ------------------------------------------------------------------
    #  2. Load reference-year forcing for the Nordell case
    # ------------------------------------------------------------------
    REF_FORCING = "Nordell_Fig4_Forcing_AllVars_refYear.csv"
    dt          = 3 * 3600   # 3-hour time step for validation run
    forc        = load_reference_forcing(REF_FORCING, dt)

    # ------------------------------------------------------------------
    #  3. Configure the Nordell 1998 validation case
    # ------------------------------------------------------------------
    which_case = "nordell1998"
    ensure_attr(base, "meta", SimpleNamespace())
    base.meta.enable_volume = True

    p0_CH_noIns, p0_beta_noIns = configure_case(which_case, base, p0)

    print(f"\nCase        : {which_case}")
    print(f"  V0        : {base.meta.V0:.0f} m³")
    print(f"  rho_s     : {base.meta.rho_s:.0f} kg/m³")
    print(f"  Hs_eff    : {base.meta.Hs_eff:.1f} m")
    print(f"  Time step : {dt/3600:.1f} h  ({forc.Nt} steps, "
          f"{forc.days[-1]:.1f} days)")

    # ------------------------------------------------------------------
    #  4. Run model for each insulation thickness
    # ------------------------------------------------------------------
    Hi_list = [0.0, 0.10, 0.20]
    outputs = []

    print("\n" + "-" * 60)
    print(f"  {'Hi [m]':>8}  {'V_end [m³]':>12}  "
          f"{'V_end / V0':>12}  {'Melt-out [day]':>16}")
    print("  " + "-" * 56)

    for Hi in Hi_list:
        p     = copy.deepcopy(p0)
        p.Hi  = Hi

        if p.Hi <= 0:
            p.CH     = p0_CH_noIns
            p.beta_w = p0_beta_noIns
        else:
            p.CH     = p0.CH
            p.beta_w = p0.beta_w

        out = run_snow_model(p, forc, base)
        outputs.append(out)

        melt_day = f"{out.meltout_day:.1f}" if np.isfinite(out.meltout_day) else "-"
        print(f"  {Hi:>8.2f}  {out.Vend:>12.0f}  "
              f"{out.Vend / base.meta.V0:>12.3f}  {melt_day:>16}")

    # ------------------------------------------------------------------
    #  5. Interpolate digitised paper curves onto model time grid
    # ------------------------------------------------------------------
    t_model = np.asarray(forc.days).ravel()
    V_paper = [
        np.maximum(interp_pchip(H00[:, 0], H00[:, 1], t_model), 0.0),
        np.maximum(interp_pchip(H01[:, 0], H01[:, 1], t_model), 0.0),
        np.maximum(interp_pchip(H02[:, 0], H02[:, 1], t_model), 0.0),
    ]

    # ------------------------------------------------------------------
    #  6. RMSE summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Model vs. Nordell & Sundin (1998)  -  RMSE")
    print("  " + "-" * 40)
    labels = ["No insulation", "100 mm insulation", "200 mm insulation"]
    for i, lbl in enumerate(labels):
        err = rmse(np.asarray(outputs[i].V_hist).ravel(), V_paper[i])
        print(f"  {lbl:<22} : {err:>8.1f} m³")
    print("=" * 60)
    print(f"Using insulation aging model: [ {USE_AGING_MODEL} ]")


    # ------------------------------------------------------------------
    #  7. Resolve calendar start date from reference forcing file
    # ------------------------------------------------------------------
    Tforc = clean_met(pd.read_csv(REF_FORCING))
    if "Date_refYear" in Tforc.columns:
        s0 = str(Tforc["Date_refYear"].iloc[0]).strip().replace("\ufeff", "")
        t0 = pd.to_datetime(s0[:9], format="%d-%b-%y", errors="coerce")
    elif "Time" in Tforc.columns:
        t0 = pd.to_datetime(
            str(Tforc["Time"].iloc[0]).strip().replace("\ufeff", ""), errors="coerce"
        )
    else:
        raise KeyError("No usable time column in reference forcing file.")
    if pd.isna(t0):
        raise ValueError("Could not parse start date from reference forcing file.")

    # ------------------------------------------------------------------
    #  8. Plot and save
    # ------------------------------------------------------------------
    fig = plot_volume_validation(t_model, outputs, V_paper, Hi_list, t0)

    # Save digitised curves for external use
    csv_path = "Nordell_digitized_snow_volume.csv"
    pd.DataFrame({
        "day": H00[:, 0],
        "V_no_insulation_m3":  H00[:, 1],
        "V_100mm_insulation_m3": interp_pchip(H01[:, 0], H01[:, 1], H00[:, 0]),
        "V_200mm_insulation_m3": interp_pchip(H02[:, 0], H02[:, 1], H00[:, 0]),
    }).to_csv(csv_path, index=False)
    print(f"  Digitised paper curves saved: {csv_path}")

    plt.show()


if __name__ == "__main__":
    main()