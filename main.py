################################################################################
#                                                                              #
#                        Snow Storage RC Thermal Model                         #
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
#  Simulates the thermal evolution and melt of an insulated seasonal snow      #
#  storage pile using a 3-layer RC (resistance-capacitance) network model.     #
#  A woodchip insulation cover sits above the snow; its surface temperature    #
#  Tc is solved at every time step from a full surface energy balance (SEB)    #
#  using bisection, rather than a fixed-resistance shortcut.  The snow pile    #
#  itself is divided into three vertical layers (surface, middle, bottom),     #
#  each tracking temperature, liquid water content, ice fraction, and layer    #
#  height (via Boone overburden densification).  Ground heat exchange is       #
#  handled at the base through a concrete pad with a Robin boundary            #
#  condition.  Meteorological forcing is read from an hourly CSV file and      #
#  interpolated onto a 10-minute simulation grid.                              #
#                                                                              #
#  Model characteristics:                                                      #
#    - Cover surface temperature solved via SEB bisection (qSW, qLW, qH,       #
#      qE, qRAIN, qins all balanced at each step)                              #
#    - 3-layer snowpack RC network integrated with 4th-order Runge-Kutta       #
#    - All hot-path snow physics (refreezing, percolation, densification,      #
#      RK4 integration) are JIT-compiled with Numba for performance            #
#    - Insulation conductivity and albedo evolve with moisture content and     #
#      material age (optional aging + porosity model)                          #
#    - Optional snow-volume tracking to estimate pile melt-out date            #
#                                                                              #
# ---------------------------------------------------------------------------- #
#                                                                              #
#  INPUT                                                                       #
#    Hourly meteorological CSV with columns for air temperature, wind          #
#    speed, relative humidity, global solar irradiance, precipitation, and     #
#    (optionally) soil temperature.  See README for the expected format.       #
#                                                                              #
#  OUTPUT                                                                      #
#    figures/01_temperatures.png          Snow layer temperature evolution     #
#    figures/02_meteorological_overview.png  Air/soil temp, precip, solar      #
#    figures/03_liquid_water_content.png  LWC per layer                        #
#    figures/04_melt_and_runoff.png       Cumulative melt / runoff             #
#    figures/05_heat_fluxes.png           Daily stacked surface heat fluxes    #
#    figures/06_ground_interface.png      Soil T vs bottom layer T vs q_ground #
#    figures/07_insulation_properties.png k_eff and alpha_eff evolution        #
#    figures/08_insulation_moisture.png   Cover moisture content               #
#    figures/09_rho_h_V.png               Layer density, height, and volume    #
#    figures/10_cover_temperature.png     Solved cover temperature Tc          #
#    figures/11_flux_components.png       Individual SEB terms over time       #
#    figures/12_energy_balance_layered.png  Layered energy-flow diagram        #
#    Console: energy balance table, mass balance, melt totals                  #
#                                                                              #
# ---------------------------------------------------------------------------- #
#                                                                              #
#  DEPENDENCIES                                                                #
#    numpy, pandas, matplotlib, numba                                          #
#                                                                              #
#  USAGE                                                                       #
#    Place the input CSV in the working directory (see load_primary_forcing    #
#    for the expected file path), then run:                                    #
#       python3 main.py                                                        #
#                                                                              #
################################################################################

import os
import math
import time as py_time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit


# ==============================================================================
#  PHYSICAL CONSTANTS  (module-level; shared by all routines)
# ==============================================================================

SIGMA    = 5.670374419e-8   # Stefan-Boltzmann constant          [W/m^2 K^4]
LF       = 3.34e5           # Latent heat of fusion              [J/kg]
LV       = 2.5e6            # Latent heat of vaporisation        [J/kg]
RHO_I    = 917.0            # Ice density                        [kg/m^3]
RHO_W    = 1000.0           # Liquid water density               [kg/m^3]
RHO_AIR  = 1.225            # Air density at sea level           [kg/m^3]
CP_AIR   = 1005.0           # Air specific heat                  [J/(kg K)]
C_W      = 4180.0           # Water specific heat                [J/(kg K)]
C_S      = 2100.0           # Ice/snow specific heat             [J/(kg K)]
P0       = 101325.0         # Reference pressure                 [Pa]
TFREEZE  = 273.15           # Freezing temperature               [K]
THETA_E  = 0.04             # Irreducible (field-capacity) LWC   [-]

# Toggle: True  = aging + porosity insulation model
#         False = simple moisture-only model
USE_AGING_MODEL = True


# ==============================================================================
#  HELPER UTILITIES
# ==============================================================================

def ensure_attr(obj, name, default):
    if not hasattr(obj, name):
        setattr(obj, name, default)


# ==============================================================================
#  NUMBA-JIT KERNELS
# ==============================================================================

@njit
def e_sat_scalar(T_K):
    """Magnus-type saturation vapour pressure [Pa].  Scalar only."""
    dT = T_K - TFREEZE
    if T_K >= TFREEZE:
        return 611.2 * math.exp(17.27 * dT / (dT + 237.3))
    else:
        return 611.2 * math.exp(22.46 * dT / (dT + 272.62))


@njit
def dTdt_snow(T, qtop, R12, R23, R3g, Tsoil_K, Cs):
    """ODE right-hand side for 3-layer snow temperatures [K/s].

    Cs is a 3-element array of per-layer volumetric heat capacities [J/(m^2 K)],
    computed each time step from the current ice fraction and layer height so
    that densification is accounted for in the thermal inertia.
    """
    T1, T2, T3 = T[0], T[1], T[2]
    q12 = (T1 - T2) / R12
    q23 = (T2 - T3) / R23
    q3g = (T3 - Tsoil_K) / R3g
    dT  = np.empty(3)
    dT[0] = (qtop - q12) / Cs[0]
    dT[1] = (q12  - q23) / Cs[1]
    dT[2] = (q23  - q3g) / Cs[2]
    return dT


@njit
def rk4_snow(T, qtop, R12, R23, R3g, Tsoil_K, Cs, dt):
    """Advance snow temperatures by one RK4 step.

    Cs is a 3-element per-layer heat capacity array [J/(m^2 K)].
    It is treated as frozen over the sub-step (operator-split approach),
    which is accurate because densification timescales >> dt.
    """
    k1 = dTdt_snow(T,                    qtop, R12, R23, R3g, Tsoil_K, Cs)
    k2 = dTdt_snow(T + (dt/2.0)*k1,     qtop, R12, R23, R3g, Tsoil_K, Cs)
    k3 = dTdt_snow(T + (dt/2.0)*k2,     qtop, R12, R23, R3g, Tsoil_K, Cs)
    k4 = dTdt_snow(T + dt*k3,            qtop, R12, R23, R3g, Tsoil_K, Cs)
    return T + (dt/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)


@njit
def refreezing_layer(T_layer, LWC_layer, ice_frac, dz_s):
    """Refreeze liquid water in one snow layer (Bartelt & Lehning 2002)."""
    if (T_layer >= TFREEZE) or (LWC_layer <= 0.0):
        return T_layer, LWC_layer, ice_frac, 0.0
    dT_max       = T_layer - TFREEZE
    denom        = RHO_W * (LF - dT_max * (C_S - C_W))
    dtheta_w_max = -(dT_max * (ice_frac*RHO_I*C_S + LWC_layer*RHO_W*C_W)) / denom
    dtheta_w     = min(LWC_layer, dtheta_w_max)
    dtheta_i     = (RHO_W / RHO_I) * dtheta_w
    dT           = (dtheta_w * RHO_W * LF) / (
                       ice_frac*RHO_I*C_S + LWC_layer*RHO_W*C_W)
    return (T_layer + dT,
            LWC_layer - dtheta_w,
            ice_frac  + dtheta_i,
            dtheta_w  * dz_s * RHO_W)


@njit
def percolate_water(LWC_array, heights):
    """Bucket-method liquid water percolation through layers."""
    n       = len(LWC_array)
    new_LWC = LWC_array.copy()
    for i in range(n - 1):
        if new_LWC[i] > THETA_E:
            excess         = new_LWC[i] - THETA_E
            new_LWC[i]     = THETA_E
            new_LWC[i+1]  += excess * heights[i] / heights[i+1]
    runoff = 0.0
    if new_LWC[n-1] > THETA_E:
        excess         = new_LWC[n-1] - THETA_E
        new_LWC[n-1]  = THETA_E
        runoff         = excess * heights[n-1] * RHO_W
    return new_LWC, runoff

@njit
def densification_boone(T_layers, LWC_layers, ice_fractions, heights, dz_initial, dt):
    """
    Overburden densification for 3-layer snow model (Essery et al. 2013 - Boone method).
    Numba-optimized and mass-conservative.

    Mutates heights, ice_fractions, and LWC_layers in place (no return value)
    to avoid three array allocations per time step.
    """
    c1 = 2.8e-6; c2 = 0.042; c3 = 0.046; c4 = 0.081; c5 = 0.018
    eta0 = 3.7e7; rho0 = 150.0
    dz_min = dz_initial * 0.1

    M_s = 0.0
    for i in range(3):
        rho_layer = ice_fractions[i] * RHO_I + LWC_layers[i] * RHO_W

        if rho_layer < 800.0 and heights[i] > 0.01:
            T_C  = T_layers[i] - TFREEZE
            eta  = eta0 * math.exp(c4 * (-T_C) + c5 * rho_layer)
            dRho = ((M_s * 9.81) / eta +
                    c1 * math.exp(-c2 * (-T_C) - c3 * max(0.0, rho_layer - rho0))) * dt
            dRho = min(dRho, 0.1)

            ice_fractions[i] *= (1.0 + dRho)
            LWC_layers[i]    *= (1.0 + dRho)
            h_new             = heights[i] * (1.0 - dRho)
            heights[i]        = h_new if h_new > dz_min else dz_min

        M_s += rho_layer * heights[i]


# ==============================================================================
#  SURFACE ENERGY BALANCE (SEB)
#
#  All hot-path SEB functions are JIT-compiled.  Quantities that depend only
#  on air temperature and are constant during the bisection search (sky
#  emissivity, air specific humidity, the sensible-heat aerodynamic
#  prefactor) are precomputed once in solve_cover_temperature and passed in
#  as scalars, so the inner bisection loop only re-evaluates Tc-dependent
#  terms.  solve_cover_temperature returns (Tc, qLW, qH, qE, qRAIN, qins)
#  directly, avoiding a second flux evaluation after convergence.
# ==============================================================================


@njit
def cover_fluxes_full_jit(Tc, Ta, U, RH, Pr, Ts1, Rins, f,
                           eps_c, CH, CV, CE, f_shelter, beta_w):
    """
    Full cover flux computation with all inputs explicit (no precomputation).
    Used for the final flux extraction after Tc is converged.
    Returns (qLW, qH, qE, qRAIN, qins).
    """
    e_a_Pa  = RH * e_sat_scalar(Ta)
    e_a_hPa = e_a_Pa / 100.0
    eps_sky  = min(max(1.24 * (e_a_hPa / Ta)**(1.0/7.0), 0.6), 1.0)
    qLW      = eps_sky * SIGMA * Ta**4 - eps_c * SIGMA * Tc**4

    qH = RHO_AIR * CP_AIR * CH * (1.0 + CV) * U * (Ta - Tc)

    e_c_Pa = e_sat_scalar(Tc)
    q_a    = 0.622 * e_a_Pa  / (P0 - 0.378 * e_a_Pa)
    q_star = 0.622 * e_c_Pa / (P0 - 0.378 * e_c_Pa)
    E0     = f_shelter * RHO_AIR * CE * U * (q_star - q_a)
    E      = E0 * math.exp(-beta_w * f)
    L      = LV + LF if Tc < TFREEZE else LV
    qE     = -L * E

    qRAIN  = RHO_W * C_W * Pr * (Ta - TFREEZE) if (Ta > TFREEZE and Pr > 0.0) else 0.0

    qins   = (Tc - Ts1) / Rins

    return qLW, qH, qE, qRAIN, qins


@njit
def _bisect_seb(lo, hi,
                Ta, Ts1, Rins, f, qSW_cov,
                eps_sky_Ta4, qH_coeff,
                q_a,            # air specific humidity (precomputed, constant)
                CE_U_shelter,   # f_shelter * RHO_AIR * CE * U  (precomputed)
                qRAIN,
                eps_c, beta_w, rA,
                tol=1e-6, max_iter=100):
    """
    Bisection solver for the SEB residual, fully JIT-compiled.
    Precomputed Ta-side quantities are passed as scalars.

    Residual: rA * (qSW_cov + qLW(Tc) + qH(Tc) + qE(Tc) + qRAIN) - qins(Tc) = 0

    rA is the slope/exposure area ratio scaling the incoming radiative and
    convective fluxes relative to a flat horizontal surface (rA = 1.0).
    q_a and CE_U_shelter are constant across the bisection (Ta-side only).
    The latent term applies exp(-beta_w*f) to the full (q_star - q_a).
    """
    def residual(Tc):
        # Longwave
        qLW  = eps_sky_Ta4 - eps_c * SIGMA * Tc**4
        # Sensible
        qH   = qH_coeff * (Ta - Tc)
        # Latent: exp factor applied to full (q_star - q_a)
        e_c_Pa = e_sat_scalar(Tc)
        q_star = 0.622 * e_c_Pa / (P0 - 0.378 * e_c_Pa)
        E      = CE_U_shelter * (q_star - q_a) * math.exp(-beta_w * f)
        L      = LV + LF if Tc < TFREEZE else LV
        qE     = -L * E
        # Conduction through insulation
        qins   = (Tc - Ts1) / Rins
        return rA * (qSW_cov + qLW + qH + qE + qRAIN) - qins

    flo = residual(lo)
    fhi = residual(hi)

    # Guard: if bracket is invalid just return midpoint
    if not (math.isfinite(flo) and math.isfinite(fhi)):
        return 0.5 * (lo + hi)
    if flo * fhi > 0.0:
        return 0.5 * (lo + hi)

    for _ in range(max_iter):
        mid  = 0.5 * (lo + hi)
        fmid = residual(mid)
        if not math.isfinite(fmid):
            break
        if abs(fmid) < tol or abs(hi - lo) < tol:
            return mid
        if flo * fmid < 0.0:
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid
    return 0.5 * (lo + hi)


def solve_cover_temperature(Ta, U, RH, Pr, Ts1, Rins, f, qSW_cov, p, rA=1.0):
    """
    Find cover temperature Tc by solving SEB residual via bisection.

    Precomputes all Ta-dependent quantities once before the bracket search and
    bisection, so the inner residual only evaluates Tc-dependent terms.

    Returns (Tc, qLW, qH, qE, qRAIN, qins) - no second flux evaluation needed.
    """
    # --- Precompute Ta-side quantities (constant during bisection) ---
    e_a_Pa      = RH * e_sat_scalar(Ta)
    e_a_hPa     = e_a_Pa / 100.0
    eps_sky     = min(max(1.24 * (e_a_hPa / Ta)**(1.0/7.0), 0.6), 1.0)
    eps_sky_Ta4 = eps_sky * SIGMA * Ta**4

    qH_coeff    = RHO_AIR * CP_AIR * p.CH * (1.0 + p.CV) * U   # * (Ta - Tc)

    q_a         = 0.622 * e_a_Pa / (P0 - 0.378 * e_a_Pa)
    CE_U_shelter = p.f_shelter * RHO_AIR * p.CE * U   # * (q_star - q_a) * exp(...)

    qRAIN       = (RHO_W * C_W * Pr * (Ta - TFREEZE)
                  if (Ta > TFREEZE and Pr > 0.0) else 0.0)

    # --- Bracket search ---
    Tc_min = TFREEZE + p.solve.Tc_min_C
    Tc_max = TFREEZE + p.solve.Tc_max_C

    lo = max(Ta - p.solve.bracket_dT_lo, Tc_min)
    hi = min(Ta + p.solve.bracket_dT_hi, Tc_max)

    def _res(Tc):
        qLW_  = eps_sky_Ta4 - p.eps_c * SIGMA * Tc**4
        qH_   = qH_coeff * (Ta - Tc)
        e_c   = e_sat_scalar(Tc)
        q_s   = 0.622 * e_c / (P0 - 0.378 * e_c)
        E_    = CE_U_shelter * (q_s - q_a) * math.exp(-p.beta_w * f)
        L_    = LV + LF if Tc < TFREEZE else LV
        qE_   = -L_ * E_
        qi_   = (Tc - Ts1) / Rins
        return rA * (qSW_cov + qLW_ + qH_ + qE_ + qRAIN) - qi_

    Flo, Fhi = _res(lo), _res(hi)

    for ntry in range(1, p.solve.max_expand + 1):
        if (math.isfinite(Flo) and math.isfinite(Fhi)
                and Flo * Fhi < 0.0):
            break
        factor = p.solve.expand_factor**ntry
        new_lo = max(Ta - p.solve.bracket_dT_lo * factor, Tc_min)
        new_hi = min(Ta + p.solve.bracket_dT_hi * factor, Tc_max)
        # Only re-evaluate changed endpoints (cache unchanged side)
        if new_lo != lo:
            lo, Flo = new_lo, _res(new_lo)
        if new_hi != hi:
            hi, Fhi = new_hi, _res(new_hi)

    if math.isfinite(Flo) and math.isfinite(Fhi) and Flo * Fhi < 0.0:
        Tc = _bisect_seb(lo, hi,
                         Ta, Ts1, Rins, f, qSW_cov,
                         eps_sky_Ta4, qH_coeff,
                         q_a, CE_U_shelter, qRAIN,
                         p.eps_c, p.beta_w, rA)
    else:
        # Fallback: coarse grid scan
        Tc_grid = np.linspace(lo, hi, 25)
        Fg      = np.array([_res(x) for x in Tc_grid])
        Tc      = Tc_grid[np.argmin(np.abs(Fg))]

    # --- Single final flux evaluation at converged Tc ---
    qLW, qH, qE, qRAIN_out, qins = cover_fluxes_full_jit(
        Tc, Ta, U, RH, Pr, Ts1, Rins, f,
        p.eps_c, p.CH, p.CV, p.CE, p.f_shelter, p.beta_w)

    return Tc, qLW, qH, qE, qRAIN_out, qins


# ==============================================================================
#  ADVANCED INSULATION MODEL  (aging + porosity)
# ==============================================================================

def update_insulation_properties(W, age_days, zeta, Hi, p):
    """
    Update effective insulation properties (conductivity, albedo,
    transmissivity) accounting for moisture, aging and porosity.
    """
    f      = min(1.0, max(0.0, W / p.W_sat))
    age_yr = age_days / 365.0

    # Conductivity
    k_moist        = p.k_dry + (p.k_sat - p.k_dry) * (f**p.n_k)
    porosity_factor = zeta * np.exp(-p.gamma_H * Hi) * np.exp(-p.gamma_W * f)
    k_age_factor    = 1.0 + p.delta_k_age * (1.0 - np.exp(-age_yr / p.tau_k_years))
    k_eff           = k_moist * (1.0 + porosity_factor) * k_age_factor

    # Albedo (reflected fraction; note: absorptance = 1 - alb_eff)
    alb_moist = p.alb_dry + (p.alb_wet  - p.alb_dry) * (f**p.beta_w)
    alb_eff   = min(max(alb_moist + p.delta_alb_age *
                          (1.0 - np.exp(-age_yr / p.tau_alb_years)), 0.0), 1.0)
    tau_moist   = p.tau_dry + (p.tau_wet - p.tau_dry)  * (f**p.beta_w)
    tau_eff     = min(max(tau_moist, 0.0), 1.0)

    # Updated porosity (compaction / settlement)
    new_zeta = p.zeta0 * np.exp(-p.gamma_H * Hi) * np.exp(-p.gamma_W * f)

    return k_eff, alb_eff, tau_eff, new_zeta


# ==============================================================================
#  CSV LOADING & FORCING CONSTRUCTION
# ==============================================================================

COL_ALIASES = {
    "T"  : ["Temp_C",            "Ta_C"],
    "U"  : ["Air_Vel_m/s_10m",   "Air_Vel_m_s",  "U_ms"],
    "RH" : ["RH_%",              "RH_",           "RH_pct"],
    "G"  : ["Glo_Sol_Ir_W/m2",   "Glo_Sol_Ir_W_m2", "G_W_m2"],
    "Pr" : ["Prec_m/h",          "Prec_m_h",      "Pr_mm_day"],
    "Tg" : ["Soil_Temp_320cm",   "Soil_Temp_C"],
}


def detect_columns(cols):
    return {k: next((c for c in cands if c in cols), None)
            for k, cands in COL_ALIASES.items()}


def clean_met(df):
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    return df


def load_primary_forcing(data_file):
    """
    Load hourly met CSV, build forcing arrays on a 10-min grid, and return
    base geometry + default tuning parameters.
    """
    met = clean_met(pd.read_csv(data_file))
    met["Time_dt"] = pd.to_datetime(met["Time"], format="%Y-%m-%dT%H:%M", errors="coerce")
    if met["Time_dt"].isna().any():
        met["Time_dt"] = pd.to_datetime(met["Time"], errors="coerce")
    if met["Time_dt"].isna().any():
        raise ValueError(f"Could not parse all Time values in {data_file}.")

    t0          = met["Time_dt"].iloc[0]
    met["tsec"] = (met["Time_dt"] - t0).dt.total_seconds()

    col_map = detect_columns(met.columns)
    missing = [k for k in ("T", "U", "RH", "G", "Pr") if not col_map[k]]
    if missing:
        raise KeyError(f"Missing columns in {data_file}: {missing}")

    print(f"\nLoading  : {data_file}")
    print(f"  T='{col_map['T']}'  U='{col_map['U']}'  "
          f"RH='{col_map['RH']}'  G='{col_map['G']}'  Pr='{col_map['Pr']}'")
    if col_map["Tg"]:
        print(f"  Soil T  : '{col_map['Tg']}'  (dynamic from CSV - Robin BC active)")
    else:
        print(f"  Soil T  : column not found - static fallback Tg = 2.0 °C")

    dt    = 600.0   # 10-min simulation step [s]
    ts    = met["tsec"].to_numpy(dtype=float)
    t_out = np.arange(0.0, ts[-1] + dt, dt, dtype=float)

    def interp1d(col):
        return np.interp(t_out, ts, met[col].to_numpy(dtype=float))

    forc      = SimpleNamespace()
    forc.dt   = dt
    forc.t    = t_out
    forc.Nt   = len(t_out)
    forc.days = t_out / 86400.0

    forc.Ta = interp1d(col_map["T"]) + TFREEZE
    forc.U  = np.clip(interp1d(col_map["U"]), 0.1, None)

    rh_div  = 100.0 if col_map["RH"] in ("RH_%", "RH_", "RH_pct") else 1.0
    forc.RH = np.clip(interp1d(col_map["RH"]) / rh_div, 0.0, 1.0)

    forc.G  = np.clip(interp1d(col_map["G"]), 0.0, None)

    Pr_raw  = met[col_map["Pr"]].to_numpy(dtype=float)
    Pr_mps  = (Pr_raw * 1e-3 / 86400.0 if col_map["Pr"] == "Pr_mm_day"
               else Pr_raw / 3600.0)
    forc.Pr = np.clip(np.interp(t_out, ts, Pr_mps), 0.0, None)

    if col_map["Tg"]:
        forc.Tg = interp1d(col_map["Tg"]) + TFREEZE
    else:
        forc.Tg = None

    # --- Keep hourly arrays for energy-balance diagnostics ---
    forc.Ta_h  = met[col_map["T"]].to_numpy(dtype=float)    # [°C]
    forc.U_h   = met[col_map["U"]].to_numpy(dtype=float)    # [m/s]
    forc.G_h   = met[col_map["G"]].to_numpy(dtype=float)    # [W/m^2]
    forc.Pr_h  = (met[col_map["Pr"]].to_numpy(dtype=float)  
                  / 3600.0)                                 # [m/s]
    forc.RH_h  = met[col_map["RH"]].to_numpy(dtype=float)   # [%]
    forc.t0_dt = t0                                             

    # ------------------------------------------------------------------
    #  Physical geometry  (base)
    # ------------------------------------------------------------------
    base      = SimpleNamespace()
    Hs = 4.5;  Ns = 3;  dz = Hs / Ns
    rho_s = 560.0

    base.snow = SimpleNamespace(
        Hs=Hs, Ns=Ns, dz=dz, rho=rho_s, c=C_S, k=0.45,
        Cs = rho_s * C_S * dz,
        R12 = dz / 0.45,
        R23 = dz / 0.45,
        T0  = np.array([TFREEZE - 2.0, TFREEZE - 4.0, TFREEZE - 6.0]),
    )

    # --- Concrete Pad Properties ---
    Hg_concrete = 0.30  # Concrete pad thickness [m]
    kg_concrete = 1.5   # Thermal conductivity of concrete [W/(m K)]
    h_ground    = 2.5   # Ground interface heat transfer coefficient
    base.ground = SimpleNamespace(
        Hg=Hg_concrete, kg=kg_concrete, h_ground=h_ground,
        # Robin BC: conduction through bottom snow layer
        #         + conduction through ground insulation layer
        #         + contact interface resistance  1/h_ground
        R3g = dz / 0.45 + Hg_concrete / kg_concrete + 1.0 / h_ground,
        Tg  = TFREEZE + 2.0,   # static fallback [K]; overridden if CSV has Tg
    )

    # ------------------------------------------------------------------
    #  Volume tracking parameters - adjust these for your pile geometry
    # ------------------------------------------------------------------
    V0   = 18_000.0    # Initial snow volume             [m^3]
    Aref = 4_000.0     # Reference (base) area of pile   [m^2]

    base.meta = SimpleNamespace(
        enable_volume = True,
        V0    = V0,
        rho_s = rho_s,   # bulk density used for volume -> mass conversion
        Hs_eff= Hs,
        Aref  = Aref,
        rA    = 1.0,     # area ratio (flat surface = 1.0)
    )

    # ------------------------------------------------------------------
    #  Default tuning parameters  (p0)
    # ------------------------------------------------------------------
    p0 = SimpleNamespace(
        # Surface exchange
        CH=1.8e-3, CE=1.0e-3, CV=2.0,
        eps_c=0.95, f_shelter=0.4,
        # Insulation geometry
        Hi=0.20,
        # Insulation conductivity
        k_dry=0.07, k_sat=0.12, n_k=1.5,
        # Insulation moisture
        W_sat=100.0, W_field=40.0, KD=5e-6,
        # Optical
        alb_dry=0.65, alb_wet=0.50,
        tau_dry=0.25, tau_wet=0.10,
        a_snow=0.55,
        # Wind & evaporation
        beta_w=3.0, U10=1.0,
        # aging + porosity (can be toggled on/off)
        delta_k_age=0.5,    tau_k_years=2.0,
        delta_alb_age=0.05, tau_alb_years=2.0,
        zeta0=0.25, gamma_H=0.5, gamma_W=2.0,
    )

    p0.solve = SimpleNamespace(
        Tc_min_C=-50, Tc_max_C=50,
        bracket_dT_lo=40, bracket_dT_hi=20,
        max_expand=6, expand_factor=1.6,
    )

    return forc, base, p0


# ==============================================================================
#  CORE SIMULATION
# ==============================================================================

def run_snow_model(p, forc, base):
    """
    Forward-integrate the 3-layer RC snow model using the SEB cover-flux
    solver, with dynamic Boone overburden densification tracking.

    Loop-invariant constants (the constant tail of R3g, the snow thermal
    resistance reciprocal, and several scalar tuning parameters) are
    hoisted out of the time loop before integration begins.
    """
    dt, Nt  = forc.dt, forc.Nt
    sn, gr  = base.snow, base.ground

    # --- Hoist loop-invariant constants ---
    k_s_inv   = 1.0 / (2.0 * sn.k)          # reciprocal for resistance calc
    R3g_const = gr.Hg / gr.kg + 1.0 / gr.h_ground   # constant tail of R3g
    a_snow    = p.a_snow
    W_sat     = p.W_sat
    KD        = p.KD
    W_field   = p.W_field
    max_surf_flux = max(float(np.nanmax(forc.G)) * 1.5, 500.0)

    # Soil temperature: resolved once as array or scalar
    Tg_arr    = forc.Tg          # None or ndarray
    Tg_static = gr.Tg

    # --- Initial state ---
    T        = np.array(sn.T0, dtype=float).copy()
    W        = 5.0                              # cover moisture [% of sat]
    LWC_snow = np.zeros(3, dtype=float)
    ice_frac = np.full(3, 0.4, dtype=float)
    heights  = np.full(3, sn.dz, dtype=float)
    age_days = 0.0
    zeta     = p.zeta0

    # --- History arrays ---
    T_hist        = np.full((3, Nt), np.nan)
    LWC_hist      = np.full((3, Nt), np.nan)
    heights_hist  = np.full((3, Nt), np.nan)
    ice_frac_hist = np.full((3, Nt), np.nan)
    Tc_hist       = np.full(Nt, np.nan)
    W_hist        = np.full(Nt, np.nan)
    k_eff_hist    = np.full(Nt, np.nan)
    alb_hist      = np.full(Nt, np.nan)
    fsat_hist     = np.full(Nt, np.nan)

    qSW_hist     = np.full(Nt, np.nan)
    qLW_hist     = np.full(Nt, np.nan)
    qH_hist      = np.full(Nt, np.nan)
    qE_hist      = np.full(Nt, np.nan)
    qRAIN_hist   = np.full(Nt, np.nan)
    qins_hist    = np.full(Nt, np.nan)
    qtop_hist    = np.full(Nt, np.nan)
    qground_hist = np.full(Nt, np.nan)

    Ta_hist    = forc.Ta.copy()
    Tsoil_hist = np.full(Nt, np.nan)
    G_hist     = forc.G.copy()
    Pr_hist    = forc.Pr.copy()

    melt_top = np.zeros(Nt)
    melt_mid = np.zeros(Nt)
    melt_bot = np.zeros(Nt)
    runoff_h = np.zeros(Nt)
    refrozen = np.zeros(Nt)
    Cs_hist  = np.full((3, Nt), np.nan)   # per-layer dynamic heat capacity [J/(m^2 K)]

    # Store initial state
    T_hist[:, 0]        = T
    LWC_hist[:, 0]      = LWC_snow
    heights_hist[:, 0]  = heights
    ice_frac_hist[:, 0] = ice_frac
    W_hist[0]           = W
    # Initial per-layer heat capacity
    Cs_hist[:, 0]       = ice_frac * RHO_I * C_S * heights

    prog = {int(Nt * q): f"{int(q*100)}%" for q in (0.25, 0.5, 0.75)}

    for k in range(Nt - 1):
        if k in prog:
            print(f"  Progress: {prog[k]}")
        # -- Forcing at step k --
        Ta   = forc.Ta[k]
        U    = forc.U[k]            # already clipped ≥ 0.1 at load time
        RH   = forc.RH[k]
        G    = forc.G[k]
        Pr   = forc.Pr[k]
        Tg_k = Tg_arr[k] if (Tg_arr is not None) else Tg_static

        Tsoil_hist[k] = Tg_k

        # -- Insulation properties --
        f = min(1.0, max(0.0, W / W_sat))
        fsat_hist[k] = f

        if USE_AGING_MODEL and p.Hi > 0:
            k_eff, alb_eff, tau_eff, zeta = update_insulation_properties(
                W, age_days, zeta, p.Hi, p)
        else:
            k_eff     = p.k_dry + (p.k_sat - p.k_dry) * (f**p.n_k)
            alb_eff = p.alb_dry * (1 - f) + p.alb_wet * f
            tau_eff   = p.tau_dry * (1 - f) + p.tau_wet * f

        k_eff_hist[k] = k_eff
        alb_hist[k] = alb_eff

        Rins     = max(p.Hi / k_eff, 1e-4) if p.Hi > 0 else 1e-4
        qSW_cov  = (1 - alb_eff) * (1 - tau_eff) * G
        qSW_into = a_snow * (1 - alb_eff) * tau_eff * G

        # -- Area ratio (slope/exposure correction; 1.0 for a flat surface) --
        rA = getattr(base.meta, "rA", 1.0)

        # -- Solve cover SEB for Tc; fluxes returned directly, no re-evaluation --
        Tc, qLW, qH, qE, qRAIN, qins = solve_cover_temperature(
            Ta, U, RH, Pr, T[0], Rins, f, qSW_cov, p, rA)
        Tc_hist[k] = Tc

        # -- Conductive flux into the snowpack (rA-scaled solar included) --
        qtop = min(max(qins + rA * qSW_into, -max_surf_flux), max_surf_flux)

        # -- Record rA-scaled flux histories --
        qSW_hist[k]   = rA * qSW_cov + rA * qSW_into
        qLW_hist[k]   = rA * qLW
        qH_hist[k]    = rA * qH
        qE_hist[k]    = rA * qE
        qRAIN_hist[k] = rA * qRAIN
        qins_hist[k]  = qins
        qtop_hist[k]  = qtop

        # -- Dynamic Thermal Resistances (hoisted constant R3g tail) --
        R12_dyn = (heights[0] + heights[1]) * k_s_inv
        R23_dyn = (heights[1] + heights[2]) * k_s_inv
        R3g_dyn =  heights[2]               * k_s_inv + R3g_const

        qground_hist[k] = (Tg_k - T[2]) / R3g_dyn

        # -- Dynamic per-layer volumetric heat capacity [J/(m^2 K)] --
        # Ice contribution dominates; liquid water contribution included for
        # completeness (C_W ≈ 2x C_S, so LWC matters at field-capacity pore fill).
        Cs_layers = (ice_frac * RHO_I * C_S + LWC_snow * RHO_W * C_W) * heights
        Cs_hist[:, k] = Cs_layers

        # -- RK4 temperature integration --
        Tnew = rk4_snow(T, qtop, R12_dyn, R23_dyn, R3g_dyn, Tg_k, Cs_layers, dt)

        # -- Refreezing --
        tot_refrozen = 0.0
        for i in range(3):
            Tnew[i], LWC_snow[i], ice_frac[i], rf = refreezing_layer(
                Tnew[i], LWC_snow[i], ice_frac[i], heights[i])
            tot_refrozen += rf
        refrozen[k] = tot_refrozen

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

        # -- Boone Densification (in-place; no return value) --
        densification_boone(Tnew, LWC_snow, ice_frac, heights, sn.dz, dt)

        # -- Water Percolation --
        LWC_snow, runoff = percolate_water(LWC_snow, heights)
        runoff_h[k] = runoff

        # -- Update cover moisture --
        L_evap = LV + LF if Tc < TFREEZE else LV
        E_rate = -qE / L_evap
        eta_r  = max(0.0, 1.0 - f)
        m_in   = eta_r * RHO_W * Pr
        D      = KD * max(0.0, W - W_field)
        W      = min(max(W + (m_in - E_rate - D) * dt, 0.0), W_sat)

        if USE_AGING_MODEL and p.Hi > 0:
            age_days += dt / 86400.0

        # Advance state
        T = Tnew
        T_hist[:, k+1]        = T
        LWC_hist[:, k+1]      = LWC_snow
        heights_hist[:, k+1]  = heights
        ice_frac_hist[:, k+1] = ice_frac
        W_hist[k+1]           = W
        Cs_hist[:, k+1]       = (ice_frac * RHO_I * C_S + LWC_snow * RHO_W * C_W) * heights

    # Fill last step records
    Tsoil_hist[-1]  = forc.Tg[-1] if forc.Tg is not None else gr.Tg
    Tc_hist[-1]     = Tc_hist[-2]
    k_eff_hist[-1]  = k_eff_hist[-2]
    alb_hist[-1]  = alb_hist[-2]
    fsat_hist[-1]   = fsat_hist[-2]
    qSW_hist[-1]    = qSW_hist[-2]
    qLW_hist[-1]    = qLW_hist[-2]
    qH_hist[-1]     = qH_hist[-2]
    qE_hist[-1]     = qE_hist[-2]
    qRAIN_hist[-1]  = qRAIN_hist[-2]
    qins_hist[-1]   = qins_hist[-2]
    qtop_hist[-1]   = qtop_hist[-2]
    qground_hist[-1]= qground_hist[-2]

    melt_rate = (melt_top + melt_mid + melt_bot) / dt

    # -- Volume tracking --
    V_hist = None
    meltout_day = np.nan
    if getattr(base.meta, "enable_volume", False):
        V0   = base.meta.V0
        Aref = base.meta.Aref
        rho_s_vol = base.meta.rho_s
        cum_melt = np.cumsum(melt_top[:Nt-1] + melt_mid[:Nt-1] + melt_bot[:Nt-1])
        cum_melt = np.append(cum_melt, cum_melt[-1])
        V_hist   = np.maximum(V0 - Aref * (RHO_I / rho_s_vol) * cum_melt, 0.0)
        hits = np.where(V_hist <= 1e-6 * V0)[0]
        meltout_day = float(forc.days[hits[0]]) if len(hits) else np.nan

    out = SimpleNamespace(
        T_hist       = T_hist,        
        LWC_hist     = LWC_hist,      
        heights_hist = heights_hist,   
        ice_frac_hist= ice_frac_hist, 
        Cs_hist      = Cs_hist,        # per-layer dynamic Cs [J/(m^2 K)] at each step
        Tc_hist      = Tc_hist,
        Ta_hist      = Ta_hist,
        Tsoil_hist   = Tsoil_hist,
        G_hist       = G_hist,
        Pr_hist      = Pr_hist,
        W_hist       = W_hist,
        k_eff_hist   = k_eff_hist,
        alb_hist     = alb_hist,
        fsat_hist    = fsat_hist,
        qSW_hist     = qSW_hist,
        qLW_hist     = qLW_hist,
        qH_hist      = qH_hist,
        qE_hist      = qE_hist,
        qRAIN_hist   = qRAIN_hist,
        qins_hist    = qins_hist,
        qtop_hist    = qtop_hist,
        qground_hist = qground_hist,
        melt_top     = melt_top,
        melt_mid     = melt_mid,
        melt_bot     = melt_bot,
        melt_rate    = melt_rate,
        runoff_h     = runoff_h,
        refrozen     = refrozen,
        melt_total   = float(np.nansum(melt_top + melt_mid + melt_bot)),
        V_hist      = V_hist,
        meltout_day = meltout_day,
    )
    return out


# ==============================================================================
#  CONSOLE DIAGNOSTICS
# ==============================================================================

def print_energy_balance(out, forc, base):
    """
    Print energy-balance and mass-balance summary to stdout.

    Two budgets are reported:

    1. COVER SURFACE FLUXES - individual terms at the cover top and pile
       bottom, integrated over the simulation period.  These are informational
       only; they cannot form a closed balance against snowpack-internal
       changes because:
         (a) qSW_hist = qSW_cov + qSW_into, so solar absorbed by the cover
             and solar transmitted into the snow are both included here, while
             only qSW_into enters the snowpack via qtop.
         (b) E_dT now uses a per-layer dynamic Cs integrated step-by-step,
             accounting for densification-driven changes in layer mass.
       These are structural features of the quasi-static cover SEB formulation,
       not physics errors.

    2. SNOWPACK BUDGET - uses qtop (the conductive flux actually delivered to
       the top snow layer) and qground as inputs.  This budget closes to within
       numerical integration error (~1-2 MJ/m^2) and is the meaningful check.
    """
    t  = forc.t
    sn = base.snow

    # --- Flux integrals [J/m^2] ---
    E_solar  = float(np.trapezoid(np.nan_to_num(out.qSW_hist),    t))
    E_LW     = float(np.trapezoid(np.nan_to_num(out.qLW_hist),    t))
    E_H      = float(np.trapezoid(np.nan_to_num(out.qH_hist),     t))
    E_evap   = float(np.trapezoid(np.nan_to_num(out.qE_hist),     t))
    E_rain   = float(np.trapezoid(np.nan_to_num(out.qRAIN_hist),  t))
    E_ground = float(np.trapezoid(np.nan_to_num(out.qground_hist),t))
    E_qtop   = float(np.trapezoid(np.nan_to_num(out.qtop_hist),   t))

    # --- Snowpack internal energy change [J/m^2] ---
    E_melt   = out.melt_total * RHO_I * LF
    E_refroz = float(np.nansum(out.refrozen)) * LF

    # Sensible storage: integrate Cs(t) * dT step by step (midpoint rule).
    # This is exact given the operator-split structure: within each RK4 step
    # Cs is frozen, so the energy stored in step k is Cs_k * (T_k+1 - T_k). 
    T_arr  = out.T_hist                  # shape (3, Nt)
    Cs_arr = out.Cs_hist                 # shape (3, Nt)
    dT_arr = np.diff(T_arr, axis=1)      # shape (3, Nt-1)
    # Use Cs at the start of each step (consistent with how RK4 was driven)
    E_dT = float(np.sum(Cs_arr[:, :-1] * dT_arr))
    E_internal = E_dT + E_melt - E_refroz

    n_days = t[-1] / 86400.0

    print("\n" + "=" * 60)
    print("Energy Balance Diagnostics")
    print("=" * 60)

    print(f"\n1. COVER SURFACE FLUXES [MJ/m^2]  (informational):")
    print(f"  Solar absorbed at cover  : {E_solar/1e6:>10.3f}")
    print(f"  Sensible heat (qH)       : {E_H/1e6:>10.3f}")
    print(f"  Rain heat input          : {E_rain/1e6:>10.3f}")
    print(f"  Ground heat input        : {E_ground/1e6:>10.3f}")
    print(f"  Net longwave (qLW)       : {E_LW/1e6:>10.3f}")
    print(f"  Latent heat (qE)         : {E_evap/1e6:>10.3f}")
    print(f"  Conducted into snow(qtop): {E_qtop/1e6:>10.3f}")

    print(f"\n2. SNOWPACK ENERGY BUDGET [MJ/m^2]  (primary closure check):")
    print(f"  Heat input via qtop      : {E_qtop/1e6:>10.3f}")
    print(f"  Heat input via ground    : {E_ground/1e6:>10.3f}")
    print(f"  " + "-" * 36)
    print(f"  Total heat entering snow : {(E_qtop + E_ground)/1e6:>10.3f}")
    print(f"  Consumed by melt         : {E_melt/1e6:>10.3f}")
    print(f"  Released by refreezing   : {-E_refroz/1e6:>10.3f}")
    print(f"  Sensible storage (dT)    : {E_dT/1e6:>10.3f}")
    print(f"  " + "-" * 36)
    print(f"  Total internal change    : {E_internal/1e6:>10.3f}")
    print(f"  Residual (input-change)  : {(E_qtop + E_ground - E_internal)/1e6:>10.3f}")

    print(f"\n3. MASS BALANCE:")
    print(f"  Total melt               : {out.melt_total:>10.3f} m w.e.")
    print(f"  Total runoff             : {float(np.nansum(out.runoff_h)):>10.3f} kg/m^2")
    if n_days > 0:
        print(f"  Avg melt rate            : {out.melt_total/n_days*1000:>10.3f} mm/day")


# ==============================================================================
#  12-FIGURE DIAGNOSTIC PLOTS  (matches mainPlot.py figure layout)
# ==============================================================================


def plot_diagnostics(out, forc, base, fig_dir="figures"):
    """
    Generate and save diagnostic PNG figures to fig_dir/.

    Figures
    -------
    01  Snow layer temperatures (T1, T2, T3)
    02  Air/ground temperature, precipitation, and solar irradiance overview
    03  Liquid water content per layer
    04  Cumulative melt and runoff
    05  Daily stacked heat-flux components at the surface and ground
    06  Ground interface (soil T, bottom snow T, q_ground)
    07  Insulation k_eff and alpha_eff
    08  Insulation moisture content
    09  Layer density, total pile height, and snow volume evolution
    10  Cover temperature Tc vs air temperature
    11  Individual SEB flux terms over time
    12  Layered cross-section energy-flow diagram
    """
    os.makedirs(fig_dir, exist_ok=True)
    days = forc.days
    sn   = base.snow
    t0_date = forc.t0_dt  # Starting datetime from CSV

    # --- Calendar x-axis helper ---
    def set_calendar_xaxis(ax, days_array, tick_interval=15):
        """Apply calendar date formatting to x-axis."""
        xt = np.arange(days_array.min(), days_array.max() + 1e-9, tick_interval)
        xt_labels = [(t0_date + pd.Timedelta(days=float(x))).strftime("%d-%b") for x in xt]
        ax.set_xticks(xt)
        ax.set_xticklabels(xt_labels, rotation=45)
        ax.set_xlim([days_array.min(), days_array.max()])

    def save(fig, name):
        fig.savefig(os.path.join(fig_dir, name), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    saved {name}")

    # ================================================================== #
    #  1. Snow Layer Temperatures                                         #
    # ================================================================== #
    fig, ax = plt.subplots(figsize=(10, 5))
    
    l_t1, = ax.plot(days, out.T_hist[0] - TFREEZE,
                    color="tab:blue",   lw=1.5, label="T1 (surface)")
    l_t2, = ax.plot(days, out.T_hist[1] - TFREEZE,
                    color="tab:orange", lw=1.5, label="T2 (middle)")
    l_t3, = ax.plot(days, out.T_hist[2] - TFREEZE,
                    color="tab:green",    lw=1.5, label="T3 (bottom)")
    
    ax.axhline(0, color="gray", ls=":", alpha=0.5, label="Freezing line")
    
    set_calendar_xaxis(ax, days)
    ax.set_ylabel("Snow Temperature (°C)")
    ax.set_title("Snow Layer Temperature Evolution")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    save(fig, "01_temperatures.png")

    # ================================================================== #
    #  2. Triple Plot: Air+Ground temp (left), Precip (upper right),     #
    #     Solar irradiation (lower right)                                #
    # ================================================================== #
    fig = plt.figure(figsize=(16, 10))
    
    # Create a custom GridSpec with 3 columns instead of 2
    # Left column takes 3/5 of width, right column takes 2/5
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 5, figure=fig)  # 2 rows, 5 columns total
    
    # --- Left panel: Air and Ground temperature (spans all rows, first 3 columns) ---
    ax_left = fig.add_subplot(gs[:, :3])  # All rows, columns 0-2
    
    ax_left.plot(days, out.Ta_hist - TFREEZE, "k--", lw=1, alpha=0.5, label="Air temp")
    ax_left.plot(days, out.Tsoil_hist - TFREEZE, "g--", lw=1, alpha=0.5, label="Soil temp (40 cm)")
    ax_left.axhline(0, color="gray", ls=":", alpha=0.5)
    
    set_calendar_xaxis(ax_left, days, tick_interval=30)
    ax_left.set_ylabel("Temperature (°C)")
    ax_left.set_title("Air and Ground Temperature")
    ax_left.legend(fontsize=8)
    ax_left.grid(True, alpha=0.3)
    
    # --- Upper right: Precipitation (columns 3-4) ---
    ax_upper_right = fig.add_subplot(gs[0, 3:])  # Row 0, columns 3-4
    
    ax_upper_right.fill_between(days, 0, out.Pr_hist * 1000.0 * 3600.0,
                                alpha=0.5, color="blue")
    
    set_calendar_xaxis(ax_upper_right, days, tick_interval=30)
    ax_upper_right.set_ylabel("Precip (mm/h)")
    ax_upper_right.set_title("Precipitation")
    ax_upper_right.grid(True, alpha=0.3)
    
    # --- Lower right: Solar irradiation (columns 3-4) ---
    ax_lower_right = fig.add_subplot(gs[1, 3:])  # Row 1, columns 3-4
    
    ax_lower_right.fill_between(days, 0, out.G_hist, alpha=0.5)
    
    set_calendar_xaxis(ax_lower_right, days, tick_interval=30)
    ax_lower_right.set_ylabel("Solar (W/m²)")
    ax_lower_right.set_title("Solar Radiation Input")
    ax_lower_right.grid(True, alpha=0.3)
    
    # Adjust layout with wider left panel
    plt.subplots_adjust(wspace=0.4, hspace=0.35, left=0.06, right=0.97, top=0.95, bottom=0.08)
    
    save(fig, "02_meteorological_overview.png")

    # ------------------------------------------------------------------ #
    #  3. Liquid water content                                           #
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(days, out.LWC_hist[0], lw=1.5, label="Layer 1 (surface)")
    ax.plot(days, out.LWC_hist[1], lw=1.5, label="Layer 2 (middle)")
    ax.plot(days, out.LWC_hist[2], lw=1.5, label="Layer 3 (bottom)")
    ax.axhline(THETA_E, color="r", ls="--", lw=1, label="Field capacity")
    set_calendar_xaxis(ax, days)
    ax.set_ylabel("Liquid Water Content (-)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    save(fig, "03_liquid_water_content.png")

    # ------------------------------------------------------------------ #
    #  4. Cumulative melt and runoff                                     #
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(10, 5))
    cumul_melt   = np.cumsum(np.nan_to_num(out.melt_rate) * forc.dt)
    cumul_runoff = np.cumsum(np.nan_to_num(out.runoff_h)) / RHO_W
    ax.plot(days, cumul_melt,   lw=2, label="Cumulative melt")
    ax.plot(days, cumul_runoff, lw=2, label="Cumulative runoff")
    set_calendar_xaxis(ax, days)
    ax.set_ylabel("Melt and Runoff (m w.e.)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    save(fig, "04_melt_and_runoff.png")

    # ------------------------------------------------------------------ #
    #  5. Heat flux components — stacked daily bars                      #
    # ------------------------------------------------------------------ #
    
    # -- Aggregate 10-min data to daily totals [MJ/m^2 per day] --
    dt_day = 86400.0                          
    n_steps_per_day = int(dt_day / forc.dt)   
    n_days_total = len(days) // n_steps_per_day
    trim_len = n_days_total * n_steps_per_day
    
    def daily_sum(arr):
        valid = np.nan_to_num(arr[:trim_len])
        energy = valid * forc.dt                 
        daily  = energy.reshape(n_days_total, n_steps_per_day).sum(axis=1)
        return daily / 1e6                       
    
    # Extract all fluxes (including Longwave)
    E_sw_daily     = daily_sum(out.qSW_hist)       
    E_rain_daily   = daily_sum(out.qRAIN_hist)     
    E_sens_daily   = daily_sum(out.qH_hist)        
    E_ground_daily = daily_sum(out.qground_hist)   
    E_latent_daily = daily_sum(out.qE_hist)        
    E_lw_daily     = daily_sum(out.qLW_hist)
    
    # Helper functions to isolate positive (sources) and negative (sinks)
    def pos(arr): return np.maximum(arr, 0)
    def neg(arr): return np.minimum(arr, 0)

    # Build daily date axis
    day_axis = np.arange(n_days_total)
    day_labels = [(t0_date + pd.Timedelta(days=float(d))).strftime("%d-%b") 
                  for d in day_axis]
    tick_step = max(1, n_days_total // 10)
    tick_pos = day_axis[::tick_step]
    tick_lab = [day_labels[i] for i in tick_pos]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    bar_width = 0.8
    
    # --- Plot Positive Fluxes (Stacking up from zero) ---
    bottom_pos = np.zeros(n_days_total)
    
    ax.bar(day_axis, pos(E_ground_daily), bar_width, bottom=bottom_pos, color='#8B4513', alpha=0.85, label='Ground heat')
    bottom_pos += pos(E_ground_daily)
    
    ax.bar(day_axis, pos(E_sw_daily), bar_width, bottom=bottom_pos, color='#FFD700', alpha=0.85, label='Solar radiation')
    bottom_pos += pos(E_sw_daily)
    
    ax.bar(day_axis, pos(E_sens_daily), bar_width, bottom=bottom_pos, color='#FF8C00', alpha=0.85, label='Sensible (+)')
    bottom_pos += pos(E_sens_daily)
    
    ax.bar(day_axis, pos(E_rain_daily), bar_width, bottom=bottom_pos, color='#4682B4', alpha=0.85, label='Rain heat')
    bottom_pos += pos(E_rain_daily)
    
    ax.bar(day_axis, pos(E_latent_daily), bar_width, bottom=bottom_pos, color='#1ABC9C', alpha=0.85, label='Latent (+ / Condens)')
    bottom_pos += pos(E_latent_daily)

    # --- Plot Negative Fluxes (Stacking down from zero) ---
    bottom_neg = np.zeros(n_days_total)
    
    ax.bar(day_axis, neg(E_lw_daily), bar_width, bottom=bottom_neg, color='#9B59B6', alpha=0.85, label='Longwave cooling')
    bottom_neg += neg(E_lw_daily)
    
    ax.bar(day_axis, neg(E_sens_daily), bar_width, bottom=bottom_neg, color='#E67E22', alpha=0.85, label='Sensible (-)')
    bottom_neg += neg(E_sens_daily)
    
    ax.bar(day_axis, neg(E_latent_daily), bar_width, bottom=bottom_neg, color='#2E8B57', alpha=0.85, label='Latent (- / Evap)')
    bottom_neg += neg(E_latent_daily)

    ax.bar(day_axis, neg(E_ground_daily), bar_width, bottom=bottom_neg, color='#A0522D', alpha=0.85, label='Ground cooling')
    
    # Zero line
    ax.axhline(0, color='black', lw=1.2)
    
    # Axis labels and formatting
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab, rotation=45, ha='right', fontsize=8)
    ax.set_xlim(-0.5, n_days_total - 0.5)
    ax.set_ylabel('Daily energy flux (MJ/m²)', fontsize=12)
    
    # Clean up legend (limit to 4 columns to avoid clutter)
    ax.legend(fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=True)
    ax.grid(True, alpha=0.3, axis='y')
    
    fig.tight_layout()
    save(fig, "05_heat_fluxes.png")

    # ------------------------------------------------------------------ #
    #  6. Ground interface                                               #
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(days, out.Tsoil_hist - TFREEZE, "g-",  lw=1.5, label="Soil temp (320 cm)")
    ax.plot(days, out.T_hist[2]  - TFREEZE, "b-",  lw=1.5, label="T3 (bottom snow)")
    ax.plot(days, np.nan_to_num(out.qground_hist),  "r-",  lw=1,   label="Ground heat flux")
    ax.axhline(0, color="gray", ls=":", alpha=0.5)
    set_calendar_xaxis(ax, days)
    ax.set(ylabel="Temperature (°C) / Flux (W/m^2)",
           title="Ground Interface")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    save(fig, "06_ground_interface.png")

    # ------------------------------------------------------------------ #
    #  7 & 8. Insulation properties (only when insulation is present)    #
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(10, 5))
    ax2b = ax.twinx()
    ln1 = ax.plot(days, np.nan_to_num(out.k_eff_hist), "b-", lw=1.5,
                  label="k_eff (W/m/K)")
    ln2 = ax2b.plot(days, np.nan_to_num(out.alb_hist), "r-", lw=1.5,
                    label="alpha_eff (-)")
    set_calendar_xaxis(ax, days)
    ax.set_ylabel("Effective conductivity (W/m/K)", color="b")
    ax2b.set_ylabel("Insulation albedo (-)", color="r")
    ax.tick_params(axis="y", labelcolor="b")
    ax2b.tick_params(axis="y", labelcolor="r")
    lns  = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, fontsize=8)
    ax.grid(True, alpha=0.3)
    save(fig, "07_insulation_properties.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(days, np.nan_to_num(out.W_hist), lw=1.5, color="steelblue")
    set_calendar_xaxis(ax, days)
    ax.set(ylabel="Moisture (% of saturation)",
           title="Insulation Moisture Content")
    ax.grid(True, alpha=0.3)
    save(fig, "08_insulation_moisture.png")

    # ------------------------------------------------------------------ #
    #  9. SWE, Density Evolution, and Snow Volume                        #
    # ------------------------------------------------------------------ #
    fig = plt.figure(figsize=(18, 10))
    
    # Use subplot2grid for more control and better tight_layout compatibility
    # Main density heatmap (left side, full height)
    ax_density = plt.subplot2grid((2, 5), (0, 0), rowspan=2, colspan=3, fig=fig)
    
    # Top-right: Total height evolution
    ax_height = plt.subplot2grid((2, 5), (0, 3), colspan=2, fig=fig)
    
    # Bottom-right: Volume evolution
    ax_volume = plt.subplot2grid((2, 5), (1, 3), colspan=2, fig=fig)
    
    # Calculate layer densities over time
    n_timesteps = len(days)
    density_map = np.zeros((3, n_timesteps))
    
    for i in range(3):
        density_map[i, :] = (out.ice_frac_hist[i] * RHO_I + 
                            np.nan_to_num(out.LWC_hist[i]) * RHO_W)
    
    # Flip the density map vertically so bottom layer is at the bottom of the plot
    density_map_flipped = np.flipud(density_map)
    
    # Create the imshow plot
    im = ax_density.imshow(density_map_flipped, 
                          aspect='auto',
                          origin='lower',
                          cmap='PuBu',
                          interpolation='spline16',
                          extent=[days[0], days[-1], 0, 3],
                          vmin=300,
                          vmax=700)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax_density, fraction=0.035, pad=0.02)
    cbar.set_label('Density (kg/m³)', rotation=270, labelpad=15)
    
    # Customize y-axis
    ax_density.set_yticks([0.5, 1.5, 2.5])
    ax_density.set_yticklabels(['Layer 3\n(Bottom)', 'Layer 2\n(Middle)', 'Layer 1\n(Surface)'])
    ax_density.set_ylabel('Snow Layer')
    ax_density.set_title('Snow Density Evolution', fontsize=11)
    
    # Add calendar x-axis to density heatmap
    xt = np.arange(days.min(), days.max() + 1e-9, 15)
    xt_labels = [(t0_date + pd.Timedelta(days=float(x))).strftime("%d-%b") for x in xt]
    ax_density.set_xticks(xt)
    ax_density.set_xticklabels(xt_labels, rotation=45)
    
    # Add contour lines
    contour_levels = [400, 450, 500, 550]
    y_coords = np.array([0.5, 1.5, 2.5])
    CS = ax_density.contour(days, y_coords, density_map_flipped, 
                           levels=contour_levels, 
                           colors='white', 
                           linewidths=0.8, 
                           alpha=0.6)
    
    # Manual label positions
    time_min = days[0]
    time_max = days[-1]
    time_margin = 0.2 * (time_max - time_min)
    time_safe_min = time_min + time_margin
    time_safe_max = time_max - time_margin
    
    manual_positions = []
    for level_idx in range(len(CS.levels)):
        paths = CS.get_paths()
        if level_idx < len(paths):
            vertices = paths[level_idx].vertices
            if len(vertices) > 2:
                safe_points = vertices[(vertices[:, 0] >= time_safe_min) & 
                                      (vertices[:, 0] <= time_safe_max)]
                if len(safe_points) > 0:
                    mid_idx = len(safe_points) // 2
                    point = safe_points[mid_idx]
                    manual_positions.append(point)
                elif len(vertices) > 0:
                    mid_idx = len(vertices) // 2
                    manual_positions.append(vertices[mid_idx])
    
    if manual_positions:
        ax_density.clabel(CS, inline=True, fontsize=9, fmt='%d kg/m^3', 
                         manual=manual_positions)
    else:
        ax_density.clabel(CS, inline=True, fontsize=9, fmt='%d kg/m^3')
    
    # Add text annotations
    ax_density.annotate('Fresh snow\n(low density)', 
                       xy=(0.02, 0.95),
                       xycoords='axes fraction',
                       fontsize=9,
                       color='darkblue',
                       verticalalignment='top',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax_density.annotate('Compacted snow\n(high density)', 
                       xy=(0.98, 0.05),
                       xycoords='axes fraction',
                       fontsize=9,
                       color='darkred',
                       horizontalalignment='right',
                       verticalalignment='bottom',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax_density.grid(False)
    
    # Top-right: Total height evolution
    tot_height = np.sum(out.heights_hist, axis=0)
    ax_height.plot(days, tot_height, 'b-', lw=2)
    ax_height.fill_between(days, 0, tot_height, alpha=0.2, color='blue')
    set_calendar_xaxis(ax_height, days, tick_interval=30)  # Wider ticks for smaller subplot
    ax_height.set_ylabel('Total Height (m)')
    ax_height.set_title('Snow Pile Height', fontsize=10)
    ax_height.grid(True, alpha=0.3)
    
    # Add initial height line
    initial_height = 3 * base.snow.dz
    ax_height.axhline(initial_height, color='gray', ls='--', alpha=0.5, 
                     label=f'Initial: {initial_height:.1f} m')
    ax_height.legend(fontsize=7, loc='center')
    
    # Bottom-right: Volume evolution
    if out.V_hist is not None:
        ax_volume.fill_between(days, 0, out.V_hist / 1000.0,
                              alpha=0.3, color="steelblue")
        ax_volume.plot(days, out.V_hist / 1000.0,
                      color="steelblue", lw=2)
        ax_volume.axhline(base.meta.V0 / 1000.0, color="gray", ls=":",
                         lw=1, label=f'V₀ = {base.meta.V0/1000:.0f} × 10 m³')
        
        if np.isfinite(out.meltout_day):
            ax_volume.axvline(out.meltout_day, color="red", ls="--", lw=1.5,
                            label=f'Melt-out: day {out.meltout_day:.1f}')
        
        set_calendar_xaxis(ax_volume, days, tick_interval=30)  # Wider ticks for smaller subplot
        ax_volume.set_ylabel('Volume (m³)')
        ax_volume.set_title('Snow Volume', fontsize=10)
        ax_volume.legend(fontsize=7, loc='center')
        ax_volume.grid(True, alpha=0.3)
    
    # Adjust spacing
    plt.subplots_adjust(left=0.10, right=0.95, top=0.95, bottom=0.10,
                       wspace=0.55, hspace=0.40)
    
    save(fig, "09_rho_h_V.png")

    # ------------------------------------------------------------------ #
    #  10. Cover temperature                                             #
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(days, out.Tc_hist - TFREEZE, "tab:purple",
            lw=1.5, alpha=0.8, label="Cover temp Tc (SEB)")
    ax.plot(days, out.Ta_hist    - TFREEZE, "k--", lw=1, alpha=0.6, label="Air temp")
    ax.axhline(0, color="gray", ls=":", alpha=0.5)
    set_calendar_xaxis(ax, days)
    ax.set(ylabel="Temperature (°C)",
           title="Cover Temperature (SEB bisection solve)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    save(fig, "10_cover_temperature.png")

    # ------------------------------------------------------------------ #
    #  11. Individual SEB flux terms                                     #
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(days, np.nan_to_num(out.qLW_hist),   lw=1, alpha=0.8, label="Long-wave (qLW)")
    ax.plot(days, np.nan_to_num(out.qH_hist),    lw=1, alpha=0.8, label="Sensible (qH)")
    ax.plot(days, np.nan_to_num(out.qE_hist),    lw=1, alpha=0.8, label="Latent (qE)")
    ax.plot(days, np.nan_to_num(out.qRAIN_hist), lw=1, alpha=0.8, label="Rain (qRAIN)")
    ax.plot(days, np.nan_to_num(out.qins_hist),  lw=1.5, ls="--",
            color="black", label="Insulation flux (qins)")
    ax.axhline(0, color="gray", ls=":", alpha=0.4)
    set_calendar_xaxis(ax, days)
    ax.set(ylabel="Heat flux (W/m²)",
           title="Individual SEB Flux Components at Cover Layer")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    save(fig, "11_flux_components.png")

    print(f"\n  All figures saved to '{fig_dir}/'")

def plot_energy_balance_layered(out, forc, base, fig_dir="figures"):
    """
    Layered cross-section diagram showing energy flows through each physical layer:
    WOODCHIP COVER → SNOW → CONCRETE BASE
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Rectangle
    import numpy as np
    import os
    
    t = forc.t
    
    # ---- Energy integrals [MJ/m²] ----
    E_solar  = np.trapezoid(np.nan_to_num(out.qSW_hist),    t) / 1e6
    E_sens   = np.trapezoid(np.nan_to_num(out.qH_hist),     t) / 1e6
    E_rain   = np.trapezoid(np.nan_to_num(out.qRAIN_hist),  t) / 1e6
    E_ground = np.trapezoid(np.nan_to_num(out.qground_hist),t) / 1e6
    E_LW     = np.trapezoid(np.nan_to_num(out.qLW_hist),    t) / 1e6
    E_latent = np.trapezoid(np.nan_to_num(out.qE_hist),     t) / 1e6
    E_qtop   = np.trapezoid(np.nan_to_num(out.qtop_hist),   t) / 1e6
    
    E_melt   = out.melt_total * RHO_I * LF / 1e6
    E_refroz = np.nansum(out.refrozen) * LF / 1e6
    Cs_arr = out.Cs_hist
    T_arr  = out.T_hist
    dT_arr = np.diff(T_arr, axis=1)
    E_dT   = np.sum(Cs_arr[:, :-1] * dT_arr) / 1e6
    
    # ---- Create figure (Square aspect ratio eliminates distortion) ----
    fig, ax = plt.subplots(figsize=(13, 13))  
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 14)
    ax.axis('off')
    fig.patch.set_facecolor('white')
    
    # ---- Colors ----
    C_COVER    = '#C19A6B'
    C_SNOW     = '#E8F4FD'
    C_CONCRETE = '#95A5A6'
    C_SOLAR    = '#F39C12'
    C_SENS     = '#E74C3C'
    C_RAIN     = '#3498DB'
    C_GROUND   = '#8B4513'
    C_LW       = '#E67E22'
    C_LATENT   = '#1ABC9C'
    C_IN       = '#27AE60'
    
    # ---- Layer geometry ----
    LAYER_LEFT  = 2.5
    LAYER_RIGHT = 11.5
    LAYER_WIDTH = LAYER_RIGHT - LAYER_LEFT
    
    COVER_TOP    = 11.5
    COVER_BOT    = 10.5   # Thickness = 1.0
    SNOW_TOP     = 10.5
    SNOW_BOT     = 6.5    # Thickness = 4.0
    CONCRETE_TOP = 6.5
    CONCRETE_BOT = 5.2    # Thickness = 1.3

    FONT_SIZE = 18        # Noticeably larger text labels
    ARROW_LEN = 1.6       # Standardized arrow size
    
    # ---- Helper functions ----
    def draw_layer(y_top, y_bot, color, label, hatch=None):
        rect = Rectangle((LAYER_LEFT, y_bot), LAYER_WIDTH, y_top - y_bot,
                        facecolor=color, edgecolor='#2C3E50', linewidth=2.5,
                        zorder=2)
        if hatch:
            rect.set_hatch(hatch)
        ax.add_patch(rect)
        ax.text(LAYER_LEFT + LAYER_WIDTH/2, (y_top + y_bot)/2, label,
               ha='center', va='center', fontsize=22, fontweight='bold',  # Increased header size
               color='#2C3E50', zorder=3)
    
    def draw_energy_flow(x, y, direction, value, color, label, 
                         offset_x=0, offset_y=0, fontsize=FONT_SIZE, angle=0):
        
        if direction == 'down':
            dx, dy = 0, -ARROW_LEN
            ha, va = 'center', 'top'
        elif direction == 'up':
            dx, dy = 0, ARROW_LEN
            ha, va = 'center', 'bottom'
        elif direction == 'ne':
            rad = np.radians(angle) if angle else np.radians(45)
            dx, dy = ARROW_LEN * np.cos(rad), ARROW_LEN * np.sin(rad)
            ha, va = 'left', 'bottom'
        
        ax.annotate('', xy=(x + dx, y + dy), xytext=(x, y),
                    arrowprops=dict(arrowstyle='->', color=color, lw=6,  # Thicker lines
                                   connectionstyle='arc3,rad=0'),
                    zorder=5)
        
        label_x = x + dx/2 + offset_x
        label_y = y + dy/2 + offset_y
        
        ax.text(label_x, label_y, f'{label}\n{value:.0f} MJ/m²',
                ha=ha, va=va, fontsize=fontsize, fontweight='bold',
                color='#2C3E50',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                        edgecolor='#BDC3C7', linewidth=1.5, alpha=0.95),
                zorder=6)
    
    def draw_simple_box(x, y, w, h, label, value, fontsize=FONT_SIZE):
        ax.text(x, y, f'{label}\n{value:.0f} MJ/m²',
                ha='center', va='center', fontsize=fontsize, fontweight='bold',
                color='#2C3E50',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                        edgecolor='#BDC3C7', linewidth=1.5, alpha=0.95),
                zorder=8)
        
    def draw_simple_box_neg(x, y, w, h, label, value, fontsize=FONT_SIZE):
        ax.text(x, y, f'{label}\n-{abs(value):.0f} MJ/m²',
                ha='center', va='center', fontsize=fontsize, fontweight='bold',
                color='#2C3E50',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                        edgecolor='#BDC3C7', linewidth=1.5, alpha=0.95),
                zorder=8)
    
    # =======================================
    # DRAW THE THREE LAYERS
    # =======================================
    draw_layer(COVER_TOP, COVER_BOT, C_COVER, 'WOODCHIP COVER', hatch='///')
    draw_layer(SNOW_TOP, SNOW_BOT, C_SNOW, 'SNOW')
    draw_layer(CONCRETE_TOP, CONCRETE_BOT, C_CONCRETE, 'CONCRETE BASE', hatch='...')
    
    # =======================================
    # WOODCHIP COVER FLOWS
    # =======================================
    # Downward inputs spaced out evenly across the top edge
    draw_energy_flow(LAYER_LEFT + 2.5, COVER_TOP + ARROW_LEN, 'down', E_solar, C_SOLAR,
                    'Solar', offset_x=0.0, offset_y=1.0, fontsize=FONT_SIZE)
    draw_energy_flow(LAYER_LEFT + 4.5, COVER_TOP + ARROW_LEN, 'down', E_sens, C_SENS,
                    'Air temp.', offset_x=0.0, offset_y=1.0, fontsize=FONT_SIZE)
    draw_energy_flow(LAYER_LEFT + 6.5, COVER_TOP + ARROW_LEN, 'down', E_rain, C_RAIN,
                    'Rain', offset_x=0.0, offset_y=1.0, fontsize=FONT_SIZE)
    
    # Longwave loss — pointing up/away diagonally from the top right edge
    draw_energy_flow(LAYER_RIGHT - 1.0, COVER_TOP, 'ne', E_LW, C_LW,
                    'Longwave\nemission', offset_x=0.75, offset_y=-0.3, 
                    fontsize=FONT_SIZE, angle=45)
    
    # Latent heat — evaporation pointing up away from the cover bottom
    draw_energy_flow(LAYER_LEFT + 0.8, COVER_BOT, 'up', E_latent, C_LATENT,
                    'Latent heat\n(evaporation)', offset_x=-1.5, offset_y=-0.5, fontsize=FONT_SIZE)
    
    # Heat conducted into snow pack from the cover bottom
    draw_energy_flow(LAYER_LEFT + LAYER_WIDTH/2, COVER_BOT, 'down', E_qtop, C_IN,
                    'Heat into\nsnowpack', offset_x=1.3, offset_y=0.2, fontsize=FONT_SIZE)
    
    # =======================================
    # SNOWPACK INTERNAL ENERGY CHANGES
    # =======================================
    # Balanced directly along the exact vertical midpoint of the snow layer
    SNOW_MID = (SNOW_TOP + SNOW_BOT) / 2
    box_y = SNOW_MID
    
    melt_x = LAYER_LEFT + LAYER_WIDTH * 0.20
    warm_x = LAYER_LEFT + LAYER_WIDTH * 0.50
    refroz_x = LAYER_LEFT + LAYER_WIDTH * 0.80
    
    draw_simple_box(melt_x,       box_y-1, 0, 0, 'Melt', E_melt,         fontsize=FONT_SIZE)
    draw_simple_box(warm_x,       box_y-1, 0, 0, 'Warming', E_dT,         fontsize=FONT_SIZE)
    draw_simple_box_neg(refroz_x, box_y-1, 0, 0, 'Refreezing', E_refroz, fontsize=FONT_SIZE)
    
    # =======================================
    # CONCRETE / GROUND FLOW
    # =======================================
    # Ground heat flux pointing straight up into the bottom of concrete base
    draw_energy_flow(LAYER_LEFT + LAYER_WIDTH/2 , CONCRETE_BOT - ARROW_LEN, 'up', E_ground, C_GROUND,
                    'Ground heat', offset_x=-1.3, offset_y=-0.2, fontsize=FONT_SIZE)
    
    plt.tight_layout()
    
    os.makedirs(fig_dir, exist_ok=True)
    fig.savefig(os.path.join(fig_dir, "12_energy_balance_layered.png"),
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"    saved 12_energy_balance_layered.png")

# ==============================================================================
#  MAIN
# ==============================================================================

def main():
    print("=" * 60)
    print("  Snow Storage RC Model")
    print("=" * 60)

    # ------------------------------------------------------------------
    #  1. Load forcing and build parameters
    # ------------------------------------------------------------------
    DATA_FILE = "DATA_2024_40cm.csv"

    try:
        forc, base, p0 = load_primary_forcing(DATA_FILE)
    except FileNotFoundError:
        print(f"\nERROR: '{DATA_FILE}' not found.")
        print("Please check the file paths and re-run.")
        return

    n_hours = int(round(forc.t[-1] / 3600.0))
    print(f"\nForcing  : {n_hours} h  ({n_hours/24:.1f} days)")
    print(f"Time step: {forc.dt:.0f} s   ({forc.dt/60:.0f} min) | "
          f"Steps: {forc.Nt}")
    print(f"Snow Hs  : {base.snow.Hs:.1f} m   |  rho_s = {base.snow.rho:.0f} kg/m^3")
    print(f"Hi       : {p0.Hi:.2f} m  |  "
          f"Aging model: {USE_AGING_MODEL}")

    # ------------------------------------------------------------------
    #  2. Run model
    # ------------------------------------------------------------------
    print("\nRunning simulation...")
    t_wall = py_time.time()
    out = run_snow_model(p0, forc, base)
    elapsed = py_time.time() - t_wall
    print(f"  Progress: 100%")
    print(f"\nSimulation complete.  Elapsed: {elapsed:.2f} s")
    print(f"Total melt : {out.melt_total*1000:.1f} mm w.e.")
    if out.V_hist is not None:
        print(f"V0         : {base.meta.V0:.0f} m^3  |  Aref = {base.meta.Aref:.0f} m^2")
        print(f"V_end      : {out.V_hist[-1]:.0f} m^3  "
              f"({100*out.V_hist[-1]/base.meta.V0:.1f} % of V0 remaining)")
        if np.isfinite(out.meltout_day):
            print(f"Melt-out   : day {out.meltout_day:.1f}")
        else:
            print("Melt-out   : not reached within simulation period")

    # ------------------------------------------------------------------
    #  3. Energy balance diagnostics
    # ------------------------------------------------------------------
    print_energy_balance(out, forc, base)

    # ------------------------------------------------------------------
    #  4. Figures
    # ------------------------------------------------------------------
    print("\nGenerating figures...")
    plot_diagnostics(out, forc, base, fig_dir="figures")
    plot_energy_balance_layered(out, forc, base, fig_dir="figures")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()