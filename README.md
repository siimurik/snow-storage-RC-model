# Snow Storage RC Model

A physics-based thermal model for seasonal snow storage piles, implemented in Python and Fortran. Simulates the temperature evolution, melting, refreezing and densification of a multi-layer snowpack driven by real hourly Estonian meteorological data.

---

## Overview

Seasonal snow storage is a technique for retaining winter snowfall to provide cooling during summer. A covering of woodchip insulation dramatically reduces melt losses. The code models this system using a 3-layer RC (resistance-capacitance) thermal network, where each snow layer has its own temperature, liquid water content and density that evolve in response to atmospheric and ground boundary conditions.

Results are validated against historical digitized volume data from Nordell & Sundin (1998).

---

## Features

* **3-layer RC snow model**: Tracks temperature and liquid water content per layer, integrated with a 4th-order Runge-Kutta (RK4) scheme at a 10-minute time step.
* **Full Surface Energy Balance (SEB)**: Cover surface temperature is solved dynamically at every time step using a bisection method balancing shortwave, longwave, sensible, latent, rain, and conduction fluxes.
* **Advanced insulation model**: Effective conductivity and solar absorptivity evolve with woodchip moisture content, material age, and porosity compaction.
* **Boone Overburden Densification**: Dynamic tracking of layer thickness and ice fraction as the snowpack compacts under its own weight.
* **Refreezing & Percolation**: Cold-content refreezing approach following Bartelt & Lehning (2002), and bucket-method drainage with an irreducible water content threshold.
* **Ground Robin BC**: Soil temperature read directly from CSV; handles combined conduction and interface resistance at the snow-ground boundary via a concrete pad.
* **Dual Implementations**: Available as a high-performance Numba JIT-compiled Python script and a fast OpenMP-enabled Fortran program.
* **Comprehensive Diagnostics**: Python version outputs 12 detailed diagnostic figures automatically saved to the `figures/` directory, including a layered cross-section energy flow diagram.

---

## Project Structure

| File | Description |
|---|---|
| `main.py` | Python implementation utilizing `numba` for JIT-compiled physics kernels and `matplotlib` for generating diagnostic plots. |
| `main.f90` | [cite_start]Fortran implementation utilizing `iso_fortran_env` for fast, compiled execution of the thermal network[cite: 1, 7]. |
| `validat.py` | Validation script that runs the model against digitized snow volume data from Nordell & Sundin (1998) for various insulation thicknesses. |

---

## System Schematic

```text
  ╔══════════════════════════════════════════╗
  ║                  AIR                     ║
  ║   Ta  |  U_wind  |  I_solar  |  RH       ║
  ╚═══════════════════╤══════════════════════╝
                      │  SEB Bisection Solver (qSW, qLW, qH, qE, qRAIN)
  ╔═══════════════════╧══════════════════════╗
  ║           WOODCHIP INSULATION            ║
  ║   thickness Hi  |  k_eff(W, age, zeta)   ║
  ║   moisture W    |  alpha_eff(W, age)     ║
  ╚═══════════════════╤══════════════════════╝
                      │  q_ins = (Tc - T1) / R_ins
  ╔═══════════════════╧══════════════════════╗
  ║          SNOW LAYER 1  (surface)         ║
  ║   T1  |  LWC1  |  ice_frac1  |  height1  ║
  ╠══════════════════════════════════════════╣
  ║               R_12 = (h1+h2) / k_snow    ║
  ╠══════════════════════════════════════════╣
  ║          SNOW LAYER 2  (middle)          ║
  ║   T2  |  LWC2  |  ice_frac2  |  height2  ║
  ╠══════════════════════════════════════════╣
  ║               R_23 = (h2+h3) / k_snow    ║
  ╠══════════════════════════════════════════╣
  ║          SNOW LAYER 3  (bottom)          ║
  ║   T3  |  LWC3  |  ice_frac3  |  height3  ║
  ╚═══════════════════╤══════════════════════╝
                      │  Robin BC (Snow + Concrete Pad + Ground)
  ╔═══════════════════╧══════════════════════╗
  ║                  GROUND                  ║
  ║   T_soil (dynamic from CSV or static)    ║
  ╚══════════════════════════════════════════╝

```

---

## Installation

### Python Dependencies

Python 3.9 or later is recommended.

```bash
pip install numpy matplotlib numba pandas scipy

```
---

## Input Data

The model reads hourly meteorological data from a CSV file. The default forcing file is currently named `DATA_2024_40cm.csv`.

### Required Columns

The model adapts to several common column aliases:

| Column Alias | Description | Unit |
| --- | --- | --- |
| `Time` / `Date_refYear` | Timestamp | string |
| `Temp_C` / `Ta_C` | Air temperature | °C |
| `Air_Vel_m/s_10m` / `U_ms` | Wind speed at 10 m height | m/s |
| `Prec_m/h` / `Pr_mm_day` | Precipitation rate | m/h or mm/day |
| `Glo_Sol_Ir_W/m2` / `G_W_m2` | Global solar irradiance | W/m² |
| `RH_%` / `RH_pct` | Relative humidity | % |
| `Soil_Temp_C` | Soil temperature | °C |

---

## License

[MIT](LICENSE.md)

```

```
