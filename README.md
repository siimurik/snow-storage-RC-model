# Snow Storage RC Model

A physics-based thermal model for seasonal snow storage piles, implemented in
Python. Simulates the temperature evolution, melting, refreezing, and runoff of
a multi-layer snowpack driven by real hourly meteorological data.

---

## Overview

Seasonal snow storage is a technique for retaining winter snowfall to provide
cooling during summer. A covering of woodchip insulation dramatically reduces
melt losses. **snowsim** models this system using a 3-layer RC
(resistance-capacitance) thermal network, where each snow layer has its own
temperature and liquid water content that evolve in response to atmospheric and
ground boundary conditions.

Results are benchmarked against two empirical melt-rate models (Skogsberg 2005;
Skogsberg & Nordell 2001), a transient 1D implicit finite-difference solver
through the insulation layer, and optionally against SNOWPACK simulation output.

---

## System Schematic

```
  ╔══════════════════════════════════════════╗
  ║                  AIR                     ║
  ║   T_air  |  U_wind  |  I_solar  |  RH    ║
  ╚═══════════════════╤══════════════════════╝
                      │  h_eff = h_conv + h_rad
                      │  (convection + longwave radiation)
  ╔═══════════════════╧══════════════════════╗
  ║           WOODCHIP INSULATION            ║
  ║   thickness Hi  |  k_eff(W, age)         ║
  ║   moisture W    |  alpha_eff(W, age)     ║
  ╚═══════════════════╤══════════════════════╝
                      │  R_ins = Hi / k_eff
                      │
  ╔═══════════════════╧══════════════════════╗
  ║          SNOW LAYER 1  (surface)         ║
  ║   T1  |  LWC1  |  ice_frac1              ║
  ╠══════════════════════════════════════════╣
  ║               R_12 = dz/k_snow           ║
  ╠══════════════════════════════════════════╣
  ║          SNOW LAYER 2  (middle)          ║
  ║   T2  |  LWC2  |  ice_frac2              ║
  ╠══════════════════════════════════════════╣
  ║               R_23 = dz/k_snow           ║
  ╠══════════════════════════════════════════╣
  ║          SNOW LAYER 3  (bottom)          ║
  ║   T3  |  LWC3  |  ice_frac3              ║
  ╚═══════════════════╤══════════════════════╝
                      │  Robin BC
                      │  q = (T_soil - T3) / (L/k_soil + 1/h_ground)
  ╔═══════════════════╧══════════════════════╗
  ║                  GROUND                  ║
  ║   T_soil (from CSV, 3.2 m depth)         ║
  ╚══════════════════════════════════════════╝

  Percolation:  LWC1 → LWC2 → LWC3 → runoff   (bucket method)
  Refreezing:   applied per layer when T < 0°C  (cold-content method)
```

---

## Features

- **3-layer RC snow model** — temperature and liquid water content per layer,
  integrated with a 4th-order Runge-Kutta (RK4) scheme at a 10-minute time step
- **Advanced insulation model** — effective conductivity `k_eff` and solar
  absorptivity `alpha_eff` evolve with woodchip moisture content and material age
- **Refreezing** — cold-content approach following Bartelt & Lehning (2002)
- **Percolation** — bucket-method drainage between layers with irreducible water
  content threshold; excess becomes runoff
- **Ground Robin BC** — soil temperature read directly from CSV; combined
  conduction + interface resistance at the snow-ground boundary
- **Transient 1D reference solver** — implicit finite-difference scheme through
  the insulation layer with wind-dependent outer HTC; JIT-compiled with Numba
- **Empirical model comparison** — Skogsberg (2005) and Skogsberg & Nordell
  (2001) melt-rate formulas evaluated over the same forcing
- **Optional SNOWPACK validation** — loads a SNOWPACK `.met` output file for
  side-by-side cumulative melt and SWE comparison
- **12 output figures** saved automatically to `figures/`

---

## Installation

Python 3.9 or later is recommended.

### Install dependencies

```bash
pip install numpy matplotlib numba pandas
```

| Package      | Purpose                                      |
|--------------|----------------------------------------------|
| `numpy`      | Array operations and numerical integration   |
| `matplotlib` | Plotting and figure output                   |
| `numba`      | JIT compilation of the RK4 and 1D solvers    |
| `pandas`     | SNOWPACK `.met` file parsing (optional)      |

> **Note:** On first run, Numba will compile the JIT functions. This adds
> roughly 2–3 seconds of startup time but only occurs once per installation.

---

## Input Data

The model reads hourly meteorological data from a CSV file named
`DATA_2024.csv`, which must be placed in the working directory.

### Required columns

| Column              | Description                        | Unit    |
|---------------------|------------------------------------|---------|
| `Time`              | Timestamp                          | string  |
| `Temp_C`            | Air temperature                    | °C      |
| `Air_Vel_m/s_10m`   | Wind speed at 10 m height          | m/s     |
| `Prec_m/h`          | Precipitation rate                 | m/h     |
| `Glo_Sol_Ir_W/m2`   | Global solar irradiance            | W/m²    |
| `RH_%`              | Relative humidity                  | %       |
| `Soil_Temp_320cm`   | Soil temperature at 3.2 m depth    | °C      |


### Optional SNOWPACK validation file

Place a SNOWPACK `.met` output file at:

```
output/snow_storage_snow_storage.met
```

If found, cumulative runoff and SWE from SNOWPACK are included in the
comparison plots. If not found, the simulation runs without it.

---

## Usage

```bash
python3 main.py
```

The simulation will print progress, energy balance diagnostics and SMR
comparison statistics to the console. Figures are saved to `figures/`.

### Key parameters

All tunable inputs are collected at the top of the code under clearly
labelled sections. The most commonly adjusted values are:

| Parameter        | Description                                  | Default |
|------------------|----------------------------------------------|---------|
| `Hs`             | Total snow pile thickness                    | 4.5 m   |
| `Hi`             | Woodchip insulation thickness                | 0.20 m  |
| `k_snow`         | Snow thermal conductivity                    | 0.45 W/mK |
| `k_i_base`       | Insulation base conductivity                 | 0.07 W/mK |
| `h_ground`       | Ground interface heat transfer coefficient   | 2.5 W/m²K |
| `rho_s`          | Snow bulk density                            | 400 kg/m³ |
| `theta_e`        | Irreducible liquid water content             | 0.04    |

Feature flags near the top of the file enable or disable physics modules:

```python
USE_ADVANCED_INSULATION = True   # moisture- and age-dependent insulation
USE_REFREEZING          = True   # refreeze liquid water in sub-zero layers
USE_PERCOLATION         = True   # bucket-method percolation between layers
```

---



## License

[MIT](LICENSE)
