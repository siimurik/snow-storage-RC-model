! =============================================================================
!  main.f90  -  Snow Storage RC Thermal Model
!
!  Authors : Siim Erik Pugal, siim.pugal@taltech.ee                            
!            Hossein Alimohammadi,  hossein.alimohammadi@taltech.ee            
!  Date    : June 2026                                                         
!  License : MIT                                                               
!
!  Compilation (gfortran):
!    gfortran main.f90 -o main
!
!  On Windows:
!    gfortran -g -fbacktrace -fcheck=all -Wall -Wextra main.f90 -o main.exe
! =============================================================================

program main
    use iso_fortran_env, only: real64, int32
    use ieee_arithmetic, only: ieee_value, ieee_quiet_nan, ieee_is_nan
    implicit none

    ! =========================================================================
    !  Physical constants
    ! =========================================================================
    real(real64), parameter :: SIGMA       = 5.670374419d-8
    real(real64), parameter :: LF          = 3.34d5
    real(real64), parameter :: LV          = 2.5d6
    real(real64), parameter :: RHO_I       = 917.0d0
    real(real64), parameter :: RHO_W       = 1000.0d0
    real(real64), parameter :: RHO_AIR     = 1.225d0
    real(real64), parameter :: CP_AIR      = 1005.0d0
    real(real64), parameter :: C_W         = 4180.0d0
    real(real64), parameter :: C_S         = 2100.0d0
    real(real64), parameter :: P0          = 101325.0d0
    real(real64), parameter :: TFREEZE     = 273.15d0
    real(real64), parameter :: THETA_E     = 0.04d0

    logical, parameter :: USE_AGING_MODEL = .true.

    ! =========================================================================
    !  Type definitions
    ! =========================================================================
    type :: SolveParams
        real(real64) :: Tc_min_C      = -50.0d0
        real(real64) :: Tc_max_C      =  50.0d0
        real(real64) :: bracket_dT_lo =  40.0d0
        real(real64) :: bracket_dT_hi =  20.0d0
        integer      :: max_expand    =  6
        real(real64) :: expand_factor =  1.6d0
    end type SolveParams

    type :: TuningParams
        real(real64) :: CH          = 1.8d-3
        real(real64) :: CE          = 1.0d-3
        real(real64) :: CV          = 2.0d0
        real(real64) :: eps_c       = 0.95d0
        real(real64) :: f_shelter   = 0.4d0
        real(real64) :: Hi          = 0.20d0
        real(real64) :: k_dry       = 0.07d0
        real(real64) :: k_sat       = 0.12d0
        real(real64) :: n_k         = 1.5d0
        real(real64) :: W_sat       = 100.0d0
        real(real64) :: W_field     = 40.0d0
        real(real64) :: KD          = 5.0d-6
        real(real64) :: alb_dry     = 0.65d0
        real(real64) :: alb_wet     = 0.50d0
        real(real64) :: tau_dry     = 0.25d0
        real(real64) :: tau_wet     = 0.10d0
        real(real64) :: a_snow      = 0.55d0
        real(real64) :: beta_w      = 3.0d0
        real(real64) :: U10         = 1.0d0
        real(real64) :: delta_k_age     = 0.5d0
        real(real64) :: tau_k_years     = 2.0d0
        real(real64) :: delta_alpha_age = 0.05d0
        real(real64) :: tau_alpha_years = 2.0d0
        real(real64) :: zeta0      = 0.25d0
        real(real64) :: gamma_H    = 0.5d0
        real(real64) :: gamma_W    = 2.0d0
        type(SolveParams) :: solve
    end type TuningParams

    type :: SnowBase
        real(real64) :: Hs, dz, rho, c, k, Cs, R12, R23
        integer      :: Ns
        real(real64), dimension(3) :: T0
    end type SnowBase

    type :: GroundBase
        real(real64) :: Hg, kg, h_ground, R3g, Tg
    end type GroundBase

    type :: MetaBase
        logical      :: enable_volume
        real(real64) :: V0, rho_s, Hs_eff, Aref, rA
    end type MetaBase

    type :: Forcing
        real(real64)              :: dt
        real(real64), allocatable :: t(:), days(:)
        integer                   :: Nt
        real(real64), allocatable :: Ta(:), U(:), RH(:), G(:), Pr(:), Tg(:)
    end type Forcing

    type :: Output
        real(real64), allocatable :: T_hist(:,:), LWC_hist(:,:)
        real(real64), allocatable :: heights_hist(:,:), ice_frac_hist(:,:)
        real(real64), allocatable :: Cs_hist(:,:)
        real(real64), allocatable :: Tc_hist(:), Ta_hist(:), Tsoil_hist(:)
        real(real64), allocatable :: G_hist(:), Pr_hist(:)
        real(real64), allocatable :: W_hist(:), k_eff_hist(:), alpha_hist(:)
        real(real64), allocatable :: fsat_hist(:)
        real(real64), allocatable :: qSW_hist(:), qLW_hist(:), qH_hist(:)
        real(real64), allocatable :: qE_hist(:), qRAIN_hist(:)
        real(real64), allocatable :: qins_hist(:), qtop_hist(:)
        real(real64), allocatable :: qground_hist(:)
        real(real64), allocatable :: melt_top(:), melt_mid(:), melt_bot(:)
        real(real64), allocatable :: melt_rate(:)
        real(real64), allocatable :: runoff_h(:), refrozen(:), V_hist(:)
        real(real64)              :: melt_total, meltout_day
    end type Output

    ! =========================================================================
    !  Local variables
    ! =========================================================================
    type(TuningParams) :: main_p
    type(SnowBase)     :: main_sn
    type(GroundBase)   :: main_gr
    type(MetaBase)     :: main_meta
    type(Forcing)      :: main_forc
    type(Output)       :: main_out
    real(real64)       :: elapsed, start_time, end_time
    character(len=256) :: data_file

    data_file = "DATA_2024_40cm.csv"

    print *, "============================================================"
    print *, "  Snow Storage RC Model"
    print *, "============================================================"

    call load_primary_forcing(data_file, main_forc, main_sn, main_gr, &
                              main_meta, main_p)

    print '(A,I0,A,F6.1,A)', "Forcing  : ", &
          int(main_forc%t(main_forc%Nt)/3600.0d0), &
          " h  (", main_forc%t(main_forc%Nt)/86400.0d0, " days)"
    print '(A,F6.1,A,F4.1,A,I0)', "Time step: ", main_forc%dt, &
          " s   (", main_forc%dt/60.0d0, " min) | Steps: ", main_forc%Nt
    print '(A,F5.1,A,F6.0,A)', "Snow Hs  : ", main_sn%Hs, &
          " m   |  rho_s = ", main_sn%rho, " kg/m^3"
    print '(A,F5.2,A,L1)', "Hi       : ", main_p%Hi, &
          " m   |  Aging model: ", USE_AGING_MODEL

    print *, ""
    print *, "Running simulation..."
    call cpu_time(start_time)
    call run_snow_model(main_p, main_forc, main_sn, main_gr, main_meta, &
                        main_out)
    call cpu_time(end_time)
    elapsed = end_time - start_time

    print '(A,F7.1,A)', "Total melt : ", main_out%melt_total * 1000.0d0, &
          " mm w.e."
    if (allocated(main_out%V_hist)) then
        print '(A,F9.0,A,F9.0,A)', "V0         : ", main_meta%V0, &
              " m^3  |  Aref = ", main_meta%Aref, " m^2"
        print '(A,F9.0,A,F6.1,A)', "V_end      : ", &
              main_out%V_hist(main_forc%Nt), &
              " m^3  (", &
              100.0d0 * main_out%V_hist(main_forc%Nt) / main_meta%V0, &
              " % of V0 remaining)"
        if (.not. ieee_is_nan(main_out%meltout_day)) then
            print '(A,F7.1)', "Melt-out   : day ", main_out%meltout_day
        else
            print *, "Melt-out  : not reached within simulation period"
        end if
    end if

    call print_energy_balance(main_out, main_forc)
    !call debug_energy_balance(main_out, main_forc, main_sn)

    print *, ""
    print '(A,F6.3,A)', "Simulation complete. Elapsed time: ", elapsed, " s"

    print *, ""
    print *, "============================================================"
    print *, "Analysis complete!"
    print *, "============================================================"

contains

    ! =====================================================================
    !  CORE SIMULATION LOOP
    ! =====================================================================
    subroutine run_snow_model(p, forc, sn, gr, meta, out)
        type(TuningParams), intent(in) :: p
        type(Forcing), intent(in)      :: forc
        type(SnowBase), intent(in)     :: sn
        type(GroundBase), intent(in)   :: gr
        type(MetaBase), intent(in)     :: meta
        type(Output), intent(out)      :: out

        integer      :: k, Nt, i
        real(real64) :: dt, k_s_inv, R3g_const, a_snow, W_sat, KD, W_field
        real(real64) :: max_surf_flux
        real(real64) :: Tg_k, W, age_days, zeta, f
        real(real64) :: k_eff, alpha_eff, tau_eff, Rins, qSW_cov, qSW_into
        real(real64) :: Tc, qLW, qH, qE, qRAIN_out, qins, qtop
        real(real64) :: R12_dyn, R23_dyn, R3g_dyn, tot_refrozen, rf
        real(real64) :: dE, runoff, L_evap, E_rate, eta_r, m_in, D
        real(real64) :: rA

        real(real64), dimension(3) :: T, LWC_snow, ice_frac, heights
        real(real64), dimension(3) :: Tnew, Cs_layers
        real(real64) :: cum_melt

        dt = forc%dt
        Nt = forc%Nt

        allocate(out%T_hist(3, Nt), out%LWC_hist(3, Nt))
        allocate(out%heights_hist(3, Nt), out%ice_frac_hist(3, Nt))
        allocate(out%Cs_hist(3, Nt))
        allocate(out%Tc_hist(Nt), out%Ta_hist(Nt), out%Tsoil_hist(Nt))
        allocate(out%G_hist(Nt), out%Pr_hist(Nt))
        allocate(out%W_hist(Nt), out%k_eff_hist(Nt), out%alpha_hist(Nt))
        allocate(out%fsat_hist(Nt))
        allocate(out%qSW_hist(Nt), out%qLW_hist(Nt), out%qH_hist(Nt))
        allocate(out%qE_hist(Nt), out%qRAIN_hist(Nt))
        allocate(out%qins_hist(Nt), out%qtop_hist(Nt), out%qground_hist(Nt))
        allocate(out%melt_top(Nt), out%melt_mid(Nt), out%melt_bot(Nt))
        allocate(out%melt_rate(Nt))
        allocate(out%runoff_h(Nt), out%refrozen(Nt))

        out%T_hist = 0.0d0
        out%LWC_hist = 0.0d0
        out%heights_hist = 0.0d0
        out%ice_frac_hist = 0.0d0
        out%Cs_hist = 0.0d0
        out%Tc_hist = 0.0d0
        out%W_hist = 0.0d0
        out%k_eff_hist = 0.0d0
        out%alpha_hist = 0.0d0
        out%fsat_hist = 0.0d0
        out%qSW_hist = 0.0d0
        out%qLW_hist = 0.0d0
        out%qH_hist = 0.0d0
        out%qE_hist = 0.0d0
        out%qRAIN_hist = 0.0d0
        out%qins_hist = 0.0d0
        out%qtop_hist = 0.0d0
        out%qground_hist = 0.0d0
        out%melt_top = 0.0d0
        out%melt_mid = 0.0d0
        out%melt_bot = 0.0d0
        out%runoff_h = 0.0d0
        out%refrozen = 0.0d0

        k_s_inv   = 1.0d0 / (2.0d0 * sn%k)
        R3g_const = gr%Hg / gr%kg + 1.0d0 / gr%h_ground
        a_snow    = p%a_snow
        W_sat     = p%W_sat
        KD        = p%KD
        W_field   = p%W_field
        max_surf_flux = max(maxval(forc%G) * 1.5d0, 500.0d0)

        ! Area ratio (slope/exposure correction; 1.0 for a flat surface)
        rA = meta%rA

        T        = sn%T0
        W        = 5.0d0
        LWC_snow = 0.0d0
        ice_frac = 0.4d0
        heights  = sn%dz
        age_days = 0.0d0
        zeta     = p%zeta0

        out%T_hist(:, 1)        = T
        out%LWC_hist(:, 1)      = LWC_snow
        out%heights_hist(:, 1)  = heights
        out%ice_frac_hist(:, 1) = ice_frac
        out%W_hist(1)           = W

        Cs_layers = (ice_frac * RHO_I * C_S + LWC_snow * RHO_W * C_W) &
                    * heights
        out%Cs_hist(:, 1)       = Cs_layers

        out%Ta_hist = forc%Ta
        out%G_hist  = forc%G
        out%Pr_hist = forc%Pr

        do k = 1, Nt - 1
            if (mod(k, Nt/4) == 0) &
                print *, "  Progress: ", int(real(k)/Nt*100)+1, "%"

            Tg_k = forc%Tg(k)
            out%Tsoil_hist(k) = Tg_k

            f = min(1.0d0, max(0.0d0, W / W_sat))
            out%fsat_hist(k) = f

            if (USE_AGING_MODEL .and. p%Hi > 0.0d0) then
                call update_insulation_properties(W, age_days, p, k_eff, &
                                                  alpha_eff, tau_eff, zeta)
            else
                k_eff     = p%k_dry + (p%k_sat - p%k_dry) * (f**p%n_k)
                alpha_eff = p%alb_dry * (1.0d0 - f) + p%alb_wet * f
                tau_eff   = p%tau_dry * (1.0d0 - f) + p%tau_wet * f
            end if

            out%k_eff_hist(k) = k_eff
            out%alpha_hist(k) = alpha_eff

            Rins = max(p%Hi / k_eff, 1.0d-4)
            qSW_cov  = (1.0d0 - alpha_eff) * (1.0d0 - tau_eff) * forc%G(k)
            qSW_into = a_snow * (1.0d0 - alpha_eff) * tau_eff * forc%G(k)

            call solve_cover_temperature(forc%Ta(k), forc%U(k), &
                                         forc%RH(k), forc%Pr(k), &
                                         T(1), Rins, f, qSW_cov, p, rA, &
                                         Tc, qLW, qH, qE, qRAIN_out, qins)

            out%Tc_hist(k) = Tc

            ! Conductive flux into the snowpack (rA-scaled solar included)
            qtop = min(max(qins + rA * qSW_into, -max_surf_flux), &
                      max_surf_flux)

            ! Record rA-scaled flux histories
            out%qSW_hist(k)   = rA * qSW_cov + rA * qSW_into
            out%qLW_hist(k)   = rA * qLW
            out%qH_hist(k)    = rA * qH
            out%qE_hist(k)    = rA * qE
            out%qRAIN_hist(k) = rA * qRAIN_out
            out%qins_hist(k)  = qins
            out%qtop_hist(k)  = qtop

            R12_dyn = (heights(1) + heights(2)) * k_s_inv
            R23_dyn = (heights(2) + heights(3)) * k_s_inv
            R3g_dyn =  heights(3) * k_s_inv + R3g_const

            out%qground_hist(k) = (Tg_k - T(3)) / R3g_dyn

            Cs_layers = (ice_frac * RHO_I * C_S + LWC_snow * RHO_W * C_W) &
                        * heights
            out%Cs_hist(:, k) = Cs_layers

            call rk4_snow(T, qtop, R12_dyn, R23_dyn, R3g_dyn, Tg_k, &
                         Cs_layers, dt, Tnew)

            tot_refrozen = 0.0d0
            do i = 1, 3
                call refreezing_layer(Tnew(i), LWC_snow(i), ice_frac(i), &
                                     heights(i), &
                                     Tnew(i), LWC_snow(i), ice_frac(i), rf)
                tot_refrozen = tot_refrozen + rf
            end do
            out%refrozen(k) = tot_refrozen

            ! Melting -- all three layers use the dynamic per-layer Cs
            if (Tnew(1) > TFREEZE) then
                dE = Cs_layers(1) * (Tnew(1) - TFREEZE)
                out%melt_top(k) = dE / (RHO_I * LF)
                LWC_snow(1) = LWC_snow(1) + out%melt_top(k) / heights(1)
                Tnew(1) = TFREEZE
            end if

            if (Tnew(2) > TFREEZE) then
                dE = Cs_layers(2) * (Tnew(2) - TFREEZE)
                out%melt_mid(k) = dE / (RHO_I * LF)
                LWC_snow(2) = LWC_snow(2) + out%melt_mid(k) / heights(2)
                Tnew(2) = TFREEZE
            end if

            if (Tnew(3) > TFREEZE) then
                dE = Cs_layers(3) * (Tnew(3) - TFREEZE)
                out%melt_bot(k) = dE / (RHO_I * LF)
                LWC_snow(3) = LWC_snow(3) + out%melt_bot(k) / heights(3)
                Tnew(3) = TFREEZE
            end if

            call densification_boone(Tnew, LWC_snow, ice_frac, heights, &
                                     sn%dz, dt)
            call percolate_water(LWC_snow, heights, 3, runoff)
            out%runoff_h(k) = runoff

            if (Tc < TFREEZE) then
                L_evap = LV + LF
            else
                L_evap = LV
            end if
            E_rate = -qE / L_evap
            eta_r  = max(0.0d0, 1.0d0 - f)
            m_in   = eta_r * RHO_W * forc%Pr(k)
            D      = KD * max(0.0d0, W - W_field)
            W      = min(max(W + (m_in - E_rate - D) * dt, 0.0d0), W_sat)

            if (USE_AGING_MODEL .and. p%Hi > 0.0d0) &
                age_days = age_days + dt / 86400.0d0

            T = Tnew
            out%T_hist(:, k+1)        = T
            out%LWC_hist(:, k+1)      = LWC_snow
            out%heights_hist(:, k+1)  = heights
            out%ice_frac_hist(:, k+1) = ice_frac
            out%W_hist(k+1)           = W

            out%Cs_hist(:, k+1) = (ice_frac * RHO_I * C_S &
                                   + LWC_snow * RHO_W * C_W) * heights
        end do

        out%Tsoil_hist(Nt)   = forc%Tg(Nt)
        out%Tc_hist(Nt)      = out%Tc_hist(Nt-1)
        out%k_eff_hist(Nt)   = out%k_eff_hist(Nt-1)
        out%alpha_hist(Nt)   = out%alpha_hist(Nt-1)
        out%fsat_hist(Nt)    = out%fsat_hist(Nt-1)
        out%W_hist(Nt)       = out%W_hist(Nt-1)
        out%qSW_hist(Nt)     = out%qSW_hist(Nt-1)
        out%qLW_hist(Nt)     = out%qLW_hist(Nt-1)
        out%qH_hist(Nt)      = out%qH_hist(Nt-1)
        out%qE_hist(Nt)      = out%qE_hist(Nt-1)
        out%qRAIN_hist(Nt)   = out%qRAIN_hist(Nt-1)
        out%qins_hist(Nt)    = out%qins_hist(Nt-1)
        out%qtop_hist(Nt)    = out%qtop_hist(Nt-1)
        out%qground_hist(Nt) = out%qground_hist(Nt-1)

        out%melt_rate  = (out%melt_top + out%melt_mid + out%melt_bot) / dt
        out%melt_total = sum(out%melt_top) + sum(out%melt_mid) &
                         + sum(out%melt_bot)

        if (meta%enable_volume) then
            allocate(out%V_hist(Nt))
            out%meltout_day = ieee_value(1.0d0, ieee_quiet_nan)
            cum_melt = 0.0d0
            do k = 1, Nt
                cum_melt = cum_melt + out%melt_top(k) + out%melt_mid(k) &
                          + out%melt_bot(k)
                out%V_hist(k) = max(meta%V0 - meta%Aref &
                                    * (RHO_I / meta%rho_s) * cum_melt, 0.0d0)
                if (out%V_hist(k) <= 1.0d-6 * meta%V0 &
                    .and. ieee_is_nan(out%meltout_day)) then
                    out%meltout_day = forc%days(k)
                end if
            end do
        end if
    end subroutine run_snow_model

    ! =====================================================================
    !  LOAD CSV FORCING
    ! =====================================================================
    subroutine load_primary_forcing(csv_file, forc, sn, gr, meta, p)
    character(len=*), intent(in)    :: csv_file
    type(Forcing), intent(out)      :: forc
    type(SnowBase), intent(out)     :: sn
    type(GroundBase), intent(out)   :: gr
    type(MetaBase), intent(out)     :: meta
    type(TuningParams), intent(out) :: p

    integer, parameter :: max_lines = 100000
    integer            :: unit, ios, n_lines, i
    character(len=512) :: line
    character(len=32)  :: datetime_str

    real(real64), allocatable :: Ta_C(:), U_ms(:), G_Wm2(:), Pr_mh(:)
    real(real64), allocatable :: RH_pct(:), Tg_C(:), t_sec(:)
    real(real64) :: dt, t_total, val
    integer      :: Nt_out
    real(real64), allocatable :: t_out(:)

    p = TuningParams()
    p%solve = SolveParams()

    sn%Hs = 4.5d0; sn%Ns = 3; sn%dz = sn%Hs / dble(sn%Ns)
    sn%rho = 560.0d0; sn%c = C_S; sn%k = 0.45d0
    sn%Cs = sn%rho * sn%c * sn%dz
    sn%R12 = sn%dz / 0.45d0; sn%R23 = sn%dz / 0.45d0
    sn%T0 = [TFREEZE - 2.0d0, TFREEZE - 4.0d0, TFREEZE - 6.0d0]

    gr%Hg = 0.30d0; gr%kg = 1.5d0; gr%h_ground = 2.5d0
    gr%R3g = sn%dz / 0.45d0 + gr%Hg / gr%kg + 1.0d0 / gr%h_ground
    gr%Tg = TFREEZE + 2.0d0

    meta%enable_volume = .true.
    meta%V0 = 18000.0d0; meta%Aref = 4000.0d0
    meta%rho_s = sn%rho; meta%Hs_eff = sn%Hs
    meta%rA = 1.0d0     ! area ratio (flat surface = 1.0)

    allocate(Ta_C(max_lines), U_ms(max_lines), G_Wm2(max_lines), &
             Pr_mh(max_lines), RH_pct(max_lines), Tg_C(max_lines), &
             t_sec(max_lines))
    Ta_C = 0.0d0; U_ms = 0.0d0; G_Wm2 = 0.0d0
    Pr_mh = 0.0d0; RH_pct = 0.0d0; Tg_C = 0.0d0

    open(newunit=unit, file=trim(csv_file), status='old', action='read', &
         iostat=ios)
    if (ios /= 0) then
        print *, "ERROR: Cannot open ", trim(csv_file)
        stop
    end if

    ! Skip header
    read(unit, '(A)', iostat=ios) line
    print *, "CSV Header: ", trim(line)

    n_lines = 0
    do
        read(unit, '(A)', iostat=ios) line
        if (ios /= 0) exit
        if (len_trim(line) == 0) cycle

        n_lines = n_lines + 1
        ! Expected CSV columns:
        !   Time,Temp_C,Air_Vel_m/s_10m,Prec_m/h,Glo_Sol_Ir_W/m2,RH_%,
        !   Soil_Temp_C
        ! Time format: 2024-04-01T00:00 (no spaces)
        read(line, *, iostat=ios) datetime_str, Ta_C(n_lines), &
                                   U_ms(n_lines), Pr_mh(n_lines), &
                                   G_Wm2(n_lines), RH_pct(n_lines), &
                                   Tg_C(n_lines)
        if (ios /= 0) then
            print *, "WARNING: Could not read line ", n_lines
            print *, trim(line)
            n_lines = n_lines - 1
            cycle
        end if
        t_sec(n_lines) = dble(n_lines - 1) * 3600.0d0
    end do
    close(unit)

    print *, "Loaded ", n_lines, " lines from CSV"
    !if (n_lines >= 1) then
    !    print '(A,F8.2,A,F8.2,A,F8.3,A,F8.1,A,F8.1,A,F8.1)', &
    !          "Sample row 1: Ta=", Ta_C(1), " U=", U_ms(1), &
    !          " Pr=", Pr_mh(1), " G=", G_Wm2(1), " RH=", RH_pct(1), &
    !          " Tg=", Tg_C(1)
    !end if
    !if (n_lines >= 2) then
    !    print '(A,F8.2,A,F8.2,A,F8.3,A,F8.1,A,F8.1,A,F8.1)', &
    !          "Sample row 2: Ta=", Ta_C(2), " U=", U_ms(2), &
    !          " Pr=", Pr_mh(2), " G=", G_Wm2(2), " RH=", RH_pct(2), &
    !          " Tg=", Tg_C(2)
    !end if

    dt       = 600.0d0
    t_total  = t_sec(n_lines)
    Nt_out   = int(t_total / dt) + 1
    allocate(t_out(Nt_out))

    do i = 1, Nt_out
        t_out(i) = dble(i - 1) * dt
    end do

    forc%dt   = dt
    forc%t    = t_out
    forc%Nt   = Nt_out
    allocate(forc%days(Nt_out), forc%Ta(Nt_out), forc%U(Nt_out))
    allocate(forc%RH(Nt_out))
    allocate(forc%G(Nt_out), forc%Pr(Nt_out), forc%Tg(Nt_out))

    forc%days = t_out / 86400.0d0

    do i = 1, Nt_out
        forc%Ta(i) = interp1d(t_out(i), t_sec(1:n_lines), Ta_C(1:n_lines), &
                              n_lines) + TFREEZE

        val = interp1d(t_out(i), t_sec(1:n_lines), U_ms(1:n_lines), n_lines)
        forc%U(i) = max(val, 0.1d0)

        val = interp1d(t_out(i), t_sec(1:n_lines), RH_pct(1:n_lines), &
                       n_lines) / 100.0d0
        forc%RH(i) = min(max(val, 0.0d0), 1.0d0)

        val = interp1d(t_out(i), t_sec(1:n_lines), G_Wm2(1:n_lines), n_lines)
        forc%G(i) = max(val, 0.0d0)

        val = interp1d(t_out(i), t_sec(1:n_lines), Pr_mh(1:n_lines), &
                       n_lines) / 3600.0d0
        forc%Pr(i) = max(val, 0.0d0)

        forc%Tg(i) = interp1d(t_out(i), t_sec(1:n_lines), Tg_C(1:n_lines), &
                              n_lines) + TFREEZE
    end do

    deallocate(Ta_C, U_ms, G_Wm2, Pr_mh, RH_pct, Tg_C, t_sec)
end subroutine load_primary_forcing

    ! =====================================================================
    !  PRINT ENERGY BALANCE
    ! =====================================================================
    subroutine print_energy_balance(out, forc)
        type(Output), intent(in)   :: out
        type(Forcing), intent(in)  :: forc

        real(real64) :: E_solar, E_LW, E_H, E_evap, E_rain, E_ground, E_qtop
        real(real64) :: E_melt, E_refroz, E_dT, E_internal
        integer :: n, k

        n = forc%Nt
        E_solar  = trapz(out%qSW_hist, forc%t, n)
        E_LW     = trapz(out%qLW_hist, forc%t, n)
        E_H      = trapz(out%qH_hist,  forc%t, n)
        E_evap   = trapz(out%qE_hist,  forc%t, n)
        E_rain   = trapz(out%qRAIN_hist, forc%t, n)
        E_ground = trapz(out%qground_hist, forc%t, n)
        E_qtop   = trapz(out%qtop_hist, forc%t, n)

        E_melt   = out%melt_total * RHO_I * LF
        E_refroz = sum(out%refrozen) * LF

        E_dT = 0.0d0
        do k = 1, n - 1
            E_dT = E_dT + sum(out%Cs_hist(:, k) &
                              * (out%T_hist(:, k+1) - out%T_hist(:, k)))
        end do

        E_internal = E_dT + E_melt - E_refroz

        print *, ""
        print *, "============================================================"
        print *, "Energy Balance Diagnostics"
        print *, "============================================================"
        print *, ""
        print *, "1. COVER SURFACE FLUXES [MJ/m^2]  (informational):"
        print '(A,F10.3)', "  Solar absorbed at cover  : ", E_solar/1.0d6
        print '(A,F10.3)', "  Sensible heat (qH)       : ", E_H/1.0d6
        print '(A,F10.3)', "  Rain heat input          : ", E_rain/1.0d6
        print '(A,F10.3)', "  Ground heat input        : ", E_ground/1.0d6
        print '(A,F10.3)', "  Net longwave (qLW)       : ", E_LW/1.0d6
        print '(A,F10.3)', "  Latent heat (qE)         : ", E_evap/1.0d6
        print '(A,F10.3)', "  Conducted into snow(qtop): ", E_qtop/1.0d6

        print *, ""
        print *, "2. SNOWPACK ENERGY BUDGET [MJ/m^2]  (primary closure check):"
        print '(A,F10.3)', "  Heat input via qtop      : ", E_qtop/1.0d6
        print '(A,F10.3)', "  Heat input via ground    : ", E_ground/1.0d6
        print *, "  ------------------------------------"
        print '(A,F10.3)', "  Total heat entering snow : ", &
              (E_qtop + E_ground)/1.0d6
        print '(A,F10.3)', "  Consumed by melt         : ", E_melt/1.0d6
        print '(A,F10.3)', "  Released by refreezing   : ", -E_refroz/1.0d6
        print '(A,F10.3)', "  Sensible storage (dT)    : ", E_dT/1.0d6
        print *, "  ------------------------------------"
        print '(A,F10.3)', "  Total internal change    : ", E_internal/1.0d6
        print '(A,F10.3)', "  Residual (input-change)  : ", &
              (E_qtop + E_ground - E_internal)/1.0d6

        print *, ""
        print *, "3. MASS BALANCE:"
        print '(A,F10.3,A)', "  Total melt               : ", &
              out%melt_total, " m w.e."
        print '(A,F10.3,A)', "  Total runoff             : ", &
              sum(out%runoff_h), " kg/m^2"
        if (forc%t(n) > 0.0d0) then
            print '(A,F10.3,A)', "  Avg melt rate            : ", &
                  out%melt_total/(forc%t(n)/86400.0d0)*1000.0d0, " mm/day"
        end if
    end subroutine print_energy_balance

    ! =====================================================================
    !  DEBUG ENERGY BALANCE
    !
    !  Verbose diagnostic comparing several alternative formulations of the
    !  sensible-heat storage term and verifying mass conservation between
    !  melt, runoff, and stored liquid water.  Useful when validating a new
    !  forcing dataset or after changing the densification scheme; can be
    !  commented out of the call in the main program once a dataset is
    !  validated.
    ! =====================================================================
    subroutine debug_energy_balance(out, forc, sn)
        type(Output), intent(in)   :: out
        type(Forcing), intent(in)  :: forc
        type(SnowBase), intent(in) :: sn

        real(real64) :: E_qtop, E_ground, E_melt_calc_RHO_W
        real(real64) :: E_melt_calc_RHO_I, E_refroz
        real(real64) :: Cs_beg, Cs_end, E_dT_dynamic, E_dT_fixed, E_dT_avg
        real(real64) :: E_dT_beg, E_dT_end
        real(real64) :: internal_dynamic, residual_dynamic
        real(real64) :: internal_fixed, residual_fixed
        real(real64) :: internal_avg, residual_avg, total_melt_layer
        real(real64) :: melt_in_ice_eq
        real(real64) :: melt_water_mass, total_runoff, initial_water
        real(real64) :: stored_water, mass_balance
        real(real64) :: rho_beg(3), rho_end(3)
        integer :: n, k

        n = forc%Nt

        print *, ""
        print *, "============================================================"
        print *, "DETAILED ENERGY BALANCE DEBUG"
        print *, "============================================================"

        E_qtop   = trapz(out%qtop_hist, forc%t, n)
        E_ground = trapz(out%qground_hist, forc%t, n)

        print *, ""
        print *, "1. INPUT FLUXES [MJ/m2]:"
        print '(A,F14.6)', "   Integral qtop dt    = ", E_qtop/1.0d6
        print '(A,F14.6)', "   Integral qground dt = ", E_ground/1.0d6
        print '(A,F14.6)', "   Total input         = ", &
              (E_qtop + E_ground)/1.0d6

        E_melt_calc_RHO_W = out%melt_total * RHO_W * LF
        E_melt_calc_RHO_I = out%melt_total * RHO_I * LF

        print *, ""
        print *, "2. MELT ENERGY [MJ/m2]:"
        print '(A,F14.6,A)', "   Total melt = ", out%melt_total, " m w.e."
        print '(A,F14.6)',   "   E_melt (RHO_W*LF) = ", &
              E_melt_calc_RHO_W/1.0d6
        print '(A,F14.6)',   "   E_melt (RHO_I*LF) = ", &
              E_melt_calc_RHO_I/1.0d6
        print '(A,F14.6)',   "   Difference        = ", &
              (E_melt_calc_RHO_W - E_melt_calc_RHO_I)/1.0d6

        E_refroz = sum(out%refrozen) * LF

        print *, ""
        print *, "3. REFREEZING ENERGY [MJ/m2]:"
        print '(A,F14.6,A)', "   Total refrozen = ", sum(out%refrozen), &
              " kg/m2"
        print '(A,F14.6)',   "   E_refroz       = ", E_refroz/1.0d6

        rho_beg = out%ice_frac_hist(:, 1) * RHO_I &
                 + out%LWC_hist(:, 1) * RHO_W
        rho_end = out%ice_frac_hist(:, n) * RHO_I &
                 + out%LWC_hist(:, n) * RHO_W

        Cs_beg = sum((out%ice_frac_hist(:, 1) * RHO_I * C_S &
                     + out%LWC_hist(:, 1) * RHO_W * C_W) &
                     * out%heights_hist(:, 1))
        Cs_end = sum((out%ice_frac_hist(:, n) * RHO_I * C_S &
                     + out%LWC_hist(:, n) * RHO_W * C_W) &
                     * out%heights_hist(:, n))

        E_dT_dynamic = 0.0d0
        do k = 1, n - 1
            E_dT_dynamic = E_dT_dynamic &
                          + sum(out%Cs_hist(:, k) &
                               * (out%T_hist(:, k+1) - out%T_hist(:, k)))
        end do

        E_dT_fixed = sn%Cs * sum(out%T_hist(:, n) - out%T_hist(:, 1))
        E_dT_avg   = 0.5d0 * (Cs_beg + Cs_end) &
                    * sum(out%T_hist(:, n) - out%T_hist(:, 1))
        E_dT_beg   = Cs_beg * sum(out%T_hist(:, n) - out%T_hist(:, 1))
        E_dT_end   = Cs_end * sum(out%T_hist(:, n) - out%T_hist(:, 1))

        print *, ""
        print *, "4. SENSIBLE HEAT STORAGE:"
        print '(A,3F10.3)', "   Init temps (C): ", out%T_hist(:, 1) - TFREEZE
        print '(A,3F10.3)', "   Final temps(C): ", out%T_hist(:, n) - TFREEZE
        print '(A,3F10.3)', "   Init heights  : ", out%heights_hist(:, 1)
        print '(A,3F10.3)', "   Final heights : ", out%heights_hist(:, n)
        print '(A,3F10.1)', "   Init density  : ", rho_beg
        print '(A,3F10.1)', "   Final density : ", rho_end
        print '(A,F10.1)',  "   Init Cs (total):", Cs_beg
        print '(A,F10.1)',  "   Final Cs(total):", Cs_end
        print '(A,F10.1)',  "   Fixed sn%Cs    :", sn%Cs

        print *, ""
        print '(A,F14.6,A)', "   E_dT (DYNAMIC) : ", E_dT_dynamic/1.0d6, &
              " MJ/m2"
        print '(A,F14.6,A)', "   E_dT (fixed)   : ", E_dT_fixed/1.0d6, &
              " MJ/m2"
        print '(A,F14.6,A)', "   E_dT (average) : ", E_dT_avg/1.0d6, &
              " MJ/m2"

        internal_dynamic = E_dT_dynamic + E_melt_calc_RHO_I - E_refroz
        residual_dynamic = (E_qtop + E_ground) - internal_dynamic

        internal_fixed = E_dT_fixed + E_melt_calc_RHO_I - E_refroz
        residual_fixed = (E_qtop + E_ground) - internal_fixed

        internal_avg = E_dT_avg + E_melt_calc_RHO_I - E_refroz
        residual_avg = (E_qtop + E_ground) - internal_avg

        print *, ""
        print *, "5. BALANCE WITH E_dT ASSUMPTIONS:"
        print *, "   Method 1 (DYNAMIC step-wise Cs):"
        print '(A,F10.3,A,F10.3,A)', "     Internal = ", &
              internal_dynamic/1.0d6, ", Residual = ", &
              residual_dynamic/1.0d6, " MJ/m2"
        print *, "   Method 2 (fixed legacy Cs):"
        print '(A,F10.3,A,F10.3,A)', "     Internal = ", &
              internal_fixed/1.0d6, ", Residual = ", &
              residual_fixed/1.0d6, " MJ/m2"
        print *, "   Method 3 (average endpoint Cs):"
        print '(A,F10.3,A,F10.3,A)', "     Internal = ", &
              internal_avg/1.0d6, ", Residual = ", &
              residual_avg/1.0d6, " MJ/m2"

        total_melt_layer = sum(out%melt_top) + sum(out%melt_mid) &
                          + sum(out%melt_bot)
        melt_in_ice_eq   = total_melt_layer * RHO_I / RHO_W

        print *, ""
        print *, "6 & 7. MASS BALANCE VERIFICATION:"
        print '(A,F14.6,A)', "   Sum of layer melts = ", total_melt_layer, &
              " m"
        print '(A,F14.6,A)', "   out%melt_total     = ", out%melt_total, &
              " m w.e."

        if (abs(total_melt_layer - out%melt_total) < 1.0d-6) then
            melt_water_mass = out%melt_total * RHO_W
            print *, "   Melt is in m w.e."
        else
            melt_water_mass = total_melt_layer * RHO_I
            print *, "   Melt is in m ice eq."
        end if

        total_runoff  = sum(out%runoff_h)
        initial_water = sum(out%LWC_hist(:,1) * out%heights_hist(:,1)) &
                        * RHO_W
        stored_water  = sum(out%LWC_hist(:,n) * out%heights_hist(:,n)) &
                        * RHO_W
        mass_balance  = melt_water_mass - total_runoff &
                       - (stored_water - initial_water)

        print '(A,F10.2,A)', "   Melt water mass      = ", melt_water_mass, &
              " kg/m2"
        print '(A,F10.2,A)', "   Total runoff         = ", total_runoff, &
              " kg/m2"
        print '(A,F10.2,A)', "   Initial stored water = ", initial_water, &
              " kg/m2"
        print '(A,F10.2,A)', "   Final stored water   = ", stored_water, &
              " kg/m2"
        print '(A,F10.2,A)', "   Mass bal residual    = ", mass_balance, &
              " kg/m2"
    end subroutine debug_energy_balance

    ! =====================================================================
    !  INTERPOLATION 1D
    ! =====================================================================
    pure real(real64) function interp1d(t, x, y, n)
        real(real64), intent(in) :: t
        integer, intent(in) :: n
        real(real64), dimension(n), intent(in) :: x, y
        integer :: i
        if (t <= x(1)) then
            interp1d = y(1)
            return
        end if
        if (t >= x(n)) then
            interp1d = y(n)
            return
        end if
        do i = 1, n-1
            if (t >= x(i) .and. t <= x(i+1)) then
                interp1d = y(i) &
                          + (y(i+1) - y(i)) * (t - x(i)) / (x(i+1) - x(i))
                return
            end if
        end do
        interp1d = y(n)
    end function interp1d

    ! =====================================================================
    !  TRAPEZOIDAL INTEGRATION
    ! =====================================================================
    pure real(real64) function trapz(y, x, n)
        integer, intent(in)                     :: n
        real(real64), dimension(n), intent(in)  :: y, x
        integer :: i
        trapz = 0.0d0
        do i = 1, n - 1
            trapz = trapz + 0.5d0 * (y(i+1) + y(i)) * (x(i+1) - x(i))
        end do
    end function trapz

    ! =====================================================================
    !  PHYSICS KERNELS
    ! =====================================================================
    pure real(real64) function e_sat_scalar(T_K)
        real(real64), intent(in) :: T_K
        real(real64) :: dT
        dT = T_K - TFREEZE
        if (T_K >= TFREEZE) then
            e_sat_scalar = 611.2d0 * exp(17.27d0 * dT / (dT + 237.3d0))
        else
            e_sat_scalar = 611.2d0 * exp(22.46d0 * dT / (dT + 272.62d0))
        end if
    end function e_sat_scalar

    pure subroutine dTdt_snow(T, qtop, R12, R23, R3g, Tsoil_K, Cs, dT)
        real(real64), dimension(3), intent(in)  :: T, Cs
        real(real64), intent(in)                :: qtop, R12, R23, R3g
        real(real64), intent(in)                :: Tsoil_K
        real(real64), dimension(3), intent(out) :: dT
        real(real64) :: q12, q23, q3g

        q12 = (T(1) - T(2)) / R12
        q23 = (T(2) - T(3)) / R23
        q3g = (T(3) - Tsoil_K) / R3g

        dT(1) = (qtop - q12) / Cs(1)
        dT(2) = (q12  - q23) / Cs(2)
        dT(3) = (q23  - q3g) / Cs(3)
    end subroutine dTdt_snow

    subroutine rk4_snow(T, qtop, R12, R23, R3g, Tsoil_K, Cs, dt, Tnew)
        real(real64), dimension(3), intent(in)  :: T, Cs
        real(real64), intent(in)                :: qtop, R12, R23, R3g
        real(real64), intent(in)                :: Tsoil_K, dt
        real(real64), dimension(3), intent(out) :: Tnew
        real(real64), dimension(3) :: k1, k2, k3, k4, Tmp

        call dTdt_snow(T, qtop, R12, R23, R3g, Tsoil_K, Cs, k1)
        Tmp = T + (dt/2.0d0) * k1
        call dTdt_snow(Tmp, qtop, R12, R23, R3g, Tsoil_K, Cs, k2)
        Tmp = T + (dt/2.0d0) * k2
        call dTdt_snow(Tmp, qtop, R12, R23, R3g, Tsoil_K, Cs, k3)
        Tmp = T + dt * k3
        call dTdt_snow(Tmp, qtop, R12, R23, R3g, Tsoil_K, Cs, k4)

        Tnew = T + (dt/6.0d0) * (k1 + 2.0d0*k2 + 2.0d0*k3 + k4)
    end subroutine rk4_snow

    subroutine refreezing_layer(T_in, LWC_in, ice_frac_in, dz_s, &
                                T_out, LWC_out, ice_frac_out, refrozen_mass)
        real(real64), intent(in)  :: T_in, LWC_in, ice_frac_in, dz_s
        real(real64), intent(out) :: T_out, LWC_out, ice_frac_out
        real(real64), intent(out) :: refrozen_mass
        real(real64) :: dT_max, denom, dtheta_w_max, dtheta_w, dtheta_i, dT
        real(real64) :: heat_cap

        if (T_in >= TFREEZE .or. LWC_in <= 0.0d0) then
            T_out         = T_in
            LWC_out       = LWC_in
            ice_frac_out  = ice_frac_in
            refrozen_mass = 0.0d0
            return
        end if

        dT_max = T_in - TFREEZE
        denom  = RHO_W * (LF - dT_max * (C_S - C_W))
        heat_cap = ice_frac_in*RHO_I*C_S + LWC_in*RHO_W*C_W

        if (denom <= 0.0d0 .or. heat_cap <= 0.0d0) then
            T_out         = T_in
            LWC_out       = LWC_in
            ice_frac_out  = ice_frac_in
            refrozen_mass = 0.0d0
            return
        end if

        dtheta_w_max = -(dT_max * heat_cap) / denom
        dtheta_w     = min(LWC_in, dtheta_w_max)
        dtheta_i     = (RHO_W / RHO_I) * dtheta_w
        dT           = (dtheta_w * RHO_W * LF) / heat_cap

        T_out         = T_in + dT
        LWC_out       = LWC_in - dtheta_w
        ice_frac_out  = ice_frac_in + dtheta_i
        refrozen_mass = dtheta_w * dz_s * RHO_W
    end subroutine refreezing_layer

    subroutine percolate_water(LWC, heights, n, runoff)
        integer, intent(in)                       :: n
        real(real64), dimension(n), intent(inout) :: LWC
        real(real64), dimension(n), intent(in)    :: heights
        real(real64), intent(out)                 :: runoff
        integer      :: i
        real(real64) :: excess

        do i = 1, n - 1
            if (LWC(i) > THETA_E) then
                excess     = LWC(i) - THETA_E
                LWC(i)     = THETA_E
                LWC(i+1)   = LWC(i+1) + excess * heights(i) / heights(i+1)
            end if
        end do

        runoff = 0.0d0
        if (LWC(n) > THETA_E) then
            excess  = LWC(n) - THETA_E
            LWC(n)  = THETA_E
            runoff  = excess * heights(n) * RHO_W
        end if
    end subroutine percolate_water

    subroutine densification_boone(T_layers, LWC_layers, ice_fractions, &
                                   heights, dz_initial, dt)
        real(real64), dimension(3), intent(in)    :: T_layers
        real(real64), dimension(3), intent(inout) :: LWC_layers
        real(real64), dimension(3), intent(inout) :: ice_fractions, heights
        real(real64), intent(in)                  :: dz_initial, dt

        real(real64), parameter :: c1 = 2.8d-6, c2 = 0.042d0, c3 = 0.046d0
        real(real64), parameter :: c4 = 0.081d0, c5 = 0.018d0
        real(real64), parameter :: eta0 = 3.7d7, rho0 = 150.0d0
        real(real64) :: dz_min, M_s, rho_layer, T_C, eta, dRho, h_new
        integer :: i

        dz_min = dz_initial * 0.1d0
        M_s = 0.0d0

        do i = 1, 3
            rho_layer = ice_fractions(i) * RHO_I + LWC_layers(i) * RHO_W

            if (rho_layer < 800.0d0 .and. heights(i) > 0.01d0) then
                T_C  = T_layers(i) - TFREEZE
                eta  = eta0 * exp(c4 * (-T_C) + c5 * rho_layer)

                if (eta > 0.0d0) then
                    dRho = ((M_s * 9.81d0) / eta &
                           + c1 * exp(-c2 * (-T_C) &
                                     - c3 * max(0.0d0, rho_layer - rho0))) &
                          * dt
                else
                    dRho = c1 * exp(-c2 * (-T_C) &
                                    - c3 * max(0.0d0, rho_layer - rho0)) * dt
                end if

                dRho = min(dRho, 0.1d0)

                ice_fractions(i) = ice_fractions(i) * (1.0d0 + dRho)
                LWC_layers(i)    = LWC_layers(i) * (1.0d0 + dRho)
                h_new            = heights(i) * (1.0d0 - dRho)
                heights(i)       = max(h_new, dz_min)
            end if

            M_s = M_s + rho_layer * heights(i)
        end do
    end subroutine densification_boone

    subroutine cover_fluxes_full(Tc, Ta, U, RH, Pr, Ts1, Rins, f, &
                                 eps_c, CH, CV, CE, f_shelter, beta_w, &
                                 qLW, qH, qE, qRAIN, qins)
        real(real64), intent(in)  :: Tc, Ta, U, RH, Pr, Ts1, Rins, f
        real(real64), intent(in)  :: eps_c, CH, CV, CE, f_shelter, beta_w
        real(real64), intent(out) :: qLW, qH, qE, qRAIN, qins

        real(real64) :: e_a_Pa, e_a_kPa, eps_sky, e_c_Pa, q_a, q_star
        real(real64) :: E0, E, L

        e_a_Pa  = RH * e_sat_scalar(Ta)
        e_a_kPa = e_a_Pa / 1000.0d0
        eps_sky = min(max(1.24d0 * (e_a_kPa / Ta)**(1.0d0/7.0d0), 0.6d0), &
                     1.0d0)

        qLW = eps_sky * SIGMA * Ta**4 - eps_c * SIGMA * Tc**4
        qH = RHO_AIR * CP_AIR * CH * (1.0d0 + CV) * U * (Ta - Tc)

        e_c_Pa = e_sat_scalar(Tc)
        q_a    = 0.622d0 * e_a_Pa  / (P0 - 0.378d0 * e_a_Pa)
        q_star = 0.622d0 * e_c_Pa / (P0 - 0.378d0 * e_c_Pa)

        E0     = f_shelter * RHO_AIR * CE * U * (q_star - q_a)
        E      = E0 * exp(-beta_w * f)

        if (Tc < TFREEZE) then
            L = LV + LF
        else
            L = LV
        end if

        qE = -L * E

        if (Ta > TFREEZE .and. Pr > 0.0d0) then
            qRAIN = RHO_W * C_W * Pr * (Ta - TFREEZE)
        else
            qRAIN = 0.0d0
        end if

        qins = (Tc - Ts1) / Rins
    end subroutine cover_fluxes_full

    real(real64) function bisect_seb(lo, hi, Ta, Ts1, Rins, f, qSW_cov, &
                                     eps_sky_Ta4, qH_coeff, q_a, &
                                     CE_U_shelter, qRAIN, eps_c, beta_w, rA)
        use ieee_arithmetic, only: ieee_is_finite
        real(real64), intent(in) :: lo, hi, Ta, Ts1, Rins, f, qSW_cov
        real(real64), intent(in) :: eps_sky_Ta4, qH_coeff, q_a, CE_U_shelter
        real(real64), intent(in) :: qRAIN, eps_c, beta_w, rA

        real(real64), parameter :: tol = 1.0d-6
        integer, parameter      :: max_iter = 100
        real(real64) :: a, b, fa, fb, mid, fmid
        integer      :: iter

        a = lo
        b = hi
        call seb_residual(a, Ta, Ts1, Rins, f, qSW_cov, eps_sky_Ta4, &
                          qH_coeff, q_a, CE_U_shelter, qRAIN, eps_c, &
                          beta_w, rA, fa)
        call seb_residual(b, Ta, Ts1, Rins, f, qSW_cov, eps_sky_Ta4, &
                          qH_coeff, q_a, CE_U_shelter, qRAIN, eps_c, &
                          beta_w, rA, fb)

        if (.not. (ieee_is_finite(fa) .and. ieee_is_finite(fb))) then
            bisect_seb = 0.5d0 * (a + b)
            return
        end if
        if (fa * fb > 0.0d0) then
            bisect_seb = 0.5d0 * (a + b)
            return
        end if

        do iter = 1, max_iter
            mid = 0.5d0 * (a + b)
            call seb_residual(mid, Ta, Ts1, Rins, f, qSW_cov, eps_sky_Ta4, &
                              qH_coeff, q_a, CE_U_shelter, qRAIN, eps_c, &
                              beta_w, rA, fmid)
            if (.not. ieee_is_finite(fmid)) exit
            if (abs(fmid) < tol .or. abs(b - a) < tol) then
                bisect_seb = mid
                return
            end if
            if (fa * fmid < 0.0d0) then
                b  = mid
                fb = fmid
            else
                a  = mid
                fa = fmid
            end if
        end do
        bisect_seb = 0.5d0 * (a + b)
    end function bisect_seb

    ! SEB residual: rA scales the incoming radiative/convective/rain
    ! fluxes relative to the conductive flux through the insulation,
    ! where rA is the slope/exposure area ratio (1.0 for a flat surface).
    subroutine seb_residual(Tc, Ta, Ts1, Rins, f, qSW_cov, &
                            eps_sky_Ta4, qH_coeff, q_a, CE_U_shelter, &
                            qRAIN, eps_c, beta_w, rA, res)
        real(real64), intent(in)  :: Tc, Ta, Ts1, Rins, f, qSW_cov
        real(real64), intent(in)  :: eps_sky_Ta4, qH_coeff, q_a, CE_U_shelter
        real(real64), intent(in)  :: qRAIN, eps_c, beta_w, rA
        real(real64), intent(out) :: res

        real(real64) :: qLW, qH, e_c_Pa, q_star, E, L, qE, qins

        qLW = eps_sky_Ta4 - eps_c * SIGMA * Tc**4
        qH  = qH_coeff * (Ta - Tc)
        e_c_Pa = e_sat_scalar(Tc)
        q_star = 0.622d0 * e_c_Pa / (P0 - 0.378d0 * e_c_Pa)
        E      = CE_U_shelter * (q_star - q_a) * exp(-beta_w * f)

        if (Tc < TFREEZE) then
            L = LV + LF
        else
            L = LV
        end if

        qE  = -L * E
        qins = (Tc - Ts1) / Rins
        res = rA * (qSW_cov + qLW + qH + qE + qRAIN) - qins
    end subroutine seb_residual

    subroutine solve_cover_temperature(Ta, U, RH, Pr, Ts1, Rins, f, &
                                       qSW_cov, p, rA, &
                                       Tc, qLW, qH, qE, qRAIN_out, qins)
        use ieee_arithmetic, only: ieee_is_finite
        real(real64), intent(in)     :: Ta, U, RH, Pr, Ts1, Rins, f, qSW_cov
        type(TuningParams), intent(in) :: p
        real(real64), intent(in)     :: rA
        real(real64), intent(out)    :: Tc, qLW, qH, qE, qRAIN_out, qins

        real(real64) :: e_a_Pa, e_a_kPa, eps_sky, eps_sky_Ta4, qH_coeff
        real(real64) :: q_a, CE_U_shelter, qRAIN
        real(real64) :: Tc_min, Tc_max, lo, hi, Flo, Fhi, new_lo, new_hi
        integer      :: ntry, idx_min(1)
        real(real64), dimension(25)  :: Tc_grid, Fg

        e_a_Pa      = RH * e_sat_scalar(Ta)
        e_a_kPa     = e_a_Pa / 1000.0d0
        eps_sky     = min(max(1.24d0 * (e_a_kPa / Ta)**(1.0d0/7.0d0), &
                             0.6d0), 1.0d0)
        eps_sky_Ta4 = eps_sky * SIGMA * Ta**4
        qH_coeff = RHO_AIR * CP_AIR * p%CH * (1.0d0 + p%CV) * U
        q_a = 0.622d0 * e_a_Pa / (P0 - 0.378d0 * e_a_Pa)
        CE_U_shelter = p%f_shelter * RHO_AIR * p%CE * U

        if (Ta > TFREEZE .and. Pr > 0.0d0) then
            qRAIN = RHO_W * C_W * Pr * (Ta - TFREEZE)
        else
            qRAIN = 0.0d0
        end if

        Tc_min = TFREEZE + p%solve%Tc_min_C
        Tc_max = min(TFREEZE + p%solve%Tc_max_C, Ta + 2.0d0)
        lo = max(Ta - p%solve%bracket_dT_lo, Tc_min)
        hi = min(Ta + p%solve%bracket_dT_hi, Tc_max)

        call seb_residual(lo, Ta, Ts1, Rins, f, qSW_cov, eps_sky_Ta4, &
                          qH_coeff, q_a, CE_U_shelter, qRAIN, p%eps_c, &
                          p%beta_w, rA, Flo)
        call seb_residual(hi, Ta, Ts1, Rins, f, qSW_cov, eps_sky_Ta4, &
                          qH_coeff, q_a, CE_U_shelter, qRAIN, p%eps_c, &
                          p%beta_w, rA, Fhi)

        do ntry = 1, p%solve%max_expand
            if (ieee_is_finite(Flo) .and. ieee_is_finite(Fhi) &
                .and. Flo * Fhi < 0.0d0) exit
            new_lo = max(Ta - p%solve%bracket_dT_lo &
                        * (p%solve%expand_factor**ntry), Tc_min)
            new_hi = min(Ta + p%solve%bracket_dT_hi &
                        * (p%solve%expand_factor**ntry), Tc_max)
            if (abs(new_lo - lo) > epsilon(1.0d0)) then
                lo = new_lo
                call seb_residual(lo, Ta, Ts1, Rins, f, qSW_cov, &
                                  eps_sky_Ta4, qH_coeff, q_a, &
                                  CE_U_shelter, qRAIN, p%eps_c, p%beta_w, &
                                  rA, Flo)
            end if
            if (abs(new_hi - hi) > epsilon(1.0d0)) then
                hi = new_hi
                call seb_residual(hi, Ta, Ts1, Rins, f, qSW_cov, &
                                  eps_sky_Ta4, qH_coeff, q_a, &
                                  CE_U_shelter, qRAIN, p%eps_c, p%beta_w, &
                                  rA, Fhi)
            end if
        end do

        if (ieee_is_finite(Flo) .and. ieee_is_finite(Fhi) &
            .and. Flo * Fhi < 0.0d0) then
            Tc = bisect_seb(lo, hi, Ta, Ts1, Rins, f, qSW_cov, &
                            eps_sky_Ta4, qH_coeff, q_a, CE_U_shelter, &
                            qRAIN, p%eps_c, p%beta_w, rA)
        else
            call linspace(lo, hi, 25, Tc_grid)
            do ntry = 1, 25
                call seb_residual(Tc_grid(ntry), Ta, Ts1, Rins, f, &
                                  qSW_cov, eps_sky_Ta4, qH_coeff, q_a, &
                                  CE_U_shelter, qRAIN, p%eps_c, p%beta_w, &
                                  rA, Fg(ntry))
            end do
            idx_min = minloc(abs(Fg))
            Tc = Tc_grid(idx_min(1))
        end if

        call cover_fluxes_full(Tc, Ta, U, RH, Pr, Ts1, Rins, f, p%eps_c, &
                               p%CH, p%CV, p%CE, p%f_shelter, p%beta_w, &
                               qLW, qH, qE, qRAIN_out, qins)
    end subroutine solve_cover_temperature

    subroutine update_insulation_properties(W, age_days, p, k_eff, &
                                            alpha_eff, tau_eff, zeta)
        real(real64), intent(in)     :: W, age_days
        type(TuningParams), intent(in) :: p
        real(real64), intent(out)    :: k_eff, alpha_eff, tau_eff
        real(real64), intent(inout)  :: zeta

        real(real64) :: f, age_yr, k_moist, porosity_factor, k_age_factor
        real(real64) :: alpha_moist, tau_moist

        f      = min(1.0d0, max(0.0d0, W / p%W_sat))
        age_yr = age_days / 365.0d0
        k_moist = p%k_dry + (p%k_sat - p%k_dry) * (f**p%n_k)

        porosity_factor = zeta * exp(-p%gamma_H * p%Hi) &
                          * exp(-p%gamma_W * f)
        k_age_factor    = 1.0d0 + p%delta_k_age &
                          * (1.0d0 - exp(-age_yr / p%tau_k_years))
        k_eff = k_moist * (1.0d0 + porosity_factor) * k_age_factor
        alpha_moist = p%alb_dry + (p%alb_wet - p%alb_dry) * (f**p%beta_w)
        alpha_eff = min(max(alpha_moist + p%delta_alpha_age &
                            * (1.0d0 - exp(-age_yr / p%tau_alpha_years)), &
                            0.0d0), 1.0d0)
        tau_moist = p%tau_dry + (p%tau_wet - p%tau_dry) * (f**p%beta_w)
        tau_eff   = min(max(tau_moist, 0.0d0), 1.0d0)

        zeta = p%zeta0 * exp(-p%gamma_H * p%Hi) * exp(-p%gamma_W * f)
    end subroutine update_insulation_properties

    subroutine linspace(a, b, n, arr)
        real(real64), intent(in)               :: a, b
        integer, intent(in)                    :: n
        real(real64), dimension(n), intent(out):: arr
        integer :: i
        if (n == 1) then
            arr(1) = a
        else
            do i = 1, n
                arr(i) = a + (b - a) * dble(i - 1) / dble(n - 1)
            end do
        end if
    end subroutine linspace

end program main