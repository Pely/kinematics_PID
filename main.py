import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from tqdm import trange
from get_mass import get_m
from setup_classes import *
from get_pid import *
from kinematics import *
from kinematics_3body import generate_3body_events

if __name__ == '__main__':

    # ============================================================
    # INPUT: Instrument / detector setup
    # ============================================================
    # These objects define the materials the ions traverse
    # and are used later for energy-loss calculations (dE/dx)
    
    jet = Jet(1, '240psi', 2500, 'helium')
    jet.material = jet.material+ '_' + jet.pressure # Always ensure jet material string matches stopping-power tables
    # status, pressure, thickness (um), material
    # 2.5mm 150 = (8.48e-5g/cm3 3.19x10^18 at/cm2)

    mcp = MCPs(0, 0.5, 'mylar')
    # status, thickness (um), material

    stripfoil = StripFoil(1, 0.23008, 'carbon')
    # status, thickness (um), material
    # 0.2 um = 45 ug/cm2
    # 0.2308 um = 52 ug/cm2

    ic = IC(3.0, 'mylar', 1, '50T', 12700, 50800, 3200, 'isobutane')
    # ic = IC(3.0, 'mylar', 1, '50T', 12700, 54060, 3200, 'isobutane')
    # window thickness (um), window material, gas status, gas pressure,
    # dl1 thickness (um), de thickness, dl2 thickness, gas material

    dssd = DSSD(0.5, 'aluminium', 300, 'silicon')
    # dl thickness, dl material (um), det thickness, det material   

    # ============================================================
    # INPUT: Reaction setup
    # ============================================================
    # Naming follows SECAR convention:
    # beam + target → heavy recoil + light ejectile(s)
    
    target = 'He4' # particle1
    beam = 'Kr86' # particle2
    recoil = 'Sr89' # particle3. Traditional kinematics would call it ejectile. But in SECAR this is recoil
    ejectile = 'n1' # particle4. n0 for gamma, nX for X neutrons up to X=3. # n2 → 3-body, n1 → 2-body
    beam_q = 26
    recoil_q = 28
    beam_energy_u = 2.9863 # MeV/u
    excE_list_MeV = [0.0, 1.0, 5.0, 8.0] # Put any list of excitation energies (MeV) here (e.g. [0.0, 0.5, 1.0, 2.0])
    tuning_energy = 2.6731  # MeV/u 2.6833 88 , 2.6731 89
    dedx_lib = 'SRIM'
    # dedx_lib = 'VICAR' # this option requires user to calculate stopping power tables for ion in each material
    
    # Decide kinematics mode automatically
    kinematics_mode = "3body" if ejectile == 'n2' else "2body"

    kinematics_check_plot = False
    energy_offset_plot = True
    
    # ============================================================
    # INPUT: Plotting & Monte-Carlo parameters
    # ============================================================
    n_rec = 100 # number of recoil particles generated in Monte-Carlo (theta_com and theta_lab)
    E_range = [0, 180] # DSSD energy axis range in MeV
    delE_range = [0, 100] # IC_dE energy axis range in MeV
    
    # Currently not used in this version
    # leaky_b_range = [0.5, 1.0] # Calculate leaky beam PID for beam energy fraction in this range.
    # leaky_b_step = 0.01
    # +/- Leaky beam brho acceptance as a fraction of the recoil Brho value
    # brho_accept = 0.03
    # +/- Leaky beam velocity acceptance as fraction of the recoil velocity value
    # vel_accept = 0.03

    # ======== End of user input =================#


    # ============================================================
    # CALCULATIONS: reaction masses and objects
    # ============================================================
    # Compile VICAR stopping if requested
    if dedx_lib == 'VICAR':
        os.system('cd ./VICAR_dedx/ && g++ stp.cxx && cd ../')
        
    # Masses of ions in GeV/c2 for kinematics routines
    m_list = [get_m(target), get_m(beam), get_m(recoil), get_m(ejectile)]  # micro-amu
    for i in range(4):
        m_list[i] = m_list[i] * 931.494 * 1e-9  # GeV/c2
    # Beam energy in GeV for kinematics routines
    beam_energy = beam_energy_u*get_m(beam)*10**-9  # GeV
    
    # ============================================================
    # Diagnostic: beam energy at front / middle / back of target
    # Reaction at ground state
    # ============================================================
    if(kinematics_check_plot):
        recoil_depths = {'front': 0.0, 'middle': 0.5, 'back': 1.0}
        
        excE_GeV = excE_list_MeV[0]*10**-3
        rxn = Reaction(1, beam, recoil, m_list[0], m_list[1], m_list[2], m_list[3], excE_GeV, beam_energy)
        # status, beam, recoil, m1 (GeV/c2), m2, m3, m4, exc3, eb (GeV) 

        def beam_energy_at_depth_MeV(depth_fraction):
            rxn_distance = jet.thickness * float(depth_fraction)
            return calculate_Eout(rxn.beam, rxn.eb * 1000.0, jet.material, rxn_distance, dedx_lib)

        Eb_depth_MeV = {k: beam_energy_at_depth_MeV(frac) for k, frac in recoil_depths.items()}


    # ============================================================
    # MONTE CARLO LOOP
    # ============================================================
    # Physics sequence per event:
    #   1) sample reaction depth
    #   2) beam energy loss up to reaction
    #   3) kinematics (2-body or 3-body)
    #   4) recoil energy loss through remaining target + stripper
    #   5) store distributions + CSV output
    #   6) PID calculation

    rng = np.random.default_rng(12345)

    # PID arrays
    delE_recoil_all = []
    E_recoil_all = []

    # Beam after full target (+stripfoil) PID (single point)
    delE_beam = []
    E_beam = []

    # Distributions for plots + CSV
    theta_recoil_prod_mrad = []
    E_recoil_prod_MeV = []

    theta_recoil_after_strip_mrad = []
    E_recoil_after_strip_MeV = []

    # Neutron kinematics (production only)
    theta_n1, E_n1_MeV = [], []
    theta_n2, E_n2_MeV = [], []

    theta_ejec, E_ejec = [], []
    
    # CSV rows
    recoil_csv_rows = []   # recoil AFTER stripper foil
    light_csv_rows  = []   # light ejectile(s) AT PRODUCTION
    
    for excE_MeV in excE_list_MeV:

        excE_GeV = excE_MeV * 1e-3  # GeV 
        rxn = Reaction(1, beam, recoil, m_list[0], m_list[1], m_list[2], m_list[3], excE_GeV, beam_energy)
        # status, beam, recoil, m1 (GeV/c2), m2, m3, m4, exc3, eb (GeV) 

        for i_evt in trange(n_rec, desc=f"{kinematics_mode} kinematics | events", unit="evt"):
            
            # --- sample random reaction position in target
            depth_fraction = rng.uniform(0.0, 1.0)
            rxn_distance = jet.thickness * depth_fraction

            # Beam energy at reaction point (MeV) after pre-reaction energy loss
            Ebeam_rxn_MeV = calculate_Eout(rxn.beam, rxn.eb * 1000.0, jet.material, rxn_distance, dedx_lib)

            # ========================================================
            # KINEMATICS AT PRODUCTION
            # ========================================================
            recoil_solutions = []
            
            if kinematics_mode == "3body":
                
                # Masses in atomic units for 3-body generator
                masses_u = {
                    beam:   get_m(beam)   * 1e-6,
                    target: get_m(target) * 1e-6,
                    recoil: get_m(recoil) * 1e-6,
                    "n":    1.00866491588
                }
                # Generate one 3-body event at this updated beam energy
                ev = generate_3body_events(
                    Ebeam_MeV=Ebeam_rxn_MeV,
                    Nevents=1,
                    masses_u=masses_u,
                    beam_name=beam,
                    target_name=target,
                    recoil_name=recoil,
                    rng=rng,
                    return_neutrons=True
                )[0]
                # --- recoil at production (kinematics)
                Erec_prod = ev["recoil"]["E"]                 # MeV (kinetic)
                th_rec = ev["recoil"]["theta"]                # rad
                phi_rec = ev["recoil"]["phi"]
                recoil_solutions.append((Erec_prod, th_rec, phi_rec))
                
                # ========================================================
                # NEUTRON KINEMATICS (production only)
                # ========================================================
                if "n1" in ev:
                    theta_n1.append(ev["n1"]["theta"]) #rad
                    E_n1_MeV.append(ev["n1"]["E"])
                    light_csv_rows.append({
                        "E_MeV": float(ev["n1"]["E"]),
                        "theta_deg": float(np.degrees(abs(ev["n1"]["theta"])))
                    })
                if "n2" in ev:
                    theta_n2.append(ev["n2"]["theta"])
                    E_n2_MeV.append(ev["n2"]["E"])
                    light_csv_rows.append({
                        "E_MeV": float(ev["n2"]["E"]),
                        "theta_deg": float(np.degrees(abs(ev["n2"]["theta"])))
                    })

            else:
                # --- standard 2-body kinematics (two solutions)
                th_com = generate_random_theta_com()
                th_rec = cm2lab_theta(th_com, m_list[0], m_list[1], m_list[2], m_list[3], rxn.exc3, Ebeam_rxn_MeV/1000)
                ev = kinematics(rxn.m1, rxn.m2, rxn.m3, rxn.m4, rxn.exc3, Ebeam_rxn_MeV/1000, th_rec)
                
                for sol in ev["solutions"]:
                    
                    # Heavy recoil solutions
                    Erec_GeV = sol["E3_GeV"]
                    Erec_prod = Erec_GeV * 1e3
                    phi_rec = np.degrees(rng.uniform(0.0, 2.0 * np.pi))
                    recoil_solutions.append((Erec_prod, th_rec, phi_rec))

                    # Light ejectile solutions
                    Eeject_GeV = sol["E4_GeV"]
                    th_eject_rad = sol["theta4_rad"]
                    Eeject_MeV = Eeject_GeV * 1e3
                    theta_eject_deg = np.rad2deg(th_eject_rad)
                    theta_ejec.append(theta_eject_deg)
                    E_ejec.append(Eeject_MeV)
                    
                    light_csv_rows.append({
                        "E_MeV": float(Eeject_MeV),
                        "theta_deg": float(theta_eject_deg)
                    })
            
            # ========================================================
            # TRANSPORT: remaining target + stripper foil
            # ========================================================
            for Erec_prod, th_rec, phi_rec in recoil_solutions:
                
                theta_recoil_prod_mrad.append(th_rec * 1e3)
                E_recoil_prod_MeV.append(Erec_prod)
                
                # transport through remaining target
                Erec_after_target = calculate_Eout(
                    rxn.recoil,
                    Erec_prod,
                    jet.material,
                    jet.thickness - rxn_distance,
                    dedx_lib)

                # transport through stripper foil
                if stripfoil.status:
                    Erec_after_strip = calculate_Eout(
                        rxn.recoil,
                        Erec_after_target,
                        stripfoil.material,
                        stripfoil.thickness,
                        dedx_lib
                    )
                else:
                    Erec_after_strip = Erec_after_target

                theta_recoil_after_strip_mrad.append(th_rec * 1e3)
                E_recoil_after_strip_MeV.append(Erec_after_strip)

                # ========================================================
                # CSV OUTPUT (recoil after stripper foil)
                # ========================================================
                recoil_csv_rows.append({
                    "E_recoil_after_strip_MeV": float(Erec_after_strip),
                    "theta_recoil_mrad": float(th_rec * 1e3),
                    "phi_recoil_rad": float(phi_rec),
                })
                
                # ========================================================
                # PID CALCULATION (IC dE vs DSSD E)
                # ========================================================
                pid_rec = get_pid(rxn.recoil, Erec_after_strip, mcp, ic, dssd, dedx_lib)
                delE_recoil_all.append(pid_rec[0])
                E_recoil_all.append(pid_rec[1])

    # ============================================================
    # BEAM AFTER FULL TARGET (+ STRIPPER)
    # ============================================================
    Ebeam_after_target = calculate_Eout(rxn.beam, rxn.eb * 1000.0, jet.material, jet.thickness, dedx_lib)
    if stripfoil.status:
        Ebeam_after_strip = calculate_Eout(rxn.beam, Ebeam_after_target, stripfoil.material, stripfoil.thickness, dedx_lib)
    else:
        Ebeam_after_strip = Ebeam_after_target

    pid_beam = get_pid(rxn.beam, Ebeam_after_strip, mcp, ic, dssd, dedx_lib)
    delE_beam.append(pid_beam[0])
    E_beam.append(pid_beam[1])

    # ============================================================
    # SAVE CSV
    # ============================================================
    # --- Save recoil distribution after stripper foil
    df_recoil = pd.DataFrame(recoil_csv_rows)
    recoil_csv_name = f"{recoil}_recoils_after_strip.csv"
    df_recoil.to_csv(recoil_csv_name, index=False)
    print("Saved recoil-after-strip CSV:", recoil_csv_name)
    
    df_light = pd.DataFrame(light_csv_rows)
    light_csv_name = f"{recoil}_{ejectile}_production.csv"
    df_light.to_csv(light_csv_name, index=False)
    print("Saved light-ejectile production CSV:", light_csv_name)

    # ============================================================
    # SAVE PID DATA (for re-plotting later)
    # ============================================================
    np.savez(
        f"pid_data_{recoil}_E{beam_energy_u:.3f}_{beam}.npz",
        E_recoil = np.array(E_recoil_all),
        dE_recoil = np.array(delE_recoil_all),
        E_beam=np.asarray(E_beam),
        dE_beam=np.asarray(delE_beam)
    )

    # ============================================================
    # PLOTTING
    # ============================================================  

    ms = 10  # marker size
    # --- PID plot: recoils + beam after target
    figPID, axPID = plt.subplots(figsize=(8, 6))
    figPID.suptitle(
        'PID *** {}({},{}){} *** Ebeam = {:.3f} MeV/u *** dedx = {}'.format(
            beam, target, ejectile, recoil, beam_energy_u, dedx_lib
        ),
        fontname="Times New Roman",
        fontweight="bold",
        fontsize=14
    )

    axPID.scatter(E_recoil_all, delE_recoil_all, s=ms, label=f'{recoil} recoils')
    axPID.scatter(E_beam, delE_beam, s=ms*3, marker='x', label='Unreacted beam')
    axPID.set_xlabel("DSSD E (MeV)")
    axPID.set_ylabel("IC dE (MeV)")
    axPID.set_xlim(*E_range)
    axPID.set_ylim(*delE_range)
    axPID.grid(True, alpha=0.3)
    axPID.legend(fontsize=9)
    plt.tight_layout()
    figPID.savefig(f"{recoil}_pid_recoils.png", dpi=300, bbox_inches="tight")

    # --- Kinematics at production
    figK, (axR, axL) = plt.subplots(2, figsize=(8, 8), sharex=False)
    # figK.suptitle("Recoil at production", fontname="Times New Roman", fontweight="bold", fontsize=14)

    axR.scatter(theta_recoil_prod_mrad, E_recoil_prod_MeV, s=2)
    axR.set_ylabel("E (MeV)")
    axR.set_xlabel(r"$\theta_{recoil}$ (mrad)")
    axR.grid(True, alpha=0.3)
    axR.set_title("Heavy recoil at production")

    if(kinematics_mode=='3body'): 
        axL.scatter(np.degrees(theta_n1), E_n1_MeV, s=2, label="n1")
        axL.scatter(np.degrees(theta_n2), E_n2_MeV, s=2, label="n2")
    else:
        axL.scatter(theta_ejec, E_ejec, s=2)
    axL.set_title("Light ejectile at production")
    axL.set_ylabel("E (MeV)")
    axL.set_xlabel(fr"$\theta_{ejectile}$ (deg)")
    axL.grid(True, alpha=0.3)
    if(kinematics_mode=='3body'): axL.legend()

    plt.tight_layout()
    figK.savefig(f"{recoil}_kinematics_production.png", dpi=300, bbox_inches="tight")


    # --- Heavy recoil distribution AFTER stripper foil 
    figStrip, axStrip = plt.subplots(figsize=(8, 6))
    figStrip.suptitle(f"{recoil} recoil after stripper foil", fontname="Times New Roman", fontweight="bold", fontsize=14)
    axStrip.scatter(theta_recoil_after_strip_mrad, E_recoil_after_strip_MeV, s=2, c='red')
    axStrip.set_xlabel(r"$\theta_{recoil}$ (mrad)")
    axStrip.set_ylabel("E after strip (MeV)")
    axStrip.grid(True, alpha=0.3)
    plt.tight_layout()
    figStrip.savefig(f"{recoil}_kinematics_recoil_after_strip.png", dpi=300, bbox_inches="tight")

    # --- Diagnostic: show 3-body kinematics at three fixed beam energies (front/middle/back)
    if kinematics_check_plot:

        figDiag, axs = plt.subplots(3, figsize=(8, 10), sharex=True, sharey=True)
        figDiag.suptitle(
            "Recoil kinematics at E_beam@rxn (front / middle / back)",
            fontname="Times New Roman",
            fontweight="bold",
            fontsize=14
        )
        theta_all = []
        Erec_all = []
        for ax, (depth_label, Eb_MeV) in zip(axs, Eb_depth_MeV.items()):

            if kinematics_mode == "3body":
                evs = generate_3body_events(
                    Ebeam_MeV=Eb_MeV,
                    Nevents=3000,
                    masses_u=masses_u,
                    beam_name=beam,
                    target_name=target,
                    recoil_name=recoil,
                    rng=rng,
                    return_neutrons=False
                )

                th = [e["recoil"]["theta"] * 1e3 for e in evs]
                Er = [e["recoil"]["E"] for e in evs]

            else:
                th, Er = [], []
                for _ in range(3000):
                    th_com = generate_random_theta_com()
                    th_lab = cm2lab_theta(th_com, m_list[0], m_list[1], m_list[2], m_list[3], rxn.exc3, Eb_MeV / 1000.0)
                    ev = kinematics(rxn.m1, rxn.m2, rxn.m3, rxn.m4, rxn.exc3, Eb_MeV / 1000.0, th_lab)

                    for Erec_GeV in ev[:2]:
                        if not np.isfinite(Erec_GeV) or Erec_GeV <= 0:
                            continue

                        th.append(th_lab * 1e3)
                        Er.append(Erec_GeV * 1e3)

            # plot
            ax.scatter(th, Er, s=1)

            # collect global limits
            theta_all.extend(th)
            Erec_all.extend(Er)

            ax.set_title(f"{depth_label}: Ebeam@rxn = {Eb_MeV:.3f} MeV")
            ax.set_ylabel("Recoil E (MeV)")
            ax.grid(True, alpha=0.3)

        theta_all = np.array(theta_all)
        Erec_all = np.array(Erec_all)

        xmin, xmax = theta_all.min(), theta_all.max()
        ymin, ymax = Erec_all.min(), Erec_all.max()
        dx = 0.02 * (xmax - xmin)
        dy = 0.02 * (ymax - ymin)
        xmin -= dx; xmax += dx
        ymin -= dy; ymax += dy
        for ax in axs:
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            
        axs[-1].set_xlabel(r"$\theta_{recoil}$ (mrad)")

        plt.tight_layout()
        figDiag.savefig(f"{recoil}_kinematics_recoil.png",dpi=300, bbox_inches="tight")

    if(energy_offset_plot):
        tuning_energy_tot = tuning_energy * get_m(recoil)* 1e-6    #MeV
        E_recoil_after_strip_MeV = np.array(E_recoil_after_strip_MeV)   
        theta_recoil_after_strip_mrad = np.array(theta_recoil_after_strip_mrad)
        e_offset = 100 * (E_recoil_after_strip_MeV - tuning_energy_tot) / tuning_energy_tot
        
        figEoff, axEoff = plt.subplots(figsize=(8, 6))
        figEoff.suptitle(f"{recoil} recoil after stripper foil", fontname="Times New Roman", fontweight="bold", fontsize=14)
        axEoff.scatter(theta_recoil_after_strip_mrad, e_offset, s=2, c='red')
        axEoff.set_xlabel(r"$\theta_{recoil}$ (mrad)")
        axEoff.set_ylabel("E offset after strip (%)")
        axEoff.grid(True, alpha=0.3)
        plt.tight_layout()
        figEoff.savefig(f"{recoil}_ene_offset_recoil_after_strip.png", dpi=300, bbox_inches="tight")
        
    plt.show()
