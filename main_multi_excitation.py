import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from tqdm import trange
from get_mass import get_m
from setup_classes import *
from get_pid import *
from kinematics import *              # expects kinematics() to return dict with 'solutions'
from kinematics_3body import generate_3body_events

if __name__ == '__main__':

    # ============================================================
    # INPUT: Instrument / detector setup
    # ============================================================
    jet = Jet(1, '240psi', 2500, 'helium')
    jet.material = jet.material + '_' + jet.pressure  # must match stopping-power tables

    mcp = MCPs(0, 0.5, 'mylar')
    stripfoil = StripFoil(1, 0.23008, 'carbon')

    ic = IC(3.0, 'mylar', 1, '50T', 12700, 50800, 3200, 'isobutane')
    dssd = DSSD(0.5, 'aluminium', 300, 'silicon')

    # ============================================================
    # INPUT: Reaction setup
    # ============================================================
    target   = 'He4'
    beam     = 'Kr86'
    recoil   = 'Sr89'
    ejectile = 'n1'     # n2 -> 3body, otherwise 2body

    beam_q   = 26
    recoil_q = 28

    beam_energy_u = 2.9863  # MeV/u

    # --- NEW: run multiple excitation energies (MeV) and accumulate
    # Put any list you want here (e.g. [0.0, 0.5, 1.0, 2.0])
    excE_list_MeV = [0.0]

    dedx_lib = 'SRIM'  # or 'VICAR'

    # Decide kinematics mode automatically
    kinematics_mode = "3body" if ejectile == 'n2' else "2body"

    # ============================================================
    # INPUT: Plotting & Monte-Carlo parameters
    # ============================================================
    n_rec = 1000
    E_range = [0, 180]
    delE_range = [0, 100]

    # ============================================================
    # SETUP: masses and reaction object template
    # ============================================================
    m_list = [get_m(target), get_m(beam), get_m(recoil), get_m(ejectile)]  # micro-amu
    for i in range(4):
        m_list[i] = m_list[i] * 931.494 * 1e-9  # GeV/c^2

    beam_energy_GeV = beam_energy_u * get_m(beam) * 1e-9  # GeV (MeV/u * micro-amu * 1e-9)
    rxn = Reaction(1, beam, recoil, m_list[0], m_list[1], m_list[2], m_list[3], 0.0, beam_energy_GeV)

    if dedx_lib == 'VICAR':
        os.system('cd ./VICAR_dedx/ && g++ stp.cxx && cd ../')

    rng = np.random.default_rng(12345)

    # ============================================================
    # ACCUMULATED ARRAYS (across all excitation energies)
    # ============================================================
    # PID
    delE_recoil_all, E_recoil_all = [], []
    delE_beam, E_beam = [], []

    # Heavy recoil kinematics / transport
    theta_recoil_prod_mrad, E_recoil_prod_MeV = [], []
    theta_recoil_after_strip_mrad, E_recoil_after_strip_MeV = [], []

    # Light ejectile (production only)
    theta_light_rad, E_light_MeV, light_label = [], [], []  # label is n1/n2/ejectile

    # CSV rows
    recoil_csv_rows = []   # recoil AFTER stripper foil
    light_csv_rows  = []   # light ejectile(s) AT PRODUCTION

    # ============================================================
    # MAIN LOOP OVER EXCITATION ENERGIES
    # ============================================================
    for excE_MeV in excE_list_MeV:

        rxn.exc3 = excE_MeV * 1e-3  # GeV

        for i_evt in trange(n_rec, desc=f"{kinematics_mode} | Ex={excE_MeV:.3f} MeV", unit="evt"):

            # ----------------------------
            # Sample random reaction depth
            # ----------------------------
            depth_fraction = rng.uniform(0.0, 1.0)
            rxn_distance = jet.thickness * depth_fraction

            # Beam energy at reaction point (MeV)
            Ebeam_rxn_MeV = calculate_Eout(rxn.beam, rxn.eb * 1000.0, jet.material, rxn_distance, dedx_lib)

            # ----------------------------
            # KINEMATICS AT PRODUCTION
            # ----------------------------
            recoil_solutions = []  # list of (Erec_MeV, theta_rec_rad, phi_rec_rad)

            if kinematics_mode == "3body":

                # Include excitation energy in recoil mass (amu) for 3-body generator
                recoil_mass_u_exc = get_m(recoil) * 1e-6 + (excE_MeV / 931.494)

                masses_u = {
                    beam:   get_m(beam)   * 1e-6,
                    target: get_m(target) * 1e-6,
                    recoil: recoil_mass_u_exc,
                    "n":    1.00866491588
                }

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

                # heavy recoil at production
                Erec_prod = ev["recoil"]["E"]          # MeV
                th_rec    = ev["recoil"]["theta"]      # rad
                phi_rec   = ev["recoil"]["phi"]        # rad
                recoil_solutions.append((Erec_prod, th_rec, phi_rec))

                # light ejectiles (neutrons) at production
                if "n1" in ev:
                    theta_light_rad.append(abs(ev["n1"]["theta"]))
                    E_light_MeV.append(ev["n1"]["E"])
                    light_label.append("n1")
                    light_csv_rows.append({
                        "excE_MeV": float(excE_MeV),
                        "depth_fraction": float(depth_fraction),
                        "Ebeam_rxn_MeV": float(Ebeam_rxn_MeV),
                        "particle": "n1",
                        "E_MeV": float(ev["n1"]["E"]),
                        "theta_rad": float(abs(ev["n1"]["theta"])),
                        "theta_deg": float(np.degrees(abs(ev["n1"]["theta"])))
                    })
                if "n2" in ev:
                    theta_light_rad.append(abs(ev["n2"]["theta"]))
                    E_light_MeV.append(ev["n2"]["E"])
                    light_label.append("n2")
                    light_csv_rows.append({
                        "excE_MeV": float(excE_MeV),
                        "depth_fraction": float(depth_fraction),
                        "Ebeam_rxn_MeV": float(Ebeam_rxn_MeV),
                        "particle": "n2",
                        "E_MeV": float(ev["n2"]["E"]),
                        "theta_rad": float(abs(ev["n2"]["theta"])),
                        "theta_deg": float(np.degrees(abs(ev["n2"]["theta"])))
                    })

            else:
                # --- 2-body kinematics (returns heavy+light per physical branch)
                th_com = generate_random_theta_com()
                th_rec = cm2lab_theta(th_com, m_list[0], m_list[1], m_list[2], m_list[3], rxn.exc3, Ebeam_rxn_MeV/1000.0)

                ev = kinematics(rxn.m1, rxn.m2, rxn.m3, rxn.m4, rxn.exc3, Ebeam_rxn_MeV/1000.0, th_rec)

                for sol in ev["solutions"]:
                    Erec_prod = sol["E3_GeV"] * 1e3     # MeV
                    phi_rec   = rng.uniform(0.0, 2.0 * np.pi)

                    recoil_solutions.append((Erec_prod, th_rec, phi_rec))

                    # light ejectile at production
                    Eeject_MeV = sol["E4_GeV"] * 1e3
                    th_ej_rad  = abs(sol["theta4_rad"])

                    theta_light_rad.append(th_ej_rad)
                    E_light_MeV.append(Eeject_MeV)
                    light_label.append(ejectile)

                    light_csv_rows.append({
                        "excE_MeV": float(excE_MeV),
                        "depth_fraction": float(depth_fraction),
                        "Ebeam_rxn_MeV": float(Ebeam_rxn_MeV),
                        "particle": str(ejectile),
                        "E_MeV": float(Eeject_MeV),
                        "theta_rad": float(th_ej_rad),
                        "theta_deg": float(np.degrees(th_ej_rad))
                    })

            # ----------------------------
            # TRANSPORT: remaining target + stripper foil
            # ----------------------------
            for Erec_prod, th_rec, phi_rec in recoil_solutions:

                # production distributions (for kinematics plots)
                theta_recoil_prod_mrad.append(th_rec * 1e3)
                E_recoil_prod_MeV.append(Erec_prod)

                # energy loss through remaining target
                Erec_after_target = calculate_Eout(
                    rxn.recoil,
                    Erec_prod,
                    jet.material,
                    jet.thickness - rxn_distance,
                    dedx_lib
                )

                # energy loss through stripper foil
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

                # after-strip distributions (requested)
                theta_recoil_after_strip_mrad.append(th_rec * 1e3)
                E_recoil_after_strip_MeV.append(Erec_after_strip)

                # recoil-after-strip CSV
                recoil_csv_rows.append({
                    "excE_MeV": float(excE_MeV),
                    "depth_fraction": float(depth_fraction),
                    "Ebeam_rxn_MeV": float(Ebeam_rxn_MeV),
                    "E_recoil_after_strip_MeV": float(Erec_after_strip),
                    "theta_recoil_mrad": float(th_rec * 1e3),
                    "phi_recoil_rad": float(phi_rec),
                })

                # PID point for this recoil
                pid_rec = get_pid(rxn.recoil, Erec_after_strip, mcp, ic, dssd, dedx_lib)
                delE_recoil_all.append(pid_rec[0])
                E_recoil_all.append(pid_rec[1])

    # ============================================================
    # BEAM AFTER FULL TARGET (+ STRIPPER)  (single point)
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
    # SAVE CSV OUTPUTS
    # ============================================================
    df_recoil = pd.DataFrame(recoil_csv_rows)
    recoil_csv_name = f"{recoil}_recoils_after_strip_multiEx.csv"
    df_recoil.to_csv(recoil_csv_name, index=False)
    print("Saved recoil-after-strip CSV:", recoil_csv_name)

    df_light = pd.DataFrame(light_csv_rows)
    light_csv_name = f"{recoil}_{ejectile}_light_ejectiles_production_multiEx.csv"
    df_light.to_csv(light_csv_name, index=False)
    print("Saved light-ejectile production CSV:", light_csv_name)

    # PID save (for re-plotting later)
    np.savez(
        f"pid_data_{recoil}_E{beam_energy_u:.3f}_{beam}_multiEx.npz",
        E_recoil=np.asarray(E_recoil_all),
        dE_recoil=np.asarray(delE_recoil_all),
        E_beam=np.asarray(E_beam),
        dE_beam=np.asarray(delE_beam)
    )

    # ============================================================
    # PLOTTING (cumulative over all excitation energies)
    # ============================================================
    ms = 10

    # --- PID
    figPID, axPID = plt.subplots(figsize=(8, 6))
    figPID.suptitle(
        f'PID cumulative *** {beam}({target},{ejectile}){recoil} *** Ebeam={beam_energy_u:.3f} MeV/u *** dedx={dedx_lib}',
        fontname="Times New Roman",
        fontweight="bold",
        fontsize=14
    )
    axPID.scatter(E_recoil_all, delE_recoil_all, s=ms, label=f'{recoil} recoils (all Ex)')
    axPID.scatter(E_beam, delE_beam, s=ms*3, marker='x', label='Unreacted beam')
    axPID.set_xlabel("DSSD E (MeV)")
    axPID.set_ylabel("IC dE (MeV)")
    axPID.set_xlim(*E_range)
    axPID.set_ylim(*delE_range)
    axPID.grid(True, alpha=0.3)
    axPID.legend(fontsize=9)
    plt.tight_layout()
    figPID.savefig(f"{recoil}_pid_multiEx.png", dpi=300, bbox_inches="tight")

    # --- Kinematics at production: heavy recoil
    figK, (axR, axL) = plt.subplots(2, figsize=(8, 8), sharex=False)
    axR.scatter(theta_recoil_prod_mrad, E_recoil_prod_MeV, s=2)
    axR.set_title("Heavy recoil at production (all Ex)")
    axR.set_xlabel(r"$\theta_{recoil}$ (mrad)")
    axR.set_ylabel("E (MeV)")
    axR.grid(True, alpha=0.3)

    # --- Light ejectile(s) at production
    axL.scatter(np.degrees(theta_light_rad), E_light_MeV, s=2)
    axL.set_title("Light ejectile(s) at production (all Ex)")
    axL.set_xlabel(r"$\theta_{light}$ (deg)")
    axL.set_ylabel("E (MeV)")
    axL.grid(True, alpha=0.3)

    plt.tight_layout()
    figK.savefig(f"{recoil}_kinematics_production_multiEx.png", dpi=300, bbox_inches="tight")

    # --- Heavy recoil after stripper foil (requested)
    figStrip, axStrip = plt.subplots(figsize=(8, 6))
    figStrip.suptitle(f"{recoil} recoil after stripper foil (all Ex)", fontname="Times New Roman", fontweight="bold", fontsize=14)
    axStrip.scatter(theta_recoil_after_strip_mrad, E_recoil_after_strip_MeV, s=2)
    axStrip.set_xlabel(r"$\theta_{recoil}$ (mrad)")
    axStrip.set_ylabel("E after strip (MeV)")
    axStrip.grid(True, alpha=0.3)
    plt.tight_layout()
    figStrip.savefig(f"{recoil}_recoil_after_strip_multiEx.png", dpi=300, bbox_inches="tight")

    plt.show()
