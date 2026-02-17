import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from get_mass import get_m

# ============================================================
# USER SETTINGS
# ============================================================
HEAVY_CSV = "Sr89_recoils_after_strip.csv"
LIGHT_CSV = "Sr89_n1_production.csv"

# Light ejectile angle cut (DEGREES)
THETA_LIGHT_MIN_DEG = 15.0
THETA_LIGHT_MAX_DEG = 35.0

# Column names (edit if yours differ)
# --- heavy recoil after strip
HEAVY_THETA_COL = "theta_recoil_mrad"            # mrad
HEAVY_E_COL     = "E_recoil_after_strip_MeV"     # MeV

# --- light ejectile at production
# If your light theta is stored in radians, set LIGHT_THETA_IN = "rad"
# If stored in degrees, set LIGHT_THETA_IN = "deg"
LIGHT_THETA_COL = "theta_deg"          # or "theta_ejectile_deg"
LIGHT_E_COL     = "E_MeV"              # MeV
LIGHT_THETA_IN  = "deg"                         # "rad" or "deg"

# If you have event ids in both CSVs, set this to that column name
EVENT_ID_COL = None   # set to None if you don't have one

# --- energy offset plot settings
MAKE_EOFFSET_PLOT = True
TUNING_ENERGY_MEVU = 2.6731   # MeV/u (your tune)
RECOIL_NAME = "Sr89"          # used to get mass in u


def to_deg(theta_series, units):
    if units == "deg":
        return theta_series.astype(float)
    if units == "rad":
        return np.degrees(theta_series.astype(float))
    raise ValueError("units must be 'rad' or 'deg'")

def recoil_energy_offset_percent(Erec_after_strip_MeV, recoil_name, tuning_energy_MeVu):
    """
    E_offset(%) = 100 * (Erec - Etune_tot) / Etune_tot
    where Etune_tot = tuning_energy(MeV/u) * mass_u
    """
    mass_u = get_m(recoil_name) * 1e-6   # your get_m returns micro-amu
    Etune_tot = tuning_energy_MeVu * mass_u
    return 100.0 * (Erec_after_strip_MeV - Etune_tot) / Etune_tot


def main():
    heavy = pd.read_csv(HEAVY_CSV)
    light = pd.read_csv(LIGHT_CSV)

    # ---- build a merge key to associate light <-> heavy
    if EVENT_ID_COL and (EVENT_ID_COL in heavy.columns) and (EVENT_ID_COL in light.columns):
        merged = heavy.merge(light, on=EVENT_ID_COL, how="inner", suffixes=("_heavy", "_light"))
        match_mode = f"merge on column '{EVENT_ID_COL}'"
    else:
        # fallback: assume row-to-row alignment (only safe if you wrote them that way)
        heavy = heavy.reset_index().rename(columns={"index": "__rowid"})
        light = light.reset_index().rename(columns={"index": "__rowid"})
        merged = heavy.merge(light, on="__rowid", how="inner", suffixes=("_heavy", "_light"))
        match_mode = "merge by row index (__rowid) — verify this is valid for your files!"

    # ---- extract arrays
    thetaH_mrad = merged[HEAVY_THETA_COL].astype(float).to_numpy()
    EH_MeV      = merged[HEAVY_E_COL].astype(float).to_numpy()
    
    if MAKE_EOFFSET_PLOT:
        EH_arr = np.array(EH_MeV, dtype=float)
        thetaH_arr = np.array(thetaH_mrad, dtype=float)
        eoff_pct = recoil_energy_offset_percent(EH_arr, RECOIL_NAME, TUNING_ENERGY_MEVU)

    thetaL_deg = to_deg(merged[LIGHT_THETA_COL], LIGHT_THETA_IN).to_numpy()
    EL_MeV     = merged[LIGHT_E_COL].astype(float).to_numpy()

    # ---- apply light angle cut
    in_cut = (thetaL_deg >= THETA_LIGHT_MIN_DEG) & (thetaL_deg <= THETA_LIGHT_MAX_DEG)

    n_tot = len(merged)
    n_cut = int(np.count_nonzero(in_cut))
    frac  = (n_cut / n_tot) if n_tot > 0 else 0.0

    print(f"[INFO] Association mode: {match_mode}")
    print(f"[INFO] Light angle cut: {THETA_LIGHT_MIN_DEG}–{THETA_LIGHT_MAX_DEG} deg")
    print(f"[INFO] Matched events: {n_tot}")
    print(f"[INFO] Events passing light cut: {n_cut} ({frac:.3%})")

    # ============================================================
    # PLOTTING
    # ============================================================
    nrows = 3 if MAKE_EOFFSET_PLOT else 2
    fig, axs = plt.subplots(nrows, 1, figsize=(9, 11 if MAKE_EOFFSET_PLOT else 9), sharex=False)
    axH = axs[1]
    axL = axs[0]
    axOff = axs[2] if MAKE_EOFFSET_PLOT else None

    fig.suptitle("Heavy recoil (after stripper) and light ejectile (production)", fontweight="bold")
    
    # ---- Light ejectile subplot
    axL.scatter(thetaL_deg, EL_MeV, s=2, alpha=0.5, label="Light ejectile (all)")
    axL.axvspan(THETA_LIGHT_MIN_DEG, THETA_LIGHT_MAX_DEG, alpha=0.2, label="Angle cut (shaded)")
    axL.scatter(thetaL_deg[in_cut], EL_MeV[in_cut], s=4, alpha=0.9, label="Light ejectile (in cut)")
    axL.set_xlabel(r"$\theta_{\mathrm{ejectile}}$ (deg)")
    axL.set_ylabel("E at production (MeV)")
    axL.grid(True, alpha=0.3)
    axL.legend(loc="best", fontsize=9)

    # ---- Heavy recoil subplot
    axH.scatter(thetaH_mrad, EH_MeV, s=2, alpha=0.5, label="All heavy recoils")
    axH.scatter(thetaH_mrad[in_cut], EH_MeV[in_cut], s=4, alpha=0.9, label="Heavy recoils (light in shaded cut)")
    axH.set_xlabel(r"$\theta_{\mathrm{recoil}}$ (mrad)")
    axH.set_ylabel("E after strip (MeV)")
    axH.grid(True, alpha=0.3)
    axH.legend(loc="best", fontsize=9)
    
    if MAKE_EOFFSET_PLOT:
        axOff.scatter(thetaH_arr, eoff_pct, s=2, alpha=0.35, label="All heavy recoils")
        axOff.scatter(thetaH_arr[in_cut], eoff_pct[in_cut], s=4, alpha=0.9, label="Heavy recoils (light in cut)")

        axOff.set_xlabel(r"$\theta_{\mathrm{recoil}}$ (mrad)")
        axOff.set_ylabel("Energy offset after strip (%)")
        axOff.grid(True, alpha=0.3)
        axOff.legend(loc="best", fontsize=9)

        axOff.set_title(f"Energy offset vs tune ({TUNING_ENERGY_MEVU:.4f} MeV/u)")


    plt.tight_layout()
    outname = "heavy_light_kinematics_with_light_angle_cut.png"
    fig.savefig(outname, dpi=300, bbox_inches="tight")
    print(f"[INFO] Saved figure: {outname}")

    plt.show()


if __name__ == "__main__":
    main()
