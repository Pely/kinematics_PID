import numpy as np

U_TO_MEV = 931.49410242  # 1 u = 931.494... MeV/c^2 (natural units c=1)

def kallen(x, y, z):
    return x*x + y*y + y*y + z*z - 2*x*y - 2*x*z - 2*y*z

def boost_fourvec(E, p, beta):
    """
    Boost 4-vector (E, p[3]) by velocity beta[3] (|beta|<1).
    Returns (E', p').
    """
    b2 = float(np.dot(beta, beta))
    if b2 < 1e-30:
        return float(E), p.copy()
    if b2 >= 1.0:
        raise ValueError("Superluminal beta encountered.")
    gamma = 1.0 / np.sqrt(1.0 - b2)
    bp = float(np.dot(beta, p))
    Eprime = gamma * (E + bp)
    pprime = p + ((gamma - 1.0) / b2) * bp * beta + gamma * E * beta
    return float(Eprime), pprime

def p_from_E_m(E, m):
    arg = E*E - m*m
    return np.sqrt(np.maximum(arg, 0.0))

def theta_lab_rad(pvec):
    pmag = float(np.linalg.norm(pvec))
    if pmag <= 0.0:
        return np.nan
    return float(np.arccos(np.clip(pvec[2] / pmag, -1.0, 1.0)))

def phi_lab_rad(pvec):
    """Lab azimuth angle φ in radians, in (-π, π]."""
    return float(np.arctan2(pvec[1], pvec[0]))

def isotropic_unit(rng):
    """Random isotropic unit vector."""
    u = rng.uniform(-1.0, 1.0)
    phi = rng.uniform(0.0, 2.0*np.pi)
    s = np.sqrt(1.0 - u*u)
    return np.array([s*np.cos(phi), s*np.sin(phi), u], dtype=float)

def generate_3body_events(
    Ebeam_MeV,
    Nevents,
    masses_u,
    beam_name,
    target_name,
    recoil_name,
    rng=None,
    return_neutrons=True
):
    """
    Flat-matrix-element 3-body phase space sampler for:
        beam + target -> recoil + n + n

    Inputs:
      - Ebeam_MeV: projectile kinetic energy in lab (MeV) at reaction point
      - masses_u: dict of atomic masses in u, must include beam_name, target_name, recoil_name, and 'n'
      - Names: beam_name, target_name, recoil_name
      - return_neutrons: if False, returns only recoil kinematics.

    Output: list of events, each event is:
      {
        "recoil": {"E": T_recoil_MeV, "theta": theta_lab_rad},
        "n1": {"E": T_n1_MeV, "theta": theta_lab_rad},
        "n2": {"E": T_n2_MeV, "theta": theta_lab_rad},
      }
      Energies are kinetic energies in MeV, angles are lab polar angles in radians.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Masses in MeV
    m_a = masses_u[beam_name]   * U_TO_MEV
    m_A = masses_u[target_name] * U_TO_MEV
    m_b = masses_u[recoil_name] * U_TO_MEV
    m_n = masses_u["n"]         * U_TO_MEV

    # Projectile in lab (along +z)
    E_a = m_a + float(Ebeam_MeV)
    p_a = p_from_E_m(E_a, m_a)
    Ptot_E = E_a + m_A
    Ptot_p = np.array([0.0, 0.0, p_a], dtype=float)

    s = Ptot_E*Ptot_E - float(np.dot(Ptot_p, Ptot_p))
    if s <= 0.0:
        raise RuntimeError("Non-physical s encountered.")
    W = np.sqrt(s)

    threshold = m_b + 2.0*m_n
    if W < threshold:
        raise RuntimeError(f"Below threshold: W={W:.6f} < m_b+2m_n={threshold:.6f} MeV")

    beta_cm = Ptot_p / Ptot_E  # lab->CM boost velocity

    # Treat nn as intermediate system X with mass mX
    mX_min = 2.0*m_n
    mX_max = W - m_b
    if mX_max <= mX_min:
        raise RuntimeError("No available phase space: mX_max <= mX_min")

    def pstar_cm(mX):
        lam = (s - (m_b + mX)**2) * (s - (m_b - mX)**2)
        return np.sqrt(np.maximum(lam, 0.0)) / (2.0 * W)

    def q_nn_in_X(mX):
        En = 0.5 * mX
        return np.sqrt(np.maximum(En*En - m_n*m_n, 0.0))

    # Precompute rejection max
    grid = np.linspace(mX_min, mX_max, 2000)
    w_grid = pstar_cm(grid) * q_nn_in_X(grid)
    w_max = float(np.max(w_grid))
    if w_max <= 0.0:
        raise RuntimeError("w_max <= 0: check masses/threshold.")

    events = []

    for _ in range(int(Nevents)):
        # Rejection sample mX with weight ~ p* q (flat ME)
        while True:
            mX = rng.uniform(mX_min, mX_max)
            w = pstar_cm(mX) * q_nn_in_X(mX)
            if rng.uniform(0.0, w_max) < w:
                break

        # In CM: b and X are back-to-back
        p_b = pstar_cm(mX)
        E_b = np.sqrt(m_b*m_b + p_b*p_b)
        E_X = np.sqrt(mX*mX + p_b*p_b)

        n_b = isotropic_unit(rng)
        p_b_vec = p_b * n_b
        p_X_vec = -p_b_vec

        # In X rest frame: n and n back-to-back with magnitude q
        q = q_nn_in_X(mX)
        E_n = 0.5*mX  # each neutron energy in X frame

        n_n1 = isotropic_unit(rng)
        p_n1_X = q * n_n1
        p_n2_X = -p_n1_X

        # Boost neutrons from X rest frame to CM frame
        beta_X = p_X_vec / E_X  # CM velocity of X

        E_n1_cm, p_n1_cm = boost_fourvec(E_n, p_n1_X, beta_X)
        E_n2_cm, p_n2_cm = boost_fourvec(E_n, p_n2_X, beta_X)

        # Now boost b and neutrons from CM to lab
        E_b_lab,  p_b_lab  = boost_fourvec(E_b,   p_b_vec,  beta_cm)
        E_n1_lab, p_n1_lab = boost_fourvec(E_n1_cm, p_n1_cm, beta_cm)
        E_n2_lab, p_n2_lab = boost_fourvec(E_n2_cm, p_n2_cm, beta_cm)

        # Convert to kinetic energies
        T_b = E_b_lab - m_b
        T_n1 = E_n1_lab - m_n
        T_n2 = E_n2_lab - m_n

        ev = {
            "recoil": {"E": float(T_b), "theta": theta_lab_rad(p_b_lab), "phi": phi_lab_rad(p_b_lab)}
        }
        if return_neutrons:
            ev["n1"] = {"E": float(T_n1), "theta": theta_lab_rad(p_n1_lab), "phi": phi_lab_rad(p_n1_lab)}
            ev["n2"] = {"E": float(T_n2), "theta": theta_lab_rad(p_n2_lab), "phi": phi_lab_rad(p_n2_lab)}
        events.append(ev)

    return events
