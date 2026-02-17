import math
import random


def kinematics(m1: float, m2: float, m3: float, m4: float, exc3: float, eb: float, t_lab: float):
    """
    2-body relativistic kinematics.

    Inputs:
      m1,m2,m3,m4 : masses in GeV/c^2
      exc3        : excitation energy of particle 3 in GeV
      eb          : beam kinetic energy in GeV
      t_lab       : LAB polar angle of particle 3 (recoil) in rad

    Returns:
      dict with:
        "solutions": list of dicts, each solution has:
            {"E3_GeV": ..., "theta3_rad": ..., "E4_GeV": ..., "theta4_rad": ...}
        "theta_max_rad": float or None
    """
    m3 = m3 + exc3

    # --- beam in lab
    w2l = eb + m2
    g2l = w2l / m2
    b2l = (1.0 - 1.0 / (g2l * g2l)) ** 0.5
    p2l = g2l * b2l * m2

    w1l = m1
    wtl = w1l + w2l

    # --- CM boost
    b = p2l / (w2l + m1)
    g = 1.0 / (1.0 - b * b) ** 0.5

    wtc = wtl / g

    # --- threshold check
    if wtc < (m3 + m4):
        return {"solutions": [], "theta_max_rad": None}

    # --- CM energies/momenta
    w3c = (wtc * wtc + m3 * m3 - m4 * m4) / (2 * wtc)
    p3c = (w3c * w3c - m3 * m3) ** 0.5
    g3c = w3c / m3
    b3c = (1.0 - 1.0 / (g3c * g3c)) ** 0.5
    k3c = b / b3c

    w4c = (wtc * wtc + m4 * m4 - m3 * m3) / (2 * wtc)
    p4c = (w4c * w4c - m4 * m4) ** 0.5
    if m4 == 0:
        b4c = 1.0
    else:
        g4c = w4c / m4
        b4c = (1.0 - 1.0 / (g4c * g4c)) ** 0.5
    k4c = b / b4c

    # --- solve for CM angle(s) that correspond to the given lab angle t_lab of particle 3
    t3l = t_lab

    xx = (m1 - m4) * (m1 + m4) + (m2 - m3) * (2 * m1 + m2 - m3) + 2 * eb * (m1 - m3)
    aa = math.cos(t3l) * math.cos(t3l) + g * g * math.sin(t3l) * math.sin(t3l)
    bb = (g * math.sin(t3l)) ** 2 * 2.0 * k3c
    cc = 4.0 * math.cos(t3l) * math.cos(t3l) * (
        math.cos(t3l) * math.cos(t3l)
        + g * g * math.sin(t3l) * math.sin(t3l) * xx * (g3c / g + 1.0)
        / ((g3c * g3c - 1.0) * (2.0 * m3 * wtl))
    )

    solutions = []
    tmax = None

    # numerical guard (can go slightly negative due to precision)
    disc = cc
    if disc < 0 and disc > -1e-12:
        disc = 0.0

    if xx > 0:
        # theta_max doesn't exist; only 1 physical solution
        if math.cos(t3l) >= 0.0:
            t3c = math.acos((-bb + math.sqrt(disc)) / (2.0 * aa))
        else:
            t3c = math.acos((-bb - math.sqrt(disc)) / (2.0 * aa))

        e3l, t4l, e4l = cm2lab(m3, t3c, w3c, b3c, m4, w4c, b4c, k4c, g, b)
        solutions.append({
            "E3_GeV": e3l, "theta3_rad": t3l,
            "E4_GeV": e4l, "theta4_rad": t4l
        })

    else:
        # theta_max exists; potentially 2 solutions
        xa = -g * xx * (g3c + g) / (2.0 * m3 * wtl * (g3c * g3c - 1.0))
        tmax = math.acos((xa / (xa + 1.0)) ** 0.5)

        if t3l <= tmax:
            t3c = math.acos((-bb - math.sqrt(disc)) / (2.0 * aa))
            e3l, t4l, e4l = cm2lab(m3, t3c, w3c, b3c, m4, w4c, b4c, k4c, g, b)
            solutions.append({
                "E3_GeV": e3l, "theta3_rad": t3l,
                "E4_GeV": e4l, "theta4_rad": t4l
            })

            t3c = math.acos((-bb + math.sqrt(disc)) / (2.0 * aa))
            e3l, t4l, e4l = cm2lab(m3, t3c, w3c, b3c, m4, w4c, b4c, k4c, g, b)
            solutions.append({
                "E3_GeV": e3l, "theta3_rad": t3l,
                "E4_GeV": e4l, "theta4_rad": t4l
            })

    return {"solutions": solutions, "theta_max_rad": tmax}



def cm2lab(m3, t3c, w3c, b3c, m4, w4c, b4c, k4c, g, b):
    t4c = -math.pi + t3c
    t4l = math.atan(math.sin(t4c) / (g * (math.cos(t4c) + k4c)))
    if t4c * t4l < 0.0:
        t4l = t4l - math.pi
    t4l = abs(t4l) 
    w3l = w3c * g * (1.0 + b * b3c * math.cos(t3c))
    e3l = w3l - m3
    w4l = w4c * g * (1.0 + b * b4c * math.cos(t4c))
    e4l = w4l - m4
    solution = [e3l, t4l, e4l]
    return solution


def cm2lab_theta(t3c, m1, m2, m3, m4, exc3, eb):
    m3 = m3 + exc3
    w2l = eb + m2
    g2l = w2l / m2
    b2l = (1.0 - 1.0 / (g2l * g2l)) ** 0.5
    p2l = g2l * b2l * m2
    w1l = m1
    wtl = w1l + w2l
    b = p2l / (w2l + m1)
    g = 1.0 / (1.0 - b * b) ** 0.5
    wtc = wtl / g
    e3l = 0
    if wtc < (m3 + m4):
        return e3l
    elif wtc > (m3 + m4):
        w3c = (wtc * wtc + m3 * m3 - m4 * m4) / (2 * wtc)
        g3c = w3c / m3
        b3c = (1.0 - 1.0 / (g3c * g3c)) ** 0.5
    w3l = w3c * g * (1.0 + b * b3c * math.cos(t3c))
    t3l = math.asin( math.sin(t3c) * ((w3c**2-m3**2)/(w3l**2 - m3**2))**0.5)
    return t3l


def generate_random_theta_com():
    rs = 0
    while rs > 1 or rs < 0.01:
        x = random.uniform(-1,1)
        y = random.uniform(-1,1)
        z = random.uniform(-1,1)
        rs = (x**2 + y**2 + z**2)**0.5
    el = (abs(x**2 + y**2))**0.5
    if abs(el) < 1e-9: el += 3e-9
    if abs(z) < 1e-9: z += 3e-9
    th = math.acos(z/rs)
    return th


def get_brho(energy, mass, charge):
    total_energy = energy*1e-3 + mass
    pc = (total_energy**2 - mass**2)**0.5
    return 3.33564*pc/charge
    

def get_energy_from_brho(brho, mass, charge):
    pc = brho*charge/3.33564
    total_energy = (pc**2 + mass**2)**0.5
    energy = total_energy - mass
    return energy


def get_vel(energy, mass):
    c = 29.9792 # cm/ns
    total_energy = energy * 1e-3 + mass
    return c*(1-(mass/total_energy)**2)**0.5

