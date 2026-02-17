from Eout import calculate_Eout
from kinematics import kinematics

def e_after_JENSA_with_reaction_depth(rxn, theta_lab, jet, stripfoil, dedx_lib, depth_fraction):
    #Wrapper to evaluate recoil energy entering SECAR for a reaction happening at a given depth in the target/jet.
    try:
        return e_after_JENSA(rxn, theta_lab, jet, stripfoil, dedx_lib, depth_fraction=depth_fraction)
    except TypeError:
        pass
    return e_after_JENSA(rxn, theta_lab, jet, stripfoil, dedx_lib)

def e_after_JENSA(reaction, theta, jet, stripfoil, dedx_lib):
    ion = reaction.beam
    e_ion_after_jet_list = []
    e_ion_after_stripfoil_list = []
    if jet.status:
        jet.material = 'helium_' + jet.pressure
        if reaction.status:
            rxn_distance = jet.thickness / 2
            e_ion_at_rxn = calculate_Eout(ion, reaction.eb * 1000, jet.material, rxn_distance, dedx_lib)
            ion = reaction.recoil
            reacKin = kinematics(reaction.m1, reaction.m2, reaction.m3,
                                         reaction.m4, reaction.exc3, e_ion_at_rxn / 1000, theta)
            # kinematics function returns an array of recoil energies (GeV) and theta_max
            reacKin.pop() # Remove theta_max from the array
            for e_ion_after_rxn in reacKin:
                e_ion_after_jet_list.append(calculate_Eout(ion, e_ion_after_rxn * 1000, jet.material, jet.thickness - rxn_distance, dedx_lib))
        else:
            e_ion_after_jet_list.append(calculate_Eout(ion, reaction.eb * 1000, jet.material, jet.thickness, dedx_lib))
    else:
        e_ion_after_jet_list.append(reaction.eb * 1000)

    for e_ion_after_jet in e_ion_after_jet_list:
        if stripfoil.status:
            e_ion_after_stripfoil_list.append(calculate_Eout(ion, e_ion_after_jet, stripfoil.material, stripfoil.thickness, dedx_lib))
        else:
            e_ion_after_stripfoil_list.append(e_ion_after_jet)

    return e_ion_after_stripfoil_list


def get_pid(ion, e_ion_after_stripfoil, mcp, ic, dssd, dedx_lib):

    if mcp.status:
        e_ion_after_mcp = calculate_Eout(ion, e_ion_after_stripfoil, mcp.material, mcp.thickness, dedx_lib)
    else:
        e_ion_after_mcp = e_ion_after_stripfoil

    e_ion_after_ic_window = calculate_Eout(ion, e_ion_after_mcp, ic.window_material, ic.window_thickness, dedx_lib)
    if ic.gas_status:
        ic.gas_material = 'isobutane_' + ic.pressure
        e_ion_after_ic_dl1 = calculate_Eout(ion, e_ion_after_ic_window, ic.gas_material, ic.dl1_thickness, dedx_lib)
        e_ion_after_ic_de = calculate_Eout(ion, e_ion_after_ic_dl1, ic.gas_material, ic.de_thickness, dedx_lib)
        e_ion_after_ic_dl2 = calculate_Eout(ion, e_ion_after_ic_de, ic.gas_material, ic.dl2_thickness, dedx_lib)
        delE = e_ion_after_ic_dl1 - e_ion_after_ic_de
    else:
        e_ion_after_ic_dl2 = e_ion_after_ic_window
        delE = 0

    e_ion_after_dssd_dl = calculate_Eout(ion, e_ion_after_ic_dl2, dssd.dl_material, dssd.dl_thickness, dedx_lib)
    e_ion_after_dssd = calculate_Eout(ion, e_ion_after_dssd_dl, dssd.det_material, dssd.det_thickness, dedx_lib)
    E = e_ion_after_dssd_dl - e_ion_after_dssd

    pid = [delE, E]

    # print(e_ion_after_jet, e_ion_after_mcp, e_ion_after_ic_window, e_ion_after_ic_dl1, e_ion_after_ic_de, e_ion_after_ic_dl2, e_ion_after_dssd_dl, e_ion_after_dssd)

    return pid