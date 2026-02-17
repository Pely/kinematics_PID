from SRIM_to_data import process_file
from scipy.interpolate import interp1d
import pandas as pd
import numpy as np
from os.path import exists
from generate_dedx_output import generate_dedx_output
import matplotlib.pyplot as plt

# Read text file containing energy and range
# For the input energy, find the range
# Subtract material thickness from range
# Find the energy corresponding to the remaining range


def calculate_Eout(ion:str, input_energy, material:str, thickness, lib):

    if lib == 'SRIM':
        PATH_TO_FILE = './SRIM_outputs/{}_{}.txt'.format(ion, material)
        if not(exists(PATH_TO_FILE)):
            generate_dedx_output(lib, ion, material)
            input('********* Are you sure you created the asked file? Press enter if yes. ********* \n')
        if exists(PATH_TO_FILE):
            df = process_file(PATH_TO_FILE)

    elif lib == 'VICAR':
        PATH_TO_FILE = './VICAR_outputs/{}_{}.txt'.format(ion, material)
        if not(exists(PATH_TO_FILE)):
            generate_dedx_output(lib, ion, material)
        if exists(PATH_TO_FILE):
            df = pd.read_csv(PATH_TO_FILE, sep="\t", names=['ionE MeV', 'range um'])

    else:
        raise Exception("No library path found")

    # energy = df['ionE MeV'].to_numpy()
    # range = df['range um'].to_numpy()

    energy = np.array(df['ionE MeV'])
    range = np.array(df['range um'])

    spline1 = interp1d(energy, range, kind='cubic')
    spline2 = interp1d(range, energy, kind='cubic')
    if input_energy < min(energy):
        range_in = 0
    else: range_in = spline1(input_energy)
    # print(ion, material, input_energy, range_in)
    range_out = range_in - thickness
    if range_out < min(range):
        energy_out = 0
    else: energy_out = spline2(range_out)

    return energy_out


