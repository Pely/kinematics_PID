import os
import pandas as pd
import re
from materials import materials
from get_mass import get_m


def generate_dedx_output(lib, ion:str, material:str):

    target_material = materials(material)
    den_tar = target_material[0]
    avg_a_tar = target_material[1]
    avg_z_tar = target_material[2]

    filename = ''
    if lib == 'SRIM':
        filename = './SRIM_outputs/{}_{}.txt'.format(ion, material)
        print('Generate and copy SRIM stopping power table with:')
        print('\t name = {}'.format(filename))
        print('\t ion mass = {} amu'.format(1e-6*get_m(ion)))
        print('\t energy range = 10 keV to 900 MeV')
        print('\t target density = {} g/cm3'.format(den_tar))
        input('********* Press enter when you are done ********* \n')
    else:
        filename = './VICAR_outputs/{}_{}.txt'.format(ion, material)

        df_ele = pd.read_csv("elements.txt", sep="\t", names=["z","el"])

        recoil = re.findall('(\d+|[A-Za-z]+)', ion)
        el_rec = recoil[0]
        a_rec = recoil[1]

        z_rec = df_ele['z'][df_ele['el'] == el_rec].values[0]

        command = './VICAR_dedx/a.out {} {} {} {} {}'.format(den_tar, avg_a_tar, avg_z_tar, a_rec, z_rec)
        os.system(command)

        mv_command = 'mv ./VICAR_outputs/temp.txt {}'.format(filename)
        os.system(mv_command)