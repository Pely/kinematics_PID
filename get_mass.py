import pandas as pd
import re


def get_m(ion:str) -> float:
    df_ele = pd.read_csv("elements.txt", sep="\t", names=["z","el"])
    df_mas = pd.read_csv("masses.txt", sep="\t", names=["a", "z", "mas"])

    ion_list = re.findall('(\d+|[A-Za-z]+)', ion)
    el_ion = ion_list[0]
    a_ion = ion_list[1]

    if el_ion == 'n':
        z_ion =0
    else:
        z_ion = df_ele['z'][df_ele['el'] == el_ion].values[0]
    if a_ion == '0':
        mas_ion = 0
    else:
        mas_ion = df_mas['mas'][df_mas['a'] == a_ion][df_mas['z'] == str(z_ion)].values[0]

    return float(mas_ion)