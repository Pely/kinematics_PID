import re

def materials(material:str):
    density = 0
    average_a = 0
    average_z = 0

    material_input = re.findall('(\d+|[A-Za-z]+)', material)

    if len(material_input) < 3:
        if material_input[0] == 'carbon':
            density = 2.253E+00 # g/cm3
            average_a = 12
            average_z = 6

        if material_input[0] == 'mylar':
            density = 1.3970E+00 # g/cm3
            average_a = (12*10 + 1*8 + 16*4)/(10 + 8 + 4) # C10H8O4
            average_z = (6*10 + 1*8 + 8*4)/(10 + 8 + 4)

        elif material_input[0] == 'silicon':
            density = 2.3212E+00 #g/cm3
            average_a = 28
            average_z = 14

        elif material_input[0] == 'aluminium':
            density = 2.7020E+00 #g/cm3
            average_a = 26
            average_z = 13

    elif len(material_input) == 3:
        if material_input[0] == 'isobutane':
            # density formula claculated from LISE values b/w 25 and 50 torrs
            density = float(material_input[1])*3.18e-6 - 8.41e-8 
            average_a = (12*4 + 1*10)/(4 + 10) # C4H10
            average_z = (6*4 + 1*10)/(4 + 10)
        
        if material_input[0] == 'helium':
            # 150 psi = 3.19e18 at/cm2
            # width = 2.5mm.
            # 150 psi = 3.19e18/0.25 at/cm3.
            # 1 at/cm3 = 4/N_a g/cm3
            # 1 psi = 3.19e18*4 / (0.25*150*N_a) g/cm3
            density = float(material_input[1])*3.19e18*4 / (0.25*150*6.022e23) # g/cm3
            average_a = 4
            average_z = 2

    return [density, average_a, average_z]