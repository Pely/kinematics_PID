class Reaction:
    def __init__(self, status, beam_ion, recoil_ion, m1:float, m2:float, m3:float, m4:float, exc3, eb):
        self.status = status
        self.beam = beam_ion
        self.recoil = recoil_ion
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.m4 = m4
        self.exc3 = exc3
        self.eb = eb


class Jet:
    def __init__(self, status, pressure, thickness, material):
        self.status = status
        self.pressure = pressure
        self.thickness = thickness
        self.material = material


class StripFoil:
    def __init__(self, status, thickness, material):
        self.status = status
        self.thickness = thickness
        self.material = material


class MCPs:
    def __init__(self, status, thickness, material):
        self.status = status
        self.thickness = thickness
        self.material = material


class IC:
    def __init__(self, window_thickness, window_material, gas_status, pressure, dl1_thickness, de_thickness,
                 dl2_thickness, gas_material):
        self.window_thickness = window_thickness
        self.window_material = window_material
        self.gas_status = gas_status
        self.pressure = pressure
        self.dl1_thickness = dl1_thickness
        self.de_thickness = de_thickness
        self.dl2_thickness = dl2_thickness
        self.gas_material = gas_material


class DSSD:
    def __init__(self, dl_thickness, dl_material, det_thickness, det_material):
        self.dl_thickness = dl_thickness
        self.dl_material = dl_material
        self.det_thickness = det_thickness
        self.det_material = det_material