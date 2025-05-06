from fmi2 import Fmi2FMU, Fmi2Status

class Integrator(Fmi2FMU):
    def __init__(self, reference_to_attr=None):
        super().__init__()
        # Giriş: entegre edilecek sinyal
        self.in_signal = 0.0
        # Çıktı: entegre edilmiş değer
        self.integrated_out = 0.0
        # İçsel state: entegrasyon sonucu (başlangıçta 0)
        self.x = 0.0

        # Value reference eşlemesi:
        # Input: 0: in_signal
        # Output: 1: integrated_out
        if reference_to_attr is None:
            self.reference_to_attr = {
                0: "in_signal",
                1: "integrated_out"
            }
        else:
            self.reference_to_attr = reference_to_attr

    def instantiate(self, instanceName, resourceLocation):
        return Fmi2Status.ok

    def setup_experiment(self, startTime, stopTime, tolerance):
        self.x = 0.0
        return Fmi2Status.ok

    def enter_initialization_mode(self):
        return Fmi2Status.ok

    def exit_initialization_mode(self):
        return Fmi2Status.ok

    def do_step(self, currentTime, stepSize, noSetFMUStatePriorToCurrentPoint):
        # Euler entegrasyon yöntemiyle state güncellemesi
        self.x = self.x + self.in_signal * stepSize
        self.integrated_out = self.x
        return Fmi2Status.ok

    def get_real(self, refs):
        outvals = []
        for ref in refs:
            attr = self.reference_to_attr[ref]
            outvals.append(getattr(self, attr))
        return outvals, Fmi2Status.ok

    def set_real(self, refs, values):
        for ref, val in zip(refs, values):
            attr = self.reference_to_attr[ref]
            setattr(self, attr, val)
        return Fmi2Status.ok

def create_fmu_instance():
    return Integrator()
