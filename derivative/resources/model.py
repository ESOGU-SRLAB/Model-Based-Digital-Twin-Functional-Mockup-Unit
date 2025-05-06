from fmi2 import Fmi2FMU, Fmi2Status

class DerivativeCalculator(Fmi2FMU):
    def __init__(self, reference_to_attr=None):
        super().__init__()
        # Giriş: in_signal
        self.in_signal = 0.0
        # Çıktı: derivative_out
        self.derivative_out = 0.0
        # Önceki sinyal değeri ve önceki zaman (başlangıçta 0)
        self.prev_signal = 0.0
        self.prev_time = 0.0

        # Value reference eşlemesi:
        # Input: 0: in_signal
        # Output: 1: derivative_out
        if reference_to_attr is None:
            self.reference_to_attr = {
                0: "in_signal",
                1: "derivative_out"
            }
        self.reference_to_attr = reference_to_attr
    
    def instantiate(self, instanceName, resourceLocation):
        return Fmi2Status.ok

    def setup_experiment(self, startTime, stopTime, tolerance):
        self.prev_time = startTime
        return Fmi2Status.ok

    def enter_initialization_mode(self):
        # Başlangıçta ilk sinyal değerini sıfır kabul ediyoruz.
        self.prev_signal = self.in_signal
        return Fmi2Status.ok

    def exit_initialization_mode(self):
        return Fmi2Status.ok

    def do_step(self, currentTime, stepSize, noSetFMUStatePriorToCurrentPoint):
        # Fark oranı yöntemi: (current - previous) / dt
        dt = currentTime + stepSize - self.prev_time
        if dt == 0:
            self.derivative_out = 0.0
        else:
            self.derivative_out = (self.in_signal - self.prev_signal) / dt
        # Güncelle önceki değerler:
        self.prev_signal = self.in_signal
        self.prev_time = currentTime + stepSize
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
    return DerivativeCalculator()
