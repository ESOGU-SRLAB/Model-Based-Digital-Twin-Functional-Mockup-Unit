import math
from fmi2 import Fmi2FMU, Fmi2Status

class SinusGenerator(Fmi2FMU):
    def __init__(self, reference_to_attr=None):
        super().__init__()
        # Eğer dışarıdan bir mapping verilmişse, onu kullan; aksi takdirde varsayılan değeri kullan.
        if reference_to_attr is None:
            self.reference_to_attr = {
                0: "amplitude",
                1: "frequency",
                2: "sin_out"
            }
        else:
            self.reference_to_attr = reference_to_attr

        # Parametreler: amplitude ve frequency, başlangıç değerleri
        self.amplitude = 1.0
        self.frequency = 1.0
        # Çıktı: sinüs sinyali
        self.sin_out = 0.0
        # Simülasyon zamanı
        self.t = 0.0

    def instantiate(self, instanceName, resourceLocation):
        return Fmi2Status.ok

    def setup_experiment(self, startTime, stopTime, tolerance):
        self.t = startTime
        return Fmi2Status.ok

    def enter_initialization_mode(self):
        return Fmi2Status.ok

    def exit_initialization_mode(self):
        return Fmi2Status.ok

    def do_step(self, currentTime, stepSize, noSetFMUStatePriorToCurrentPoint):
        # Zamanı güncelle
        self.t = currentTime + stepSize
        # Sinüs sinyal hesapla
        self.sin_out = self.amplitude * math.sin(2 * math.pi * self.frequency * self.t)
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
    return SinusGenerator()
