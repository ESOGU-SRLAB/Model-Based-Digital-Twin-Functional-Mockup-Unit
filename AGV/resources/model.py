import math
import numpy as np
from fmi2 import Fmi2FMU, Fmi2Status
import yaml  # Opsiyonel: ROS YAML dosyalarından parametre okunması için

###############################################################################
# AGV FMU Parametreleri ve Ayarları
###############################################################################
DT = 0.1  # Zaman adım büyüklüğü (s)

# URDF'den alınan temel parametreler (dinamik modelleme için ileride kullanılabilir)
WHEEL_SEPARATION = 0.42      # metre
WHEEL_DIAMETER = 0.2         # metre
WHEEL_RADIUS = WHEEL_DIAMETER / 2.0
BASE_MASS = 10.0             # kg
WHEEL_MASS = 1.0             # kg

# PD kontrol için kazançlar
KP_V = 1.0       # Pozisyon hatası -> doğrusal hız kazancı
KP_OMEGA = 1.0   # Açı hatası -> açısal hız kazancı

# Maksimum doğrusal ve açısal hız (saturasyon)
V_MAX = 1.0      # m/s
OMEGA_MAX = 1.0  # rad/s

# Hata toleransları
POS_TOL = 0.05      # Hedef konuma yaklaşım eşiği
ANGLE_TOL = 0.05    # Hedef açıyı yakalama eşiği

###############################################################################
# Yardımcı Fonksiyonlar
###############################################################################
def normalize_angle(angle):
    """Açıyı [-pi, pi] aralığına çeker."""
    return (angle + math.pi) % (2 * math.pi) - math.pi

def saturate(value, limit):
    """value'yi [-limit, limit] aralığında sınırlandırır."""
    return max(min(value, limit), -limit)

###############################################################################
# FMU class for AGV with dual-mode (direct velocity or target position/orientation)
###############################################################################
class Model(Fmi2FMU):
    """
    Bir AGV (diferansiyel sürüş) modeli. İki mod içerir:
      1) Doğrudan hız modu: (des_v, des_omega) != (0, 0) verilir ve (des_x, des_y, des_theta) = (0,0,0).
      2) Hedef modu: (des_x, des_y, des_theta) hedef olarak verilir, (des_v, des_omega) = (0,0).
         Robot hedefe yaklaşınca pozisyon hatasını azaltır, ardından son aşamada açı hatasını düzeltir.
         
    FMI giriş/çıkış tanımı:
      Inputs (valueRefs):
        0: des_v       (m/s)
        1: des_omega   (rad/s)
        2: des_x       (m)
        3: des_y       (m)
        4: des_theta   (rad)
      Outputs (valueRefs):
        5: x           (m)
        6: y           (m)
        7: theta       (rad)
    """
    def __init__(self, reference_to_attr=None):
        super().__init__()
        self.reference_to_attr = reference_to_attr if reference_to_attr else {}
        
        # Robotun başlangıç konumu
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        
        # Girişler (varsayılan değerler)
        self.des_v = 0.0
        self.des_omega = 0.0
        self.des_x = 0.0
        self.des_y = 0.0
        self.des_theta = 0.0
        
        # FMI valueReference eşlemesi
        self.reference_to_attr = {
            0: "des_v",
            1: "des_omega",
            2: "des_x",
            3: "des_y",
            4: "des_theta",
            5: "x",
            6: "y",
            7: "theta"
        }
    
    def get_variable_name(self, ref):
        return self.reference_to_attr[ref]

    def instantiate(self, instanceName, resourceLocation):
        return Fmi2Status.ok
    
    def setup_experiment(self, startTime, stopTime, tolerance):
        return Fmi2Status.ok
    
    def enter_initialization_mode(self):
        return Fmi2Status.ok
    
    def exit_initialization_mode(self):
        return Fmi2Status.ok
    
    def set_real(self, refs, values):
        for ref, val in zip(refs, values):
            attr = self.get_variable_name(ref)
            setattr(self, attr, val)
        return Fmi2Status.ok
    
    def get_real(self, refs):
        outvals = []
        for ref in refs:
            attr = self.get_variable_name(ref)
            outvals.append(getattr(self, attr))
        return outvals, Fmi2Status.ok
    
    def do_step(self, currentTime, stepSize, noSetFMUStatePriorToCurrentPoint):
        """
        Her simülasyon adımında iki modu kontrol ederiz:
        
          1) Hedef modu:
             - Eğer (des_x, des_y) != (0,0) veya pos_error büyükse, robot hedefe ilerler.
             - pos_error yeterince küçükse, robot son açıyı düzeltmeye çalışır (des_theta).
          2) Doğrudan hız modu:
             - Eğer (des_x, des_y, des_theta) = (0,0,0) ise, robot des_v ve des_omega'yı uygular.
             
        Bu tasarım, her iki senaryoda da beklenen davranışları tolere edecek şekilde kurgulanmıştır.
        """
        
        # Robotun mevcut durumu
        x_curr = self.x
        y_curr = self.y
        theta_curr = self.theta
        
        # Hedef konum ve açı
        x_target = self.des_x
        y_target = self.des_y
        theta_target = normalize_angle(self.des_theta)  # son açı
        
        # Doğrudan hız girişleri
        v_direct = self.des_v
        omega_direct = self.des_omega
        
        # Konum hatası ve hedef yönü
        pos_error = math.sqrt((x_target - x_curr)**2 + (y_target - y_curr)**2)
        
        # İstenen yöne bakma açısı (atan2)
        desired_heading = math.atan2(y_target - y_curr, x_target - x_curr) if pos_error > 1e-6 else theta_curr
        heading_error = normalize_angle(desired_heading - theta_curr)
        
        # Son açı hatası (robot hedefe ulaştıktan sonra istenen final açı)
        final_angle_error = normalize_angle(theta_target - theta_curr)
        
        # Varsayılan komutlar
        v_command = 0.0
        omega_command = 0.0
        
        # "Doğrudan hız" kullanmak mı istiyoruz?
        # Koşul: Kullanıcı (des_x, des_y, des_theta) = (0,0,0) vermiş olsun.
        #        Veya pos_error < POS_TOL ve final_angle_error < ANGLE_TOL => hedeften memnun.
        #        Bu durumda direct velocity devreye girebilir.
        use_direct_mode = (
            abs(x_target) < 1e-9 and
            abs(y_target) < 1e-9 and
            abs(theta_target) < 1e-9
        )
        
        # 1) Eğer doğrudan hız modu aktifse, "hedef moduna" girmeyelim
        if use_direct_mode:
            v_command = v_direct
            omega_command = omega_direct
        
        else:
            # 2) Hedef modu
            #    (Aşama A) Eğer robot hedef konuma yeterince uzaksa, hedefe yönelip ilerle
            if pos_error > POS_TOL:
                # Doğrusal hız = KP_V * pos_error
                v_command = KP_V * pos_error
                # Dönüş hızı = KP_OMEGA * heading_error (hedefe bakacak şekilde dön)
                omega_command = KP_OMEGA * heading_error
                
            else:
                # (Aşama B) Robot hedefe yeterince yakın, şimdi son açı hatasını düzelt
                if abs(final_angle_error) > ANGLE_TOL:
                    # Açı düzeltmesi
                    v_command = 0.0
                    omega_command = KP_OMEGA * final_angle_error
                else:
                    # Hedef konum ve açıya zaten yeterince yakın
                    v_command = 0.0
                    omega_command = 0.0
        
        # Hız saturasyonu
        v_command = saturate(v_command, V_MAX)
        omega_command = saturate(omega_command, OMEGA_MAX)
        
        # Euler entegrasyonu ile konum ve açı güncelle
        self.x += v_command * math.cos(theta_curr) * DT
        self.y += v_command * math.sin(theta_curr) * DT
        self.theta += omega_command * DT
        self.theta = normalize_angle(self.theta)
        
        return Fmi2Status.ok
    
    def terminate(self):
        return Fmi2Status.ok
    
    def reset(self):
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        return Fmi2Status.ok

def create_fmu_instance():
    return Model()
