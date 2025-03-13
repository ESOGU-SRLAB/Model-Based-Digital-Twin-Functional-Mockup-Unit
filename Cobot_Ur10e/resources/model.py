import math
import numpy as np
from fmi2 import Fmi2FMU, Fmi2Status

###############################################################################
# UR10e Robot Model Data
###############################################################################
NUM_JOINTS = 6
GRAVITY = 9.81

# --------------------------------------------------------------------------
# 1) Kinematik Parametreler (DH benzeri veya transform matrisleri)
#    Gerçek değerler "default_kinematics.yaml" ve "physical_parameters.yaml" dosyalarından alınmalıdır.
# --------------------------------------------------------------------------
# DH parametre tablosu: (a [m], d [m], alpha [rad])
DH_PARAMS = [
    (0.1807,  0.0,     math.pi/2),   # Joint 1
    (0.6127,  0.0,     0.0      ),   # Joint 2
    (0.57155, 0.17415, 0.0      ),   # Joint 3
    (0.11985, 0.0,     math.pi/2),   # Joint 4
    (0.11655, 0.0,    -math.pi/2),   # Joint 5
    (0.0,     0.0,     0.0      )    # Joint 6
]

# --------------------------------------------------------------------------
# 2) Dinamik Parametreler: Kütle, atalet tensörü, kütle merkezi
#    "physical_parameters.yaml" içeriğine dayalı.
# --------------------------------------------------------------------------
# Link kütleleri [kg]
LINK_MASSES = np.array([7.369, 13.051, 3.989, 2.1, 1.98, 0.615])

# --------------------------------------------------------------------------
# Eklem limitleri (joint_limits.yaml) - örnek değerler:
JOINT_VEL_LIMITS = [2.094, 2.094, 3.142, 3.142, 3.142, 3.142]  # rad/s
JOINT_TORQUE_LIMITS = [330.0, 330.0, 150.0, 54.0, 54.0, 54.0]    # Nm

###############################################################################
# Kinematik Fonksiyonlar
###############################################################################
def dh_transform(theta, d, a, alpha):
    """
    Create a 4x4 transformation matrix from DH parameters (radians).
    """
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array([
        [ ct,    -st*ca,  st*sa,  a*ct ],
        [ st,     ct*ca, -ct*sa,  a*st ],
        [  0,        sa,     ca,     d ],
        [  0,         0,      0,     1 ]
    ])

def forward_kinematics(q_rad):
    """
    Returns end-effector pose [x, y, z, roll, pitch, yaw] in the base frame.
    q_rad is a 6-element array (radians).
    """
    T = np.eye(4)
    for i in range(NUM_JOINTS):
        theta_i = q_rad[i]  # Gerekirse eklem offsetleri eklenebilir.
        a_i, d_i, alpha_i = DH_PARAMS[i]
        T_i = dh_transform(theta_i, d_i, a_i, alpha_i)
        T = T @ T_i
    x, y, z = T[0, 3], T[1, 3], T[2, 3]
    roll, pitch, yaw = matrix_to_rpy(T)
    return np.array([x, y, z, roll, pitch, yaw])

def matrix_to_rpy(T):
    """
    Convert a rotation matrix to roll, pitch, yaw (X-Y-Z sequence).
    """
    r11, r12, r13 = T[0,0], T[0,1], T[0,2]
    r21, r22, r23 = T[1,0], T[1,1], T[1,2]
    r31, r32, r33 = T[2,0], T[2,1], T[2,2]
    beta = math.atan2(-r31, math.sqrt(r11*r11 + r21*r21))
    alpha = math.atan2(r21, r11)
    gamma = math.atan2(r32, r33)
    return np.array([alpha, beta, gamma])

def compute_jacobian(q_rad, delta=1e-6):
    """
    Numerical Jacobian for the end-effector pose with respect to each joint angle.
    Returns a 6x6 Jacobian.
    """
    base_pose = forward_kinematics(q_rad)
    J = np.zeros((6, NUM_JOINTS))
    for j in range(NUM_JOINTS):
        perturbed = q_rad.copy()
        perturbed[j] += delta
        pose_pert = forward_kinematics(perturbed)
        J[:, j] = (pose_pert - base_pose) / delta
    return J

def inverse_kinematics_pose(des_pose, init_q_rad, max_iter=100, alpha=0.01, tol=1e-4):
    """
    Iteratif IK: des_pose = [x, y, z, roll, pitch, yaw].
    Returns joint angles (radians).
    """
    q = init_q_rad.copy()
    for _ in range(max_iter):
        current_pose = forward_kinematics(q)
        error = des_pose - current_pose
        if np.linalg.norm(error[:3]) < tol and np.linalg.norm(error[3:]) < tol:
            break
        J = compute_jacobian(q)
        J_inv = np.linalg.pinv(J, rcond=1e-3)
        q += alpha * (J_inv @ error)
    return q

###############################################################################
# Newton-Euler Dinamik Hesaplamalar (Revize Edilmiş Model)
###############################################################################
def compute_dynamics(q, dq, ddq):
    """
    Basit Newton-Euler yaklaşımı: 
      tau = I*ddq + gravity_term + friction_term
    Burada:
      - I: atalet (diagonal yaklaşım)
      - gravity_term: LINK_MASSES * GRAVITY * L_offset * sin(q)
      - friction_term: Coulomb friction (statik) modeli
    """
    tau = np.zeros(NUM_JOINTS)
    # Gerçek UR10e değerlerine yakın atalet: Örnek değerler (radyan cinsinden)
    # Örneğin: Joint 1: 0.05, Joint 2: 0.77, Joint 3: 0.31, Joint 4: 0.02, Joint 5: 0.02, Joint 6: 0.01
    I_approx = np.array([0.05, 0.77, 0.31, 0.02, 0.02, 0.01])
    # Örnek: Yerçekimi etkisi için L_offset (yaklaşık yarıçap benzeri, m)
    L_offset = 0.1  
    # Coulomb friction katsayıları (statik): Üretici verisine yakın örnek değerler
    FRICTION_COEFF = np.array([0.5, 0.5, 0.3, 0.2, 0.2, 0.1])
    
    for i in range(NUM_JOINTS):
        inertia_term = I_approx[i] * ddq[i]
        grav_term = LINK_MASSES[i] * GRAVITY * L_offset * math.sin(q[i])
        # Eğer dq = 0 ise friction term sıfır; aksi halde, friction = friction_coeff * sign(dq)
        friction_term = FRICTION_COEFF[i] * np.sign(dq[i]) if dq[i] != 0 else 0.0
        tau[i] = inertia_term + grav_term + friction_term
    return tau

###############################################################################
# FMU Class for UR10e Cobot (Revize Edilmiş)
###############################################################################
class Model(Fmi2FMU):
    """
    FMI Co-Simulation model for UR10e robot:
      - Ters kinematik: End-effector hedef pose (x_des, y_des, z_des, roll_des, pitch_des, yaw_des)
        kullanılarak istenen eklem açıları (q_des) hesaplanır.
      - Basit PD kontrol: tau_cmd = kp*(q_des - q) - kd*dq
      - Newton-Euler benzeri dinamik hesaplama ile net tork (tau_eff) üzerinden
        eklem ivmesi (ddq) hesaplanır.
      - Euler entegrasyonu ile eklem açıları (q) ve hızları (dq) güncellenir.
      - Eklem limit saturasyonu (hız ve tork limitleri)
      
    Inputs (valueRefs):
      0: x_des
      1: y_des
      2: z_des
      3: roll_des
      4: pitch_des
      5: yaw_des
      6: kp
      7: kd
      
    Outputs (valueRefs):
      8:  q1
      9:  q2
      10: q3
      11: q4
      12: q5
      13: q6
      14: dq1
      15: dq2
      16: dq3
      17: dq4
      18: dq5
      19: dq6
      20: ee_x
      21: ee_y
      22: ee_z
      23: ee_roll
      24: ee_pitch
      25: ee_yaw
      26: tau1
      27: tau2
      28: tau3
      29: tau4
      30: tau5
      31: tau6
    """
    def __init__(self, reference_to_attr=None):
        super().__init__()
        self.reference_to_attr = reference_to_attr if reference_to_attr else {}
        
        # Internal states: eklem açıları ve hızları (radians, rad/s)
        self.q_rad = np.zeros(NUM_JOINTS)
        self.dq_rad = np.zeros(NUM_JOINTS)
        
        # Inputs (varsayılan değerler)
        self.x_des = 0.5
        self.y_des = 0.0
        self.z_des = 0.5
        self.roll_des = 0.0
        self.pitch_des = 0.0
        self.yaw_des = 0.0
        self.kp = 10.0
        self.kd = 2.0
        
        # Outputs:
        self.q_out = np.zeros(NUM_JOINTS)      # q1..q6
        self.dq_out = np.zeros(NUM_JOINTS)     # dq1..dq6
        self.ee_pose = np.zeros(6)             # [x, y, z, roll, pitch, yaw]
        self.tau_out = np.zeros(NUM_JOINTS)      # tau1..tau6
        
        # Value references mapping:
        self.reference_to_attr = {
            # Inputs
            0:  "x_des",
            1:  "y_des",
            2:  "z_des",
            3:  "roll_des",
            4:  "pitch_des",
            5:  "yaw_des",
            6:  "kp",
            7:  "kd",
            # Outputs
            8:  "q1",
            9:  "q2",
            10: "q3",
            11: "q4",
            12: "q5",
            13: "q6",
            14: "dq1",
            15: "dq2",
            16: "dq3",
            17: "dq4",
            18: "dq5",
            19: "dq6",
            20: "ee_x",
            21: "ee_y",
            22: "ee_z",
            23: "ee_roll",
            24: "ee_pitch",
            25: "ee_yaw",
            26: "tau1",
            27: "tau2",
            28: "tau3",
            29: "tau4",
            30: "tau5",
            31: "tau6"
        }
    
    def get_variable_name(self, ref):
        return self.reference_to_attr[ref]
    
    # Property getters to expose outputs with the same names as in modelDescription.xml
    @property
    def q1(self): return self.q_out[0]
    @property
    def q2(self): return self.q_out[1]
    @property
    def q3(self): return self.q_out[2]
    @property
    def q4(self): return self.q_out[3]
    @property
    def q5(self): return self.q_out[4]
    @property
    def q6(self): return self.q_out[5]
    
    @property
    def dq1(self): return self.dq_out[0]
    @property
    def dq2(self): return self.dq_out[1]
    @property
    def dq3(self): return self.dq_out[2]
    @property
    def dq4(self): return self.dq_out[3]
    @property
    def dq5(self): return self.dq_out[4]
    @property
    def dq6(self): return self.dq_out[5]
    
    @property
    def ee_x(self): return self.ee_pose[0]
    @property
    def ee_y(self): return self.ee_pose[1]
    @property
    def ee_z(self): return self.ee_pose[2]
    @property
    def ee_roll(self): return self.ee_pose[3]
    @property
    def ee_pitch(self): return self.ee_pose[4]
    @property
    def ee_yaw(self): return self.ee_pose[5]
    
    @property
    def tau1(self): return self.tau_out[0]
    @property
    def tau2(self): return self.tau_out[1]
    @property
    def tau3(self): return self.tau_out[2]
    @property
    def tau4(self): return self.tau_out[3]
    @property
    def tau5(self): return self.tau_out[4]
    @property
    def tau6(self): return self.tau_out[5]
    
    # FMI2 lifecycle methods
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
            val = getattr(self, attr)
            outvals.append(val)
        return outvals, Fmi2Status.ok
    
    def do_step(self, currentTime, stepSize, noSetFMUStatePriorToCurrentPoint):
        """
        1) Ters kinematik: des_pose -> q_des
        2) Basit PD kontrol: tau_cmd = kp*(q_des - q) - kd*dq
        3) Newton-Euler benzeri dinamik hesaplama: tau_dyn
        4) Net tork üzerinden eklem ivmesi: ddq = (tau_cmd - tau_dyn) / I_approx
        5) Euler entegrasyonu ve eklem limit saturasyonu
        6) Çıktıları güncelle
        """
        # 1) Ters Kinematik
        des_pose = np.array([self.x_des, self.y_des, self.z_des,
                             self.roll_des, self.pitch_des, self.yaw_des])
        q_des = inverse_kinematics_pose(des_pose, self.q_rad)
        
        # 2) PD Kontrol
        error = q_des - self.q_rad
        tau_cmd = self.kp * error - self.kd * self.dq_rad
        
        # 3) Dinamik hesaplama (gravity + inertia + friction)
        tau_dyn = compute_dynamics(self.q_rad, self.dq_rad, np.zeros(NUM_JOINTS))
        tau_eff = tau_cmd - tau_dyn
        
        # 4) Eklem ivmesi: Güncel inertia değerlerini gerçek UR10e değerlerine yakın seçiyoruz.
        I_approx = np.array([0.05, 0.77, 0.31, 0.02, 0.02, 0.01])
        ddq = tau_eff / I_approx
        
        # 5) Euler entegrasyonu
        dq_new = self.dq_rad + ddq * stepSize
        q_new = self.q_rad + dq_new * stepSize
        
        # Eklem hız ve tork limit saturasyonu
        for i in range(NUM_JOINTS):
            if abs(dq_new[i]) > JOINT_VEL_LIMITS[i]:
                dq_new[i] = np.sign(dq_new[i]) * JOINT_VEL_LIMITS[i]
            if abs(tau_cmd[i]) > JOINT_TORQUE_LIMITS[i]:
                tau_cmd[i] = np.sign(tau_cmd[i]) * JOINT_TORQUE_LIMITS[i]
        
        self.q_rad = q_new
        self.dq_rad = dq_new
        
        # 6) Çıktıları güncelle
        self.q_out = self.q_rad.copy()
        self.dq_out = self.dq_rad.copy()
        self.ee_pose = forward_kinematics(self.q_rad)
        self.tau_out = tau_cmd.copy()
        
        return Fmi2Status.ok
    
    def terminate(self):
        return Fmi2Status.ok
    
    def reset(self):
        self.q_rad = np.zeros(NUM_JOINTS)
        self.dq_rad = np.zeros(NUM_JOINTS)
        return Fmi2Status.ok

def create_fmu_instance():
    return Model()
