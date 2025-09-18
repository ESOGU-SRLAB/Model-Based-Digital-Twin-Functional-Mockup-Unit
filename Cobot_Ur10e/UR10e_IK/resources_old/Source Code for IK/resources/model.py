# model.py — UR10e IK FMU (FMI 2.0 Co-Simulation)
# Variable names aligned to modelDescription.xml: x_ik, y_ik, z_ik, roll_ik, pitch_ik, yaw_ik, q1_ik..q6_ik

from __future__ import annotations
import math, os, pickle
from typing import List, Optional, Tuple
import numpy as np
from ikpy.chain import Chain
from ikpy.link import DHLink
from fmi2 import Fmi2FMU, Fmi2Status

# ---------- UR10e DH (meters, radians) ----------
UR10E_DH = {
    "a":     [0.0, 0.613, 0.572, 0.0,   0.0,   0.0],
    "d":     [0.181, 0.0,  0.0,   0.174, 0.120, 0.117],
    "alpha": [math.pi/2, 0.0, 0.0, math.pi/2, -math.pi/2, 0.0],
}
DEFAULT_JOINT_LIMITS: List[Tuple[float, float]] = [(-2*math.pi, 2*math.pi)]*6

# Orientation mode: 'auto' | 'euler' | 'rvec'
ORI_MODE = os.getenv("UR10E_IK_INPUT", "rvec").lower().strip()
if ORI_MODE not in {"auto","euler","rvec"}:
    ORI_MODE = "auto"

# ---------- math helpers ----------
def wrap_to_pi(x): return np.arctan2(np.sin(x), np.cos(x))
def rot_x(a): c,s=math.cos(a),math.sin(a); return np.array([[1,0,0],[0,c,-s],[0,s,c]])
def rot_y(a): c,s=math.cos(a),math.sin(a); return np.array([[c,0,s],[0,1,0],[-s,0,c]])
def rot_z(a): c,s=math.cos(a),math.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def euler_xyz_to_R(roll, pitch, yaw):
    # Rz(yaw) * Ry(pitch) * Rx(roll)
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)

def rvec_to_R(rx, ry, rz):
    r = np.array([rx, ry, rz], dtype=float)
    th = float(np.linalg.norm(r))
    if th < 1e-12: return np.eye(3)
    k = r / th
    K = np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]], dtype=float)
    return np.eye(3) + math.sin(th)*K + (1.0-math.cos(th))*(K@K)

def T_from_R_p(R, p):
    T = np.eye(4); T[:3,:3]=R; T[:3,3]=np.asarray(p, dtype=float); return T

def dh_fk(q):
    a,d,al = UR10E_DH["a"], UR10E_DH["d"], UR10E_DH["alpha"]
    T = np.eye(4)
    for i in range(6):
        ct,st = math.cos(q[i]), math.sin(q[i])
        ca,sa = math.cos(al[i]), math.sin(al[i])
        A = np.array([[ct, -st*ca,  st*sa, a[i]*ct],
                      [st,  ct*ca, -ct*sa, a[i]*st],
                      [0.0,    sa,     ca, d[i]],
                      [0.0,   0.0,    0.0, 1.0]])
        T = T @ A
    return T

def rot_angle(Rerr):
    tr = np.trace(Rerr); val = max(-1.0, min(1.0, (tr-1.0)/2.0)); return math.acos(val)
def pose_err(T_fk, T_tgt):
    dp = T_fk[:3,3] - T_tgt[:3,3]
    pos_mm = 1000.0*float(np.linalg.norm(dp))
    ang = rot_angle(T_fk[:3,:3].T @ T_tgt[:3,:3])
    return pos_mm + 1000.0*ang

# ---------- IK wrapper ----------
def build_chain(dh, limits=None):
    if limits is None: limits = DEFAULT_JOINT_LIMITS
    links=[]
    for i in range(6):
        lo,hi = limits[i]
        links.append(DHLink(name=f"joint_{i+1}", d=float(dh["d"][i]), a=float(dh["a"][i]),
                            alpha=float(dh["alpha"][i]), theta=0.0, bounds=(float(lo),float(hi))))
    return Chain(name="ur10e_chain", links=links)

class UR10eIK:
    def __init__(self, dh, limits=None):
        self.chain = build_chain(dh, limits)
        self._last = np.zeros(6)
    def solve(self, T, seed=None, max_iter=600):
        if seed is None: seed = self._last
        seed = np.asarray(seed, dtype=float).reshape(-1)
        if seed.size < 6: seed = np.pad(seed, (0, 6-seed.size))
        try:
            q = self.chain.inverse_kinematics_frame(target=T, initial_position=seed,
                                                    orientation_mode="all", max_iter=int(max_iter))
        except TypeError:
            q = self.chain.inverse_kinematics_frame(T, seed)
        q = np.asarray(q, dtype=float).reshape(-1)[:6]
        return wrap_to_pi(q)

def canon_q2_q4(q):
    q = wrap_to_pi(np.asarray(q, dtype=float))
    if q[1] > 0.0 and q[3] > 0.0:
        q[1] = wrap_to_pi(q[1]-math.pi); q[3] = wrap_to_pi(q[3]-math.pi)
    return q

# ---------- FMI model ----------
class Model(Fmi2FMU):
    def __init__(self, reference_to_attr=None):
        super().__init__(reference_to_attr)

        # Map valueReference -> attribute **aligned with _ik names in modelDescription.xml**
        self.reference_to_attr = {
            0:"x_ik", 1:"y_ik", 2:"z_ik",
            3:"roll_ik", 4:"pitch_ik", 5:"yaw_ik",
            6:"q1_ik", 7:"q2_ik", 8:"q3_ik", 9:"q4_ik", 10:"q5_ik", 11:"q6_ik",
        }
        for n in self.reference_to_attr.values(): setattr(self, n, 0.0)

        self._ik = UR10eIK(UR10E_DH)
        self._seed = np.zeros(6)
        self._last_key = None
        self._last_mode = None  # 'euler' or 'rvec'
        self._update_outputs()

    def _T_euler(self):
        R = euler_xyz_to_R(self.roll_ik, self.pitch_ik, self.yaw_ik)
        return T_from_R_p(R, [self.x_ik, self.y_ik, self.z_ik])
    def _T_rvec(self):
        R = rvec_to_R(self.roll_ik, self.pitch_ik, self.yaw_ik)
        return T_from_R_p(R, [self.x_ik, self.y_ik, self.z_ik])

    def _update_outputs(self):
        key = (self.x_ik, self.y_ik, self.z_ik, self.roll_ik, self.pitch_ik, self.yaw_ik)
        if self._last_key is not None and np.allclose(np.array(key), self._last_key, atol=1e-12):
            return
        self._last_key = np.array(key, dtype=float)

        try:
            seed = self._seed
            if ORI_MODE == "euler":
                T = self._T_euler()
                q = self._ik.solve(T, seed)
            elif ORI_MODE == "rvec":
                T = self._T_rvec()
                q = self._ik.solve(T, seed)
            else:  # auto
                T_eu = self._T_euler(); q_eu = self._ik.solve(T_eu, seed); err_eu = pose_err(dh_fk(q_eu), T_eu)
                T_rv = self._T_rvec();  q_rv = self._ik.solve(T_rv, seed); err_rv = pose_err(dh_fk(q_rv), T_rv)
                if self._last_mode == "euler" and err_rv > 0.9*err_eu: mode, T, q = "euler", T_eu, q_eu
                elif self._last_mode == "rvec" and err_eu > 0.9*err_rv: mode, T, q = "rvec",  T_rv, q_rv
                else:
                    mode, T, q = ("euler", T_eu, q_eu) if err_eu <= err_rv else ("rvec", T_rv, q_rv)
                self._last_mode = mode

            q = canon_q2_q4(q)
            self._seed = q.copy(); self._ik._last = q.copy()
            self.q1_ik, self.q2_ik, self.q3_ik, self.q4_ik, self.q5_ik, self.q6_ik = [float(a) for a in q]

        except Exception as e:
            print(f"[UR10e_IK_FMU] IK failure: {e}")

    # ---- FMI 2.0 required ----
    def get_variable_name(self, vr): return self.reference_to_attr[vr]
    def set_real(self, refs, vals):
        for r,v in zip(refs, vals): setattr(self, self.get_variable_name(r), float(v))
        return Fmi2Status.ok
    def get_real(self, refs):
        return [float(getattr(self, self.get_variable_name(r))) for r in refs], Fmi2Status.ok
    def instantiate(self, instanceName, resourceLocation): return Fmi2Status.ok
    def setup_experiment(self, startTime, stopTime, tolerance): return Fmi2Status.ok
    def enter_initialization_mode(self): return Fmi2Status.ok
    def exit_initialization_mode(self): self._update_outputs(); return Fmi2Status.ok
    def do_step(self, current_time, step_size, no_prior): self._update_outputs(); return Fmi2Status.ok
    def terminate(self): return Fmi2Status.ok
    def reset(self):
        for n in self.reference_to_attr.values(): setattr(self, n, 0.0)
        self._seed[:] = 0.0; self._ik._last[:] = 0.0; self._last_key = None; self._last_mode = None
        self._update_outputs(); return Fmi2Status.ok
    def serialize(self):
        state = {
            "x_ik": self.x_ik, "y_ik": self.y_ik, "z_ik": self.z_ik,
            "roll_ik": self.roll_ik, "pitch_ik": self.pitch_ik, "yaw_ik": self.yaw_ik,
            "q": [self.q1_ik, self.q2_ik, self.q3_ik, self.q4_ik, self.q5_ik, self.q6_ik],
            "seed": self._seed.tolist(), "mode": self._last_mode,
        }
        return Fmi2Status.ok, pickle.dumps(state)
    def deserialize(self, bytes_):
        d = pickle.loads(bytes_)
        self.x_ik=float(d["x_ik"]); self.y_ik=float(d["y_ik"]); self.z_ik=float(d["z_ik"])
        self.roll_ik=float(d["roll_ik"]); self.pitch_ik=float(d["pitch_ik"]); self.yaw_ik=float(d["yaw_ik"])
        (self.q1_ik,self.q2_ik,self.q3_ik,self.q4_ik,self.q5_ik,self.q6_ik) = [float(v) for v in d["q"]]
        self._seed = np.asarray(d.get("seed",[self.q1_ik,self.q2_ik,self.q3_ik,self.q4_ik,self.q5_ik,self.q6_ik]), dtype=float)
        self._ik._last = self._seed.copy()
        self._last_mode = d.get("mode", None)
        self._last_key = None
        return Fmi2Status.ok

def create_fmu_instance(): return Model()
