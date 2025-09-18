# model.py — UR10e Inverse Kinematics FMU (Co-Simulation, FMI 2.0)
"""
Inputs  (Real, refs 0..5):   x, y, z  [m], roll, pitch, yaw  [rad]  (Euler XYZ: roll→pitch→yaw)
Outputs (Real, refs 6..11):  q1..q6  [rad]

Design goals
------------
• DH-only framing (d1, d6 are inside DH vectors).
• Deterministic, ROS2-friendly: keep last solution as IK seed.
• Post-canonicalize just (q2, q4) to the UR elbow-down branch:
    if q2>0 and q4>0: q2 -= π; q4 -= π; then wrap to (−π, π].
"""

from __future__ import annotations

import math
import pickle
from typing import List, Optional, Tuple

import numpy as np
from ikpy.chain import Chain
from ikpy.link import DHLink

from fmi2 import Fmi2FMU, Fmi2Status


# ── 1) UR10e DH parameters (meters, radians) ──────────────────────────────────
UR10E_DH = {
    "a":     [0.0,   0.613, 0.572, 0.0,   0.0,   0.0],
    "d":     [0.181, 0.0,   0.0,   0.174, 0.120, 0.117],
    "alpha": [math.pi/2, 0.0, 0.0, math.pi/2, -math.pi/2, 0.0],
}
DEFAULT_JOINT_LIMITS: List[Tuple[float, float]] = [(-2*math.pi, 2*math.pi)] * 6


# ── 2) Math helpers (Euler XYZ) ────────────────────────────────────────────────
def wrap_to_pi(x: np.ndarray | float) -> np.ndarray | float:
    return np.arctan2(np.sin(x), np.cos(x))

def rot_x(rx: float) -> np.ndarray:
    cr, sr = math.cos(rx), math.sin(rx)
    return np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])

def rot_y(ry: float) -> np.ndarray:
    cp, sp = math.cos(ry), math.sin(ry)
    return np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])

def rot_z(rz: float) -> np.ndarray:
    cy, sy = math.cos(rz), math.sin(rz)
    return np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])

def euler_xyz_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    # Convention used by FMU: Rz(yaw) * Ry(pitch) * Rx(roll)
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)

def build_transform(x: float, y: float, z: float, roll: float, pitch: float, yaw: float) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = euler_xyz_to_matrix(roll, pitch, yaw)
    T[:3, 3] = np.array([x, y, z], dtype=float)
    return T


# ── 3) IK chain & solver (IKPy) ───────────────────────────────────────────────
def build_ikpy_chain(dh: dict, joint_limits: Optional[List[Tuple[float, float]]] = None) -> Chain:
    a, d, alpha = dh["a"], dh["d"], dh["alpha"]
    if joint_limits is None:
        joint_limits = DEFAULT_JOINT_LIMITS

    links: List[DHLink] = []
    for i in range(6):
        lo, hi = joint_limits[i]
        links.append(
            DHLink(
                name=f"joint_{i+1}",
                d=float(d[i]),
                a=float(a[i]),
                alpha=float(alpha[i]),
                theta=0.0,
                bounds=(float(lo), float(hi)),
            )
        )
    return Chain(name="ur10e_chain", links=links)

class UR10eIK:
    def __init__(self, dh: dict, limits: Optional[List[Tuple[float, float]]] = None) -> None:
        self.chain = build_ikpy_chain(dh, limits)
        self._last_solution = np.zeros(6, dtype=float)

    def solve(self, T_target: np.ndarray, q_seed: Optional[np.ndarray] = None, max_iter: int = 800) -> np.ndarray:
        if q_seed is None:
            q_seed = self._last_solution
        q_seed = np.asarray(q_seed, dtype=float).reshape(-1)
        if q_seed.size < 6:
            q_seed = np.pad(q_seed, (0, 6 - q_seed.size))
        try:
            q = self.chain.inverse_kinematics_frame(
                target=T_target,
                initial_position=q_seed,
                orientation_mode="all",
                max_iter=int(max_iter),
            )
        except TypeError:
            q = self.chain.inverse_kinematics_frame(T_target, q_seed)
        q = np.asarray(q, dtype=float).reshape(-1)
        if q.size > 6:
            q = q[:6]
        return wrap_to_pi(q)


# ── 4) Canonicalize just (q2, q4) to UR elbow-down branch ─────────────────────
def canonicalize_q2_q4(q: np.ndarray) -> np.ndarray:
    q = wrap_to_pi(np.asarray(q, dtype=float).copy())
    # If both shoulder & wrist are on the +π branch, move them back together.
    if q[1] > 0.0 and q[3] > 0.0:
        q[1] = wrap_to_pi(q[1] - math.pi)
        q[3] = wrap_to_pi(q[3] - math.pi)
    return q


# ── 5) FMI 2.0 Co-Simulation model ────────────────────────────────────────────
class Model(Fmi2FMU):
    """
    Inputs:  x, y, z, roll, pitch, yaw (Euler XYZ), meters & radians
    Outputs: q1..q6 (radians)
    """

    def __init__(self, reference_to_attr=None):
        super().__init__(reference_to_attr)

        # valueReference → attribute (must match modelDescription.xml)
        self.reference_to_attr = {
            0: "x", 1: "y", 2: "z",
            3: "roll", 4: "pitch", 5: "yaw",
            6: "q1", 7: "q2", 8: "q3", 9: "q4", 10: "q5", 11: "q6",
        }
        for name in self.reference_to_attr.values():
            setattr(self, name, 0.0)

        self._ik = UR10eIK(UR10E_DH, limits=DEFAULT_JOINT_LIMITS)
        self._last_q: np.ndarray = np.zeros(6)
        self._last_T: Optional[np.ndarray] = None

        self._update_outputs()

    # ---- helpers ----
    def _target_T(self) -> np.ndarray:
        return build_transform(self.x, self.y, self.z, self.roll, self.pitch, self.yaw)

    def _update_outputs(self) -> None:
        T = self._target_T()
        # micro-optimization: skip if inputs haven’t changed
        if self._last_T is not None and np.allclose(T, self._last_T, atol=1e-9):
            return
        self._last_T = T

        try:
            q_raw = self._ik.solve(T, q_seed=self._last_q)
        except Exception as exc:
            print(f"[UR10e_IK_FMU] IK failure: {exc}")
            return

        # Enforce UR elbow-down branch (q2,q4 together), then persist
        q_best = canonicalize_q2_q4(q_raw)
        self._last_q = q_best.copy()
        self._ik._last_solution = q_best.copy()

        self.q1, self.q2, self.q3, self.q4, self.q5, self.q6 = [float(a) for a in q_best]

    # ---- FMI 2.0 API ----
    def get_variable_name(self, vr):
        return self.reference_to_attr[vr]

    def set_real(self, refs, values):
        for ref, val in zip(refs, values):
            setattr(self, self.get_variable_name(ref), float(val))
        return Fmi2Status.ok

    def get_real(self, refs):
        return [float(getattr(self, self.get_variable_name(ref))) for ref in refs], Fmi2Status.ok

    def instantiate(self, instanceName, resourceLocation):
        return Fmi2Status.ok

    def setup_experiment(self, startTime, stopTime, tolerance):
        return Fmi2Status.ok

    def enter_initialization_mode(self):
        return Fmi2Status.ok

    def exit_initialization_mode(self):
        self._update_outputs()
        return Fmi2Status.ok

    def do_step(self, current_time, step_size, no_prior):
        self._update_outputs()
        return Fmi2Status.ok

    def terminate(self):
        return Fmi2Status.ok

    def reset(self):
        for name in self.reference_to_attr.values():
            setattr(self, name, 0.0)
        self._ik._last_solution[:] = 0.0
        self._last_q[:] = 0.0
        self._last_T = None
        self._update_outputs()
        return Fmi2Status.ok

    # Optional: keep state/seed across checkpoints
    def serialize(self):
        state = {
            "x": self.x, "y": self.y, "z": self.z,
            "roll": self.roll, "pitch": self.pitch, "yaw": self.yaw,
            "q": [self.q1, self.q2, self.q3, self.q4, self.q5, self.q6],
            "seed": self._last_q.tolist(),
        }
        return Fmi2Status.ok, pickle.dumps(state)

    def deserialize(self, bytes_):
        data = pickle.loads(bytes_)
        self.x = float(data["x"]); self.y = float(data["y"]); self.z = float(data["z"])
        self.roll = float(data["roll"]); self.pitch = float(data["pitch"]); self.yaw = float(data["yaw"])
        (self.q1, self.q2, self.q3, self.q4, self.q5, self.q6) = [float(v) for v in data["q"]]
        self._last_q = np.asarray(data.get("seed", [self.q1, self.q2, self.q3, self.q4, self.q5, self.q6]), dtype=float)
        self._ik._last_solution = self._last_q.copy()
        self._last_T = None
        return Fmi2Status.ok


def create_fmu_instance():
    return Model()
