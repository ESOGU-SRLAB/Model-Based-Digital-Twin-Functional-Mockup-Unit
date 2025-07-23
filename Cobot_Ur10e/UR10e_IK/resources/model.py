"""
UR10e Inverse Kinematics – Analytic Closed‑Form Solver
=====================================================

Pure‑Python module that implements the closed‑form inverse kinematics
for the **UR10e** industrial manipulator.  Given an end‑effector pose
(position + XYZ‐Euler orientation, **metres / radians**), the solver
returns **four** valid joint‑angle solutions
(elbow‑up/down × wrist‑flip), each as a 6‑element NumPy array
[θ₁ … θ₆] in **radians**.

The routine follows the derivation in *Universal Robots Analytic IK*
notes (same as the PDF you shared).  All geometry constants are taken
from the official UR10e CAD (converted to metres).

Usage
-----
>>> import ur10e_ik as ik
>>> import numpy as np
>>> q_sets = ik.solve_ik(x=0.5, y=0.1, z=0.3,
...                      roll=0.0, pitch=np.pi/2, yaw=0.0)
>>> print(q_sets.shape)  # (4,6)

You can optionally pass a homogeneous 4×4 pose matrix instead of the
RPY components.
"""
from __future__ import annotations

import math
from typing import Iterable, List, Tuple

import numpy as np

# ────────────────────────────────────────────────────────────────
# 1) Robot geometric constants (metres)
# ────────────────────────────────────────────────────────────────
# Denavit–Hartenberg parameters – UR10e (base frame at robot pedestal)
#   Link |   a_i‑1  |  d_i   |  α_i‑1  | θ_i (variable)
#  --------------------------------------------------------------
#    1   |    0     | 0.1625 |  +π/2   | θ1
#    2   | ‑0.425   |   0    |     0   | θ2
#    3   | ‑0.3922  |   0    |     0   | θ3
#    4   |    0     | 0.1333 |  +π/2   | θ4
#    5   |    0     | 0.0997 | ‑π/2   | θ5
#    6   |    0     | 0.0996 |     0   | θ6 (flange)

# Link lengths (converted from mm → m)
_d1, _a2, _a3, _d4, _d5, _d6 = (
    0.1625,
    -0.425,
    -0.3922,
    0.1333,
    0.0997,
    0.0996,
)

# Axis unit vectors (for building wrist centre)
_Z_AXIS = np.array([0.0, 0.0, 1.0])

# Numerical tolerance (metres / radians)
_EPS = 1e-9

# ────────────────────────────────────────────────────────────────
# 2) Helper functions
# ────────────────────────────────────────────────────────────────

def _rot_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """XYZ Euler → 3×3 rotation matrix."""
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    # R = R_x(roll) * R_y(pitch) * R_z(yaw)
    R = np.array(
        [
            [cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy],
            [cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy],
            [   -sp ,              sr * cp ,              cr * cp ],
        ]
    )
    return R


def _dh(a: float, d: float, alpha: float, theta: float) -> np.ndarray:
    """Return a homogeneous DH transform."""
    sa, ca = math.sin(alpha), math.cos(alpha)
    st, ct = math.sin(theta), math.cos(theta)
    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


# ────────────────────────────────────────────────────────────────
# 3) Analytic inverse kinematics solver
# ────────────────────────────────────────────────────────────────


def solve_ik(
    *,
    x: float | None = None,
    y: float | None = None,
    z: float | None = None,
    roll: float | None = None,
    pitch: float | None = None,
    yaw: float | None = None,
    pose: np.ndarray | None = None,
) -> np.ndarray:
    """Return **four** analytic IK solutions for the given pose.

    Parameters
    ----------
    x, y, z : float
        End‑effector position in **metres**.
    roll, pitch, yaw : float
        XYZ Euler orientation (*radians*).  If `pose` is provided, these
        are ignored.
    pose : np.ndarray, optional
        4×4 homogeneous matrix.  If supplied, (x, y, z, roll, pitch, yaw)
        are ignored.

    Returns
    -------
    np.ndarray, shape (4, 6)
        The four IK solution sets in radians.
    """
    if pose is None:
        if None in (x, y, z, roll, pitch, yaw):
            raise ValueError("Either 'pose' or all scalar pose components must be provided.")
        R06 = _rot_from_rpy(roll, pitch, yaw)
        p06 = np.array([x, y, z], dtype=float)
        pose = np.eye(4)
        pose[:3, :3] = R06
        pose[:3, 3] = p06
    else:
        pose = np.asarray(pose, dtype=float)
        R06 = pose[:3, :3]
        p06 = pose[:3, 3]

    # 1) Wrist centre (WC) – intersection of axes 5 & 6
    pwc = p06 - _d6 * R06 @ _Z_AXIS  # 3‑vector
    pwc_xy_norm = math.hypot(pwc[0], pwc[1])

    # Guard against unreachable positions
    if pwc_xy_norm < _EPS:
        raise ValueError("Pose too close to base axis – singular configuration.")

    # ────────────────────────────────────────────────────────────
    # θ1 (two solutions)
    # ────────────────────────────────────────────────────────────
    # Based on projection onto base plane – see UR analytic derivation.
    # r = distance from base to WC projected on XY
    r = pwc_xy_norm
    # horizontal offset from joint 2 to 4 is d4
    if abs(_d4 / r) > 1.0:
        raise ValueError("Unreachable pose – r < d4.")
    phi = math.atan2(pwc[1], pwc[0])
    delta = math.acos(_d4 / r)
    theta1_a = phi + delta + math.pi / 2.0  # elbow‑down
    theta1_b = phi - delta + math.pi / 2.0  # elbow‑up

    # Normalise to [‑π, π]
    theta1_opts = [_wrap_to_pi(theta1_a), _wrap_to_pi(theta1_b)]

    solutions: List[np.ndarray] = []

    for theta1 in theta1_opts:
        # Position of joint‑2 origin (after rotating by θ1)
        T10 = _dh(0.0, _d1, math.pi / 2.0, theta1)
        p10 = T10[:3, 3]
        pwc_1 = pwc - p10  # vector in base frame

        # Transform WC into frame‑1
        R10 = T10[:3, :3]
        pwc_1 = R10.T @ pwc_1

        # Coordinates in plane of joints 2–3
        x1, y1, z1 = pwc_1  # note: y1 should be ~0 for UR due to design

        # ρ: planar distance from joint‑2 to WC in x‑z plane
        rho = math.hypot(x1, z1)

        # ─── θ3 (elbow – two solutions) ───────────────────────
        # Law of cosines on the triangle (a2, a3, rho)
        cos_theta3 = (rho ** 2 - _a2 ** 2 - _a3 ** 2) / (2.0 * _a2 * _a3)
        # Clamp for numerical robustness
        cos_theta3 = max(min(cos_theta3, 1.0), -1.0)
        theta3_a = math.acos(cos_theta3)  # elbow‑down (positive)
        theta3_b = -theta3_a             # elbow‑up   (negative)

        for theta3 in (theta3_a, theta3_b):
            # ─── θ2 -------------------------------------------------
            # Using geometry: θ2 = atan2(z1, x1) ‑ atan2(a3*sinθ3, a2 + a3*cosθ3)
            k1 = _a2 + _a3 * math.cos(theta3)
            k2 = _a3 * math.sin(theta3)
            theta2 = math.atan2(z1, x1) - math.atan2(k2, k1)
            theta2 = _wrap_to_pi(theta2)

            # ─── R36 and θ4‑θ6 -----------------------------------
            # Compute R03 from θ1‑θ3, then R36 = R03ᵀ * R06
            T21 = _dh(_a2, 0.0, 0.0, theta2)
            T32 = _dh(_a3, 0.0, 0.0, theta3)
            T30 = T10 @ T21 @ T32  # base → frame3
            R03 = T30[:3, :3]
            R36 = R03.T @ R06

            # θ5 from R36 element (two solutions ±)
            cos_theta5 = _clamp(R36[2, 2])
            # In theory cosθ5 = R36_33; if |cos| > 1 (numerical noise) clamp.
            if abs(cos_theta5) > 1.0:
                cos_theta5 = _clamp(cos_theta5)
            sin_theta5 = math.sqrt(max(0.0, 1 - cos_theta5 ** 2))
            theta5_opts = [math.atan2(sin_theta5, cos_theta5), math.atan2(-sin_theta5, cos_theta5)]

            for theta5 in theta5_opts:
                # θ4 and θ6 depend on sinθ5 (avoid division by 0)
                if abs(sin_theta5) < _EPS:
                    # Singular: treat θ4=0, θ6 = atan2(-R36[0,1], R36[0,0])
                    theta4 = 0.0
                    theta6 = math.atan2(-R36[0, 1], R36[0, 0])
                else:
                    theta4 = math.atan2(R36[2, 1] / sin_theta5, R36[2, 0] / sin_theta5)
                    theta6 = math.atan2(R36[1, 2] / sin_theta5, -R36[0, 2] / sin_theta5)

                q = np.array([
                    _wrap_to_pi(theta1),
                    _wrap_to_pi(theta2),
                    _wrap_to_pi(theta3),
                    _wrap_to_pi(theta4),
                    _wrap_to_pi(theta5),
                    _wrap_to_pi(theta6),
                ])
                solutions.append(q)

    # Deduplicate (numeric tolerance) and return first four unique sets
    unique_solutions = _unique_rows(np.vstack(solutions), tol=1e-6)
    if unique_solutions.shape[0] < 4:
        raise RuntimeError("Less than four unique IK solutions found – pose may be near singular.")
    return unique_solutions[:4]


# ────────────────────────────────────────────────────────────────
# 4) Internal utilities
# ────────────────────────────────────────────────────────────────

def _wrap_to_pi(angle: float) -> float:
    """Wrap angle to (‑π, π]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _clamp(x: float, /, *, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(min(x, hi), lo)


def _unique_rows(arr: np.ndarray, *, tol: float = 1e-9) -> np.ndarray:
    """Return unique rows of *arr* (2‑D) within *tol* L‑∞ norm."""
    unique = []
    for row in arr:
        if not any(np.allclose(row, u, atol=tol) for u in unique):
            unique.append(row)
    return np.vstack(unique)


# ────────────────────────────────────────────────────────────────
# 5) Self‑test (executed when run standalone)
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Simple random test – verifies FK(θ) ≈ pose for all solutions
    # FK imported lazily to avoid heavy deps if not needed.
    print("[ur10e_ik] Running quick numerical self‑check…")
    import random

    try:
        from math import tau  # Python 3.11+
    except ImportError:
        tau = 2 * math.pi

    # Random target pose via existing FK (for consistency)
    random_q = np.array([random.uniform(-math.pi, math.pi) for _ in range(6)])

    def _fk(q: Iterable[float]) -> np.ndarray:
        q1, q2, q3, q4, q5, q6 = q
        T01 = _dh(0.0, _d1, math.pi / 2.0, q1)
        T12 = _dh(_a2, 0.0, 0.0, q2)
        T23 = _dh(_a3, 0.0, 0.0, q3)
        T34 = _dh(0.0, _d4, math.pi / 2.0, q4)
        T45 = _dh(0.0, _d5, -math.pi / 2.0, q5)
        T56 = _dh(0.0, _d6, 0.0, q6)
        return (((((T01 @ T12) @ T23) @ T34) @ T45) @ T56)

    target_pose = _fk(random_q)

    sols = solve_ik(pose=target_pose)
    max_err = 0.0
    for q in sols:
        pose_i = _fk(q)
        pos_err = np.linalg.norm(pose_i[:3, 3] - target_pose[:3, 3])
        rot_err = np.linalg.norm(pose_i[:3, :3] - target_pose[:3, :3])
        max_err = max(max_err, pos_err, rot_err)
    print(f"  ✓ self‑test max FK error: {max_err:.2e} m/rad")
    assert max_err < 1e-6, "IK solutions do not reproduce target pose within tolerance!"
    print("[ur10e_ik] Self‑check passed – solver ready.")
