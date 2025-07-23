"""
UR10e Inverse Kinematics FMU (Co‑Simulation) - REVISED
====================================================
*Aligned with validated FK model DH conventions*

This revision aligns the IK model with the validated FK model by:
1. Using NEGATIVE link lengths (a2 = -0.613, a3 = -0.572) to match FK
2. Removing Craig theta offsets to match FK (no theta offsets)
3. Maintaining same physical constants and frame definitions

The goal is to achieve exact FK→IK→FK round-trip consistency.
"""

from __future__ import annotations
import math
import pickle
from typing import List

import numpy as np
from fmi2 import Fmi2FMU, Fmi2Status

# ──────────────────────────────────────────────────────────
# 1) Constants (metres) & DH arrays - ALIGNED WITH FK MODEL
# ──────────────────────────────────────────────────────────
LB   = 0.181            # base‑frame Z offset
a2   = -0.613           # link‑2 length (NEGATIVE to match FK)
a3   = -0.572           # link‑3 length (NEGATIVE to match FK)
d4   = 0.174            # wrist‑1 offset
d5   = 0.120            # wrist‑2 offset
LTP  = 0.117            # flange → TCP (treated as d₆)

d6 = LTP               # alias for clarity

# DH arrays matching FK model exactly
_d    = np.array([LB,   0.0, 0.0, d4,  d5,  LTP])  # d values including base offset
_a    = np.array([0.0,  a2,  a3,  0.0, 0.0, 0.0])  # NEGATIVE a2, a3 like FK
_alph = np.array([0.0, math.pi/2, 0.0, math.pi/2, -math.pi/2, 0.0])  # Alpha values matching FK

NUM_JOINTS = 6

# ──────────────────────────────────────────────────────────
# 2) Helper functions
# ──────────────────────────────────────────────────────────

def _clip(val: float, lo: float = -1.0, hi: float = 1.0) -> float:
    """Clip value to range [lo, hi]"""
    return max(lo, min(hi, val))

def create_dh_transform_matrix(alpha, a, d, theta):
    """Creates DH transformation matrix - SAME AS FK MODEL"""
    return np.array([
        [math.cos(theta), -math.sin(theta) * math.cos(alpha),  math.sin(theta) * math.sin(alpha), a * math.cos(theta)],
        [math.sin(theta),  math.cos(theta) * math.cos(alpha), -math.cos(theta) * math.sin(alpha), a * math.sin(theta)],
        [0,                math.sin(alpha),                    math.cos(alpha),                   d],
        [0,                0,                                  0,                                 1]
    ])

def get_transform_matrix(joint_idx: int, theta: float) -> np.ndarray:
    """Get transformation matrix for joint i using same convention as FK"""
    return create_dh_transform_matrix(_alph[joint_idx], _a[joint_idx], _d[joint_idx], theta)

def _invert_h(T: np.ndarray) -> np.ndarray:
    """Invert homogeneous transformation matrix"""
    R, t = T[:3, :3], T[:3, 3]
    Ti   = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti

def rpy_to_matrix(roll: float, pitch: float, yaw: float,
                  x: float, y: float, z: float) -> np.ndarray:
    """Convert RPY + position to homogeneous transformation matrix"""
    cr, sr = math.cos(roll),  math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw),   math.sin(yaw)
    R = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [  -sp,            cp*sr,            cp*cr],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T

# ──────────────────────────────────────────────────────────
# 3) Forward Kinematics (for verification) - SAME AS FK MODEL
# ──────────────────────────────────────────────────────────

def forward_kinematics_check(q):
    """FK implementation identical to FK model for verification"""
    # Sequential multiplication: T_0^6 = A_1 * A_2 * ... * A_6
    T_0_6 = np.eye(4)
    for i in range(NUM_JOINTS):
        A_i = get_transform_matrix(i, q[i])
        T_0_6 = T_0_6 @ A_i
    
    # Tool frame (no additional offset for now)
    T_6_Tool = np.eye(4)
    T_final = T_0_6 @ T_6_Tool
    
    return T_final

# ──────────────────────────────────────────────────────────
# 4) Analytical IK solver - REVISED FOR FK COMPATIBILITY
# ──────────────────────────────────────────────────────────

def analytical_ik_solver(T_target: np.ndarray) -> List[np.ndarray]:
    """
    Analytical IK solver using same DH conventions as FK model
    Returns up to 8 solutions for joint angles
    """
    print(f"\n=== IK SOLVER DEBUG ===")
    print(f"Target transformation matrix:")
    print(f"T_target =\n{T_target}")
    
    solutions = []
    
    # Extract position and orientation from target
    px, py, pz = T_target[0, 3], T_target[1, 3], T_target[2, 3]
    R = T_target[:3, :3]
    
    print(f"\nTarget position: [{px:.6f}, {py:.6f}, {pz:.6f}]")
    print(f"Target distance from base: {math.sqrt(px**2 + py**2 + pz**2):.6f}m")
    print(f"Target rotation matrix:\n{R}")
    
    # Wrist center position (subtracting d6 along approach vector)
    wrist_center = np.array([px, py, pz]) - d6 * R[:, 2]
    print(f"\nWrist center calculation:")
    print(f"  d6 = {d6}")
    print(f"  R[:, 2] (approach vector) = {R[:, 2]}")
    print(f"  Wrist center = [{wrist_center[0]:.6f}, {wrist_center[1]:.6f}, {wrist_center[2]:.6f}]")
    
    # Joint 1 solutions (shoulder pan)
    # Two solutions for theta1 based on wrist center position
    theta1_solutions = []
    
    # Calculate theta1 using wrist center
    r = math.sqrt(wrist_center[0]**2 + wrist_center[1]**2)
    print(f"\nJoint 1 calculation:")
    print(f"  r (radial distance) = {r:.6f}")
    print(f"  d4 = {d4}")
    print(f"  Check: r >= |d4| ? {r:.6f} >= {abs(d4):.6f} = {r >= abs(d4)}")
    
    if r >= abs(d4):  # Check if solution exists
        # Two solutions for theta1
        phi = math.atan2(wrist_center[1], wrist_center[0])
        psi = math.acos(d4 / r)
        
        theta1_1 = phi + psi + math.pi/2
        theta1_2 = phi - psi + math.pi/2
        
        print(f"  phi = atan2({wrist_center[1]:.6f}, {wrist_center[0]:.6f}) = {phi:.6f} rad")
        print(f"  psi = acos({d4:.6f}/{r:.6f}) = {psi:.6f} rad")
        print(f"  theta1_1 = {theta1_1:.6f} rad")
        print(f"  theta1_2 = {theta1_2:.6f} rad")
        
        theta1_solutions = [theta1_1, theta1_2]
    else:
        print(f"  ERROR: No theta1 solution - wrist center too close to base axis!")
        return []  # No solution possible
    
    # For each theta1, find remaining joint angles
    print(f"\n=== PROCESSING THETA1 SOLUTIONS ===")
    for idx, theta1 in enumerate(theta1_solutions):
        print(f"\n--- Processing theta1 solution {idx+1}: {theta1:.6f} rad ---")
        
        # Transform wrist center to joint 1 frame
        T01 = get_transform_matrix(0, theta1)
        T01_inv = _invert_h(T01)
        wc_1 = T01_inv @ np.append(wrist_center, 1)
        
        print(f"  T01 transform matrix:\n{T01}")
        print(f"  Wrist center in joint 1 frame: [{wc_1[0]:.6f}, {wc_1[1]:.6f}, {wc_1[2]:.6f}]")
        
        # Joint 2 and 3 (shoulder lift and elbow)
        # This is a 2-DOF problem in the arm plane
        wx, wy, wz = wc_1[0] - d4, wc_1[1], wc_1[2]
        print(f"  Adjusted wrist position: wx={wx:.6f}, wy={wy:.6f}, wz={wz:.6f}")
        
        # Distance from joint 2 to wrist center
        r2 = math.sqrt(wx**2 + wy**2 + wz**2)
        print(f"  Distance r2 from joint 2 to wrist center: {r2:.6f}")
        
        # Check if target is reachable
        max_reach = abs(a2) + abs(a3)
        min_reach = abs(abs(a2) - abs(a3))
        print(f"  Reachability check:")
        print(f"    |a2| = {abs(a2):.6f}, |a3| = {abs(a3):.6f}")
        print(f"    max_reach = {max_reach:.6f}, min_reach = {min_reach:.6f}")
        print(f"    r2 in range? {min_reach:.6f} <= {r2:.6f} <= {max_reach:.6f} = {min_reach <= r2 <= max_reach}")
        
        if r2 > abs(a2) + abs(a3) or r2 < abs(abs(a2) - abs(a3)):
            print(f"  SKIP: Target unreachable for this theta1")
            continue
        
        # Joint 3 solutions (elbow)
        cos_theta3 = (r2**2 - a2**2 - a3**2) / (2 * a2 * a3)
        cos_theta3_orig = cos_theta3
        cos_theta3 = _clip(cos_theta3, -1.0, 1.0)
        
        print(f"  Joint 3 calculation:")
        print(f"    cos_theta3 = ({r2:.6f}² - {a2:.6f}² - {a3:.6f}²) / (2 * {a2:.6f} * {a3:.6f})")
        print(f"    cos_theta3 = {cos_theta3_orig:.6f} (clipped to {cos_theta3:.6f})")
        
        theta3_1 = math.acos(cos_theta3)
        theta3_2 = -theta3_1
        
        print(f"    theta3_1 = {theta3_1:.6f} rad")
        print(f"    theta3_2 = {theta3_2:.6f} rad")
        
        for theta3_idx, theta3 in enumerate([theta3_1, theta3_2]):
            print(f"\n    --- Processing theta3 solution {theta3_idx+1}: {theta3:.6f} rad ---")
            
            # Joint 2 solution (shoulder lift)
            s3 = math.sin(theta3)
            k1 = a2 + a3 * math.cos(theta3)
            k2 = a3 * s3
            
            print(f"      s3 = sin({theta3:.6f}) = {s3:.6f}")
            print(f"      k1 = {a2:.6f} + {a3:.6f} * cos({theta3:.6f}) = {k1:.6f}")
            print(f"      k2 = {a3:.6f} * {s3:.6f} = {k2:.6f}")
            
            # Solve for theta2
            r_xy = math.sqrt(wx**2 + wy**2)
            print(f"      r_xy = sqrt({wx:.6f}² + {wy:.6f}²) = {r_xy:.6f}")
            
            if r_xy < 1e-6:
                print(f"      WARNING: r_xy too small, may cause numerical issues")
            
            theta2 = math.atan2(wz, r_xy) - math.atan2(k2, k1)
            print(f"      theta2 = atan2({wz:.6f}, {r_xy:.6f}) - atan2({k2:.6f}, {k1:.6f})")
            print(f"      theta2 = {theta2:.6f} rad")
            
            # Calculate orientation part (joints 4, 5, 6)
            # Get rotation matrix for first 3 joints
            T02 = get_transform_matrix(1, theta2)
            T23 = get_transform_matrix(2, theta3)
            T03 = T01 @ T02 @ T23
            
            print(f"      T03 (first 3 joints) rotation matrix:\n{T03[:3, :3]}")
            
            # Required rotation for last 3 joints
            R03 = T03[:3, :3]
            R36 = R03.T @ R
            
            print(f"      R36 (required rotation for last 3 joints):\n{R36}")
            
            # Extract joint 4, 5, 6 from R36
            # Joint 5 (wrist 2)
            cos_theta5 = R36[2, 2]
            cos_theta5_orig = cos_theta5
            cos_theta5 = _clip(cos_theta5, -1.0, 1.0)
            
            print(f"      Joint 5 calculation:")
            print(f"        cos_theta5 = R36[2,2] = {cos_theta5_orig:.6f} (clipped to {cos_theta5:.6f})")
            
            theta5_1 = math.acos(cos_theta5)
            theta5_2 = -theta5_1
            
            print(f"        theta5_1 = {theta5_1:.6f} rad")
            print(f"        theta5_2 = {theta5_2:.6f} rad")
            
            for theta5_idx, theta5 in enumerate([theta5_1, theta5_2]):
                print(f"\n        --- Processing theta5 solution {theta5_idx+1}: {theta5:.6f} rad ---")
                
                s5 = math.sin(theta5)
                print(f"          s5 = sin({theta5:.6f}) = {s5:.6f}")
                
                if abs(s5) < 1e-6:  # Singularity case
                    print(f"          SINGULARITY: |s5| < 1e-6, using arbitrary theta4=0")
                    theta4 = 0.0  # Choose arbitrary value
                    theta6 = math.atan2(R36[1, 0], R36[0, 0])
                    print(f"          theta4 = {theta4:.6f} rad (arbitrary)")
                    print(f"          theta6 = atan2({R36[1, 0]:.6f}, {R36[0, 0]:.6f}) = {theta6:.6f} rad")
                else:
                    # Joint 4 and 6
                    theta4 = math.atan2(R36[1, 2], R36[0, 2])
                    theta6 = math.atan2(R36[2, 1], -R36[2, 0])
                    print(f"          theta4 = atan2({R36[1, 2]:.6f}, {R36[0, 2]:.6f}) = {theta4:.6f} rad")
                    print(f"          theta6 = atan2({R36[2, 1]:.6f}, {-R36[2, 0]:.6f}) = {theta6:.6f} rad")
                
                # Normalize angles to [-pi, pi]
                solution = np.array([theta1, theta2, theta3, theta4, theta5, theta6])
                solution_orig = solution.copy()
                solution = (solution + math.pi) % (2 * math.pi) - math.pi
                
                print(f"          Solution before normalization: {np.degrees(solution_orig)}")
                print(f"          Solution after normalization:  {np.degrees(solution)}")
                
                # Verify solution by FK
                verification_result = verify_solution(solution, T_target)
                print(f"          Verification result: {verification_result}")
                
                if verification_result:
                    solutions.append(solution)
                    print(f"          ✓ VALID SOLUTION ADDED")
                else:
                    print(f"          ✗ SOLUTION REJECTED (failed verification)")
    
    print(f"\n=== IK SOLVER SUMMARY ===")
    print(f"Total valid solutions found: {len(solutions)}")
    for i, sol in enumerate(solutions):
        print(f"Solution {i+1}: {np.degrees(sol)}")
    
    return solutions

def verify_solution(q: np.ndarray, T_target: np.ndarray, tol: float = 1e-4) -> bool:
    """Verify IK solution by computing FK and comparing with target"""
    try:
        T_fk = forward_kinematics_check(q)
        
        # Check position error
        pos_error = np.linalg.norm(T_fk[:3, 3] - T_target[:3, 3])
        
        # Check orientation error (using rotation matrix difference)
        R_error = T_fk[:3, :3] @ T_target[:3, :3].T
        trace_val = np.trace(R_error)
        trace_clipped = _clip((trace_val - 1) / 2, -1.0, 1.0)
        angle_error = abs(math.acos(trace_clipped))
        
        print(f"            VERIFICATION:")
        print(f"              FK position: {T_fk[:3, 3]}")
        print(f"              Target pos:  {T_target[:3, 3]}")
        print(f"              Pos error:   {pos_error:.8f}m (tol: {tol:.8f})")
        print(f"              Angle error: {angle_error:.8f}rad (tol: {tol:.8f})")
        print(f"              Trace value: {trace_val:.8f} (clipped: {trace_clipped:.8f})")
        
        is_valid = pos_error < tol and angle_error < tol
        print(f"              Valid: {is_valid}")
        
        return is_valid
    except Exception as e:
        print(f"            VERIFICATION ERROR: {e}")
        return False

# ──────────────────────────────────────────────────────────
# 5) FMU Co‑Simulation wrapper (unchanged API)
# ──────────────────────────────────────────────────────────
class Model(Fmi2FMU):
    """Co‑simulation FMU exposing the revised IK solver."""

    def __init__(self, reference_to_attr=None):
        super().__init__(reference_to_attr)
        self.reference_to_attr = {
            0: 'x_t', 1: 'y_t', 2: 'z_t',
            3: 'roll_t', 4: 'pitch_t', 5: 'yaw_t',
            6: 'q1', 7: 'q2', 8: 'q3', 9: 'q4', 10: 'q5', 11: 'q6'
        }
        for attr in self.reference_to_attr.values():
            setattr(self, attr, 0.0)
        self.q_current = np.zeros(NUM_JOINTS)

    def _update_logic(self):
        """Update joint angles from pose inputs"""
        print(f"\n=== FMU UPDATE LOGIC ===")
        print(f"Input pose: x={self.x_t:.6f}, y={self.y_t:.6f}, z={self.z_t:.6f}")
        print(f"Input orientation (rad): roll={self.roll_t:.6f}, pitch={self.pitch_t:.6f}, yaw={self.yaw_t:.6f}")
        print(f"Input orientation: roll={self.roll_t:.6f} rad, pitch={self.pitch_t:.6f} rad, yaw={self.yaw_t:.6f} rad")
        
        T_target = rpy_to_matrix(self.roll_t, self.pitch_t, self.yaw_t,
                                 self.x_t, self.y_t, self.z_t)
        
        print(f"Target transformation matrix constructed:")
        print(f"T_target =\n{T_target}")
        
        all_solutions = analytical_ik_solver(T_target)

        if all_solutions:
            print(f"\n=== SOLUTION SELECTION ===")
            print(f"Found {len(all_solutions)} valid solutions")
            
            # Select first valid solution (TODO: implement better selection)
            selected_q = all_solutions[0]
            print(f"Selected solution: {selected_q} rad")
            for i in range(NUM_JOINTS):
                old_val = getattr(self, f'q{i+1}')
                setattr(self, f'q{i+1}', selected_q[i])
                print(f"  q{i+1}: {old_val:.6f} -> {selected_q[i]:.6f} rad")
            self.q_current = selected_q
            print(f"✓ IK solution successfully applied")
        else:
            print(f"\n=== NO SOLUTION FOUND ===")
            print("Warning: No IK solution found for target pose")
            for i in range(NUM_JOINTS):
                val = getattr(self, f'q{i+1}')
                print(f"  q{i+1}: {val:.6f} rad")
        print(f"=== END UPDATE LOGIC ===\n")

    # FMU interface methods (unchanged)
    def serialize(self):
        return Fmi2Status.ok, pickle.dumps(self.q_current)

    def deserialize(self, bytes_data):
        self.q_current = pickle.loads(bytes_data)
        return Fmi2Status.ok

    def get_variable_name(self, vr):
        return self.reference_to_attr[vr]

    def set_real(self, refs, values):
        for ref, val in zip(refs, values):
            setattr(self, self.get_variable_name(ref), val)
        return Fmi2Status.ok

    def get_real(self, refs):
        return [getattr(self, self.get_variable_name(ref)) for ref in refs], Fmi2Status.ok

    def enter_initialization_mode(self): 
        self._update_logic()
        return Fmi2Status.ok
    
    def exit_initialization_mode(self):  # FIXED: Added 'self' parameter
        self._update_logic()
        return Fmi2Status.ok
    
    def do_step(self, current_time, step_size, no_set_fmu_state_prior_to_current_point):
        self._update_logic()
        return Fmi2Status.ok
    
    def reset(self):
        for attr in self.reference_to_attr.values():
            setattr(self, attr, 0.0)
        self.q_current = np.zeros(NUM_JOINTS)
        return Fmi2Status.ok
    
    def setup_experiment(self, start_time, stop_time=None, tolerance=None): 
        return Fmi2Status.ok
    
    def terminate(self): 
        return Fmi2Status.ok

# ──────────────────────────────────────────────────────────
# 6) Testing utilities
# ──────────────────────────────────────────────────────────

def test_round_trip(q_test: np.ndarray, verbose: bool = True) -> tuple:
    """Test FK->IK->FK round trip"""
    # Forward kinematics
    T_target = forward_kinematics_check(q_test)
    
    # Inverse kinematics
    solutions = analytical_ik_solver(T_target)
    
    if not solutions:
        if verbose:
            print("No IK solutions found!")
        return False, None, None
    
    # Test each solution
    best_solution = None
    min_error = float('inf')
    
    for i, q_sol in enumerate(solutions):
        # Forward kinematics of solution
        T_verify = forward_kinematics_check(q_sol)
        
        # Calculate errors
        pos_error = np.linalg.norm(T_verify[:3, 3] - T_target[:3, 3])
        R_error = T_verify[:3, :3] @ T_target[:3, :3].T
        angle_error = abs(math.acos(_clip((np.trace(R_error) - 1) / 2, -1.0, 1.0)))
        
        total_error = pos_error + angle_error
        
        if verbose:
            print(f"Solution {i+1}: pos_error={pos_error:.6f}m, angle_error={angle_error:.6f}rad")
            print(f"  Original: {q_test} rad")
            print(f"  Solution: {q_sol} rad")
        
        if total_error < min_error:
            min_error = total_error
            best_solution = q_sol
    
    success = min_error < 1e-4
    return success, best_solution, min_error

if __name__ == "__main__":
    print("Testing revised UR10e IK model...")
    
    # Test with a simple configuration
    q_test = np.array([0.0, -math.pi/2, 0.0, 0.0, math.pi/2, 0.0])
    success, solution, error = test_round_trip(q_test)
    
    if success:
        print(f"Round-trip test PASSED with error: {error:.6f}")
    else:
        print(f"Round-trip test FAILED with error: {error:.6f}")