#!/usr/bin/env python3
"""
Test script to verify Forward Kinematics and IK accuracy
"""

import numpy as np
import math
from ikpy.chain import Chain
from ikpy.link import DHLink

# UR10e DH parameters
UR10E_DH = {
    "a":     [0.0,   -0.613, -0.572, 0.0,   0.0,   0.0],
    "d":     [0.181, 0.0,   0.0,   0.174, 0.120, 0.117],
    "alpha": [math.pi/2, 0.0, 0.0, math.pi/2, -math.pi/2, 0.0],
}

def build_chain():
    """Build IKPy chain for UR10e"""
    a, d, alpha = UR10E_DH["a"], UR10E_DH["d"], UR10E_DH["alpha"]
    links = []
    for i in range(6):
        links.append(DHLink(
            name=f"joint_{i+1}",
            d=float(d[i]),
            a=float(a[i]),
            alpha=float(alpha[i]),
            theta=0.0,
            bounds=(-2*math.pi, 2*math.pi)
        ))
    return Chain(name="ur10e", links=links)

def test_forward_kinematics():
    """Test FK against your real robot data"""
    chain = build_chain()
    
    # Your test cases: [q1, q2, q3, q4, q5, q6] and expected [x, y, z]
    test_cases = [
        # Test 1: Zero configuration
        {"q": [0, 0, 0, 0, 0, 0], 
         "expected": [-1.183, -0.2907, 0.0603]},
        
        # Test 2
        {"q": [0, -1.57, 1.57, 0, 0, 0],
         "expected": [-0.5712, -0.2896, 0.6721]},
         
        # Test 3
        {"q": [0, -1.57, 1.57, 0, 1.57, 0],
         "expected": [-0.686, -0.174, 0.6726]},
         
        # Test 4
        {"q": [1.57, -1.57, 1.57, 0, 0, 0],
         "expected": [0.2896, -0.5712, 0.6721]},
         
        # Test 5
        {"q": [0.349, -0.523, 1.047, -0.785, 0.523, 1.57],
         "expected": [-0.95, -0.6388, 0.0997]},
    ]
    
    print("FORWARD KINEMATICS VERIFICATION")
    print("=" * 60)
    
    for i, test in enumerate(test_cases, 1):
        q = test["q"]
        expected_pos = test["expected"]
        
        # Compute FK
        T = chain.forward_kinematics(q)
        computed_pos = T[:3, 3]
        
        # Calculate error
        error = np.linalg.norm(computed_pos - expected_pos)
        
        print(f"\nTest {i}:")
        print(f"  Joint angles: {q}")
        print(f"  Expected pos: {expected_pos}")
        print(f"  Computed pos: [{computed_pos[0]:.4f}, {computed_pos[1]:.4f}, {computed_pos[2]:.4f}]")
        print(f"  Position error: {error:.6f} m")
        
        if error > 0.01:  # 10mm threshold
            print(f"  ⚠️  LARGE ERROR! DH parameters might be incorrect!")

def matrix_to_euler_xyz(R):
    """Extract Euler angles from rotation matrix (XYZ convention)"""
    # This assumes Rz(yaw) * Ry(pitch) * Rx(roll)
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6
    
    if not singular:
        x = math.atan2(R[2,1], R[2,2])  # roll
        y = math.atan2(-R[2,0], sy)     # pitch  
        z = math.atan2(R[1,0], R[0,0])  # yaw
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
        
    return x, y, z

def test_ik_accuracy():
    """Test if IK can recover the original joint angles"""
    chain = build_chain()
    
    print("\n\nINVERSE KINEMATICS RECOVERY TEST")
    print("=" * 60)
    print("Testing if IK can recover original joint angles from FK poses")
    
    test_configs = [
        [0, 0, 0, 0, 0, 0],
        [0, -1.57, 1.57, 0, 0, 0],
        [0.349, -0.523, 1.047, -0.785, 0.523, 1.57],
    ]
    
    for i, q_original in enumerate(test_configs, 1):
        print(f"\nTest {i}:")
        print(f"  Original joints: {[f'{q:.3f}' for q in q_original]}")
        
        # Forward kinematics
        T_target = chain.forward_kinematics(q_original)
        
        # Try to recover with IK
        q_recovered = chain.inverse_kinematics_frame(
            T_target,
            initial_position=q_original,  # Use original as seed
            orientation_mode="all"
        )
        
        if len(q_recovered) > 6:
            q_recovered = q_recovered[:6]
            
        # Compare
        joint_error = np.linalg.norm(np.array(q_recovered) - np.array(q_original))
        
        print(f"  Recovered joints: {[f'{q:.3f}' for q in q_recovered]}")
        print(f"  Joint error: {joint_error:.6f} rad")
        
        # Verify FK of recovered solution
        T_check = chain.forward_kinematics(q_recovered)
        pos_error = np.linalg.norm(T_check[:3, 3] - T_target[:3, 3])
        print(f"  Position error after recovery: {pos_error:.6f} m")
        
        if joint_error > 0.1:
            print(f"  ⚠️  IK found a different solution branch!")

if __name__ == "__main__":
    test_forward_kinematics()
    test_ik_accuracy()