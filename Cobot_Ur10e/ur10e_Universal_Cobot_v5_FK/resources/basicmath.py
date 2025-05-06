import math

# Given joint angles in degrees
angles_degrees = [20, 40, 60, 50, 70, 10]

# Convert degrees to radians and calculate sine and cosine for each angle
angles_radians = [math.radians(angle) for angle in angles_degrees]
cosines = [math.cos(angle) for angle in angles_radians]
sines = [math.sin(angle) for angle in angles_radians]

# Print the results
for i, (angle_deg, angle_rad, cos_val, sin_val) in enumerate(zip(angles_degrees, angles_radians, cosines, sines), 1):
    print(f"θ{i} = {angle_deg}° (radians: {angle_rad:.4f}) -> cos(θ{i}) = {cos_val:.4f}, sin(θ{i}) = {sin_val:.4f}")