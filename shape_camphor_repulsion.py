import numpy as np
import matplotlib.pyplot as plt
import imageio
from scipy.ndimage import gaussian_filter
from shapely.geometry import Polygon
import os
import csv
import tempfile
import sys

# Parameters
L = 260
scale = 1
triangle_height_mm = 17
triangle_base_mm = 6
triangle_height = int(triangle_height_mm * scale)
triangle_base = int(triangle_base_mm * scale)
dx = 1.0
dt = 0.1
D = 1.0
k = 0.8
A = float(sys.argv[1])
gamma0 = 1.0
v_mobility = 0.002
rot_mobility = 0.000
n_frames = 14400
coated_heights_mm = [4]
coated_heights_px = [int(h * scale) for h in coated_heights_mm]
nt = 0.2
nr = 1.0

N_particles = int(sys.argv[2])
s2 = (triangle_base / 2)**2 + triangle_height**2
circum_radius = s2 / (2 * triangle_height)
interaction_sigma = 2 * circum_radius
edge_repulsion_strength = int(sys.argv[3])
edge_sigma = triangle_base
damping = 0.2
molarity = 1.0
alignment_strength = float(sys.argv[4])
wall_align = 0.8
alignment_range = 0.5 * circum_radius

R = 100
cx, cy = L / 2, L / 2

os.makedirs(f"outputs/Rep_{edge_repulsion_strength}_Align_{alignment_strength}", exist_ok=True)

def triangle_polygon(x, y, theta):
    tip = (x + (2/3) * triangle_height * np.cos(theta),
           y + (2/3) * triangle_height * np.sin(theta))
    base_center = (x - (1/3) * triangle_height * np.cos(theta),
                   y - (1/3) * triangle_height * np.sin(theta))
    wx = np.cos(theta + np.pi/2)
    wy = np.sin(theta + np.pi/2)
    left = (base_center[0] - triangle_base/2 * wx, base_center[1] - triangle_base/2 * wy)
    right = (base_center[0] + triangle_base/2 * wx, base_center[1] + triangle_base/2 * wy)
    return Polygon([left, tip, right])

def get_coated_points(x, y, theta, source_height):
    points = []
    base_x = x - np.cos(theta) * (triangle_height / 3)
    base_y = y - np.sin(theta) * (triangle_height / 3)
    wx = np.cos(theta + np.pi/2)
    wy = np.sin(theta + np.pi/2)
    for h in range(source_height):
        frac = h / triangle_height
        width = max(1, int(triangle_base * (1 - frac**2)))
        cx_i = base_x + np.cos(theta) * h
        cy_i = base_y + np.sin(theta) * h
        for w in range(-width // 2, width // 2 + 1):
            px = cx_i + w * wx
            py = cy_i + w * wy
            dx_local = ((px - x + L/2) % L) - L/2
            dy_local = ((py - y + L/2) % L) - L/2
            norm = np.sqrt(dx_local**2 + dy_local**2)
            if norm > 1e-8:
                nx, ny = dx_local / norm, dy_local / norm
            else:
                nx, ny = 0.0, 0.0
            points.append((int(px) % L, int(py) % L, nx, ny))
    return points

def add_camphor(c, coated_points):
    for px, py, *_ in coated_points:
        c[py % L, px % L] += A
    return c

def compute_force(coated_points, c_field, x, y):
    F = np.zeros(2)
    for px, py, nx, ny in coated_points:
        gamma = gamma0 - A * c_field[py % L, px % L]
        F += gamma * np.array([nx, ny])
    return F

# Create temp files for logging particle data
particle_logs = []
particle_writers = []
for pid in range(N_particles):
    tmp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, newline='')
    writer = csv.writer(tmp_file)
    writer.writerow(["particle_no", "x", "y", "frame_no"])
    particle_logs.append(tmp_file)
    particle_writers.append(writer)

for source_height in coated_heights_px:
    c = np.zeros((L, L))
    particles = []
    for _ in range(N_particles):
        x = np.random.uniform(L/4, 3*L/4)
        y = np.random.uniform(L/4, 3*L/4)
        theta = np.random.uniform(0, 2*np.pi)
        particles.append([x, y, theta])

    for t in range(n_frames):
        c = c + D * (np.roll(c, 1, axis=0) + np.roll(c, -1, axis=0) +
                     np.roll(c, 1, axis=1) + np.roll(c, -1, axis=1) - 4 * c) / dx**2 * dt
        c -= k * c * dt

        for i, (x, y, theta) in enumerate(particles):
            coated_points = get_coated_points(x, y, theta, source_height)
            c = add_camphor(c, coated_points)

        c = gaussian_filter(c, sigma=0.5)
        triangle_shapes = [triangle_polygon(px, py, ptheta) for px, py, ptheta in particles]
        new_particles = []

        for i, (x, y, theta) in enumerate(particles):
            coated_points = get_coated_points(x, y, theta, source_height)
            F = compute_force(coated_points, c, x, y)
            tau = 0

            for j in range(len(particles)):
                if i == j:
                    continue

                poly_i = triangle_shapes[i]
                poly_j = triangle_shapes[j]
                d = poly_i.distance(poly_j)

                if d < 0.01:
                    rij = np.array([x - particles[j][0], y - particles[j][1]])
                    norm = np.linalg.norm(rij)
                    if norm > 0:
                        F += (edge_repulsion_strength * np.exp(-(d / 2)**2)) * (rij / norm)

                    # Alignment torque
                    theta_j = particles[j][2]
                    angle_diff = (theta_j - theta + np.pi) % (2 * np.pi) - np.pi
                    alignment_factor = np.exp(-(d / alignment_range)**2)
                    tau += alignment_strength * np.sin(angle_diff) * alignment_factor

            vx, vy = v_mobility * F
            vx += np.random.normal(0, nt)
            vy += np.random.normal(0, nt)
            vx *= (1 - damping)
            vy *= (1 - damping)

            omega = rot_mobility * tau + np.random.normal(0, nr)
            omega *= (1 - damping)

            x += vx * dt
            y += vy * dt
            theta = (theta + omega * dt) % (2 * np.pi)

            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            if dist >= R:
                nx = (x - cx) / dist
                ny = (y - cy) / dist
                tx = -ny
                ty = nx
                tangent_theta = np.arctan2(ty, tx)
                theta += wall_align * np.sin(tangent_theta - theta)
                x = cx + nx * (R - 0.01 * interaction_sigma)
                y = cy + ny * (R - 0.01 * interaction_sigma)

            x = np.clip(x, 1, L - 2)
            y = np.clip(y, 1, L - 2)

            particle_writers[i].writerow([i + 1, x, y, t + 1])
            new_particles.append([x, y, theta])

        particles = new_particles

# Close temp files
for f in particle_logs:
    f.close()

# Combine all logs into final CSV
output_path = f"outputs/Rep_{edge_repulsion_strength}_Align_{alignment_strength}/particle_trajectories_N_{N_particles}_A_{A}_rep_{edge_repulsion_strength}_Al_{alignment_strength}.csv"
with open(output_path, "w", newline='') as fout:
    writer = csv.writer(fout)
    writer.writerow(["particle_no", "x", "y", "frame_no"])
    for f in particle_logs:
        with open(f.name, "r") as fin:
            reader = csv.reader(fin)
            next(reader)  # skip header
            for row in reader:
                writer.writerow(row)
        os.remove(f.name)

