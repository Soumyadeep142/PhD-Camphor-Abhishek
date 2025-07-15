import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from shapely.geometry import Polygon
import tempfile
import csv
import os
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
n_frames = 14400
nt = 0.2
nr = 1.0
N_particles = int(sys.argv[2])
edge_repulsion_strength = int(sys.argv[3])
alignment_strength = int(sys.argv[4])
alignment_range = 2*triangle_base

cx, cy = L / 2, L / 2

os.makedirs(f"outputs/Align_{alignment_strength}_Rep_{edge_repulsion_strength}", exist_ok=True)

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

def get_coated_points(x, y, theta):
    points = []
    base_x = x - np.cos(theta) * (triangle_height / 3)
    base_y = y - np.sin(theta) * (triangle_height / 3)
    wx = np.cos(theta + np.pi/2)
    wy = np.sin(theta + np.pi/2)
    for h in range(triangle_height):
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

def add_camphor(c, points):
    for px, py, *_ in points:
        c[py % L, px % L] += A
    return c

def compute_force(points, c_field, x, y):
    F = np.zeros(2)
    tau = 0.0
    for px, py, nx, ny in points:
        gamma = gamma0 - A * c_field[py % L, px % L]
        dx_local = ((px - x + L/2) % L) - L/2
        dy_local = ((py - y + L/2) % L) - L/2
        r = np.array([dx_local, dy_local])
        n = np.array([nx, ny])
        F += gamma * n
       
    return F

particle_logs = []
particle_writers = []
for pid in range(N_particles):
    tmp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, newline='')
    writer = csv.writer(tmp_file)
    writer.writerow(["particle_no", "x", "y", "frame_no"])
    particle_logs.append(tmp_file)
    particle_writers.append(writer)

c = np.zeros((L, L))
particles = []
for _ in range(N_particles):
    x = np.random.uniform(L/4, 3*L/4)
    y = np.random.uniform(L/4, 3*L/4)
    theta = np.random.uniform(0, 2*np.pi)
    particles.append([x, y, theta])

for t in range(n_frames):
    c += D * (np.roll(c, 1, axis=0) + np.roll(c, -1, axis=0) +
              np.roll(c, 1, axis=1) + np.roll(c, -1, axis=1) - 4 * c) / dx**2 * dt
    c -= k * c * dt

    for i, (x, y, theta) in enumerate(particles):
        coated_points = get_coated_points(x, y, theta)
        c = add_camphor(c, coated_points)

    c = gaussian_filter(c, sigma=0.5)
    triangle_shapes = [triangle_polygon(px, py, ptheta) for px, py, ptheta in particles]
    new_particles = []

    for i, (x, y, theta) in enumerate(particles):
        coated_points = get_coated_points(x, y, theta)
        F= compute_force(coated_points, c, x, y)
        tau=0

        for j in range(len(particles)):
            if i == j:
                continue
            poly_i = triangle_shapes[i]
            poly_j = triangle_shapes[j]
            d = poly_i.distance(poly_j)

            if d < 1e-2:
                rij = np.array([particles[i][0] - particles[j][0], particles[i][1] - particles[j][1]])
                norm = np.linalg.norm(rij)
                if norm > 0:
                    F += (edge_repulsion_strength * np.exp(-(d / 2)**2)) * (rij / norm)
            elif d < alignment_range:
                theta_i = particles[i][2]
                theta_j = particles[j][2]
                tau += alignment_strength * np.sin(theta_j - theta_i) * np.exp(-(d / 2)**2)

        noise = np.random.normal(0, nt, 2)
        vx, vy = v_mobility * F + noise
        omega = tau + np.random.normal(0, nr)

        x += vx * dt
        y += vy * dt
        theta = (theta + omega * dt) % (2 * np.pi)

        x = np.clip(x, 1, L - 2)
        y = np.clip(y, 1, L - 2)

        particle_writers[i].writerow([i + 1, x, y, t + 1])
        new_particles.append([x, y, theta])

    particles = new_particles

for f in particle_logs:
    f.close()

with open(f"outputs/Align_{alignment_strength}_Rep_{edge_repulsion_strength}/particle_trajectories_N_{N_particles}_A_{A}_rep_align_{alignment_strength}_{edge_repulsion_strength}.csv", "w", newline='') as fout:
    writer = csv.writer(fout)
    writer.writerow(["particle_no", "x", "y", "frame_no"])
    for f in particle_logs:
        with open(f.name, "r") as fin:
            reader = csv.reader(fin)
            next(reader)
            for row in reader:
                writer.writerow(row)
        os.remove(f.name)
