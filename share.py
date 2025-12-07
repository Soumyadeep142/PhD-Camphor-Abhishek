import numpy as np
import matplotlib.pyplot as plt
import imageio
from scipy.ndimage import gaussian_filter
from shapely.geometry import Polygon
import os
import csv
import tempfile
import math
import sys

# ------------------------------
# Utility functions
# ------------------------------
def angle_diff(a, b):
    """Signed smallest difference a - b in [-pi, pi]."""
    return (a - b + np.pi) % (2*np.pi) - np.pi

# ------------------------------
# Parameters
# ------------------------------
L = 260
scale = 1
triangle_height_mm = 17
triangle_base_mm = 6
triangle_height = int(triangle_height_mm * scale)
triangle_base = int(triangle_base_mm * scale)
dx = 1.0
dt = 0.016
D = 1
k = 2
A =float(sys.argv[1])
tau = float(sys.argv[5])
gamma0 = 1
v_mobility = 0.01
rot_mobility = 1.0
n_frames = 14400
coated_heights_mm = [4]
coated_heights_px = [int(h * scale) for h in coated_heights_mm]

N_particles =  int(sys.argv[2])
edge_repulsion_strength = 0
alignment_strength = float(sys.argv[4])
align_dist = float(sys.argv[6])
rep_dist = 20
align_angle = np.pi
align_angle_deg = 180
rep_param = 1.0
align_param = 10.0
delta = 4 * np.pi / 36
delta_deg = 20
mtv_power = 2.0
nt = 0.02
nr = 0.01
run_id=int(sys.argv[7])
# Circular boundary
cx, cy = L/2, L/2
R = 90
R_out = 100
wall_align = float(sys.argv[3])

# collision rotation kick
#rot_theta = 0.00005

ensemble_dir = f"outputs/Align_Dist_{align_dist}_Align_Str_{alignment_strength}/N_{N_particles}_A_{A}"
os.makedirs(ensemble_dir, exist_ok=True)
video_dir = ensemble_dir

# ------------------------------
# Geometry functions
# ------------------------------
def triangle_polygon(x, y, theta):
    """Return a shapely Polygon for a triangle centered at (x,y) with orientation theta."""
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
    """Generate camphor source points on coated region."""
    points = []
    base_x = x - np.cos(theta) * (triangle_height / 3)
    base_y = y - np.sin(theta) * (triangle_height / 3)
    wx = np.cos(theta + np.pi/2)
    wy = np.sin(theta + np.pi/2)
    for h in range(source_height):
        frac = h / max(1.0, triangle_height)
        width = max(1, int(triangle_base * (1 - frac)))
        cx_i = base_x + np.cos(theta) * h
        cy_i = base_y + np.sin(theta) * h
        for w in range(-width // 2, width // 2 + 1):
            px = cx_i + w * wx
            py = cy_i + w * wy
            dx_local = (px - x)
            dy_local = (py - y)
            norm = np.hypot(dx_local, dy_local)
            if norm > 1e-8:
                nx, ny = dx_local / norm, dy_local / norm
            else:
                nx, ny = 0.0, 0.0
            points.append((px, py, nx, ny))
    return points

def add_camphor(c, coated_points, A_de):
    """Add camphor A_t at integer grid locations corresponding to coated_points."""
    for px, py, *_ in coated_points:
        ix = int(round(px)) % L
        iy = int(round(py)) % L
        c[iy, ix] += A_de
    return c

def compute_force(coated_points, c_field, x, y):
    """Compute net translational force by summing local gamma * normal."""
    F = np.zeros(2)
    for px, py, nx, ny in coated_points:
        ix = int(round(px)) % L
        iy = int(round(py)) % L
        gamma = gamma0 - A_de * c_field[iy, ix]/N_particles
        F += gamma * np.array([nx, ny])
    return F

# ------------------------------
# SAT Collision
# ------------------------------
def sat_collision(poly1: Polygon, poly2: Polygon):
    """Separating Axis Theorem for polygon collision."""
    polygons = [poly1, poly2]
    min_overlap = np.inf
    mtv_axis = None
    for poly in polygons:
        coords = np.array(poly.exterior.coords[:-1])
        for i in range(len(coords)):
            edge = coords[(i+1)%len(coords)] - coords[i]
            axis = np.array([-edge[1], edge[0]])
            axis_norm = np.linalg.norm(axis)
            if axis_norm < 1e-8:
                continue
            axis /= axis_norm
            def project(polygon):
                pts = np.array(polygon.exterior.coords[:-1])
                dots = pts.dot(axis)
                return np.min(dots), np.max(dots)
            minA, maxA = project(poly1)
            minB, maxB = project(poly2)
            overlap = min(maxA, maxB) - max(minA, minB)
            if overlap <= 0:
                return False, np.array([0.0, 0.0])
            if overlap < min_overlap:
                min_overlap = overlap
                mtv_axis = axis
    if mtv_axis is None:
        return False, np.array([0.0, 0.0])
    center1 = np.array(poly1.centroid.coords[0])
    center2 = np.array(poly2.centroid.coords[0])
    direction = center2 - center1
    if np.dot(direction, mtv_axis) < 0:
        mtv_axis = -mtv_axis
    mtv = mtv_axis * mtv_power * min_overlap
    return True, mtv

# ------------------------------
# CSV logging setup
# ------------------------------
particle_logs = []
particle_writers = []
for pid in range(N_particles):
    tmp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, newline='')
    writer = csv.writer(tmp_file)
    writer.writerow(["particle_no", "x", "y", "frame_no"])
    particle_logs.append(tmp_file)
    particle_writers.append(writer)

# ------------------------------
# Simulation
# ------------------------------
fig, ax = plt.subplots(figsize=(6,6), dpi=100)

for source_height in coated_heights_px:
    c = np.zeros((L, L))

    fixed_positions = [
        (cx - 50, cy+10),  # Particle 1
        (cx + 50, cy)      # Particle 2
    ]
    initial_orientations = [2*np.pi/3, np.pi/18]
    particles = []
    for i in range(N_particles):
        x, y = fixed_positions[i] if i < len(fixed_positions) else (np.random.uniform(cx-R/2, cx+R/2),
                                                                    np.random.uniform(cy-R/2, cy+R/2))
        theta = initial_orientations[i] if i < len(initial_orientations) else np.random.uniform(0, 2*np.pi)
        particles.append([x, y, theta])

    video_path = f"{video_dir}/Expo_N_particles_{N_particles}_A_{A}_k_{k}_alignment_strength_{alignment_strength}_align_dist_{align_dist}_delta_deg_{delta_deg}_wall_{wall_align}_tau_{tau}_run_{run_id}.mp4"
    writer = imageio.get_writer(video_path, fps=60)

    for t in range(n_frames):
        t_phy = t*dt
        A_de = A*np.exp(-t_phy / tau)
        # diffusion-decay
        c = c + D * (np.roll(c, 1, axis=0) + np.roll(c, -1, axis=0) +
                     np.roll(c, 1, axis=1) + np.roll(c, -1, axis=1) - 4 * c) / dx**2 * dt
        c -= k * c * dt

        # add camphor
        for i, (x, y, theta) in enumerate(particles):
            coated_points = get_coated_points(x, y, theta, source_height)
            c = add_camphor(c, coated_points, A_de)

        # smooth
        c = gaussian_filter(c, sigma=0.5)

        # precompute shapes
        triangle_shapes = [triangle_polygon(px, py, ptheta) for px, py, ptheta in particles]
        new_particles = []

        # --- Wall alignment first ---
        wall_thetas = np.zeros(N_particles)
        for i, (x, y, theta) in enumerate(particles):
            dist_to_center = np.hypot(x - cx, y - cy)
            if dist_to_center < 1e-8:
                dist_to_center = 1e-8
            nx = (x - cx) / dist_to_center
            ny = (y - cy) / dist_to_center
            tx, ty = -ny, nx
            tx_cw, ty_cw = ny, -nx   
            dist_from_wall = R - dist_to_center
            sigma_wall = 4.0
            align_factor = np.exp(-(dist_from_wall**2) / (2 * sigma_wall**2))
            tangent_theta_ccw = np.arctan2(ty, tx) % (2*np.pi)
            tangent_theta_cw  = np.arctan2(ty_cw, tx_cw)  % (2*np.pi)
            if abs(angle_diff(theta, tangent_theta_ccw)) < abs(angle_diff(theta, tangent_theta_cw)):
                target_theta = tangent_theta_ccw
            else:
                target_theta = tangent_theta_cw
            wall_thetas[i] = theta - wall_align * align_factor * np.sin(angle_diff(theta, target_theta))

        # --- Mutual alignment torque computation ---
        taus = np.zeros(N_particles)
        forces = np.zeros((N_particles, 2)) 
        for i in range(N_particles):
            x_i, y_i, theta_i = particles[i]
            for j in range(i + 1, N_particles):
                x_j, y_j, theta_j = particles[j]
                dx_ij = x_i - x_j
                dy_ij = y_i - y_j
                center_dist = np.hypot(dx_ij, dy_ij)
                if center_dist < align_dist:
                    diff = theta_j - theta_i
                    taus[i] += alignment_strength * np.sin(diff - delta)
                    taus[j] -= alignment_strength * np.sin(diff - delta)

        # # --- Repulsion (use center_dist instead of d) ---
        #         if center_dist < rep_dist and center_dist > 1e-8:
        #             rij = np.array([dx_ij, dy_ij])         # vector from j -> i
        #         # magnitude of repulsive force (use center_dist, not undefined d)
        #             F_mag = edge_repulsion_strength * np.exp(-(center_dist / rep_param)**2)
        #     # unit vector from j to i
        #             F_vec = F_mag * (rij / center_dist)
        #     # apply equal and opposite forces
        #             forces[i] += F_vec
        #             forces[j] -= F_vec

        # --- integrate motion ---
        for i, (x, y, theta) in enumerate(particles):
            coated_points = get_coated_points(x, y, theta, source_height)
            F = compute_force(coated_points, c, x, y)
            F_total = F

            vx, vy = v_mobility * F_total
            omega = rot_mobility * taus[i]

            new_x = x + vx * dt + np.random.normal(0, nt)*np.sqrt(dt)
            new_y = y + vy * dt + np.random.normal(0, nt)*np.sqrt(dt)
            new_theta = (wall_thetas[i] + omega * dt + np.random.normal(0, nr)*np.sqrt(dt)) % (2*np.pi)   #np.random.normal(0, nr)*np.sqrt(dt)
            new_poly = triangle_polygon(new_x, new_y, new_theta)

            # Collision resolution
            for j_idx in range(len(particles)):
                if i == j_idx:
                    continue
                ox, oy, otheta = particles[j_idx]
                other_poly = triangle_polygon(ox, oy, otheta)
                collided, mtv = sat_collision(new_poly, other_poly)
                if collided:
                    correction = 0.5 * mtv
                    new_x -= correction[0]
                    new_y -= correction[1]
                    particles[j_idx][0] += correction[0]
                    particles[j_idx][1] += correction[1]
                    #new_theta = (new_theta + rot_theta) % (2*np.pi)
                    new_theta = (new_theta) % (2*np.pi)
                    new_poly = triangle_polygon(new_x, new_y, new_theta)

            if np.hypot(new_x - cx, new_y - cy) > R:
                nx = (new_x - cx) / np.hypot(new_x - cx, new_y - cy)
                ny = (new_y - cy) / np.hypot(new_x - cx, new_y - cy)
                new_x = cx + nx * (R - 1e-3)
                new_y = cy + ny * (R - 1e-3)

            particle_writers[i].writerow([i+1, new_x, new_y, t+1])
            new_particles.append([new_x, new_y, new_theta])

        particles = new_particles

        # Visualization
        if True:
            ax.clear()
            ax.set_xlim(0, L)
            ax.set_ylim(0, L)
            ax.set_aspect('equal')
            ax.axis('off')
            ax.imshow(c, cmap='viridis', origin='lower', vmin=0, vmax=5)
            circle = plt.Circle((cx, cy), R_out, color='red', fill=False)
            ax.add_patch(circle)
            for px, py, ptheta in particles:
                poly = triangle_polygon(px, py, ptheta)
                x_poly, y_poly = poly.exterior.xy
                ax.fill(x_poly, y_poly, color='white', edgecolor='black', lw=0.5)
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            writer.append_data(image)

    writer.close()
    print(f"Movie saved to: {video_path}")

# Merge CSVs
csv_path = f"{video_dir}/Expo_N_particles_{N_particles}_A_{A}_k_{k}_alignment_strength_{alignment_strength}_align_dist_{align_dist}_delta_deg_{delta_deg}_wall_{wall_align}_tau_{tau}run_{run_id}.csv"
with open(csv_path, "w", newline='') as fout:
    writer = csv.writer(fout)
    writer.writerow(["particle_no","x","y","frame_no"])
    for f in particle_logs:
        with open(f.name, "r") as fin:
            reader = csv.reader(fin)
            next(reader)
            for row in reader:
                writer.writerow(row)
        os.remove(f.name)

print(f"CSV saved to: {csv_path}")
