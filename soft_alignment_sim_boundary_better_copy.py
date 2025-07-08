import numpy as np
import matplotlib.pyplot as plt
import imageio
from scipy.ndimage import gaussian_filter
import os

# Parameters
L = 260
scale = 1
triangle_height_mm = 17
triangle_base_mm = 6
triangle_height = int(triangle_height_mm * scale)
triangle_base = int(triangle_base_mm * scale)
dx = 1.0
dt = 0.01
D = 1.0
k = 0.8
A = 10.0
gamma0 = 1.0
v_mobility = 0.2
rot_mobility = 0.003
n_frames = 600
coated_heights_mm = [4]
coated_heights_px = [int(h * scale) for h in coated_heights_mm]
nt = 0.2
nr = 1.0

N_particles = 20
s2 = (triangle_base / 2)**2 + triangle_height**2
circum_radius = s2 / (2 * triangle_height)
interaction_sigma =2 * circum_radius
repulsion_strength = 1500.0
damping = 0.2
alignment_strength = 0.3
alignment_range = 5*circum_radius

R = L / 2 - 30
cx, cy = L / 2, L / 2

os.makedirs("outputs", exist_ok=True)

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

def compute_force_and_torque(coated_points, c_field, x, y):
    F = np.zeros(2)
    tau = 0.0
    for px, py, nx, ny in coated_points:
        gamma = gamma0 - A * c_field[py % L, px % L]
        dx_local = ((px - x + L/2) % L) - L/2
        dy_local = ((py - y + L/2) % L) - L/2
        r = np.array([dx_local, dy_local])
        n = np.array([nx, ny])
        F += gamma * n
        tau += np.cross(r, gamma * n)
    return F, tau

def compute_pairwise_repulsion(x_i, x_j):
    dx = x_j[0] - x_i[0]
    dy = x_j[1] - x_i[1]
    r = np.sqrt(dx**2 + dy**2)
    #if r < 1e-5:
    #    return np.zeros(2)
    force_mag = repulsion_strength * np.exp(-(r / interaction_sigma)**2)
    fx, fy = -force_mag * dx / r, -force_mag * dy / r
    return np.array([fx, fy])

def compute_alignment(particles, i):
    x, y, theta = particles[i]
    sin_sum, cos_sum = 0.0, 0.0
    for j, (xj, yj, thetaj) in enumerate(particles):
        if i == j:
            continue
        dx, dy = xj - x, yj - y
        dist = np.sqrt(dx**2 + dy**2)
        if dist < alignment_range:
            sin_sum += np.sin(thetaj)
            cos_sum += np.cos(thetaj)
    if sin_sum == 0 and cos_sum == 0:
        return 0.0
    avg_theta = np.arctan2(sin_sum, cos_sum)
    return alignment_strength * np.sin(avg_theta - theta)

for source_height in coated_heights_px:
    c = np.zeros((L, L))
    particles = []
    for _ in range(N_particles):
        x = np.random.uniform(L/4, 3*L/4)
        y = np.random.uniform(L/4, 3*L/4)
        theta = np.random.uniform(0, 2*np.pi)
        particles.append([x, y, theta])

    frames = []
    for t in range(n_frames):
        c = c + D * (np.roll(c, 1, axis=0) + np.roll(c, -1, axis=0) +
                     np.roll(c, 1, axis=1) + np.roll(c, -1, axis=1) - 4 * c) / dx**2 * dt
        c -= k * c * dt

        new_particles = []
        for i, (x, y, theta) in enumerate(particles):
            coated_points = get_coated_points(x, y, theta, source_height)
            c = add_camphor(c, coated_points)

        c = gaussian_filter(c, sigma=0.5)

        for i, (x, y, theta) in enumerate(particles):
            coated_points = get_coated_points(x, y, theta, source_height)
            F, tau = compute_force_and_torque(coated_points, c, x, y)

            for j, (xj, yj, _) in enumerate(particles):
                if i != j:
                    F += compute_pairwise_repulsion([x, y], [xj, yj])

            vx, vy = v_mobility * F
            vx += np.random.normal(0, nt)
            vy += np.random.normal(0, nt)
            vx *= (1 - damping)
            vy *= (1 - damping)

            omega = rot_mobility * tau + compute_alignment(particles, i) + np.random.normal(0, nr)
            omega *= (1 - damping)

            x += vx * dt
            y += vy * dt
            theta = (theta + omega * dt) % (2 * np.pi)

            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            if dist >= R:
                nx = (x - cx) / dist
                ny = (y - cy) / dist
                tx = -ny
                ty = nx
                tangent_theta = np.arctan2(ty, tx)
                theta += 0.8 * np.sin(tangent_theta - theta)
                x = cx + nx * (R - 0.01*interaction_sigma)
                y = cy + ny * (R - 0.01*interaction_sigma)
                F1=0
                for k, (xk, yk, _) in enumerate(particles):
                    if j != k:
                        F1 += compute_pairwise_repulsion([x, y], [xk, yk])

                vx, vy = v_mobility * F1
                vx += np.random.normal(0, nt)
                vy += np.random.normal(0, nt)
                vx *= (1 - damping)
                vy *= (1 - damping)

                omega = rot_mobility * tau + compute_alignment(particles, i) + np.random.normal(0, nr)
                omega *= (1 - damping)
                
                x+=vx*dt
                y+=vy*dt

            
            x = np.clip(x, 1, L - 2)
            y = np.clip(y, 1, L - 2)

            new_particles.append([x, y, theta])

        particles = new_particles

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_xlim(0,L)
        ax.set_ylim(0,L)
        ax.imshow(c, cmap='viridis', origin='lower', vmin=0, vmax=5)
        circle = plt.Circle((cx, cy), R, color='red', fill=False)
        ax.add_patch(circle)
        for x, y, theta in particles:
            tip = (x + (2/3) * triangle_height * np.cos(theta),
                   y + (2/3) * triangle_height * np.sin(theta))
            base_center = (x - (1/3) * triangle_height * np.cos(theta),
                           y - (1/3) * triangle_height * np.sin(theta))
            wx = np.cos(theta + np.pi/2)
            wy = np.sin(theta + np.pi/2)
            left = (base_center[0] - triangle_base/2 * wx, base_center[1] - triangle_base/2 * wy)
            right = (base_center[0] + triangle_base/2 * wx, base_center[1] + triangle_base/2 * wy)
            ax.fill([left[0], tip[0], right[0]], [left[1], tip[1], right[1]], color='black', alpha=0.8)

        ax.axis('off')
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())
        frames.append(frame)
        plt.close(fig)

    path = f"outputs/PhysCamphor_CircularBC_Align_{int(source_height / scale)}mm.mp4"
    imageio.mimsave(path, frames, fps=20, format='FFMPEG')

