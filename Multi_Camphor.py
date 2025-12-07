import multiprocessing
import subprocess
from itertools import product

# -----------------------------
# Parameter ranges
# -----------------------------
A_values = [5.0, 10.0, 15.0, 20.0, 25.0]
N_values = [10, 20, 30]
wall_align_values=[0.06]
alignment_strength_values=[13] #8,10
tau_values=[250]
align_dist_values=[12] #10
N_TRIALS = 20

# Optional: include more values if your script needs them
# Align_values = [...]
# R_values = [...]

# -----------------------------
# Worker function
# -----------------------------
def run_simulation(params):
    A, N, W, As,tau,align, run_id = params
    print(f"Running simulation with A={A}, N_particles={N}, wall_align={W}, alignment_strength={As} tau={tau} align_dist={align} run={run_id}")

    # Call your simulation script
    subprocess.run([
        "python3", "share.py",   # <-- replace with your actual filename
        str(A),
        str(N),
        str(W),
        str(As),
        str(tau),
        str(align),
        str(run_id)
    ])

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    cpu_count = multiprocessing.cpu_count()
    print(f"Detected {cpu_count} cores. Running in parallel...\n")

    # Create all parameter combinations INCLUDING run_id = 1..10
    param_combinations = []
    for combo in product(A_values, N_values, wall_align_values,
                         alignment_strength_values, tau_values, align_dist_values):
        A, N, W, As, tau, align = combo
        for run_id in range(1, N_TRIALS + 1):
            param_combinations.append((A, N, W, As, tau, align, run_id))

    # Run parallel
    with multiprocessing.Pool(cpu_count) as pool:
        pool.map(run_simulation, param_combinations)

    print("\nAll simulations completed.")

