import multiprocessing
import subprocess
from itertools import product

# Define ranges for A and N
A_values = [2.0, 6.0, 10.0,  16.0, 20.0]
N_values = [10, 20,30,40,50]
Align_values=[0.1, 0.01, 1, 5,10,15,20]
R_values=[10000,100000,1000000,10000000]

def run_simulation(params):
    A, N, R, Al = params
    print(f"Running simulation with A={A}, N_particles={N} Repel={R} Align={Al}")
    subprocess.run([
        "python3", "shape_camphor_align_repel.py",  # Replace with your actual filename
        str(A), str(N), str(R), str(Al)
    ])

if __name__ == "__main__":
    cpu_count = multiprocessing.cpu_count()
    print(f"Detected {cpu_count} cores. Running in parallel...")

    param_combinations = list(product(A_values, N_values, R_values, Align_values))

    # Run in parallel using all available cores
    with multiprocessing.Pool(cpu_count) as pool:
        pool.map(run_simulation, param_combinations)

