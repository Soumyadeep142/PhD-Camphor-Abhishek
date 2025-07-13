import multiprocessing
import subprocess
from itertools import product

# Define ranges for A and N
A_values = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]
N_values = [5,10,15,20,25,30,35,40,45,50]
R_values=[100,1000,10000,1000000]

def run_simulation(params):
    A, N, R = params
    print(f"Running simulation with A={A}, N_particles={N}")
    subprocess.run([
        "python3", "shape_camphor.py",  # Replace with your actual filename
        str(A), str(N), str(R)
    ])

if __name__ == "__main__":
    cpu_count = multiprocessing.cpu_count()
    print(f"Detected {cpu_count} cores. Running in parallel...")

    param_combinations = list(product(A_values, N_values, R_values))

    # Run in parallel using all available cores
    with multiprocessing.Pool(cpu_count) as pool:
        pool.map(run_simulation, param_combinations)

