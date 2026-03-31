import argparse
import pickle
import math
import numpy as np
from scipy.special import gamma
import scipy.io as sio
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def calculate_crate(ncs, dim, niche_radius, bounds=(-1.0, 1.0)):
    \"\"\"
    Calculate the Spatial Coverage Rate (CRate) based on the formulated formula:
    CRate(X) = Volume(Union(Chi(xi))) / V_space  ≈ (NCS * V_niche) / V_space
    
    Args:
        ncs (int): Number of Critical Scenarios (unique niches found)
        dim (int): Dimensionality of the search space.
        niche_radius (float): Radius defining the niche volume.
        bounds (tuple): The domain of each dimension. Assumes hypercube.
        
    Returns:
        float: the CRate in range [0, 1]
    \"\"\"
    domain_length = bounds[1] - bounds[0]
    V_space = domain_length ** dim
    V_niche = (math.pi ** (dim / 2.0) / gamma(dim / 2.0 + 1.0)) * (niche_radius ** dim)
    crate = min(1.0, ncs * V_niche / V_space)
    return crate

def calculate_ncs(hazardous_points, fitness_values, niche_radius, danger_threshold):
    \"\"\"
    Recalculate the Number of Critical Scenarios (NCS) using standard greedy niche clustering.
    Only points with fitness < danger_threshold are considered.
    \"\"\"
    valid_mask = fitness_values < danger_threshold
    valid_points = hazardous_points[valid_mask]
    valid_fitness = fitness_values[valid_mask]
    
    if len(valid_points) == 0:
        return 0, np.array([]), np.array([])
        
    # Sort by fitness (most dangerous first)
    sort_idx = np.argsort(valid_fitness)
    sorted_points = valid_points[sort_idx]
    sorted_fitness = valid_fitness[sort_idx]
    
    representative_seeds = []
    representative_fitness = []
    
    for pt, fit in zip(sorted_points, sorted_fitness):
        is_novel = True
        for seed in representative_seeds:
            if np.linalg.norm(pt - seed) < niche_radius:
                is_novel = False
                break
        if is_novel:
            representative_seeds.append(pt)
            representative_fitness.append(fit)
            
    return len(representative_seeds), np.array(representative_seeds), np.array(representative_fitness)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Coverage Rate (CRate) Calculator')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the search_results.pkl or .mat containing hazardous_points_history or representative seeds')
    parser.add_argument('--dim', type=int, default=3, help='Dimensionality of search space')
    parser.add_argument('--niche_radius', type=float, default=0.05, help='Niche radius used to compute hypercube volume')
    parser.add_argument('--danger_threshold', type=float, default=-0.3, help='Threshold to consider a scenario critical')
    parser.add_argument('--bounds', type=float, nargs=2, default=[-1.0, 1.0], help='Search space bounds per dimension')
    parser.add_argument('--output_dir', type=str, default='.', help='Directory to save the processed output')
    
    args = parser.parse_args()
    
    try:
        if args.input_file.endswith('.pkl'):
            with open(args.input_file, 'rb') as f:
                data = pickle.load(f)
        else:
            data = sio.loadmat(args.input_file)
            
        print(f"Successfully loaded data from {args.input_file}")
    except Exception as e:
        print(f"Error loading file {args.input_file}: {e}")
        exit(1)
        
    # Validate and extract necessary data
    # The user specifies that this script takes found dangerous test cases and true values.
    # We rely on 'representative_seeds' and 'representative_fitness' directly from the deployment script.
    
    if 'representative_seeds' not in data or 'representative_fitness' not in data:
        print("Error: The input data must contain 'representative_seeds' and 'representative_fitness' keys.")
        exit(1)
        
    rep_seeds = np.array(data['representative_seeds'])
    rep_fitness = np.array(data['representative_fitness'])
    
    # Optional Re-verification step: count them manually with the precise radius
    ncs, filtered_seeds, filtered_fitness = calculate_ncs(rep_seeds, rep_fitness, args.niche_radius, args.danger_threshold)
    
    print(f"\\n--- CRate Calculation ---")
    print(f"Dimension (d): {args.dim}")
    print(f"Niche Radius (r): {args.niche_radius}")
    print(f"Danger Threshold: {args.danger_threshold}")
    print(f"Number of Critical Scenarios (NCS): {ncs}")
    
    crate = calculate_crate(ncs, args.dim, args.niche_radius, args.bounds)
    print(f"\\nFinal CRate (Spatial Coverage): {crate * 100:.4f}%")
    
    # Save output
    os.makedirs(args.output_dir, exist_ok=True)
    out_dict = {
        'ncs': ncs,
        'crate': crate,
        'dim': args.dim,
        'niche_radius': args.niche_radius,
        'filtered_seeds': filtered_seeds,
        'filtered_fitness': filtered_fitness
    }
    
    output_path = os.path.join(args.output_dir, 'coverage_metrics.mat')
    sio.savemat(output_path, out_dict)
    print(f"Coverage metrics saved to: {output_path}")
