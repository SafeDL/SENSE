# -*- coding: utf-8 -*-
"""
Quick view of RLSAN search results with coverage metrics

Usage:
  python quick_view.py --results_path search_results.pkl
  python quick_view.py --results_path search_results.pkl --grid_x grid_x.pkl --grid_y grid_y.pkl
"""

import pickle
import os.path as osp
import argparse
import numpy as np


def compute_coverage_rate(hazardous_points, grid_x_path, grid_y_path,
                         collision_threshold=0.3, grid_resolution=30):
    """Compute failure domain coverage rate"""
    try:
        if not osp.exists(grid_x_path) or not osp.exists(grid_y_path):
            return None, None, None

        with open(grid_x_path, 'rb') as f:
            grid_x = pickle.load(f)
        with open(grid_y_path, 'rb') as f:
            grid_y = np.array(pickle.load(f))

        # Build ground truth
        search_space_min = -1.0
        search_space_max = 1.0
        grid_step = (search_space_max - search_space_min) / (grid_resolution - 1)

        collision_mask = grid_y > collision_threshold
        collision_points = grid_x[collision_mask]

        ground_truth_cells = set()
        for point in collision_points:
            indices = np.floor((point - search_space_min) / grid_step).astype(int)
            indices = np.clip(indices, 0, grid_resolution - 1)
            key = f"{indices[0]}_{indices[1]}_{indices[2]}"
            ground_truth_cells.add(key)

        # Compute captured cells
        captured = set()
        for point in hazardous_points:
            indices = np.floor((point - search_space_min) / grid_step).astype(int)
            indices = np.clip(indices, 0, grid_resolution - 1)
            key = f"{indices[0]}_{indices[1]}_{indices[2]}"
            if key in ground_truth_cells:
                captured.add(key)

        coverage_rate = (len(captured) / len(ground_truth_cells) * 100) if ground_truth_cells else 0.0
        return len(ground_truth_cells), len(captured), coverage_rate

    except Exception as e:
        print(f"[!] Error computing coverage: {e}")
        return None, None, None


def quick_view(results_path: str, grid_x_path=None, grid_y_path=None, collision_threshold=0.3):
    """Quick view of results"""
    if not osp.exists(results_path):
        print(f"[!] File not found: {results_path}")
        return False

    print(f"[*] Loading: {osp.abspath(results_path)}")

    with open(results_path, 'rb') as f:
        results = pickle.load(f)

    print("\n" + "="*70)
    print("RLSAN SEARCH RESULTS - QUICK VIEW".center(70))
    print("="*70 + "\n")

    # Basic info
    print("[1] SEARCH SUMMARY")
    total_evals = results.get('total_evaluations', 'N/A')
    real_sims = results.get('real_simulations', 'N/A')
    print(f"  Total Evaluations:        {total_evals}")
    print(f"  Real CARLA Simulations:   {real_sims}")
    print(f"  Search Time:              {results.get('search_time', 'N/A'):.2f} s")

    # Raw failures
    print("\n[2] RAW FAILURES")
    raw_count = results.get('raw_failures_count', 0)
    auc_raw = results.get('auc_fdc_raw', 0.0)
    print(f"  Total Found:              {raw_count}")
    print(f"  AUC-FDC:                  {auc_raw:.4e}")

    # Compute nAUC for raw
    if isinstance(total_evals, int) and total_evals > 0 and raw_count > 0:
        nauc_raw = auc_raw / (total_evals * raw_count)
        print(f"  nAUC-FDC:                 {nauc_raw:.4e}")

    n50_raw = results.get('n_50_raw', -1)
    if n50_raw == -1:
        print(f"  N_50:                     Not reached ({raw_count}/50)")
    else:
        print(f"  N_50:                     {n50_raw}")

    # Representative failures
    print("\n[3] REPRESENTATIVE FAILURES")
    rep_count = results.get('representative_failures_count', 0)
    auc_rep = results.get('auc_fdc_representative', 0.0)
    print(f"  Total Found:              {rep_count}")
    print(f"  AUC-FDC:                  {auc_rep:.4e}")

    # Compute nAUC for representative
    if isinstance(total_evals, int) and total_evals > 0 and rep_count > 0:
        nauc_rep = auc_rep / (total_evals * rep_count)
        print(f"  nAUC-FDC:                 {nauc_rep:.4e}")

    n50_rep = results.get('n_50_representative', -1)
    if n50_rep == -1:
        print(f"  N_50:                     Not reached ({rep_count}/50)")
    else:
        print(f"  N_50:                     {n50_rep}")

    if rep_count > 0:
        print(f"  Deduplication Ratio:      {raw_count/rep_count:.2f}x")

    # Failure Domain Coverage
    if grid_x_path and grid_y_path:
        print("\n[4] FAILURE DOMAIN COVERAGE (vs Grid Search)")
        hazardous_points = results.get('hazardous_points', np.array([]))
        gt_cells, captured_cells, coverage = compute_coverage_rate(
            hazardous_points, grid_x_path, grid_y_path, collision_threshold
        )
        if coverage is not None:
            print(f"  Ground Truth Cells:       {gt_cells}")
            print(f"  Captured Cells:           {captured_cells}")
            print(f"  Coverage Rate:            {coverage:.2f}%")
        else:
            print("  [!] Could not compute coverage rate")

    print("\n" + "="*70)

    # FDC curve sample
    fdc_raw = results.get('fdc_curve_raw', [])
    if fdc_raw:
        print("\n[5] RAW FAILURES FDC CURVE (sample)")
        print("  Budget    Failures")
        step = max(1, len(fdc_raw) // 10)
        for i in range(0, len(fdc_raw), step):
            budget, failures = fdc_raw[i]
            print(f"  {budget:>6d}    {failures:>6d}")
        if len(fdc_raw) % step != 0:
            budget, failures = fdc_raw[-1]
            print(f"  {budget:>6d}    {failures:>6d}")

    fdc_rep = results.get('fdc_curve_representative', [])
    if fdc_rep:
        print("\n[6] REPRESENTATIVE FAILURES FDC CURVE (sample)")
        print("  Budget    Failures")
        step = max(1, len(fdc_rep) // 10)
        for i in range(0, len(fdc_rep), step):
            budget, failures = fdc_rep[i]
            print(f"  {budget:>6d}    {failures:>6d}")
        if len(fdc_rep) % step != 0:
            budget, failures = fdc_rep[-1]
            print(f"  {budget:>6d}    {failures:>6d}")

    print("\n" + "="*70)
    print("[✓] Data loaded successfully!\n")
    return True


def main():
    parser = argparse.ArgumentParser(description='Quick view of RLSAN search results')
    parser.add_argument('--results_path', type=str, default='../../../../log/search_results.pkl',
                       help='Path to search_results.pkl file')
    parser.add_argument('--grid_x', type=str, default='../../surrogate/train_data/scenario01_grid_x.pkl',
                       help='Path to grid X coordinates (optional)')
    parser.add_argument('--grid_y', type=str, default='../../surrogate/train_data/scenario01_grid_y.pkl',
                       help='Path to grid Y values (optional)')
    parser.add_argument('--collision_threshold', type=float, default=0.3,
                       help='Collision threshold for grid search')
    args = parser.parse_args()

    quick_view(args.results_path, args.grid_x, args.grid_y, args.collision_threshold)


if __name__ == '__main__':
    main()
