#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基线方法对比分析脚本

功能：
- 加载所有基线搜索结果
- 计算缺失的指标（覆盖率、AUC等）
- 生成对比表格和图表
- 输出详细报告
"""

import pickle
import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import argparse
from datetime import datetime

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class BaselineAnalyzer:
    """Analyze and compare baseline methods"""

    def __init__(self, results_dir='log/baselines',
                 grid_x_path=None, grid_y_path=None, collision_threshold=0.3):
        """
        Initialize analyzer

        Args:
            results_dir: Directory containing baseline results
            grid_x_path: Path to grid X coordinates (scenario01_grid_x.pkl)
            grid_y_path: Path to grid Y values (scenario01_grid_y.pkl)
            collision_threshold: Threshold for collision detection
        """
        self.results_dir = results_dir
        self.grid_x_path = grid_x_path
        self.grid_y_path = grid_y_path
        self.collision_threshold = collision_threshold

        # Grid search parameters
        self.grid_x_data = None
        self.grid_y_data = None
        self.ground_truth_cells = set()
        self.grid_resolution = 30
        self.search_space_min = -1.0
        self.search_space_max = 1.0
        self.grid_step = (self.search_space_max - self.search_space_min) / (self.grid_resolution - 1)

        self.methods = {}
        self.method_names = []

        # Load grid data if available
        if grid_x_path and grid_y_path:
            self._load_grid_data(grid_x_path, grid_y_path)

    def _load_grid_data(self, grid_x_path: str, grid_y_path: str):
        """Load grid search data and compute ground truth"""
        try:
            if not osp.exists(grid_x_path):
                print(f"[!] Warning: Grid X file not found: {grid_x_path}")
                return
            if not osp.exists(grid_y_path):
                print(f"[!] Warning: Grid Y file not found: {grid_y_path}")
                return

            print(f"[*] Loading grid X from: {grid_x_path}")
            with open(grid_x_path, 'rb') as f:
                self.grid_x_data = pickle.load(f)

            print(f"[*] Loading grid Y from: {grid_y_path}")
            with open(grid_y_path, 'rb') as f:
                grid_y_raw = pickle.load(f)
            self.grid_y_data = np.array(grid_y_raw)

            # Build ground truth: find all grid cells with collisions
            self._build_ground_truth()
            print(f"[✓] Grid data loaded: {self.grid_x_data.shape[0]} points, {len(self.ground_truth_cells)} collision cells")

        except Exception as e:
            print(f"[!] Error loading grid data: {e}")

    def _build_ground_truth(self):
        """Build ground truth set of collision grid cells"""
        if self.grid_x_data is None or self.grid_y_data is None:
            return

        # Find collision points
        collision_mask = self.grid_y_data > self.collision_threshold
        collision_points = self.grid_x_data[collision_mask]

        # Convert to grid indices and store as keys
        for point in collision_points:
            grid_idx = self._point_to_grid_index(point)
            key = f"{grid_idx[0]}_{grid_idx[1]}_{grid_idx[2]}"
            self.ground_truth_cells.add(key)

    def _point_to_grid_index(self, point: np.ndarray) -> np.ndarray:
        """Convert continuous point to grid index"""
        indices = np.floor((point - self.search_space_min) / self.grid_step).astype(int)
        # Clamp to valid range
        indices = np.clip(indices, 0, self.grid_resolution - 1)
        return indices

    def _compute_coverage_rate(self, hazardous_points):
        """Compute failure domain coverage rate"""
        # 如果没有网格数据或没有危险点，返回 0
        if not self.ground_truth_cells:
            print(f"[!] Warning: No ground truth cells available (grid data not loaded)")
            return 0.0, 0, 0

        if len(hazardous_points) == 0:
            return 0.0, 0, len(self.ground_truth_cells)

        hazardous_points = np.atleast_2d(hazardous_points)

        # Coverage with raw failures (all discovered points)
        captured_raw = set()
        for point in hazardous_points:
            grid_idx = self._point_to_grid_index(point)
            key = f"{grid_idx[0]}_{grid_idx[1]}_{grid_idx[2]}"
            if key in self.ground_truth_cells:
                captured_raw.add(key)

        captured_cells_raw = len(captured_raw)
        coverage_rate_raw = (captured_cells_raw / len(self.ground_truth_cells) * 100) if self.ground_truth_cells else 0.0

        return coverage_rate_raw, captured_cells_raw, len(self.ground_truth_cells)

    def load_baseline_results(self):
        """Load all baseline results from directory"""
        if not osp.exists(self.results_dir):
            print(f"[!] Baselines directory not found: {self.results_dir}")
            return

        print(f"[*] Loading baseline results from: {self.results_dir}")

        baseline_files = {
            'rs': 'random_search_results.pkl',
            'ga': 'genetic_algorithm_results.pkl',
            'bo': 'bayesian_optimization_results.pkl',
            'rnns': 'random_neighbourhood_search_results.pkl',
            'ras': 'repulsive_adaptive_sampling_results.pkl',
            'ltc': 'learning_to_collide_results.pkl',
        }

        baseline_names = {
            'rs': 'Random Search',
            'ga': 'Genetic Algorithm',
            'bo': 'Bayesian Optimization',
            'rnns': 'Random Neighbourhood Search',
            'ras': 'Repulsive Adaptive Sampling',
            'ltc': 'Learning to Collide',
        }

        for key, filename in baseline_files.items():
            path = osp.join(self.results_dir, filename)
            if osp.exists(path):
                try:
                    with open(path, 'rb') as f:
                        data = pickle.load(f)

                    # Extract basic metrics
                    hazardous_points = data.get('hazardous_points', np.array([]))
                    total_evals = data.get('total_evaluations', 1)
                    raw_failures = data.get('raw_failures_count', len(hazardous_points))
                    rep_failures = data.get('representative_failures_count', raw_failures)
                    search_time = data.get('search_time', 0.0)
                    real_sims = data.get('real_simulations', 0)
                    surrogate_calls = data.get('surrogate_calls', 0)

                    # 总是尝试从网格数据计算覆盖率（如果可用）
                    # 即使基线结果中已有覆盖率数据，也要重新计算以确保一致性
                    if self.ground_truth_cells:
                        coverage_rate_raw, captured_cells_raw, ground_truth_cells = \
                            self._compute_coverage_rate(hazardous_points)
                        print(f"  [*] {baseline_names[key]}: Computed coverage from grid data")
                    else:
                        # 如果没有网格数据，使用基线结果中的数据（可能为 0）
                        coverage_rate_raw = data.get('coverage_rate_raw', 0.0)
                        captured_cells_raw = data.get('captured_cells_raw', 0)
                        ground_truth_cells = data.get('ground_truth_cells', 0)
                        if coverage_rate_raw == 0.0 and ground_truth_cells == 0:
                            print(f"  [!] {baseline_names[key]}: No coverage data (grid data not loaded)")

                    # Compute nAUC if AUC data is available
                    auc_fdc_raw = data.get('auc_fdc_raw', 0.0)
                    nauc_fdc_raw = 0.0
                    if auc_fdc_raw > 0 and total_evals > 0 and raw_failures > 0:
                        nauc_fdc_raw = auc_fdc_raw / (total_evals * raw_failures)

                    auc_fdc_rep = data.get('auc_fdc_representative', 0.0)
                    nauc_fdc_rep = 0.0
                    if auc_fdc_rep > 0 and total_evals > 0 and rep_failures > 0:
                        nauc_fdc_rep = auc_fdc_rep / (total_evals * rep_failures)

                    # Prepare unified data
                    unified = {
                        'method': baseline_names[key],
                        'algorithm': key,
                        'hazardous_points': hazardous_points,
                        'raw_failures_count': raw_failures,
                        'representative_failures_count': rep_failures,
                        'total_evaluations': total_evals,
                        'real_simulations': real_sims,
                        'surrogate_calls': surrogate_calls,
                        'search_time': search_time,
                        'coverage_rate_raw': coverage_rate_raw,
                        'captured_cells_raw': captured_cells_raw,
                        'ground_truth_cells': ground_truth_cells,
                        # FDC metrics (may not be available for baselines)
                        'auc_fdc_raw': auc_fdc_raw,
                        'auc_fdc_representative': auc_fdc_rep,
                        'nauc_fdc_raw': nauc_fdc_raw,
                        'nauc_fdc_representative': nauc_fdc_rep,
                        'n_50_raw': data.get('n_50_raw', -1),
                        'n_50_representative': data.get('n_50_representative', -1),
                        'fdc_curve_raw': data.get('fdc_curve_raw', []),
                        'fdc_curve_representative': data.get('fdc_curve_representative', []),
                    }

                    self.methods[key] = unified
                    self.method_names.append(baseline_names[key])
                    print(f"[✓] {baseline_names[key]} results loaded")
                except Exception as e:
                    print(f"[!] Error loading {baseline_names[key]}: {e}")
            else:
                print(f"[!] {baseline_names[key]} results not found at {path}")

    def print_comparison(self):
        """Print comparison table"""
        if not self.methods:
            print("[!] No results loaded")
            return

        print("\n" + "="*200)
        print("BASELINE METHODS COMPARISON".center(200))
        print("="*200)

        # Header with all metrics
        print(f"{'Method':<25} {'Raw':<10} {'Rep':<10} {'AUC-FDC':<14} {'nAUC-FDC':<14} {'Coverage%':<12} {'Time(s)':<10}")
        print("-"*200)

        for method_key in self.methods.keys():
            results = self.methods[method_key]
            method_name = results.get('method', 'Unknown')
            raw_failures = results.get('raw_failures_count', 0)
            rep_failures = results.get('representative_failures_count', 0)
            auc_fdc = results.get('auc_fdc_raw', 0.0)
            nauc_fdc = results.get('nauc_fdc_raw', 0.0)
            coverage = results.get('coverage_rate_raw', 0.0)
            time_val = results.get('search_time', 0.0)

            auc_str = f"{auc_fdc:.2e}" if auc_fdc > 0 else "N/A"
            nauc_str = f"{nauc_fdc:.4e}" if nauc_fdc > 0 else "N/A"
            coverage_str = f"{coverage:.1f}" if coverage > 0 else "N/A"

            print(f"{method_name:<25} {raw_failures:<10} {rep_failures:<10} {auc_str:<14} {nauc_str:<14} {coverage_str:<12} {time_val:<10.1f}")

        print("="*200 + "\n")

    def plot_comparison(self, output_dir='results'):
        """Create comparison plots"""
        if not self.methods:
            print("[!] No results to plot")
            return

        os.makedirs(output_dir, exist_ok=True)

        # Extract data
        method_names = list(self.methods.keys())
        labels = [self.methods[m].get('method', m) for m in method_names]
        raw_failures = [self.methods[m].get('raw_failures_count', 0) for m in method_names]
        rep_failures = [self.methods[m].get('representative_failures_count', 0) for m in method_names]
        coverage_rates = [self.methods[m].get('coverage_rate_raw', 0.0) for m in method_names]
        search_times = [self.methods[m].get('search_time', 0.0) for m in method_names]
        total_evals = [self.methods[m].get('total_evaluations', 1) for m in method_names]
        real_sims = [self.methods[m].get('real_simulations', 0) for m in method_names]
        auc_fdc_values = [self.methods[m].get('auc_fdc_raw', 0.0) for m in method_names]
        nauc_fdc_values = [self.methods[m].get('nauc_fdc_raw', 0.0) for m in method_names]

        # Color scheme
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        method_colors = colors[:len(method_names)]

        fig, axes = plt.subplots(3, 3, figsize=(20, 14))

        # Plot 1: Raw Failures
        ax = axes[0, 0]
        bars = ax.bar(labels, raw_failures, color=method_colors, alpha=0.7, edgecolor='black', linewidth=2)
        for bar, count in zip(bars, raw_failures):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(count)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax.set_ylabel('Number of Raw Failures', fontsize=12, fontweight='bold')
        ax.set_title('Raw Failures Found', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Plot 2: Representative Failures
        ax = axes[0, 1]
        bars = ax.bar(labels, rep_failures, color=method_colors, alpha=0.7, edgecolor='black', linewidth=2)
        for bar, count in zip(bars, rep_failures):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(count)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax.set_ylabel('Number of Representative Failures', fontsize=12, fontweight='bold')
        ax.set_title('Representative Failures (Deduplicated)', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Plot 3: Coverage Rate
        ax = axes[0, 2]
        bars = ax.bar(labels, coverage_rates, color=method_colors, alpha=0.7, edgecolor='black', linewidth=2)
        for bar, rate in zip(bars, coverage_rates):
            height = bar.get_height()
            if rate > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{rate:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax.axhline(y=100, color='green', linestyle='--', alpha=0.5, linewidth=2, label='Perfect Coverage')
        ax.set_ylabel('Coverage Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title('Failure Domain Coverage Rate', fontsize=13, fontweight='bold')
        ax.set_ylim(0, 110)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Plot 4: Search Time
        ax = axes[1, 0]
        bars = ax.bar(labels, search_times, color=method_colors, alpha=0.7, edgecolor='black', linewidth=2)
        for bar, t in zip(bars, search_times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{t:.0f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax.set_ylabel('Search Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_title('Computational Time', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Plot 5: AUC-FDC
        ax = axes[1, 1]
        bars = ax.bar(labels, auc_fdc_values, color=method_colors, alpha=0.7, edgecolor='black', linewidth=2)
        for bar, auc in zip(bars, auc_fdc_values):
            height = bar.get_height()
            if auc > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{auc:.2e}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_ylabel('AUC-FDC Value', fontsize=12, fontweight='bold')
        ax.set_title('Area Under Failure Discovery Curve', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Plot 6: nAUC-FDC (Normalized AUC)
        ax = axes[1, 2]
        bars = ax.bar(labels, nauc_fdc_values, color=method_colors, alpha=0.7, edgecolor='black', linewidth=2)
        for bar, nauc in zip(bars, nauc_fdc_values):
            height = bar.get_height()
            if nauc > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{nauc:.4e}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_ylabel('nAUC-FDC Value', fontsize=12, fontweight='bold')
        ax.set_title('Normalized AUC (nAUC = AUC / (T × N))', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Plot 7: Evaluation Budget Breakdown
        ax = axes[2, 0]
        x_pos = np.arange(len(labels))
        width = 0.35
        ax.bar(x_pos - width/2, real_sims, width, label='Real Simulations',
               color='#f39c12', alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.bar(x_pos + width/2, [s - r for s, r in zip(total_evals, real_sims)], width,
               label='Surrogate Calls', color='#2ecc71', alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Number of Evaluations', fontsize=12, fontweight='bold')
        ax.set_title('Evaluation Budget Breakdown', fontsize=13, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Plot 8: Efficiency (Failures per Second)
        ax = axes[2, 1]
        efficiency = [r/max(t, 0.01) if t > 0 else 0 for r, t in zip(raw_failures, search_times)]
        bars = ax.bar(labels, efficiency, color=method_colors, alpha=0.7, edgecolor='black', linewidth=2)
        for bar, eff in zip(bars, efficiency):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{eff:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax.set_ylabel('Failures Found per Second', fontsize=12, fontweight='bold')
        ax.set_title('Search Efficiency', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Plot 9: Summary Statistics Text
        ax = axes[2, 2]
        ax.axis('off')
        summary_text = "SUMMARY STATISTICS\n" + "="*35 + "\n\n"
        summary_text += f"Total Methods: {len(method_names)}\n"
        best_raw = max(zip(labels, raw_failures), key=lambda x: x[1]) if raw_failures else (None, 0)
        best_coverage = max(zip(labels, coverage_rates), key=lambda x: x[1]) if coverage_rates else (None, 0)
        best_nauc = max(zip(labels, nauc_fdc_values), key=lambda x: x[1]) if nauc_fdc_values else (None, 0)

        if best_raw[0]:
            summary_text += f"\nBest (Raw): {best_raw[0]}\n  ({int(best_raw[1])} failures)\n"
        if best_coverage[0] and best_coverage[1] > 0:
            summary_text += f"\nBest Coverage: {best_coverage[0]}\n  ({best_coverage[1]:.1f}%)\n"
        if best_nauc[0] and best_nauc[1] > 0:
            summary_text += f"\nBest nAUC: {best_nauc[0]}\n  ({best_nauc[1]:.2e})\n"

        ax.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
               family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        output_path = osp.join(output_dir, 'baseline_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[+] Saved: {output_path}")
        plt.close()

    def generate_report(self, output_dir='results'):
        """Generate comprehensive comparison report"""
        if not self.methods:
            print("[!] No results to report")
            return

        os.makedirs(output_dir, exist_ok=True)

        report_path = osp.join(output_dir, 'baseline_comparison_report.txt')

        with open(report_path, 'w') as f:
            f.write("="*160 + "\n")
            f.write("BASELINE METHODS COMPARISON REPORT\n")
            f.write("="*160 + "\n\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Summary table
            f.write("="*160 + "\n")
            f.write("SUMMARY METRICS\n")
            f.write("="*160 + "\n\n")

            f.write(f"{'Method':<25} {'Raw':<10} {'Rep':<10} {'AUC-FDC':<14} {'nAUC-FDC':<14} {'Coverage%':<12} {'Time(s)':<10}\n")
            f.write("-"*200 + "\n")

            for method_key in self.methods.keys():
                results = self.methods[method_key]
                method_name = results.get('method', 'Unknown')
                raw_failures = results.get('raw_failures_count', 0)
                rep_failures = results.get('representative_failures_count', 0)
                auc_fdc = results.get('auc_fdc_raw', 0.0)
                nauc_fdc = results.get('nauc_fdc_raw', 0.0)
                coverage = results.get('coverage_rate_raw', 0.0)
                time_val = results.get('search_time', 0.0)

                auc_str = f"{auc_fdc:.2e}" if auc_fdc > 0 else "N/A"
                nauc_str = f"{nauc_fdc:.4e}" if nauc_fdc > 0 else "N/A"
                coverage_str = f"{coverage:.1f}" if coverage > 0 else "N/A"

                f.write(f"{method_name:<25} {raw_failures:<10} {rep_failures:<10} {auc_str:<14} {nauc_str:<14} {coverage_str:<12} {time_val:<10.1f}\n")

            f.write("\n" + "="*160 + "\n")
            f.write("DETAILED ANALYSIS\n")
            f.write("="*160 + "\n\n")

            # Detailed results for each method
            for method_key in self.methods.keys():
                results = self.methods[method_key]
                method_name = results.get('method', 'Unknown')

                f.write(f"\n{method_name}\n")
                f.write("-" * 200 + "\n")

                # Basic metrics
                f.write(f"[BASIC STATISTICS]\n")
                f.write(f"  Total Evaluations:                {results.get('total_evaluations', 'N/A')}\n")
                f.write(f"  Real Simulations:                 {results.get('real_simulations', 'N/A')}\n")
                f.write(f"  Surrogate Calls:                  {results.get('surrogate_calls', 'N/A')}\n")
                f.write(f"  Search Time:                      {results.get('search_time', 0.0):.2f}s\n\n")

                # Failure metrics
                f.write(f"[FAILURE DISCOVERY]\n")
                f.write(f"  Raw Failures Found:               {results.get('raw_failures_count', 0)}\n")
                f.write(f"  Representative Failures:          {results.get('representative_failures_count', 0)}\n")
                if results.get('raw_failures_count', 0) > 0 and results.get('representative_failures_count', 0) > 0:
                    dedupe_ratio = results.get('raw_failures_count', 1) / max(results.get('representative_failures_count', 1), 1)
                    f.write(f"  Deduplication Ratio:              {dedupe_ratio:.2f}x\n\n")
                else:
                    f.write(f"\n")

                # FDC metrics
                auc_fdc = results.get('auc_fdc_raw', 0.0)
                nauc_fdc = results.get('nauc_fdc_raw', 0.0)
                n_50 = results.get('n_50_raw', -1)
                if auc_fdc > 0 or nauc_fdc > 0:
                    f.write(f"[FAILURE DISCOVERY CURVE (FDC)]\n")
                    f.write(f"  AUC-FDC (Raw):                    {auc_fdc:.4e}\n")
                    f.write(f"  nAUC-FDC (Normalized):            {nauc_fdc:.4e}\n")
                    if n_50 == -1:
                        f.write(f"  N_50 (Budget to find 50):         Not reached ({results.get('raw_failures_count', 0)}/50)\n\n")
                    else:
                        f.write(f"  N_50 (Budget to find 50):         {n_50}\n\n")

                # Coverage metrics
                coverage_rate = results.get('coverage_rate_raw', 0.0)
                if coverage_rate > 0 or results.get('ground_truth_cells', 0) > 0:
                    f.write(f"[FAILURE DOMAIN COVERAGE]\n")
                    f.write(f"  Coverage Rate:                    {coverage_rate:.2f}%\n")
                    f.write(f"  Captured Grid Cells:              {results.get('captured_cells_raw', 0)}\n")
                    f.write(f"  Ground Truth Cells:               {results.get('ground_truth_cells', 0)}\n\n")

                # Efficiency metrics
                total_evals = results.get('total_evaluations', 1)
                search_time = results.get('search_time', 0.0)
                raw_failures = results.get('raw_failures_count', 0)

                f.write(f"[EFFICIENCY METRICS]\n")
                if total_evals > 0:
                    f.write(f"  Failures per Evaluation:          {(raw_failures/total_evals)*100:.2f}%\n")
                if search_time > 0:
                    f.write(f"  Failures per Second:              {raw_failures/search_time:.2f}\n")

            f.write("\n" + "="*160 + "\n")
            f.write("KEY FINDINGS\n")
            f.write("="*160 + "\n\n")

            # Find the best method for each metric
            hazardous_dict = {self.methods[m].get('method', m): self.methods[m].get('raw_failures_count', 0)
                            for m in self.methods.keys()}
            rep_hazardous_dict = {self.methods[m].get('method', m): self.methods[m].get('representative_failures_count', 0)
                                for m in self.methods.keys()}
            coverage_dict = {self.methods[m].get('method', m): self.methods[m].get('coverage_rate_raw', 0.0)
                           for m in self.methods.keys()}
            time_dict = {self.methods[m].get('method', m): self.methods[m].get('search_time', 0.0)
                        for m in self.methods.keys()}
            efficiency_dict = {self.methods[m].get('method', m):
                             self.methods[m].get('raw_failures_count', 0) / max(self.methods[m].get('search_time', 0.01), 0.01)
                             for m in self.methods.keys()}

            best_hazardous = max(hazardous_dict, key=hazardous_dict.get)
            best_rep = max(rep_hazardous_dict, key=rep_hazardous_dict.get)
            best_coverage = max(coverage_dict, key=coverage_dict.get)
            best_time = min(time_dict, key=time_dict.get)
            best_efficiency = max(efficiency_dict, key=efficiency_dict.get)

            f.write(f"Best in Raw Failures:               {best_hazardous} ({hazardous_dict[best_hazardous]})\n")
            f.write(f"Best in Representative Failures:    {best_rep} ({rep_hazardous_dict[best_rep]})\n")
            if coverage_dict[best_coverage] > 0:
                f.write(f"Best in Coverage Rate:              {best_coverage} ({coverage_dict[best_coverage]:.2f}%)\n")
            f.write(f"Fastest Search:                     {best_time} ({time_dict[best_time]:.2f}s)\n")
            f.write(f"Best Efficiency:                    {best_efficiency} ({efficiency_dict[best_efficiency]:.2f} failures/s)\n")

            f.write("\n" + "="*200 + "\n")
            f.write("METRIC INTERPRETATION\n")
            f.write("="*200 + "\n\n")
            f.write("1. Raw Failures: Number of test cases triggering collisions (unfiltered)\n")
            f.write("2. Representative Failures: Unique/deduplicated failure cases\n")
            f.write("3. AUC-FDC: Area Under Failure Discovery Curve\n")
            f.write("   - Larger values indicate better early-stage search efficiency\n")
            f.write("   - Reflects how quickly the algorithm discovers failures\n\n")
            f.write("4. nAUC-FDC: Normalized AUC = AUC / (T_total × N_max)\n")
            f.write("   - Normalizes AUC by total budget and discovered failures\n")
            f.write("   - Enables fair comparison across different experiments\n")
            f.write("   - Larger values indicate better efficiency per unit budget\n\n")
            f.write("5. Coverage Rate: Percentage of grid cells (from grid search) found by algorithm\n")
            f.write("6. Search Time: Total wall-clock time for the search process\n")
            f.write("7. Total Evals: Sum of surrogate calls and real simulations\n")
            f.write("8. Real Sims: Actual CARLA simulations executed\n")
            f.write("9. Surrogate: Fast predictions from surrogate model\n")
            f.write("10. Captured Grid Cells: Number of unique failure domain cells discovered\n")
            f.write("11. Ground Truth Cells: Total number of collision cells in the test space\n")
            f.write("12. Efficiency: Failures found per second of search time\n\n")

        print(f"[+] Report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Baseline Methods Comparison Analysis')
    parser.add_argument('--baselines_dir', type=str, default='../log/baselines',
                       help='Directory containing baseline results')
    parser.add_argument('--grid_x', type=str, default='../rlsan/src/surrogate/train_data/scenario01_grid_x.pkl',
                       help='Path to grid X coordinates (scenario01_grid_x.pkl)')
    parser.add_argument('--grid_y', type=str, default='../rlsan/src/surrogate/train_data/scenario01_grid_y.pkl',
                       help='Path to grid Y values (scenario01_grid_y.pkl)')
    parser.add_argument('--output_dir', type=str, default='results/baseline_comparison',
                       help='Directory to save comparison analysis')
    parser.add_argument('--collision_threshold', type=float, default=0.3,
                       help='Threshold for collision detection in grid search')

    args = parser.parse_args()


    # Initialize analyzer
    analyzer = BaselineAnalyzer(
        results_dir=args.baselines_dir,
        grid_x_path=args.grid_x,
        grid_y_path=args.grid_y,
        collision_threshold=args.collision_threshold
    )

    # Load results
    analyzer.load_baseline_results()

    if not analyzer.methods:
        print("[!] No results loaded")
        return

    # Print comparison
    analyzer.print_comparison()

    # Generate visualizations
    analyzer.plot_comparison(args.output_dir)

    # Generate report
    analyzer.generate_report(args.output_dir)

    print(f"\n[✓] Analysis complete! Results saved to: {osp.abspath(args.output_dir)}/")


if __name__ == '__main__':
    main()
