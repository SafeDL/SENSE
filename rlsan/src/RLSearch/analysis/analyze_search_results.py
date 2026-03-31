# -*- coding: utf-8 -*-
"""
Analyze and visualize RLSAN search results with Failure Domain Coverage

Usage:
  python analysis/analyze_search_results.py --results_path log/search_results.pkl
  python analysis/analyze_search_results.py --results_path log/search_results.pkl --grid_x surrogate/train_data/scenario01_grid_x.pkl --grid_y surrogate/train_data/scenario01_grid_y.pkl
"""

import pickle
import os
import os.path as osp
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime

matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


class SearchResultsAnalyzer:
    """Analyze search results with coverage metrics"""

    def __init__(self, results_path: str, grid_x_path: str = None, grid_y_path: str = None, collision_threshold: float = 0.3):
        """
        Initialize analyzer

        Args:
            results_path: Path to search_results.pkl file
            grid_x_path: Path to grid X coordinates (scenario01_grid_x.pkl)
            grid_y_path: Path to grid Y values (scenario01_grid_y.pkl)
            collision_threshold: Threshold for collision detection (default: 0.3)
        """
        if not osp.exists(results_path):
            raise FileNotFoundError(f"Results file not found: {results_path}")

        print(f"[*] Loading results from: {results_path}")
        with open(results_path, 'rb') as f:
            self.results = pickle.load(f)

        # Grid search parameters
        self.grid_x_data = None
        self.grid_y_data = None
        self.ground_truth_cells = set()
        self.coverage_rate = 0.0
        self.captured_cells = 0
        self.collision_threshold = collision_threshold
        self.grid_resolution = 30  # 每维30个点
        self.search_space_min = -1.0
        self.search_space_max = 1.0
        self.grid_step = (self.search_space_max - self.search_space_min) / (self.grid_resolution - 1)

        # Try to load grid data
        if grid_x_path and grid_y_path:
            self._load_grid_data(grid_x_path, grid_y_path)

        self._validate_results()
        self._extract_data()

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
        # Clamp to valid range [0, grid_resolution-1]
        indices = np.clip(indices, 0, self.grid_resolution - 1)
        return indices

    def _compute_coverage_rate(self):
        """Compute failure domain coverage rate (both raw and representative)"""
        if not self.ground_truth_cells or len(self.hazardous_points) == 0:
            self.coverage_rate_raw = 0.0
            self.coverage_rate_rep = 0.0
            self.captured_cells_raw = 0
            self.captured_cells_rep = 0
            return

        # Coverage with raw failures (all discovered points)
        captured_raw = set()
        for point in self.hazardous_points:
            grid_idx = self._point_to_grid_index(point)
            key = f"{grid_idx[0]}_{grid_idx[1]}_{grid_idx[2]}"
            if key in self.ground_truth_cells:
                captured_raw.add(key)

        self.captured_cells_raw = len(captured_raw)
        self.coverage_rate_raw = (self.captured_cells_raw / len(self.ground_truth_cells) * 100) if self.ground_truth_cells else 0.0

        # Coverage with representative failures (use representative_points directly if available)
        representative_points = self.results.get('representative_points', np.array([]))

        if len(representative_points) > 0:
            # Use representative_points directly from results
            captured_rep = set()
            for point in representative_points:
                grid_idx = self._point_to_grid_index(point)
                key = f"{grid_idx[0]}_{grid_idx[1]}_{grid_idx[2]}"
                if key in self.ground_truth_cells:
                    captured_rep.add(key)

            self.captured_cells_rep = len(captured_rep)
            self.coverage_rate_rep = (self.captured_cells_rep / len(self.ground_truth_cells) * 100) if self.ground_truth_cells else 0.0
        else:
            # Fallback: use raw coverage if representative_points not available
            self.captured_cells_rep = self.captured_cells_raw
            self.coverage_rate_rep = self.coverage_rate_raw

    def _validate_results(self):
        """验证结果数据的完整性"""
        required_keys = [
            'hazardous_points', 'raw_failures_count', 'representative_failures_count',
            'total_evaluations', 'real_simulations', 'search_time',
            'auc_fdc_raw', 'n_50_raw', 'fdc_curve_raw',
            'auc_fdc_representative', 'n_50_representative', 'fdc_curve_representative'
        ]

        missing = [k for k in required_keys if k not in self.results]
        if missing:
            print(f"[!] Warning: Missing keys: {missing}")
        else:
            print(f"[✓] All required keys found")

    def _extract_data(self):
        """Extract key data and compute normalized metrics"""
        self.hazardous_points = self.results.get('hazardous_points', np.array([]))
        self.raw_failures = self.results.get('raw_failures_count', 0)
        self.representative_failures = self.results.get('representative_failures_count', 0)
        self.total_evals = self.results.get('total_evaluations', 0)
        self.real_sims = self.results.get('real_simulations', 0)
        self.search_time = self.results.get('search_time', 0)

        # Raw metrics
        self.auc_fdc_raw = self.results.get('auc_fdc_raw', 0.0)
        self.n_50_raw = self.results.get('n_50_raw', -1)
        self.fdc_curve_raw = self.results.get('fdc_curve_raw', [])

        # Representative metrics
        self.auc_fdc_rep = self.results.get('auc_fdc_representative', 0.0)
        self.n_50_rep = self.results.get('n_50_representative', -1)
        self.fdc_curve_rep = self.results.get('fdc_curve_representative', [])

        # Compute normalized AUC metrics: nAUC = AUC / (T_total * N_max)
        # nAUC_raw = auc_fdc_raw / (total_evals * raw_failures)
        # nAUC_rep = auc_fdc_rep / (total_evals * representative_failures)
        if self.total_evals > 0 and self.raw_failures > 0:
            self.nauc_fdc_raw = self.auc_fdc_raw / (self.total_evals * self.raw_failures)
        else:
            self.nauc_fdc_raw = 0.0

        if self.total_evals > 0 and self.representative_failures > 0:
            self.nauc_fdc_rep = self.auc_fdc_rep / (self.total_evals * self.representative_failures)
        else:
            self.nauc_fdc_rep = 0.0

        # Compute failure domain coverage rate
        self._compute_coverage_rate()

    def print_summary(self):
        """Print summary statistics"""
        print("\n" + "="*70)
        print("SEARCH RESULTS SUMMARY".center(70))
        print("="*70)

        print("\n[1] BASIC STATISTICS")
        print(f"  Total Evaluations (Surrogate + Real):  {self.total_evals:>10}")
        print(f"  Real CARLA Simulations:                {self.real_sims:>10}")
        print(f"  Surrogate Model Calls:                 {self.total_evals - self.real_sims:>10}")
        print(f"  Search Time:                           {self.search_time:>10.2f} s")

        print("\n[2] RAW FAILURES (Unfiltered)")
        print(f"  Total Found:                           {self.raw_failures:>10}")
        print(f"  AUC-FDC (Raw):                         {self.auc_fdc_raw:>10.4e}")
        print(f"  nAUC-FDC (Raw):                        {self.nauc_fdc_raw:>10.4e}")
        if self.n_50_raw == -1:
            print(f"  N_50 (Raw):                            {'Not reached':>10} ({self.raw_failures}/50)")
        else:
            print(f"  N_50 (Raw):                            {self.n_50_raw:>10}")

        print("\n[3] REPRESENTATIVE FAILURES (Deduplicated)")
        print(f"  Total Found:                           {self.representative_failures:>10}")
        print(f"  AUC-FDC (Rep):                         {self.auc_fdc_rep:>10.4e}")
        print(f"  nAUC-FDC (Rep):                        {self.nauc_fdc_rep:>10.4e}")
        if self.n_50_rep == -1:
            print(f"  N_50 (Rep):                            {'Not reached':>10} ({self.representative_failures}/50)")
        else:
            print(f"  N_50 (Rep):                            {self.n_50_rep:>10}")

        print("\n[4] EFFICIENCY METRICS")
        if self.total_evals > 0:
            raw_efficiency = (self.raw_failures / self.total_evals) * 100
            rep_efficiency = (self.representative_failures / self.total_evals) * 100
            print(f"  Raw Failures per Evaluation:           {raw_efficiency:>10.2f}%")
            print(f"  Representative per Evaluation:         {rep_efficiency:>10.2f}%")

        if self.search_time > 0:
            print(f"  Raw Failures per Second:               {self.raw_failures/self.search_time:>10.2f}")
            print(f"  Representative per Second:             {self.representative_failures/self.search_time:>10.2f}")

        if self.ground_truth_cells:
            print("\n[5] FAILURE DOMAIN COVERAGE (vs Grid Search)")
            print(f"  Ground Truth Collision Cells:          {len(self.ground_truth_cells):>10}")
            print(f"  Captured by Raw Failures:              {self.captured_cells_raw:>10} ({self.coverage_rate_raw:>6.2f}%)")
            print(f"  Captured by Representative:            {self.captured_cells_rep:>10} ({self.coverage_rate_rep:>6.2f}%)")

        print("="*70 + "\n")

    def plot_fdc_curves(self, output_dir: str = 'results'):
        """绘制 Failure Discovery Curve"""
        os.makedirs(output_dir, exist_ok=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # ===== Left: Raw Failures FDC =====
        if self.fdc_curve_raw:
            budgets_raw = [p[0] for p in self.fdc_curve_raw]
            failures_raw = [p[1] for p in self.fdc_curve_raw]

            ax1.plot(budgets_raw, failures_raw, 'b-o', linewidth=2, markersize=4, label='FDC Curve')
            ax1.axhline(y=50, color='r', linestyle='--', alpha=0.7, label='Target (N=50)')

            if self.n_50_raw != -1:
                ax1.axvline(x=self.n_50_raw, color='g', linestyle='--', alpha=0.7, label=f'N_50 = {self.n_50_raw}')
                ax1.plot(self.n_50_raw, 50, 'g*', markersize=15)

            ax1.set_xlabel('Evaluation Budget', fontsize=11, fontweight='bold')
            ax1.set_ylabel('Cumulative Failures Found', fontsize=11, fontweight='bold')
            ax1.set_title(f'FDC - Raw Failures\n(AUC-FDC = {self.auc_fdc_raw:.4e})', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='best')

        # ===== Right: Representative Failures FDC =====
        if self.fdc_curve_rep:
            budgets_rep = [p[0] for p in self.fdc_curve_rep]
            failures_rep = [p[1] for p in self.fdc_curve_rep]

            ax2.plot(budgets_rep, failures_rep, 'r-s', linewidth=2, markersize=4, label='FDC Curve')
            ax2.axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='Target (N=50)')

            if self.n_50_rep != -1:
                ax2.axvline(x=self.n_50_rep, color='purple', linestyle='--', alpha=0.7, label=f'N_50 = {self.n_50_rep}')
                ax2.plot(self.n_50_rep, 50, 'mo', markersize=12)

            ax2.set_xlabel('Evaluation Budget', fontsize=11, fontweight='bold')
            ax2.set_ylabel('Cumulative Failures Found', fontsize=11, fontweight='bold')
            ax2.set_title(f'FDC - Representative Failures\n(AUC-FDC = {self.auc_fdc_rep:.4e})', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='best')

        plt.tight_layout()
        output_path = osp.join(output_dir, 'fdc_curves.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[+] Saved: {output_path}")
        plt.close()

    def plot_comparison_metrics(self, output_dir: str = 'results'):
        """Plot performance metrics comparison (6 subplots with normalized AUC)"""
        os.makedirs(output_dir, exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))

        # ===== [0,0] Total Failures Comparison =====
        ax = axes[0, 0]
        categories = ['Raw', 'Representative']
        counts = [self.raw_failures, self.representative_failures]
        colors = ['#3498db', '#e74c3c']
        bars = ax.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(count)}', ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_ylabel('Number of Failures', fontsize=11, fontweight='bold')
        ax.set_title('Total Failures Discovered', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # ===== [0,1] AUC-FDC Comparison =====
        ax = axes[0, 1]
        aucs = [self.auc_fdc_raw, self.auc_fdc_rep]
        bars = ax.bar(categories, aucs, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

        for bar, auc in zip(bars, aucs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{auc:.2e}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_ylabel('AUC-FDC Value', fontsize=11, fontweight='bold')
        ax.set_title('AUC-FDC (Absolute)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # ===== [0,2] Normalized AUC Comparison =====
        ax = axes[0, 2]
        naucs = [self.nauc_fdc_raw, self.nauc_fdc_rep]
        bars = ax.bar(categories, naucs, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

        for bar, nauc in zip(bars, naucs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{nauc:.4e}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_ylabel('nAUC-FDC Value', fontsize=11, fontweight='bold')
        ax.set_title('nAUC-FDC (Normalized)\nnAUC = AUC / (T_total × N_max)', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # ===== [1,0] N_50 Comparison =====
        ax = axes[1, 0]
        n50_values = []
        n50_labels = []

        if self.n_50_raw != -1:
            n50_values.append(self.n_50_raw)
            n50_labels.append(f'Raw\n({self.n_50_raw})')
        else:
            n50_labels.append(f'Raw\n(Not reached)')

        if self.n_50_rep != -1:
            n50_values.append(self.n_50_rep)
            n50_labels.append(f'Rep\n({self.n_50_rep})')
        else:
            n50_labels.append(f'Rep\n(Not reached)')

        if n50_values:
            bars = ax.bar(range(len(n50_values)), n50_values, color=colors[:len(n50_values)],
                         alpha=0.7, edgecolor='black', linewidth=1.5)
            for bar, n50 in zip(bars, n50_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(n50)}', ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_xticks(range(len(n50_labels)))
        ax.set_xticklabels(n50_labels)
        ax.set_ylabel('Evaluation Budget', fontsize=11, fontweight='bold')
        ax.set_title('Budget to Find 50 Failures (N_50)\nLower is Better', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # ===== [1,1] Evaluation Budget Breakdown =====
        ax = axes[1, 1]
        surrogate_calls = self.total_evals - self.real_sims
        budget_parts = [surrogate_calls, self.real_sims]
        budget_labels = [f'Surrogate\n({surrogate_calls})', f'Real CARLA\n({self.real_sims})']
        colors_budget = ['#2ecc71', '#f39c12']

        wedges, texts, autotexts = ax.pie(budget_parts, labels=budget_labels, colors=colors_budget,
                                           autopct='%1.1f%%', startangle=90,
                                           textprops={'fontsize': 10, 'fontweight': 'bold'})
        ax.set_title(f'Evaluation Budget Breakdown\n(Total: {self.total_evals})',
                    fontsize=12, fontweight='bold')

        # ===== [1,2] Efficiency Metrics =====
        ax = axes[1, 2]
        ax.axis('off')

        # Create efficiency metrics text
        metrics_text = "EFFICIENCY METRICS\n" + "="*35 + "\n\n"

        if self.total_evals > 0:
            raw_eff = (self.raw_failures / self.total_evals) * 100
            rep_eff = (self.representative_failures / self.total_evals) * 100
            metrics_text += f"Raw Failures/Eval:         {raw_eff:.2f}%\n"
            metrics_text += f"Representative/Eval:       {rep_eff:.2f}%\n\n"

        if self.search_time > 0:
            raw_per_sec = self.raw_failures / self.search_time
            rep_per_sec = self.representative_failures / self.search_time
            metrics_text += f"Raw Failures/Second:       {raw_per_sec:.2f}\n"
            metrics_text += f"Representative/Second:     {rep_per_sec:.2f}\n\n"

        if self.representative_failures > 0:
            dedupe_ratio = self.raw_failures / self.representative_failures
            metrics_text += f"Deduplication Ratio:       {dedupe_ratio:.2f}x\n"

        metrics_text += "\nFormula: nAUC = AUC / (T_total × N_max)"

        ax.text(0.1, 0.5, metrics_text, fontsize=10, verticalalignment='center',
               family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        output_path = osp.join(output_dir, 'comparison_metrics.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[+] Saved: {output_path}")
        plt.close()

    def plot_coverage_visualization(self, output_dir: str = 'results'):
        """Visualize failure domain coverage in 2D projections"""
        if not self.ground_truth_cells or self.grid_x_data is None:
            print("[!] Skipping coverage visualization (no grid data)")
            return

        os.makedirs(output_dir, exist_ok=True)

        # Prepare data
        grid_collision_points = self.grid_x_data[(np.array(self.grid_y_data) > self.collision_threshold)]
        found_points = np.atleast_2d(self.hazardous_points)

        # Create 2D projections: (x1,x2), (x1,x3), (x2,x3)
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        projections = [
            (0, 1, 'x1-x2 Projection'),
            (0, 2, 'x1-x3 Projection'),
            (1, 2, 'x2-x3 Projection')
        ]

        for ax, (dim1, dim2, title) in zip(axes, projections):
            # Plot Grid Search collision points (ground truth)
            ax.scatter(grid_collision_points[:, dim1], grid_collision_points[:, dim2],
                      c='red', s=50, alpha=0.5, label='Grid Search (Ground Truth)', marker='s')

            # Plot RLSAN found points
            if len(found_points) > 0:
                ax.scatter(found_points[:, dim1], found_points[:, dim2],
                          c='blue', s=30, alpha=0.7, label='RLSAN Found', marker='o')

            # Mark captured cells (grid cell centers)
            if hasattr(self, 'captured_cells_raw') and self.captured_cells_raw > 0:
                captured_centers = []
                for point in found_points:
                    grid_idx = self._point_to_grid_index(point)
                    key = f"{grid_idx[0]}_{grid_idx[1]}_{grid_idx[2]}"
                    if key in self.ground_truth_cells:
                        # Compute grid cell center
                        center = np.array([grid_idx[0], grid_idx[1], grid_idx[2]]) * self.grid_step + self.search_space_min
                        captured_centers.append(center)

                if captured_centers:
                    captured_centers = np.array(captured_centers)
                    ax.scatter(captured_centers[:, dim1], captured_centers[:, dim2],
                              c='green', s=150, alpha=0.8, marker='*', label=f'Captured Cells ({len(set([tuple(p) for p in captured_centers]))})',
                              edgecolors='darkgreen', linewidth=2)

            ax.set_xlabel(f'Dimension {dim1+1}', fontsize=11, fontweight='bold')
            ax.set_ylabel(f'Dimension {dim2+1}', fontsize=11, fontweight='bold')
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)
            ax.legend(loc='best', fontsize=9)

        plt.tight_layout()
        output_path = osp.join(output_dir, 'coverage_visualization_2d.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[+] Saved: {output_path}")
        plt.close()

    def plot_coverage_comparison(self, output_dir: str = 'results'):
        """Plot coverage comparison: raw vs representative"""
        if not self.ground_truth_cells:
            print("[!] Skipping coverage comparison (no grid data)")
            return

        os.makedirs(output_dir, exist_ok=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # ===== Left: Coverage Rate Comparison =====
        labels = ['Raw Failures\n(All 791 points)', 'Representative\n(104 points)']
        coverage_rates = [self.coverage_rate_raw, self.coverage_rate_rep]
        captured = [self.captured_cells_raw, self.captured_cells_rep]
        colors = ['#3498db', '#e74c3c']

        bars = ax1.bar(labels, coverage_rates, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

        # Add value labels
        for bar, rate, cap in zip(bars, coverage_rates, captured):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{rate:.2f}%\n({cap} cells)',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

        ax1.axhline(y=100, color='green', linestyle='--', alpha=0.5, linewidth=2, label='Perfect Coverage')
        ax1.set_ylabel('Coverage Rate (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Failure Domain Coverage Comparison', fontsize=13, fontweight='bold')
        ax1.set_ylim(0, 110)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.legend()

        # ===== Right: Cells Captured Comparison =====
        categories = ['Ground Truth', 'Captured (Raw)', 'Captured (Rep)']
        cell_counts = [len(self.ground_truth_cells), self.captured_cells_raw, self.captured_cells_rep]
        bar_colors = ['#95a5a6', '#3498db', '#e74c3c']

        bars = ax2.bar(categories, cell_counts, color=bar_colors, alpha=0.7, edgecolor='black', linewidth=2)

        # Add value labels
        for bar, count in zip(bars, cell_counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(count)}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

        ax2.set_ylabel('Number of Cells', fontsize=12, fontweight='bold')
        ax2.set_title('Captured Grid Cells Comparison', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        output_path = osp.join(output_dir, 'coverage_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[+] Saved: {output_path}")
        plt.close()

    def plot_convergence_analysis(self, output_dir: str = 'results'):
        """Plot convergence analysis"""
        os.makedirs(output_dir, exist_ok=True)

        fig = plt.figure(figsize=(14, 5))

        # ===== Left: Discovery Rate Over Time =====
        if self.fdc_curve_raw and self.fdc_curve_rep:
            ax1 = plt.subplot(1, 2, 1)

            budgets_raw = np.array([p[0] for p in self.fdc_curve_raw])
            failures_raw = np.array([p[1] for p in self.fdc_curve_raw])

            budgets_rep = np.array([p[0] for p in self.fdc_curve_rep])
            failures_rep = np.array([p[1] for p in self.fdc_curve_rep])

            # 计算发现率 (failures per 100 evaluations)
            if len(budgets_raw) > 1:
                rate_raw = np.diff(failures_raw) / np.diff(budgets_raw) * 100
                ax1.plot(budgets_raw[1:], rate_raw, 'b-o', linewidth=2, markersize=4, label='Raw Failure Rate')

            if len(budgets_rep) > 1:
                rate_rep = np.diff(failures_rep) / np.diff(budgets_rep) * 100
                ax1.plot(budgets_rep[1:], rate_rep, 'r-s', linewidth=2, markersize=4, label='Rep Failure Rate')

            ax1.set_xlabel('Evaluation Budget', fontsize=11, fontweight='bold')
            ax1.set_ylabel('Discovery Rate (failures per 100 evals)', fontsize=11, fontweight='bold')
            ax1.set_title('Failure Discovery Rate Over Time', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='best')

        # ===== Right: Cumulative Comparison =====
        ax2 = plt.subplot(1, 2, 2)

        if self.fdc_curve_raw:
            budgets_raw = [p[0] for p in self.fdc_curve_raw]
            failures_raw = [p[1] for p in self.fdc_curve_raw]
            ax2.plot(budgets_raw, failures_raw, 'b-o', linewidth=2, markersize=4, label='Raw Failures')

        if self.fdc_curve_rep:
            budgets_rep = [p[0] for p in self.fdc_curve_rep]
            failures_rep = [p[1] for p in self.fdc_curve_rep]
            ax2.plot(budgets_rep, failures_rep, 'r-s', linewidth=2, markersize=4, label='Representative Failures')

        ax2.set_xlabel('Evaluation Budget', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Cumulative Failures', fontsize=11, fontweight='bold')
        ax2.set_title('Cumulative Failure Discovery - Comparison', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best')

        plt.tight_layout()
        output_path = osp.join(output_dir, 'convergence_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[+] Saved: {output_path}")
        plt.close()

    def generate_report(self, output_dir: str = 'results'):
        """生成完整的分析报告"""
        os.makedirs(output_dir, exist_ok=True)

        report_path = osp.join(output_dir, 'analysis_report.txt')

        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("RLSAN SEARCH RESULTS ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Basic Statistics
            f.write("="*80 + "\n")
            f.write("1. BASIC STATISTICS\n")
            f.write("="*80 + "\n")
            f.write(f"Total Evaluations (Surrogate + Real):  {self.total_evals}\n")
            f.write(f"Real CARLA Simulations:                {self.real_sims}\n")
            f.write(f"Surrogate Model Calls:                 {self.total_evals - self.real_sims}\n")
            f.write(f"Search Time:                           {self.search_time:.2f} seconds\n")
            f.write(f"Average Time per Evaluation:           {self.search_time/max(self.total_evals, 1):.4f} s\n\n")

            # Raw Failures
            f.write("="*80 + "\n")
            f.write("2. RAW FAILURES (Unfiltered)\n")
            f.write("="*80 + "\n")
            f.write(f"Total Found:                           {self.raw_failures}\n")
            f.write(f"AUC-FDC (Raw):                         {self.auc_fdc_raw:.4e}\n")
            f.write(f"nAUC-FDC (Raw):                        {self.nauc_fdc_raw:.4e}\n")
            if self.n_50_raw == -1:
                f.write(f"N_50 (Raw):                            Not reached ({self.raw_failures}/50)\n")
            else:
                f.write(f"N_50 (Raw):                            {self.n_50_raw}\n")
            f.write(f"FDC Curve Points:                      {len(self.fdc_curve_raw)}\n\n")

            # Representative Failures
            f.write("="*80 + "\n")
            f.write("3. REPRESENTATIVE FAILURES (Deduplicated)\n")
            f.write("="*80 + "\n")
            f.write(f"Total Found:                           {self.representative_failures}\n")
            f.write(f"AUC-FDC (Representative):              {self.auc_fdc_rep:.4e}\n")
            f.write(f"nAUC-FDC (Representative):             {self.nauc_fdc_rep:.4e}\n")
            if self.n_50_rep == -1:
                f.write(f"N_50 (Representative):                 Not reached ({self.representative_failures}/50)\n")
            else:
                f.write(f"N_50 (Representative):                 {self.n_50_rep}\n")
            f.write(f"FDC Curve Points:                      {len(self.fdc_curve_rep)}\n")
            f.write(f"Deduplication Ratio:                   {self.raw_failures/max(self.representative_failures, 1):.2f}x\n\n")

            # Efficiency Metrics
            f.write("="*80 + "\n")
            f.write("4. EFFICIENCY METRICS\n")
            f.write("="*80 + "\n")
            if self.total_evals > 0:
                f.write(f"Raw Failures per Evaluation:           {(self.raw_failures/self.total_evals)*100:.2f}%\n")
                f.write(f"Representative per Evaluation:         {(self.representative_failures/self.total_evals)*100:.2f}%\n")
            if self.search_time > 0:
                f.write(f"Raw Failures per Second:               {self.raw_failures/self.search_time:.2f}\n")
                f.write(f"Representative per Second:             {self.representative_failures/self.search_time:.2f}\n")

            # Failure Domain Coverage
            if self.ground_truth_cells:
                f.write("\n" + "="*80 + "\n")
                f.write("5. FAILURE DOMAIN COVERAGE (vs Grid Search)\n")
                f.write("="*80 + "\n")
                f.write(f"Ground Truth Collision Cells:           {len(self.ground_truth_cells)}\n")
                f.write(f"\nCoverage with Raw Failures (all points):\n")
                f.write(f"  Captured Cells:                        {self.captured_cells_raw}\n")
                f.write(f"  Coverage Rate (CRate_raw):             {self.coverage_rate_raw:.2f}%\n")
                f.write(f"  Formula: CRate = |C_RLSAN| / |C_fail| × 100%\n\n")
                f.write(f"Coverage with Representative Failures (deduplicated):\n")
                f.write(f"  Captured Cells:                        {self.captured_cells_rep}\n")
                f.write(f"  Coverage Rate (CRate_rep):             {self.coverage_rate_rep:.2f}%\n")
                f.write(f"  Formula: CRate = |C_RLSAN| / |C_fail| × 100%\n\n")

            f.write("\n" + "="*80 + "\n")
            f.write(f"{'6' if self.ground_truth_cells else '5'}. INTERPRETATION\n")
            f.write("="*80 + "\n")
            f.write("AUC-FDC (Area Under Failure Discovery Curve):\n")
            f.write("  - Larger values indicate better early-stage search efficiency\n")
            f.write("  - Reflects how quickly the algorithm discovers failures\n\n")
            f.write("nAUC-FDC (Normalized Area Under Curve):\n")
            f.write("  - Formula: nAUC = AUC / (T_total × N_max)\n")
            f.write("  - Normalizes AUC by total budget and discovered failures\n")
            f.write("  - Enables fair comparison across different experiments\n")
            f.write("  - Larger values indicate better efficiency per unit budget\n\n")
            f.write("N_50 (Budget to Find 50 Failures):\n")
            f.write("  - Smaller values indicate higher search efficiency\n")
            f.write("  - More stable metric than 'first failure' (avoids luck factor)\n\n")
            f.write("Raw vs Representative:\n")
            f.write("  - Raw: All discovered failures (quantity metric)\n")
            f.write("  - Representative: After deduplication/clustering (quality metric)\n")
            f.write(f"  - Current deduplication ratio: {self.raw_failures/max(self.representative_failures, 1):.2f}x\n\n")

            if self.ground_truth_cells:
                f.write("Failure Domain Coverage (FDC):\n")
                f.write("  - Measures diversity of discovered failures compared to ground truth\n")
                f.write("  - Test space χ is partitioned into M disjoint cells C = {c1, c2, ..., cM}\n")
                f.write("  - Ground truth C_fail: all cells with collisions detected by grid search\n")
                f.write("  - RLSAN result C_RLSAN: cells in C_fail that contain found failures\n")
                f.write("  - Coverage Rate = |C_RLSAN| / |C_fail| × 100%\n")
                f.write("  - Higher coverage indicates better coverage of failure domain\n\n")

        print(f"[+] Saved: {report_path}")

    def run_all_analysis(self, output_dir: str = 'results'):
        """Run comprehensive analysis and visualization"""
        print("\n[*] Running comprehensive analysis...")

        self.print_summary()
        self.plot_fdc_curves(output_dir)
        self.plot_comparison_metrics(output_dir)
        self.plot_convergence_analysis(output_dir)

        # New visualization for coverage
        if self.ground_truth_cells:
            self.plot_coverage_visualization(output_dir)
            self.plot_coverage_comparison(output_dir)

        self.generate_report(output_dir)

        print(f"\n[✓] Analysis complete! Results saved to: {osp.abspath(output_dir)}/")


def main():
    parser = argparse.ArgumentParser(description='Analyze RLSAN search results with coverage metrics')
    parser.add_argument('--results_path', type=str, default='../../../../log/new_search_results.pkl',
                       help='Path to search_results.pkl file')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Directory to save analysis plots and reports')
    parser.add_argument('--grid_x', type=str, default='../../surrogate/train_data/scenario01_grid_x.pkl',
                       help='Path to grid X coordinates (scenario01_grid_x.pkl)')
    parser.add_argument('--grid_y', type=str, default='../../surrogate/train_data/scenario01_grid_y.pkl',
                       help='Path to grid Y values (scenario01_grid_y.pkl)')
    parser.add_argument('--collision_threshold', type=float, default=0.3,
                       help='Threshold for collision detection in grid search')

    args = parser.parse_args()

    try:
        analyzer = SearchResultsAnalyzer(
            results_path=args.results_path,
            grid_x_path=args.grid_x,
            grid_y_path=args.grid_y,
            collision_threshold=args.collision_threshold
        )
        analyzer.run_all_analysis(args.output_dir)
    except FileNotFoundError as e:
        print(f"[!] Error: {e}")
        print(f"[!] Tried to load from: {osp.abspath(args.results_path)}")
        exit(1)
    except Exception as e:
        print(f"[!] Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == '__main__':
    main()
