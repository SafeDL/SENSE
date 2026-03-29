# -*- coding: utf-8 -*-
"""
Surrogate-Enhanced Baseline Framework

为baselines提供代理模型加速支持，使搜索更加高效
"""

import pickle
import numpy as np
import torch
import gpytorch
from typing import Tuple, Optional, Any


def load_surrogate_model(model_path: str) -> Tuple[Any, Any]:
    """
    Load surrogate model using the same logic as RLSAN

    Args:
        model_path: Path to surrogate_model_1000.pkl

    Returns:
        (gp_model, gp_likelihood) tuple or (None, None) if loading fails
    """
    if not model_path:
        return None, None

    try:
        # First try RLSAN's loader
        try:
            from rlsan.src.RLSearch.utils import load_surrogate_model as load_gp
            return load_gp(model_path)
        except ImportError:
            # Fall back to direct loading
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, dict):
                    return data['model'], data['likelihood']
                elif isinstance(data, tuple):
                    return data[0], data[1]
                else:
                    raise ValueError("Unknown model format")
    except Exception as e:
        print(f"[!] Failed to load surrogate model: {e}")
        return None, None


class SurrogateModel:
    """Wrapper for GP surrogate model"""

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize surrogate model

        Args:
            model_path: Path to surrogate_model_1000.pkl (or None for no model)
        """
        self.gp_model = None
        self.gp_likelihood = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if model_path:
            print(f"[*] Loading surrogate model from: {model_path}")
            self.gp_model, self.gp_likelihood = load_surrogate_model(model_path)

            if self.gp_model is not None:
                # Move model to device
                self.gp_model.to(self.device)
                if self.gp_likelihood is not None:
                    self.gp_likelihood.to(self.device)
                print(f"[✓] Surrogate model loaded (device: {self.device})")
            else:
                print(f"[!] Failed to load surrogate model from {model_path}")
        else:
            print(f"[*] No surrogate model path provided")

    def predict(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict risk scores using surrogate model

        Args:
            points: shape (N, 3) - test points

        Returns:
            means: (N,) - predicted risk scores
            variances: (N,) - prediction uncertainty (variance, aligned with ADScenarioEnv)
        """
        if self.gp_model is None:
            raise RuntimeError("[!] Surrogate model not initialized!")

        points = np.atleast_2d(points)
        x_tensor = torch.FloatTensor(points).to(self.device)

        self.gp_model.eval()
        if self.gp_likelihood is not None:
            self.gp_likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            if self.gp_likelihood is not None:
                posterior = self.gp_likelihood(self.gp_model(x_tensor))
            else:
                posterior = self.gp_model(x_tensor)

            means = posterior.mean.cpu().numpy()
            # Use variance instead of stddev for consistency with ADScenarioEnv
            variances = posterior.variance.cpu().numpy() if hasattr(posterior, 'variance') else np.zeros_like(means)

        return means, variances


class SurrogateEvaluator:
    """Surrogate-based evaluation wrapper with fallback to real runner"""

    def __init__(self, surrogate_model: Optional[SurrogateModel] = None,
                 real_runner: Optional[Any] = None,
                 uncertainty_threshold: float = 0.00217,
                 use_real_sim: bool = False):
        """
        Initialize evaluator with surrogate model and/or real runner

        Args:
            surrogate_model: SurrogateModel instance (can be None)
            real_runner: CarlaRunner for real simulation (optional)
            uncertainty_threshold: Threshold for using real simulation (variance, aligned with ADScenarioEnv)
            use_real_sim: Whether to use real simulation for uncertain points
        """
        self.surrogate = surrogate_model
        self.real_runner = real_runner
        self.uncertainty_threshold = uncertainty_threshold
        self.use_real_sim = use_real_sim

        self.surrogate_call_count = 0          # 调用代理模型的次数
        self.surrogate_final_count = 0         # 最终使用代理模型预测的次数
        self.real_simulation_count = 0         # 真实仿真的次数
        self.evaluation_count = 0              # 总的不同点评估数

    def evaluate(self, point: np.ndarray) -> float:
        """
        Evaluate a single point using surrogate model or real runner

        Args:
            point: (3,) - test point

        Returns:
            risk_score: float - predicted risk
        """
        point = np.atleast_2d(point)
        self.evaluation_count += 1  # 计数一个不同的点

        # Priority 1: Use surrogate if available and valid
        if self.surrogate is not None and self.surrogate.gp_model is not None:
            try:
                mean, variance = self.surrogate.predict(point)
                score = float(mean[0])
                self.surrogate_call_count += 1

                # Priority 2: Optionally use real sim for uncertain points (hybrid mode)
                if self.use_real_sim and self.real_runner is not None:
                    if variance[0] > self.uncertainty_threshold:
                        try:
                            score_ret, _ = self.real_runner.run(point)
                            score = float(score_ret[0])
                            self.real_simulation_count += 1
                            # 不增加evaluation_count，因为已经在上面计数过了
                        except Exception as e:
                            print(f"[!] Error in real simulation: {e}")
                            # 保持代理模型分数
                            self.surrogate_final_count += 1
                            return score
                    else:
                        # 代理模型不确定性低，使用代理预测
                        self.surrogate_final_count += 1
                        return score
                else:
                    # 不使用真实仿真，直接返回代理预测
                    self.surrogate_final_count += 1
                    return score

                return score
            except Exception as e:
                print(f"[!] Surrogate prediction failed: {e}, falling back to real runner")
                # Fall through to real runner

        # Priority 3: Fall back to real runner
        if self.real_runner is not None:
            try:
                score_ret, _ = self.real_runner.run(point)
                score = float(score_ret[0])
                self.real_simulation_count += 1
                return score
            except Exception as e:
                raise RuntimeError(f"[!] Both surrogate and real runner failed: {e}")

        # No evaluator available
        raise RuntimeError("[!] No surrogate model or real runner available!")

    def get_stats(self) -> dict:
        """Get evaluation statistics"""
        return {
            'total_evaluations': self.evaluation_count,
            'surrogate_calls': self.surrogate_call_count,
            'surrogate_final': self.surrogate_final_count,
            'real_simulations': self.real_simulation_count,
        }

    def reset_stats(self):
        """Reset statistics counters"""
        self.surrogate_call_count = 0
        self.surrogate_final_count = 0
        self.real_simulation_count = 0
        self.evaluation_count = 0


def _compute_fdc_metrics(all_scores: np.ndarray, collision_threshold: float = 0.3) -> dict:
    """
    Compute Failure Discovery Curve metrics

    Args:
        all_scores: Historical scores in evaluation order
        collision_threshold: Threshold for collision detection

    Returns:
        dict containing FDC curve, AUC, and n_50
    """
    collision_mask = all_scores > collision_threshold

    # Build cumulative failure discovery curve
    fdc_curve = []
    cumulative_failures = 0

    for i, is_collision in enumerate(collision_mask):
        if is_collision:
            cumulative_failures += 1
        fdc_curve.append((i + 1, cumulative_failures))  # (budget, failures found)

    if len(fdc_curve) == 0:
        return {
            'auc_fdc': 0.0,
            'n_50': -1,
            'fdc_curve': []
        }

    # Compute AUC using trapezoidal rule
    budgets = np.array([p[0] for p in fdc_curve])
    failures = np.array([p[1] for p in fdc_curve])
    auc_fdc = np.trapz(failures, budgets)

    # Find N_50: budget needed to find 50 failures
    n_50 = -1
    for budget, num_failures in fdc_curve:
        if num_failures >= 50:
            n_50 = budget
            break

    return {
        'auc_fdc': auc_fdc,
        'n_50': n_50,
        'fdc_curve': fdc_curve
    }


def prepare_baseline_results(scenarios: np.ndarray, scores: np.ndarray,
                            evaluator: SurrogateEvaluator,
                            method_name: str,
                            search_time: float,
                            grid_x_data: Optional[np.ndarray] = None,
                            grid_y_data: Optional[np.ndarray] = None,
                            collision_threshold: float = 0.3,
                            all_scores_history: Optional[np.ndarray] = None) -> dict:
    """
    Prepare baseline results in RLSAN-compatible format

    Args:
        scenarios: All evaluated scenarios
        scores: All evaluation scores (hazardous only, or all if used for FDC)
        evaluator: SurrogateEvaluator instance
        method_name: Name of the baseline method
        search_time: Total search time in seconds
        grid_x_data: Grid X coordinates (optional)
        grid_y_data: Grid Y values (optional)
        collision_threshold: Threshold for collision detection
        all_scores_history: Historical scores in evaluation order (for FDC computation)

    Returns:
        results: Dictionary compatible with RLSAN format
    """
    # Identify hazardous points (collision cases)
    collision_mask = scores > collision_threshold
    hazardous_points = scenarios[collision_mask]

    raw_failures_count = len(hazardous_points)
    total_evaluations = evaluator.evaluation_count
    real_simulations = evaluator.real_simulation_count
    surrogate_calls = evaluator.surrogate_call_count

    # NOTE: Grid-based metrics (coverage_rate, captured_cells, ground_truth_cells)
    # are NOT computed here. They will be computed in the analysis script.
    # This function only stores raw data needed for later analysis.
    coverage_rate_raw = 0.0
    captured_cells_raw = 0
    ground_truth_cells = 0

    # Compute FDC metrics if evaluation history is provided
    auc_fdc_raw = 0.0
    n_50_raw = -1
    fdc_curve_raw = []
    nauc_fdc_raw = 0.0

    if all_scores_history is not None:
        fdc_metrics = _compute_fdc_metrics(all_scores_history, collision_threshold)
        auc_fdc_raw = fdc_metrics['auc_fdc']
        n_50_raw = fdc_metrics['n_50']
        fdc_curve_raw = fdc_metrics['fdc_curve']

        # Compute normalized AUC: nAUC = AUC / (T_total * N_max)
        if total_evaluations > 0 and raw_failures_count > 0:
            nauc_fdc_raw = auc_fdc_raw / (total_evaluations * raw_failures_count)
        else:
            nauc_fdc_raw = 0.0

    # Prepare results dictionary - store ONLY raw data
    # Grid-based coverage metrics will be computed in analysis script
    results = {
        'method': method_name,
        'algorithm': 'baseline',
        'hazardous_points': np.array(hazardous_points),
        'all_samples': scenarios,
        'all_scores': scores,
        'raw_failures_count': raw_failures_count,
        'representative_failures_count': raw_failures_count,  # Will be deduplicated in analysis
        'total_evaluations': total_evaluations,
        'real_simulations': real_simulations,
        'surrogate_calls': surrogate_calls,
        'search_time': search_time,
        # Grid coverage metrics - NOT computed here, computed in analysis script
        'coverage_rate_raw': coverage_rate_raw,
        'coverage_rate_rep': coverage_rate_raw,
        'captured_cells_raw': captured_cells_raw,
        'captured_cells_rep': captured_cells_raw,
        'ground_truth_cells': ground_truth_cells,
        # FDC metrics - computed here because they need evaluation history
        'auc_fdc_raw': auc_fdc_raw,
        'auc_fdc_representative': auc_fdc_raw,  # Same as raw for baselines
        'n_50_raw': n_50_raw,
        'n_50_representative': n_50_raw,  # Same as raw for baselines
        'fdc_curve_raw': fdc_curve_raw,
        'fdc_curve_representative': fdc_curve_raw,  # Same as raw for baselines
        'nauc_fdc_raw': nauc_fdc_raw,
        'nauc_fdc_representative': nauc_fdc_raw,  # Same as raw for baselines
    }

    return results
