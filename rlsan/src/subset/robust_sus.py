import numpy as np
import torch
import gpytorch
import time
from tqdm import tqdm
import pickle
import os
from rlsan.src.surrogate.build_surrogate_new import GPModel, update_GPModel

class RobustSubsetSimulation:
    def __init__(
            self,
            surrogate_model,
            surrogate_likelihood,
            simulator_fn,
            input_dim,
            failure_threshold=0.3,
            p0=0.1,
            uncertainty_tol=0.1,
            max_levels=10,
            device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Robust Subset Simulation (SuS) with Self-Healing Surrogate.

        Args:
            surrogate_model: Trained GP model.
            surrogate_likelihood: Likelihood for the GP.
            simulator_fn: Function f(x) -> y (scalar) or (y, success_bool).
                          Should return a scalar performance metric where > threshold means failure.
            input_dim: Dimension of the input space.
            failure_threshold: The y-value defining failure (default 0.3).
            p0: Conditional probability (percentile) for each level (default 0.1).
            uncertainty_tol: Epistemic uncertainty (std) threshold to trigger simulator.
            max_levels: Maximum number of SuS levels.
            device: 'cpu' or 'cuda'.
        """
        self.model = surrogate_model
        self.likelihood = surrogate_likelihood
        self.simulator_fn = simulator_fn
        self.input_dim = input_dim
        self.failure_threshold = failure_threshold
        self.p0 = p0
        self.uncertainty_tol = uncertainty_tol
        self.max_levels = max_levels
        self.device = device
        
        self.model.to(self.device)
        self.likelihood.to(self.device)
        self.model.eval()
        self.likelihood.eval()

        self.levels = []
        self.thresholds = []
        self.collision_probs = []
        
        # Buffer for "Self-Healing"
        self.new_data_buffer_x = []
        self.new_data_buffer_y = []
        
        # Statistics
        self.simulator_calls = 0

    def get_prediction(self, x_batch):
        """
        Get GP prediction (mean, std).
        """
        self.model.eval()
        self.likelihood.eval()
        
        if not isinstance(x_batch, torch.Tensor):
            x_batch = torch.tensor(x_batch, dtype=torch.float32).to(self.device)
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = self.likelihood(self.model(x_batch))
            mean = posterior.mean.cpu().numpy()
            std = posterior.stddev.cpu().numpy()
            
        return mean, std

    def self_healing_evaluate(self, x):
        """
        Evaluate x using Surrogate. 
        If uncertain, call Simulator and update Surrogate.
        """
        # 1. Check Surrogate
        mu, sigma = self.get_prediction(x.reshape(1, -1))
        mu = mu[0]
        sigma = sigma[0]
        
        # 2. Uncertainty Trigger
        if sigma > self.uncertainty_tol:
            print(f"[Self-Healing] High uncertainty ({sigma:.4f} > {self.uncertainty_tol}) detected at {x[:3]}... Triggering Simulator.")
            
            # Call Simulator
            real_val = self.simulator_fn(x)
            self.simulator_calls += 1
            
            # Store for update
            self.new_data_buffer_x.append(x)
            self.new_data_buffer_y.append(real_val)
            
            # Instant update (lightweight)
            self._update_surrogate_online()
            
            return real_val, True  # (Value, UsedSimulator)
        else:
            return mu, False

    def _update_surrogate_online(self):
        """
        Update the GP model with buffered data (Lightweight).
        Does not perform full retraining, just refits the posterior conditioning.
        """
        if len(self.new_data_buffer_x) == 0:
            return

        # Prepare new data
        X_new = torch.tensor(np.array(self.new_data_buffer_x), dtype=torch.float32).to(self.device)
        Y_new = torch.tensor(np.array(self.new_data_buffer_y), dtype=torch.float32).view(-1).to(self.device)

        # Get old data
        X_old = self.model.train_inputs[0]
        Y_old = self.model.train_targets

        # Concatenate
        X_all = torch.cat([X_old, X_new])
        Y_all = torch.cat([Y_old, Y_new])

        # Set training data (GPyTorch handles the rest for ExactGP)
        self.model.set_train_data(inputs=X_all, targets=Y_all, strict=False)
        
        # Clear buffer (data is now in the model)
        self.new_data_buffer_x = []
        self.new_data_buffer_y = []
        
        # print("Surrogate updated with new experience.")

    def hybrid_initialization(self, niche_seeds, n_samples):
        """
        Step 1: Hybrid Initialization.
        Mix Niche Seeds (from RL) with Global Random Samples.
        """
        n_niche = len(niche_seeds)
        n_random = n_samples - n_niche
        
        if n_random < 0:
            print(f"Warning: Niche seeds ({n_niche}) > n_samples ({n_samples}). Truncating niche seeds.")
            initial_samples = niche_seeds[:n_samples]
        else:
            # Uniform random in [-1, 1] (Assuming normalized input space)
            random_samples = np.random.uniform(-1, 1, size=(n_random, self.input_dim))
            initial_samples = np.vstack([niche_seeds, random_samples])
            
        return initial_samples

    def adaptive_threshold(self, performance_values):
        """
        Step 2: Calculate intermediate threshold b_i.
        """
        # Sort descending (assuming larger y is worse/closer to failure)
        sorted_vals = np.sort(performance_values)[::-1]
        
        # Pick the site at p0 quantile
        # e.g., if N=1000, p0=0.1, we pick the 100-th largest value
        target_idx = int(self.p0 * len(performance_values))
        b_i = sorted_vals[target_idx]
        
        # Never go beyond the physical failure threshold
        if b_i > self.failure_threshold:
            b_i = self.failure_threshold
            
        return b_i

    def mmh_transition(self, seeds, current_threshold, n_chain_steps, proposal_std=0.2):
        """
        Step 3: Modified Metropolis-Hastings (MMH) to fill the level.
        """
        n_seeds = len(seeds)
        samples = []
        performance_vals = []
        
        # Determine chain length (N / n_seeds) to get back to N samples
        # Note: In standard SuS, we need N total samples. We have n_seeds seeds.
        # usually len_chain = N / n_seeds
        
        # For each seed, run a Markov Chain
        for i, seed in enumerate(tqdm(seeds, desc=f"MMH Evolution (Thresh={current_threshold:.3f})")):
            current_x = seed['x']
            current_y = seed['y'] # This should be > threshold already
            
            # Store the seed itself as the first element of the chain
            samples.append(current_x)
            performance_vals.append(current_y)
            
            for _ in range(n_chain_steps - 1):
                # 1. Proposal: Simple Gaussian Drift
                # (Can be improved to Component-wise for high dim)
                candidate_x = current_x + np.random.normal(0, proposal_std, size=self.input_dim)
                
                # Clip to domain [-1, 1]
                candidate_x = np.clip(candidate_x, -1, 1)
                
                # 2. Evaluation with Self-Healing
                candidate_y, used_sim = self.self_healing_evaluate(candidate_x)
                
                # 3. Acceptance Criterion (Indicator Function: I(y > b))
                # In SuS, stationary distribution is Conditional PDF given y > b.
                # If candidate is in the region (y > b), we accept with probability 1 (since proposal is symmetric-ish)
                # Actually, strictly, A = min(1, p(x')/p(x) * I(x')/I(x)) since we want uniform in Failure Region
                # If prior is uniform, p(x')=p(x), so A = 1 if I(x')=1.
                
                if candidate_y > current_threshold:
                    # Accept
                    current_x = candidate_x
                    current_y = candidate_y
                else:
                    # Reject (stay at current)
                    # current_x and current_y remain same
                    pass
                
                samples.append(current_x)
                performance_vals.append(current_y)
                
        return np.array(samples), np.array(performance_vals)

    def run(self, niche_seeds, n_samples_per_level=1000):
        """
        Main Execution Loop.
        """
        print(f"--- Starting Robust Subset Simulation ---")
        print(f"Goal: P(Y > {self.failure_threshold})")
        print(f"Method: Hybrid Init ({len(niche_seeds)} niches) + Self-healing GP")
        
        # --- Level 0 ---
        # 1. Init
        samples = self.hybrid_initialization(niche_seeds, n_samples_per_level)
        
        # 2. Evaluate (Batch)
        # For Level 0, we can rely on Surrogate for batch prediction to start,
        # but technically we should verify seeds? 
        # For safety/simplicity, let's iterate and use self_healing_evaluate
        y_values = []
        print("Evaluating Level 0 samples...")
        for x in tqdm(samples):
            # Evaluate using self-healing trigger
            val, _ = self.self_healing_evaluate(x)
            y_values.append(val)
        y_values = np.array(y_values)
        
        current_samples = samples
        current_y = y_values
        
        # SuS Loop
        for level in range(self.max_levels):
            # 1. Calculate Threshold
            b_current = self.adaptive_threshold(current_y)
            self.levels.append({
                'level': level,
                'threshold': b_current,
                'samples': current_samples,
                'y': current_y
            })
            self.thresholds.append(b_current)
            
            print(f"Level {level}: Threshold = {b_current:.4f} (Target: {self.failure_threshold})")
            
            # Check convergence
            if b_current >= self.failure_threshold:
                print("Target threshold reached!")
                b_current = self.failure_threshold # Clamp
                # The probability of this level is simply count / total
                count_fail = np.sum(current_y > self.failure_threshold)
                p_level = count_fail / len(current_y)
                self.collision_probs.append(p_level)
                break
            else:
                self.collision_probs.append(self.p0)
            
            # 2. Identify Seeds for next level
            # These are samples where y > b_current
            seed_mask = current_y >= b_current
            seeds_x = current_samples[seed_mask]
            seeds_y = current_y[seed_mask]
            
            seeds = [{'x': sx, 'y': sy} for sx, sy in zip(seeds_x, seeds_y)]
            print(f"Selected {len(seeds)} kernels (seeds) for next level.")
            
            # 3. MCMC Expansion
            # We need n_samples_per_level total. 
            n_seeds = len(seeds)
            chain_len = int(n_samples_per_level / n_seeds)
            
            next_samples, next_y = self.mmh_transition(seeds, b_current, chain_len)
            
            # Shuffle slightly to remove correlation artifacts in list order (optional)
            perm = np.random.permutation(len(next_samples))
            current_samples = next_samples[perm]
            current_y = next_y[perm]
            
            # Truncate if we generated slightly more/less due to rounding
            if len(current_samples) > n_samples_per_level:
                current_samples = current_samples[:n_samples_per_level]
                current_y = current_y[:n_samples_per_level]

        # Calculate Final Pf
        Pf = np.prod(self.collision_probs)
        cov = np.sqrt(np.sum([(1-p)/(p*n_samples_per_level) for p in self.collision_probs]))
        
        print(f"--- SuS Completed ---")
        print(f"Estimated Pf: {Pf:.4e}")
        print(f"C.O.V: {cov:.4f}")
        print(f"Total Simulator Calls (Self-Healing): {self.simulator_calls}")
        
        return {
            'Pf': Pf,
            'cov': cov,
            'levels': self.levels,
            'simulator_calls': self.simulator_calls
        }
