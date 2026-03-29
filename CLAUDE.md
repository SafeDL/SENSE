# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SENSE is an autonomous driving research platform combining multiple approaches for finding edge cases and validating autonomous driving systems:

- **TCP** (Trajectory-guided Control Prediction): End-to-end autonomous driving using trajectory guidance and control prediction with PyTorch
- **pylot**: Modular autonomous vehicle platform for testing perception, prediction, and planning components
- **safebench**: Testing framework for generating and evaluating dangerous scenarios
- **rlsan**: Reinforcement learning-driven particle swarm search for finding critical test cases
- **baselines**: Reference implementations of alternative methods (random search, genetic algorithms, Bayesian optimization)

## Architecture Overview

```
SENSE/
├── TCP/                      # Main autonomous driving model training/evaluation
│   ├── TCP/                  # Core training code
│   ├── leaderboard/          # CARLA leaderboard evaluation
│   ├── scenario_runner/      # Scenario execution
│   ├── roach/                # Baseline implementation
│   └── environment.yml       # Conda environment (Python 3.7, PyTorch 1.10, CUDA 11.3)
├── pylot/                    # Modular AV platform
│   ├── pylot/                # Core modules (detection, tracking, planning, control)
│   ├── configs/              # Configuration files for different components
│   └── requirements.txt       # Dependencies
├── safebench/                # Testing framework
├── rlsan/                    # RL-driven search implementation
├── baselines/                # Baseline comparison methods
└── scripts/
    ├── run.py                # Main SafeBench runner
    ├── run_batch_simulation.py
    └── run_rlsan_search.py
```

## Key Technologies

- **Deep Learning**: PyTorch, PyTorch Lightning, TensorFlow-GPU
- **Computer Vision**: OpenCV, Pillow, scikit-image, imgaug, albumentations
- **Simulation**: CARLA 0.9.10.1, scenario_runner
- **RL**: Gym, stable-baselines3
- **Data**: LMDB, HDF5, NuScenes dataset format

## Development Setup

### TCP Setup
```bash
# Create and activate conda environment
conda env create -f TCP/environment.yml --name TCP
conda activate TCP
export PYTHONPATH=$PYTHONPATH:PATH_TO_SENSE
```

### pylot Setup
```bash
cd pylot
./install.sh
pip install -e ./
export CARLA_HOME=$PYLOT_HOME/dependencies/CARLA_0.9.10.1/
```

## Common Development Commands

### CARLA Simulator
Before running training/evaluation, launch CARLA:
```bash
cd CARLA_ROOT
./CarlaUE4.sh --world-port=2000 -opengl
```

### TCP Training
1. Set dataset path in `TCP/config.py`
2. Run training:
```bash
python TCP/train.py --gpus NUM_OF_GPUS
```

### TCP Evaluation
1. Set paths in `leaderboard/scripts/run_evaluation.sh`
2. Run evaluation:
```bash
sh leaderboard/scripts/run_evaluation.sh
```

### TCP Data Collection
1. Configure paths in `leaderboard/scripts/data_collection.sh`
2. Start collection:
```bash
sh leaderboard/scripts/data_collection.sh
```
3. Filter and pack data:
```bash
python tools/filter_data.py
python tools/gen_data.py
```

### SafeBench Testing
```bash
python scripts/run.py \
  --exp_name <exp_name> \
  --output_dir <output_dir> \
  --mode eval \
  --agent_cfg behavior.yaml \
  --scenario_cfg lc.yaml \
  --device cuda:0
```

### RLSAN Search
```bash
python scripts/run_rlsan_search.py --exp_name <name> --device cuda:0
```

### Batch Simulations
```bash
python scripts/run_batch_simulation.py --threads 8 --device cuda:0
```

### Pylot Execution
```bash
# Demo with all components
python3 pylot.py --flagfile=configs/demo.conf

# Specific component (e.g., detection)
python3 pylot.py --flagfile=configs/detection.conf

# With evaluation
python3 pylot.py --flagfile=configs/detection.conf --evaluate_obstacle_detection
```

## Important Configuration Files

- **TCP/config.py**: Dataset paths, model hyperparameters
- **TCP/leaderboard/scripts/run_evaluation.sh**: Evaluation settings (CARLA path, model checkpoint, routes)
- **TCP/leaderboard/scripts/data_collection.sh**: Data collection paths and parameters
- **pylot/configs/*.conf**: Configuration for different pylot components
- **rlsan/config/standard.yaml**: RL search hyperparameters

## Key Project Files

- **TCP/TCP/train.py**: Main training loop for trajectory + control models
- **TCP/TCP/config.py**: Configuration and hyperparameters
- **pylot/pylot.py**: Main entry point for component orchestration
- **scripts/run.py**: SafeBench scenario runner with argparse-based CLI
- **safebench/carla_runner.py**: CARLA environment wrapper
- **safebench/carla_runner_simple.py**: Simplified runner for development

## Data and Models

- **TCP Dataset**: ~115GB from HuggingFace/GoogleDrive/BaiduYun
  - Training: `TCP/train.py` expects dataset configured in `TCP/config.py`
  - Format: CARLA sensor data with trajectory and control labels
- **Checkpoints**: Model `.pth` files specified in evaluation configs
- **CARLA Routes**: XML route files for leaderboard evaluation

## Python Version and Dependencies

- **Python 3.7** (TCP) or **3.8+** (pylot)
- Core: NumPy, SciPy, scikit-learn, scikit-image, Pillow
- GPU: CUDA 11.3, PyTorch 1.10, TensorFlow-GPU 2.5
- Testing: pytest
- Visualization: matplotlib, Jupyter

## Git and Citation

The repository acknowledges these dependencies:
- [Transfuser](https://github.com/autonomousvision/transfuser)
- [Roach](https://github.com/zhejz/carla-roach)
- [CARLA Leaderboard](https://github.com/carla-simulator/leaderboard)
- [Scenario Runner](https://github.com/carla-simulator/scenario_runner)

See repository-specific LICENSE files (Apache 2.0) in TCP/ and pylot/ subdirectories.

## Common Workflows

### Adding a New Scenario
1. Create YAML config in appropriate config directory
2. Load with `load_config()` in run.py
3. Scenario runner executes based on `policy_type` field

### Training Custom Models
1. Set `PYTHONPATH` to include project root
2. Prepare dataset with `safebench/util/run_util.py` utilities
3. Use `set_seed()` and `set_torch_variable()` from `safebench/util/torch_util.py` for reproducibility
4. Train with specified device and seed

### Running Evaluations
1. Configure CARLA path and port
2. Specify checkpoint paths
3. Use appropriate runner (CarlaRunner for training, leaderboard runner for evaluation)
4. Results saved to `--output_dir`
