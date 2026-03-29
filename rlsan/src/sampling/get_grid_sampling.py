"""
在指定的测试空间内进行grid search划分,作为构建代理模型和后续对比的基准
"""
import numpy as np
import os
import pickle as pkl
from utils import Grid_Search


if __name__ == "__main__":
    np.random.seed(42)
    test_parameters = {}
    test_parameters['x1'] = [-1, 1]
    test_parameters['x2'] = [-1, 1]
    test_parameters['x3'] = [-1, 1]
    test_parameters['x4'] = [-1, 1]
    sampled_parameters = Grid_Search(parameters=test_parameters, step=30)
    with open(os.path.join('../../results/grid/scenario01', 'grid_x.pkl'), 'wb') as f:
        pkl.dump(sampled_parameters, f)

