import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def extract_metrics(mat_file):
    """从 .mat 文件提取指标"""
    try:
        data = sio.loadmat(mat_file)
        return data
    except Exception as e:
        print(f"错误读取 {mat_file}: {e}")
        return None

def print_results(data, filename):
    """打印结果"""
    print(f"\n{'='*60}")
    print(f"文件: {filename}")
    print(f"{'='*60}")

    if data is None:
        return

    # 打印所有变量
    for key in sorted(data.keys()):
        if not key.startswith('__'):
            value = data[key]
            if isinstance(value, np.ndarray):
                if value.size == 1:
                    print(f"{key}: {value.item()}")
                elif value.size <= 10:
                    print(f"{key}: {value.flatten()}")
                else:
                    print(f"{key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"{key}: {value}")

# 处理 DDPG 数据
ddpg_multi = extract_metrics('ddpg_multi_run_data.mat')
print_results(ddpg_multi, 'ddpg_multi_run_data.mat')

ddpg_run1 = extract_metrics('ddpg_run_1.mat')
print_results(ddpg_run1, 'ddpg_run_1.mat')
