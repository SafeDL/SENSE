"""
代理模型比较实验框架
对比不同机器学习方法实现的代理模型性能，包括：
- RF (随机森林)
- RBF (径向基函数插值)
- MLP (多层感知机神经网络)
- XGBoost
- KNN (K近邻)
- LightGBM (轻量级梯度提升)
- Random (随机基线)
"""
import numpy as np
import pickle
import time
import warnings
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from scipy.interpolate import RBFInterpolator
import xgboost as xgb
import scipy.io as sio
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ==========================================
# 1. 基类定义
# ==========================================
class SurrogateModel(ABC):
    """代理模型基类"""
    
    @abstractmethod
    def fit(self, X_train, y_train):
        """训练模型"""
        pass
    
    @abstractmethod
    def predict(self, X_test):
        """预测，返回 (mean, std)"""
        pass
    
    @abstractmethod
    def get_name(self):
        """返回模型名称"""
        pass


# ==========================================
# 2. 各种代理模型实现
# ==========================================

class RandomForestSurrogate(SurrogateModel):
    """基于随机森林的代理模型"""
    
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
    def get_name(self):
        return "rf"
    
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train.ravel())
    
    def predict(self, X_test):
        mean_preds = self.model.predict(X_test)
        # 使用各树预测的标准差作为不确定性估计
        tree_preds = np.array([tree.predict(X_test) for tree in self.model.estimators_])
        std_preds = np.std(tree_preds, axis=0)
        return mean_preds, std_preds


class MLPSurrogate(SurrogateModel):
    """基于多层感知机(MLP)神经网络的代理模型"""
    
    def __init__(self):
        self.model = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
            warm_start=True  # 支持增量训练
        )
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_fitted = False
        
    def get_name(self):
        return "mlp"
    
    def fit(self, X_train, y_train):
        if not self.is_fitted:
            X_scaled = self.scaler_x.fit_transform(X_train)
            y_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
            self.is_fitted = True
        else:
            X_scaled = self.scaler_x.transform(X_train)
            y_scaled = self.scaler_y.transform(y_train.reshape(-1, 1)).ravel()
        
        # 限制样本数量以加快训练
        max_samples = min(len(X_train), 5000)
        if len(X_train) > max_samples:
            indices = np.random.choice(len(X_train), max_samples, replace=False)
            X_scaled = X_scaled[indices]
            y_scaled = y_scaled[indices]
        
        self.model.fit(X_scaled, y_scaled)
    
    def predict(self, X_test):
        X_scaled = self.scaler_x.transform(X_test)
        y_scaled = self.model.predict(X_scaled)
        mean_preds = self.scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).ravel()
        # MLP没有不确定性估计
        std_preds = np.zeros(len(X_test))
        return mean_preds, std_preds


class RBFSurrogate(SurrogateModel):
    """基于RBF插值的代理模型"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        
    def get_name(self):
        return "rbf"
    
    def fit(self, X_train, y_train):
        # 标准化输入
        X_scaled = self.scaler.fit_transform(X_train)
        # RBF插值 (限制样本数量以避免内存问题)
        max_samples = min(len(X_train), 3000)
        if len(X_train) > max_samples:
            indices = np.random.choice(len(X_train), max_samples, replace=False)
            X_scaled = X_scaled[indices]
            y_train = y_train[indices]
        
        self.model = RBFInterpolator(X_scaled, y_train.ravel(), kernel='thin_plate_spline')
    
    def predict(self, X_test):
        X_scaled = self.scaler.transform(X_test)
        # 分批预测
        batch_size = 10000
        n = len(X_test)
        mean_preds = np.zeros(n)
        
        for i in range(0, n, batch_size):
            batch_indices = slice(i, min(i + batch_size, n))
            mean_preds[batch_indices] = self.model(X_scaled[batch_indices])
        
        # RBF没有不确定性估计
        std_preds = np.zeros(n)
        return mean_preds, std_preds


class XGBSurrogate(SurrogateModel):
    """基于XGBoost的代理模型"""
    
    def __init__(self):
        self.model = xgb.XGBRegressor(
            n_estimators=50,
            max_depth=6,
            learning_rate=0.1,
            tree_method='hist',
            random_state=42,
            device="cuda",
            n_jobs=-1
        )
        
    def get_name(self):
        return "xgb"
    
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train.ravel())
    
    def predict(self, X_test):
        mean_preds = self.model.predict(X_test)
        # XGBoost没有不确定性估计
        std_preds = np.zeros(len(X_test))
        return mean_preds, std_preds


class KNNSurrogate(SurrogateModel):
    """基于KNN的代理模型"""
    
    def __init__(self, n_neighbors=5):
        self.model = KNeighborsRegressor(n_neighbors=n_neighbors, n_jobs=-1)
        self.scaler = StandardScaler()
        
    def get_name(self):
        return "knn"
    
    def fit(self, X_train, y_train):
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train.ravel())
    
    def predict(self, X_test):
        X_scaled = self.scaler.transform(X_test)
        mean_preds = self.model.predict(X_scaled)
        std_preds = np.zeros(len(X_test))
        return mean_preds, std_preds


class LightGBMSurrogate(SurrogateModel):
    """基于LightGBM的代理模型"""
    
    def __init__(self):
        self.model = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=8,
            learning_rate=0.1,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
    def get_name(self):
        return "lgbm"
    
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train.ravel())
    
    def predict(self, X_test):
        mean_preds = self.model.predict(X_test)
        std_preds = np.zeros(len(X_test))
        return mean_preds, std_preds


class RandomSurrogate(SurrogateModel):
    """随机预测基线 - 使用固定随机种子确保可复现"""
    
    def __init__(self):
        self.y_mean = 0
        self.y_std = 1
        self._cached_preds = None
        self._cache_size = 0
        
    def get_name(self):
        return "random"
    
    def fit(self, X_train, y_train):
        self.y_mean = np.mean(y_train)
        self.y_std = np.std(y_train)
        # 重置缓存
        self._cached_preds = None
        self._cache_size = 0
    
    def predict(self, X_test):
        n = len(X_test)
        # 使用缓存确保相同X_test返回相同结果
        if self._cached_preds is None or n != self._cache_size:
            # 使用固定种子生成，确保可复现
            rng = np.random.RandomState(42)
            self._cached_preds = rng.normal(self.y_mean, self.y_std, n)
            self._cache_size = n
        
        std_preds = np.ones(n) * self.y_std
        return self._cached_preds.copy(), std_preds


# ==========================================
# 3. 采集函数 (用于主动学习)
# ==========================================
def acquisition_function_simple(model, X_data, active_mask, top_n=5):
    """
    简化的采集函数：基于预测不确定性和距离
    """
    candidate_indices_full = np.where(~active_mask)[0]
    if len(candidate_indices_full) == 0:
        return []
    
    # 下采样
    subsample_size = 30000
    if len(candidate_indices_full) > subsample_size:
        eval_indices = np.random.choice(len(candidate_indices_full), size=subsample_size, replace=False)
        candidate_indices = candidate_indices_full[eval_indices]
    else:
        candidate_indices = candidate_indices_full
    
    X_cand = X_data[candidate_indices]
    mean_pred, std_pred = model.predict(X_cand)
    
    # 如果模型有不确定性估计，使用UCB策略
    if np.sum(std_pred) > 0:
        ucb = mean_pred + 1.5 * std_pred
        best_local = np.argsort(ucb)[-top_n:]
    else:
        # 否则随机选择高预测值点
        best_local = np.argsort(mean_pred)[-top_n:]
    
    return candidate_indices[best_local]


# ==========================================
# 4. F1-Score 计算
# ==========================================
def compute_f1_score(y_pred, y_true, threshold=0.3):
    """
    计算F1-Score指标
    """
    pred_cls = (y_pred > threshold).astype(int)
    true_cls = (y_true.ravel() > threshold).astype(int)
    return f1_score(true_cls, pred_cls)


# ==========================================
# 5. 主训练流程
# ==========================================
def train_surrogate_model(model, X_data, y_data, num_iterations=200, initial_samples=500, samples_per_iter=25):
    """
    训练单个代理模型并记录F1-Score曲线
    
    Args:
        model: SurrogateModel 实例
        X_data: 全量输入数据
        y_data: 全量输出数据
        num_iterations: 迭代次数
        initial_samples: 初始采样数
        samples_per_iter: 每轮新增样本数
    
    Returns:
        test_numbers: 测试样本数列表
        f1_scores: 对应的F1-Score列表
    """
    np.random.seed(42)
    
    # 初始采样
    initial_indices = np.random.choice(len(X_data), size=initial_samples, replace=False)
    active_mask = np.zeros(len(X_data), dtype=bool)
    active_mask[initial_indices] = True
    
    X_train = X_data[initial_indices]
    Y_train = y_data[initial_indices]
    
    iterations = []
    f1_scores = []
    
    model_name = model.get_name()
    pbar = tqdm(range(num_iterations), desc=f"Training {model_name}")
    
    for iteration in pbar:
        # 训练
        try:
            model.fit(X_train, Y_train)
        except Exception as e:
            print(f"Error training {model_name} at iteration {iteration}: {e}")
            break
        
        # 评估 (每1轮)
        mean_pred, _ = model.predict(X_data)
        f1 = compute_f1_score(mean_pred, y_data)
        
        iterations.append(iteration)
        f1_scores.append(f1)
        
        pbar.set_description(f"{model_name}: Iter={iteration}, F1={f1:.4f}")
        
        # 采样新点
        new_indices = acquisition_function_simple(model, X_data, active_mask, top_n=samples_per_iter)
        if len(new_indices) == 0:
            break
        
        active_mask[new_indices] = True
        X_train = np.vstack([X_train, X_data[new_indices]])
        Y_train = np.vstack([Y_train, y_data[new_indices]])
    
    return iterations, f1_scores


def run_comparison_experiment(X_data, y_data, output_path):
    """
    运行所有模型的对比实验
    """
    results = {
        'iterations': None,  # 将使用第一个模型的iterations
    }
    
    # 计算真实碰撞案例总数
    true_collisions = np.sum(y_data.ravel() > 0.3)
    print(f"\nTotal collision cases in dataset: {true_collisions}")
    
    # 创建所有模型
    models = [
        # RBFSurrogate(),
        RandomForestSurrogate(),
        MLPSurrogate(),
        # XGBSurrogate(),
        # KNNSurrogate(),
        # LightGBMSurrogate(),
        # RandomSurrogate(),
    ]
    
    for model in models:
        print(f"\n{'='*50}")
        print(f"Training {model.get_name().upper()} model...")
        print(f"{'='*50}")
        
        start_time = time.time()
        iters, f1_scores = train_surrogate_model(
            model, X_data, y_data,
            num_iterations=500,
            initial_samples=500,
            samples_per_iter=5
        )
        elapsed = time.time() - start_time
        print(f"{model.get_name()} training completed in {elapsed:.2f}s")
        
        # ========== 最终评估 ==========
        mean_pred, _ = model.predict(X_data)
        pred_cls = (mean_pred > 0.3).astype(int)
        true_cls = (y_data.ravel() > 0.3).astype(int)
        
        # 捕获的碰撞案例数 (True Positive)
        collision_captured = np.sum(pred_cls * true_cls)
        # 覆盖率 (Recall)
        coverage = collision_captured / (true_collisions + 1e-8)
        
        print(f"\n[Final Evaluation - {model.get_name().upper()}]")
        print(f"  Collision cases captured: {collision_captured} / {true_collisions}")
        print(f"  Coverage: {coverage:.4f} ({coverage*100:.2f}%)")
        
        # 保存结果
        model_name = model.get_name()
        results[f'{model_name}_f1'] = np.array(f1_scores)
        
        if results['iterations'] is None:
            results['iterations'] = np.array(iters)
    
    # 保存为.mat文件
    sio.savemat(output_path, results)
    print(f"\nResults saved to {output_path}")
    
    return results


# ==========================================
# 6. 入口函数
# ==========================================
if __name__ == "__main__":
    # 加载数据
    print("Loading data...")
    with open('./train_data/scenario05_grid_x.pkl', 'rb') as f:
        grid_x = pickle.load(f)[:80000]
    with open('./train_data/scenario05_grid_y.pkl', 'rb') as f:
        grid_y = np.array(pickle.load(f)).reshape(-1, 1)[:80000]
    
    print(f"Data Loaded: X={grid_x.shape}, Y={grid_y.shape}")
    print(f"Total Failures in Dataset: {np.sum(grid_y > 0.3)}")
    
    # 运行对比实验
    output_mat_path = '../../../matlab_scripts/surrogate_comparison.mat'
    results = run_comparison_experiment(grid_x, grid_y, output_mat_path)
    
    print("\n" + "="*50)
    print("Experiment completed!")
    print("="*50)
