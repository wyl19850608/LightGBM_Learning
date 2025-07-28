import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Dict, Tuple, Optional, Union
from sklearn.feature_selection import f_classif

class FeaturePreprocessor:
    """特征预处理工具，处理多类型特征并进行特征选择"""

    def __init__(self,
                 continuous_features: List[str] = None,
                 categorical_features: List[str] = None,
                 boolean_features: List[str] = None,
                 string_features: List[str] = None,
                 missing_threshold: float = 0.3,
                 chi2_k: int = 100):
        """
        初始化特征预处理器

        参数:
            continuous_features: 连续型特征列表
            categorical_features: 分类特征列表
            boolean_features: 布尔特征列表
            string_features: 字符串特征列表
            missing_threshold: 缺失值阈值，超过此比例将删除特征
            chi2_k: 卡方检验保留的特征数量
        """
        self.continuous_features = continuous_features or []
        self.categorical_features = categorical_features or []
        self.boolean_features = boolean_features or []
        self.string_features = string_features or []

        # 合并所有类别型特征
        self.all_categorical = self.categorical_features + self.boolean_features + self.string_features

        # 配置参数
        self.missing_threshold = missing_threshold
        self.chi2_k = chi2_k

        # 存储预处理模型
        self.scalers = {}  # 连续特征标准化器
        self.encoders = {}  # 分类特征编码器
        self.selected_features = None  # 最终选择的特征
        self.feature_importance = None  # 特征重要性分数

    def fit(self, df: pd.DataFrame, target: np.ndarray) -> 'FeaturePreprocessor':
        """拟合预处理模型"""
        # 1. 处理缺失值
        self._handle_missing_values(df)

        # 2. 编码分类特征
        self._encode_categorical_features(df)

        # 3. 标准化连续特征
        self._standardize_continuous_features(df)

        # 4. 特征选择
        self._select_features(df, target)

        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """转换数据"""
        # 复制数据避免修改原始数据
        df = df.copy()

        # 1. 处理缺失值
        for col in self.continuous_features:
            if col in df.columns:
                df[col].fillna(self.continuous_impute_values[col], inplace=True)

        for col in self.all_categorical:
            if col in df.columns:
                df[col].fillna("Unknown", inplace=True)

        # 2. 编码分类特征
        for col in self.all_categorical:
            if col in df.columns:
                df[col] = df[col].map(lambda x: self.encoders[col].get(x, -1))

        # 3. 标准化连续特征
        for col in self.continuous_features:
            if col in df.columns:
                df[col] = self.scalers[col].transform(df[col].values.reshape(-1, 1)).flatten()

        # 4. 提取最终特征
        X = np.hstack([
            df[col].values.reshape(-1, 1)
            for col in self.selected_features
            if col in df.columns
        ])

        return X

    def fit_transform(self, df: pd.DataFrame, target: np.ndarray) -> np.ndarray:
        """拟合并转换数据"""
        self.fit(df, target)
        return self.transform(df)

    def _handle_missing_values(self, df: pd.DataFrame) -> None:
        """处理缺失值"""
        # 计算缺失率
        missing_ratio = df.isnull().sum() / len(df)

        # 删除缺失率过高的特征
        features_to_drop = []
        for col, ratio in missing_ratio.items():
            if ratio > self.missing_threshold:
                features_to_drop.append(col)
                if col in self.continuous_features:
                    self.continuous_features.remove(col)
                if col in self.all_categorical:
                    self.all_categorical.remove(col)

        if features_to_drop:
            print(f"删除缺失率过高的特征: {features_to_drop}")
            df.drop(features_to_drop, axis=1, inplace=True)

        # 连续特征用众数填充
        self.continuous_impute_values = {}
        for col in self.continuous_features:
            if col in df.columns:
                self.continuous_impute_values[col] = df[col].mode()[0]
                df[col].fillna(self.continuous_impute_values[col], inplace=True)

        # 分类特征用"Unknown"填充
        for col in self.all_categorical:
            if col in df.columns:
                df[col].fillna("Unknown", inplace=True)

    def _encode_categorical_features(self, df: pd.DataFrame) -> None:
        """编码分类特征"""
        for col in self.all_categorical:
            if col in df.columns:
                # 获取所有可能的值（包括缺失值标记）
                unique_values = df[col].unique()
                # 创建值到索引的映射
                self.encoders[col] = {val: i for i, val in enumerate(unique_values)}
                # 转换为索引
                df[col] = df[col].map(self.encoders[col])

    def _standardize_continuous_features(self, df: pd.DataFrame) -> None:
        """标准化连续特征"""
        for col in self.continuous_features:
            if col in df.columns:
                scaler = StandardScaler()
                df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1)).flatten()
                self.scalers[col] = scaler

    def _select_features(self, df: pd.DataFrame, target: np.ndarray) -> None:
        """特征选择"""
        # 分离连续特征和分类特征
        continuous_valid = [f for f in self.continuous_features if f in df.columns]
        categorical_valid = [f for f in self.all_categorical if f in df.columns]

        # 对连续特征使用ANOVA F-value
        if continuous_valid:
            X_continuous = np.hstack([
                df[col].values.reshape(-1, 1)
                for col in continuous_valid
            ])
            anova_selector = SelectKBest(score_func=f_classif, k=min(self.chi2_k, len(continuous_valid)))
            X_continuous_selected = anova_selector.fit_transform(X_continuous, target)
            continuous_mask = anova_selector.get_support()
            selected_continuous = [continuous_valid[i] for i in range(len(continuous_valid)) if continuous_mask[i]]
            continuous_importance = {
                continuous_valid[i]: score
                for i, score in enumerate(anova_selector.scores_)
                if continuous_mask[i]
            }
        else:
            selected_continuous = []
            continuous_importance = {}

        # 对分类特征使用卡方检验
        if categorical_valid:
            X_categorical = np.hstack([
                df[col].values.reshape(-1, 1)
                for col in categorical_valid
            ])
            # 确保卡方检验的输入非负
            chi2_selector = SelectKBest(score_func=chi2, k=min(self.chi2_k, len(categorical_valid)))
            X_categorical_selected = chi2_selector.fit_transform(X_categorical, target)
            categorical_mask = chi2_selector.get_support()
            selected_categorical = [categorical_valid[i] for i in range(len(categorical_valid)) if categorical_mask[i]]
            categorical_importance = {
                categorical_valid[i]: score
                for i, score in enumerate(chi2_selector.scores_)
                if categorical_mask[i]
            }
        else:
            selected_categorical = []
            categorical_importance = {}

        # 合并选择的特征
        self.selected_features = selected_continuous + selected_categorical
        self.feature_importance = {**continuous_importance, **categorical_importance}

        print(f"特征选择完成: 从 {len(continuous_valid) + len(categorical_valid)} 个特征中选择了 {len(self.selected_features)} 个")


class DeepFM:
    """DeepFM模型的NumPy实现"""

    def __init__(self,
                 feature_dim: int,
                 embedding_dim: int = 8,
                 hidden_units: List[int] = [128, 64],
                 learning_rate: float = 0.001,
                 epochs: int = 10,
                 batch_size: int = 256,
                 lambda_w: float = 0.01,
                 lambda_v: float = 0.01,
                 lambda_h: float = 0.01):
        """
        初始化DeepFM模型

        参数:
            feature_dim: 特征维度
            embedding_dim: 嵌入维度
            hidden_units: DNN隐藏层单元数
            learning_rate: 学习率
            epochs: 训练轮数
            batch_size: 批次大小
            lambda_w: 一阶权重正则化系数
            lambda_v: 嵌入层正则化系数
            lambda_h: DNN层正则化系数
        """
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.lambda_w = lambda_w
        self.lambda_v = lambda_v
        self.lambda_h = lambda_h

        # 模型参数
        self.w0 = 0.0  # 偏置项
        self.w = np.zeros(feature_dim)  # 一阶权重
        self.V = np.random.normal(0, 0.01, (feature_dim, embedding_dim))  # 嵌入矩阵

        # DNN参数
        self.weights = []
        self.biases = []

        # 初始化DNN权重和偏置
        prev_dim = feature_dim * embedding_dim
        for units in hidden_units:
            self.weights.append(np.random.normal(0, 0.01, (prev_dim, units)))
            self.biases.append(np.zeros(units))
            prev_dim = units

        # 输出层
        self.weights.append(np.random.normal(0, 0.01, (prev_dim, 1)))
        self.biases.append(np.zeros(1))

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid激活函数"""
        return 1.0 / (1.0 + np.exp(-x))

    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU激活函数"""
        return np.maximum(0, x)

    def _forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """前向传播"""
        # FM部分
        # 一阶项
        linear_terms = self.w0 + np.dot(X, self.w)

        # 二阶交互项
        sum_square = np.square(np.dot(X, self.V))
        square_sum = np.dot(np.square(X), np.square(self.V))
        interaction_terms = 0.5 * np.sum(sum_square - square_sum, axis=1)

        # DNN部分
        # 嵌入层输出
        embed_output = X.reshape(-1, self.feature_dim, 1) * self.V
        embed_flatten = embed_output.reshape(-1, self.feature_dim * self.embedding_dim)

        # 存储每一层的输出用于反向传播
        layer_outputs = [embed_flatten]

        # 前向传播通过DNN
        dnn_output = embed_flatten
        for i in range(len(self.weights)):
            dnn_output = np.dot(dnn_output, self.weights[i]) + self.biases[i]
            layer_outputs.append(dnn_output)
            if i < len(self.weights) - 1:  # 最后一层不使用激活函数
                dnn_output = self._relu(dnn_output)
                layer_outputs[-1] = dnn_output  # 更新激活后的输出

        # 合并FM和DNN输出
        y_pred = linear_terms + interaction_terms + dnn_output.flatten()
        y_prob = self._sigmoid(y_pred)

        return y_prob, embed_flatten, layer_outputs

    def _backward(self, X: np.ndarray, y_true: np.ndarray, y_prob: np.ndarray,
                  embed_flatten: np.ndarray, layer_outputs: List[np.ndarray]) -> None:
        """反向传播"""
        # 计算梯度
        m = X.shape[0]
        error = y_prob - y_true

        # FM部分梯度
        grad_w0 = np.sum(error) / m
        grad_w = np.dot(X.T, error) / m + self.lambda_w * self.w

        # 二阶交互项梯度
        grad_V = np.zeros_like(self.V)
        for i in range(m):
            x = X[i]
            xv = X[i].reshape(-1, 1) * self.V  # 计算嵌入值
            sum_v = np.sum(xv, axis=0)
            for f in range(self.feature_dim):
                if x[f] != 0:
                    grad_V[f] += x[f] * (sum_v - xv[f]) * error[i]

        grad_V = grad_V / m + self.lambda_v * self.V

        # DNN部分梯度
        delta = error.reshape(-1, 1)

        for i in reversed(range(len(self.weights))):
            if i < len(self.weights) - 1:  # 非输出层
                # 计算ReLU导数
                relu_grad = (layer_outputs[i+1] > 0).astype(float)
                delta = delta * relu_grad

            # 计算权重和偏置梯度
            prev_output = layer_outputs[i]
            grad_w_dnn = np.dot(prev_output.T, delta) / m + self.lambda_h * self.weights[i]
            grad_b_dnn = np.sum(delta, axis=0) / m

            # 更新梯度
            delta = np.dot(delta, self.weights[i].T)

            # 更新权重和偏置
            self.weights[i] -= self.learning_rate * grad_w_dnn
            self.biases[i] -= self.learning_rate * grad_b_dnn

        # 更新FM参数
        self.w0 -= self.learning_rate * grad_w0
        self.w -= self.learning_rate * grad_w
        self.V -= self.learning_rate * grad_V

    def fit(self, X: np.ndarray, y: np.ndarray, val_data: Tuple[np.ndarray, np.ndarray] = None) -> None:
        """训练模型"""
        for epoch in range(self.epochs):
            # 生成批次
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)

            for start_idx in range(0, X.shape[0], self.batch_size):
                end_idx = min(start_idx + self.batch_size, X.shape[0])
                batch_indices = indices[start_idx:end_idx]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]

                # 前向传播
                y_prob, embed_flatten, layer_outputs = self._forward(X_batch)

                # 反向传播
                self._backward(X_batch, y_batch, y_prob, embed_flatten, layer_outputs)

            # 计算训练集损失
            y_train_prob, _, _ = self._forward(X)
            train_loss = -np.mean(y * np.log(y_train_prob + 1e-10) +
                                  (1 - y) * np.log(1 - y_train_prob + 1e-10))

            # 计算验证集损失（如果有）
            if val_data is not None:
                X_val, y_val = val_data
                y_val_prob, _, _ = self._forward(X_val)
                val_loss = -np.mean(y_val * np.log(y_val_prob + 1e-10) +
                                    (1 - y_val) * np.log(1 - y_val_prob + 1e-10))
                print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        y_prob, _, _ = self._forward(X)
        return y_prob

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """预测类别"""
        y_prob = self.predict_proba(X)
        return (y_prob >= threshold).astype(int)

# 示例用法
def main():
    # 生成示例数据
    np.random.seed(42)
    n_samples = 10000
    n_features = 50

    # 创建示例DataFrame
    data = {
        f'num_{i}': np.random.normal(0, 1, n_samples)
        for i in range(10)  # 10个连续型特征
    }

    # 10个分类特征
    for i in range(10):
        data[f'cat_{i}'] = np.random.choice(['A', 'B', 'C', 'D', 'E', 'Unknown'], n_samples)

    # 5个布尔特征
    for i in range(5):
        data[f'bool_{i}'] = np.random.choice([True, False, None], n_samples)

    # 5个字符串特征
    for i in range(5):
        data[f'str_{i}'] = np.random.choice(['apple', 'banana', 'cherry', 'date', 'elderberry', 'Unknown'], n_samples)

    # 添加一些缺失值
    for col in data:
        mask = np.random.random(n_samples) < 0.1  # 10%的缺失率
        data[col][mask] = np.nan





    # 创建目标变量
    X = pd.DataFrame(data)
    y = np.random.binomial(1, 0.5, n_samples)

    #输出一共有多少列
    print(f"Number of features: {len(X.columns)}")

    # 定义特征类型
    # 定义特征类型
    continuous_features = [f'num_{i}' for i in range(10)]  # 修正这里
    categorical_features = [f'cat_{i}' for i in range(10)]  # 修正这里
    boolean_features = [f'bool_{i}' for i in range(5)]     # 修正这里
    string_features = [f'str_{i}' for i in range(5)]       # 修正这里

    #输出每个特征的统计
    # for column in X.columns:
    #     print(f"列 {column} 的数值出现次数（包含空值）：")
    #     value_counts = X[column].value_counts(dropna=False)
    #     print(value_counts)
    #     print("\n")




    # 预处理数据
    preprocessor = FeaturePreprocessor(
        continuous_features=continuous_features,
        categorical_features=categorical_features,
        boolean_features=boolean_features,
        string_features=string_features,
        missing_threshold=0.3,
        chi2_k=40  # 选择40个特征
    )

    X_processed = preprocessor.fit_transform(X, y)

    rows, columns = X_processed.shape
    print(f"DataFrame有 {rows} 行")

    # 划分训练集和测试集
    train_size = int(0.8 * len(X))
    X_train, X_test = X_processed[:train_size], X_processed[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # 初始化并训练DeepFM模型
    feature_dim = X_processed.shape[1]
    model = DeepFM(
        feature_dim=feature_dim,
        embedding_dim=8,
        hidden_units=[128, 64],
        learning_rate=0.001,
        epochs=10,
        batch_size=256
    )

    model.fit(X_train, y_train, val_data=(X_test, y_test))

    # 评估模型
    y_pred_prob = model.predict_proba(X_test)
    y_pred = model.predict(X_test)

    # 计算准确率和AUC
    accuracy = np.mean(y_pred == y_test)
    print(f"Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
