import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, f1_score, roc_auc_score,
                             precision_recall_curve, auc)
import pickle
import time

class FeatureBinner:
    """特征分箱处理器，支持配置分箱参数并保存/加载"""
    def __init__(self):
        self.binners = {}  # 存储每个特征的分箱器
        self.bin_params = {}  # 存储分箱参数

    def fit(self, X: pd.DataFrame, config: dict) -> None:
        """拟合分箱器"""
        for feature, params in config.items():
            if feature not in X.columns:
                continue

            n_bins = params.get('n_bins', 5)
            strategy = params.get('strategy', 'quantile')

            binner = KBinsDiscretizer(
                n_bins=n_bins,
                strategy=strategy,
                encode='ordinal',
                subsample=None
            )

            binner.fit(X[[feature]])
            self.binners[feature] = binner
            self.bin_params[feature] = {
                'n_bins': n_bins,
                'strategy': strategy,
                'bin_edges': binner.bin_edges_[0].tolist()
            }

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """应用分箱转换"""
        X_copy = X.copy()
        for feature, binner in self.binners.items():
            if feature in X_copy.columns:
                binned_data = binner.transform(X_copy[[feature]]).flatten()
                X_copy[feature] = binned_data
        return X_copy

    def fit_transform(self, X: pd.DataFrame, config: dict) -> pd.DataFrame:
        """拟合并转换数据"""
        self.fit(X, config)
        return self.transform(X)

    def save(self, path: str) -> None:
        """保存分箱参数"""
        with open(path, "wb") as f:
            pickle.dump(self.bin_params, f)

    def load(self, path: str) -> None:
        """加载分箱参数并重建分箱器"""
        with open(path, "rb") as f:
            self.bin_params = pickle.load(f)

        for feature, params in self.bin_params.items():
            binner = KBinsDiscretizer(
                n_bins=params['n_bins'],
                strategy=params['strategy'],
                encode='ordinal',
                subsample=None
            )
            binner.bin_edges_ = [np.array(params['bin_edges'])]
            self.binners[feature] = binner


class DeepFMDataset(Dataset):
    def __init__(self, df, numeric_features, categorical_features, label_col=None,
                 encoders=None, binner=None, outlier_bounds=None, zscore_stats=None,
                 is_train=True, config=None):
        # 确保输入是DataFrame
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"预期DataFrame类型，实际得到{type(df)}")

        self.config = config if config is not None else {}
        self.df = df.copy().reset_index(drop=True)
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.label_col = label_col
        self.is_train = is_train
        self.binner = binner
        self.outlier_bounds = outlier_bounds
        self.zscore_stats = zscore_stats

        # 确保所有需要的特征都存在
        required_features = numeric_features + categorical_features
        if label_col:
            required_features += [label_col]
        missing_features = [f for f in required_features if f not in self.df.columns]
        if missing_features:
            raise ValueError(f"数据中缺少必要特征: {missing_features}")

        # 预处理流程
        self._handle_missing_values()
        self._handle_outliers()
        self._zscore_normalize()
        self._handle_binning()
        self._encode_categorical(encoders)

        # 准备模型输入数据
        self.numeric_data = self.df[self.numeric_features].values.astype(np.float32)
        self.categorical_data = self.df[[f"{f}_encoded" for f in categorical_features]].values.astype(np.int64)

        if label_col and label_col in self.df.columns:
            self.labels = self.df[label_col].values.astype(np.float32)
        else:
            self.labels = None

    def _handle_missing_values(self):
        """处理缺失值"""
        # 数值特征用中位数填充
        for col in self.numeric_features:
            if self.df[col].isnull().any():
                fill_value = self.df[col].median()
                self.df[col].fillna(fill_value, inplace=True)

        # 类别特征用特定值填充
        cat_fill_value = self.config.get("categorical_fill_value", "Missing")
        for col in self.categorical_features:
            if self.df[col].isnull().any():
                self.df[col].fillna(cat_fill_value, inplace=True)

    def _handle_outliers(self):
        """处理异常值"""
        outlier_cfg = self.config.get("outlier_handling", {})
        if not outlier_cfg.get("enabled", False) or not self.numeric_features:
            return

        if self.is_train:
            # 训练时计算异常值边界
            self.outlier_bounds = {}
            iqr_multiplier = outlier_cfg.get("iqr_params", {}).get("iqr_multiplier", 1.5)
            lower_quantile = outlier_cfg.get("iqr_params", {}).get("lower_quantile", 0.25)
            upper_quantile = outlier_cfg.get("iqr_params", {}).get("upper_quantile", 0.75)

            for feature in self.numeric_features:
                q1 = self.df[feature].quantile(lower_quantile)
                q3 = self.df[feature].quantile(upper_quantile)
                iqr = q3 - q1
                lower_bound = q1 - iqr_multiplier * iqr
                upper_bound = q3 + iqr_multiplier * iqr

                self.outlier_bounds[feature] = {
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound
                }

                # 处理异常值
                upper_strategy = outlier_cfg.get("iqr_params", {}).get("upper_strategy", "clip")
                lower_strategy = outlier_cfg.get("iqr_params", {}).get("lower_strategy", "clip")

                if upper_strategy == "clip":
                    self.df[feature] = self.df[feature].clip(upper=upper_bound)
                if lower_strategy == "clip":
                    self.df[feature] = self.df[feature].clip(lower=lower_bound)
        else:
            # 推理时使用预计算的边界
            if self.outlier_bounds:
                upper_strategy = outlier_cfg.get("iqr_params", {}).get("upper_strategy", "clip")
                lower_strategy = outlier_cfg.get("iqr_params", {}).get("lower_strategy", "clip")

                for feature, bounds in self.outlier_bounds.items():
                    if upper_strategy == "clip":
                        self.df[feature] = self.df[feature].clip(upper=bounds["upper_bound"])
                    if lower_strategy == "clip":
                        self.df[feature] = self.df[feature].clip(lower=bounds["lower_bound"])

    def _zscore_normalize(self):
        """标准化数值特征"""
        zscore_cfg = self.config.get("zscore_normalization", {})
        if not zscore_cfg.get("enabled", False) or not self.numeric_features:
            return

        if self.is_train:
            # 训练时计算均值和标准差
            self.zscore_stats = {}
            for feature in self.numeric_features:
                mean = self.df[feature].mean()
                std = self.df[feature].std()
                self.zscore_stats[feature] = {"mean": mean, "std": std}
                # 避免除以零
                if std > 0:
                    self.df[feature] = (self.df[feature] - mean) / std
        else:
            # 推理时使用预计算的均值和标准差
            if self.zscore_stats:
                for feature, stats in self.zscore_stats.items():
                    if stats["std"] > 0:
                        self.df[feature] = (self.df[feature] - stats["mean"]) / stats["std"]

    def _handle_binning(self):
        """处理数值特征分箱"""
        if self.binner:
            if self.is_train:
                bin_config = self.config.get("bin_config", {})
                self.df[self.numeric_features] = self.binner.fit_transform(
                    self.df[self.numeric_features], bin_config)
            else:
                self.df[self.numeric_features] = self.binner.transform(
                    self.df[self.numeric_features])

    def _encode_categorical(self, encoders):
        """编码分类特征"""
        self.categorical_encoders = {} if encoders is None else encoders
        self.categorical_dims = {}

        for feature in self.categorical_features:
            if self.is_train:
                encoder = LabelEncoder()
                # 确保特征是字符串类型
                if self.df[feature].dtype != 'object':
                    self.df[feature] = self.df[feature].astype(str)

                # 确保缺失值填充值被包含在训练编码器中
                missing_value = self.config.get("categorical_fill_value", "Missing")
                if missing_value not in self.df[feature].unique():
                    # 临时添加一行包含缺失值，确保编码器能识别它
                    temp_df = self.df.copy()
                    temp_row = pd.DataFrame({feature: [missing_value]})
                    temp_df = pd.concat([temp_df, temp_row], ignore_index=True)
                    encoded = encoder.fit_transform(temp_df[feature])
                    encoded = encoded[:-1]  # 移除临时添加的行的编码结果
                else:
                    encoded = encoder.fit_transform(self.df[feature])

                self.categorical_encoders[feature] = encoder
                self.categorical_dims[feature] = len(encoder.classes_)
            else:
                if feature not in self.categorical_encoders:
                    raise ValueError(f"编码器中缺少特征 {feature} 的编码信息")

                encoder = self.categorical_encoders[feature]
                missing_value = self.config.get("categorical_fill_value", "Missing")

                # 检查填充值是否在编码器中
                if missing_value not in encoder.classes_:
                    raise ValueError(
                        f"编码器未包含缺失值填充值 '{missing_value}' 的编码信息，"
                        f"请重新训练模型以包含此值"
                    )

                # 处理预测数据中可能出现的未知类别
                mask = ~self.df[feature].isin(encoder.classes_)
                if mask.any():
                    self.df.loc[mask, feature] = missing_value

                encoded = encoder.transform(self.df[feature])
                self.categorical_dims[feature] = len(encoder.classes_)

            self.df[f"{feature}_encoded"] = encoded

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        numeric = torch.tensor(self.numeric_data[idx], dtype=torch.float32)
        categorical = torch.tensor(self.categorical_data[idx], dtype=torch.long)

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return numeric, categorical, label
        return numeric, categorical


class DeepFM(nn.Module):
    def __init__(self, numeric_dim, categorical_features, categorical_dims,
                 embed_dim=32, mlp_layers=[256, 128, 64], dropout=0.3):
        super(DeepFM, self).__init__()
        self.numeric_dim = numeric_dim
        self.embed_dim = embed_dim

        # FM部分 - 一阶特征权重
        self.first_order_numeric = nn.Linear(numeric_dim, 1)
        self.first_order_categorical = nn.ModuleList([
            nn.Embedding(categorical_dims[feat], 1)
            for feat in categorical_features
        ])

        # FM部分 - 二阶特征嵌入
        self.embeddings = nn.ModuleList([
            nn.Embedding(categorical_dims[feat], embed_dim)
            for feat in categorical_features
        ])

        # DNN部分
        input_dnn_dim = numeric_dim + len(categorical_features) * embed_dim
        dnn_layers = []
        for dim in mlp_layers:
            dnn_layers.append(nn.Linear(input_dnn_dim, dim))
            dnn_layers.append(nn.BatchNorm1d(dim))
            dnn_layers.append(nn.ReLU())
            dnn_layers.append(nn.Dropout(dropout))
            input_dnn_dim = dim

        self.dnn = nn.Sequential(*dnn_layers)

        # 输出层
        self.output_layer = nn.Linear(mlp_layers[-1] + 1, 1)  # 1来自FM的输出
        self.sigmoid = nn.Sigmoid()

    def forward(self, numeric, categorical):
        # FM一阶部分
        first_order = self.first_order_numeric(numeric)
        for i, emb in enumerate(self.first_order_categorical):
            first_order += emb(categorical[:, i])

        # FM二阶部分
        embeddings = []
        for i, emb in enumerate(self.embeddings):
            embeddings.append(emb(categorical[:, i]))  # (batch_size, embed_dim)

        embeddings = torch.stack(embeddings, dim=1)  # (batch_size, num_categorical, embed_dim)
        sum_square = torch.sum(embeddings, dim=1) ** 2  # (batch_size, embed_dim)
        square_sum = torch.sum(embeddings ** 2, dim=1)  # (batch_size, embed_dim)
        fm_second_order = 0.5 * torch.sum(sum_square - square_sum, dim=1, keepdim=True)  # (batch_size, 1)

        # FM总输出
        fm_output = first_order + fm_second_order

        # DNN部分
        flat_embeddings = embeddings.view(embeddings.size(0), -1)  # (batch_size, num_categorical*embed_dim)
        dnn_input = torch.cat([numeric, flat_embeddings], dim=1)  # 拼接数值特征和嵌入特征
        dnn_output = self.dnn(dnn_input)  # (batch_size, mlp_layers[-1])

        # 联合输出
        combined = torch.cat([fm_output, dnn_output], dim=1)  # (batch_size, 1 + mlp_layers[-1])
        output = self.output_layer(combined)
        output = self.sigmoid(output)

        return output


class DeepFMTrainer:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=CONFIG["learning_rate"],
            weight_decay=CONFIG["weight_decay"]
        )

        # 训练记录
        self.train_loss_history = []
        self.train_metrics_history = []
        self.val_metrics_history = []
        self.best_auc = 0.0
        self.best_model_state = None
        self.best_threshold = 0.5
        self.test_confusion_matrix = None

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for numeric, categorical, labels in dataloader:
            numeric = numeric.to(self.device)
            categorical = categorical.to(self.device)
            labels = labels.to(self.device).unsqueeze(1)

            self.optimizer.zero_grad()
            outputs = self.model(numeric, categorical)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * numeric.size(0)
            all_preds.extend(outputs.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

        avg_loss = total_loss / len(dataloader.dataset)
        metrics = calculate_detailed_metrics(all_labels, all_preds, self.best_threshold)
        return avg_loss, metrics

    def evaluate(self, dataloader, threshold=0.5):
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                # 处理有标签和无标签数据
                if len(batch) == 3:
                    numeric, categorical, labels = batch
                else:
                    numeric, categorical = batch
                    labels = None

                numeric = numeric.to(self.device)
                categorical = categorical.to(self.device)

                outputs = self.model(numeric, categorical)

                if labels is not None:
                    labels = labels.to(self.device).unsqueeze(1)
                    all_preds.extend(outputs.detach().cpu().numpy())
                    all_labels.extend(labels.detach().cpu().numpy())

        metrics = calculate_detailed_metrics(all_labels, all_preds, threshold)
        return metrics

    def find_best_threshold(self, dataloader):
        """在验证集上寻找最佳阈值"""
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 3:
                    numeric, categorical, labels = batch
                else:
                    numeric, categorical = batch
                    labels = None

                numeric = numeric.to(self.device)
                categorical = categorical.to(self.device)
                outputs = self.model(numeric, categorical)

                if labels is not None:
                    labels = labels.to(self.device).unsqueeze(1)
                    all_preds.extend(outputs.detach().cpu().numpy())
                    all_labels.extend(labels.detach().cpu().numpy())

        all_preds = np.array(all_preds).flatten()
        all_labels = np.array(all_labels).flatten()

        best_score = 0.0
        best_threshold = 0.5

        # 搜索最佳阈值
        for threshold in np.arange(0.1, 0.91, 0.01):
            y_pred = (all_preds >= threshold).astype(int)
            if CONFIG.get("optimize_for", "") == "f1":
                score = f1_score(all_labels, y_pred)
            else:  # 默认使用PR-AUC
                precision, recall, _ = precision_recall_curve(all_labels, all_preds)
                score = auc(recall, precision)

            if score > best_score:
                best_score = score
                best_threshold = threshold

        return best_threshold

    def train(self, train_loader, val_loader, epochs=15):
        for epoch in range(epochs):
            start_time = time.time()

            # 训练
            train_loss, train_metrics = self.train_epoch(train_loader)
            self.train_loss_history.append(train_loss)
            self.train_metrics_history.append(train_metrics)

            # 验证
            val_metrics = self.evaluate(val_loader, self.best_threshold)
            self.val_metrics_history.append(val_metrics)

            # 打印 epoch 结果
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"训练损失: {train_loss:.4f} | 训练F1: {train_metrics['overall']['f1_score']:.4f}")
            print(f"验证F1: {val_metrics['overall']['f1_score']:.4f} | 验证AUC-PR: {val_metrics['overall']['auc_pr']:.4f}")
            print(f"耗时: {time.time() - start_time:.2f}秒\n")

            # 更新最佳模型
            if val_metrics['overall']['auc_pr'] > self.best_auc:
                self.best_auc = val_metrics['overall']['auc_pr']
                self.best_model_state = self.model.state_dict()
                # 如果启用阈值调优，在最佳模型上重新计算阈值
                if CONFIG.get("threshold_tuning", False):
                    self.best_threshold = self.find_best_threshold(val_loader)

        # 加载最佳模型
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)

    def predict(self, dataloader):
        """对数据进行预测，兼容有标签和无标签数据"""
        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for batch in dataloader:
                # 处理有标签和无标签数据
                if len(batch) == 3:
                    numeric, categorical, _ = batch  # 忽略标签
                else:
                    numeric, categorical = batch

                numeric = numeric.to(self.device)
                categorical = categorical.to(self.device)

                outputs = self.model(numeric, categorical)
                all_preds.extend(outputs.detach().cpu().numpy())

        return np.array(all_preds).flatten()


# 辅助函数
def calculate_detailed_metrics(y_true, y_pred, threshold=0.5):
    """计算详细的评估指标"""
    y_true = np.array(y_true).flatten()
    y_pred_proba = np.array(y_pred).flatten()
    y_pred_binary = (y_pred_proba >= threshold).astype(int)

    # 计算各种指标
    f1 = f1_score(y_true, y_pred_binary)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    auc_pr = auc(recall, precision)

    return {
        'overall': {
            'f1_score': f1,
            'roc_auc': roc_auc,
            'auc_pr': auc_pr
        }
    }