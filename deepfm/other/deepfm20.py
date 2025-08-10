import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score,
                             precision_score, recall_score, confusion_matrix,
                             precision_recall_curve, auc)
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime
import json
import pickle
import warnings
from typing import List, Optional, Literal, Tuple, Dict
import argparse

warnings.filterwarnings('ignore')

# 配置参数
CONFIG = {
    "test_size": 0.2,
    "val_size": 0.2,
    "batch_size": 256,
    "epochs": 5,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "embed_dim": 32,
    "mlp_layers": [256, 128, 64],
    "dropout": 0.3,
    "numeric_features": ['age', 'lim_use_rate'],
    "categorical_features": ['gender_cd'],
    "label_col": "label",
    "bin_config": {
        'age': {'n_bins': 20, 'strategy': 'quantile'}
    },
    "features_list": ['A_flow_id', 'user_id', 'unique_id', 'inviite_time', 'create_tm', 'label',
                      'is_coupon_used', 'valid_begin_date', 'valid_end_date', 'push_status',
                      'delay_type', 'delay_days', 'push_cnt', 'sms_chharge_cnt', 'tel_call_type',
                      'tel_call_dur', 'touch_status', 'click_status', 'is_coupon_issue', 'is_credit',
                      'first_biz_name', 'is_apply', 'is_wdraw', 'wdraw_cnt', 'wdraw_amt', 'wdraw_amt_t0',
                      'wdraw_amt_t3', 'wdraw_amt_t7', 'wdraw_amt_t10', 'wdraw_amt_t14', 'wdraw_amt_t15',
                      'wdraw_amt_t30', 'second_biz_name', 'third_biz_name', 'tel_call_inav_cnt', 'plan_id',
                      'subscribe_no', 't20_cust_id', 'db_src', 'cust_date', 'prod_cd', 'channel_task_id',
                      'plan_name', 'cust_gp_code', 'cust_gp_name', 'touch_name', 'touch_id', 'touuch_type',
                      'channel_name', 'channel_id', 'apply_amt', 'apply_Ent', 'aprv_status', 'cust_recv_time',
                      'decision_id', 'decision_namme', 'touch_time', 't19_user_status', 'user_mobile_status',
                      'is_realname_cert', 'is_mobile_realname_cert', 'phy_del_op_side', 't2_cust_id',
                      'user_name', 't19_mobile_no', 't19_id_no'],
    # 新增：异常值处理配置
    "outlier_handling": {
        "enabled": True,
        "features": ['age', 'lim_use_rate'],
        "iqr_params": {
            "iqr_multiplier": 1.5,
            "lower_quantile": 0.25,
            "upper_quantile": 0.75,
            "upper_strategy": "cap",
            "lower_strategy": "cap",
            "new_column_suffix": None
        }
    },
    # 新增：标准化配置
    "zscore_normalization": {
        "enabled": True,
        "features": ['age', 'lim_use_rate']
    },
    # 新增：预处理参数保存路径
    "preprocess_params": {
        "outlier_bounds_path": "outlier_bounds.pkl",
        "zscore_stats_path": "zscore_stats.pkl",
        "encoders_path": "encoders.pkl",
        "binner_path": "binner.pkl"
    }
}


def handle_outliers_by_iqr(
        df: pd.DataFrame,
        features: List[str],
        upper_strategy: Literal["cap", "keep"] = "cap",
        lower_strategy: Literal["cap", "keep"] = "cap",
        new_column_suffix: Optional[str] = "_handled",
        inplace: bool = False,
        iqr_multiplier: float = 1.5,
        lower_quantile: float = 0.25,
        upper_quantile: float = 0.75
) -> Tuple[pd.DataFrame, List[str], Dict[str, Dict[str, float]]]:
    if not (0 < lower_quantile < upper_quantile < 1):
        raise ValueError("分位数参数必须满足 0 < lower_quantile < upper_quantile < 1")

    if not inplace or new_column_suffix is not None:
        df = df.copy()

    processed_columns = []
    fill_values = {}

    for feat in features:
        if feat not in df.columns:
            print(f"警告: 特征 {feat} 不存在，已跳过")
            continue

        q_low = df[feat].quantile(lower_quantile)
        q_high = df[feat].quantile(upper_quantile)
        iqr = q_high - q_low
        lower_bound = q_low - iqr_multiplier * iqr
        upper_bound = q_high + iqr_multiplier * iqr

        fill_values[feat] = {
            "lower_bound": lower_bound,
            "upper_bound": upper_bound
        }

        target_col = feat if new_column_suffix is None else f"{feat}{new_column_suffix}"
        if new_column_suffix is not None and target_col not in df.columns:
            df[target_col] = df[feat].copy()

        is_outlier_lower = df[feat] < lower_bound
        is_outlier_upper = df[feat] > upper_bound
        lower_count = is_outlier_lower.sum()
        upper_count = is_outlier_upper.sum()

        if lower_strategy == "cap" and lower_count > 0:
            df.loc[is_outlier_lower, target_col] = lower_bound
        if upper_strategy == "cap" and upper_count > 0:
            df.loc[is_outlier_upper, target_col] = upper_bound

        processed_columns.append(target_col)
        print(f"特征 {feat}: 处理列={target_col}, 下界异常值{lower_count}个, 上界异常值{upper_count}个")

    processed_columns = list(dict.fromkeys(processed_columns))
    return df, processed_columns, fill_values


def handle_outliers_by_bounds(
        df: pd.DataFrame,
        bounds: Dict[str, Tuple[float, float]],
        upper_strategy: Literal["cap", "keep"] = "cap",
        lower_strategy: Literal["cap", "keep"] = "cap",
        new_column_suffix: Optional[str] = "_handled",
        inplace: bool = False
) -> Tuple[pd.DataFrame, List[str], Dict[str, Dict[str, float]]]:
    for feat, (lower, upper) in bounds.items():
        if lower >= upper:
            raise ValueError(f"特征 {feat} 的下界({lower})必须小于上界({upper})")

    if not inplace or new_column_suffix is not None:
        df = df.copy()

    processed_columns = []
    fill_values = {}

    for feat, (lower_bound, upper_bound) in bounds.items():
        if feat not in df.columns:
            print(f"警告: 特征 {feat} 不存在，已跳过")
            continue

        fill_values[feat] = {
            "lower_bound": lower_bound,
            "upper_bound": upper_bound
        }

        target_col = feat if new_column_suffix is None else f"{feat}{new_column_suffix}"
        if new_column_suffix is not None and target_col not in df.columns:
            df[target_col] = df[feat].copy()

        is_outlier_lower = df[feat] < lower_bound
        is_outlier_upper = df[feat] > upper_bound
        lower_count = is_outlier_lower.sum()
        upper_count = is_outlier_upper.sum()

        if lower_strategy == "cap" and lower_count > 0:
            df.loc[is_outlier_lower, target_col] = lower_bound
        if upper_strategy == "cap" and upper_count > 0:
            df.loc[is_outlier_upper, target_col] = upper_bound

        processed_columns.append(target_col)
        print(f"特征 {feat}: 处理列={target_col}, 下界异常值{lower_count}个, 上界异常值{upper_count}个")

    processed_columns = list(dict.fromkeys(processed_columns))
    return df, processed_columns, fill_values


def zscore_normalize(df: pd.DataFrame, features: List[str], inplace: bool = False,
                     stats: Optional[dict] = None) -> (pd.DataFrame, dict):
    if not inplace:
        df = df.copy()

    zscore_stats = {}

    for feat in features:
        if feat not in df.columns:
            print(f"警告: 特征 {feat} 不在DataFrame中，已跳过")
            continue

        if stats is not None and feat in stats:
            mean, std = stats[feat]
        else:
            mean = df[feat].mean(skipna=True)
            std = df[feat].std(skipna=True)

            if std < 1e-10:
                std = 1e-10
                print(f"警告: 特征 {feat} 的标准差为0，已替换为{std}")

            zscore_stats[feat] = (mean, std)

        df[feat] = (df[feat] - mean) / std

    return_df = df if inplace else df
    return_stats = zscore_stats if stats is None else stats

    return return_df, return_stats


def add_days_from_today(df, date_column, new_column=None, handle_nulls=True):
    if new_column is None:
        new_column = f"{date_column}_day_diff"

    if date_column not in df.columns:
        raise ValueError(f"列 {date_column} 不存在于DataFrame中")

    today = pd.Timestamp.today().normalize()

    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        df = df.assign(**{
            date_column: pd.to_datetime(df[date_column], errors='coerce', format='mixed')
        })

    invalid_mask = df[date_column].isna()
    invalid_count = invalid_mask.sum()

    if invalid_count > 0:
        print(f"警告：有{invalid_count}个日期值无法转换为有效的日期格式")

        if handle_nulls:
            df.loc[invalid_mask, date_column] = today

    df[new_column] = (today - df[date_column]).dt.days.astype('int32')

    if new_column not in CONFIG['numeric_features']:
        CONFIG['numeric_features'].append(new_column)
        print(f"已将新列 {new_column} 添加到CONFIG['numeric_features']")

    return df


def extract_user_features(df, user_id_col='user_id', label_col='label'):
    if user_id_col not in df.columns:
        raise ValueError(f"用户ID列 {user_id_col} 不存在于DataFrame中")
    if label_col not in df.columns:
        raise ValueError(f"标签列 {label_col} 不存在于DataFrame中")

    df[label_col] = pd.to_numeric(
        df[label_col],
        errors='coerce',
        downcast='integer'
    )

    def process_group(group):
        labels = group[label_col]
        mask_1 = (labels == 1)
        if mask_1.any():
            return mask_1.idxmax()
        mask_0 = (labels == 0)
        if mask_0.any():
            return mask_0.idxmax()
        return group.index[0]

    keep_indices = df.groupby(
        user_id_col,
        group_keys=False,
        observed=True
    ).apply(process_group)

    return df.loc[keep_indices]


def load_data_from_folder(folder_path):
    """加载文件夹中所有CSV文件并合并为一个DataFrame"""
    all_files = []

    # 遍历文件夹中的所有文件
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            file_path = os.path.join(folder_path, file)
            try:
                df = pd.read_csv(file_path)
                # 检查并添加features_list（原代码中缺少定义，这里补充默认处理）
                if 'features_list' in CONFIG:
                    if len(df.columns) == len(CONFIG['features_list']):
                        df.columns = CONFIG['features_list']
                all_files.append(df)
                print(f"已加载 {file}，包含 {len(df)} 行数据")
            except Exception as e:
                print(f"加载 {file} 时出错: {str(e)}")

    if not all_files:
        raise ValueError(f"在 {folder_path} 中未找到CSV文件")

    # 合并所有DataFrame
    combined_df = pd.concat(all_files, axis=0, ignore_index=True)
    print(f"\n合并后的数据集大小: {len(combined_df)} 行")
    return combined_df


class FeatureBinner:
    def __init__(self):
        self.binners = {}
        self.bin_params = {}

    def fit(self, X: pd.DataFrame, config: dict) -> None:
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
        X_copy = X.copy()
        for feature, binner in self.binners.items():
            if feature in X_copy.columns:
                binned_data = binner.transform(X_copy[[feature]]).flatten()
                X_copy[feature] = binned_data
        return X_copy

    def fit_transform(self, X: pd.DataFrame, config: dict) -> pd.DataFrame:
        self.fit(X, config)
        return self.transform(X)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.bin_params, f)

    def load(self, path: str) -> None:
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


class DeepFM(nn.Module):
    def __init__(self, cat_feature_sizes, num_numeric_features, embed_dim, mlp_layers, dropout):
        super(DeepFM, self).__init__()
        self.embed_dim = embed_dim

        # 嵌入层
        self.embeddings = nn.ModuleList()
        for size in cat_feature_sizes:
            self.embeddings.append(nn.Embedding(size, embed_dim))

        # FM部分 - 一阶特征
        self.first_order_cat = nn.ModuleList()
        for size in cat_feature_sizes:
            self.first_order_cat.append(nn.Embedding(size, 1))
        self.first_order_num = nn.Linear(num_numeric_features, 1)

        # MLP部分
        input_dim = num_numeric_features + embed_dim * len(cat_feature_sizes)
        mlp_layers = [input_dim] + mlp_layers
        self.mlp = nn.Sequential()
        for i in range(len(mlp_layers) - 1):
            self.mlp.add_module(f'linear_{i}', nn.Linear(mlp_layers[i], mlp_layers[i+1]))
            self.mlp.add_module(f'bn_{i}', nn.BatchNorm1d(mlp_layers[i+1]))
            self.mlp.add_module(f'relu_{i}', nn.ReLU())
            self.mlp.add_module(f'dropout_{i}', nn.Dropout(dropout))

        # 输出层
        self.output = nn.Linear(mlp_layers[-1] + 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_cat, x_num):
        # 一阶特征
        first_order = self.first_order_num(x_num)
        for i, emb in enumerate(self.first_order_cat):
            first_order += emb(x_cat[:, i])

        # 二阶特征 (FM)
        embeddings = []
        for i, emb in enumerate(self.embeddings):
            embeddings.append(emb(x_cat[:, i]))
        embeddings = torch.stack(embeddings, dim=1)  # (batch, num_cat, embed_dim)

        sum_square = torch.sum(embeddings, dim=1) ** 2
        square_sum = torch.sum(embeddings ** 2, dim=1)
        fm_second_order = 0.5 * torch.sum(sum_square - square_sum, dim=1, keepdim=True)

        # MLP部分
        mlp_input = torch.cat([x_num, embeddings.view(-1, embeddings.shape[1] * embeddings.shape[2])], dim=1)
        mlp_out = self.mlp(mlp_input)

        # 合并输出
        total = first_order + fm_second_order + self.output(torch.cat([mlp_out, first_order], dim=1))
        return self.sigmoid(total)


class CustomDataset(Dataset):
    def __init__(self, x_cat, x_num, y):
        self.x_cat = x_cat
        self.x_num = x_num
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.x_cat[idx], dtype=torch.long),
            torch.tensor(self.x_num[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32)
        )


def preprocess_training_data(df, config):
    # 异常值处理
    outlier_cfg = config["outlier_handling"]
    outlier_bounds = None
    if outlier_cfg["enabled"] and outlier_cfg["features"]:
        df, _, outlier_bounds = handle_outliers_by_iqr(
            df=df,
            features=outlier_cfg["features"],
            upper_strategy=outlier_cfg["iqr_params"]["upper_strategy"],
            lower_strategy=outlier_cfg["iqr_params"]["lower_strategy"],
            new_column_suffix=outlier_cfg["iqr_params"]["new_column_suffix"],
            iqr_multiplier=outlier_cfg["iqr_params"]["iqr_multiplier"],
            lower_quantile=outlier_cfg["iqr_params"]["lower_quantile"],
            upper_quantile=outlier_cfg["iqr_params"]["upper_quantile"]
        )

    # 标准化
    zscore_cfg = config["zscore_normalization"]
    zscore_stats = None
    if zscore_cfg["enabled"] and zscore_cfg["features"]:
        df, zscore_stats = zscore_normalize(
            df=df,
            features=zscore_cfg["features"]
        )

    # 分箱处理
    binner = FeatureBinner()
    if config["bin_config"]:
        df = binner.fit_transform(df, config["bin_config"])

    # 类别特征编码
    encoders = {}
    for cat_feat in config["categorical_features"]:
        le = LabelEncoder()
        df[cat_feat] = le.fit_transform(df[cat_feat].astype(str))
        encoders[cat_feat] = le

    return df, encoders, binner, outlier_bounds, zscore_stats


def preprocess_inference_data(df, config, encoders, binner, outlier_bounds, zscore_stats):
    # 异常值处理
    outlier_cfg = config["outlier_handling"]
    if outlier_cfg["enabled"] and outlier_bounds and outlier_cfg["features"]:
        bounds_dict = {feat: (params["lower_bound"], params["upper_bound"])
                       for feat, params in outlier_bounds.items()}
        df, _, _ = handle_outliers_by_bounds(
            df=df,
            bounds=bounds_dict,
            upper_strategy=outlier_cfg["iqr_params"]["upper_strategy"],
            lower_strategy=outlier_cfg["iqr_params"]["lower_strategy"],
            new_column_suffix=outlier_cfg["iqr_params"]["new_column_suffix"]
        )

    # 标准化
    zscore_cfg = config["zscore_normalization"]
    if zscore_cfg["enabled"] and zscore_stats and zscore_cfg["features"]:
        df, _ = zscore_normalize(
            df=df,
            features=zscore_cfg["features"],
            stats=zscore_stats
        )

    # 分箱处理
    if binner and binner.bin_params:
        df = binner.transform(df)

    # 类别特征编码
    for cat_feat, le in encoders.items():
        if cat_feat in df.columns:
            df[cat_feat] = df[cat_feat].astype(str)
            unseen = ~df[cat_feat].isin(le.classes_)
            if unseen.any():
                print(f"警告: 特征 {cat_feat} 存在未见过的类别，已替换为0")
                df.loc[unseen, cat_feat] = le.classes_[0]
            df[cat_feat] = le.transform(df[cat_feat])

    return df


def save_preprocessors(save_dir, encoders, binner, outlier_bounds, zscore_stats, config):
    os.makedirs(save_dir, exist_ok=True)

    # 保存编码器
    encoders_path = os.path.join(save_dir, config["preprocess_params"]["encoders_path"])
    with open(encoders_path, "wb") as f:
        pickle.dump(encoders, f)

    # 保存分箱器
    if binner and binner.bin_params:
        binner_path = os.path.join(save_dir, config["preprocess_params"]["binner_path"])
        binner.save(binner_path)

    # 保存异常值边界
    if outlier_bounds:
        outlier_path = os.path.join(save_dir, config["preprocess_params"]["outlier_bounds_path"])
        with open(outlier_path, "wb") as f:
            pickle.dump(outlier_bounds, f)

    # 保存标准化统计量
    if zscore_stats:
        zscore_path = os.path.join(save_dir, config["preprocess_params"]["zscore_stats_path"])
        with open(zscore_path, "wb") as f:
            pickle.dump(zscore_stats, f)


def load_preprocessors(load_dir, config):
    # 加载编码器
    encoders = {}
    encoders_path = os.path.join(load_dir, config["preprocess_params"]["encoders_path"])
    if os.path.exists(encoders_path):
        with open(encoders_path, "rb") as f:
            encoders = pickle.load(f)

    # 加载分箱器
    binner = FeatureBinner()
    binner_path = os.path.join(load_dir, config["preprocess_params"]["binner_path"])
    if os.path.exists(binner_path):
        binner.load(binner_path)

    # 加载异常值边界
    outlier_bounds = None
    outlier_path = os.path.join(load_dir, config["preprocess_params"]["outlier_bounds_path"])
    if os.path.exists(outlier_path):
        with open(outlier_path, "rb") as f:
            outlier_bounds = pickle.load(f)

    # 加载标准化统计量
    zscore_stats = None
    zscore_path = os.path.join(load_dir, config["preprocess_params"]["zscore_stats_path"])
    if os.path.exists(zscore_path):
        with open(zscore_path, "rb") as f:
            zscore_stats = pickle.load(f)

    return encoders, binner, outlier_bounds, zscore_stats


def train_model(train_loader, val_loader, cat_feature_sizes, num_numeric_features, config, device):
    model = DeepFM(
        cat_feature_sizes=cat_feature_sizes,
        num_numeric_features=num_numeric_features,
        embed_dim=config["embed_dim"],
        mlp_layers=config["mlp_layers"],
        dropout=config["dropout"]
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )

    best_auc = 0.0
    for epoch in range(config["epochs"]):
        model.train()
        train_loss = 0.0
        for x_cat, x_num, y in train_loader:
            x_cat, x_num, y = x_cat.to(device), x_num.to(device), y.to(device).view(-1, 1)

            optimizer.zero_grad()
            outputs = model(x_cat, x_num)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x_cat.size(0)

        train_loss /= len(train_loader.dataset)

        # 验证
        model.eval()
        val_loss = 0.0
        y_true = []
        y_pred = []
        with torch.no_grad():
            for x_cat, x_num, y in val_loader:
                x_cat, x_num, y = x_cat.to(device), x_num.to(device), y.to(device).view(-1, 1)
                outputs = model(x_cat, x_num)
                loss = criterion(outputs, y)
                val_loss += loss.item() * x_cat.size(0)

                y_true.extend(y.cpu().numpy())
                y_pred.extend(outputs.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_auc = roc_auc_score(y_true, y_pred)

        print(f"Epoch {epoch+1}/{config['epochs']}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), "best_model.pth")

    return model


def evaluate_model(model, test_loader, device):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for x_cat, x_num, y in test_loader:
            x_cat, x_num = x_cat.to(device), x_num.to(device)
            outputs = model(x_cat, x_num)
            y_true.extend(y.numpy())
            y_pred.extend(outputs.cpu().numpy())

    y_pred = np.round(y_pred).astype(int).flatten()
    y_true = np.array(y_true).flatten()

    metrics = {
        "auc": roc_auc_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred)
    }

    print("\n测试集指标:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    plt.title("混淆矩阵")
    plt.savefig("confusion_matrix.png")
    plt.close()

    return metrics


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    if args.mode == "train":
        # 加载数据
        df = load_data_from_folder(args.data_path)
        print(f"加载数据完成，形状: {df.shape}")

        # 特征工程
        df = extract_user_features(df)
        date_cols = ['inviite_time', 'create_tm', 'valid_begin_date', 'valid_end_date']
        for col in date_cols:
            if col in df.columns:
                df = add_days_from_today(df, col)

        # 预处理
        df, encoders, binner, outlier_bounds, zscore_stats = preprocess_training_data(df, CONFIG)

        # 保存预处理参数
        save_preprocessors(args.save_dir, encoders, binner, outlier_bounds, zscore_stats, CONFIG)

        # 准备特征和标签
        X_cat = df[CONFIG["categorical_features"]].values
        X_num = df[CONFIG["numeric_features"]].values
        y = df[CONFIG["label_col"]].values

        # 计算类别特征大小
        cat_feature_sizes = [len(encoders[feat].classes_) for feat in CONFIG["categorical_features"]]

        # 划分数据集
        X_cat_train, X_cat_test, X_num_train, X_num_test, y_train, y_test = train_test_split(
            X_cat, X_num, y, test_size=CONFIG["test_size"], random_state=42, stratify=y
        )
        X_cat_train, X_cat_val, X_num_train, X_num_val, y_train, y_val = train_test_split(
            X_cat_train, X_num_train, y_train, test_size=CONFIG["val_size"], random_state=42, stratify=y_train
        )

        # 创建数据集和数据加载器
        train_dataset = CustomDataset(X_cat_train, X_num_train, y_train)
        val_dataset = CustomDataset(X_cat_val, X_num_val, y_val)
        test_dataset = CustomDataset(X_cat_test, X_num_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

        # 训练模型
        model = train_model(
            train_loader, val_loader, cat_feature_sizes, len(CONFIG["numeric_features"]), CONFIG, device
        )

        # 评估模型
        model.load_state_dict(torch.load("best_model.pth"))
        evaluate_model(model, test_loader, device)

        # 保存最终模型
        torch.save(model.state_dict(), os.path.join(args.save_dir, "final_model.pth"))
        print(f"模型已保存至 {args.save_dir}")

    elif args.mode == "inference":
        # 加载数据
        df = load_data_from_folder(args.data_path)
        print(f"加载预测数据完成，形状: {df.shape}")

        # 特征工程
        df = extract_user_features(df)
        date_cols = ['inviite_time', 'create_tm', 'valid_begin_date', 'valid_end_date']
        for col in date_cols:
            if col in df.columns:
                df = add_days_from_today(df, col)

        # 加载预处理参数
        encoders, binner, outlier_bounds, zscore_stats = load_preprocessors(args.model_dir, CONFIG)

        # 预处理
        df = preprocess_inference_data(df, CONFIG, encoders, binner, outlier_bounds, zscore_stats)

        # 准备特征
        X_cat = df[CONFIG["categorical_features"]].values
        X_num = df[CONFIG["numeric_features"]].values

        # 计算类别特征大小
        cat_feature_sizes = [len(encoders[feat].classes_) for feat in CONFIG["categorical_features"]]

        # 创建数据集和数据加载器
        dataset = CustomDataset(X_cat, X_num, np.zeros(len(df)))  # 标签用0填充
        loader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=False)

        # 加载模型
        model = DeepFM(
            cat_feature_sizes=cat_feature_sizes,
            num_numeric_features=len(CONFIG["numeric_features"]),
            embed_dim=CONFIG["embed_dim"],
            mlp_layers=CONFIG["mlp_layers"],
            dropout=CONFIG["dropout"]
        ).to(device)
        model.load_state_dict(torch.load(os.path.join(args.model_dir, "final_model.pth")))
        model.eval()

        # 预测
        predictions = []
        with torch.no_grad():
            for x_cat, x_num, _ in loader:
                x_cat, x_num = x_cat.to(device), x_num.to(device)
                outputs = model(x_cat, x_num)
                predictions.extend(outputs.cpu().numpy().flatten())

        # 保存结果
        df["prediction"] = predictions
        df.to_csv(args.output_path, index=False)
        print(f"预测结果已保存至 {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepFM模型训练与预测")
    parser.add_argument("--mode", type=str, choices=["train", "inference"], default="inference", help="运行模式：train或inference")
    parser.add_argument("--data_path", type=str,default="./", help="数据文件路径")
    parser.add_argument("--save_dir", type=str, default="./model", help="训练时模型保存目录")
    parser.add_argument("--model_dir", type=str, default="./model", help="预测时模型加载目录")
    parser.add_argument("--output_path", type=str, default="./predictions.csv", help="预测结果保存路径")
    args = parser.parse_args()

    main(args)