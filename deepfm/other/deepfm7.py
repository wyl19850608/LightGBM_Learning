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
                             precision_score, recall_score,
                             confusion_matrix, precision_recall_curve, auc)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import time
from datetime import datetime
import json
import pickle  # 用于保存和加载编码器、分箱器
import warnings

# 忽略警告信息
warnings.filterwarnings('ignore')


# 特征分箱处理器
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

            # 获取分箱参数，默认5箱，quantile策略
            n_bins = params.get('n_bins', 5)
            strategy = params.get('strategy', 'quantile')

            # 初始化分箱器
            binner = KBinsDiscretizer(
                n_bins=n_bins,
                strategy=strategy,
                encode='ordinal',  # 输出整数编码
                subsample=None
            )

            # 拟合分箱器
            binner.fit(X[[feature]])

            # 保存分箱器和参数
            self.binners[feature] = binner
            self.bin_params[feature] = {
                'n_bins': n_bins,
                'strategy': strategy,
                'bin_edges': binner.bin_edges_[0].tolist()  # 保存分箱边界
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
        """拟合并转换转换数据"""
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

        # 重建分箱器
        for feature, params in self.bin_params.items():
            binner = KBinsDiscretizer(
                n_bins=params['n_bins'],
                strategy=params['strategy'],
                encode='ordinal',
                subsample=None
            )
            binner.bin_edges_ = [np.array(params['bin_edges'])]
            self.binners[feature] = binner


# 配置参数
CONFIG = {
    "test_size": 0.2,
    "val_size": 0.2,
    "batch_size": 256,
    "epochs": 15,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "embed_dim": 32,
    "mlp_layers": [256, 128, 64],
    "dropout": 0.3,
    # 特征列表
    "numeric_features": [
        'age', 'last_30d_tel_succ_cs', 'loss_model_ggroup_v3',
        'risk_ms11_1_model_score', 'standard_score_group_v6_1',
        'last_month_expire_coupon_cnt', 'number_of_gold_be_used',
        'last_10d_lin_e_cnt', 'last_10d_gu_e_cnt',
        'ayht10_all_respond_score', 'call_anss_score_t10',
        'xyl_model_107', 'avail_cash', 'avg_limuse_rate',
        'pril_bal', 'crdt_lim_yx', 'lim_use_rate', 'zaidai_ctrl_rate'
    ],  # 数值特征列表

    "categorical_features": [
        'yls_cust_type_v2', 'cust_types_01', 'cust_types_02',
        'if_sms_yn', 'elec_types', 'igender_cd', 'icust_ty',
        'if_500up_yn', 'is_login', 'sms_types', 'if_bj_30_yn',
        'member_types', 'if_bj_10_yn'
    ],  # 类别特征列表

    "features_list": [
        'user_id', 'unique_id', 'invite_time', 'label', 'age',
        'gender_cd', 'residence_city_cd', 'house_hold_type',
        'marital_stattus_cd', 'edu_deg_cd', 'm_income', 'income_ind',
        'cust_ty', 'custtypes_01', 'cust_types_02', 'yls_cust_type',
        'xyl_tag', 'last_fq_dt', 'last_fq_txk_amt', 'last_tx_dt',
        'last_tx_amt', 'last_30d_login_app_days', 'last_30d_push_touch_times',
        'last_30d_sms_send_succ_cs', 'last_30d_tel_sCC_cs', 'last_5d_coupon_cnt',
        'last_10d_coupon_cnt', 'last_op_time', 'last_pboc_cx_dtуспсq03nwww.',
        'querycncq03wwlw', 'querycncebookswwflaged', 'сdaccapwwwww.',
        'гvaccapwwwwww.', 'гуграфиям', 'густармим', 'rvclaapwwww.',
        'rvblsapwwwww.', 'rvnbmwww.', 'rvnbawpwwwww.', 'rvapsappywww.',
        'rvapmapwwwwww.', 'rvapaapwwwww.', 'repay_date_days',
        'als_m1_id_bank_allnum', 'als_m1_id_nbank_allnum',
        'als_m3_id_bank_allnum', 'als_m3_id_nbank_allnum', 'br_modle_score',
        'query_times_bank_90d', 'query_imes_cfc_90d', 'risk_ms9_model_score',
        'loss_model_group_v3', 'yls_cust_typev2', 'risk_ms11_1_model_score',
        'standard_score_group_v6_1', 'latest_login_days', 'total_loan_cnt',
        'total_loan_amt', 'last_1y_cps_num', 'last_month_expire_coupon_cnt',
        'number_of_gold_be_used', 'cuir_mon_use_gold_cnt', 'late_1mon_use_gold_cnt',
        'late_3mon_use_gold_इंडो', 'late_6mon_use_gold_cnt', 'cur_mon_use_points_cnt',
        'late_1mon_use_points_cnt', 'late_3mon_use_points_cnt', 'late_6mon_use_points_cnt',
        'cur_mon_poins_sign_cnt', 'late_imon_points_sign_cnt', 'late_3mon_points_sign_cnt',
        'late_6mon_points_sign_cnt', 'cur_mon_points_luck_gift_cnt',
        'late_1Imon_points_luck_gift_cnt', 'late_3mon_points_luck_gift_cnt',
        'late_6mon_points_luck_gift_cnt', 'cur_mon_cps_order_types',
        'late_imon_cps_order_types', 'late_3mon_cps_order_types',
        'late_6mon_cps_order_types', 'last_mon_game_accum_click_cnt',
        'last_mon_game_accum_expo_cnt', 'last_mon_read_accum_click_cnt',
        'last_mon_read_accum_expo_cnt', 'cur_yili_vipterms', 'cur_juhui_vip_terms',
        'last_login_ayh_time', 'last_login_app_time', 'last_uas_reject_time',
        'last_30d_fq_cs', 'last_30d_tx_cs', 'last_30d_login_ayh_days',
        'last_5d_lin_e_cnt', 'last_10d_lin_e_cnt', 'last_Sd_gu_e_cnt',
        'last_10d_gu_e_cnt', 'last_yunying_jj_new_rate', 'last_yunying_jj_beegin_time',
        'last_yunying_jj_end_time', 'last_90d_yunying_jj_cs', 'last_180d_yunying_jj_cs',
        'partner_yx_contr_cnt', 'ayht10_all_respond_score', 'call_ans_score_t10',
        'member_types', 'xyl_model_107', 'kzr_types', 'elec_types', 'sms_types',
        'if_sms_yn', 'if_ele_yn', 'if_500up_yn', 'if_bj_10_yn', 'if_bj_30_yn',
        'avail_bal_cash', 'send_coupon_cnt_1mon', 'use_coupon_cnt_1mon',
        'valid_coupon_cnt', 'coupon_use', 'clkk_wyyh_count', 'clk_llydg_count',
        'clk_yhq_sy_count', 'clk_hyzx_sy_count', 'clk_jryhq_count', 'zaidai_days',
        'zaidai_days/mob_3y as zaidai_rate', 'avg_lim_use_rate', 'pril_bal',
        'crdt_lim_yx', 'lim_use_rate', 'zaidai_ctrl_rate', 'is_login',
        'list_call_times_d', 'list_ansr_times_d', 'list_ansr_dur_d',
        'list_manu_call_times_d', 'list_manu_ansr_times_d', 'list_manu_ansr_dur_d',
        'list_ivr_call_times_d', 'list_ivr_ansr_timees_d', 'list_ivr_ansr_dur_d',
        'list_call_times', 'list_ansr_times', 'list_ansr_dur', 'cust_call_times',
        'cust_ansr_times', 'cust_ansr_dur', 'cust_call_times_mon',
        'cust_ansr_times_mon', 'cust_ansr_dur_mon', 'cust_manu_call_times',
        'cust_manu_ansr_times', 'cust_manu_ansr_dur', 'cust_manu_call_times_mon',
        'cust_manu_ansr_times_mon', 'cust_manu_ansr_dur_mon', 'cust_ivr_call_times',
        'cust_ivr_ansr_times', 'cust_ivr_ansr_dur', 'otust_ivr_call_times_mon',
        'cust_ivr_ansr_times_mon', 'cust_ivr_ansr_dur_mon'
    ],
    # 分箱配置 - 每个数值特征的分箱参数
    "bin_config": {
        # 示例配置：'age': {'n_bins': 5, 'strategy': 'quantile'},
        # 未配置的特征将保持原样
    },
    "label_col": "label",
    "data_folder": "./",
    # 缺失值处理策略
    "numeric_missing_strategy": "mean",  # 选项: "mean", "median", "constant"
    "numeric_fill_value": 0,
    "categorical_missing_strategy": "constant",  # 选项: "mode", "constant"
    "categorical_fill_value": "Missing",
    # 不平衡处理配置
    "imbalance_method": "both",  # 选项: 'weight', 'undersample', 'both'
    "undersample_ratio": 3,  # 下采样的负正样本比例
    # 阈值选择
    "threshold_tuning": False,
    "optimize_for": "f1",  # 选项: 'f1', 'precision', 'recall'
    "positive_threshold": 0.5,
    "min_threshold": 0.1,
    "max_threshold": 0.9,
    "min_pos_samples": 20  # 每个数据集所需的最小正样本数
}


# 保存编码器和分箱器
def save_preprocessors(encoders, binner, save_dir):
    """保存编码器和分箱器到指定目录"""
    # 保存编码器
    encoders_path = os.path.join(save_dir, "encoders.pkl")
    with open(encoders_path, "wb") as f:
        pickle.dump(encoders, f)
    print(f"编码器已保存至: {encoders_path}")

    # 保存分箱器
    if binner and binner.bin_params:
        binner_path = os.path.join(save_dir, "binner.pkl")
        binner.save(binner_path)
        print(f"分箱器已保存至: {binner_path}")


# 加载编码器和分箱器
def load_preprocessors(load_dir):
    """从指定目录加载编码器和分箱器"""
    # 加载编码器
    encoders_path = os.path.join(load_dir, "encoders.pkl")
    if not os.path.exists(encoders_path):
        raise FileNotFoundError(f"编码器文件不存在: {encoders_path}")

    with open(encoders_path, "rb") as f:
        encoders = pickle.load(f)
    print(f"编码器已从 {encoders_path} 加载")

    # 加载分箱器
    binner = FeatureBinner()
    binner_path = os.path.join(load_dir, "binner.pkl")
    if os.path.exists(binner_path):
        binner.load(binner_path)
        print(f"分箱器已从 {binner_path} 加载")

    return encoders, binner


# 从文件夹加载数据
def load_data_from_folder(folder_path):
    """加载文件夹中所有CSV文件并合并为一个DataFrame"""
    all_files = []

    # 遍历文件夹中的所有文件
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            file_path = os.path.join(folder_path, file)
            try:
                df = pd.read_csv(file_path)
                # 检查并添加features_list
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


# 处理缺失值
def handle_missing_values(df, numeric_features, categorical_features):
    """根据配置处理缺失值"""
    df = df.copy()

    # 处理数值型特征的缺失值
    for col in numeric_features:
        if col in df.columns and df[col].isnull().any():
            if CONFIG["numeric_missing_strategy"] == "mean":
                fill_value = df[col].mean()
            elif CONFIG["numeric_missing_strategy"] == "median":
                fill_value = df[col].median()
            else:  # 常数填充
                fill_value = CONFIG["numeric_fill_value"]
            df[col] = df[col].fillna(fill_value)

    # 处理分类型特征的缺失值
    for col in categorical_features:
        if col in df.columns and df[col].isnull().any():
            if CONFIG["categorical_missing_strategy"] == "mode":
                fill_value = df[col].mode()[0]
            else:  # 常数填充
                fill_value = CONFIG["categorical_fill_value"]
            df[col] = df[col].fillna(fill_value)

    return df


# 下采样函数
def undersample_data(df, label_col, ratio=5):
    """对多数类进行下采样"""
    pos_df = df[df[label_col] == 1]
    neg_df = df[df[label_col] == 0]
    n_pos = len(pos_df)
    n_neg = min(len(neg_df), n_pos * ratio)
    neg_sample = neg_df.sample(n_neg, random_state=42)
    balanced_df = pd.concat([pos_df, neg_sample], axis=0).sample(frac=1, random_state=42)
    return balanced_df


# 计算详细指标
def calculate_detailed_metrics(y_true, y_pred_proba, threshold=0.5):
    """计算详细指标，包括每个类别的统计信息"""
    # 转换为分类预测结果
    y_pred = (y_pred_proba >= threshold).astype(int)

    # 整体指标
    acc = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    f1 = f1_score(y_true, y_pred)

    # 计算两个类别的精确率和召回率
    precision_neg = precision_score(y_true, y_pred, pos_label=0)
    recall_neg = recall_score(y_true, y_pred, pos_label=0)
    f1_neg = f1_score(y_true, y_pred, pos_label=0)

    precision_pos = precision_score(y_true, y_pred)
    recall_pos = recall_score(y_true, y_pred)
    f1_pos = f1_score(y_true, y_pred)

    # 类别计数
    count_pos = sum(y_true)
    count_neg = len(y_true) - count_pos

    # 计算PR曲线下面积
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba)
    aucpr = auc(recall_curve, precision_curve)

    # 构建结果字典
    metrics = {
        'overall': {
            "accuracy": acc,
            "roc_auc": roc_auc,
            "auc_pr": aucpr,
            "f1_score": f1
        },
        'positive': {
            "precision": precision_pos,
            "recall": recall_pos,
            "f1": f1_pos,
            "support": count_pos
        },
        'negative': {
            "precision": precision_neg,
            "recall": recall_neg,
            "f1": f1_neg,
            "support": count_neg
        },
        "threshold": threshold
    }

    return metrics


# 打印详细指标
def print_detailed_metrics(metrics, dataset_name):
    """打印详细的分类指标"""
    # 打印标题
    title = f"{dataset_name} 数据集指标"
    print("\n" + "=" * 70)
    print(title.center(70))
    print("=" * 70)

    # 整体指标
    print("\n整体指标:")
    print(f"  准确率: {metrics['overall']['accuracy']:.4f}")
    print(f"  ROC曲线下面积: {metrics['overall']['roc_auc']:.4f}")
    print(f"  PR曲线下面积: {metrics['overall']['auc_pr']:.4f}")
    print(f"  F1分数: {metrics['overall']['f1_score']:.4f}")
    print(f"  预测阈值: {metrics['threshold']:.4f}")

    # 正样本指标
    print("\n正样本(少数类)指标:")
    print(f"  精确率: {metrics['positive']['precision']:.4f}")
    print(f"  召回率: {metrics['positive']['recall']:.4f}")
    print(f"  F1分数: {metrics['positive']['f1']:.4f}")
    print(f"  样本数: {metrics['positive']['support']}")

    # 负样本指标
    print("\n负样本(多数类)指标:")
    print(f"  精确率: {metrics['negative']['precision']:.4f}")
    print(f"  召回率: {metrics['negative']['recall']:.4f}")
    print(f"  F1分数: {metrics['negative']['f1']:.4f}")
    print(f"  样本数: {metrics['negative']['support']}")

    # 底部装饰线
    print("=" * 70 + "\n")
    return metrics


# 保存结果
def save_results(trainer, config, test_metrics, encoders, binner, results_dir=None):
    """将所有训练结果保存到文件，包括编码器和分箱器"""
    # 创建结果目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if results_dir is None:
        results_dir = f"results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    # 1. 保存配置文件
    with open(os.path.join(results_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    # 2. 保存训练历史
    train_history = []
    for i, metrics in enumerate(trainer.train_metrics_history):
        epoch_data = {
            "epoch": i + 1,
            "train_loss": trainer.train_loss_history[i] if i < len(trainer.train_loss_history) else None,
            "train_accuracy": metrics["overall"]["accuracy"],
            "train_auc_pr": metrics["overall"]["auc_pr"],
            "train_f1": metrics["overall"]["f1_score"],
            "val_accuracy": trainer.val_metrics_history[i]["overall"]["accuracy"],
            "val_auc_pr": trainer.val_metrics_history[i]["overall"]["auc_pr"],
            "val_f1": trainer.val_metrics_history[i]["overall"]["f1_score"]
        }
        train_history.append(epoch_data)

    pd.DataFrame(train_history).to_csv(os.path.join(results_dir, "training_history.csv"), index=False)

    # 3. 保存最佳模型
    torch.save(trainer.best_model_state, os.path.join(results_dir, "best_model.pth"))

    # 4. 保存最终模型
    torch.save(trainer.model.state_dict(), os.path.join(results_dir, "final_model.pth"))

    # 5. 保存测试结果
    test_results = {
        "test_accuracy": test_metrics["overall"]["accuracy"],
        "test_auc_pr": test_metrics["overall"]["auc_pr"],
        "test_f1": test_metrics["overall"]["f1_score"],
        "test_threshold": test_metrics["threshold"],
        "positive_precision": test_metrics["positive"]["precision"],
        "positive_recall": test_metrics["positive"]["recall"],
        "positive_f1": test_metrics["positive"]["f1"],
        "negative_precision": test_metrics["negative"]["precision"],
        "negative_recall": test_metrics["negative"]["recall"],
        "negative_f1": test_metrics["negative"]["f1"],
        "best_val_auc_pr": trainer.best_auc,
        "best_threshold": trainer.best_threshold
    }

    with open(os.path.join(results_dir, "test_results.json"), "w") as f:
        json.dump(test_results, f, indent=4)

    # 6. 保存混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(trainer.test_confusion_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['负样本', '正样本'],
                yticklabels=['负样本', '正样本'])
    plt.title('测试集混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()

    # 7. 保存编码器和分箱器
    save_preprocessors(encoders, binner, results_dir)

    return results_dir


# DeepFM数据集类
class DeepFMDataset(Dataset):
    def __init__(self, df, numeric_features, categorical_features, label_col,
                 encoders=None, binner=None, is_train=True):
        self.df = df.copy().reset_index(drop=True)
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.label_col = label_col
        self.is_train = is_train
        self.binner = binner

        # 处理缺失值
        self.df = handle_missing_values(self.df, numeric_features, categorical_features)

        # 应用分箱处理（仅对训练集拟合，对验证/测试集仅转换）
        if self.binner:
            if is_train:
                self.df[self.numeric_features] = self.binner.fit_transform(
                    self.df[self.numeric_features], CONFIG["bin_config"])
            else:
                self.df[self.numeric_features] = self.binner.transform(
                    self.df[self.numeric_features])

        # 处理分类型特征
        self.categorical_encoders = {} if encoders is None else encoders
        self.categorical_dims = {}
        self._encode_categorical()

    def _encode_categorical(self):
        for col in self.categorical_features:
            if col in self.df.columns:  # 确保列存在
                if self.is_train or col not in self.categorical_encoders:
                    le = LabelEncoder()
                    self.df[col] = le.fit_transform(self.df[col].astype(str))
                    self.categorical_encoders[col] = le
                else:
                    le = self.categorical_encoders[col]
                    # 处理未见过的类别 - 映射到特殊值
                    self.df[col] = self.df[col].apply(lambda x: x if x in le.classes_ else 'UNKNOWN')
                    # 如果'UNKNOWN'不在编码器类别中，则添加
                    if 'UNKNOWN' not in le.classes_:
                        le.classes_ = np.append(le.classes_, 'UNKNOWN')
                    self.df[col] = le.transform(self.df[col].astype(str))

                self.categorical_dims[col] = len(le.classes_)

        return self.categorical_encoders

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 数值特征
        numeric_data = self.df[self.numeric_features].iloc[idx].values if self.numeric_features else np.array([])
        numeric_tensor = torch.tensor(numeric_data.astype(np.float32), dtype=torch.float32)

        # 分类特征
        categorical_data = []
        for col in self.categorical_features:
            if col in self.df.columns:
                categorical_data.append(self.df.loc[idx, col])
            else:
                categorical_data.append(0)  # 默认值

        categorical_tensor = torch.tensor(categorical_data, dtype=torch.long)

        # 标签
        label = torch.tensor(self.df.loc[idx, self.label_col], dtype=torch.float32)

        return {
            'numeric': numeric_tensor,
            'categorical': categorical_tensor,
            'label': label
        }


# DeepFM模型
class DeepFM(nn.Module):
    def __init__(self, numeric_dim, categorical_dims, embed_dim=32, mlp_layers=[256, 128, 64], dropout=0.3):
        super().__init__()

        # 确保有数值特征或分类特征
        self.has_numeric = numeric_dim > 0
        self.has_categorical = len(categorical_dims) > 0

        # ========== FM部分 ==========
        # FM一阶部分（分类特征）
        self.fm_first_order_cat = nn.ModuleDict()
        if self.has_categorical:
            for col, num_embeddings in categorical_dims.items():
                self.fm_first_order_cat[col] = nn.Embedding(num_embeddings, 1)

        # FM一阶部分（数值特征）
        if self.has_numeric:
            self.fm_first_order_num = nn.Linear(numeric_dim, 1)

        # FM二阶部分（分类特征）
        self.fm_second_order = nn.ModuleDict()
        if self.has_categorical:
            for col, num_embeddings in categorical_dims.items():
                self.fm_second_order[col] = nn.Embedding(num_embeddings, embed_dim)

        # ========== DNN部分 ==========
        # DNN嵌入层（分类特征）
        self.dnn_embeddings = nn.ModuleDict()
        if self.has_categorical:
            for col, num_embeddings in categorical_dims.items():
                self.dnn_embeddings[col] = nn.Embedding(num_embeddings, embed_dim)

        # DNN数值特征处理
        if self.has_numeric:
            self.numeric_layer = nn.Linear(numeric_dim, embed_dim)

        # DNN全连接层
        input_dim = embed_dim * len(categorical_dims) + (embed_dim if self.has_numeric else 0)
        layers = []
        for i, out_dim in enumerate(mlp_layers):
            layers.append(nn.Linear(input_dim, out_dim))
            layers.append(nn.BatchNorm1d(out_dim))  # 添加批归一化
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = out_dim

        self.dnn = nn.Sequential(*layers) if layers else nn.Identity()

        # 输出层
        # FM部分输出维度: 一阶(1) + 二阶(1)
        fm_output_dim = 2 if self.has_categorical else 1
        # DNN部分输出维度
        dnn_output_dim = mlp_layers[-1] if mlp_layers else 0
        # 合并输出层
        self.output_layer = nn.Linear(fm_output_dim + dnn_output_dim, 1)

    def forward(self, numeric, categorical):
        batch_size = numeric.shape[0] if self.has_numeric else categorical.shape[0]
        components = []

        # ========== FM部分 ==========
        # 一阶特征
        fm_first_total = torch.zeros(batch_size, 1, device=numeric.device if self.has_numeric else categorical.device)

        # 分类特征一阶项
        fm_first_cat_list = []
        for i, col in enumerate(self.fm_first_order_cat):
            # 输入形状: [batch_size] -> 输出形状: [batch_size, 1]
            emb = self.fm_first_order_cat[col](categorical[:, i])
            fm_first_cat_list.append(emb)

        # 拼接所有嵌入: [batch_size, num_cat_features]
        fm_first_cat = torch.cat(fm_first_cat_list, dim=1)
        # 求和: [batch_size, 1]
        fm_first_cat = fm_first_cat.sum(dim=1, keepdim=True)
        fm_first_total += fm_first_cat

        # 数值特征一阶项
        if self.has_numeric:
            fm_first_num = self.fm_first_order_num(numeric).view(-1, 1)
            fm_first_total += fm_first_num

        components.append(fm_first_total)

        # 二阶特征
        if self.has_categorical:
            fm_second_embeds = torch.cat([
                self.fm_second_order[col](categorical[:, i])
                for i, col in enumerate(self.fm_second_order)
            ], dim=1)  # (batch_size, embed_dim * num_cat_features)

            # FM二阶计算
            square_of_sum = torch.sum(fm_second_embeds, dim=1).pow(2)
            sum_of_square = torch.sum(fm_second_embeds.pow(2), dim=1)
            fm_second_order = 0.5 * (square_of_sum - sum_of_square).unsqueeze(1)
            components.append(fm_second_order)

        # ========== DNN部分 ==========
        dnn_embeds = []

        # 分类特征嵌入
        if self.has_categorical:
            dnn_embeds.extend([
                self.dnn_embeddings[col](categorical[:, i])
                for i, col in enumerate(self.dnn_embeddings)
            ])

        # 数值特征嵌入
        if self.has_numeric:
            numeric_embed = self.numeric_layer(numeric)
            dnn_embeds.append(numeric_embed)

        if dnn_embeds:
            dnn_input = torch.cat(dnn_embeds, dim=1)
            dnn_output = self.dnn(dnn_input)
            components.append(dnn_output)

        # 合并所有部分
        total = torch.cat(components, dim=1)

        # 最终输出
        output = self.output_layer(total)
        return torch.sigmoid(output).squeeze(1)


# DeepFM训练器
class DeepFMTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader, optimizer, device='cpu',
                 load_from_checkpoint=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.device = device

        # 损失函数（支持类别权重）
        if CONFIG["imbalance_method"] in ['weight', 'both']:
            # 计算正样本权重
            total_samples = len(train_loader.dataset)
            pos_samples = sum(1 for x in train_loader.dataset if x['label'] == 1)
            neg_samples = total_samples - pos_samples
            pos_weight = torch.tensor(neg_samples / pos_samples, device=device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.criterion = nn.BCELoss()

        # 训练跟踪
        self.best_auc = 0
        self.best_model_state = None
        self.best_threshold = CONFIG["positive_threshold"]
        self.train_metrics_history = []  # 存储训练指标历史
        self.val_metrics_history = []  # 存储验证指标历史
        self.train_loss_history = []  # 存储训练损失历史
        self.test_confusion_matrix = None  # 存储测试混淆矩阵

        # 加载检查点（如果提供）
        if load_from_checkpoint:
            self.load_checkpoint(load_from_checkpoint)

    def load_checkpoint(self, checkpoint_path):
        """从检查点加载模型状态和训练历史"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_auc = checkpoint.get('best_auc', 0)
        self.best_model_state = checkpoint.get('best_model_state')
        self.train_metrics_history = checkpoint.get('train_metrics_history', [])
        self.val_metrics_history = checkpoint.get('val_metrics_history', [])
        self.train_loss_history = checkpoint.get('train_loss_history', [])
        print(f"已从 {checkpoint_path} 加载检查点")

    def save_checkpoint(self, path):
        """保存训练检查点"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_auc': self.best_auc,
            'best_model_state': self.best_model_state,
            'train_metrics_history': self.train_metrics_history,
            'val_metrics_history': self.val_metrics_history,
            'train_loss_history': self.train_loss_history
        }
        torch.save(checkpoint, path)
        print(f"检查点已保存至 {path}")

    def train_one_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for batch in self.train_loader:
            numeric = batch['numeric'].to(self.device)
            categorical = batch['categorical'].to(self.device)
            labels = batch['label'].to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(numeric, categorical)
            loss = self.criterion(outputs, labels)

            # 反向传播和优化
            loss.backward()
            self.optimizer.step()

            # 累计损失和预测结果
            total_loss += loss.item() * numeric.size(0)
            all_preds.extend(outputs.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

        # 计算平均损失和指标
        avg_loss = total_loss / len(self.train_loader.dataset)
        metrics = calculate_detailed_metrics(np.array(all_labels), np.array(all_preds), self.best_threshold)

        return avg_loss, metrics

    def evaluate(self, data_loader, is_test=False):
        """评估模型"""
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in data_loader:
                numeric = batch['numeric'].to(self.device)
                categorical = batch['categorical'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(numeric, categorical)
                all_preds.extend(outputs.detach().cpu().numpy())
                all_labels.extend(labels.detach().cpu().numpy())

        # 转换为numpy数组
        y_true = np.array(all_labels)
        y_pred_proba = np.array(all_preds)

        # 阈值调优（仅在验证集上）
        if CONFIG["threshold_tuning"] and not is_test:
            best_metric = -1
            best_thresh = CONFIG["positive_threshold"]

            # 尝试不同阈值
            for thresh in np.arange(CONFIG["min_threshold"], CONFIG["max_threshold"], 0.01):
                metrics = calculate_detailed_metrics(y_true, y_pred_proba, thresh)

                # 根据优化目标选择最佳阈值
                if CONFIG["optimize_for"] == 'f1':
                    current_metric = metrics['positive']['f1']
                elif CONFIG["optimize_for"] == 'precision':
                    current_metric = metrics['positive']['precision']
                else:  # recall
                    current_metric = metrics['positive']['recall']

                # 确保有足够的正样本预测
                n_pos_pred = sum(y_pred_proba >= thresh)
                if n_pos_pred >= CONFIG["min_pos_samples"] and current_metric > best_metric:
                    best_metric = current_metric
                    best_thresh = thresh

            self.best_threshold = best_thresh

        # 计算指标
        metrics = calculate_detailed_metrics(y_true, y_pred_proba, self.best_threshold)

        # 保存测试集混淆矩阵
        if is_test:
            y_pred = (y_pred_proba >= self.best_threshold).astype(int)
            self.test_confusion_matrix = confusion_matrix(y_true, y_pred)

        return metrics

    def train(self, epochs):
        """完整训练过程"""
        print(f"开始训练，共 {epochs} 个epoch，使用设备: {self.device}")

        for epoch in range(epochs):
            start_time = time.time()

            # 训练一个epoch
            train_loss, train_metrics = self.train_one_epoch()
            self.train_loss_history.append(train_loss)
            self.train_metrics_history.append(train_metrics)

            # 在验证集上评估
            val_metrics = self.evaluate(self.val_loader)
            self.val_metrics_history.append(val_metrics)

            # 保存最佳模型
            if val_metrics['overall']['auc_pr'] > self.best_auc:
                self.best_auc = val_metrics['overall']['auc_pr']
                self.best_model_state = self.model.state_dict()

            # 打印 epoch 信息
            epoch_time = time.time() - start_time
            print(f"\nEpoch {epoch+1}/{epochs} - 耗时: {epoch_time:.2f}s")
            print(f"训练损失: {train_loss:.4f} - 训练F1: {train_metrics['overall']['f1_score']:.4f}")
            print(f"验证F1: {val_metrics['overall']['f1_score']:.4f} - 验证AUC-PR: {val_metrics['overall']['auc_pr']:.4f}")
            print(f"当前最佳验证AUC-PR: {self.best_auc:.4f}")

        # 训练结束后，使用最佳模型评估测试集
        print("\n训练完成，使用最佳模型评估测试集...")
        self.model.load_state_dict(self.best_model_state)
        test_metrics = self.evaluate(self.test_loader, is_test=True)

        return test_metrics


# 主函数
def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载数据
    print("\n加载数据...")
    data = load_data_from_folder(CONFIG["data_folder"])
    # 只保留需要的列：数值特征、类别特征和标签列
    required_columns = (CONFIG["numeric_features"] +
                        CONFIG["categorical_features"] +
                        [CONFIG["label_col"]])

    # 过滤数据，只保留必要的列
    # 处理可能的列名不匹配问题
    available_columns = [col for col in required_columns if col in data.columns]
    missing_columns = [col for col in required_columns if col not in data.columns]

    if missing_columns:
        print(f"警告：以下必要列在数据中未找到，将被忽略: {missing_columns}")

    data = data[available_columns].copy()
    print(f"过滤后的数据包含 {len(data.columns)} 列特征和标签")



    # 分割训练集和测试集
    print("\n分割数据集...")
    train_val_df, test_df = train_test_split(
        data, test_size=CONFIG["test_size"], random_state=42, stratify=data[CONFIG["label_col"]]
    )

    # 分割训练集和验证集
    train_df, val_df = train_test_split(
        train_val_df, test_size=CONFIG["val_size"], random_state=42, stratify=train_val_df[CONFIG["label_col"]]
    )

    # 处理类别不平衡（下采样）
    if CONFIG["imbalance_method"] in ['undersample', 'both']:
        print(f"对训练集进行下采样，负正样本比例为 {CONFIG['undersample_ratio']}:1")
        train_df = undersample_data(train_df, CONFIG["label_col"], CONFIG["undersample_ratio"])

    # 初始化分箱器
    binner = FeatureBinner() if CONFIG["bin_config"] else None

    # 创建数据集
    print("\n创建数据集...")
    train_dataset = DeepFMDataset(
        train_df, CONFIG["numeric_features"], CONFIG["categorical_features"],
        CONFIG["label_col"], is_train=True, binner=binner
    )
    val_dataset = DeepFMDataset(
        val_df, CONFIG["numeric_features"], CONFIG["categorical_features"],
        CONFIG["label_col"], encoders=train_dataset.categorical_encoders,
        binner=binner, is_train=False
    )
    test_dataset = DeepFMDataset(
        test_df, CONFIG["numeric_features"], CONFIG["categorical_features"],
        CONFIG["label_col"], encoders=train_dataset.categorical_encoders,
        binner=binner, is_train=False
    )

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=4)

    # 初始化模型
    print("\n初始化模型...")
    model = DeepFM(
        numeric_dim=len(CONFIG["numeric_features"]),
        categorical_dims=train_dataset.categorical_dims,
        embed_dim=CONFIG["embed_dim"],
        mlp_layers=CONFIG["mlp_layers"],
        dropout=CONFIG["dropout"]
    )

    # 初始化优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"]
    )

    # 初始化训练器
    trainer = DeepFMTrainer(
        model, train_loader, val_loader, test_loader, optimizer, device=device
    )

    # 训练模型
    print("\n开始训练...")
    test_metrics = trainer.train(epochs=CONFIG["epochs"])

    # 打印测试集指标
    print_detailed_metrics(test_metrics, "测试集")

    # 保存结果
    print("\n保存结果...")
    results_dir = save_results(
        trainer, CONFIG, test_metrics,
        train_dataset.categorical_encoders, binner
    )
    print(f"所有结果已保存至: {results_dir}")


def predict_new_data(new_data_path, model_dir, output_path=None):
    """
    对新数据进行预测

    参数:
        new_data_path: 新数据文件路径或文件夹路径
        model_dir: 模型和预处理文件所在目录
        output_path: 预测结果保存路径，为None则不保存

    返回:
        包含预测结果的DataFrame
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 1. 加载配置
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)
    print("配置文件加载完成")

    # 2. 加载编码器和分箱器
    encoders, binner = load_preprocessors(model_dir)
    print("预处理工具加载完成")

    # 3. 加载模型
    model_path = os.path.join(model_dir, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    # 准备数据集以获取分类特征维度
    dummy_df = pd.DataFrame({col: ["dummy"] for col in config["categorical_features"]})
    dummy_dataset = DeepFMDataset(
        dummy_df, config["numeric_features"], config["categorical_features"],
        config["label_col"], encoders=encoders, binner=binner, is_train=False
    )

    # 初始化模型并加载权重
    model = DeepFM(
        numeric_dim=len(config["numeric_features"]),
        categorical_dims=dummy_dataset.categorical_dims,
        embed_dim=config["embed_dim"],
        mlp_layers=config["mlp_layers"],
        dropout=config["dropout"]
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("模型加载完成")

    # 4. 加载和预处理新数据
    print("加载并预处理新数据...")
    # 判断输入是文件还是文件夹
    if os.path.isdir(new_data_path):
        new_data = load_data_from_folder(new_data_path)
    else:
        new_data = pd.read_csv(new_data_path)

    # 过滤需要的列
    required_columns = config["numeric_features"] + config["categorical_features"]
    available_columns = [col for col in required_columns if col in new_data.columns]
    missing_columns = [col for col in required_columns if col not in new_data.columns]

    if missing_columns:
        print(f"警告：以下特征在新数据中未找到，将使用默认值: {missing_columns}")

    # 为缺失的特征添加默认值
    for col in missing_columns:
        if col in config["numeric_features"]:
            new_data[col] = config["numeric_fill_value"]
        else:
            new_data[col] = config["categorical_fill_value"]

    new_data = new_data[required_columns].copy()

    # 5. 创建数据集和数据加载器
    predict_dataset = DeepFMDataset(
        new_data, config["numeric_features"], config["categorical_features"],
        config["label_col"], encoders=encoders, binner=binner, is_train=False
    )
    predict_loader = DataLoader(
        predict_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=4
    )

    # 6. 进行预测
    print("开始预测...")
    all_preds = []

    with torch.no_grad():
        for batch in predict_loader:
            numeric = batch['numeric'].to(device)
            categorical = batch['categorical'].to(device)

            outputs = model(numeric, categorical)
            all_preds.extend(outputs.detach().cpu().numpy())

    # 7. 处理预测结果
    predictions = np.array(all_preds)
    new_data["pred_prob"] = predictions
    new_data["pred_label"] = (predictions >= config["positive_threshold"]).astype(int)

    # 8. 保存结果
    if output_path:
        new_data.to_csv(output_path, index=False)
        print(f"预测结果已保存至: {output_path}")

    return new_data

if __name__ == "__main__":
    main()