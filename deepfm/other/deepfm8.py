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
import pickle
import warnings

warnings.filterwarnings('ignore')


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
    "numeric_features": [
        'age', 'last_30d_tel_succ_cs', 'loss_model_ggroup_v3',
        'risk_ms11_1_model_score', 'standard_score_group_v6_1',
        'last_month_expire_coupon_cnt', 'number_of_gold_be_used',
        'last_10d_lin_e_cnt', 'last_10d_gu_e_cnt',
        'ayht10_all_respond_score', 'call_anss_score_t10',
        'xyl_model_107', 'avail_cash', 'avg_limuse_rate',
        'pril_bal', 'crdt_lim_yx', 'lim_use_rate', 'zaidai_ctrl_rate'
    ],
    "categorical_features": [
        'yls_cust_type_v2', 'cust_types_01', 'cust_types_02',
        'if_sms_yn', 'elec_types', 'igender_cd', 'icust_ty',
        'if_500up_yn', 'is_login', 'sms_types', 'if_bj_30_yn',
        'member_types', 'if_bj_10_yn'
    ],
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
    "bin_config": {
        # 核心特征：分箱数20（保留更多分布信息）
        'age': {'n_bins': 20, 'strategy': 'quantile'},
        'last_30d_tel_succ_cs': {'n_bins': 20, 'strategy': 'quantile'},
        'risk_ms11_1_model_score': {'n_bins': 20, 'strategy': 'quantile'},
        'standard_score_group_v6_1': {'n_bins': 20, 'strategy': 'quantile'},

        # 重要特征：分箱数15
        'loss_model_ggroup_v3': {'n_bins': 15, 'strategy': 'quantile'},
        'ayht10_all_respond_score': {'n_bins': 15, 'strategy': 'quantile'},
        'xyl_model_107': {'n_bins': 15, 'strategy': 'quantile'},
        'avg_limuse_rate': {'n_bins': 15, 'strategy': 'quantile'},
        'lim_use_rate': {'n_bins': 15, 'strategy': 'quantile'},

        # 一般特征：分箱数10
        'last_month_expire_coupon_cnt': {'n_bins': 10, 'strategy': 'quantile'},
        'number_of_gold_be_used': {'n_bins': 10, 'strategy': 'quantile'},
        'last_10d_lin_e_cnt': {'n_bins': 10, 'strategy': 'quantile'},
        'last_10d_gu_e_cnt': {'n_bins': 10, 'strategy': 'quantile'},
        'call_anss_score_t10': {'n_bins': 10, 'strategy': 'quantile'},
        'pril_bal': {'n_bins': 10, 'strategy': 'quantile'},
        'crdt_lim_yx': {'n_bins': 10, 'strategy': 'quantile'},
        'zaidai_ctrl_rate': {'n_bins': 10, 'strategy': 'quantile'},

        # 计数/金额类特征：分箱数10（避免极端值稀疏导致过拟合）
        'avail_cash': {'n_bins': 10, 'strategy': 'quantile'}
    },
    "label_col": "label",
    "data_folder": "./",
    "numeric_missing_strategy": "mean",
    "numeric_fill_value": 0,
    "categorical_missing_strategy": "constant",
    "categorical_fill_value": "Missing",
    "imbalance_method": "both",
    "undersample_ratio": 3,
    "threshold_tuning": False,
    "optimize_for": "f1",
    "positive_threshold": 0.5,
    "min_threshold": 0.1,
    "max_threshold": 0.9,
    "min_pos_samples": 20
}


def save_preprocessors(encoders, binner, save_dir):
    """保存编码器和分箱器到指定目录"""
    encoders_path = os.path.join(save_dir, "encoders.pkl")
    with open(encoders_path, "wb") as f:
        pickle.dump(encoders, f)
    print(f"编码器已保存至: {encoders_path}")

    if binner and binner.bin_params:
        binner_path = os.path.join(save_dir, "binner.pkl")
        binner.save(binner_path)
        print(f"分箱器已保存至: {binner_path}")


def load_preprocessors(load_dir):
    """从指定目录加载编码器和分箱器"""
    encoders_path = os.path.join(load_dir, "encoders.pkl")
    if not os.path.exists(encoders_path):
        raise FileNotFoundError(f"编码器文件不存在: {encoders_path}")

    with open(encoders_path, "rb") as f:
        encoders = pickle.load(f)
    print(f"编码器已从 {encoders_path} 加载")

    binner = FeatureBinner()
    binner_path = os.path.join(load_dir, "binner.pkl")
    if os.path.exists(binner_path):
        binner.load(binner_path)
        print(f"分箱器已从 {binner_path} 加载")

    return encoders, binner


def load_data_from_folder(folder_path):
    """加载文件夹中所有CSV文件并合并为一个DataFrame"""
    all_files = []

    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            file_path = os.path.join(folder_path, file)
            try:
                df = pd.read_csv(file_path)
                if 'features_list' in CONFIG:
                    if len(df.columns) == len(CONFIG['features_list']):
                        df.columns = CONFIG['features_list']
                all_files.append(df)
                print(f"已加载 {file}，包含 {len(df)} 行数据")
            except Exception as e:
                print(f"加载 {file} 时出错: {str(e)}")

    if not all_files:
        raise ValueError(f"在 {folder_path} 中未找到CSV文件")

    combined_df = pd.concat(all_files, axis=0, ignore_index=True)
    print(f"\n合并后的数据集大小: {len(combined_df)} 行")
    return combined_df


def handle_missing_values(df, numeric_features, categorical_features):
    """根据配置处理缺失值"""
    df = df.copy()

    for col in numeric_features:
        if col in df.columns and df[col].isnull().any():
            if CONFIG["numeric_missing_strategy"] == "mean":
                fill_value = df[col].mean()
            elif CONFIG["numeric_missing_strategy"] == "median":
                fill_value = df[col].median()
            else:
                fill_value = CONFIG["numeric_fill_value"]
            df[col] = df[col].fillna(fill_value)

    for col in categorical_features:
        if col in df.columns and df[col].isnull().any():
            if CONFIG["categorical_missing_strategy"] == "mode":
                fill_value = df[col].mode()[0]
            else:
                fill_value = CONFIG["categorical_fill_value"]
            df[col] = df[col].fillna(fill_value)

    return df


def undersample_data(df, label_col, ratio=5):
    """对多数类进行下采样"""
    pos_df = df[df[label_col] == 1]
    neg_df = df[df[label_col] == 0]
    n_pos = len(pos_df)
    n_neg = min(len(neg_df), n_pos * ratio)
    neg_sample = neg_df.sample(n_neg, random_state=42)
    balanced_df = pd.concat([pos_df, neg_sample], axis=0).sample(frac=1, random_state=42)
    return balanced_df


def calculate_detailed_metrics(y_true, y_pred_proba, threshold=0.5):
    """计算详细指标，包括每个类别的统计信息"""
    y_pred = (y_pred_proba >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    f1 = f1_score(y_true, y_pred)

    precision_neg = precision_score(y_true, y_pred, pos_label=0)
    recall_neg = recall_score(y_true, y_pred, pos_label=0)
    f1_neg = f1_score(y_true, y_pred, pos_label=0)

    precision_pos = precision_score(y_true, y_pred)
    recall_pos = recall_score(y_true, y_pred)
    f1_pos = f1_score(y_true, y_pred)

    count_pos = sum(y_true)
    count_neg = len(y_true) - count_pos

    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba)
    aucpr = auc(recall_curve, precision_curve)

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


def print_detailed_metrics(metrics, dataset_name):
    """打印详细的分类指标"""
    title = f"{dataset_name} 数据集指标"
    print("\n" + "=" * 70)
    print(title.center(70))
    print("=" * 70)

    print("\n整体指标:")
    print(f"  准确率: {metrics['overall']['accuracy']:.4f}")
    print(f"  ROC曲线下面积: {metrics['overall']['roc_auc']:.4f}")
    print(f"  PR曲线下面积: {metrics['overall']['auc_pr']:.4f}")
    print(f"  F1分数: {metrics['overall']['f1_score']:.4f}")
    print(f"  预测阈值: {metrics['threshold']:.4f}")

    print("\n正样本(少数类)指标:")
    print(f"  精确率: {metrics['positive']['precision']:.4f}")
    print(f"  召回率: {metrics['positive']['recall']:.4f}")
    print(f"  F1分数: {metrics['positive']['f1']:.4f}")
    print(f"  样本数: {metrics['positive']['support']}")

    print("\n负样本(多数类)指标:")
    print(f"  精确率: {metrics['negative']['precision']:.4f}")
    print(f"  召回率: {metrics['negative']['recall']:.4f}")
    print(f"  F1分数: {metrics['negative']['f1']:.4f}")
    print(f"  样本数: {metrics['negative']['support']}")

    print("=" * 70 + "\n")
    return metrics


def save_results(trainer, config, test_metrics, encoders, binner, results_dir=None):
    """将所有训练结果保存到文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if results_dir is None:
        results_dir = f"results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

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

    torch.save(trainer.best_model_state, os.path.join(results_dir, "best_model.pth"))
    torch.save(trainer.model.state_dict(), os.path.join(results_dir, "final_model.pth"))

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

    save_preprocessors(encoders, binner, results_dir)

    return results_dir


class DeepFMDataset(Dataset):
    def __init__(self, df, numeric_features, categorical_features, label_col,
                 encoders=None, binner=None, is_train=True):
        self.df = df.copy().reset_index(drop=True)
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.label_col = label_col
        self.is_train = is_train
        self.binner = binner

        self.df = handle_missing_values(self.df, numeric_features, categorical_features)

        if self.binner:
            if is_train:
                self.df[self.numeric_features] = self.binner.fit_transform(
                    self.df[self.numeric_features], CONFIG["bin_config"])
            else:
                self.df[self.numeric_features] = self.binner.transform(
                    self.df[self.numeric_features])

        self.categorical_encoders = {} if encoders is None else encoders
        self.categorical_dims = {}
        self._encode_categorical()

        self.numeric_data = self.df[self.numeric_features].values.astype(np.float32)
        self.categorical_data = self.df[[f"{col}_encoded" for col in self.categorical_features]].values.astype(np.int64)
        self.labels = self.df[label_col].values.astype(np.float32)

    def _encode_categorical(self):
        """编码分类特征"""
        for col in self.categorical_features:
            if col not in self.df.columns:
                continue

            if self.is_train:
                encoder = LabelEncoder()
                # 处理非字符串类型的分类值
                self.df[col] = self.df[col].astype(str)
                encoded = encoder.fit_transform(self.df[col])
                self.categorical_encoders[col] = encoder
                self.categorical_dims[col] = len(encoder.classes_)
            else:
                if col not in self.categorical_encoders:
                    raise ValueError(f"编码器不存在: {col}")
                encoder = self.categorical_encoders[col]
                # 处理训练集中未出现的类别
                self.df[col] = self.df[col].astype(str)
                encoded = []
                for val in self.df[col]:
                    if val in encoder.classes_:
                        encoded.append(encoder.transform([val])[0])
                    else:
                        encoded.append(len(encoder.classes_))  # 分配一个新索引
                encoded = np.array(encoded)
                # 更新类别数（如果有新类别）
                self.categorical_dims[col] = max(len(encoder.classes_), np.max(encoded) + 1)

            self.df[f"{col}_encoded"] = encoded

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        numeric = self.numeric_data[idx]
        categorical = self.categorical_data[idx]
        label = self.labels[idx]
        return torch.tensor(numeric), torch.tensor(categorical), torch.tensor(label)


class DeepFM(nn.Module):
    def __init__(self, numeric_dim, cat_dims, embed_dim=32, mlp_layers=[256, 128, 64], dropout=0.3):
        super(DeepFM, self).__init__()
        self.numeric_dim = numeric_dim
        self.cat_dims = cat_dims
        self.embed_dim = embed_dim

        # FM部分 - 一阶特征
        self.fm_first_order_num = nn.Linear(numeric_dim, 1)
        self.fm_first_order_cat = nn.ModuleList([
            nn.Embedding(num_embeddings=dim + 1, embedding_dim=1)  # +1 防止索引越界
            for dim in cat_dims
        ])

        # FM部分 - 二阶特征（嵌入层）
        self.fm_second_order_cat = nn.ModuleList([
            nn.Embedding(num_embeddings=dim + 1, embedding_dim=embed_dim)  # +1 防止索引越界
            for dim in cat_dims
        ])

        # MLP部分
        mlp_input_dim = numeric_dim + embed_dim * len(cat_dims)
        self.mlp = nn.Sequential()
        for i, hidden_dim in enumerate(mlp_layers):
            self.mlp.add_module(f"linear_{i}", nn.Linear(mlp_input_dim, hidden_dim))
            self.mlp.add_module(f"relu_{i}", nn.ReLU())
            self.mlp.add_module(f"dropout_{i}", nn.Dropout(dropout))
            mlp_input_dim = hidden_dim

        # 输出层
        self.output = nn.Linear(mlp_layers[-1] + 1, 1)  # +1 来自FM的输出
        self.sigmoid = nn.Sigmoid()

    def forward(self, numeric, categorical):
        # FM一阶部分
        first_order_num = self.fm_first_order_num(numeric)
        first_order_cat = []
        for i, emb in enumerate(self.fm_first_order_cat):
            first_order_cat.append(emb(categorical[:, i]))
        first_order_cat = torch.stack(first_order_cat, dim=1).sum(dim=1)
        fm_first_order = first_order_num + first_order_cat

        # FM二阶部分
        second_order_emb = []
        for i, emb in enumerate(self.fm_second_order_cat):
            second_order_emb.append(emb(categorical[:, i]))
        second_order_emb = torch.stack(second_order_emb, dim=1)  # [batch, n_cat, embed_dim]

        sum_square = torch.pow(torch.sum(second_order_emb, dim=1), 2)
        square_sum = torch.sum(torch.pow(second_order_emb, 2), dim=1)
        fm_second_order = 0.5 * torch.sum(sum_square - square_sum, dim=1, keepdim=True)

        # FM总输出
        fm_output = fm_first_order + fm_second_order

        # MLP部分
        flat_emb = second_order_emb.view(second_order_emb.size(0), -1)  # 展平嵌入向量
        mlp_input = torch.cat([numeric, flat_emb], dim=1)
        mlp_output = self.mlp(mlp_input)

        # 最终输出
        total_output = self.output(torch.cat([fm_output, mlp_output], dim=1))
        return self.sigmoid(total_output).squeeze(1)


class DeepFMTrainer:
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=CONFIG["learning_rate"],
            weight_decay=CONFIG["weight_decay"]
        )

        self.train_loss_history = []
        self.train_metrics_history = []
        self.val_metrics_history = []
        self.best_auc = 0
        self.best_model_state = None
        self.best_threshold = CONFIG["positive_threshold"]
        self.test_confusion_matrix = None

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        for numeric, categorical, labels in dataloader:
            numeric = numeric.to(self.device)
            categorical = categorical.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(numeric, categorical)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * numeric.size(0)
            all_preds.extend(outputs.cpu().detach().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(dataloader.dataset)
        metrics = calculate_detailed_metrics(np.array(all_labels), np.array(all_preds), self.best_threshold)
        return avg_loss, metrics

    def evaluate(self, dataloader, threshold=None):
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for numeric, categorical, labels in dataloader:
                numeric = numeric.to(self.device)
                categorical = categorical.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(numeric, categorical)
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        threshold = threshold if threshold is not None else self.best_threshold
        metrics = calculate_detailed_metrics(np.array(all_labels), np.array(all_preds), threshold)
        return metrics

    def train(self, epochs, train_loader, val_loader=None):
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 50)

            train_loss, train_metrics = self.train_epoch(train_loader)
            self.train_loss_history.append(train_loss)
            self.train_metrics_history.append(train_metrics)
            print(f"训练损失: {train_loss:.4f}")
            print_detailed_metrics(train_metrics, "训练集")

            if val_loader:
                val_metrics = self.evaluate(val_loader)
                self.val_metrics_history.append(val_metrics)
                print_detailed_metrics(val_metrics, "验证集")

                # 保存最佳模型
                if val_metrics["overall"]["auc_pr"] > self.best_auc:
                    self.best_auc = val_metrics["overall"]["auc_pr"]
                    self.best_model_state = self.model.state_dict()
                    # 调整最佳阈值（如果启用）
                    if CONFIG["threshold_tuning"]:
                        self.best_threshold = self.tune_threshold(val_loader)

        # 加载最佳模型
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)

        return self.evaluate(val_loader) if val_loader else train_metrics

    def tune_threshold(self, dataloader):
        """调整分类阈值以优化指定指标"""
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for numeric, categorical, labels in dataloader:
                numeric = numeric.to(self.device)
                categorical = categorical.to(self.device)
                outputs = self.model(numeric, categorical)
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        best_score = -1
        best_threshold = CONFIG["positive_threshold"]
        thresholds = np.arange(CONFIG["min_threshold"], CONFIG["max_threshold"], 0.01)

        for threshold in thresholds:
            y_pred = (all_preds >= threshold).astype(int)
            if CONFIG["optimize_for"] == "f1":
                score = f1_score(all_labels, y_pred)
            elif CONFIG["optimize_for"] == "precision":
                score = precision_score(all_labels, y_pred)
            else:  # recall
                score = recall_score(all_labels, y_pred)

            if score > best_score:
                best_score = score
                best_threshold = threshold

        return best_threshold


def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载数据
    print("\n加载数据...")
    data_folder = CONFIG["data_folder"]
    df = load_data_from_folder(data_folder)

    # 过滤特征
    required_cols = CONFIG["numeric_features"] + CONFIG["categorical_features"] + [CONFIG["label_col"]]
    df = df[required_cols].dropna(subset=[CONFIG["label_col"]])
    print(f"过滤后的数据包含 {len(df.columns) - 1} 列特征和标签")

    # 分割数据集
    print("\n分割数据集...")
    train_val_df, test_df = train_test_split(df, test_size=CONFIG["test_size"], random_state=42, stratify=df[CONFIG["label_col"]])
    train_df, val_df = train_test_split(train_val_df, test_size=CONFIG["val_size"], random_state=42, stratify=train_val_df[CONFIG["label_col"]])

    # 处理类别不平衡
    if CONFIG["imbalance_method"] in ['undersample', 'both']:
        print(f"对训练集进行下采样，负正样本比例为 {CONFIG['undersample_ratio']}:1")
        train_df = undersample_data(train_df, CONFIG["label_col"], CONFIG["undersample_ratio"])

    # 创建分箱器
    binner = FeatureBinner() if CONFIG["bin_config"] else None

    # 创建数据集
    print("\n创建数据集...")
    train_dataset = DeepFMDataset(
        train_df, CONFIG["numeric_features"], CONFIG["categorical_features"],
        CONFIG["label_col"], binner=binner, is_train=True
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

    # 获取分类特征维度
    cat_dims = [train_dataset.categorical_dims[col] for col in CONFIG["categorical_features"] if col in train_dataset.categorical_dims]
    numeric_dim = len(CONFIG["numeric_features"])

    # 初始化模型
    print("\n初始化模型...")
    model = DeepFM(
        numeric_dim=numeric_dim,
        cat_dims=cat_dims,
        embed_dim=CONFIG["embed_dim"],
        mlp_layers=CONFIG["mlp_layers"],
        dropout=CONFIG["dropout"]
    )

    # 初始化训练器
    trainer = DeepFMTrainer(model, device)

    # 开始训练
    print("\n开始训练...")
    print(f"开始训练，共 {CONFIG['epochs']} 个epoch，使用设备: {device}")
    test_metrics = trainer.train(epochs=CONFIG["epochs"], train_loader=train_loader, val_loader=val_loader)

    # 在测试集上评估
    print("\n在测试集上评估...")
    test_metrics = trainer.evaluate(test_loader, threshold=trainer.best_threshold)
    print_detailed_metrics(test_metrics, "测试集")

    # 计算混淆矩阵
    y_pred = (np.array([p for _, _, p in test_dataset]) >= trainer.best_threshold).astype(int)
    y_true = np.array([l for _, _, l in test_dataset])
    trainer.test_confusion_matrix = confusion_matrix(y_true, y_pred)

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
    # main()

    result = predict_new_data("./", "./results_20250731_011338", "./output_path")
    print("\n预测完成，部分结果:")
    print(result[["pred_prob", "pred_label"]].head())