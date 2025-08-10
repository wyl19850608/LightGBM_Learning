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
        """拟合并并转换数据"""
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
        'last_10d_coupon_cnt', 'last_op_time', 'last_pboc_cx_dt',
        'querycncq03wwlw', 'querycncebookswwflaged', 'cdaccapwwwww',
        'gvaccapwwwwww', 'rvclaapwwww', 'rvblsapwwwww', 'rvnbmwww',
        'rvnbawpwwwww', 'rvapsappywww', 'rvapmapwwwwww', 'rvapaapwwwww',
        'repay_date_days', 'als_m1_id_bank_allnum', 'als_m1_id_nbank_allnum',
        'als_m3_id_bank_allnum', 'als_m3_id_nbank_allnum', 'br_modle_score',
        'query_times_bank_90d', 'query_imes_cfc_90d', 'risk_ms9_model_score',
        'loss_model_group_v3', 'yls_cust_typev2', 'risk_ms11_1_model_score',
        'standard_score_group_v6_1', 'latest_login_days', 'total_loan_cnt',
        'total_loan_amt', 'last_1y_cps_num', 'last_month_expire_coupon_cnt',
        'number_of_gold_be_used', 'cuir_mon_use_gold_cnt', 'late_1mon_use_gold_cnt',
        'late_3mon_use_gold_cnt', 'late_6mon_use_gold_cnt', 'cur_mon_use_points_cnt',
        'late_1mon_use_points_cnt', 'late_3mon_use_points_cnt', 'late_6mon_use_points_cnt',
        'cur_mon_poins_sign_cnt', 'late_imon_points_sign_cnt', 'late_3mon_points_sign_cnt',
        'late_6mon_points_sign_cnt', 'cur_mon_points_luck_gift_cnt',
        'late_1mon_points_luck_gift_cnt', 'late_3mon_points_luck_gift_cnt',
        'late_6mon_points_luck_gift_cnt', 'cur_mon_cps_order_types',
        'late_imon_cps_order_types', 'late_3mon_cps_order_types',
        'late_6mon_cps_order_types', 'last_mon_game_accum_click_cnt',
        'last_mon_game_accum_expo_cnt', 'last_mon_read_accum_click_cnt',
        'last_mon_read_accum_expo_cnt', 'cur_yili_vipterms', 'cur_juhui_vip_terms',
        'last_login_ayh_time', 'last_login_app_time', 'last_uas_reject_time',
        'last_30d_fq_cs', 'last_30d_tx_cs', 'last_30d_login_ayh_days',
        'last_5d_lin_e_cnt', 'last_10d_lin_e_cnt', 'last_5d_gu_e_cnt',
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
        'cust_ivr_ansr_times', 'cust_ivr_ansr_dur', 'cust_ivr_call_times_mon',
        'cust_ivr_ansr_times_mon', 'cust_ivr_ansr_dur_mon'
    ],
    "bin_config": {
        'age': {'n_bins': 20, 'strategy': 'quantile'},
        'last_30d_tel_succ_cs': {'n_bins': 20, 'strategy': 'quantile'},
        'risk_ms11_1_model_score': {'n_bins': 20, 'strategy': 'quantile'},
        'standard_score_group_v6_1': {'n_bins': 20, 'strategy': 'quantile'},
        'loss_model_ggroup_v3': {'n_bins': 15, 'strategy': 'quantile'},
        'ayht10_all_respond_score': {'n_bins': 15, 'strategy': 'quantile'},
        'xyl_model_107': {'n_bins': 15, 'strategy': 'quantile'},
        'avg_limuse_rate': {'n_bins': 15, 'strategy': 'quantile'},
        'lim_use_rate': {'n_bins': 15, 'strategy': 'quantile'},
        'last_month_expire_coupon_cnt': {'n_bins': 10, 'strategy': 'quantile'},
        'number_of_gold_be_used': {'n_bins': 10, 'strategy': 'quantile'},
        'last_10d_lin_e_cnt': {'n_bins': 10, 'strategy': 'quantile'},
        'last_10d_gu_e_cnt': {'n_bins': 10, 'strategy': 'quantile'},
        'call_anss_score_t10': {'n_bins': 10, 'strategy': 'quantile'},
        'pril_bal': {'n_bins': 10, 'strategy': 'quantile'},
        'crdt_lim_yx': {'n_bins': 10, 'strategy': 'quantile'},
        'zaidai_ctrl_rate': {'n_bins': 10, 'strategy': 'quantile'},
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
    """根据配置处理缺失缺失值"""
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
    # 修复：将列表转换为numpy数组
    y_pred_proba = np.array(y_pred_proba)
    # 如果是二维数组，取第一列（适用于二分类问题）
    if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] > 1:
        y_pred_proba = y_pred_proba[:, 0]

    y_pred = (y_pred_proba >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    try:
        roc_auc = roc_auc_score(y_true, y_pred_proba)
    except ValueError:
        roc_auc = 0.0  # 当只有一个类别时AUC无法计算

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
    def __init__(self, df, numeric_features, categorical_features, label_col=None,
                 encoders=None, binner=None, is_train=True):
        self.df = df.copy().reset_index(drop=True)
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.label_col = label_col
        self.is_train = is_train
        self.binner = binner

        # 确保所有需要的特征都存在
        required_features = numeric_features + categorical_features
        if label_col:
            required_features += [label_col]
        missing_features = [f for f in required_features if f not in self.df.columns]
        if missing_features:
            raise ValueError(f"数据中缺少必要特征: {missing_features}")

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

        # 处理标签（预测时可能没有标签）
        if label_col and label_col in self.df.columns:
            self.labels = self.df[label_col].values.astype(np.float32)
        else:
            self.labels = None

    def _encode_categorical(self):
        """编码分类特征"""
        for col in self.categorical_features:
            if col not in self.df.columns:
                continue

            if self.is_train:
                encoder = LabelEncoder()
                # 处理非字符串类型的分类特征
                self.df[col] = self.df[col].astype(str)
                self.df[f"{col}_encoded"] = encoder.fit_transform(self.df[col])
                self.categorical_encoders[col] = encoder
                self.categorical_dims[col] = len(encoder.classes_)
            else:
                if col in self.categorical_encoders:
                    encoder = self.categorical_encoders[col]
                    # 处理训练集中未见过的类别
                    self.df[col] = self.df[col].astype(str)
                    mask = ~self.df[col].isin(encoder.classes_)
                    self.df.loc[mask, col] = encoder.classes_[0]  # 用第一个类别替换未知类别
                    self.df[f"{col}_encoded"] = encoder.transform(self.df[col])
                    self.categorical_dims[col] = len(encoder.classes_)
                else:
                    raise ValueError(f"编码器不存在: {col}，请先在训练集上拟合编码器")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        result = {
            'numeric': torch.tensor(self.numeric_data[idx], dtype=torch.float32),
            'categorical': torch.tensor(self.categorical_data[idx], dtype=torch.long)
        }
        if self.labels is not None:
            result['label'] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return result


class FactorizationMachine(nn.Module):
    """ factorization Machine 部分 """
    def __init__(self, input_dim, latent_dim):
        super(FactorizationMachine, self).__init__()
        self.w0 = nn.Parameter(torch.zeros(1))
        self.w = nn.Parameter(torch.randn(input_dim))
        self.v = nn.Parameter(torch.randn(input_dim, latent_dim))

    def forward(self, x):
        # 线性部分
        linear_part = self.w0 + torch.matmul(x, self.w)

        # 交互部分
        square_of_sum = torch.matmul(x, self.v).pow(2).sum(1, keepdim=True)
        sum_of_square = torch.matmul(x.pow(2), self.v.pow(2)).sum(1, keepdim=True)
        interaction_part = 0.5 * (square_of_sum - sum_of_square).squeeze()

        return linear_part + interaction_part


class MLP(nn.Module):
    """ 多层感知机部分 """
    def __init__(self, input_dim, layers, dropout=0.3):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for output_dim in layers:
            self.layers.append(nn.Linear(input_dim, output_dim))
            self.layers.append(nn.BatchNorm1d(output_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout))
            input_dim = output_dim

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DeepFM(nn.Module):
    """ DeepFM 模型 """
    def __init__(self, numeric_dim, categorical_dims, embed_dim, mlp_layers, dropout=0.3):
        super(DeepFM, self).__init__()

        # 嵌入层
        self.embeddings = nn.ModuleList()
        for dim in categorical_dims.values():
            self.embeddings.append(nn.Embedding(dim, embed_dim))

        # FM部分输入维度：数值特征维度 + 分类特征数量
        fm_input_dim = numeric_dim + len(categorical_dims)
        self.fm = FactorizationMachine(fm_input_dim, embed_dim)

        # MLP部分输入维度：数值特征维度 + 分类特征数量 * 嵌入维度
        mlp_input_dim = numeric_dim + len(categorical_dims) * embed_dim
        self.mlp = MLP(mlp_input_dim, mlp_layers, dropout)

        # 输出层
        self.output_layer = nn.Linear(mlp_layers[-1] + 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, numeric, categorical):
        # 处理分类特征的嵌入
        embeds = []
        for i, embed_layer in enumerate(self.embeddings):
            embeds.append(embed_layer(categorical[:, i]))
        embed_cat = torch.cat(embeds, dim=1)

        # FM部分的输入：数值特征 + 嵌入向量的和
        fm_categorical = torch.stack(embeds, dim=1).sum(dim=2)  # 每个分类特征的嵌入求和
        fm_input = torch.cat([numeric, fm_categorical], dim=1)
        fm_output = self.fm(fm_input).unsqueeze(1)

        # MLP部分的输入：数值特征 + 展平的嵌入向量
        mlp_input = torch.cat([numeric, embed_cat], dim=1)
        mlp_output = self.mlp(mlp_input)

        # 合并FM和MLP的输出
        combined = torch.cat([fm_output, mlp_output], dim=1)
        output = self.sigmoid(self.output_layer(combined)).squeeze()

        return output


class Trainer:
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=CONFIG["learning_rate"],
            weight_decay=CONFIG["weight_decay"]
        )

        # 训练历史记录
        self.train_loss_history = []
        self.train_metrics_history = []
        self.val_loss_history = []
        self.val_metrics_history = []

        # 最佳模型相关
        self.best_auc = -1
        self.best_model_state = None
        self.best_threshold = CONFIG["positive_threshold"]
        self.test_confusion_matrix = None

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for batch in train_loader:
            numeric = batch['numeric'].to(self.device)
            categorical = batch['categorical'].to(self.device)
            labels = batch['label'].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(numeric, categorical)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * numeric.size(0)
            all_preds.extend(outputs.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

        avg_loss = total_loss / len(train_loader.dataset)
        metrics = calculate_detailed_metrics(all_labels, all_preds, self.best_threshold)

        return avg_loss, metrics

    def evaluate(self, data_loader, threshold=None):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in data_loader:
                numeric = batch['numeric'].to(self.device)
                categorical = batch['categorical'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(numeric, categorical)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * numeric.size(0)
                all_preds.extend(outputs.detach().cpu().numpy())
                all_labels.extend(labels.detach().cpu().numpy())

        avg_loss = total_loss / len(data_loader.dataset)
        threshold = self.best_threshold if threshold is None else threshold
        metrics = calculate_detailed_metrics(all_labels, all_preds, threshold)

        # 计算混淆矩阵（仅用于测试集）
        y_pred = (np.array(all_preds) >= threshold).astype(int)
        cm = confusion_matrix(all_labels, y_pred)

        return avg_loss, metrics, cm, all_preds, all_labels

    def train(self, train_loader, val_loader, epochs=10):
        print(f"开始训练，共 {epochs} 个 epoch...")

        for epoch in range(epochs):
            start_time = time.time()

            # 训练一个epoch
            train_loss, train_metrics = self.train_epoch(train_loader)
            self.train_loss_history.append(train_loss)
            self.train_metrics_history.append(train_metrics)

            # 在验证集上评估
            val_loss, val_metrics, _, _, _ = self.evaluate(val_loader)
            self.val_loss_history.append(val_loss)
            self.val_metrics_history.append(val_metrics)

            # 保存最佳模型
            current_auc = val_metrics['overall']['auc_pr']
            if current_auc > self.best_auc:
                self.best_auc = current_auc
                self.best_model_state = self.model.state_dict()
                # 如果启用阈值调优，在验证集上寻找最佳阈值
                if CONFIG["threshold_tuning"]:
                    _, _, _, preds, labels = self.evaluate(val_loader, threshold=0.5)
                    self.best_threshold = self.find_optimal_threshold(labels, preds)

            # 打印epoch信息
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{epochs} - 时间: {epoch_time:.2f}s")
            print(f"训练损失: {train_loss:.4f} - 验证损失: {val_loss:.4f}")
            print(f"训练F1: {train_metrics['overall']['f1_score']:.4f} - 验证F1: {val_metrics['overall']['f1_score']:.4f}")
            print(f"训练AUC-PR: {train_metrics['overall']['auc_pr']:.4f} - 验证AUC-PR: {val_metrics['overall']['auc_pr']:.4f}")
            print("-" * 70)

        print("训练完成！")
        return self

    def find_optimal_threshold(self, y_true, y_pred_proba):
        """寻找最优阈值以优化指定指标"""
        thresholds = np.arange(CONFIG["min_threshold"], CONFIG["max_threshold"], 0.01)
        scores = []

        for threshold in thresholds:
            y_pred = (np.array(y_pred_proba) >= threshold).astype(int)
            if CONFIG["optimize_for"] == "f1":
                score = f1_score(y_true, y_pred)
            elif CONFIG["optimize_for"] == "precision":
                score = precision_score(y_true, y_pred)
            elif CONFIG["optimize_for"] == "recall":
                score = recall_score(y_true, y_pred)
            else:
                score = f1_score(y_true, y_pred)
            scores.append(score)

        best_idx = np.argmax(scores)
        return thresholds[best_idx]


def train_model(data_folder):
    """主函数：加载数据、预处理、训练模型并保存结果"""
    # 加载数据
    print("加载数据...")
    df = load_data_from_folder(data_folder)

    # 数据预处理
    print("数据预处理...")
    # 分割训练集和测试集
    train_df, test_df = train_test_split(
        df,
        test_size=CONFIG["test_size"],
        random_state=42,
        stratify=df[CONFIG["label_col"]]
    )

    # 分割验证集
    train_df, val_df = train_test_split(
        train_df,
        test_size=CONFIG["val_size"],
        random_state=42,
        stratify=train_df[CONFIG["label_col"]]
    )

    # 处理类别不平衡
    if CONFIG["imbalance_method"] in ["undersample", "both"]:
        print(f"对训练集进行下采样，正负样本比例 {CONFIG['undersample_ratio']}:1")
        train_df = undersample_data(
            train_df,
            CONFIG["label_col"],
            ratio=CONFIG["undersample_ratio"]
        )

    # 创建分箱器
    binner = FeatureBinner() if CONFIG["bin_config"] else None

    # 创建数据集
    print("创建数据集...")
    train_dataset = DeepFMDataset(
        train_df,
        numeric_features=CONFIG["numeric_features"],
        categorical_features=CONFIG["categorical_features"],
        label_col=CONFIG["label_col"],
        binner=binner,
        is_train=True
    )

    val_dataset = DeepFMDataset(
        val_df,
        numeric_features=CONFIG["numeric_features"],
        categorical_features=CONFIG["categorical_features"],
        label_col=CONFIG["label_col"],
        encoders=train_dataset.categorical_encoders,
        binner=binner,
        is_train=False
    )

    test_dataset = DeepFMDataset(
        test_df,
        numeric_features=CONFIG["numeric_features"],
        categorical_features=CONFIG["categorical_features"],
        label_col=CONFIG["label_col"],
        encoders=train_dataset.categorical_encoders,
        binner=binner,
        is_train=False
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=4
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=4
    )

    # 初始化模型
    print("初始化模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    model = DeepFM(
        numeric_dim=len(CONFIG["numeric_features"]),
        categorical_dims=train_dataset.categorical_dims,
        embed_dim=CONFIG["embed_dim"],
        mlp_layers=CONFIG["mlp_layers"],
        dropout=CONFIG["dropout"]
    )

    # 初始化训练器
    trainer = Trainer(model, device)

    # 开始训练
    print("开始训练...")
    trainer.train(train_loader, val_loader, epochs=CONFIG["epochs"])

    # 加载最佳模型并在测试集上评估
    print("在测试集上评估最佳模型...")
    trainer.model.load_state_dict(trainer.best_model_state)
    test_loss, test_metrics, test_cm, _, _ = trainer.evaluate(test_loader, trainer.best_threshold)
    trainer.test_confusion_matrix = test_cm
    print_detailed_metrics(test_metrics, "测试集")

    # 保存结果
    print("保存结果...")
    results_dir = save_results(
        trainer,
        CONFIG,
        test_metrics,
        train_dataset.categorical_encoders,
        binner
    )

    print(f"所有结果已保存至: {results_dir}")
    return results_dir


def predict_new_data(new_data, model_dir, threshold=None):
    """
    对新数据进行预测

    参数:
    new_data: DataFrame，包含需要预测的新数据
    model_dir: 字符串，模型保存目录
    threshold: 预测阈值，None则使用训练时确定的最佳阈值

    返回:
    DataFrame，包含原始数据和预测结果
    """
    # 加载配置
    config_path = os.path.join(model_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        config = CONFIG  # 使用默认配置

    # 加载预处理工具
    encoders, binner = load_preprocessors(model_dir)

    # 加载测试结果以获取最佳阈值
    test_results_path = os.path.join(model_dir, "test_results.json")
    if os.path.exists(test_results_path):
        with open(test_results_path, "r") as f:
            test_results = json.load(f)
        best_threshold = test_results["best_threshold"]
    else:
        best_threshold = config["positive_threshold"]

    # 使用指定阈值或最佳阈值
    threshold = threshold if threshold is not None else best_threshold

    # 创建数据集（注意：预测时is_train=False，且可能没有标签）
    dataset = DeepFMDataset(
        new_data,
        numeric_features=config["numeric_features"],
        categorical_features=config["categorical_features"],
        label_col=None,  # 预测时不需要标签
        encoders=encoders,
        binner=binner,
        is_train=False
    )

    # 创建数据加载器
    data_loader = DataLoader(
        dataset,
        batch_size=config.get("batch_size", 256),
        shuffle=False,
        num_workers=4
    )

    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepFM(
        numeric_dim=len(config["numeric_features"]),
        categorical_dims=dataset.categorical_dims,
        embed_dim=config.get("embed_dim", 32),
        mlp_layers=config.get("mlp_layers", [256, 128, 64]),
        dropout=config.get("dropout", 0.3)
    )

    # 加载最佳模型权重
    model_path = os.path.join(model_dir, "best_model.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, "final_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 进行预测
    all_preds = []
    with torch.no_grad():
        for batch in data_loader:
            numeric = batch['numeric'].to(device)
            categorical = batch['categorical'].to(device)

            outputs = model(numeric, categorical)
            all_preds.extend(outputs.detach().cpu().numpy())

    # 生成预测结果
    predictions = np.array(all_preds)
    predicted_labels = (predictions >= threshold).astype(int)

    # 将结果添加到原始数据中
    result_df = new_data.copy()
    result_df["pred_probability"] = predictions
    result_df["pred_label"] = predicted_labels
    result_df["threshold_used"] = threshold

    return result_df


def load_and_predict(new_data_path, model_dir, output_path=None, threshold=None):
    """
    加载新数据文件并进行预测

    参数:
    new_data_path: 新数据文件路径（CSV）
    model_dir: 模型保存目录
    output_path: 预测结果保存路径，None则不保存
    threshold: 预测阈值

    返回:
    DataFrame，包含预测结果
    """
    # 加载新数据
    print(f"加载新数据: {new_data_path}")
    new_df = pd.read_csv(new_data_path)

    # 检查是否有features_list配置，用于列名映射
    config_path = os.path.join(model_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        if 'features_list' in config and len(new_df.columns) == len(config['features_list']):
            new_df.columns = config['features_list']

    # 进行预测
    print("开始预测...")
    result_df = predict_new_data(new_df, model_dir, threshold)

    # 保存结果
    if output_path:
        result_df.to_csv(output_path, index=False)
        print(f"预测结果已保存至: {output_path}")
    else:
        print("预测完成")

    return result_df


if __name__ == "__main__":
    # 训练模型
    # results_dir = train_model("./")

    # 示例：使用训练好的模型进行预测
    # 请将下面的路径替换为实际的模型目录和新数据路径
    # model_directory = "results_20231015_143022"  # 替换为实际的模型目录
    # new_data_file = "new_data.csv"  # 替换为新数据文件路径
    # output_file = "prediction_results.csv"  # 预测结果保存路径
    result = predict_new_data("./", "./results_20250731_030409", "./output_path")
    pass
