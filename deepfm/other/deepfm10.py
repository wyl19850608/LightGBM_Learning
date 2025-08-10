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

        # 确保所有需要的特征都存在
        required_features = numeric_features + categorical_features + [label_col]
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
        self.labels = self.df[label_col].values.astype(np.float32)

    def _encode_categorical(self):
        """编码分类分类特征"""
        for col in self.categorical_features:
            if col not in self.df.columns:
                continue

            if self.is_train:
                encoder = LabelEncoder()
                # 处理非字符串类型的分类值
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
                    if mask.any():
                        self.df.loc[mask, col] = encoder.classes_[0]  # 用第一个类别替换未知值
                    self.df[f"{col}_encoded"] = encoder.transform(self.df[col])
                    self.categorical_dims[col] = len(encoder.classes_)
                else:
                    raise ValueError(f"编码器不存在: {col}，请先训练模型获取编码器")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return (
            self.numeric_data[idx],
            self.categorical_data[idx],
            self.labels[idx]
        )


class DeepFM(nn.Module):
    def __init__(self, numeric_dim, categorical_dims, embed_dim=32, mlp_layers=[256, 128, 64], dropout=0.3):
        super(DeepFM, self).__init__()
        self.numeric_dim = numeric_dim
        self.embed_dim = embed_dim

        # 一阶部分 - 线性层（使用与训练时一致的命名）
        self.fm_first_order_num = nn.Linear(numeric_dim, 1)
        self.fm_first_order_cat = nn.ModuleList()
        for dim in categorical_dims.values():
            self.fm_first_order_cat.append(nn.Embedding(dim, 1))

        # 二阶部分 - FM交互（使用与训练时一致的命名）
        self.fm_second_order_cat = nn.ModuleList()
        for dim in categorical_dims.values():
            self.fm_second_order_cat.append(nn.Embedding(dim, embed_dim))

        # 深度部分 - MLP（使用与训练时一致的结构和命名）
        mlp_input_dim = numeric_dim + len(categorical_dims) * embed_dim
        mlp_layers = [mlp_input_dim] + mlp_layers

        self.mlp = nn.Sequential()
        for i in range(1, len(mlp_layers)):
            self.mlp.add_module(f"linear_{i-1}", nn.Linear(mlp_layers[i-1], mlp_layers[i]))
            self.mlp.add_module(f"relu_{i-1}", nn.ReLU())
            self.mlp.add_module(f"dropout_{i-1}", nn.Dropout(dropout))

        # 输出层（使用与训练时一致的命名）
        self.output = nn.Linear(mlp_layers[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, numeric, categorical):
        # 一阶特征 - 数值特征
        first_order_num = self.fm_first_order_num(numeric)

        # 一阶特征 - 分类特征
        first_order_cat = []
        for i, emb in enumerate(self.fm_first_order_cat):
            first_order_cat.append(emb(categorical[:, i]))
        first_order_cat = torch.cat(first_order_cat, dim=1).sum(dim=1, keepdim=True)

        # 一阶总输出
        first_order_out = first_order_num + first_order_cat

        # 二阶特征交互
        second_order = []
        for i, emb in enumerate(self.fm_second_order_cat):
            second_order.append(emb(categorical[:, i]))
        second_order = torch.stack(second_order, dim=1)  # [batch_size, num_cats, embed_dim]

        # FM二阶交互计算
        sum_square = torch.pow(torch.sum(second_order, dim=1), 2)
        square_sum = torch.sum(torch.pow(second_order, 2), dim=1)
        fm_out = 0.5 * torch.sum(torch.sub(sum_square, square_sum), dim=1, keepdim=True)

        # 深度部分
        flat_embeddings = second_order.view(second_order.size(0), -1)  # 展平嵌入
        deep_input = torch.cat([numeric, flat_embeddings], dim=1)
        deep_out = self.mlp(deep_input)
        deep_out = self.output(deep_out)

        # 合并所有部分
        total = first_order_out + fm_out + deep_out
        return self.sigmoid(total).squeeze(1)


class DeepFMTrainer:
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)

        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=CONFIG["learning_rate"],
            weight_decay=CONFIG["weight_decay"]
        )

        self.train_loss_history = []
        self.train_metrics_history = []
        self.val_metrics_history = []
        self.best_auc = -1
        self.best_model_state = None
        self.best_threshold = 0.5
        self.test_confusion_matrix = None

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        for numeric, categorical, labels in train_loader:
            numeric = numeric.to(self.device)
            categorical = categorical.to(self.device)
            labels = labels.to(self.device)

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

    def evaluate(self, data_loader, threshold=0.5):
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for numeric, categorical, labels in data_loader:
                numeric = numeric.to(self.device)
                categorical = categorical.to(self.device)

                outputs = self.model(numeric, categorical)
                all_preds.extend(outputs.detach().cpu().numpy())
                all_labels.extend(labels.detach().cpu().numpy())

        metrics = calculate_detailed_metrics(all_labels, all_preds, threshold)
        return metrics, all_preds, all_labels

    def find_best_threshold(self, val_loader):
        """在验证集上寻找最佳阈值"""
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for numeric, categorical, labels in val_loader:
                numeric = numeric.to(self.device)
                categorical = categorical.to(self.device)
                outputs = self.model(numeric, categorical)
                all_preds.extend(outputs.detach().cpu().numpy())
                all_labels.extend(labels.detach().cpu().numpy())

        best_score = -1
        best_threshold = 0.5

        # 尝试不同阈值
        for threshold in np.arange(0.1, 0.91, 0.01):
            y_pred = (np.array(all_preds) >= threshold).astype(int)
            if CONFIG["optimize_for"] == "f1":
                score = f1_score(all_labels, y_pred)
            elif CONFIG["optimize_for"] == "auc_pr":
                precision, recall, _ = precision_recall_curve(all_labels, all_preds)
                score = auc(recall, precision)
            else:  # 默认使用f1
                score = f1_score(all_labels, y_pred)

            if score > best_score:
                best_score = score
                best_threshold = threshold

        return best_threshold

    def train(self, train_loader, val_loader, epochs):
        for epoch in range(epochs):
            start_time = time.time()

            # 训练
            train_loss, train_metrics = self.train_epoch(train_loader)
            self.train_loss_history.append(train_loss)
            self.train_metrics_history.append(train_metrics)

            # 验证
            val_metrics, _, _ = self.evaluate(val_loader, self.best_threshold)
            self.val_metrics_history.append(val_metrics)

            # 打印 epoch 信息
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"训练损失: {train_loss:.4f} | 训练F1: {train_metrics['overall']['f1_score']:.4f}")
            print(f"验证F1: {val_metrics['overall']['f1_score']:.4f} | 验证AUC-PR: {val_metrics['overall']['auc_pr']:.4f}")
            print(f"耗时: {time.time() - start_time:.2f}秒\n")

            # 更新最佳模型
            if val_metrics['overall']['auc_pr'] > self.best_auc:
                self.best_auc = val_metrics['overall']['auc_pr']
                self.best_model_state = self.model.state_dict()

                # 如果启用阈值调优，在最佳模型上重新计算阈值
                if CONFIG["threshold_tuning"]:
                    self.best_threshold = self.find_best_threshold(val_loader)
                    print(f"更新最佳阈值: {self.best_threshold:.4f}")

        # 加载最佳模型
        self.model.load_state_dict(self.best_model_state)
        return self


def train_model(data_folder, results_dir=None):
    """训练DeepFM模型的主函数"""
    # 1. 加载数据
    print("加载数据...")
    df = load_data_from_folder(data_folder)

    # 2. 数据预处理与划分
    print("数据预处理...")
    # 检查标签列是否存在
    if CONFIG["label_col"] not in df.columns:
        raise ValueError(f"标签列 {CONFIG['label_col']} 不存在于数据中")

    # 划分训练集和测试集
    train_df, test_df = train_test_split(
        df, test_size=CONFIG["test_size"], random_state=42, stratify=df[CONFIG["label_col"]]
    )

    # 从训练集中划分验证集
    train_df, val_df = train_test_split(
        train_df, test_size=CONFIG["val_size"], random_state=42, stratify=train_df[CONFIG["label_col"]]
    )

    # 处理类别不平衡
    if CONFIG["imbalance_method"] in ["undersample", "both"]:
        print(f"对训练集进行下采样，正负样本比例 {CONFIG['undersample_ratio']}:1")
        train_df = undersample_data(train_df, CONFIG["label_col"], CONFIG["undersample_ratio"])

    # 3. 创建数据集和数据加载器
    print("创建数据集...")
    binner = FeatureBinner() if CONFIG["bin_config"] else None

    # 训练集（会生成编码器）
    train_dataset = DeepFMDataset(
        train_df, CONFIG["numeric_features"], CONFIG["categorical_features"],
        CONFIG["label_col"], encoders=None, binner=binner, is_train=True
    )
    encoders = train_dataset.categorical_encoders

    # 验证集和测试集（使用训练好的编码器）
    val_dataset = DeepFMDataset(
        val_df, CONFIG["numeric_features"], CONFIG["categorical_features"],
        CONFIG["label_col"], encoders=encoders, binner=binner, is_train=False
    )

    test_dataset = DeepFMDataset(
        test_df, CONFIG["numeric_features"], CONFIG["categorical_features"],
        CONFIG["label_col"], encoders=encoders, binner=binner, is_train=False
    )

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2)

    # 4. 初始化模型
    print("初始化模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    model = DeepFM(
        numeric_dim=len(CONFIG["numeric_features"]),
        categorical_dims=train_dataset.categorical_dims,
        embed_dim=CONFIG["embed_dim"],
        mlp_layers=CONFIG["mlp_layers"],
        dropout=CONFIG["dropout"]
    )

    # 5. 训练模型
    print("开始训练...")
    trainer = DeepFMTrainer(model, device)
    trainer.train(train_loader, val_loader, epochs=CONFIG["epochs"])

    # 6. 在测试集上评估
    print("在测试集上评估...")
    test_metrics, test_preds, test_labels = trainer.evaluate(
        test_loader, threshold=trainer.best_threshold
    )
    print_detailed_metrics(test_metrics, "测试")

    # 保存混淆矩阵
    y_pred = (np.array(test_preds) >= trainer.best_threshold).astype(int)
    trainer.test_confusion_matrix = confusion_matrix(test_labels, y_pred)

    # 7. 保存结果
    print("保存结果...")
    results_dir = save_results(trainer, CONFIG, test_metrics, encoders, binner, results_dir)
    print(f"所有结果已保存至: {results_dir}")

    return results_dir


def load_adjusted_model(model, model_path, device):
    """调整状态字典键名以匹配模型结构"""
    # 加载原始状态字典
    state_dict = torch.load(model_path, map_location=device)

    # 创建新的状态字典，调整键名以匹配当前模型
    new_state_dict = {}

    # 映射旧键到新键
    key_mapping = {
        # 一阶数值特征
        "first_order.weight": "fm_first_order_num.weight",
        "first_order.bias": "fm_first_order_num.bias",

        # MLP层
        "mlp.linear_1.weight": "mlp.linear_0.weight",
        "mlp.linear_1.bias": "mlp.linear_0.bias",
        "mlp.linear_2.weight": "mlp.linear_1.weight",
        "mlp.linear_2.bias": "mlp.linear_1.bias",
        "mlp.linear_3.weight": "mlp.linear_2.weight",
        "mlp.linear_3.bias": "mlp.linear_2.bias",
        "mlp.output.weight": "output.weight",
        "mlp.output.bias": "output.bias"
    }

    # 处理嵌入层
    i = 0
    while True:
        old_emb_key = f"embeddings.{i}.weight"
        new_emb_key1 = f"fm_first_order_cat.{i}.weight"  # 一阶分类特征
        new_emb_key2 = f"fm_second_order_cat.{i}.weight"  # 二阶分类特征

        if old_emb_key in state_dict:
            # 如果存在一阶分类特征的权重，则使用
            if new_emb_key1 in state_dict:
                new_state_dict[new_emb_key1] = state_dict[new_emb_key1]
            else:
                # 否则使用嵌入层权重作为替代
                new_state_dict[new_emb_key1] = state_dict[old_emb_key][:, :1]  # 取第一个维度

            # 处理二阶分类特征
            if new_emb_key2 in state_dict:
                new_state_dict[new_emb_key2] = state_dict[new_emb_key2]
            else:
                new_state_dict[new_emb_key2] = state_dict[old_emb_key]

            i += 1
        else:
            break

    # 处理其他键
    for old_key, new_key in key_mapping.items():
        if old_key in state_dict:
            new_state_dict[new_key] = state_dict[old_key]
        elif new_key in state_dict:
            new_state_dict[new_key] = state_dict[new_key]

    # 加载调整后的状态字典，设置strict=False以忽略不匹配的键
    model.load_state_dict(new_state_dict, strict=False)
    return model


def predict_new_data(new_data_path, model_dir, output_path=None):
    """使用训练好的模型预测新数据"""
    # 1. 加载模型配置
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    # 2. 加载预处理工具
    encoders, binner = load_preprocessors(model_dir)
    print("预处理工具加载完成")

    # 3. 加载新数据
    print("加载新数据...")
    new_df = load_data_from_folder(new_data_path)

    # 4. 准备数据集
    # 创建虚拟数据来获取分类特征维度（解决预测时无标签的问题）
    dummy_data = {}
    # 添加数值特征
    for col in config["numeric_features"]:
        dummy_data[col] = [0.0]
    # 添加分类特征
    for col in config["categorical_features"]:
        dummy_data[col] = ["dummy"]
    # 添加标签列（预测时不会用到）
    dummy_data[config["label_col"]] = [0]

    dummy_df = pd.DataFrame(dummy_data)
    dummy_dataset = DeepFMDataset(
        dummy_df, config["numeric_features"], config["categorical_features"],
        config["label_col"], encoders=encoders, binner=binner, is_train=False
    )

    # 5. 准备预测数据（确保包含所有必要特征）
    required_features = config["numeric_features"] + config["categorical_features"]
    missing_features = [f for f in required_features if f not in new_df.columns]
    if missing_features:
        raise ValueError(f"新数据中缺少必要特征: {missing_features}")

    # 添加临时标签列（预测时不会用到）
    if config["label_col"] not in new_df.columns:
        new_df[config["label_col"]] = 0

    predict_dataset = DeepFMDataset(
        new_df, config["numeric_features"], config["categorical_features"],
        config["label_col"], encoders=encoders, binner=binner, is_train=False
    )

    predict_loader = DataLoader(predict_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=2)

    # 6. 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    model = DeepFM(
        numeric_dim=len(config["numeric_features"]),
        categorical_dims=dummy_dataset.categorical_dims,
        embed_dim=config["embed_dim"],
        mlp_layers=config["mlp_layers"],
        dropout=config["dropout"]
    )

    model_path = os.path.join(model_dir, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    # 使用调整后的模型加载函数
    model = load_adjusted_model(model, model_path, device)
    model.to(device)
    model.eval()
    print("模型加载完成")

    # 7. 进行预测
    print("开始预测...")
    all_preds = []

    with torch.no_grad():
        for numeric, categorical, _ in predict_loader:
            numeric = numeric.to(device)
            categorical = categorical.to(device)

            outputs = model(numeric, categorical)
            all_preds.extend(outputs.detach().cpu().numpy())

    # 8. 保存预测结果
    new_df["prediction_prob"] = all_preds
    new_df["prediction"] = (np.array(all_preds) >= config["positive_threshold"]).astype(int)

    if output_path is None:
        output_path = os.path.join(os.getcwd(), "predictions")
    os.makedirs(output_path, exist_ok=True)

    output_file = os.path.join(output_path, f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    new_df.to_csv(output_file, index=False)
    print(f"预测结果已保存至: {output_file}")

    return new_df


if __name__ == "__main__":
    # 训练模型示例
    results_dir = train_model("./")

    # 预测新数据示例
    # result = predict_new_data("./", "./results_20250731_011338", "./output_path")
