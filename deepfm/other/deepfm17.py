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
                             precision_score, recall_score, precision_recall_curve, auc)
import time
from datetime import datetime
import json
import pickle
import warnings
import gc
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
    "bin_config": {'age': {'n_bins': 20, 'strategy': 'quantile'}},
    "features_list": ["user_id","unique_id","gender_cd","residence_city_cd","house_hold_type","marital_stattus_cd","edu_deg_cd","m_income","income_ind","cust_ty","custtypes_01","yls_cust_type","xyl_tag","last_fq_dt","last_fq_txk_amt","last_tx_dt","last_tx_amt","last_30d_login_app_days","last_30d_push_touch_times","last_30d_sms_send_succ_cs","last_30d_tel_sCC_cs","last_5d_coupon_cnt","last_10d_coupon_cnt","last_op_time","last_pboc_cx_dt","успсq03nwww.","querycncq03wwlw","querycncebookswwflaged","сdaccapwwwww.","гvaccapwwwwww.","гуграфиям","густармим","rvclaapwwww.","rvblsapwwwww.","rvnbmwww.","rvnbawpwwwww.","rvapsappywww.","rvapmapwwwwww.","rvapaapwwwww.","repay_date_days","als_m1_id_bank_allnum","als_m1_id_nbank_allnum","als_m3_id_bank_allnum","als_m3_id_nbank_allnum","br_modle_score","query_times_bank_90d","query_imes_cfc_90d","risk_ms9_model_score","loss_model_group_v3","yls_cust_typev2","latest_login_days","total_loan_cnt","total_loan_amt","last_1y_cps_num","cuir_mon_use_gold_cnt","late_1mon_use_gold_cnt","late_3mon_use_gold_इंडो","late_6mon_use_gold_cnt","cur_mon_use_points_cnt","late_1mon_use_points_cnt","late_3mon_use_points_cnt","late_6mon_use_points_cnt","cur_mon_poins_sign_cnt","late_imon_points_sign_cnt","late_3mon_points_sign_cnt","late_6mon_points_sign_cnt","cur_mon_points_luck_gift_cnt","late_1Imon_points_luck_gift_cnt","late_3mon_points_luck_gift_cnt","late_6mon_points_luck_gift_cnt","cur_mon_cps_order_types","late_imon_cps_order_types","late_3mon_cps_order_types","late_6mon_cps_order_types","last_mon_game_accum_click_cnt","last_mon_game_accum_expo_cnt","last_mon_read_accum_click_cnt","last_mon_read_accum_expo_cnt","cur_yili_vipterms","cur_juhui_vip_terms","last_login_ayh_time","last_login_app_time","last_uas_reject_time","last_30d_fq_cs","last_30d_tx_cs","last_30d_login_ayh_days","last_5d_lin_e_cnt","last_Sd_gu_e_cnt","last_yunying_jj_new_rate","last_yunying_jj_beegin_time","last_yunying_jj_end_time","last_90d_yunying_jj_cs","last_180d_yunying_jj_cs","partner_yx_contr_cnt","ayht10_all_respond_score","call_ans_score_t10","member_types","xyl_model_107","kzr_types","elec_types","sms_types","if_sms_yn","if_ele_yn","if_500up_yn","if_bj_10_yn","if_bj_30_yn","avail_bal_cash","send_coupon_cnt_1mon","use_coupon_cnt_1mon","valid_coupon_cnt","coupon_use","clkk_wyyh_count","clk_llydg_count","clk_yhq_sy_count","clk_hyzx_sy_count","clk_jryhq_count","zaidai_days","zaidai_days/mob_3y as zaidai_rate","avg_lim_use_rate","pril_bal","crdt_lim_yx","lim_use_rate","zaidai_ctrl_rate","is_login","list_call_times_d","list_ansr_times_d","list_ansr_dur_d","list_manu_call_times_d","list_manu_ansr_times_d","list_manu_ansr_dur_d","list_ivr_call_times_d","list_ivr_ansr_timees_d","list_ivr_ansr_dur_d","list_call_times","list_ansr_times","list_ansr_dur","cust_call_times","cust_ansr_times","cust_ansr_dur","cust_call_times_mon","cust_ansr_times_mon","cust_ansr_dur_mon","cust_manu_call_times","cust_manu_ansr_times","cust_manu_ansr_dur","cust_manu_call_times_mon","cust_manu_ansr_times_mon","cust_manu_ansr_dur_mon","cust_ivr_call_times","cust_ivr_ansr_times","cust_ivr_ansr_dur","otust_ivr_call_times_mon","cust_ivr_ansr_times_mon","cust_ivr_ansr_dur_mon","age","last_30d_tel_succ_cs","loss_model_ggroup_v3","risk_ms11_1_model_score","standard_score_group_v6_1","last_month_expire_coupon_cnt","number_of_gold_be_used","last_10d_lin_e_cnt","last_10d_gu_e_cnt","call_anss_score_t10","avail_cash","avg_limuse_rate","yls_cust_type_v2","cust_types_01","cust_types_02","igender_cd","icust_ty","label"
                      ],
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


def add_days_from_today(df, date_column, new_column=None, handle_nulls=True):
    """添加日期距离当前天数的列，优化内存使用"""
    # 仅复制需要的列，避免全量复制
    if date_column not in df.columns:
        print(f"警告: 日期列 {date_column} 不存在，跳过日期特征生成")
        return df

    # 创建浅拷贝，仅复制结构而非数据
    result_df = df.copy(deep=False)
    if new_column is None:
        new_column = f"{date_column}_days_from_today"

    today = pd.Timestamp.today().normalize()
    # 直接在浅拷贝上处理，不创建额外副本
    result_df[date_column] = pd.to_datetime(result_df[date_column], errors='coerce')
    invalid_dates = result_df[date_column].isna().sum()

    if invalid_dates > 0:
        print(f"警告：有{invalid_dates}个日期值无法转换为有效的日期格式")
        if handle_nulls:
            result_df[date_column].fillna(today, inplace=True)

    # 计算新列
    result_df[new_column] = (today - result_df[date_column]).dt.days
    # 删除原始日期列以节省内存
    if date_column != new_column and date_column not in CONFIG['numeric_features'] + CONFIG['categorical_features']:
        del result_df[date_column]

    return result_df


def extract_user_features(df, user_id_col='user_id', label_col='label', is_predict=False):
    """提取用户特征，优化内存使用"""
    if user_id_col not in df.columns:
        print(f"警告: 用户ID列 {user_id_col} 不存在，跳过用户特征提取")
        return df

    # 仅复制必要的列，减少内存占用
    required_cols = [user_id_col]
    if not is_predict and label_col in df.columns:
        required_cols.append(label_col)
    required_cols.extend([col for col in df.columns
                          if col in CONFIG['numeric_features'] + CONFIG['categorical_features']])

    # 使用浅拷贝
    result_df = df[required_cols].copy(deep=False)

    if not is_predict and label_col in result_df.columns:
        # 原地转换标签列
        result_df[label_col] = pd.to_numeric(result_df[label_col], errors='coerce', downcast='integer')

        def process_group(group):
            has_positive = (group[label_col] == 1).any()
            if has_positive:
                return group[group[label_col] == 1].iloc[[0]]
            else:
                return group.iloc[[0]]
    else:
        def process_group(group):
            return group.iloc[[0]]

    # 分组后直接返回结果，不复制额外数据
    result = result_df.groupby(user_id_col, group_keys=False, observed=True).apply(process_group)
    return result

class FeatureBinner:
    """特征分箱处理器"""
    def __init__(self):
        self.binners = {}
        self.bin_params = {}

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
            # 只传入需要的列
            binner.fit(X[[feature]])
            self.binners[feature] = binner
            self.bin_params[feature] = {
                'n_bins': n_bins,
                'strategy': strategy,
                'bin_edges': binner.bin_edges_[0].tolist()
            }

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """应用分箱器，使用浅拷贝减少内存"""
        X_copy = X.copy(deep=False)
        for feature, binner in self.binners.items():
            if feature in X_copy.columns:
                X_copy[feature] = binner.transform(X_copy[[feature]]).flatten()
        return X_copy

    def fit_transform(self, X: pd.DataFrame, config: dict) -> pd.DataFrame:
        """拟合并转换"""
        self.fit(X, config)
        return self.transform(X)

    def save(self, path: str) -> None:
        """保存分箱参数"""
        with open(path, "wb") as f:
            pickle.dump(self.bin_params, f)

    def load(self, path: str) -> None:
        """加载分箱参数"""
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


def save_preprocessors(encoders, binner, save_dir):
    """保存编码器和分箱器"""
    os.makedirs(save_dir, exist_ok=True)
    encoders_path = os.path.join(save_dir, "encoders.pkl")
    with open(encoders_path, "wb") as f:
        pickle.dump(encoders, f)
    print(f"编码器已保存至: {encoders_path}")

    if binner and binner.bin_params:
        binner_path = os.path.join(save_dir, "binner.pkl")
        binner.save(binner_path)
        print(f"分箱器已保存至: {binner_path}")


def load_preprocessors(load_dir):
    """加载编码器和分箱器"""
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


def parse_feature_str(feature_str, features_list):
    """解析以逗号分隔的特征字符串"""
    values = feature_str.split(',')

    if len(values) != len(features_list):
        raise ValueError(f"特征数量不匹配: 字符串有{len(values)}个特征, 预期{len(features_list)}个")

    result = {}
    for name, value_str in zip(features_list, values):
        # 处理空值
        if value_str.strip().lower() in ['', 'null', 'none', 'nan']:
            result[name] = None
            continue

        # 尝试转换为数值类型
        try:
            result[name] = int(value_str)
        except ValueError:
            try:
                result[name] = float(value_str)
            except ValueError:
                result[name] = value_str.strip()

    return result


def parse_file_to_df(file_path, features_list):
    """将文件解析为DataFrame"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                features = parse_feature_str(line, features_list)
                data.append(features)
            except Exception as e:
                print(f"解析文件 {file_path} 第 {line_num} 行出错: {str(e)}")
                continue

    df = pd.DataFrame(data)
    df = df.reindex(columns=features_list)  # 确保列顺序一致
    return df


def load_data_from_folder(folder_path, is_predict=False):
    """加载文件夹中所有文件并合并为DataFrame，优化内存"""
    all_files = []
    features_list = CONFIG["features_list"]

    # 确定需要加载的列
    required_features = CONFIG["numeric_features"] + CONFIG["categorical_features"]
    if not is_predict and CONFIG["label_col"] in features_list:
        required_features.append(CONFIG["label_col"])
    if 'last_op_time' in features_list:
        required_features.append('last_op_time')
    if 'user_id' in features_list:
        required_features.append('user_id')

    # 过滤出实际存在的列
    required_features = [f for f in required_features if f in features_list]

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path) and not file.startswith('.'):
            try:
                print(f"正在解析 {file}...")
                df = parse_file_to_df(file_path, features_list)

                # 只保留需要的列
                df = df[required_features].copy(deep=False)

                # 转换数值特征为float32以节省内存
                numeric_features = CONFIG.get("numeric_features", [])
                for feature in numeric_features:
                    if feature in df.columns:
                        if df[feature].dtype == 'object':
                            df[feature] = df[feature].astype(str).str.replace(',', '').str.replace('%', '')
                        df[feature] = pd.to_numeric(df[feature], errors='coerce').astype(np.float32)

                # 转换标签列为float32
                if not is_predict and CONFIG["label_col"] in df.columns:
                    df[CONFIG["label_col"]] = pd.to_numeric(df[CONFIG["label_col"]], errors='coerce').astype(np.float32)

                all_files.append(df)
                print(f"已加载 {file}，包含 {len(df)} 行数据")
            except Exception as e:
                print(f"加载 {file} 时出错: {str(e)}")
                continue

    if not all_files:
        raise ValueError(f"在 {folder_path} 中未找到可解析的文件")

    # 合并数据
    combined_df = pd.concat(all_files, axis=0, ignore_index=True)

    # 清理临时变量
    del all_files
    gc.collect()

    # 去重减少数据量
    combined_df = combined_df.drop_duplicates()

    # 日期特征处理
    if 'last_op_time' in combined_df.columns:
        combined_df = add_days_from_today(combined_df, 'last_op_time')

    # 用户特征提取（预测时也需要）
    if 'user_id' in combined_df.columns:
        combined_df = extract_user_features(combined_df, is_predict=is_predict)

    # 只保留最终需要的列
    keep_cols = CONFIG["numeric_features"] + CONFIG["categorical_features"]
    if not is_predict and CONFIG["label_col"] in combined_df.columns:
        keep_cols.append(CONFIG["label_col"])
    # 保留用户ID用于结果关联
    if 'user_id' in combined_df.columns:
        keep_cols.append('user_id')

    combined_df = combined_df[keep_cols]

    print(f"合并去重后的数据集大小: {len(combined_df)} 行")
    return combined_df


def handle_missing_values(df, numeric_features, categorical_features):
    """处理缺失值，使用原地操作减少内存"""
    # 浅拷贝，避免全量复制
    df = df.copy(deep=False)

    # 处理数值特征缺失值
    for col in numeric_features:
        if col in df.columns and df[col].isnull().any():
            if CONFIG["numeric_missing_strategy"] == "mean":
                fill_value = df[col].mean()
            elif CONFIG["numeric_missing_strategy"] == "median":
                fill_value = df[col].median()
            else:
                fill_value = CONFIG["numeric_fill_value"]
            # 原地填充
            df[col].fillna(fill_value, inplace=True)

    # 处理分类特征缺失值
    for col in categorical_features:
        if col in df.columns and df[col].isnull().any():
            if CONFIG["categorical_missing_strategy"] == "mode":
                fill_value = df[col].mode()[0]
            else:
                fill_value = CONFIG["categorical_fill_value"]
            # 原地填充
            df[col].fillna(fill_value, inplace=True)

    return df


def undersample_data(df, label_col, ratio=5):
    """下采样处理数据不平衡，通过索引操作减少复制"""
    if label_col not in df.columns:
        print(f"警告: 标签列 {label_col} 不存在，跳过下采样")
        return df

    # 仅获取索引而非数据副本
    pos_idx = df[df[label_col] == 1].index
    neg_idx = df[df[label_col] == 0].index

    n_pos = len(pos_idx)
    n_neg = min(len(neg_idx), n_pos * ratio)

    # 通过索引采样，避免数据复制
    neg_sample_idx = np.random.choice(neg_idx, size=n_neg, replace=False)
    balanced_idx = np.concatenate([pos_idx, neg_sample_idx])
    np.random.shuffle(balanced_idx)

    return df.loc[balanced_idx]


def calculate_detailed_metrics(y_true, y_pred_proba, threshold=0.5):
    """计算详细评估指标"""
    y_pred_proba = np.array(y_pred_proba)
    if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] > 1:
        y_pred_proba = y_pred_proba[:, 0]

    y_pred = (y_pred_proba >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    try:
        roc_auc = roc_auc_score(y_true, y_pred_proba)
    except ValueError:
        roc_auc = 0.0

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

    return {
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


def print_detailed_metrics(metrics, dataset_name):
    """打印详细指标"""
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
    """保存训练结果"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if results_dir is None:
        results_dir = f"results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    # 保存配置
    with open(os.path.join(results_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    # 保存训练历史
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

    # 保存模型
    torch.save(trainer.best_model_state, os.path.join(results_dir, "best_model.pth"))
    torch.save(trainer.model.state_dict(), os.path.join(results_dir, "final_model.pth"))

    # 保存测试结果
    test_results = {
        "test_accuracy": test_metrics["overall"]["accuracy"],
        "test_auc_pr": test_metrics["overall"]["auc_pr"],
        "test_f1": test_metrics["overall"]["f1_score"],
        "test_threshold": test_metrics["threshold"],
        "positive_precision": test_metrics["positive"]["precision"],
        "positive_recall": test_metrics["positive"]["recall"],
        "negative_precision": test_metrics["negative"]["precision"],
        "negative_recall": test_metrics["negative"]["recall"]
    }

    with open(os.path.join(results_dir, "test_results.json"), "w") as f:
        json.dump(test_results, f, indent=4)

    # 保存预处理工具
    save_preprocessors(encoders, binner, results_dir)
    print(f"所有结果已保存至: {results_dir}")
    return results_dir


class DeepFMDataset(Dataset):
    """DeepFM数据集类，优化内存使用"""
    def __init__(self, df, numeric_features, categorical_features, label_col=None,
                 encoders=None, binner=None, is_train=True):
        # 仅保留必需的特征列
        required_cols = numeric_features + categorical_features
        if label_col and label_col in df.columns:
            required_cols.append(label_col)
        # 保留用户ID用于预测结果关联
        if 'user_id' in df.columns:
            required_cols.append('user_id')

        self.df = df[required_cols].copy(deep=False).reset_index(drop=True)

        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.label_col = label_col
        self.is_train = is_train
        self.binner = binner
        self.user_ids = self.df['user_id'].values if 'user_id' in self.df.columns else None

        # 处理缺失值
        self.df = handle_missing_values(self.df, numeric_features, categorical_features)

        # 分箱处理
        if self.binner and numeric_features:
            numeric_data = self.df[numeric_features]
            if is_train:
                binned_data = self.binner.fit_transform(numeric_data, CONFIG["bin_config"])
            else:
                binned_data = self.binner.transform(numeric_data)
            self.df[numeric_features] = binned_data

        # 编码分类特征
        self.categorical_encoders = {} if encoders is None else encoders
        self.categorical_dims = {}
        self._encode_categorical()

        # 转换为numpy数组（更省内存）
        self.numeric_data = self.df[numeric_features].values.astype(np.float32)
        self.categorical_data = self.df[[f"{f}_encoded" for f in categorical_features]].values.astype(np.int32)

        # 标签处理
        if label_col and label_col in self.df.columns:
            self.labels = self.df[label_col].values.astype(np.float32)
        else:
            self.labels = None

        # 释放DataFrame内存
        del self.df
        gc.collect()

    def _encode_categorical(self):
        """优化编码逻辑，减少中间数据复制"""
        for feature in self.categorical_features:
            if self.is_train:
                encoder = LabelEncoder()
                data = self.df[feature].astype(str).values
                missing_value = CONFIG["categorical_fill_value"]

                # 检查是否需要添加缺失值样本
                if missing_value not in np.unique(data):
                    # 只扩展当前特征的数组，不复制整个DataFrame
                    data = np.append(data, missing_value)
                    encoded = encoder.fit_transform(data)
                    encoded = encoded[:-1]  # 移除临时添加的元素
                else:
                    encoded = encoder.fit_transform(data)

                self.categorical_encoders[feature] = encoder
                self.categorical_dims[feature] = len(encoder.classes_)
            else:
                encoder = self.categorical_encoders[feature]
                data = self.df[feature].astype(str).values
                missing_value = CONFIG["categorical_fill_value"]

                # 处理未知类别
                mask = ~np.isin(data, encoder.classes_)
                if np.any(mask):
                    data[mask] = missing_value
                encoded = encoder.transform(data)
                self.categorical_dims[feature] = len(encoder.classes_)

            self.df[f"{feature}_encoded"] = encoded

    def __len__(self):
        return len(self.numeric_data)

    def __getitem__(self, idx):
        sample = {
            'numeric': torch.tensor(self.numeric_data[idx], dtype=torch.float32),
            'categorical': torch.tensor(self.categorical_data[idx], dtype=torch.long)
        }
        if self.user_ids is not None:
            sample['user_id'] = self.user_ids[idx]
        if self.labels is not None:
            sample['label'] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return sample


class DeepFM(nn.Module):
    """DeepFM模型"""
    def __init__(self, numeric_dim, categorical_dims, embed_dim=32, mlp_layers=[256, 128, 64], dropout=0.3):
        super(DeepFM, self).__init__()
        self.numeric_dim = numeric_dim
        self.embed_dim = embed_dim

        # 嵌入层
        self.embeddings = nn.ModuleList([
            nn.Embedding(cat_dim, embed_dim) for cat_dim in categorical_dims
        ])

        # FM部分 - 一阶特征
        self.first_order = nn.Linear(numeric_dim + len(categorical_dims), 1)

        # 深度部分
        input_dim = numeric_dim + len(categorical_dims) * embed_dim
        mlp_layers = [input_dim] + mlp_layers

        self.mlp = nn.Sequential()
        for i in range(1, len(mlp_layers)):
            self.mlp.add_module(f"linear_{i}", nn.Linear(mlp_layers[i-1], mlp_layers[i]))
            self.mlp.add_module(f"relu_{i}", nn.ReLU())
            self.mlp.add_module(f"dropout_{i}", nn.Dropout(dropout))

        self.final = nn.Linear(mlp_layers[-1] + 1, 1)  # 1是FM二阶输出
        self.sigmoid = nn.Sigmoid()

    def forward(self, numeric, categorical):
        # 一阶特征
        first_order_cat = torch.cat([emb(categorical[:, i]).sum(dim=1) for i, emb in enumerate(self.embeddings)], dim=1)
        first_order_input = torch.cat([numeric, first_order_cat], dim=1)
        first_order_out = self.first_order(first_order_input)

        # 二阶特征（FM）
        embeddings = [emb(categorical[:, i]) for i, emb in enumerate(self.embeddings)]
        sum_square = torch.sum(torch.stack(embeddings, dim=1), dim=1) ** 2
        square_sum = torch.sum(torch.stack(embeddings, dim=1) **2, dim=1)
        fm_second_order = 0.5 * torch.sum(sum_square - square_sum, dim=1, keepdim=True)

        # 深度部分
        deep_input = torch.cat([numeric] + embeddings, dim=1)
        deep_out = self.mlp(deep_input)

        # 合并输出
        total = first_order_out + fm_second_order + self.final(deep_out)
        return self.sigmoid(total)


class Trainer:
    """模型训练器"""
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=CONFIG["learning_rate"],
            weight_decay=CONFIG["weight_decay"]
        )
        self.train_loss_history = []
        self.train_metrics_history = []
        self.val_metrics_history = []
        self.best_val_score = -1
        self.best_model_state = None

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for batch in dataloader:
            numeric = batch['numeric'].to(self.device)
            categorical = batch['categorical'].to(self.device)
            labels = batch['label'].to(self.device).unsqueeze(1)

            self.optimizer.zero_grad()
            outputs = self.model(numeric, categorical)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * numeric.size(0)
            all_preds.extend(outputs.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

        avg_loss = total_loss / len(dataloader.dataset)
        metrics = calculate_detailed_metrics(all_labels, all_preds, CONFIG["positive_threshold"])
        return avg_loss, metrics

    def evaluate(self, dataloader):
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                numeric = batch['numeric'].to(self.device)
                categorical = batch['categorical'].to(self.device)
                labels = batch['label'].to(self.device).unsqueeze(1)

                outputs = self.model(numeric, categorical)
                all_preds.extend(outputs.detach().cpu().numpy())
                all_labels.extend(labels.detach().cpu().numpy())

        metrics = calculate_detailed_metrics(all_labels, all_preds, CONFIG["positive_threshold"])
        return metrics

    def train(self, train_loader, val_loader, epochs):
        for epoch in range(epochs):
            start_time = time.time()
            train_loss, train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)

            self.train_loss_history.append(train_loss)
            self.train_metrics_history.append(train_metrics)
            self.val_metrics_history.append(val_metrics)

            epoch_time = time.time() - start_time

            print(f"Epoch {epoch+1}/{epochs} | 时间: {epoch_time:.2f}s")
            print(f"训练损失: {train_loss:.4f} | 训练F1: {train_metrics['overall']['f1_score']:.4f}")
            print(f"验证F1: {val_metrics['overall']['f1_score']:.4f} | 验证AUC-PR: {val_metrics['overall']['auc_pr']:.4f}")

            # 保存最佳模型
            current_score = val_metrics['overall'][CONFIG["optimize_for"]]
            if current_score > self.best_val_score:
                self.best_val_score = current_score
                self.best_model_state = self.model.state_dict()
                print(f"保存最佳模型 (当前{CONFIG['optimize_for']}: {current_score:.4f})")


def predict(model, dataloader, device):
    """预测函数，返回预测概率和用户ID（如果有）"""
    model.eval()
    all_preds = []
    all_user_ids = []

    with torch.no_grad():
        for batch in dataloader:
            numeric = batch['numeric'].to(device)
            categorical = batch['categorical'].to(device)
            outputs = model(numeric, categorical)

            all_preds.extend(outputs.detach().cpu().numpy().flatten())

            # 收集用户ID（如果存在）
            if 'user_id' in batch:
                all_user_ids.extend(batch['user_id'].numpy())

    # 构建预测结果
    results = {
        'predictions': all_preds,
        'threshold': CONFIG["positive_threshold"],
        'predictions_binary': [1 if p >= CONFIG["positive_threshold"] else 0 for p in all_preds]
    }

    # 如果有用户ID，添加到结果中
    if all_user_ids:
        results['user_ids'] = all_user_ids

    return results


def save_predictions(predictions, save_path):
    """保存预测结果"""
    df = pd.DataFrame({
        'user_id': predictions.get('user_ids', [f'pred_{i}' for i in range(len(predictions['predictions']))]),
        'prediction_prob': predictions['predictions'],
        'prediction_binary': predictions['predictions_binary']
    })
    df.to_csv(save_path, index=False)
    print(f"预测结果已保存至: {save_path}")
    return df


def train_model(train_folder, test_folder, results_dir=None):
    """训练模型并保存结果"""
    # 加载数据
    print("加载训练数据...")
    train_df = load_data_from_folder(train_folder, is_predict=True)
    print("加载测试数据...")
    test_df = load_data_from_folder(test_folder, is_predict=False)

    # 数据不平衡处理
    if CONFIG["imbalance_method"] in ["undersample", "both"]:
        train_df = undersample_data(train_df, CONFIG["label_col"], CONFIG["undersample_ratio"])

    # 划分训练集和验证集
    train_df, val_df = train_test_split(
        train_df, test_size=CONFIG["val_size"], random_state=42,
        stratify=train_df[CONFIG["label_col"]] if CONFIG["label_col"] in train_df.columns else None
    )

    # 特征处理
    binner = FeatureBinner() if CONFIG["bin_config"] else None
    train_dataset = DeepFMDataset(
        train_df,
        numeric_features=CONFIG["numeric_features"],
        categorical_features=CONFIG["categorical_features"],
        label_col=CONFIG["label_col"],
        binner=binner,
        is_train=True
    )
    encoders = train_dataset.categorical_encoders
    categorical_dims = list(train_dataset.categorical_dims.values())

    val_dataset = DeepFMDataset(
        val_df,
        numeric_features=CONFIG["numeric_features"],
        categorical_features=CONFIG["categorical_features"],
        label_col=CONFIG["label_col"],
        encoders=encoders,
        binner=binner,
        is_train=False
    )

    test_dataset = DeepFMDataset(
        test_df,
        numeric_features=CONFIG["numeric_features"],
        categorical_features=CONFIG["categorical_features"],
        label_col=CONFIG["label_col"],
        encoders=encoders,
        binner=binner,
        is_train=False
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        pin_memory=True,
        num_workers=2,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        pin_memory=True,
        num_workers=2
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        pin_memory=True,
        num_workers=2
    )

    # 初始化模型
    numeric_dim = len(CONFIG["numeric_features"])
    model = DeepFM(
        numeric_dim=numeric_dim,
        categorical_dims=categorical_dims,
        embed_dim=CONFIG["embed_dim"],
        mlp_layers=CONFIG["mlp_layers"],
        dropout=CONFIG["dropout"]
    )

    # 训练模型
    trainer = Trainer(model)
    print(f"开始训练，使用设备: {trainer.device}")
    trainer.train(train_loader, val_loader, epochs=CONFIG["epochs"])

    # 评估最佳模型
    print("\n加载最佳模型进行评估...")
    model.load_state_dict(trainer.best_model_state)
    trainer.model = model

    print("\n评估训练集...")
    train_metrics = trainer.evaluate(train_loader)
    print_detailed_metrics(train_metrics, "训练")

    print("评估验证集...")
    val_metrics = trainer.evaluate(val_loader)
    print_detailed_metrics(val_metrics, "验证")

    print("评估测试集...")
    test_metrics = trainer.evaluate(test_loader)
    print_detailed_metrics(test_metrics, "测试")

    # 保存结果
    results_dir = save_results(trainer, CONFIG, test_metrics, encoders, binner, results_dir)
    return results_dir


def predict_new_data(data_folder, model_dir, output_path=None):
    """使用训练好的模型预测新数据"""
    # 加载预处理工具
    encoders, binner = load_preprocessors(model_dir)

    # 加载预测数据
    print("加载预测数据...")
    predict_df = load_data_from_folder(data_folder, is_predict=True)

    # 创建预测数据集
    predict_dataset = DeepFMDataset(
        predict_df,
        numeric_features=CONFIG["numeric_features"],
        categorical_features=CONFIG["categorical_features"],
        label_col=None,  # 预测时没有标签
        encoders=encoders,
        binner=binner,
        is_train=False
    )

    # 创建数据加载器
    predict_loader = DataLoader(
        predict_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        pin_memory=True,
        num_workers=2
    )

    # 初始化模型
    categorical_dims = [len(encoder.classes_) for encoder in encoders.values()]
    numeric_dim = len(CONFIG["numeric_features"])
    model = DeepFM(
        numeric_dim=numeric_dim,
        categorical_dims=categorical_dims,
        embed_dim=CONFIG["embed_dim"],
        mlp_layers=CONFIG["mlp_layers"],
        dropout=CONFIG["dropout"]
    )

    # 加载模型权重
    model_path = os.path.join(model_dir, "best_model.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, "final_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    # 预测
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    predictions = predict(model, predict_loader, device)

    # 保存预测结果
    if output_path is None:
        output_path = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    result_df = save_predictions(predictions, output_path)

    return result_df


def main():
    # parser = argparse.ArgumentParser(description='DeepFM模型训练与预测')
    # subparsers = parser.add_subparsers(dest='command', help='命令选项')
    #
    # # 训练命令
    # train_parser = subparsers.add_parser('train', help='训练模型')
    # train_parser.add_argument('--train_folder', type=str, required=True, help='训练数据文件夹路径')
    # train_parser.add_argument('--test_folder', type=str, required=True, help='测试数据文件夹路径')
    # train_parser.add_argument('--results_dir', type=str, default=None, help='结果保存目录')
    #
    # # 预测命令
    # predict_parser = subparsers.add_parser('predict', help='使用模型预测')
    # predict_parser.add_argument('--data_folder', type=str, required=True, help='预测数据文件夹路径')
    # predict_parser.add_argument('--model_dir', type=str, required=True, help='模型保存目录')
    # predict_parser.add_argument('--output_path', type=str, default=None, help='预测结果输出路径')
    #
    # args = parser.parse_args()
    #
    # if args.command == 'train':
    #     train_model(args.train_folder, args.test_folder, args.results_dir)
    # elif args.command == 'predict':
    #     predict_new_data(args.data_folder, args.model_dir, args.output_path)
    # else:
    #     parser.print_help()

    train_model("./","./test_data")


if __name__ == "__main__":
    main()
