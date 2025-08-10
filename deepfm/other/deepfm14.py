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
    "epochs": 5,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "embed_dim": 32,
    "mlp_layers": [256, 128, 64],
    "dropout": 0.3,
    "numeric_features": ['age','lim_use_rate', 'total_loan_ccnt','total_loan_amt', 'delay_days', 'push_cnt',
                         'sms_charge_cnt', 'tel_call_dur', 'wdraw_cnt', 'wdraw_amt''wdraw_amt_t30', 'tel_call_inav_cnt', 'apply_amt', 'applycnt'
                                                                                                                                      'avail_bal_cash','pril_bal','crdt_lim_yx','last_30d_tel_succ_cs', 'querycncq03nwww','querycncq03wwlw',
                         'querycncq03wwpw', 'cdaccapwwwww', 'dlaccapwwwwww', 'rv/accapwwww', 'rvrpsapwwwwww', 'rvclmapwwwww','rvclaapwwwWW',
                         'rvblsapwwwww','rvnbmwpwwwwww','rvnbawpwwwwww','rvapsapwwww','rvapmapwwwwww', 'rvapaapwwwww', 'last_30d_login_app_days'
                                                                                                                       'last_30d_login_ayh_days', 'latest_login_days','last_30d_sms_send_succ_cs','partner_yx_contr_cnt',
                         'last_5d_coupon_cnt', 'last_10d_coupon_cnt','last_fq_tx_:amt','last_30d_fq_cs','last_30d_tx_cs',
                         'last_yunying_jj_new_rate', 'last_90d_yunying_jj_cs','last_180d_yunying_jj_cs', 'als_m3_id_nbank_allnum',
                         'als_m1_id_bank_allnum', 'als_m1_id_nbank_allnum','ahls_m3_id_bank_allnum'],
    "categorical_features": ['user_id','unique_id','t19_user_status','user_mobile_status','is_realname_cert',
                             'is_mobile_realname_cert','t19_is_complete_base_info','t19_is_complete_occ_info','t19_is_complete_contact_info'
                             'is_ms_staff','gender_cd', 'is_login', 'chan_type','reg_chan_no','reg_dev_app_dl_chan_no', 'ocr_ethnic',
                             'residence_pr_cd','residence_city_cd','occ_cd', 'census_pr_cd', 'census_city_cd', 'census_area_cd', 'is_coupon_used',
                             'push_status','delay_type', 'click_status', 'is_coupon_issue','is_credit', 'is_apply', 'is_wdraw', 'aprv_status',
                             'touch_status','residence_area_cd', 'kzr_types','elec_types','sms_types', 'if_sms_yn', 'if_ele_yn', 'if_500up_yn',
                             'if_bj_10_yn','if_bj_30_yn', 'cust_ty','cust_types_01', 'cust_types_02', 'xyl_tag', 'yls_cust_type_v2','member_types'
                              'last_5d_lin_e_cnt', 'last_10d_lin_e_cnt', 'last_5d_gu_e_cnt','last_10d_gu_e_cnt','flagcd','reg_dev_sys_type',
                             'tel_call_type'],
    "label_col": "label",
    "bin_config": {
        'age': {'n_bins': 20, 'strategy': 'quantile'}
    },
    "features_list": ['A_flow_id',"user_id", 'unique_id', 'inviite_time', 'create_tm', 'label', 'is_coupon_used'
                                                                                                'valid_begin_date','valid_end_date',
                      'push_status','delay_type', 'delay_days', 'push_cnt','sms_chharge_cnt','tel_call_type', 'tel_call_dur','touch_status',
                      'click_status',
                      'is_coupon_issue','is_credit', 'first_biz_name', 'is_apply''is_wdraw', 'wdraw_cnt', 'wdraw_amt', 'wdraw_amt_t0',
                      'wdraw_amt_t3',
                      'wdraw_amt_t7', 'wdraw_amt_t10', 'wdraw_amt_t14', 'wdraw_amt_t15', 'wdraw_amt_t30', 'second_biz_name', 'third_biz_name',
                      'tel_call_inav_cnt', 'plan_id', 'subscribe_no','t20_cust_id','db_src', 'cust_date', 'prod_cd', 'channel_task_id', 'plan_name',
                      'cust_gp_code', 'cust_gp_name', 'touch_name','touch_id', 'touuch_type', 'channel_name', 'channel_id', 'apply_amt','apply_Ent',
                      'aprv_status', 'cust_recv_time', 'decision_id', 'decision_namme','touch_time','t19_user_status','user_mobile_status',
                      'is_realname_cert', 'is_mobile_realname_cert', 'phy_del_op_side','t2_cust_id', 'user_name', 't19_mobile_no','t19_id_no',
                      'reg_date','reg_time','t19_reg_dt','reg_term','reg_dev_app_dl_chan_no','reg_chan_no', 'reg_2nd_chan_no',
                      'reg_attr_chan_no', 'reg_attr_2nd_chan_no', 'chan_type','reg_dev_info','reg_dev_sys_type','reg_dev_sys_vn', 'reg_app_vn',
                      'reg_dev_mfr', 'reg_dev_brand', 'reg_dev_model', 'reg_dev_compile__id','reg_imei','reg_gps','reg_lng','reg_lat'
                       'reg_network', 'time_inst', 'time_upd', 't19_etl_time','t19_is_complete_base_info','t19_is_complete_occ_info',
                      't19_is_complete_contact_info', 'logout_dt','lock_dt','is_ms_staff','age','gender_cd','residence_city_cd',
                      'ocr_ethnic','residence_pr_cd', 'occ_cd', 'census_pr_cd', 'census_city_cd', 'census_area_cd',
                      'residence_area_cd', 'cust_ty','cust_types_01',
                      'cust_types_02', 'yls_cust_type','xyl_tag', 'last_fq_dt','last_fq_tx_amt','last_tx_dt','last_tx_amt',
                      'last_30d_login_app_days',
                      'last_30d_push_touch_times','last_30d_sms_send_succ_cs','last_30d_tel_succ_cs', 'last_5d_coupon_cnt',
                      'last_10d_coupon_cnt',
                      'last_op_time', 'last_pboc_cx_dt', 'querycncq03nwww','querycncq03wwlw', 'querycncq03wwpw', 'flagcd', 'cdaccapwwwwww',
                      'dlaccapwwwwwww', 'rvaccapwwwwww', 'rvrpsapwww', 'rvclmapwwww', 'rvclaapwwwww','rvblsapwwwww', 'rvnbmwww', 'rvnbawpwwwwww',
                      'rvapsapwwwww', 'rvapmapwwwwww', 'rvapaapwww', 'pboc_repay_date_days','als_m1_id_bank_allnum', 'als_m1_id_nbank_allnum',
                      'als_m3_id_bank_allnum', 'als_m3_id_nbank_allnum', 'br_modle_score','query_times_bank_90d', 'query_times_cfc_90d',
                      'risk_ms9_model_score', 'loss_model_group_v3', 'yls_cust_type_v2','risk_ms11_1_model_score', 'standard_score_group__v6_1'
                      'last_login_ayh_time','last_login_app_time', 'last_uas_reject_time','last_30d_fq_cs', 'last_30d_tx_cs',
                      'last_30d_login_ayh_days', 'last_5d_lin_e_cnt', 'last_10d_lin_e_cnt','last_5d_gu_e_cnt', 'last_10d_gu_e_cnt',
                      'last_yunying_jj_new_rate', 'last_yunying_jj_begin_time', 'last_yunying_jj_end_time', 'last_90d_yunying_jj_cs',
                      'last_180d_yunying_jj_cs', 'partner_yx_contr_cnt','ayht10_all_respond_score', 'call_ans_score_t10', 'memlber_types'
                                                                                                                          'xyl_model_107', 'latest_login_days', 'total_loan_cnt', 'total_loan_amt','kzr_types', 'elec_types','sms_types', 'if_sms_yn',
                      'if_ele_yn','if_500up_yn', 'if_bj_10_yn','if_bj_30_yn','avail_bal_cash','avg_lim_use_rate', 'pril_bal', 'crdt_lim_yx',
                      'lim_use_rate','is_login'],
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
    """加载文件夹中所有CSV文件并合并为一个DataFrame，并确保数值特征为float类型"""
    all_files = []

    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            file_path = os.path.join(folder_path, file)
            try:
                # 读取CSV文件，所有列初始为字符串类型
                df = pd.read_csv(file_path, dtype=str)

                # 如果配置中有特征列表，检查并重命名列
                if 'features_list' in CONFIG:
                    if len(df.columns) == len(CONFIG['features_list']):
                        df.columns = CONFIG['features_list']

                # 将数值特征转换为float类型
                numeric_features = CONFIG.get("numeric_features", [])
                for feature in numeric_features + ["label"]:
                    if feature in df.columns:
                        # 处理可能的非数值字符（如逗号、百分号等）
                        df[feature] = df[feature].str.replace(',', '').str.replace('%', '')
                        # 转换为float，无法转换的设为NaN
                        df[feature] = pd.to_numeric(df[feature], errors='coerce')

                all_files.append(df)
                print(f"已加载 {file}，包含 {len(df)} 行数据")
            except Exception as e:
                print(f"加载 {file} 时出错: {str(e)}")

    if not all_files:
        raise ValueError(f"在 {folder_path} 中未找到CSV文件")

    combined_df = pd.concat(all_files, axis=0, ignore_index=True)

    # 再次确保数值特征为float类型（处理合并后可能的类型不一致）
    numeric_features = CONFIG.get("numeric_features", [])
    for feature in numeric_features + ["label"]:
        if feature in combined_df.columns:
            combined_df[feature] = pd.to_numeric(combined_df[feature], errors='coerce')

    print(f"\n合并后的数据集大小: {len(combined_df)} 行")

    # 打印数值特征的前10行
    if numeric_features:
        print("\n数值特征前10行数据:")
        # 只选择数值特征列
        numeric_df = combined_df[numeric_features].head(10)
        # 设置显示选项，确保显示所有列和行
        with pd.option_context('display.max_columns', None, 'display.expand_frame_repr', False):
            print(numeric_df)
    else:
        print("\n警告: 配置中没有定义数值特征(numeric_features)")


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

    print(f"所有结果已保存至: {results_dir}")
    return results_dir


class DeepFMDataset(Dataset):
    def __init__(self, df, numeric_features, categorical_features, label_col=None,
                 encoders=None, binner=None, is_train=True):
        # 确保输入是DataFrame
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"预期DataFrame类型，实际得到{type(df)}")

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

        # 处理缺失值
        self.df = handle_missing_values(self.df, numeric_features, categorical_features)

        # 处理数值特征分箱
        if self.binner:
            if is_train:
                self.df[self.numeric_features] = self.binner.fit_transform(
                    self.df[self.numeric_features], CONFIG["bin_config"])
            else:
                self.df[self.numeric_features] = self.binner.transform(
                    self.df[self.numeric_features])
        print("\n数值特征前10行数据:")
        # 只选择数值特征列
        numeric_df = self.df[self.numeric_features].head(10)
        # 设置显示选项，确保显示所有列和行
        with pd.option_context('display.max_columns', None, 'display.expand_frame_repr', False):
            print(numeric_df)

        # 初始化编码器
        self.categorical_encoders = {} if encoders is None else encoders
        self.categorical_dims = {}
        self._encode_categorical()

        # 准备模型输入数据
        self.numeric_data = self.df[self.numeric_features].values.astype(np.float32)
        self.categorical_data = self.df[[f"{f}_encoded" for f in categorical_features]].values.astype(np.int64)

        if label_col and label_col in self.df.columns:
            self.labels = self.df[label_col].values.astype(np.float32)
        else:
            self.labels = None




    def _encode_categorical(self):
        """编码分类特征，确保包含缺失值填充值的编码"""
        for feature in self.categorical_features:
            if self.is_train:
                encoder = LabelEncoder()
                # 确保特征是字符串类型
                if self.df[feature].dtype != 'object':
                    self.df[feature] = self.df[feature].astype(str)

                # 关键改进：确保'Missing'值被包含在训练编码器中
                missing_value = CONFIG["categorical_fill_value"]
                if missing_value not in self.df[feature].unique():
                    # 临时添加一行包含'Missing'值，确保编码器能识别它
                    temp_df = self.df.copy()
                    temp_row = pd.DataFrame({feature: [missing_value]})
                    temp_df = pd.concat([temp_df, temp_row], ignore_index=True)
                    # 训练编码器并获取编码结果
                    encoded = encoder.fit_transform(temp_df[feature])
                    # 移除临时添加的行的编码结果
                    encoded = encoded[:-1]
                else:
                    # 如果数据中已经有'Missing'值，直接训练
                    encoded = encoder.fit_transform(self.df[feature])

                self.categorical_encoders[feature] = encoder
                self.categorical_dims[feature] = len(encoder.classes_)
            else:
                # 预测阶段的编码处理
                if feature not in self.categorical_encoders:
                    raise ValueError(f"编码器中缺少特征 {feature} 的编码信息")

                encoder = self.categorical_encoders[feature]
                missing_value = CONFIG["categorical_fill_value"]

                # 检查填充值是否在编码器中
                if missing_value not in encoder.classes_:
                    raise ValueError(
                        f"编码器未包含缺失值填充值 '{missing_value}' 的编码信息，"
                        f"请重新训练模型以包含此值"
                    )

                # 处理预测数据中可能出现的未知类别
                mask = ~self.df[feature].isin(encoder.classes_)
                if mask.any():
                    # 将未知类别替换为填充值
                    self.df.loc[mask, feature] = missing_value

                # 执行编码转换
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

        all_preds = np.array(all_preds).flatten()
        all_labels = np.array(all_labels).flatten()

        best_score = 0.0
        best_threshold = 0.5

        # 搜索最佳阈值
        for threshold in np.arange(0.1, 0.91, 0.01):
            y_pred = (all_preds >= threshold).astype(int)
            if CONFIG["optimize_for"] == "f1":
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
                if CONFIG["threshold_tuning"]:
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


def train_model(data_folder, test_folder,results_dir=None):
    """训练模型并返回结果目录"""
    # 加载数据
    df = load_data_from_folder(data_folder)

    test_df = load_data_from_folder(test_folder)

    # 处理数据不平衡
    if CONFIG["imbalance_method"] in ["undersample", "both"]:
        df = undersample_data(df, CONFIG["label_col"], CONFIG["undersample_ratio"])

    # 划分数据集
    train_df, val_df = train_test_split(
        df, test_size=CONFIG["test_size"], random_state=42, stratify=df[CONFIG["label_col"]]
    )
    # train_df, val_df = train_test_split(
    #     train_val_df, test_size=CONFIG["val_size"], random_state=42, stratify=train_val_df[CONFIG["label_col"]]
    # )

    # 初始化分箱器
    binner = FeatureBinner() if CONFIG["bin_config"] else None

    # 创建数据集
    train_dataset = DeepFMDataset(
        train_df,
        numeric_features=CONFIG["numeric_features"],
        categorical_features=CONFIG["categorical_features"],
        label_col=CONFIG["label_col"],
        binner=binner,
        is_train=True
    )
    encoders = train_dataset.categorical_encoders
    categorical_dims = train_dataset.categorical_dims

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
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=4)

    # 初始化模型
    model = DeepFM(
        numeric_dim=len(CONFIG["numeric_features"]),
        categorical_features=CONFIG["categorical_features"],
        categorical_dims=categorical_dims,
        embed_dim=CONFIG["embed_dim"],
        mlp_layers=CONFIG["mlp_layers"],
        dropout=CONFIG["dropout"]
    )

    # 训练模型
    trainer = DeepFMTrainer(model)
    trainer.train(train_loader, val_loader, epochs=CONFIG["epochs"])


    # 在测试集上评估
    test_metrics = trainer.evaluate(test_loader, trainer.best_threshold)
    print_detailed_metrics(test_metrics, "测试")

    # 计算混淆矩阵
    y_true = test_df[CONFIG["label_col"]].values
    y_pred = (trainer.predict(test_loader) >= trainer.best_threshold).astype(int)
    trainer.test_confusion_matrix = confusion_matrix(y_true, y_pred)

    # 保存结果
    results_dir = save_results(trainer, CONFIG, test_metrics, encoders, binner, results_dir)
    return results_dir


def predict_new_data(data_folder, model_dir, output_path):
    """使用训练好的模型预测新数据"""
    # 加载数据
    df = load_data_from_folder(data_folder)

    # 加载预处理工具和模型配置
    encoders, binner = load_preprocessors(model_dir)

    # 创建数据集（无标签）
    dataset = DeepFMDataset(
        df,
        numeric_features=CONFIG["numeric_features"],
        categorical_features=CONFIG["categorical_features"],
        label_col=None,  # 新数据无标签
        encoders=encoders,
        binner=binner,
        is_train=False
    )

    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=4)

    # 加载模型配置
    with open(os.path.join(model_dir, "config.json"), "r") as f:
        config = json.load(f)

    # 初始化模型
    model = DeepFM(
        numeric_dim=len(config["numeric_features"]),
        categorical_features=config["categorical_features"],
        categorical_dims={feat: len(encoders[feat].classes_) for feat in config["categorical_features"]},
        embed_dim=config["embed_dim"],
        mlp_layers=config["mlp_layers"],
        dropout=config["dropout"]
    )

    # 加载最佳模型权重
    model_path = os.path.join(model_dir, "best_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    # 预测
    trainer = DeepFMTrainer(model)
    predictions = trainer.predict(dataloader)

    # 保存预测结果
    os.makedirs(output_path, exist_ok=True)
    result_df = df.copy()
    result_df["prediction_prob"] = predictions
    result_df["prediction_label"] = (predictions >= config["positive_threshold"]).astype(int)
    result_df.to_csv(os.path.join(output_path, "predictions.csv"), index=False)
    print(f"预测结果已保存至: {os.path.join(output_path, 'predictions.csv')}")

    return result_df


if __name__ == "__main__":
    # 重要提示：首次运行或修改了特征处理逻辑后，请先训练模型
    # 训练模型会生成包含'Missing'值编码的新编码器
    results_dir = train_model("./")  # 取消注释此、行进行训练

    # 训练完成后，使用新生成的results_dir进行预测
    # 注意：请将下面的模型目录替换为训练完成后实际生成的目录


    # result = predict_new_data("./", "./results_20250731_112705", "./output_path")
