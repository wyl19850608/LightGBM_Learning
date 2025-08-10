import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score,
                             precision_score, recall_score,
                             confusion_matrix, precision_recall_curve, auc)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import time
from datetime import datetime
import json
import pickle  # 用于保存和加载编码器
import warnings

# 忽略警告信息
warnings.filterwarnings('ignore')


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
    ],
    "categorical_features": [
        'yls_cust_type_v2', 'cust_types_01', 'cust_types_02',
        'if_sms_yn', 'elec_types', 'igender_cd', 'icust_ty',
        'if_500up_yn', 'is_login', 'sms_types', 'if_bj_30_yn',
        'member_types', 'if_bj_10_yn'
    ],
    "label_col": "label",
    "data_folder": "data",
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


# 保存编码器
def save_encoders(encoders, save_path):
    """保存编码器到指定路径"""
    with open(save_path, "wb") as f:
        pickle.dump(encoders, f)
    print(f"编码器已保存至: {save_path}")


# 加载编码器
def load_encoders(load_path):
    """从指定路径加载编码器"""
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"编码器文件不存在: {load_path}")

    with open(load_path, "rb") as f:
        encoders = pickle.load(f)
    print(f"编码器已从 {load_path} 加载")
    return encoders


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
def save_results(trainer, config, test_metrics, encoders, results_dir=None):
    """将所有训练结果保存到文件，包括编码器"""
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

    # 7. 保存编码器
    save_encoders(encoders, os.path.join(results_dir, "encoders.pkl"))

    return results_dir


# DeepFM数据集类
class DeepFMDataset(Dataset):
    def __init__(self, df, numeric_features, categorical_features, label_col, encoders=None, is_train=True):
        self.df = df.copy().reset_index(drop=True)
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.label_col = label_col
        self.is_train = is_train

        # 处理缺失值
        self.df = handle_missing_values(self.df, numeric_features, categorical_features)

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

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # 加载模型状态
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # 加载训练历史
        if 'best_auc' in checkpoint:
            self.best_auc = checkpoint['best_auc']
        if 'best_model_state' in checkpoint:
            self.best_model_state = checkpoint['best_model_state']
        if 'best_threshold' in checkpoint:
            self.best_threshold = checkpoint['best_threshold']
        if 'train_metrics_history' in checkpoint:
            self.train_metrics_history = checkpoint['train_metrics_history']
        if 'val_metrics_history' in checkpoint:
            self.val_metrics_history = checkpoint['val_metrics_history']
        if 'train_loss_history' in checkpoint:
            self.train_loss_history = checkpoint['train_loss_history']

        print(f"已从 {checkpoint_path} 加载检查点，继续训练")

    def save_checkpoint(self, save_path):
        """保存当前训练状态作为检查点"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_auc': self.best_auc,
            'best_model_state': self.best_model_state,
            'best_threshold': self.best_threshold,
            'train_metrics_history': self.train_metrics_history,
            'val_metrics_history': self.val_metrics_history,
            'train_loss_history': self.train_loss_history
        }

        torch.save(checkpoint, save_path)
        print(f"检查点已保存至: {save_path}")

    def train(self, class_weight=None, additional_epochs=None):
        """训练模型，支持继续训练（指定additional_epochs）"""
        # 确定训练的总轮数
        epochs = additional_epochs if additional_epochs is not None else CONFIG["epochs"]

        # 定义损失函数（带或不带权重）
        if class_weight is not None:
            class_weight_tensor = torch.tensor([class_weight[0], class_weight[1]],
                                               dtype=torch.float).to(self.device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=class_weight_tensor[1])
        else:
            criterion = nn.BCEWithLogitsLoss()

        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.7,
            patience=3,
            min_lr=1e-5,
            verbose=True
        )

        # 获取当前已训练的轮数
        start_epoch = len(self.train_loss_history)

        for epoch in range(start_epoch, start_epoch + epochs):
            # 获取并打印当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"\nEpoch {epoch + 1}/{start_epoch + epochs}, 学习率: {current_lr:.6f}")

            start_time = time.time()
            self.model.train()
            total_loss = 0.0
            train_preds = []
            train_labels = []

            # 训练循环
            for batch in self.train_loader:
                numeric = batch['numeric'].to(self.device)
                categorical = batch['categorical'].to(self.device)
                labels = batch['label'].to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(numeric, categorical)
                loss = criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * numeric.size(0)
                train_preds.extend(outputs.detach().cpu().numpy())
                train_labels.extend(labels.detach().cpu().numpy())

            # 计算训练指标
            avg_train_loss = total_loss / len(self.train_loader.dataset)
            self.train_loss_history.append(avg_train_loss)
            train_metrics = calculate_detailed_metrics(
                np.array(train_labels),
                np.array(train_preds),
                threshold=CONFIG["positive_threshold"]
            )
            self.train_metrics_history.append(train_metrics)

            # 验证
            val_metrics, _, _ = self.evaluate(self.val_loader, dataset_name="验证集")
            self.val_metrics_history.append(val_metrics)

            # 学习率调度
            scheduler.step(val_metrics['overall']['auc_pr'])

            # 保存最佳模型
            if val_metrics['overall']['auc_pr'] > self.best_auc:
                self.best_auc = val_metrics['overall']['auc_pr']
                self.best_model_state = self.model.state_dict()
                # 调整最佳阈值
                if CONFIG["threshold_tuning"]:
                    self.best_threshold = self.tune_threshold(
                        self.val_loader,
                        optimize_for=CONFIG["optimize_for"]
                    )

            # 打印训练时间
            epoch_time = time.time() - start_time
            print(f"Epoch 耗时: {epoch_time:.2f} 秒")

            # 每5轮保存一次检查点
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pth")

        # 训练结束后用最佳模型评估测试集
        self.model.load_state_dict(self.best_model_state)
        test_metrics, y_true, y_pred = self.evaluate(self.test_loader, dataset_name="测试集")
        self.test_confusion_matrix = confusion_matrix(y_true, y_pred)

        return test_metrics

    def evaluate(self, data_loader, dataset_name="验证集"):
        """评估模型在指定数据集上的表现"""
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in data_loader:
                numeric = batch['numeric'].to(self.device)
                categorical = batch['categorical'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(numeric, categorical)
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 计算指标
        metrics = calculate_detailed_metrics(
            np.array(all_labels),
            np.array(all_preds),
            threshold=self.best_threshold if dataset_name == "测试集" else CONFIG["positive_threshold"]
        )
        print_detailed_metrics(metrics, dataset_name)

        # 生成预测标签
        y_pred = (np.array(all_preds) >= metrics['threshold']).astype(int)
        return metrics, np.array(all_labels), y_pred

    def tune_threshold(self, data_loader, optimize_for='f1'):
        """调整分类阈值以优化指定指标"""
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in data_loader:
                numeric = batch['numeric'].to(self.device)
                categorical = batch['categorical'].to(self.device)
                labels = batch['label'].to(self.device)
                outputs = self.model(numeric, categorical)
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 测试不同阈值
        thresholds = np.arange(CONFIG["min_threshold"], CONFIG["max_threshold"], 0.01)
        best_score = -1
        best_thresh = CONFIG["positive_threshold"]

        for thresh in thresholds:
            y_pred = (np.array(all_preds) >= thresh).astype(int)
            if optimize_for == 'f1':
                score = f1_score(all_labels, y_pred)
            elif optimize_for == 'precision':
                score = precision_score(all_labels, y_pred)
            else:  # recall
                score = recall_score(all_labels, y_pred)

            # 确保有足够的正样本预测
            if sum(y_pred) >= CONFIG["min_pos_samples"] and score > best_score:
                best_score = score
                best_thresh = thresh

        print(f"优化 {optimize_for} 的最佳阈值: {best_thresh:.4f} (得分: {best_score:.4f})")
        return best_thresh


def train_new_model():
    """训练新模型的流程"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 1. 加载数据
    print("加载数据...")
    df = load_data_from_folder(CONFIG["data_folder"])

    # 检查必要列是否存在
    required_cols = CONFIG["numeric_features"] + CONFIG["categorical_features"] + [CONFIG["label_col"]]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"数据中缺少必要列: {missing_cols}")

    # 2. 划分数据集
    print("划分数据集...")
    train_val_df, test_df = train_test_split(
        df,
        test_size=CONFIG["test_size"],
        random_state=42,
        stratify=df[CONFIG["label_col"]]
    )
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=CONFIG["val_size"],
        random_state=42,
        stratify=train_val_df[CONFIG["label_col"]]
    )

    # 3. 处理类别不平衡
    print("处理类别不平衡...")
    if CONFIG["imbalance_method"] in ['undersample', 'both']:
        train_df = undersample_data(
            train_df,
            CONFIG["label_col"],
            ratio=CONFIG["undersample_ratio"]
        )

    # 计算类别权重
    class_weight = None
    if CONFIG["imbalance_method"] in ['weight', 'both']:
        count = Counter(train_df[CONFIG["label_col"]])
        total = len(train_df)
        class_weight = {
            0: total / (2 * count[0]),
            1: total / (2 * count[1])
        }
        print(f"类别权重: 负样本={class_weight[0]:.4f}, 正样本={class_weight[1]:.4f}")

    # 4. 创建数据集和数据加载器
    print("创建数据集...")
    train_dataset = DeepFMDataset(
        train_df,
        numeric_features=CONFIG["numeric_features"],
        categorical_features=CONFIG["categorical_features"],
        label_col=CONFIG["label_col"],
        is_train=True
    )
    # 获取训练集编码器（用于验证集和测试集）
    encoders = train_dataset.categorical_encoders

    val_dataset = DeepFMDataset(
        val_df,
        numeric_features=CONFIG["numeric_features"],
        categorical_features=CONFIG["categorical_features"],
        label_col=CONFIG["label_col"],
        encoders=encoders,
        is_train=False
    )

    test_dataset = DeepFMDataset(
        test_df,
        numeric_features=CONFIG["numeric_features"],
        categorical_features=CONFIG["categorical_features"],
        label_col=CONFIG["label_col"],
        encoders=encoders,
        is_train=False
    )

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=4)

    # 5. 初始化模型和优化器
    print("初始化模型...")
    numeric_dim = len(CONFIG["numeric_features"])
    model = DeepFM(
        numeric_dim=numeric_dim,
        categorical_dims=train_dataset.categorical_dims,
        embed_dim=CONFIG["embed_dim"],
        mlp_layers=CONFIG["mlp_layers"],
        dropout=CONFIG["dropout"]
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"]
    )

    # 6. 训练模型
    print("开始训练...")
    trainer = DeepFMTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        device=device
    )
    test_metrics = trainer.train(class_weight=class_weight)

    # 7. 保存结果（包含编码器）
    print("保存结果...")
    results_dir = save_results(trainer, CONFIG, test_metrics, encoders)
    print(f"所有结果已保存至: {results_dir}")

    return results_dir, encoders


def continue_training(previous_results_dir, additional_epochs=10):
    """基于已有模型继续训练"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 1. 加载之前的配置
    config_path = os.path.join(previous_results_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            global CONFIG
            CONFIG = json.load(f)
        print(f"已加载之前的配置: {config_path}")

    # 2. 加载数据
    print("加载数据...")
    df = load_data_from_folder(CONFIG["data_folder"])

    # 检查必要列是否存在
    required_cols = CONFIG["numeric_features"] + CONFIG["categorical_features"] + [CONFIG["label_col"]]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"数据中缺少必要列: {missing_cols}")

    # 3. 划分数据集（保持与之前一致）
    print("划分数据集...")
    train_val_df, test_df = train_test_split(
        df,
        test_size=CONFIG["test_size"],
        random_state=42,
        stratify=df[CONFIG["label_col"]]
    )
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=CONFIG["val_size"],
        random_state=42,
        stratify=train_val_df[CONFIG["label_col"]]
    )

    # 4. 处理类别不平衡
    print("处理类别不平衡...")
    if CONFIG["imbalance_method"] in ['undersample', 'both']:
        train_df = undersample_data(
            train_df,
            CONFIG["label_col"],
            ratio=CONFIG["undersample_ratio"]
        )

    # 计算类别权重
    class_weight = None
    if CONFIG["imbalance_method"] in ['weight', 'both']:
        count = Counter(train_df[CONFIG["label_col"]])
        total = len(train_df)
        class_weight = {
            0: total / (2 * count[0]),
            1: total / (2 * count[1])
        }
        print(f"类别权重: 负样本={class_weight[0]:.4f}, 正样本={class_weight[1]:.4f}")

    # 5. 加载之前的编码器
    encoders_path = os.path.join(previous_results_dir, "encoders.pkl")
    encoders = load_encoders(encoders_path)

    # 6. 创建数据集和数据加载器（使用已加载的编码器）
    print("创建数据集...")
    train_dataset = DeepFMDataset(
        train_df,
        numeric_features=CONFIG["numeric_features"],
        categorical_features=CONFIG["categorical_features"],
        label_col=CONFIG["label_col"],
        encoders=encoders,  # 使用已有的编码器
        is_train=False  # 不重新训练编码器
    )

    val_dataset = DeepFMDataset(
        val_df,
        numeric_features=CONFIG["numeric_features"],
        categorical_features=CONFIG["categorical_features"],
        label_col=CONFIG["label_col"],
        encoders=encoders,
        is_train=False
    )

    test_dataset = DeepFMDataset(
        test_df,
        numeric_features=CONFIG["numeric_features"],
        categorical_features=CONFIG["categorical_features"],
        label_col=CONFIG["label_col"],
        encoders=encoders,
        is_train=False
    )

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=4)

    # 7. 初始化模型和优化器
    print("初始化模型...")
    numeric_dim = len(CONFIG["numeric_features"])
    model = DeepFM(
        numeric_dim=numeric_dim,
        categorical_dims=train_dataset.categorical_dims,
        embed_dim=CONFIG["embed_dim"],
        mlp_layers=CONFIG["mlp_layers"],
        dropout=CONFIG["dropout"]
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"]
    )

    # 8. 加载之前的模型检查点
    checkpoint_path = os.path.join(previous_results_dir, "best_model.pth")
    print(f"从 {checkpoint_path} 加载模型继续训练")

    # 9. 创建训练器并继续训练
    print(f"开始继续训练，额外训练 {additional_epochs} 轮...")
    trainer = DeepFMTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        device=device
    )

    # 加载最佳模型权重
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # 继续训练
    test_metrics = trainer.train(class_weight=class_weight, additional_epochs=additional_epochs)

    # 10. 保存新的结果（在原有结果目录下）
    print("保存新的训练结果...")
    results_dir = save_results(trainer, CONFIG, test_metrics, encoders,
                               results_dir=previous_results_dir)
    print(f"所有结果已保存至: {results_dir}")

    return results_dir


def predict_new_data(model_path, encoders_path, new_data_path, threshold=None):
    """使用保存的模型和编码器预测新数据"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 1. 加载数据
    print(f"加载新数据: {new_data_path}")
    if new_data_path.endswith('.csv'):
        df = pd.read_csv(new_data_path)
    elif new_data_path.endswith('.xlsx'):
        df = pd.read_excel(new_data_path)
    else:
        raise ValueError("不支持的数据格式，仅支持CSV和Excel文件")

    # 2. 加载配置
    config_dir = os.path.dirname(model_path)
    config_path = os.path.join(config_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            global CONFIG
            CONFIG = json.load(f)
        print(f"已加载配置: {config_path}")

    # 3. 检查必要列是否存在
    required_cols = CONFIG["numeric_features"] + CONFIG["categorical_features"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"新数据中缺少必要列: {missing_cols}")

    # 4. 加载编码器
    encoders = load_encoders(encoders_path)

    # 5. 创建数据集（不使用标签列）
    print("创建预测数据集...")
    # 创建一个临时的标签列（预测时不会用到）
    if CONFIG["label_col"] not in df.columns:
        df[CONFIG["label_col"]] = 0  # 临时值

    pred_dataset = DeepFMDataset(
        df,
        numeric_features=CONFIG["numeric_features"],
        categorical_features=CONFIG["categorical_features"],
        label_col=CONFIG["label_col"],
        encoders=encoders,
        is_train=False
    )

    # 创建数据加载器
    pred_loader = DataLoader(pred_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=4)

    # 6. 初始化模型并加载权重
    print(f"加载模型: {model_path}")
    numeric_dim = len(CONFIG["numeric_features"])
    model = DeepFM(
        numeric_dim=numeric_dim,
        categorical_dims=pred_dataset.categorical_dims,
        embed_dim=CONFIG["embed_dim"],
        mlp_layers=CONFIG["mlp_layers"],
        dropout=CONFIG["dropout"]
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 7. 如果未指定阈值，尝试从配置中获取
    if threshold is None:
        try:
            with open(os.path.join(config_dir, "test_results.json"), "r") as f:
                test_results = json.load(f)
                threshold = test_results["best_threshold"]
                print(f"使用最佳阈值: {threshold}")
        except:
            threshold = CONFIG["positive_threshold"]
            print(f"未找到最佳阈值，使用默认值: {threshold}")

    # 8. 进行预测
    print("开始预测...")
    all_preds_proba = []

    with torch.no_grad():
        for batch in pred_loader:
            numeric = batch['numeric'].to(device)
            categorical = batch['categorical'].to(device)

            outputs = model(numeric, categorical)
            all_preds_proba.extend(outputs.cpu().numpy())

    # 转换为分类结果
    all_preds = (np.array(all_preds_proba) >= threshold).astype(int)

    # 9. 保存预测结果
    df['prediction_probability'] = all_preds_proba
    df['prediction'] = all_preds
    output_path = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(output_path, index=False)
    print(f"预测结果已保存至: {output_path}")

    return df


if __name__ == "__main__":
    # 示例用法:
    # 1. 训练新模型
    # results_dir, encoders = train_new_model()

    # 2. 继续训练（假设之前的结果目录是'results_20231020_153045'）
    # continue_training('results_20231020_153045', additional_epochs=10)

    # 3. 预测新数据（假设模型和编码器路径如下）
    # predict_new_data(
    #     model_path='results_20231020_153045/best_model.pth',
    #     encoders_path='results_20231020_153045/encoders.pkl',
    #     new_data_path='new_data.csv'
    # )

    # 默认执行训练新模型
    train_new_model()
