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
import warnings

warnings.filterwarnings('ignore')

# Configuration parameters
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
    # Predefined feature lists
    "numeric_features": ['age', 'last_30d_tel_succ_cs', 'loss_model_ggroup_v3',
                         'risk_ms11_1_model_score', 'standard_score_group_v6_1',
                         'last_month_expire_coupon_cnt', 'number_of_gold_be_used',
                         'last_10d_lin_e_cnt', 'last_10d_gu_e_cnt',
                         'ayht10_all_respond_score', 'call_anss_score_t10',
                         'xyl_model_107', 'avail_cash', 'avg_limuse_rate',
                         'pril_bal', 'crdt_lim_yx', 'lim_use_rate', 'zaidai_ctrl_rate'],
    "categorical_features": ['yls_cust_type_v2', 'cust_types_01', 'cust_types_02',
                             'if_sms_yn', 'elec_types', 'igender_cd', 'icust_ty',
                             'if_500up_yn', 'is_login', 'sms_types', 'if_bj_30_yn',
                             'member_types', 'if_bj_10_yn'],
    "label_col": "label",
    "data_folder": "data",
    # Missing value handling strategies
    "numeric_missing_strategy": "mean",  # Options: "mean", "median", "constant"
    "numeric_fill_value": 0,
    "categorical_missing_strategy": "constant",  # Options: "mode", "constant"
    "categorical_fill_value": "Missing",
    # Imbalance handling configuration
    "imbalance_method": "both",  # Options: 'weight', 'undersample', 'both'
    "undersample_ratio": 3,  # Negative:Positive ratio for undersampling
    # Threshold selection
    "threshold_tuning": False,
    "optimize_for": "f1",  # Options: 'f1', 'precision', 'recall'
    "positive_threshold": 0.5,
    "min_threshold": 0.1,
    "max_threshold": 0.9,
    "min_pos_samples": 20  # Minimum positive samples required per dataset
}

# === Load data from folder ===
def load_data_from_folder(folder_path):
    """Load all CSV files from folder and merge into one DataFrame"""
    all_files = []

    # Iterate through all files in folder
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            file_path = os.path.join(folder_path, file)
            try:
                df = pd.read_csv(file_path)
                all_files.append(df)
                print(f"Loaded {file} with {len(df)} rows")
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")

    if not all_files:
        raise ValueError(f"No CSV files found in {folder_path}")

    # Combine all DataFrames
    combined_df = pd.concat(all_files, axis=0, ignore_index=True)
    print(f"\nCombined dataset size: {len(combined_df)} rows")
    return combined_df

# === Handle missing values ===
def handle_missing_values(df, numeric_features, categorical_features):
    """Handle missing values according to configuration"""
    df = df.copy()

    # Handle numeric feature missing values
    for col in numeric_features:
        if col in df.columns and df[col].isnull().any():
            if CONFIG["numeric_missing_strategy"] == "mean":
                fill_value = df[col].mean()
            elif CONFIG["numeric_missing_strategy"] == "median":
                fill_value = df[col].median()
            else:  # constant
                fill_value = CONFIG["numeric_fill_value"]
            df[col] = df[col].fillna(fill_value)

    # Handle categorical feature missing values
    for col in categorical_features:
        if col in df.columns and df[col].isnull().any():
            if CONFIG["categorical_missing_strategy"] == "mode":
                fill_value = df[col].mode()[0]
            else:  # constant
                fill_value = CONFIG["categorical_fill_value"]
            df[col] = df[col].fillna(fill_value)

    return df

# === Undersampling function ===
def undersample_data(df, label_col, ratio=5):
    """Undersample majority class"""
    pos_df = df[df[label_col] == 1]
    neg_df = df[df[label_col] == 0]
    n_pos = len(pos_df)
    n_neg = min(len(neg_df), n_pos * ratio)
    neg_sample = neg_df.sample(n_neg, random_state=42)
    balanced_df = pd.concat([pos_df, neg_sample], axis=0).sample(frac=1, random_state=42)
    return balanced_df

# === Calculate detailed metrics ===
def calculate_detailed_metrics(y_true, y_pred_proba, threshold=0.5):
    """Calculate detailed metrics including per-class statistics"""
    # Convert to classification predictions
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Overall metrics
    acc = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    f1 = f1_score(y_true, y_pred)

    # Calculate precision and recall for both classes
    precision_neg = precision_score(y_true, y_pred, pos_label=0)
    recall_neg = recall_score(y_true, y_pred, pos_label=0)
    f1_neg = f1_score(y_true, y_pred, pos_label=0)

    precision_pos = precision_score(y_true, y_pred)
    recall_pos = recall_score(y_true, y_pred)
    f1_pos = f1_score(y_true, y_pred)

    # Class counts
    count_pos = sum(y_true)
    count_neg = len(y_true) - count_pos

    # Calculate AUC-PR (precision-recall curve area)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba)
    aucpr = auc(recall_curve, precision_curve)

    # Build results dictionary
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

# === Print detailed metrics ===
def print_detailed_metrics(metrics, dataset_name):
    """Print detailed classification metrics"""
    title = f"{dataset_name} Dataset Metrics"
    print("\n" + "=" * 70)
    print(title.center(70))
    print("=" * 70)

    # Overall metrics
    print(f"\nOverall Metrics:")
    print(f"  Accuracy: {metrics['overall']['accuracy']:.4f}")
    print(f"  ROC AUC: {metrics['overall']['roc_auc']:.4f}")
    print(f"  AUC-PR: {metrics['overall']['auc_pr']:.4f}")
    print(f"  F1 Score: {metrics['overall']['f1_score']:.4f}")
    print(f"  Prediction Threshold: {metrics['threshold']:.4f}")

    # Positive class metrics
    print("\nPositive Class (Minority) Metrics:")
    print(f"  Precision: {metrics['positive']['precision']:.4f}")
    print(f"  Recall: {metrics['positive']['recall']:.4f}")
    print(f"  F1 Score: {metrics['positive']['f1']:.4f}")
    print(f"  Support: {metrics['positive']['support']}")

    # Negative class metrics
    print("\nNegative Class (Majority) Metrics:")
    print(f"  Precision: {metrics['negative']['precision']:.4f}")
    print(f"  Recall: {metrics['negative']['recall']:.4f}")
    print(f"  F1 Score: {metrics['negative']['f1']:.4f}")
    print(f"  Support: {metrics['negative']['support']}")
    print("=" * 70 + "\n")

# === Save results ===
def save_results(trainer, config, test_metrics):
    """Save all training results to files"""
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    # 1. Save config file
    with open(os.path.join(results_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    # 2. Save training history
    train_history = []
    for i, metrics in enumerate(trainer.train_metrics_history):
        epoch_data = {
            "epoch": i+1,
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

    # 3. Save best model
    torch.save(trainer.best_model_state, os.path.join(results_dir, "best_model.pth"))

    # 4. Save final model
    torch.save(trainer.model.state_dict(), os.path.join(results_dir, "final_model.pth"))

    # 5. Save test results
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

    # 6. Save confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(trainer.test_confusion_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title('Test Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()

    return results_dir

# === DeepFM Dataset ===
class DeepFMDataset(Dataset):
    def __init__(self, df, numeric_features, categorical_features, label_col, encoders=None, is_train=True):
        self.df = df.copy().reset_index(drop=True)
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.label_col = label_col
        self.is_train = is_train

        # Handle missing values
        self.df = handle_missing_values(self.df, numeric_features, categorical_features)

        # Handle categorical features
        self.categorical_encoders = {} if encoders is None else encoders
        self.categorical_dims = {}
        self._encode_categorical()

    def _encode_categorical(self):
        for col in self.categorical_features:
            if col in self.df.columns:  # Ensure column exists
                if self.is_train or col not in self.categorical_encoders:
                    le = LabelEncoder()
                    self.df[col] = le.fit_transform(self.df[col].astype(str))
                    self.categorical_encoders[col] = le
                else:
                    le = self.categorical_encoders[col]
                    # Handle unseen categories - map to special value
                    self.df[col] = self.df[col].apply(lambda x: x if x in le.classes_ else 'UNKNOWN')
                    # Add 'UNKNOWN' to encoder classes if not present
                    if 'UNKNOWN' not in le.classes_:
                        le.classes_ = np.append(le.classes_, 'UNKNOWN')
                    self.df[col] = le.transform(self.df[col].astype(str))

                self.categorical_dims[col] = len(le.classes_)

        return self.categorical_encoders

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Numeric features
        numeric_data = self.df[self.numeric_features].iloc[idx].values if self.numeric_features else np.array([])
        numeric_tensor = torch.tensor(numeric_data.astype(np.float32), dtype=torch.float32)

        # Categorical features
        categorical_data = []
        for col in self.categorical_features:
            if col in self.df.columns:
                categorical_data.append(self.df.loc[idx, col])
            else:
                categorical_data.append(0)  # Default value

        categorical_tensor = torch.tensor(categorical_data, dtype=torch.long)

        # Label
        label = torch.tensor(self.df.loc[idx, self.label_col], dtype=torch.float32)

        return {
            'numeric': numeric_tensor,
            'categorical': categorical_tensor,
            'label': label
        }

# === DeepFM Model ===
class DeepFM(nn.Module):
    def __init__(self, numeric_dim, categorical_dims, embed_dim=32, mlp_layers=[256, 128, 64], dropout=0.3):
        super().__init__()

        # Ensure we have either numeric or categorical features
        self.has_numeric = numeric_dim > 0
        self.has_categorical = len(categorical_dims) > 0

        # ========== FM part ==========
        # FM first order (categorical features)
        self.fm_first_order_cat = nn.ModuleDict()
        if self.has_categorical:
            for col, num_embeddings in categorical_dims.items():
                self.fm_first_order_cat[col] = nn.Embedding(num_embeddings, 1)

        # FM first order (numeric features)
        if self.has_numeric:
            self.fm_first_order_num = nn.Linear(numeric_dim, 1)

        # FM second order (categorical features)
        self.fm_second_order = nn.ModuleDict()
        if self.has_categorical:
            for col, num_embeddings in categorical_dims.items():
                self.fm_second_order[col] = nn.Embedding(num_embeddings, embed_dim)

        # ========== DNN part ==========
        # DNN embeddings (categorical features)
        self.dnn_embeddings = nn.ModuleDict()
        if self.has_categorical:
            for col, num_embeddings in categorical_dims.items():
                self.dnn_embeddings[col] = nn.Embedding(num_embeddings, embed_dim)

        # DNN numeric feature processing
        if self.has_numeric:
            self.numeric_layer = nn.Linear(numeric_dim, embed_dim)

        # DNN fully connected layers
        input_dim = embed_dim * len(categorical_dims) + (embed_dim if self.has_numeric else 0)
        layers = []
        for i, out_dim in enumerate(mlp_layers):
            layers.append(nn.Linear(input_dim, out_dim))
            layers.append(nn.BatchNorm1d(out_dim))  # Add batch normalization
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = out_dim

        self.dnn = nn.Sequential(*layers) if layers else nn.Identity()

        # Output layer
        # FM part output dim: first order (1) + second order (1)
        fm_output_dim = 2 if self.has_categorical else 1
        # DNN part output dim
        dnn_output_dim = mlp_layers[-1] if mlp_layers else 0
        # Combined output layer
        self.output_layer = nn.Linear(fm_output_dim + dnn_output_dim, 1)

    def forward(self, numeric, categorical):
        batch_size = numeric.shape[0] if self.has_numeric else categorical.shape[0]
        components = []

        # ========== FM part ==========
        # First order features
        fm_first_total = torch.zeros(batch_size, 1, device=numeric.device if self.has_numeric else categorical.device)

        # Categorical first order terms
        fm_first_cat_list = []
        for i, col in enumerate(self.fm_first_order_cat):
            # Input shape: [batch_size] -> Output shape: [batch_size, 1]
            emb = self.fm_first_order_cat[col](categorical[:, i])
            fm_first_cat_list.append(emb)

        # Concatenate all embeddings: [batch_size, num_cat_features]
        fm_first_cat = torch.cat(fm_first_cat_list, dim=1)
        # Sum: [batch_size, 1]
        fm_first_cat = fm_first_cat.sum(dim=1, keepdim=True)
        fm_first_total += fm_first_cat

        # Numeric first order terms
        if self.has_numeric:
            fm_first_num = self.fm_first_order_num(numeric).view(-1, 1)
            fm_first_total += fm_first_num

        components.append(fm_first_total)

        # Second order features
        if self.has_categorical:
            fm_second_embeds = torch.cat([
                self.fm_second_order[col](categorical[:, i])
                for i, col in enumerate(self.fm_second_order)
            ], dim=1)  # (batch_size, embed_dim * num_cat_features)

            # FM second order calculation
            square_of_sum = torch.sum(fm_second_embeds, dim=1).pow(2)
            sum_of_square = torch.sum(fm_second_embeds.pow(2), dim=1)
            fm_second_order = 0.5 * (square_of_sum - sum_of_square).unsqueeze(1)
            components.append(fm_second_order)

        # ========== DNN part ==========
        dnn_embeds = []

        # Categorical feature embeddings
        if self.has_categorical:
            dnn_embeds.extend([
                self.dnn_embeddings[col](categorical[:, i])
                for i, col in enumerate(self.dnn_embeddings)
            ])

        # Numeric feature embedding
        if self.has_numeric:
            numeric_embed = self.numeric_layer(numeric)
            dnn_embeds.append(numeric_embed)

        if dnn_embeds:
            dnn_input = torch.cat(dnn_embeds, dim=1)
            dnn_output = self.dnn(dnn_input)
            components.append(dnn_output)

        # Combine all parts
        total = torch.cat(components, dim=1)

        # Final output
        output = self.output_layer(total)
        return torch.sigmoid(output).squeeze(1)

# === DeepFM Trainer ===
class DeepFMTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader, optimizer, device='cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.device = device

        # Training tracking
        self.best_auc = 0
        self.best_model_state = None
        self.best_threshold = CONFIG["positive_threshold"]
        self.train_metrics_history = []  # Store training metrics history
        self.val_metrics_history = []    # Store validation metrics history
        self.train_loss_history = []     # Store training loss history
        self.test_confusion_matrix = None  # Store test confusion matrix

    def train(self, class_weight=None):
        """Train the model"""
        # Define loss function (with or without weights)
        if class_weight is not None:
            class_weight_tensor = torch.tensor([class_weight[0], class_weight[1]],
                                               dtype=torch.float).to(self.device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=class_weight_tensor[1])
        else:
            criterion = nn.BCEWithLogitsLoss()

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.7,
            patience=3,
            min_lr=1e-5,
            verbose=True
        )

        for epoch in range(CONFIG["epochs"]):
            # Get and print current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}, Learning Rate: {current_lr:.6f}")

            start_time = time.time()
            self.model.train()
            total_loss, total_samples = 0, 0
            train_y_true, train_y_pred_proba = [], []
            epoch_grad_norms = []

            for batch in self.train_loader:
                numeric = batch['numeric'].to(self.device)
                categorical = batch['categorical'].to(self.device)
                labels = batch['label'].to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(numeric, categorical)  # Get logits before sigmoid
                loss = criterion(logits, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * numeric.size(0)
                total_samples += numeric.size(0)

                # Calculate gradient norms
                total_norm = 0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (0.5)
                epoch_grad_norms.append(total_norm)

                # Collect training predictions
                with torch.no_grad():
                    train_y_true.extend(labels.cpu().numpy())
                    train_y_pred_proba.extend(logits.cpu().numpy())

            # Print average gradient norms at end of epoch
            avg_grad_norm = sum(epoch_grad_norms) / len(epoch_grad_norms)
            min_grad_norm = min(epoch_grad_norms)
            max_grad_norm = max(epoch_grad_norms)
            print(f"Epoch {epoch+1} - Grad Norm: Avg={avg_grad_norm:.4f}, "
                  f"Min={min_grad_norm:.4f}, Max={max_grad_norm:.4f}")

            avg_loss = total_loss / total_samples
            self.train_loss_history.append(avg_loss)

            train_metrics = calculate_detailed_metrics(
                np.array(train_y_true),
                np.array(train_y_pred_proba),
                threshold=self.best_threshold
            )
            self.train_metrics_history.append(train_metrics)

            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{CONFIG['epochs']}, Train Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")
            print_detailed_metrics(train_metrics, "Training")

            # Validate and save best model
            val_metrics, threshold = self.validate()
            self.val_metrics_history.append(val_metrics)
            scheduler.step(val_metrics["overall"]["auc_pr"])

            # Save best model
            if val_metrics["overall"]["auc_pr"] > self.best_auc:
                self.best_auc = val_metrics["overall"]["auc_pr"]
                self.best_model_state = self.model.state_dict()
                self.best_threshold = threshold
                print(f"New best AUC-PR: {self.best_auc:.4f}, Threshold: {threshold:.4f}")

        print(f"Training complete. Best AUC-PR: {self.best_auc:.4f}, Best Threshold: {self.best_threshold:.4f}")
        return self.best_threshold

    def validate(self):
        """Evaluate on validation set"""
        self.model.eval()
        y_true, y_pred_proba = [], []

        with torch.no_grad():
            for batch in self.val_loader:
                numeric = batch['numeric'].to(self.device)
                categorical = batch['categorical'].to(self.device)
                labels = batch['label'].to(self.device)
                outputs = self.model(numeric, categorical)
                y_true.extend(labels.cpu().numpy())
                y_pred_proba.extend(outputs.cpu().numpy())

        # Use best threshold to calculate metrics
        if CONFIG["threshold_tuning"]:
            threshold = self.find_optimal_threshold(y_true, y_pred_proba)
        else:
            threshold = CONFIG["positive_threshold"]

        y_true = np.array(y_true)
        y_pred_proba = np.array(y_pred_proba)
        metrics = calculate_detailed_metrics(y_true, y_pred_proba, threshold)
        print_detailed_metrics(metrics, "Validation")
        return metrics, threshold

    def test(self):
        """Evaluate on test set"""
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        self.model.eval()
        y_true, y_pred_proba = [], []

        with torch.no_grad():
            for batch in self.test_loader:
                numeric = batch['numeric'].to(self.device)
                categorical = batch['categorical'].to(self.device)
                labels = batch['label'].to(self.device)
                outputs = self.model(numeric, categorical)
                y_true.extend(labels.cpu().numpy())
                y_pred_proba.extend(outputs.cpu().numpy())

        # Use best threshold from training
        if CONFIG["threshold_tuning"]:
            threshold = self.best_threshold
        else:
            threshold = CONFIG["positive_threshold"]

        y_true = np.array(y_true)
        y_pred_proba = np.array(y_pred_proba)
        metrics = calculate_detailed_metrics(y_true, y_pred_proba, threshold)
        print_detailed_metrics(metrics, "Test")

        # Plot confusion matrix
        self.test_confusion_matrix = self.plot_confusion_matrix(y_true, np.array(y_pred_proba) >= threshold)
        return metrics

    def find_optimal_threshold(self, y_true, y_pred_proba):
        """Find optimal threshold based on validation set"""
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)

        if CONFIG["optimize_for"] == "recall":
            # Maximize recall while maintaining reasonable precision
            optimal_idx = np.argmax(recall[:-1])
        elif CONFIG["optimize_for"] == "precision":
            # Maximize precision while maintaining reasonable recall
            optimal_idx = np.argmax(precision[:-1])
        else:  # Default optimize for F1 score
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            optimal_idx = np.argmax(f1_scores[:-1])

        threshold = thresholds[optimal_idx]
        return threshold

    def plot_confusion_matrix(self, y_true, y_pred, normalize=False):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300)
        plt.close()
        return cm

# === Main program ===
def main():
    # Load data from folder
    print(f"Loading data from folder: {CONFIG['data_folder']}")
    df = load_data_from_folder(CONFIG["data_folder"])
    label_col = CONFIG["label_col"]
    numeric_features = CONFIG["numeric_features"]
    categorical_features = CONFIG["categorical_features"]

    # Ensure label column exists
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in data")

    # Print initial data distribution
    original_class_dist = Counter(df[label_col])
    print("\n" + "=" * 70)
    print("Original Data Distribution".center(70))
    print("=" * 70)
    print(f"Positive (1): {original_class_dist[1]} samples")
    print(f"Negative (0): {original_class_dist[0]} samples")
    print(f"Imbalance Ratio: {original_class_dist[0]/original_class_dist[1]:.1f}:1")
    print("=" * 70 + "\n")

    # Handle imbalance (before splitting dataset)
    imbalance_method = CONFIG["imbalance_method"]
    class_weights = None

    if imbalance_method == "undersample":
        print("Applying undersampling to entire dataset...")
        df = undersample_data(df, label_col, CONFIG["undersample_ratio"])
    elif imbalance_method == "both":
        # Undersample entire dataset
        print("Applying undersampling to entire dataset...")
        df = undersample_data(df, label_col, CONFIG["undersample_ratio"])
        # Calculate class weights for loss function
        class_counts = df[label_col].value_counts()
        class_weights = [class_counts.sum() / (2.0 * class_counts[0]),
                         class_counts.sum() / (2.0 * class_counts[1])]
    elif imbalance_method == "weight":
        # Only use weight handling
        class_counts = df[label_col].value_counts()
        class_weights = [class_counts.sum() / (2.0 * class_counts[0]),
                         class_counts.sum() / (2.0 * class_counts[1])]
        print(f"Using class weights: Negative={class_weights[0]:.2f}, Positive={class_weights[1]:.2f}")

    # Print processed data distribution
    processed_class_dist = Counter(df[label_col])
    print("\n" + "=" * 70)
    print("Processed Data Distribution".center(70))
    print("=" * 70)
    print(f"Positive (1): {processed_class_dist[1]} samples")
    print(f"Negative (0): {processed_class_dist[0]} samples")
    print(f"Imbalance Ratio: {processed_class_dist[0]/processed_class_dist[1]:.1f}:1")
    print("=" * 70 + "\n")

    # Split dataset (with stratified sampling to ensure positive samples in each set)
    test_size = CONFIG["test_size"]
    val_size = CONFIG["val_size"]
    min_pos_samples = CONFIG["min_pos_samples"]

    # Ensure each set has enough positive samples
    if processed_class_dist[1] < min_pos_samples * 3:
        min_pos_samples = max(5, processed_class_dist[1] // 3)
        print(f"Adjusted min_pos_samples to {min_pos_samples} due to limited positive samples")

    # First generate test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[label_col],
        random_state=42
    )

    # Check test set positive count
    test_pos_count = test_df[label_col].sum()
    if test_pos_count < min_pos_samples:
        print(f"Warning: Test set has only {test_pos_count} positive samples. Adjusting split...")
        # Re-split to ensure test set has enough positive samples
        train_val_df, test_df = train_test_split(
            df,
            test_size=min(min_pos_samples/len(df), test_size),
            stratify=df[label_col],
            random_state=42
        )

    # Then generate validation set from remaining data
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size,
        stratify=train_val_df[label_col],
        random_state=42
    )

    # Check validation set positive count
    val_pos_count = val_df[label_col].sum()
    if val_pos_count < min_pos_samples:
        print(f"Warning: Validation set has only {val_pos_count} positive samples. Adjusting split...")
        # Re-split to ensure validation set has enough positive samples
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=min(min_pos_samples/len(train_val_df), val_size),
            stratify=train_val_df[label_col],
            random_state=42
        )

    # Print final dataset distributions
    print("\nFinal Dataset Distributions:")
    print(f"Train: {len(train_df)} samples ({train_df[label_col].sum()} positive)")
    print(f"Validation: {len(val_df)} samples ({val_df[label_col].sum()} positive)")
    print(f"Test: {len(test_df)} samples ({test_df[label_col].sum()} positive)")

    # Create datasets
    train_dataset = DeepFMDataset(
        train_df,
        numeric_features,
        categorical_features,
        label_col
    )
    encoders = train_dataset.categorical_encoders

    val_dataset = DeepFMDataset(
        val_df,
        numeric_features,
        categorical_features,
        label_col,
        encoders=encoders,
        is_train=False
    )

    test_dataset = DeepFMDataset(
        test_df,
        numeric_features,
        categorical_features,
        label_col,
        encoders=encoders,
        is_train=False
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"])

    # Initialize model
    numeric_dim = len(numeric_features)
    categorical_dims = train_dataset.categorical_dims

    model = DeepFM(
        numeric_dim=numeric_dim,
        categorical_dims=categorical_dims,
        embed_dim=CONFIG["embed_dim"],
        mlp_layers=CONFIG["mlp_layers"],
        dropout=CONFIG["dropout"]
    )

    # Print model info
    print("\nModel Architecture:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    # Define optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"]
    )

    # Create trainer
    trainer = DeepFMTrainer(
        model,
        train_loader,
        val_loader,
        test_loader,
        optimizer,
        device='cpu'
    )

    # Train model
    best_threshold = trainer.train(class_weight=class_weights)

    # Test set evaluation
    test_metrics = trainer.test()
    print("Training and evaluation completed successfully!")

    # Save results
    results_dir = save_results(trainer, CONFIG, test_metrics)
    print(f"All results saved to: {results_dir}")

if __name__ == "__main__":
    main()