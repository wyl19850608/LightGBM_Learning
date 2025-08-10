import tensorflow as tf
from tensorflow.keras import Model, optimizers, losses, metrics
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import math
import model_structure.custom_model.deepfm as deepfm
import model_structure.custom_model.mmoe as mmoe
from model_structure.custom_layer.custom_dense import CustomizedDenseLayer as Dense
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

class DeepFMTrainer:
    def __init__(self, config):
        """初始化DeepFM训练器

        Args:
            config: 配置字典，包含模型参数和训练参数
        """
        self.config = config
        self.model = None
        self.linear_feature_columns = None
        self.dnn_feature_columns = None
        self.scaler = MinMaxScaler()
        self.label_encoders = {}

    def preprocess_data(self, data, is_train=True):
        """数据预处理

        Args:
            data: 原始数据DataFrame
            is_train: 是否为训练数据

        Returns:
            处理后的特征和标签
        """
        # 分离特征和标签
        if 'label' in data.columns:
            labels = data['label'].values
            features = data.drop('label', axis=1)
        else:
            labels = None
            features = data

        # 区分离散特征和连续特征
        sparse_features = [col for col in features.columns if features[col].dtype == 'object']
        dense_features = [col for col in features.columns if col not in sparse_features]

        # 处理连续特征
        if dense_features:
            if is_train:
                features[dense_features] = self.scaler.fit_transform(features[dense_features])
            else:
                features[dense_features] = self.scaler.transform(features[dense_features])

        # 处理离散特征
        for feat in sparse_features:
            if is_train:
                self.label_encoders[feat] = LabelEncoder()
                features[feat] = self.label_encoders[feat].fit_transform(features[feat].fillna('-1'))
            else:
                features[feat] = self.label_encoders[feat].transform(features[feat].fillna('-1'))

        # 构建特征列
        if is_train:
            self.linear_feature_columns, self.dnn_feature_columns = self._build_feature_columns(
                features, sparse_features, dense_features)

        # 准备模型输入
        input_dict = {name: features[name].values for name in features.columns}

        return input_dict, labels

    def _build_feature_columns(self, features, sparse_features, dense_features):
        """构建特征列

        Args:
            features: 特征DataFrame
            sparse_features: 离散特征列表
            dense_features: 连续特征列表

        Returns:
            线性部分特征列和DNN部分特征列
        """
        linear_feature_columns = []
        dnn_feature_columns = []

        # 处理离散特征
        for feat in sparse_features:
            vocab_size = features[feat].nunique()
            # 为FM部分创建嵌入特征
            fm_embedding = tf.feature_column.embedding_column(
                tf.feature_column.categorical_column_with_identity(feat, vocab_size=vocab_size),
                dimension=self.config['embedding_dim']
            )
            linear_feature_columns.append(tf.feature_column.categorical_column_with_identity(feat, vocab_size=vocab_size))
            dnn_feature_columns.append(fm_embedding)

        # 处理连续特征
        for feat in dense_features:
            linear_feature_columns.append(tf.feature_column.numeric_column(feat))
            dnn_feature_columns.append(tf.feature_column.numeric_column(feat))

        return linear_feature_columns, dnn_feature_columns

    def build_model(self):
        """构建DeepFM模型"""
        self.model = deepfm.DeepFM(
            linear_feature_columns=self.linear_feature_columns,
            dnn_feature_columns=self.dnn_feature_columns,
            dnn_hidden_units=self.config['dnn_hidden_units'],
            dnn_dropout=self.config['dnn_dropout'],
            dnn_activation=self.config['dnn_activation'],
            task=self.config['task']
        )

        # 编译模型
        optimizer = optimizers.Adam(learning_rate=self.config['learning_rate'])

        if self.config['task'] == 'binary':
            self.model.compile(
                optimizer=optimizer,
                loss=losses.BinaryCrossentropy(),
                metrics=[metrics.AUC(name='auc'), metrics.BinaryAccuracy(name='accuracy')]
            )
        elif self.config['task'] == 'regression':
            self.model.compile(
                optimizer=optimizer,
                loss=losses.MeanSquaredError(),
                metrics=[metrics.MeanAbsoluteError(name='mae')]
            )
        elif self.config['task'] == 'multiclass':
            self.model.compile(
                optimizer=optimizer,
                loss=losses.CategoricalCrossentropy(),
                metrics=[metrics.CategoricalAccuracy(name='accuracy')]
            )

    def train(self, train_data, val_data=None):
        """训练模型

        Args:
            train_data: 训练数据DataFrame
            val_data: 验证数据DataFrame，可选
        """
        # 数据预处理
        train_input, train_labels = self.preprocess_data(train_data, is_train=True)

        # 构建模型
        self.build_model()

        # 准备回调函数
        callbacks = [
            EarlyStopping(monitor=self.config['early_stopping_monitor'],
                          patience=self.config['early_stopping_patience'],
                          mode='max' if 'auc' in self.config['early_stopping_monitor'] else 'min'),
            ModelCheckpoint(self.config['model_save_path'],
                            monitor=self.config['early_stopping_monitor'],
                            save_best_only=True,
                            mode='max' if 'auc' in self.config['early_stopping_monitor'] else 'min'),
            TensorBoard(log_dir=self.config['tensorboard_log_dir'])
        ]

        # 如果有验证数据
        validation_data = None
        if val_data is not None:
            val_input, val_labels = self.preprocess_data(val_data, is_train=False)
            validation_data = (val_input, val_labels)

        # 训练模型
        self.model.fit(
            x=train_input,
            y=train_labels,
            batch_size=self.config['batch_size'],
            epochs=self.config['epochs'],
            validation_data=validation_data,
            callbacks=callbacks,
            shuffle=True
        )

    def load_model(self, model_path=None):
        """加载已保存的模型

        Args:
            model_path: 模型路径，默认为配置中的路径
        """
        model_path = model_path or self.config['model_save_path']
        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects={'CustomizedDenseLayer': Dense}
        )

    def predict(self, data):
        """预测函数

        Args:
            data: 待预测的数据DataFrame

        Returns:
            预测结果
        """
        input_dict, _ = self.preprocess_data(data, is_train=False)
        predictions = self.model.predict(input_dict)

        # 如果是分类任务，返回类别或概率
        if self.config['task'] == 'binary':
            # 返回概率和预测标签
            pred_labels = (predictions > self.config['threshold']).astype(int).flatten()
            return {
                'probabilities': predictions.flatten(),
                'labels': pred_labels
            }
        elif self.config['task'] == 'multiclass':
            # 返回概率和预测标签
            pred_labels = np.argmax(predictions, axis=1)
            return {
                'probabilities': predictions,
                'labels': pred_labels
            }
        else:  # 回归任务
            return {'predictions': predictions.flatten()}

    def evaluate(self, test_data):
        """评估模型

        Args:
            test_data: 测试数据DataFrame

        Returns:
            评估指标字典
        """
        test_input, test_labels = self.preprocess_data(test_data, is_train=False)
        results = self.model.evaluate(test_input, test_labels, verbose=0)

        metrics_dict = dict(zip(self.model.metrics_names, results))

        # 计算额外的评估指标
        predictions = self.predict(test_data)

        if self.config['task'] == 'binary':
            metrics_dict['roc_auc'] = roc_auc_score(test_labels, predictions['probabilities'])
            metrics_dict['precision'] = precision_score(test_labels, predictions['labels'])
            metrics_dict['recall'] = recall_score(test_labels, predictions['labels'])
        elif self.config['task'] == 'multiclass':
            metrics_dict['precision'] = precision_score(test_labels, predictions['labels'], average='macro')
            metrics_dict['recall'] = recall_score(test_labels, predictions['labels'], average='macro')

        return metrics_dict

# 使用示例
if __name__ == "__main__":
    # 配置参数
    config = {
        'embedding_dim': 8,
        'dnn_hidden_units': [256, 128, 64],
        'dnn_dropout': 0.5,
        'dnn_activation': 'relu',
        'task': 'binary',  # 可选: 'binary', 'regression', 'multiclass'
        'learning_rate': 0.001,
        'batch_size': 256,
        'epochs': 50,
        'early_stopping_monitor': 'val_auc',
        'early_stopping_patience': 5,
        'model_save_path': 'deepfm_best_model.h5',
        'tensorboard_log_dir': './logs',
        'threshold': 0.5
    }

    # 示例：生成模拟数据
    def generate_sample_data(n_samples=10000, n_sparse_features=10, n_dense_features=5):
        """生成样本数据用于测试"""
        data = {}

        # 生成离散特征
        for i in range(n_sparse_features):
            data[f'sparse_{i}'] = np.random.randint(0, 10, size=n_samples)

        # 生成连续特征
        for i in range(n_dense_features):
            data[f'dense_{i}'] = np.random.normal(0, 1, size=n_samples)

        # 生成标签（简单的线性组合用于模拟）
        label = np.zeros(n_samples)
        for i in range(n_sparse_features):
            label += data[f'sparse_{i}'] * 0.1
        for i in range(n_dense_features):
            label += data[f'dense_{i}'] * 0.2

        # 二值化标签用于分类任务
        label = (label > label.mean()).astype(int)
        data['label'] = label

        return pd.DataFrame(data)

    # 生成数据
    data = generate_sample_data(n_samples=50000)

    # 划分训练集、验证集和测试集
    train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=1/3, random_state=42)

    # 初始化并训练模型
    trainer = DeepFMTrainer(config)
    trainer.train(train_data, val_data)

    # 评估模型
    test_metrics = trainer.evaluate(test_data)
    print("测试集评估指标:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")

    # 预测示例
    sample_data = test_data.drop('label', axis=1).sample(5)
    predictions = trainer.predict(sample_data)
    print("\n预测示例:")
    print("概率:", [f"{p:.4f}" for p in predictions['probabilities']])
    print("预测标签:", predictions['labels'])
