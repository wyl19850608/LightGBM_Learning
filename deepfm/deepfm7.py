import pandas as pd
import numpy as np
import json
import os
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Multiply, Add, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib

class FeatureProcessor:
    def __init__(self):
        """初始化特征处理器"""
        self.config = {
            'categorical_features': [],  # 类别特征
            'continuous_features': {},   # 连续特征及分箱配置
            'bool_features': [],         # 布尔特征
            'label_encoders': {},        # 标签编码器
            'bin_discretizers': {},      # 分箱处理器
            'feature_sizes': {},         # 每个特征的可能取值数量
            'bin_rules': {}              # 分箱规则（边界等信息）
        }

    def fit(self, df, categorical_features, continuous_features_config, bool_features):
        """拟合特征处理器"""
        # 保存特征配置
        self.config['categorical_features'] = categorical_features
        self.config['continuous_features'] = continuous_features_config
        self.config['bool_features'] = bool_features

        # 处理类别特征
        for feature in categorical_features:
            le = LabelEncoder()
            # 处理缺失值
            df[feature] = df[feature].fillna('NaN')
            le.fit(df[feature].astype(str))
            self.config['label_encoders'][feature] = le
            self.config['feature_sizes'][feature] = len(le.classes_)

        # 处理连续特征
        for feature, config in continuous_features_config.items():
            n_bins = config['n_bins']
            strategy = config.get('strategy', 'quantile')
            # 动态调整分箱数，避免分箱过细
            if df[feature].nunique() < n_bins:
                n_bins = max(2, df[feature].nunique())  # 至少保留2个分箱
                print(f"特征 {feature} 唯一值太少，调整分箱数为 {n_bins}")

            binner = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy, subsample=None)
            # 处理缺失值
            df[feature] = df[feature].fillna(df[feature].median())
            binner.fit(df[[feature]])
            self.config['bin_discretizers'][feature] = binner
            self.config['feature_sizes'][feature] = n_bins

            # 记录分箱规则
            bin_edges = binner.bin_edges_[0].tolist()
            self.config['bin_rules'][feature] = {
                'n_bins': n_bins,
                'strategy': strategy,
                'bin_edges': bin_edges
            }

        # 处理布尔特征
        for feature in bool_features:
            le = LabelEncoder()
            # 处理缺失值
            df[feature] = df[feature].fillna(False)
            le.fit(df[feature].astype(str))
            self.config['label_encoders'][feature] = le
            self.config['feature_sizes'][feature] = len(le.classes_)

        return self

    def transform(self, df):
        """转换特征"""
        features = {}
        n_samples = df.shape[0]  # 记录样本数量用于校验

        # 处理类别特征
        for feature in self.config['categorical_features']:
            if feature not in df.columns:
                raise ValueError(f"特征 {feature} 不在数据中")
            # 处理缺失值
            df[feature] = df[feature].fillna('NaN')
            # 处理未见过的类别
            df[feature] = df[feature].apply(
                lambda x: x if x in self.config['label_encoders'][feature].classes_ else 'NaN'
            )
            encoded = self.config['label_encoders'][feature].transform(df[feature].astype(str))
            if len(encoded) != n_samples:
                raise ValueError(f"类别特征 {feature} 转换后样本数不匹配: {len(encoded)} vs {n_samples}")
            features[feature] = encoded

        # 处理连续特征
        for feature in self.config['continuous_features'].keys():
            if feature not in df.columns:
                raise ValueError(f"特征 {feature} 不在数据中")
            # 处理缺失值
            df[feature] = df[feature].fillna(df[feature].median())
            # 转换为分箱索引
            binned = self.config['bin_discretizers'][feature].transform(df[[feature]]).flatten()
            if len(binned) != n_samples:
                raise ValueError(f"连续特征 {feature} 转换后样本数不匹配: {len(binned)} vs {n_samples}")
            features[feature] = binned.astype(int)

        # 处理布尔特征
        for feature in self.config['bool_features']:
            if feature not in df.columns:
                raise ValueError(f"特征 {feature} 不在数据中")
            # 处理缺失值
            df[feature] = df[feature].fillna(False)
            # 处理未见过的值
            df[feature] = df[feature].apply(
                lambda x: x if str(x) in self.config['label_encoders'][feature].classes_ else 'False'
            )
            encoded = self.config['label_encoders'][feature].transform(df[feature].astype(str))
            if len(encoded) != n_samples:
                raise ValueError(f"布尔特征 {feature} 转换后样本数不匹配: {len(encoded)} vs {n_samples}")
            features[feature] = encoded

        # 转换为适合模型输入的格式：确保每个特征是numpy数组
        feature_list = [np.asarray(features[feature]) for feature in features]
        return features, feature_list

    def save_config(self, path):
        """保存特征处理器配置，包括分箱规则"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        encoders_path = os.path.splitext(path)[0] + '_encoders'
        os.makedirs(encoders_path, exist_ok=True)

        # 保存编码器和分箱器
        for feature, le in self.config['label_encoders'].items():
            joblib.dump(le, os.path.join(encoders_path, f'{feature}_label_encoder.pkl'))
        for feature, binner in self.config['bin_discretizers'].items():
            joblib.dump(binner, os.path.join(encoders_path, f'{feature}_bin_discretizer.pkl'))

        # 保存配置（包含分箱规则）
        config_to_save = self.config.copy()
        config_to_save['label_encoders'] = list(config_to_save['label_encoders'].keys())
        config_to_save['bin_discretizers'] = list(config_to_save['bin_discretizers'].keys())

        with open(path, 'w') as f:
            json.dump(config_to_save, f, indent=4)

    def load_config(self, path):
        """加载特征处理器配置"""
        with open(path, 'r') as f:
            self.config = json.load(f)

        encoders_path = os.path.splitext(path)[0] + '_encoders'
        # 加载标签编码器
        label_encoders = {}
        for feature in self.config['label_encoders']:
            label_encoders[feature] = joblib.load(os.path.join(encoders_path, f'{feature}_label_encoder.pkl'))
        self.config['label_encoders'] = label_encoders

        # 加载分箱器
        bin_discretizers = {}
        for feature in self.config['bin_discretizers']:
            bin_discretizers[feature] = joblib.load(os.path.join(encoders_path, f'{feature}_bin_discretizer.pkl'))
        self.config['bin_discretizers'] = bin_discretizers

        return self


class DeepFM:
    def __init__(self, feature_sizes, embedding_size=8, dnn_hidden_units=[256, 128, 64]):
        """初始化DeepFM模型"""
        self.feature_sizes = feature_sizes
        self.embedding_size = embedding_size
        self.dnn_hidden_units = dnn_hidden_units
        self.model = self.build_model()

    def build_model(self):
        """构建DeepFM模型"""
        # 输入层
        inputs = {}
        for feature, size in self.feature_sizes.items():
            inputs[feature] = Input(shape=(1,), name=f'input_{feature}')

        # 嵌入层
        embeddings = {}
        linear_terms = []
        for feature, size in self.feature_sizes.items():
            # 线性部分嵌入（维度1）
            linear_emb = Embedding(size, 1, name=f'linear_emb_{feature}')(inputs[feature])
            linear_emb = Flatten(name=f'linear_flat_{feature}')(linear_emb)
            linear_terms.append(linear_emb)

            # FM和DNN部分嵌入（维度embedding_size）
            emb = Embedding(size, self.embedding_size, name=f'emb_{feature}')(inputs[feature])
            emb = Flatten(name=f'flat_{feature}')(emb)
            embeddings[feature] = emb

        # FM部分
        # 线性部分
        linear_part = Add(name='linear_sum')(linear_terms)

        # 二阶交叉部分
        embedding_list = list(embeddings.values())
        sum_square = tf.square(Add(name='sum_embeddings')(embedding_list))
        square_sum = Add(name='sum_squares')([tf.square(emb) for emb in embedding_list])
        fm_part = Multiply(name='fm_multiply')([sum_square, square_sum])
        fm_part = Dense(1, name='fm_output')(fm_part)

        # DNN部分
        dnn_input = Concatenate(name='dnn_concat')(embedding_list)
        dnn = dnn_input
        for i, units in enumerate(self.dnn_hidden_units):
            dnn = Dense(units, activation='relu', name=f'dnn_{i}')(dnn)
            dnn = BatchNormalization(name=f'batch_norm_{i}')(dnn)
            dnn = Dropout(0.3, name=f'dropout_{i}')(dnn)
        dnn_part = Dense(1, activation='relu', name='dnn_output')(dnn)

        # 合并所有部分
        output = Add(name='total_sum')([linear_part, fm_part, dnn_part])
        output = Dense(1, activation='sigmoid', name='output')(output)

        # 构建模型
        model = Model(inputs=list(inputs.values()), outputs=output)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=256, model_path='deepfm_model.h5'):
        """训练模型"""
        # 回调函数
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        model_checkpoint = ModelCheckpoint(
            model_path,
            monitor='val_accuracy',
            save_best_only=True
        )

        # 训练模型
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, model_checkpoint]
        )

        return history

    def predict(self, X):
        """预测概率"""
        return self.model.predict(X)

    def evaluate(self, X, y):
        """评估模型"""
        y_pred_proba = self.predict(X).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)

        metrics = {
            'auc': roc_auc_score(y, y_pred_proba),
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred)
        }

        return metrics

    def save_model(self, path):
        """保存模型"""
        self.model.save(path)

    def load_model(self, path):
        """加载模型"""
        self.model = tf.keras.models.load_model(path)


def main():
    # 1. 配置参数
    data_path = 'full_credit_data.csv'  # 数据路径
    model_path = 'models/deepfm_model.h5'  # 模型保存路径
    feature_config_path = 'models/feature_config.json'  # 特征配置保存路径
    embedding_size = 16  # 嵌入维度
    dnn_hidden_units = [256, 128, 64]  # DNN隐藏层
    test_size = 0.2  # 测试集比例
    random_state = 42  # 随机种子

    # 2. 定义特征类型
    # 类别特征（字符串或编码）
    categorical_features = [
        'gender_cd', 'ocr_ethnic', 'residence_pr_cd', 'residence_city_cd',
        'residence_area_cd', 'census_pr_cd', 'census_city_cd', 'census_area_cd',
        'occ_cd', 'first_biz_name', 'second_biz_name', 'third_biz_name',
        'db_src', 'prod_cd', 'cust_gp_code', 'cust_gp_name', 'touch_name',
        'touch_id', 'touch_type', 'channel_name', 'channel_id', 'plan_id',
        'plan_name', 'channel_task_id', 'subscribe_no', 'decision_id',
        'decision_name', 'tel_call_type', 'touch_status', 'click_status'
    ]

    # 连续特征及分箱配置
    continuous_features_config = {
        'age': {'n_bins': 10, 'strategy': 'quantile'},
        'pril_bal': {'n_bins': 10, 'strategy': 'quantile'},
        'crdt_lim_yx': {'n_bins': 10, 'strategy': 'quantile'},
        'avail_bal_cash': {'n_bins': 10, 'strategy': 'quantile'},
        'total_loan_cnt': {'n_bins': 8, 'strategy': 'quantile'},
        'total_loan_amt': {'n_bins': 10, 'strategy': 'quantile'},
        'apply_cnt': {'n_bins': 8, 'strategy': 'quantile'},
        'apply_amt': {'n_bins': 10, 'strategy': 'quantile'},
        'wdraw_cnt': {'n_bins': 8, 'strategy': 'quantile'},
        'wdraw_amt': {'n_bins': 10, 'strategy': 'quantile'},
        'lim_use_rate': {'n_bins': 10, 'strategy': 'uniform'},
        'push_cnt': {'n_bins': 5, 'strategy': 'quantile'},
        'sms_charge_cnt': {'n_bins': 5, 'strategy': 'quantile'},
        'tel_call_dur': {'n_bins': 8, 'strategy': 'quantile'},
        'tel_call_inav_cnt': {'n_bins': 5, 'strategy': 'quantile'},
        'delay_days': {'n_bins': 8, 'strategy': 'quantile'},
        'wdraw_amt_t0': {'n_bins': 8, 'strategy': 'quantile'},
        'wdraw_amt_t3': {'n_bins': 8, 'strategy': 'quantile'},
        'wdraw_amt_t7': {'n_bins': 8, 'strategy': 'quantile'},
        'wdraw_amt_t10': {'n_bins': 8, 'strategy': 'quantile'},
        'wdraw_amt_t14': {'n_bins': 8, 'strategy': 'quantile'},
        'wdraw_amt_t15': {'n_bins': 8, 'strategy': 'quantile'},
        'wdraw_amt_t30': {'n_bins': 8, 'strategy': 'quantile'}
    }

    # 布尔特征
    bool_features = [
        'is_coupon_issue', 'is_credit', 'is_apply', 'is_wdraw', 'is_login',
        'if_bj_10_yn', 'if_bj_30_yn'
    ]

    # 目标变量
    target = 'aprv_status'

    # 3. 加载数据
    print("加载数据...")
    df = pd.read_csv(data_path)
    print(f"原始数据样本数: {df.shape[0]}")

    # 处理目标变量
    print("处理目标变量...")
    le_target = LabelEncoder()
    df[target] = le_target.fit_transform(df[target])

    # 4. 特征工程
    print("进行特征工程...")
    feature_processor = FeatureProcessor()
    feature_processor.fit(df, categorical_features, continuous_features_config, bool_features)

    # 转换特征
    _, feature_list = feature_processor.transform(df)

    # 严格验证特征与样本数量一致
    n_samples = df.shape[0]
    for i, feat_array in enumerate(feature_list):
        if len(feat_array) != n_samples:
            raise ValueError(f"特征 {i} 样本数不匹配: {len(feat_array)} vs {n_samples}")
    print(f"特征转换后样本数: {len(feature_list[0]) if feature_list else 0}")
    print(f"特征数量: {len(feature_list)}")

    # 5. 划分训练集和测试集（核心修复：通过索引手动划分）
    print("划分训练集和测试集...")
    X = feature_list
    y = df[target].values

    # 生成样本索引并划分
    indices = np.arange(n_samples)  # 创建0到n_samples-1的索引
    train_indices, val_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )

    # 根据索引分割每个特征
    X_train = [feat[train_indices] for feat in X]  # 每个特征都按训练索引分割
    X_val = [feat[val_indices] for feat in X]      # 每个特征都按验证索引分割
    y_train = y[train_indices]
    y_val = y[val_indices]

    # 验证划分后的样本数
    print(f"训练集样本数: {len(y_train)}, 验证集样本数: {len(y_val)}")

    # 6. 构建并训练模型
    print("构建并训练模型...")
    deepfm = DeepFM(
        feature_sizes=feature_processor.config['feature_sizes'],
        embedding_size=embedding_size,
        dnn_hidden_units=dnn_hidden_units
    )

    # 训练模型
    history = deepfm.train(
        X_train, y_train,
        X_val, y_val,
        epochs=50,
        batch_size=256,
        model_path=model_path
    )

    # 7. 评估模型
    print("评估模型...")
    metrics = deepfm.evaluate(X_val, y_val)
    print("模型评估指标:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # 8. 保存特征配置
    print("保存特征配置...")
    feature_processor.save_config(feature_config_path)

    print("训练完成!")


def predict_new_data(data_path, model_path, feature_config_path):
    """预测新数据"""
    # 加载数据
    df = pd.read_csv(data_path)

    # 加载特征处理器
    feature_processor = FeatureProcessor()
    feature_processor.load_config(feature_config_path)

    # 转换特征
    _, feature_list = feature_processor.transform(df)

    # 加载模型
    deepfm = DeepFM(feature_sizes=feature_processor.config['feature_sizes'])
    deepfm.load_model(model_path)

    # 预测
    predictions = deepfm.predict(feature_list)

    # 输出预测结果
    df['prediction_prob'] = predictions.flatten()
    df['prediction'] = (predictions.flatten() > 0.5).astype(int)

    return df


if __name__ == "__main__":
    main()
    # 如果需要预测新数据，可以使用以下代码
    # predicted_df = predict_new_data('new_data.csv', 'models/deepfm_model.h5', 'models/feature_config.json')
    # predicted_df.to_csv('predicted_results.csv', index=False)