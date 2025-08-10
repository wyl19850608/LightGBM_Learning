import pandas as pd
import numpy as np
import torch
import joblib  # 用于保存和加载scaler
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import DeepFM

def prepare_data(data_path=None, scaler_path='minmax_scaler.pkl'):
    """准备数据，包括读取、预处理，并保存scaler"""
    # 如果没有提供数据路径，生成示例数据
    if data_path is None:
        np.random.seed(42)
        n_samples = 10000

        # 稀疏特征
        user_id = np.random.randint(1, 1001, size=n_samples)
        item_id = np.random.randint(1, 501, size=n_samples)
        category_id = np.random.randint(1, 51, size=n_samples)

        # 稠密特征
        age = np.random.randint(18, 65, size=n_samples)
        income = np.random.randint(3000, 20000, size=n_samples)
        score = np.random.rand(n_samples) * 5

        # 目标变量
        target = np.random.randint(0, 2, size=n_samples)

        data = pd.DataFrame({
            'user_id': user_id,
            'item_id': item_id,
            'category_id': category_id,
            'age': age,
            'income': income,
            'score': score,
            'target': target
        })
    else:
        data = pd.read_csv(data_path)

    # 定义特征
    sparse_features = ['user_id', 'item_id', 'category_id']
    dense_features = ['age', 'income', 'score']

    # 处理稀疏特征（标签编码）
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    # 处理稠密特征（归一化）
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])  # 训练时拟合
    joblib.dump(mms, scaler_path)  # 保存scaler到文件
    print(f"MinMaxScaler已保存至 {scaler_path}")

    # 构建特征列（优化后区分线性和DNN的稀疏特征处理）
    # 线性部分稀疏特征（embedding_dim=1，类似one-hot）
    linear_sparse = [
        SparseFeat(feat, vocabulary_size=data[feat].nunique(), embedding_dim=1)
        for feat in sparse_features
    ]
    # DNN部分稀疏特征（embedding_dim=4）
    dnn_sparse = [
        SparseFeat(feat, vocabulary_size=data[feat].nunique(), embedding_dim=4)
        for feat in sparse_features
    ]
    # 稠密特征（两部分共用）
    dense_cols = [DenseFeat(feat, 1) for feat in dense_features]

    linear_feature_columns = linear_sparse + dense_cols
    dnn_feature_columns = dnn_sparse + dense_cols

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 划分数据集
    train, test = train_test_split(data, test_size=0.2, random_state=42)

    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    return (train_model_input, test_model_input,
            train['target'].values, test['target'].values,
            linear_feature_columns, dnn_feature_columns)

def preprocess_new_data(new_data, sparse_features, dense_features,
                        label_encoders=None, scaler_path='minmax_scaler.pkl'):
    """预处理新数据（用于预测），使用训练时保存的scaler和label_encoder"""
    # 加载训练时保存的scaler
    mms = joblib.load(scaler_path)

    # 处理稀疏特征（使用训练时的label_encoder）
    if label_encoders is None:
        # 如果没有传入编码器，初始化（实际应从训练时保存的文件加载）
        label_encoders = {}
        for feat in sparse_features:
            lbe = LabelEncoder()
            label_encoders[feat] = lbe.fit(new_data[feat])  # 仅用于示例，实际应加载训练好的

    for feat in sparse_features:
        # 使用训练时的编码器转换，未见过的类别可填充-1或其他默认值
        new_data[feat] = label_encoders[feat].transform(new_data[feat])

    # 处理稠密特征（使用训练时的scaler转换，避免重新拟合）
    new_data[dense_features] = mms.transform(new_data[dense_features])  # 关键：用transform而非fit_transform

    return new_data

def train_model(linear_feature_columns, dnn_feature_columns, train_model_input, train_labels,
                epochs=10, batch_size=256, model_path='deepfm_model.pth'):
    """训练模型并保存"""
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model = DeepFM(linear_feature_columns=linear_feature_columns,
                   dnn_feature_columns=dnn_feature_columns,
                   task='binary',
                   l2_reg_linear=0.00001,
                   l2_reg_embedding=0.00001,
                   l2_reg_dnn=0,
                   init_std=0.0001,
                   seed=1024,
                   dnn_dropout=0.2,
                   dnn_hidden_units=(256, 128, 64),
                   dnn_activation='relu',
                   device=device)

    model.compile("adam", "binary_crossentropy", metrics=["binary_crossentropy", "auc"])

    print(f"开始训练模型，使用设备: {device}")
    model.fit(train_model_input, train_labels,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_split=0.1)

    torch.save(model.state_dict(), model_path)
    print(f"模型已保存至 {model_path}")

    return model

def load_model(linear_feature_columns, dnn_feature_columns, model_path='deepfm_model.pth'):
    """加载已保存的模型"""
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model = DeepFM(linear_feature_columns=linear_feature_columns,
                   dnn_feature_columns=dnn_feature_columns,
                   task='binary',
                   l2_reg_linear=0.00001,
                   l2_reg_embedding=0.00001,
                   l2_reg_dnn=0,
                   init_std=0.0001,
                   seed=1024,
                   dnn_dropout=0.2,
                   dnn_hidden_units=(256, 128, 64),
                   dnn_activation='relu',
                   device=device)

    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.to(device)
    model.eval()
    print(f"模型已从 {model_path} 加载")

    return model

def predict(model, test_model_input, batch_size=256):
    """预测函数"""
    print("开始预测...")
    pred_ans = model.predict(test_model_input, batch_size=batch_size)
    print("预测完成")
    return pred_ans

def main():
    # 准备数据（会保存scaler）
    print("准备数据...")
    train_model_input, test_model_input, train_labels, test_labels, \
        linear_feature_columns, dnn_feature_columns = prepare_data()

    # 训练模型
    model = train_model(linear_feature_columns, dnn_feature_columns,
                        train_model_input, train_labels, epochs=10)

    # 评估模型
    print("评估模型...")
    model.evaluate(test_model_input, test_labels)

    # 模拟新数据预测（示例）
    # 1. 生成新数据（实际中可能是外部输入）
    new_data = pd.DataFrame({
        'user_id': [100, 200],
        'item_id': [50, 60],
        'category_id': [5, 6],
        'age': [30, 40],
        'income': [8000, 15000],
        'score': [3.5, 4.2]
    })
    # 2. 预处理新数据（使用训练时的scaler）
    sparse_features = ['user_id', 'item_id', 'category_id']
    dense_features = ['age', 'income', 'score']
    processed_new_data = preprocess_new_data(new_data, sparse_features, dense_features)
    # 3. 构建模型输入
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
    new_model_input = {name: processed_new_data[name] for name in feature_names}
    # 4. 预测
    predictions = predict(model, new_model_input)
    print(f"新数据预测结果: {predictions}")

if __name__ == "__main__":
    main()
