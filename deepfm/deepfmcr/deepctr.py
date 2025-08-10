import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import DeepFM

def prepare_data(data_path=None):
    """准备数据，包括读取、预处理等"""
    # 如果没有提供数据路径，生成示例数据
    if data_path is None:
        # 生成示例数据
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

        # 目标变量（点击/不点击）
        target = np.random.randint(0, 2, size=n_samples)

        # 创建DataFrame
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
        # 从文件读取数据
        data = pd.read_csv(data_path)

    # 定义稀疏特征和稠密特征
    sparse_features = ['user_id', 'item_id', 'category_id']
    dense_features = ['age', 'income', 'score']

    # 处理稀疏特征：标签编码
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    # 处理稠密特征：归一化
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    # 计算每个稀疏特征的唯一值数量
    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(), embedding_dim=4)
                              for feat in sparse_features] + [DenseFeat(feat, 1, ) for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 划分训练集和测试集
    train, test = train_test_split(data, test_size=0.2, random_state=42)

    # 构建输入数据
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    return (train_model_input, test_model_input,
            train['target'].values, test['target'].values,
            linear_feature_columns, dnn_feature_columns)

def train_model(linear_feature_columns, dnn_feature_columns, train_model_input, train_labels,
                epochs=10, batch_size=256, model_path='deepfm_model.pth'):
    """训练模型并保存"""
    # 定义模型
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda:0'

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

    model.compile("adam", "binary_crossentropy",
                  metrics=["binary_crossentropy", "auc"])

    # 训练模型
    print(f"开始训练模型，使用设备: {device}")
    model.fit(train_model_input, train_labels,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_split=0.1)

    # 保存模型
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存至 {model_path}")

    return model

def load_model(linear_feature_columns, dnn_feature_columns, model_path='deepfm_model.pth'):
    """加载已保存的模型"""
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda:0'

    # 重建模型结构
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

    # 加载模型权重
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.to(device)
    model.eval()  # 设置为评估模式
    print(f"模型已从 {model_path} 加载")

    return model

def predict(model, test_model_input, batch_size=256):
    """使用模型进行预测"""
    print("开始预测...")
    pred_ans = model.predict(test_model_input, batch_size=batch_size)
    print("预测完成")
    return pred_ans

def evaluate_model(model, test_model_input, test_labels, batch_size=256):
    """评估模型性能"""
    print("开始评估模型...")
    result = model.evaluate(test_model_input, test_labels, batch_size=batch_size)
    print(f"评估结果: {model.metrics_names} = {result}")
    return result

def main():
    # 准备数据
    print("准备数据...")
    # 如果有自己的数据文件，可以传入路径，如 prepare_data("your_data.csv")
    train_model_input, test_model_input, train_labels, test_labels, linear_feature_columns, dnn_feature_columns = prepare_data()

    # 训练模型
    model = train_model(linear_feature_columns, dnn_feature_columns,
                        train_model_input, train_labels, epochs=10)

    # 评估模型
    evaluate_model(model, test_model_input, test_labels)

    # 使用模型进行预测
    predictions = predict(model, test_model_input)
    print(f"预测示例: {predictions[:5]}")

    # 加载模型
    loaded_model = load_model(linear_feature_columns, dnn_feature_columns)

    # 使用加载的模型进行预测
    loaded_model_predictions = predict(loaded_model, test_model_input)
    print(f"加载的模型预测示例: {loaded_model_predictions[:5]}")

if __name__ == "__main__":
    main()