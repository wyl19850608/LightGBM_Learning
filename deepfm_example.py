import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from deepctr.models import DeepFM
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names

def prepare_data():
    """准备示例数据并进行预处理"""
    # 生成示例数据
    data = {
        # 稀疏特征
        'user_id': ['user1', 'user2', 'user3', 'user4', 'user5', 'user1', 'user2', 'user3'] * 125,
        'item_id': ['item1', 'item2', 'item3', 'item4', 'item5', 'item1', 'item2', 'item3'] * 125,
        'category_id': ['cat1', 'cat2', 'cat1', 'cat2', 'cat1', 'cat1', 'cat2', 'cat1'] * 125,
        # 稠密特征
        'price': [10.0, 20.0, 15.0, 25.0, 30.0, 12.0, 18.0, 16.0] * 125,
        'click_count': [1, 2, 3, 0, 1, 2, 3, 4] * 125,
        # 目标变量：是否点击
        'label': [1, 0, 1, 0, 1, 0, 1, 0] * 125
    }

    df = pd.DataFrame(data)

    # 1. 对稀疏特征进行编码
    sparse_features = ['user_id', 'item_id', 'category_id']
    for feat in sparse_features:
        lbe = LabelEncoder()
        df[feat] = lbe.fit_transform(df[feat])

    # 2. 对稠密特征进行归一化
    dense_features = ['price', 'click_count']
    mms = MinMaxScaler(feature_range=(0, 1))
    df[dense_features] = mms.fit_transform(df[dense_features])

    # 3. 设置特征列
    fixlen_feature_columns = [
                                 # 稀疏特征配置：名称、唯一值数量、嵌入维度
                                 SparseFeat(feat, vocabulary_size=df[feat].nunique(), embedding_dim=4)
                                 for feat in sparse_features
                             ] + [
                                 # 稠密特征配置
                                 DenseFeat(feat, 1)
                                 for feat in dense_features
                             ]

    return df, fixlen_feature_columns, sparse_features, dense_features

def build_and_train_model(df, fixlen_feature_columns, sparse_features, dense_features):
    """构建并训练DeepFM模型"""
    # 获取特征名称列表
    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 划分训练集和测试集
    train, test = train_test_split(df, test_size=0.2, random_state=2021)

    # 准备训练和测试数据
    train_model_input = {name: train[name].values for name in feature_names}
    test_model_input = {name: test[name].values for name in feature_names}

    # 构建DeepFM模型
    model = DeepFM(
        linear_feature_columns=linear_feature_columns,
        dnn_feature_columns=dnn_feature_columns,
        task='binary',  # 二分类任务
        dnn_hidden_units=(128, 128),  # DNN隐藏层配置
        l2_reg_linear=0.00001,
        l2_reg_embedding=0.00001,
        l2_reg_dnn=0,
        seed=2021
    )

    # 编译模型
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['AUC', 'accuracy']
    )

    # 训练模型
    history = model.fit(
        train_model_input,
        train['label'].values,
        batch_size=32,
        epochs=10,
        verbose=1,
        validation_split=0.2
    )

    return model, test_model_input, test['label'].values

def predict_and_evaluate(model, test_model_input, test_labels):
    """使用模型进行预测并评估结果"""
    # 预测概率
    pred_ans = model.predict(test_model_input, batch_size=256)

    # 输出部分预测结果
    print("\n部分预测结果:")
    for i in range(10):
        print(f"真实值: {test_labels[i]}, 预测概率: {pred_ans[i][0]:.4f}")

    # 计算准确率（以0.5为阈值）
    pred_labels = (pred_ans > 0.5).astype(int)
    accuracy = (pred_labels.flatten() == test_labels).mean()
    print(f"\n预测准确率: {accuracy:.4f}")

    return pred_ans

if __name__ == "__main__":
    # 数据准备
    df, fixlen_feature_columns, sparse_features, dense_features = prepare_data()
    print("数据准备完成，共{}条记录".format(len(df)))

    # 模型构建与训练
    model, test_model_input, test_labels = build_and_train_model(
        df, fixlen_feature_columns, sparse_features, dense_features
    )
    print("模型训练完成")

    # 预测与评估
    predictions = predict_and_evaluate(model, test_model_input, test_labels)
    print("预测完成")
