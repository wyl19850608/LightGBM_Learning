import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def build_dnn_model(feature_columns, hidden_units=[64, 32], output_units=1, activation='relu', output_activation='sigmoid'):
    """
    构建处理混合特征的DNN模型

    参数:
    - feature_columns: 特征列定义列表
    - hidden_units: 隐藏层神经元数量列表
    - output_units: 输出层神经元数量
    - activation: 隐藏层激活函数
    - output_activation: 输出层激活函数
    """
    # 定义输入层
    feature_inputs = {
        feature.name: layers.Input(name=feature.name, shape=(), dtype=tf.float32)
        for feature in feature_columns
    }

    # 特征列转换为特征层
    feature_layer = layers.DenseFeatures(feature_columns)
    x = feature_layer(feature_inputs)

    # 添加隐藏层
    for units in hidden_units:
        x = layers.Dense(units, activation=activation)(x)
        x = layers.Dropout(0.3)(x)  # 添加Dropout防止过拟合

    # 输出层
    output = layers.Dense(output_units, activation=output_activation)(x)

    # 创建模型
    model = tf.keras.Model(inputs=feature_inputs, outputs=output)

    # 编译模型
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy' if output_units == 1 else 'sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def preprocess_data(df, label_column, categorical_columns, numerical_columns):
    """
    预处理混合类型数据

    参数:
    - df: 输入DataFrame
    - label_column: 标签列名
    - categorical_columns: 分类特征列名列表
    - numerical_columns: 数值特征列名列表
    """
    # 分离特征和标签
    X = df[numerical_columns + categorical_columns]
    y = df[label_column]

    # 定义预处理管道
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())  # 连续特征标准化
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  # 离散特征独热编码
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_columns),
            ('cat', categorical_transformer, categorical_columns)
        ])

    # 应用预处理
    X_processed = preprocessor.fit_transform(X)

    # 获取特征名称
    feature_names = (
            numerical_columns +
            preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_columns).tolist()
    )

    # 转换为TensorFlow Dataset
    dataset = tf.data.Dataset.from_tensor_slices(
        (dict(pd.DataFrame(X_processed, columns=feature_names)), y.values)
    )

    return dataset, feature_names

def prepare_datasets(dataset, feature_names, batch_size=32, test_size=0.2, val_size=0.2, shuffle=True, shuffle_buffer_size=1000):
    """
    准备训练、验证和测试数据集

    参数:
    - dataset: 输入TensorFlow Dataset
    - feature_names: 特征名称列表
    - batch_size: 批次大小
    - test_size: 测试集比例
    - val_size: 验证集比例
    - shuffle: 是否打乱数据
    - shuffle_buffer_size: 打乱缓冲区大小
    """
    # 计算数据集大小
    dataset_size = len(list(dataset))

    # 计算分割点
    test_size = int(test_size * dataset_size)
    val_size = int(val_size * dataset_size)
    train_size = dataset_size - test_size - val_size

    # 打乱数据集
    if shuffle:
        dataset = dataset.shuffle(shuffle_buffer_size, seed=42)

    # 分割数据集
    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)
    val_dataset = test_dataset.skip(val_size)
    test_dataset = test_dataset.take(val_size)

    # 批处理
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # 创建特征列
    feature_columns = []
    for feature_name in feature_names:
        feature_columns.append(tf.feature_column.numeric_column(feature_name))

    return train_dataset, val_dataset, test_dataset, feature_columns

def train_and_evaluate_model(df, label_column, categorical_columns, numerical_columns,
                             hidden_units=[64, 32], output_units=1, activation='relu',
                             output_activation='sigmoid', epochs=10, batch_size=32):
    """
    训练并评估DNN模型

    参数:
    - df: 输入DataFrame
    - label_column: 标签列名
    - categorical_columns: 分类特征列名列表
    - numerical_columns: 数值特征列名列表
    - 其他参数参考上述函数
    """
    # 预处理数据
    dataset, feature_names = preprocess_data(df, label_column, categorical_columns, numerical_columns)

    # 准备数据集
    train_dataset, val_dataset, test_dataset, feature_columns = prepare_datasets(
        dataset, feature_names, batch_size=batch_size
    )

    # 构建模型
    model = build_dnn_model(
        feature_columns,
        hidden_units=hidden_units,
        output_units=output_units,
        activation=activation,
        output_activation=output_activation
    )

    # 训练模型
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs
    )

    # 评估模型
    test_loss, test_acc = model.evaluate(test_dataset)
    print(f"测试集准确率: {test_acc:.4f}")

    return model, history

# 使用示例
if __name__ == "__main__":
    print(tf.__version__)

    # 示例数据加载 (请替换为实际数据)
    data = {
        'age': [25, 30, 45, 50, 22],  # 连续特征 (0-100)
        'income': [50000, 80000, 120000, 75000, 30000],  # 连续特征 (0-100000+)
        'rating': [4.2, 3.5, 5.0, 2.8, 4.7],  # 连续特征 (0-5)
        'gender': ['Male', 'Female', 'Male', 'Female', 'Male'],  # 离散特征 (字符串)
        'education': [1, 2, 3, 2, 1],  # 离散特征 (数字)
        'label': [1, 0, 1, 0, 1]  # 二分类标签
    }

    df = pd.DataFrame(data)

    # 定义特征列类型
    categorical_columns = ['gender', 'education']
    numerical_columns = ['age', 'income', 'rating']
    label_column = 'label'

    # 训练模型
    model, history = train_and_evaluate_model(
        df,
        label_column,
        categorical_columns,
        numerical_columns,
        hidden_units=[64, 32],
        output_units=1,  # 二分类
        activation='relu',
        output_activation='sigmoid',
        epochs=5
    )