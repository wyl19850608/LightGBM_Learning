import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# 构建 DNN 模型
class DNNModel(nn.Module):
    def __init__(self, input_size, hidden_units=[64, 32], output_units=1):
        super(DNNModel, self).__init__()
        layers_list = []
        prev_units = input_size
        for units in hidden_units:
            layers_list.append(nn.Linear(prev_units, units))
            layers_list.append(nn.ReLU())
            layers_list.append(nn.Dropout(0.3))
            prev_units = units
        layers_list.append(nn.Linear(prev_units, output_units))
        layers_list.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers_list)

    def forward(self, x):
        return self.model(x)


# 预处理数据
def preprocess_data(df, label_column, categorical_columns, numerical_columns):
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
    X_processed = preprocessor.fit_transform(X).toarray()

    return X_processed, y.values


# 准备数据集
def prepare_datasets(X, y, batch_size=32, test_size=0.2, val_size=0.2):
    # 划分训练集、验证集和测试集
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_ratio, random_state=42)

    # 创建数据集和数据加载器
    train_dataset = CustomDataset(X_train, y_train)
    val_dataset = CustomDataset(X_val, y_val)
    test_dataset = CustomDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# 训练并评估模型
def train_and_evaluate_model(df, label_column, categorical_columns, numerical_columns,
                             hidden_units=[64, 32], output_units=1, epochs=10, batch_size=32):
    # 预处理数据
    X_processed, y = preprocess_data(df, label_column, categorical_columns, numerical_columns)

    # 准备数据集
    train_loader, val_loader, test_loader = prepare_datasets(X_processed, y, batch_size=batch_size)

    # 构建模型
    input_size = X_processed.shape[1]
    model = DNNModel(input_size, hidden_units=hidden_units, output_units=output_units)

    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())

    # 训练模型
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}')

        # 验证模型
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item()
                predicted = (outputs.squeeze() > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Validation Loss: {val_loss / len(val_loader)}, Validation Accuracy: {correct / total}')

    # 评估模型
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            test_loss += loss.item()
            predicted = (outputs.squeeze() > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_acc = correct / total
    print(f"测试集准确率: {test_acc:.4f}")

    return model


# 使用示例
if __name__ == "__main__":
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
    model = train_and_evaluate_model(
        df,
        label_column,
        categorical_columns,
        numerical_columns,
        hidden_units=[64, 32],
        output_units=1,  # 二分类
        epochs=5
    )